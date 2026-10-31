"""Risk grading only: no training, sample filtering, or split decisions.

Distances are metres; probabilities are Pc, not classifier alert scores.
Invalid numeric evidence raises ValueError for the caller's validation/issue
layer. Missing evidence is retained and explicitly marked NEEDS_REVIEW.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


RISK_RULES_VERSION = "1.0"
LEVEL_RANK = {"UNKNOWN": -1, "SAFE": 0, "ELEVATED": 1, "CRITICAL": 2}
WINDOW_LABEL = "24-hour window reported by each event's latest message"


def _numeric(frame: pd.DataFrame, name: str, allow_inf: bool = False) -> pd.Series:
    if name not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    source = frame[name]
    values = pd.to_numeric(source, errors="coerce").astype(float)
    bad = (source.notna() & values.isna()) | (np.isinf(values) & (not allow_inf))
    if bad.any():
        raise ValueError(f"{name}: invalid numeric evidence at row positions "
                         f"{np.flatnonzero(bad.to_numpy()).tolist()[:10]}")
    return values


def _probability(frame: pd.DataFrame, prefix: str) -> pd.Series:
    pc = _numeric(frame, f"{prefix}_pc")
    log = _numeric(frame, f"{prefix}_log10_pc")
    if ((pc < 0) | (pc > 1) | (log > 0)).any():
        raise ValueError(f"{prefix}: Pc must be in [0, 1] and log10(Pc) <= 0")
    from_log = np.power(10.0, log)
    both = pc.notna() & log.notna()
    if (both & ~np.isclose(pc, from_log, rtol=1e-10, atol=0)).any():
        raise ValueError(f"{prefix}: probability and log10 probability disagree")
    return pc.where(pc.notna(), from_log)


def _grade(pc: float, distance: float) -> tuple[str, str]:
    critical_reasons = []
    if pd.notna(pc) and pc >= 1e-4:
        critical_reasons.append(f"Pc={pc:.12g} >= 0.0001")
    if pd.notna(distance) and distance < 1000:
        critical_reasons.append(f"miss_distance={distance:.12g} m < 1000 m")
    if critical_reasons:
        return "CRITICAL", "; ".join(critical_reasons)
    reasons = []
    if pd.notna(pc) and pc >= 1e-6:
        reasons.append(f"Pc={pc:.12g} >= 0.000001")
    if pd.notna(distance) and distance <= 5000:
        reasons.append(f"miss_distance={distance:.12g} m <= 5000 m")
    if reasons:
        return "ELEVATED", "; ".join(reasons)
    if pd.notna(pc) and pd.notna(distance):
        return "SAFE", (f"Pc={pc:.12g} < 0.000001; "
                        f"miss_distance={distance:.12g} m > 5000 m")
    return "UNKNOWN", "Available Pc and distance cannot establish a risk level"


def apply_risk_rules(
    predictions: pd.DataFrame, sigma_ratio_threshold: float | None
) -> pd.DataFrame:
    """Return a copy with risk and quality fields, retaining index and every row.

    Grade on tier_pc, the latest known Pc in final mode; the model estimate is
    kept for display only. Accept probability columns or their log10
    equivalents. Clamp conservative Pc to at least the estimate. If sigma_ratio
    is absent, derive it from radial sigmas and distance; an explicitly missing
    ratio remains missing, and an infinite ratio means zero distance. When raw
    sigma fields are supplied, either missing sigma invalidates the ratio.
    None means the training-only ratio threshold is not available yet.
    """
    if not isinstance(predictions, pd.DataFrame):
        raise TypeError("predictions must be a pandas DataFrame")
    if not predictions.columns.is_unique:
        raise ValueError("Duplicate column names are not supported")
    if sigma_ratio_threshold is not None:
        if (isinstance(sigma_ratio_threshold, bool)
                or not np.isfinite(sigma_ratio_threshold)
                or sigma_ratio_threshold < 0):
            raise ValueError("sigma_ratio_threshold must be finite and nonnegative")

    result = predictions.copy(deep=True)
    tier = _probability(result, "tier")
    pc = _probability(result, "predicted")
    conservative = _probability(result, "conservative")
    conservative = conservative.where(pc.isna() | (conservative >= pc), pc).where(
        conservative.notna()
    )
    distance = _numeric(result, "miss_distance_m")
    c_sigma = _numeric(result, "c_sigma_r")
    t_sigma = _numeric(result, "t_sigma_r")
    if ((distance < 0) | (c_sigma < 0) | (t_sigma < 0)).any():
        raise ValueError("Distance and radial sigmas must be nonnegative")
    ratio = _numeric(result, "sigma_ratio", allow_inf=True)
    if (ratio < 0).any():
        raise ValueError("sigma_ratio must be nonnegative")
    raw_sigma_supplied = "c_sigma_r" in result or "t_sigma_r" in result
    if "sigma_ratio" not in result:
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.hypot(c_sigma, t_sigma) / distance
        # Even zero sigmas at zero distance represent an undefined ratio.
        ratio = ratio.mask(distance.eq(0) & c_sigma.notna() & t_sigma.notna(), np.inf)
    if raw_sigma_supplied:
        ratio = ratio.mask(c_sigma.isna() | t_sigma.isna())

    result["tier_pc"] = tier
    result["predicted_pc"] = pc
    result["conservative_pc"] = conservative
    # Regenerate matching log values; Pc=0 has no finite log representation.
    for prefix, values in (("tier", tier), ("predicted", pc), ("conservative", conservative)):
        with np.errstate(divide="ignore"):
            logs = np.log10(values.where(values > 0))
        # Retain finite source log values when floating-point Pc underflows.
        original_log = _numeric(predictions, f"{prefix}_log10_pc")
        result[f"{prefix}_log10_pc"] = logs.where(logs.notna(), original_log)
    result["miss_distance_m"] = distance
    result["sigma_ratio"] = ratio

    levels, triggers, statuses, quality_reasons = [], [], [], []
    for pos in range(len(result)):
        row = result.iloc[pos]
        value, dist, upper, sigma_ratio = tier.iloc[pos], distance.iloc[pos], conservative.iloc[pos], ratio.iloc[pos]
        level, reason = _grade(value, dist)
        quality = []
        prior_reason = row.get("data_quality_reason")
        if pd.notna(prior_reason) and str(prior_reason).strip():
            quality.append(str(prior_reason))
        prior_status = row.get("data_quality_status", "OK")
        if pd.isna(prior_status):
            quality.append("Upstream data quality status is missing")
        elif prior_status not in ("OK", "NEEDS_REVIEW"):
            quality.append(f"Unrecognized upstream data quality status: {prior_status}")
        elif prior_status == "NEEDS_REVIEW" and not quality:
            quality.append("Upstream validation requires review")
        if pd.isna(value):
            quality.append("Latest Pc is missing")
        if pd.isna(dist):
            quality.append("Miss distance (m) is missing")
        if pd.isna(upper):
            quality.append("Conservative Pc is missing; threshold crossing cannot be checked")
        if pd.isna(sigma_ratio):
            quality.append("sigma_ratio is missing; sigma evidence is unknown (not zero)")
        elif pd.notna(value) and value < 1e-6:
            if sigma_ratio_threshold is None:
                quality.append("Training sigma_ratio_threshold is unavailable; evidence rule cannot be checked")
            elif sigma_ratio > sigma_ratio_threshold:
                quality.append(f"sigma_ratio={sigma_ratio:.12g} > training threshold={sigma_ratio_threshold:.12g} (dimensionless)")
        if pd.notna(upper):
            upper_level, upper_reason = _grade(upper, dist)
            if LEVEL_RANK[upper_level] > LEVEL_RANK[level]:
                quality.append(f"Conservative Pc={upper:.12g} raises grade {level} -> {upper_level}; "
                               f"latest Pc={value:.12g}; {upper_reason}")
        levels.append(level)
        triggers.append(reason)
        statuses.append("NEEDS_REVIEW" if quality else "OK")
        quality_reasons.append("; ".join(dict.fromkeys(quality)))
    result["risk_level"] = levels
    result["trigger_reason"] = triggers
    result["data_quality_status"] = statuses
    result["data_quality_reason"] = quality_reasons
    return result


def summarize_events(scored_df: pd.DataFrame) -> pd.DataFrame:
    """Select each event's latest message (smallest time_to_tca_hours).

    Final-mode single-row events need no time for selection. Missing timing in
    a multi-message event, or ties at its latest time, make selection ambiguous:
    preserve the event ID but clear row-level evidence and require review.
    No timestamp is inferred from row order. The 24-hour flag is based only on
    0 < latest time <= 24 and is not a forecast of absent events.
    """
    required = {"event_id", "risk_level", "trigger_reason", "data_quality_status", "data_quality_reason"}
    missing = required - set(scored_df.columns)
    if missing:
        raise ValueError(f"Missing scored fields: {sorted(missing)}")
    if scored_df["event_id"].isna().any():
        raise ValueError("event_id is missing; cannot reliably summarize events")
    if not scored_df["risk_level"].isin(LEVEL_RANK).all():
        raise ValueError("Unrecognized risk_level")
    frame = scored_df.copy(deep=True)
    frame["time_to_tca_hours"] = _numeric(frame, "time_to_tca_hours")
    rows = []
    for event_id, group in frame.groupby("event_id", sort=False, observed=True):
        times = group["time_to_tca_hours"]
        ambiguous = len(group) > 1 and (times.isna().any() or times.eq(times.min()).sum() != 1)
        if ambiguous:
            record = {column: pd.NA for column in frame.columns}
            record.update(event_id=event_id, risk_level="UNKNOWN",
                          trigger_reason="Latest message cannot be selected reliably",
                          data_quality_status="NEEDS_REVIEW",
                          data_quality_reason="Missing or tied message times; latest evidence is ambiguous")
        else:
            selected = group.iloc[0] if len(group) == 1 else group.loc[times.eq(times.min())].iloc[0]
            record = selected.to_dict()
        time = record["time_to_tca_hours"]
        record["message_count"] = len(group)
        record["in_24h_window"] = bool(pd.notna(time) and 0 < time <= 24)
        record["time_coverage_status"] = (
            "TIME_UNKNOWN" if pd.isna(time) else
            "IN_24H_WINDOW" if 0 < time <= 24 else
            "AT_OR_AFTER_TCA" if time <= 0 else "OUTSIDE_24H_WINDOW"
        )
        record["time_coverage_label"] = WINDOW_LABEL
        rows.append(record)
    extras = ["message_count", "in_24h_window", "time_coverage_status", "time_coverage_label"]
    return pd.DataFrame(rows, columns=list(frame.columns) + [c for c in extras if c not in frame])
