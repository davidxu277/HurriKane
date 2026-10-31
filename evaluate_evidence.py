"""Audit frozen predictions and evidence rules without training or inference."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml import build_samples
from rules import apply_risk_rules
from validation import validate_input


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def counts(frame):
    review = frame.data_quality_status.eq("NEEDS_REVIEW")
    low = frame.tier_pc.lt(1e-6)
    return {
        "events": len(frame),
        "needs_review_events": int(review.sum()),
        "needs_review_fraction": float(review.mean()) if len(frame) else None,
        "risk_level_counts": {str(k): int(v) for k, v in frame.risk_level.value_counts().items()},
        "safe_needs_review_events": int((review & frame.risk_level.eq("SAFE")).sum()),
        "low_latest_pc_events": int(low.sum()),
        "missing_sigma_events": int(frame.sigma_ratio.isna().sum()),
        "reason_counts_nonexclusive": {
            name: int(frame.data_quality_reason.str.contains(pattern, regex=False).sum())
            for name, pattern in {
                "large_sigma_low_pc": " > training threshold=",
                "missing_sigma": "sigma_ratio is missing",
                "conservative_grade_crossing": "raises grade",
                "upstream_warning": "input warning:",
            }.items()
        },
    }


def boundary_checks(threshold):
    cases = [
        ("ratio_equal_threshold", 1e-7, 6000, threshold, 1e-7, "SAFE", "OK"),
        ("ratio_above_threshold", 1e-7, 6000, np.nextafter(threshold, np.inf), 1e-7, "SAFE", "NEEDS_REVIEW"),
        ("pc_equal_1e-6", 1e-6, 6000, threshold * 2, 1e-6, "ELEVATED", "OK"),
        ("missing_sigma", 1e-7, 6000, np.nan, 1e-7, "SAFE", "NEEDS_REVIEW"),
        ("missing_pc", np.nan, 6000, 0, 1e-7, "UNKNOWN", "NEEDS_REVIEW"),
        ("missing_distance", 1e-7, np.nan, 0, 1e-7, "UNKNOWN", "NEEDS_REVIEW"),
        ("zero_distance", 1e-7, 0, np.inf, 1e-7, "CRITICAL", "NEEDS_REVIEW"),
        ("conservative_crossing", 1e-7, 6000, 0, 1e-5, "SAFE", "NEEDS_REVIEW"),
    ]
    results = {}
    for name, pc, distance, ratio, upper, level, status in cases:
        result = apply_risk_rules(pd.DataFrame([{
            "tier_pc": pc, "predicted_pc": 1e-9, "conservative_pc": upper,
            "miss_distance_m": distance, "sigma_ratio": ratio,
        }]), threshold).iloc[0]
        assert (result.risk_level, result.data_quality_status) == (level, status), name
        results[name] = {"risk_level": result.risk_level, "data_quality_status": result.data_quality_status, "passed": True}
    return results


def score_with_warnings(predictions, samples, issues, eligible_indices, threshold):
    """Apply exactly the service's historical-input warning propagation."""
    source = samples.set_index("event_id").loc[predictions.event_id]
    frame = predictions.copy()
    for name in ("c_sigma_r", "t_sigma_r"):
        frame[name] = source[name].to_numpy()
    pure = apply_risk_rules(frame, threshold)
    warnings = issues[issues.severity.eq("WARNING") & issues.row_position.isin(eligible_indices)]
    reasons = warnings.groupby("event_id").apply(
        lambda group: "; ".join(dict.fromkeys(
            f"input warning: {row.column}: {row.message}" for row in group.itertuples()
        )), include_groups=False,
    ).to_dict() if not warnings.empty else {}
    frame["data_quality_reason"] = frame.event_id.map(reasons).fillna("")
    frame["data_quality_status"] = np.where(frame.data_quality_reason.ne(""), "NEEDS_REVIEW", "OK")
    return pure, apply_risk_rules(frame, threshold)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="Data_raw/train_data.csv")
    parser.add_argument("--test", default="Data_raw/test_data.csv")
    parser.add_argument("--output", default="reports/evidence_holdout.json")
    args = parser.parse_args()
    frozen = ["models/metadata.json", "models/estimate.cbm", "models/conservative.cbm",
              "models/classifier.cbm", "reports/split_events.json", "reports/holdout_metrics.json",
              "reports/holdout_predictions.csv", "reports/test_predictions.csv",
              "ml.py", "train.py", "rules.py", "validation.py", "service.py", "evaluate_workload.py",
              "evaluate_evidence.py"]
    before = {name: sha256(name) for name in frozen}
    metadata = json.loads(Path("models/metadata.json").read_text(encoding="utf-8"))
    split = json.loads(Path("reports/split_events.json").read_text(encoding="utf-8"))
    sets = {name: set(ids) for name, ids in split.items()}
    assert all(len(ids) == len(sets[name]) for name, ids in split.items())
    assert not (sets["train"] & sets["validation"] or sets["train"] & sets["holdout"]
                or sets["validation"] & sets["holdout"])
    predictions = pd.read_csv("reports/holdout_predictions.csv")
    assert predictions.event_id.is_unique and set(predictions.event_id) == sets["holdout"]
    raw = pd.read_csv(args.train, low_memory=False)
    valid, issues = validate_input(raw, metadata)
    assert not issues.severity.eq("ERROR").any(), "Raw data contains validation errors"
    samples = build_samples(valid, metadata["task"], metadata["horizon_days"])
    aligned = samples.set_index("event_id").loc[predictions.event_id]
    assert np.array_equal(aligned.has_future.to_numpy(), predictions.has_future.to_numpy())
    assert np.array_equal(aligned.final_tier.to_numpy(), predictions.final_tier.to_numpy())
    assert np.allclose(aligned.known_log10_pc.to_numpy(), predictions.tier_log10_pc.to_numpy())
    assert np.allclose(aligned.miss_distance.to_numpy(), predictions.miss_distance_m.to_numpy(), equal_nan=True)
    assert np.allclose(aligned.sigma_ratio.to_numpy(), predictions.sigma_ratio.to_numpy(), equal_nan=True)
    eligible = valid.index[valid.time_to_tca.ge(metadata["horizon_days"])]
    threshold = metadata["sigma_ratio_threshold"]
    pure, scored = score_with_warnings(predictions, samples, issues, eligible, threshold)
    # Validate the rule independently on the existing cohort, including equality.
    low = scored.tier_pc.lt(1e-6)
    large = low & scored.sigma_ratio.gt(threshold)
    missing = scored.sigma_ratio.isna()
    assert scored.loc[large | missing, "data_quality_status"].eq("NEEDS_REVIEW").all()
    assert pure.risk_level.equals(scored.risk_level), "Warnings must not overwrite the risk grade"
    assert scored.loc[scored.data_quality_status.eq("NEEDS_REVIEW"), "data_quality_reason"].ne("").all()
    train = samples[samples.event_id.isin(sets["train"]) & samples.has_future]
    def quantile(mask):
        values = train.loc[mask, "sigma_ratio"].replace(np.inf, np.nan).dropna()
        return {"events_with_finite_ratio": len(values), "q90": float(values.quantile(0.9))}
    original = quantile(train.y.lt(-6))
    latest = quantile(train.known_log10_pc.lt(-6))
    assert np.isclose(original["q90"], threshold, rtol=1e-12, atol=0)
    workload = json.loads(Path("reports/workload_holdout.json").read_text(encoding="utf-8"))
    metrics = json.loads(Path("reports/holdout_metrics.json").read_text(encoding="utf-8"))
    for scope, result in (("future", workload), ("all", workload["supplementary_all"])):
        expected = metrics["tiers"][scope]
        assert result["events"] == expected["events"]
        assert result["positive_events"] == expected["critical"]["positive_events"]
        for method in ("baseline", "model"):
            assert result[method]["reviews"] == expected["critical"]["review_for_90_recall"][method]
    report = {
        "model_version": metadata["model_version"], "variant": metadata["variant"],
        "scope": "future", "role": "primary", "rule_threshold": threshold,
        "rule": "latest Pc < 1e-6 AND sigma_ratio > frozen training threshold => NEEDS_REVIEW; missing ratio also requires review",
        "counts": counts(scored[scored.has_future]),
        "rules_only_counts": counts(pure[pure.has_future]),
        "supplementary_all": {"counts": counts(scored), "rules_only_counts": counts(pure)},
        "threshold_provenance": {
            "source": "fixed training events with has_future=True; 90th percentile, finite sigma ratios",
            "training_events": len(train), "stored_threshold": threshold,
            "actual_training_filter": "final y < -6", "reproduced": original,
            "plan_filter_latest_known_pc": latest,
            "finding": "train.py filters by final y; runtime filters by latest known Pc. This differs from the plan's latest-Pc wording. The stored threshold is retained; no training, model or rule change.",
        },
        "checks": {"split_disjoint": True, "holdout_ids_exact_match": True,
                   "saved_labels_and_input_pc_match": True, "insufficient_evidence_flagged": True,
                   "missing_sigma_flagged": True, "risk_grade_preserved": True,
                   "workload_matches_main_metrics": True},
        "boundary_checks": boundary_checks(threshold),
        "method": "Reuse saved predictions; attach selected input sigmas and propagate eligible historical-message warnings as in service.py. No model prediction call or training.",
    }
    out = Path(args.output)
    scored.to_csv(out.with_name(out.stem + "_events.csv"), index=False)
    # The unlabeled test cohort is supplementary operational evidence, not accuracy evaluation.
    test_predictions = pd.read_csv("reports/test_predictions.csv")
    test_raw = pd.read_csv(args.test, low_memory=False)
    test_valid, test_issues = validate_input(test_raw, metadata)
    assert not test_issues.severity.eq("ERROR").any()
    test_samples = build_samples(test_valid, metadata["task"], metadata["horizon_days"], labeled=False)
    assert test_predictions.event_id.is_unique and set(test_predictions.event_id) == set(test_samples.event_id)
    test_eligible = test_valid.index[test_valid.time_to_tca.ge(metadata["horizon_days"])]
    test_pure, test_scored = score_with_warnings(test_predictions, test_samples, test_issues, test_eligible, threshold)
    test_scored.to_csv(out.with_name("evidence_test_events.csv"), index=False)
    report["supplementary_unlabeled_test"] = {
        "counts": counts(test_scored), "rules_only_counts": counts(test_pure),
        "note": "Existing test predictions; no accuracy or recall claim. Historical input warnings are included.",
    }
    after = {name: sha256(name) for name in frozen}
    assert before == after, "Frozen input changed during audit"
    report["checks"]["frozen_files_unchanged"] = True
    report["frozen_sha256"] = before
    report["raw_sha256"] = {args.train: sha256(args.train), args.test: sha256(args.test)}
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"primary": report["counts"], "all": report["supplementary_all"]["counts"],
                      "test": report["supplementary_unlabeled_test"]["counts"],
                      "threshold_provenance": report["threshold_provenance"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
