"""B 的单一入口：CSV DataFrame → 校验 → 缓存模型推理 → 规则 → 事件统计。"""

from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from ml import load_bundle, predict_batch
from rules import LEVEL_RANK, RISK_RULES_VERSION, apply_risk_rules, summarize_events
from validation import ISSUE_COLUMNS, validate_input


SERVICE_VERSION = "1.1"
MODEL_FILES = ("metadata.json", "estimate.cbm", "conservative.cbm", "classifier.cbm")
HISTORY_COLUMNS = [
    "event_id", "time_to_tca_hours", "risk", "miss_distance_m",
    "input_record_id", "row_position",
]
PREDICTION_COLUMNS = [
    "input_record_id", "row_position", "event_id", "time_to_tca_hours",
    "miss_distance_m", "current_tier", "predicted_final_tier",
    "p_safe", "p_elevated", "p_critical", "tier_log10_pc", "tier_pc", "predicted_log10_pc",
    "predicted_pc", "conservative_log10_pc", "conservative_pc", "alert_score",
    "baseline_score", "sigma_ratio", "c_sigma_r", "t_sigma_r",
    "relative_position_r", "relative_position_t", "relative_position_n", "c_object_type",
    "risk_level", "trigger_reason", "data_quality_status", "data_quality_reason",
    "prediction_status",
]


@lru_cache(maxsize=4)
def _cached_bundle(directory, signature):
    # signature 包含四个文件的修改时间和大小，模型更新后自动重新加载。
    return load_bundle(directory)


def clear_model_cache():
    """可选：手动清理进程内模型缓存，例如发布新模型之后。"""
    _cached_bundle.cache_clear()


def _get_bundle(model_dir):
    directory = Path(model_dir).resolve()
    missing = [name for name in MODEL_FILES if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"模型未就绪：{directory} 缺少 {', '.join(missing)}")
    signature = tuple(
        ((directory / name).stat().st_mtime_ns, (directory / name).stat().st_size)
        for name in MODEL_FILES
    )
    return _cached_bundle(str(directory), signature)


def _event_ids(df):
    """只统计可识别的 ID；缺失/无穷 ID 留在 issues 中，不创造事件编号。"""
    if list(df.columns).count("event_id") != 1:
        return pd.Series(index=df.index, dtype=object)
    ids = df["event_id"].copy()
    blank = ids.map(lambda value: isinstance(value, str) and not value.strip()).astype(bool)
    infinite = pd.to_numeric(ids, errors="coerce").isin([np.inf, -np.inf])
    return ids.mask(blank | infinite)


def _issue(event_id, code, severity, message):
    return dict(zip(ISSUE_COLUMNS, [
        pd.NA, pd.NA, event_id, pd.NA, code, severity, pd.NA, message,
    ]))


def _reasons(issues):
    reasons = {}
    for event_id, group in issues.dropna(subset=["event_id"]).groupby("event_id", sort=False):
        reasons[event_id] = "; ".join(dict.fromkeys(
            f"{row.column}: {row.message}" if pd.notna(row.column) else row.message
            for row in group.itertuples()
        ))
    return reasons


def _history(infer, source_index):
    """仅返回实际推理输入的历史观测；risk 保留原始 log10(Pc) 量纲。"""
    if infer.empty:
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    positions = infer.index.to_numpy(dtype=int)
    # current 推理不要求 risk，也不能用模型输出伪造观测历史。
    risk = pd.to_numeric(infer.reindex(columns=["risk"])["risk"], errors="coerce")
    risk = risk.where(np.isfinite(risk) & risk.le(0))
    return pd.DataFrame({
        "event_id": infer.event_id.to_numpy(),
        "time_to_tca_hours": infer.time_to_tca.to_numpy() * 24,
        "risk": risk.to_numpy(),
        "miss_distance_m": infer.miss_distance.to_numpy(),
        "input_record_id": list(source_index.take(positions)),
        "row_position": positions,
    }, columns=HISTORY_COLUMNS)


def _display_issues(issues):
    """添加当前页面使用的展示字段，保留完整校验字段供下载和排查。"""
    issues = issues.copy()
    issues["row_number"] = pd.to_numeric(issues.row_position, errors="coerce").astype("Int64") + 1
    issues["field"] = issues.column.fillna("")
    issues["problem"] = issues.severity + " [" + issues.code + "]: " + issues.message
    for index in issues.index[issues.event_id.notna()]:
        issues.at[index, "problem"] += f" (event_id={issues.at[index, 'event_id']})"
    return issues


def _summary(records, events, issues, input_rows, valid_rows, ids):
    def counts(frame):
        return {
            "total_events": len(frame),
            **{f"{level.lower()}_events": int(frame.risk_level.eq(level).sum())
               for level in LEVEL_RANK},
            "needs_review_events": int(frame.data_quality_status.eq("NEEDS_REVIEW").sum()),
        }

    window = events[events.in_24h_window.fillna(False).astype(bool)]
    predicted = records.loc[records.prediction_status.eq("PREDICTED"), "event_id"].nunique()
    errors = int(issues.severity.eq("ERROR").sum())
    blocked = len(events) - predicted
    status = "empty" if input_rows == 0 else "partial" if errors or blocked else "ok"
    if input_rows and not predicted:
        status = "invalid_input" if errors else "no_eligible_input"
    return {
        "status": status, "input_rows": input_rows, "valid_rows": valid_rows,
        "rejected_rows": input_rows - valid_rows,
        "input_events": int(ids.nunique()), "unidentified_event_rows": int(ids.isna().sum()),
        "predicted_events": int(predicted), "unpredicted_events": int(blocked),
        "issue_count": len(issues), "error_count": errors,
        "warning_count": int(issues.severity.eq("WARNING").sum()),
        **counts(events),
        "window_24h": {
            **counts(window), "label": "各事件最新消息报告的 24 小时窗口",
            "condition": "0 < time_to_tca_hours <= 24",
            "coverage_status": "OBSERVED" if len(window) else "INSUFFICIENT",
            "coverage_note": (
                "仅统计各事件最新可用输入消息报告的窗口；没有统一消息时间，不能视为同一时刻的完整覆盖。"
                if len(window) else
                "没有可可靠归入 24 小时窗口的事件，属于覆盖不足，不能解释为未来 24 小时无风险。"
            ),
            "unknown_time_events": int(events.time_coverage_status.eq("TIME_UNKNOWN").sum()),
        },
    }


def analyze_csv(df: pd.DataFrame, model_dir="models") -> dict:
    """分析上传的原始消息，返回 records/events/history/issues/summary/metadata。

    records、events、history、issues 是 DataFrame，summary、metadata 是 dict。
    final 的 records 一事件一行；current 为逐条消息。不可预测但 ID 可识别
    的事件保留一条 UNKNOWN / NEEDS_REVIEW 占位记录，prediction_status 为
    NOT_PREDICTED；正常预测为 PREDICTED。未知结果不能当成 SAFE。

    同事件任一校验 ERROR 会阻止该事件整体推理，避免回退到旧消息；文件级
    ERROR 阻止整批推理。输入消息时间有重复也会阻止该事件推理，
    防止任意选择消息或计算变化量。WARNING 按事件传递为复核提示（final
    只传播截止时刻前输入消息的 WARNING）。没有符合 horizon 的事件仍保留。

    input_record_id 还原为调用者的原索引，row_position 为 0 起始的输入位置。
    final 的 message_count 是符合 horizon 的输入消息数；input_message_count
    是上传的总消息数。其他字段与 ML/规则输出一致。events 按等级、排序分数
    降序排列；统计只基于 events，不重复累计消息。

    history 只含实际参与推理的合法历史消息，final 严格使用截止时刻前的
    输入；risk 为原始 log10(Pc)，current 未上传 risk 时留空。issues 保留
    校验字段并添加页面展示字段 row_number（从 1 开始的数据行号）、field、
    problem。summary 保留原统计并提供页面使用的 *_count 别名；24 小时
    覆盖不足时 window_24h_count 为 None，详细计数仍在 window_24h 中。

    模型按绝对目录、文件修改时间/大小在当前进程缓存，不训练、不修改原始
    数据或报告、不读取留出标签。缺少模型抛 FileNotFoundError；损坏模型、
    非法 metadata 或程序错误继续抛异常，不包装成成功的预测。返回 metadata
    是副本，页面修改它不会污染缓存。B 可直接展示 events 和下载 records。
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    bundle = _get_bundle(model_dir)
    metadata = deepcopy(bundle["metadata"])
    metadata.update(rules_version=RISK_RULES_VERSION, service_version=SERVICE_VERSION,
                    history_risk_scale="log10(Pc)")
    valid, issues = validate_input(df, metadata)
    ids = _event_ids(df)
    event_order = ids.dropna().drop_duplicates().tolist()
    input_counts = ids.value_counts()
    file_error = bool((issues.severity.eq("ERROR") & issues.row_position.isna()).any())
    reasons = _reasons(issues)
    blocked = set(issues.loc[issues.severity.eq("ERROR"), "event_id"].dropna())
    extra_issues = []

    # 推理内部使用唯一的位置索引，解决原索引重复时无法关联源消息的问题。
    bad_positions = issues.loc[issues.severity.eq("ERROR"), "row_position"].dropna().astype(int)
    positions = np.flatnonzero(~np.isin(np.arange(len(df)), bad_positions))
    if not file_error:
        valid.index = positions
    eligible = valid.iloc[:0].copy() if file_error else valid.copy()
    if not file_error and metadata["task"] == "final":
        horizon = metadata["horizon_days"]
        if isinstance(horizon, bool) or not np.isfinite(horizon) or horizon < 0:
            raise ValueError("metadata.horizon_days must be finite and nonnegative")
        eligible = eligible[eligible.time_to_tca >= horizon]
    if not file_error:
        duplicates = eligible.duplicated(["event_id", "time_to_tca"], keep=False)
        for event_id in eligible.loc[duplicates, "event_id"].drop_duplicates():
            message = "输入消息时间重复，无法可靠确定历史顺序；该事件暂停预测"
            blocked.add(event_id)
            reasons[event_id] = "; ".join(filter(None, [reasons.get(event_id), message]))
            extra_issues.append(_issue(event_id, "AMBIGUOUS_MESSAGE_TIME", "ERROR", message))

    eligible_counts = eligible.event_id.value_counts() if not file_error else pd.Series(dtype=int)
    if file_error:
        message = "; ".join(issues.loc[issues.severity.eq("ERROR"), "message"].drop_duplicates())
        reasons = {event_id: message for event_id in event_order}
        blocked = set(event_order)
        infer = eligible
    else:
        infer = eligible[~eligible.event_id.isin(blocked)]
        missing = set(event_order) - set(infer.event_id) - blocked
        for event_id in event_order:
            if event_id in missing:
                message = "没有符合模型输入截止时刻的消息；未生成预测"
                reasons[event_id] = message
                extra_issues.append(_issue(event_id, "NO_ELIGIBLE_MESSAGES", "WARNING", message))

    if infer.empty:
        records = pd.DataFrame(columns=PREDICTION_COLUMNS)
    else:
        predictions = predict_batch(infer, bundle)
        source_positions = predictions.input_record_id.to_numpy(dtype=int)
        source = infer.loc[source_positions]
        predictions["row_position"] = source_positions
        predictions["input_record_id"] = list(df.index.take(source_positions))
        for name in ["c_sigma_r", "t_sigma_r", "relative_position_r", "relative_position_t",
                     "relative_position_n", "c_object_type"]:
            predictions[name] = source[name].to_numpy()
        # 使用真实 sigma 字段，保证零距离且 sigma 缺失时也不会伪造已知比值。
        relevant_warnings = issues[
            issues.severity.eq("WARNING") & issues.row_position.isin(infer.index)
        ]
        warnings_by_event = _reasons(relevant_warnings)
        predictions["data_quality_reason"] = predictions.event_id.map(warnings_by_event).astype(object).fillna("")
        predictions["data_quality_status"] = np.where(
            predictions.data_quality_reason.ne(""), "NEEDS_REVIEW", "OK"
        )
        records = apply_risk_rules(predictions, metadata.get("sigma_ratio_threshold"))
        records["prediction_status"] = "PREDICTED"

    predicted_ids = set(records.event_id)
    placeholders = []
    for event_id in event_order:
        if event_id not in predicted_ids:
            row = dict.fromkeys(PREDICTION_COLUMNS, np.nan)
            reason = reasons.get(event_id, "没有可用的模型预测，需人工复核")
            row.update(event_id=event_id, risk_level="UNKNOWN", trigger_reason=reason,
                       data_quality_status="NEEDS_REVIEW", data_quality_reason=reason,
                       prediction_status="NOT_PREDICTED")
            placeholders.append(row)
    if placeholders:
        pending = pd.DataFrame(placeholders)
        records = pending if records.empty else pd.concat([records, pending], ignore_index=True)
    records = records.reindex(columns=PREDICTION_COLUMNS)
    events = summarize_events(records)
    for frame in (records, events):
        frame["input_message_count"] = frame.event_id.map(input_counts).fillna(0).astype(int)
    events["message_count"] = events.event_id.map(eligible_counts).fillna(0).astype(int)
    # summarize_events 在并列最新消息时清空证据，因此预测状态也需同步。
    events["prediction_status"] = events.prediction_status.fillna("NOT_PREDICTED")
    events = events.assign(_rank=events.risk_level.map(LEVEL_RANK)).sort_values(
        ["_rank", "alert_score"], ascending=[False, False], kind="stable", na_position="last"
    ).drop(columns="_rank").reset_index(drop=True)
    if extra_issues:
        issues = pd.concat([issues, pd.DataFrame(extra_issues, columns=ISSUE_COLUMNS)], ignore_index=True)
    summary = _summary(records, events, issues, len(df), len(valid), ids)
    summary["inference_rows"] = len(infer)
    summary.update(
        event_count=summary["total_events"],
        needs_review_count=summary["needs_review_events"],
        **{f"{level.lower()}_count": summary[f"{level.lower()}_events"] for level in LEVEL_RANK},
        window_24h_count=(summary["window_24h"]["total_events"]
                         if summary["window_24h"]["coverage_status"] == "OBSERVED" else None),
        window_24h_note=summary["window_24h"]["coverage_note"],
    )
    return {
        "records": records, "events": events, "history": _history(infer, df.index),
        "issues": _display_issues(issues),
        "summary": summary,
        "metadata": metadata,
    }
