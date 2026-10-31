"""原始消息的最小输入校验，不训练模型、不选择标签或筛选时间窗口。"""

from collections.abc import Mapping

import numpy as np
import pandas as pd


# 原始 CSV 字段；risk_delta 等由 ml.build_samples 生成，不要求用户上传。
RAW_FEATURES = (
    "time_to_tca", "miss_distance", "relative_speed",
    "relative_position_r", "relative_position_t", "relative_position_n",
    "c_object_type", "c_sigma_r", "t_sigma_r", "max_risk_estimate", "F10", "AP",
)
DERIVED_FEATURES = {
    "risk_delta", "miss_distance_delta", "message_count", "current_tier",
    "risk_max", "risk_std", "risk_slope", "miss_distance_min",
    "miss_distance_slope", "c_sigma_r_delta",
    "sigma_total_r", "z_r", "sigma_total_t", "z_t", "sigma_total_n", "z_n",
    "log_miss", "margin_1k", "margin_5k", "margin_1k_sigma", "margin_5k_sigma",
    "projected_miss", "p_within_1k", "p_within_5k",
}
ISSUE_COLUMNS = [
    "input_record_id", "row_position", "event_id", "column", "code",
    "severity", "value", "message",
]


def validate_input(
    df: pd.DataFrame, metadata: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """返回 (可处理记录, 问题清单)，不修改原 DataFrame。

    metadata 至少提供 task（final/current），features 可省略。必需列为
    当前 ml.py 使用的原始字段，加上 metadata 中额外声明的非派生特征；
    final 另需历史 risk，current 推理不要求标签 risk。

    input_record_id 使用原 DataFrame.index，与 ml.py 保持一致；不重置
    返回记录的索引。row_position 是从 0 开始的输入行位置，重复索引也能
    定位。已有 input_record_id 列也按该约定重建。文件级问题的行号为空。

    ERROR 行从可处理记录中排除，WARNING 行保留；缺少必需列或列名重复
    时整个输入不可处理。数值字符串转为数字，空字符串视为缺失。除事件
    ID、时间及 final 的 risk 外，数值缺失保留为 NaN；类别缺失留给 ML
    转 UNKNOWN。负时间和零距离均允许，不拟合或应用 sigma 阈值。

    调用方必须展示 issues，按事件传播质量提示并交代被排除的事件。
    尤其不能把错误的最新消息被排除后所剩的旧消息当成可靠的最新状态。
    ml.predict_batch 不会自动传递这里的质量问题，service 需要显式关联。
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    task = metadata.get("task")
    if task not in {"final", "current"}:
        raise ValueError("metadata.task must be 'final' or 'current'")
    features = metadata.get("features", RAW_FEATURES)
    if (not isinstance(features, (list, tuple))
            or not all(isinstance(name, str) for name in features)):
        raise ValueError("metadata.features must be a list of column names")
    if task == "current" and "risk" in features:
        raise ValueError("current mode must not use target risk as a feature")
    categorical = set(metadata.get("categorical_features", ["c_object_type"]))
    categorical.add("c_object_type")
    required = list(dict.fromkeys([
        "event_id", *RAW_FEATURES,
        *(name for name in features if name not in DERIVED_FEATURES),
        *(["risk"] if task == "final" else []),
    ]))
    issues = []
    rejected = np.zeros(len(df), dtype=bool)

    def report(column, code, severity, message, mask=None):
        if mask is None:
            issues.append(dict(zip(ISSUE_COLUMNS, [
                pd.NA, pd.NA, pd.NA, column, code, severity, pd.NA, message,
            ])))
            return
        positions = np.flatnonzero(np.asarray(mask, dtype=bool))
        if severity == "ERROR":
            rejected[positions] = True
        for pos in positions:
            issues.append(dict(zip(ISSUE_COLUMNS, [
                df.index[pos], int(pos), df["event_id"].iloc[pos], column,
                code, severity, df[column].iloc[pos], message,
            ])))

    # 重复列不能安全地按列名取值，必须在其他逐列校验前处理。
    if not df.columns.is_unique:
        for name in df.columns[df.columns.duplicated()].unique():
            report(name, "DUPLICATE_COLUMN", "ERROR", "Duplicate column name")
        return df.iloc[:0].copy(), pd.DataFrame(issues, columns=ISSUE_COLUMNS)

    missing_columns = [name for name in required if name not in df]
    if missing_columns:
        for name in missing_columns:
            report(name, "MISSING_COLUMN", "ERROR", "Required input column is missing")
        return df.iloc[:0].copy(), pd.DataFrame(issues, columns=ISSUE_COLUMNS)

    cleaned = df.copy(deep=True)
    cleaned["input_record_id"] = df.index
    if df.empty:
        report(None, "EMPTY_INPUT", "WARNING", "Input contains no records")
        return cleaned, pd.DataFrame(issues, columns=ISSUE_COLUMNS)

    # ID 不强制转数字，保留字符串 ID（例如 001）；只拒绝缺失或无穷 ID。
    ids = df["event_id"]
    blank_ids = ids.isna() | ids.map(lambda value: isinstance(value, str) and not value.strip())
    report("event_id", "MISSING_VALUE", "ERROR", "event_id is required", blank_ids)
    numeric_ids = pd.to_numeric(ids, errors="coerce")
    report("event_id", "NON_FINITE", "ERROR", "event_id must not be infinite",
           numeric_ids.isin([np.inf, -np.inf]))

    critical = {"time_to_tca"} | ({"risk"} if task == "final" else set())
    for name in required:
        if name == "event_id":
            continue
        source = df[name]
        blank = source.isna() | source.map(
            lambda value: isinstance(value, str) and not value.strip()
        )
        if name in categorical:
            cleaned[name] = source.mask(blank)
            report(name, "MISSING_VALUE", "WARNING",
                   "Category is missing; ML will use UNKNOWN", blank)
            continue
        values = pd.to_numeric(source.mask(blank), errors="coerce").astype(float)
        report(name, "INVALID_NUMERIC", "ERROR", "Value cannot be parsed as a number",
               ~blank & values.isna())
        report(name, "NON_FINITE", "ERROR", "Infinite input value is not allowed",
               np.isinf(values))
        report(name, "MISSING_VALUE", "ERROR" if name in critical else "WARNING",
               "Required value is missing" if name in critical else
               "Missing numeric value retained; not filled with zero", blank)
        if name in {"miss_distance", "c_sigma_r", "t_sigma_r"}:
            report(name, "NEGATIVE_VALUE", "ERROR",
                   "Distance and radial sigmas must be nonnegative",
                   values.lt(0) & np.isfinite(values))
        if name == "risk":
            report(name, "INVALID_LOG10_PC", "ERROR", "log10(Pc) must be <= 0",
                   values.gt(0) & np.isfinite(values))
        cleaned[name] = values

    return cleaned.iloc[np.flatnonzero(~rejected)].copy(), pd.DataFrame(
        issues, columns=ISSUE_COLUMNS
    )
