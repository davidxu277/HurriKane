"""比较同一召回目标下的事件复核量；只读已有预测，不训练或调整模型。"""

import argparse
import hashlib
import json
from decimal import Decimal, ROUND_CEILING
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ("event_id", "y_true", "baseline_score", "alert_score")


def _validated_frame(df):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("predictions must be a pandas DataFrame")
    if not df.columns.is_unique:
        raise ValueError("Duplicate column names are not allowed")
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Prediction file contains no events")
    frame = df[list(REQUIRED_COLUMNS)].copy()
    ids = frame.event_id
    if (ids.isna().any() or ids.astype(str).str.strip().eq("").any()
            or pd.to_numeric(ids, errors="coerce").isin([np.inf, -np.inf]).any()):
        raise ValueError("event_id must be present and finite")
    if ids.duplicated().any():
        raise ValueError("Expected one row per event; duplicate event_id found")
    for name in REQUIRED_COLUMNS[1:]:
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{name} must contain only finite numeric values")
        frame[name] = values
    if frame.y_true.gt(0).any():
        raise ValueError("y_true must be log10(Pc) <= 0")
    if not frame.alert_score.between(0, 1).all():
        raise ValueError("alert_score must be in [0, 1]")
    # 数字 ID 按数字排序，其他 ID 按字符串排序；行输入顺序不决定并列次序。
    key = ids if pd.api.types.is_numeric_dtype(ids) else ids.astype(str)
    if key.duplicated().any():
        raise ValueError("event_id values become ambiguous when ordered as strings")
    order = np.argsort(key.to_numpy(), kind="stable")
    frame = frame.iloc[order].reset_index(drop=True)
    return frame


def _ranking(scores):
    # 输入已按 event_id 排序；稳定排序保证同分时仍按 event_id 顺序。
    return np.argsort(-scores, kind="stable")


def _target_count(recall, positives):
    # 避免 0.07 * 100 的浮点误差把目标从 7 个错误上取整为 8 个。
    return int((Decimal(str(recall)) * positives).to_integral_value(rounding=ROUND_CEILING))


def _workload(positive, order, target):
    cumulative = np.cumsum(positive[order])
    if target is None:
        return {
            "reviews": None, "review_fraction": None, "recall": None,
            "true_positives": None, "missed_events": None,
        }
    reviews = int(np.searchsorted(cumulative, target) + 1)
    found = int(cumulative[reviews - 1])
    positives = int(positive.sum())
    return {
        "reviews": reviews, "review_fraction": reviews / len(positive),
        "recall": found / positives, "true_positives": found,
        "missed_events": positives - found,
    }


def evaluate_workload(
    predictions, target_recall=0.9, positive_threshold=-6.0,
    bootstrap_iterations=1000, seed=42,
):
    """返回 (报告 dict, 曲线 DataFrame)，不修改输入。

    高风险仅按 y_true >= positive_threshold 定义，不计距离规则。两种排序
    都看同一批事件，取首次达到 ceil(target_recall * 正例数) 的复核数量。
    reduction=(baseline-model)/baseline；负数表示模型需要看更多事件。

    bootstrap 用同一次事件有放回抽样同时评估两种排序，保留重复抽到的
    事件。零正例的重采样无法定义召回率，会单独计数并排除出区间；区间
    此时以存在正例的重采样为条件。没有原始正例时输出 None，不编造数值。
    """
    if not np.isfinite(target_recall) or not 0 < target_recall <= 1:
        raise ValueError("target_recall must be in (0, 1]")
    if not np.isfinite(positive_threshold) or positive_threshold > 0:
        raise ValueError("positive_threshold must be finite and <= 0")
    if isinstance(bootstrap_iterations, bool) or not isinstance(bootstrap_iterations, int) or bootstrap_iterations < 1:
        raise ValueError("bootstrap_iterations must be a positive integer")
    frame = _validated_frame(predictions)
    positive = frame.y_true.to_numpy() >= positive_threshold
    baseline_scores = frame.baseline_score.to_numpy()
    model_scores = frame.alert_score.to_numpy()
    n, p = len(frame), int(positive.sum())
    target = _target_count(target_recall, p) if p else None
    baseline_order, model_order = _ranking(baseline_scores), _ranking(model_scores)
    baseline = _workload(positive, baseline_order, target)
    model = _workload(positive, model_order, target)
    reduction = (baseline["reviews"] - model["reviews"]) / baseline["reviews"] if p else None

    rng = np.random.default_rng(seed)
    reductions = []
    zero_positive = 0
    if p:
        for _ in range(bootstrap_iterations):
            draw = rng.integers(0, n, size=n)
            labels = positive[draw]
            positives = int(labels.sum())
            if not positives:
                zero_positive += 1
                continue
            goal = _target_count(target_recall, positives)
            # 同分时按原 event_id 次序，重复副本的标签相同。
            base_order = np.lexsort((draw, -baseline_scores[draw]))
            rank_order = np.lexsort((draw, -model_scores[draw]))
            base_n = _workload(labels, base_order, goal)["reviews"]
            model_n = _workload(labels, rank_order, goal)["reviews"]
            reductions.append((base_n - model_n) / base_n)
    else:
        zero_positive = bootstrap_iterations
    lower, upper = (float(value) for value in np.quantile(reductions, [0.025, 0.975])) if len(reductions) >= 2 else (None, None)
    warnings = []
    if p < 20:
        warnings.append("高风险事件少于 20 个，样本太少，只作方向性参考。")
    if not p:
        warnings.append("没有真实高风险事件，召回率、目标复核量及其减少比例均不可评估。")
    if zero_positive:
        warnings.append(f"{zero_positive} 次 bootstrap 无正例，未纳入区间；区间以重采样存在正例为条件。")
    report = {
        "events": n, "positive_events": p,
        "positive_definition": f"y_true >= {positive_threshold:g} (log10(Pc))",
        "positive_threshold": positive_threshold, "target_recall": target_recall,
        "target_positive_events": target, "reference_only": p < 20,
        "baseline": baseline, "model": model,
        "review_reduction_fraction": reduction,
        "bootstrap": {
            "method": "paired non-stratified event bootstrap, percentile interval",
            "iterations": bootstrap_iterations, "seed": seed,
            "valid_samples": len(reductions), "zero_positive_samples": zero_positive,
            "confidence_level": 0.95, "ci95_lower": lower, "ci95_upper": upper,
        },
        "warnings": warnings,
        "tie_break": "score descending, event_id ascending; numeric IDs sort numerically",
    }
    base_tp = np.r_[0, np.cumsum(positive[baseline_order])]
    model_tp = np.r_[0, np.cumsum(positive[model_order])]
    curves = pd.DataFrame({
        "reviewed_events": np.arange(n + 1),
        "baseline_true_positives": base_tp, "model_true_positives": model_tp,
        "baseline_recall": base_tp / p if p else np.full(n + 1, np.nan),
        "model_recall": model_tp / p if p else np.full(n + 1, np.nan),
    })
    return report, curves


def save_curve(report, curves, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    figure, axes = plt.subplots(figsize=(9, 5.5))
    if report["positive_events"]:
        axes.step(curves.reviewed_events, curves.baseline_recall, where="post", label="Latest-risk baseline", linewidth=2)
        axes.step(curves.reviewed_events, curves.model_recall, where="post", label="Classifier ranking", linewidth=2)
        axes.axhline(report["target_recall"], color="gray", linestyle="--", linewidth=1)
        for key, color in [("baseline", "C0"), ("model", "C1")]:
            point = report[key]
            axes.scatter(point["reviews"], point["recall"], color=color, zorder=3)
            axes.annotate(f"{point['reviews']} reviews", (point["reviews"], point["recall"]),
                          xytext=(8, -25 if key == "baseline" else 14), textcoords="offset points", color=color)
        axes.legend(loc="lower right")
    else:
        axes.text(0.5, 0.5, "No positive events: recall/workload undefined", ha="center", transform=axes.transAxes)
    scope = "Validation (development only)" if report["split"] == "validation" else "Holdout"
    note = " | directional evidence only (<20 positives)" if report["reference_only"] else ""
    axes.set_title(f"{scope}: {report['events']} events, {report['positive_events']} positives\nTarget recall {report['target_recall']:.0%}{note}")
    axes.set_xlabel("Events reviewed (lower is better at the same recall)")
    axes.set_ylabel("Recall of Pc-defined high-risk events")
    axes.yaxis.set_major_formatter(PercentFormatter(1))
    axes.set_ylim(-0.02, 1.06)
    axes.set_xlim(0, report["events"])
    axes.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _markdown(report):
    p = report["positive_events"]
    base, model = report["baseline"], report["model"]
    title = "验证集复核量分析（开发结果）" if report["split"] == "validation" else "留出集复核量分析"
    lines = [f"# {title}", "", f"输入：`{report['source_file']}`。共 {report['events']} 个事件，{p} 个真实高风险事件。", "",
             f"高风险定义：`{report['positive_definition']}`；只看 Pc 标签，不把距离规则算成模型效果。", ""]
    if p:
        lines += [f"召回目标 {report['target_recall']:.0%}：至少找到 {report['target_positive_events']} / {p} 个高风险事件。", "",
                  "| 排序方式 | 需复核事件数 | 找到的高风险事件数 | 漏报数 | 实际召回率 |",
                  "| --- | ---: | ---: | ---: | ---: |",
                  f"| 最新 risk 基线 | {base['reviews']} | {base['true_positives']} | {base['missed_events']} | {base['recall']:.1%} |",
                  f"| 分类器 alert_score | {model['reviews']} | {model['true_positives']} | {model['missed_events']} | {model['recall']:.1%} |", ""]
        reduction = report["review_reduction_fraction"]
        direction = "减少" if reduction >= 0 else "增加"
        lines += [f"模型相对基线的复核量**{direction} {abs(reduction):.1%}**。", "",
                  "减少比例定义：`(基线复核量 - 模型复核量) / 基线复核量`；负数代表模型增加工作量。", ""]
    else:
        lines += ["没有正例：召回率、复核量和减少比例均为 N/A，不能报告为 0% 或 100%。", ""]
    boot = report["bootstrap"]
    interval = "N/A" if boot["ci95_lower"] is None else f"[{boot['ci95_lower']:.1%}, {boot['ci95_upper']:.1%}]"
    lines += [f"减少比例的 bootstrap 95% 区间：**{interval}**。", "",
              f"使用配对事件有放回抽样 {boot['iterations']} 次，种子 {boot['seed']}；有效 {boot['valid_samples']} 次，无正例 {boot['zero_positive_samples']} 次。两种排序使用同一批抽样，重复抽到的事件不去重。", ""]
    if boot["ci95_lower"] is not None and boot["ci95_lower"] <= 0 <= boot["ci95_upper"]:
        lines += ["区间跨过 0，不能据此断言模型稳定减少或增加了复核量。", ""]
    lines += [f"- {warning}" for warning in report["warnings"]]
    lines += ["", "这是对已知标签的事后排序评估；不能用它承诺未标注数据也达到相同召回率。验证集结果用于开发，最终效果需在模型和规则固定后用留出集确认。", "",
              "不根据这份结果重抽事件切分、选择更好看的种子或调整留出集。脚本不训练模型、不读取原始数据和其他预测文件。", "",
              f"输入文件 SHA256：`{report['source_sha256']}`。", "",
              "## 复现命令", "", "在项目根目录、安装了 pandas、NumPy、Matplotlib 的 Python 环境中运行：", "",
              "```powershell",
              f"python evaluate_workload.py --predictions \"{report['source_file']}\" --split {report['split']} --target-recall {report['target_recall']} --positive-threshold {report['positive_threshold']} --bootstrap {boot['iterations']} --seed {boot['seed']}",
              "```", "",
              "JSON 用于页面指标，`_curve.csv` 用于交互曲线，`_curve.png` 可直接展示，本文可交给 A/B 核对结论。验证集产物必须标注为开发结果。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--split", choices=["validation", "holdout"])
    parser.add_argument("--target-recall", type=float, default=0.9)
    parser.add_argument("--positive-threshold", type=float, default=-6.0)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", help="JSON 输出；同时生成同名 .md、_curve.csv 和 _curve.png")
    args = parser.parse_args()
    source = Path(args.predictions)
    known = {"validation_predictions.csv": "validation", "holdout_predictions.csv": "holdout"}
    split = args.split or known.get(source.name)
    if split is None:
        parser.error("非标准预测文件名需用 --split 明确 validation 或 holdout")
    if source.name in known and split != known[source.name]:
        parser.error("--split 与标准预测文件名不符，不能把验证集标为留出集")
    output = Path(args.output) if args.output else Path(f"reports/workload_{split}.json")
    if output.suffix.lower() != ".json":
        parser.error("--output must end in .json")
    curves_path = output.with_name(output.stem + "_curve.csv")
    image_path = output.with_name(output.stem + "_curve.png")
    if source.resolve() in {output.resolve(), output.with_suffix(".md").resolve(), curves_path.resolve(), image_path.resolve()}:
        parser.error("输出不能覆盖输入预测文件")
    frame = pd.read_csv(source)
    report, curves = evaluate_workload(frame, args.target_recall, args.positive_threshold, args.bootstrap, args.seed)
    report.update(split=split, purpose="development_only" if split == "validation" else "final_evaluation",
                  source_file=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    output.parent.mkdir(parents=True, exist_ok=True)
    save_curve(report, curves, image_path)
    curves.to_csv(curves_path, index=False)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(_markdown(report), encoding="utf-8")
    print(f"Events: {report['events']}; positives: {report['positive_events']}; target recall: {args.target_recall:.0%}")
    print(f"Reviews: baseline={report['baseline']['reviews']}, model={report['model']['reviews']}")
    print(f"Report: {output}; curve: {image_path}")


if __name__ == "__main__":
    main()
