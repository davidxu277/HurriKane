"""最终 Critical 复核量：有后续消息事件为主结果，全部事件为补充。"""

import argparse
import hashlib
import json
from decimal import Decimal, ROUND_CEILING
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_provenance import saved_prediction_context


REQUIRED_COLUMNS = ("event_id", "final_tier", "has_future", "baseline_score", "alert_score")
TIERS = ("Safe", "Elevated", "Critical")


def _future_flag(value):
    # 不能用 astype(bool)：字符串 "False" 也会被当成 True。
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.number)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        parsed = {"true": True, "false": False, "1": True, "0": False}
        if value.strip().lower() in parsed:
            return parsed[value.strip().lower()]
    raise ValueError("has_future must contain explicit True/False or 1/0 values, with no missing values")


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
    if not frame.final_tier.isin(TIERS).all():
        raise ValueError("final_tier must contain only Safe, Elevated or Critical, with no missing labels")
    frame["has_future"] = frame.has_future.map(_future_flag).astype(bool)
    for name in ("baseline_score", "alert_score"):
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{name} must contain only finite numeric values")
        frame[name] = values
    # 两列都是排序分数。新版 alert_score 带等级兜底与微小排序项，可能略大于 1，
    # 不能把它当碰撞概率校验或截断，否则会破坏排序。
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


def _evaluate_scope(frame, scope, target_recall, bootstrap_iterations, seed):
    positive = frame.final_tier.to_numpy() == "Critical"
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
    if not n:
        warnings.append("没有符合该评估范围的事件，复核量和召回率不可评估。")
    if p < 20:
        warnings.append("高风险事件少于 20 个，样本太少，只作方向性参考。")
    if not p:
        warnings.append("没有真实高风险事件，召回率、目标复核量及其减少比例均不可评估。")
    if zero_positive:
        warnings.append(f"{zero_positive} 次 bootstrap 无正例，未纳入区间；区间以重采样存在正例为条件。")
    if scope == "all":
        warnings.append("全部事件包含无后续消息的事件，其最终标签本身已在输入中；只能作为补充结果。")
    report = {
        "scope": scope, "role": "primary" if scope == "future" else "supplementary",
        "scope_definition": "has_future == True" if scope == "future" else "all supplied events",
        "events": n, "positive_events": p, "no_future_events": int((~frame.has_future).sum()),
        "positive_definition": "final_tier == 'Critical'",
        "target_recall": target_recall,
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
        "scope": scope,
        "reviewed_events": np.arange(n + 1),
        "baseline_true_positives": base_tp, "model_true_positives": model_tp,
        "baseline_recall": base_tp / p if p else np.full(n + 1, np.nan),
        "model_recall": model_tp / p if p else np.full(n + 1, np.nan),
    })
    return report, curves


def evaluate_workload(predictions, target_recall=0.9, bootstrap_iterations=1000, seed=42):
    """返回 (报告 dict, 曲线 DataFrame)，不修改输入或模型。

    默认高风险是 final_tier == 'Critical'，不再按 y_true/Pc 阈值重建标签。
    顶层指标仅来自 has_future == True，supplementary_all 单独报告全部事件。
    两组曲线用 scope=future/all 区分，读取曲线时必须先筛选 scope。

    baseline_score 和 alert_score 都直接使用 A 输出的排序分数。按事件 ID
    打破同分，取首次达到 ceil(target_recall * 正例数) 时的复核数量。
    reduction=(baseline-model)/baseline；负数表示模型需要复核更多事件。

    每个评估范围内采用配对事件 bootstrap：同一次有放回抽样同时评估两种
    排序，保留重复副本。零正例重采样不纳入区间并单独计数；区间此时以存在
    正例为条件。原始范围无正例或为空时返回 None，不编造 0%/100% 成绩。
    """
    if isinstance(target_recall, bool) or not np.isfinite(target_recall) or not 0 < target_recall <= 1:
        raise ValueError("target_recall must be in (0, 1]")
    if isinstance(bootstrap_iterations, bool) or not isinstance(bootstrap_iterations, int) or bootstrap_iterations < 1:
        raise ValueError("bootstrap_iterations must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    frame = _validated_frame(predictions)
    future = frame[frame.has_future]
    report, primary_curve = _evaluate_scope(future, "future", target_recall, bootstrap_iterations, seed)
    supplementary, all_curve = _evaluate_scope(frame, "all", target_recall, bootstrap_iterations, seed)
    report.update(schema_version=2, target="final_tier", input_events=len(frame),
                  excluded_no_future_events=len(frame) - len(future), supplementary_all=supplementary)
    return report, pd.concat([primary_curve, all_curve], ignore_index=True)


def save_curve(report, curves, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    figure, panels = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for axes, item in zip(panels, [report, report["supplementary_all"]]):
        data = curves[curves.scope.eq(item["scope"])]
        if item["positive_events"]:
            axes.step(data.reviewed_events, data.baseline_recall, where="post", label="Provided baseline_score", linewidth=2)
            axes.step(data.reviewed_events, data.model_recall, where="post", label="Model alert_score", linewidth=2)
            axes.axhline(item["target_recall"], color="gray", linestyle="--", linewidth=1)
            for key, color in [("baseline", "C0"), ("model", "C1")]:
                point = item[key]
                axes.scatter(point["reviews"], point["recall"], color=color, zorder=3)
                right_edge = point["reviews"] >= 0.75 * item["events"]
                y_offset = (-20 if key == "baseline" else -38) if point["recall"] >= 0.95 else (-25 if key == "baseline" else 14)
                axes.annotate(f"{point['reviews']} reviews", (point["reviews"], point["recall"]),
                              xytext=(-8 if right_edge else 8, y_offset),
                              ha="right" if right_edge else "left", textcoords="offset points", color=color)
            axes.legend(loc="lower right")
        else:
            axes.text(0.5, 0.5, "No positives: recall/workload undefined", ha="center", transform=axes.transAxes)
        scope = "Future events (PRIMARY)" if item["scope"] == "future" else "All events (SUPPLEMENTARY)"
        note = " | <20 positives: directional only" if item["reference_only"] else ""
        axes.set_title(f"{scope}\n{item['events']} events, {item['positive_events']} Critical{note}")
        axes.set_xlabel("Events reviewed")
        axes.yaxis.set_major_formatter(PercentFormatter(1))
        axes.set_ylim(-0.02, 1.06)
        axes.set_xlim(0, max(1, item["events"]))
        axes.grid(alpha=0.2)
    panels[0].set_ylabel("Recall of final Critical events")
    split = "Validation (development only)" if report["split"] == "validation" else "Holdout"
    context = report.get("provenance")
    suffix = (f" | {context['variant']} saved predictions | {context['holdout_look_count']} holdout looks"
              if context else "")
    figure.suptitle(f"{split} | final_tier = Critical | target recall {report['target_recall']:.0%}{suffix}")
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _scope_markdown(report):
    p = report["positive_events"]
    base, model = report["baseline"], report["model"]
    title = "主结果：有后续消息的事件" if report["scope"] == "future" else "补充结果：全部事件"
    lines = [f"## {title}", "", f"范围：`{report['scope_definition']}`；{report['events']} 个事件，{p} 个最终 Critical 事件。", ""]
    if p:
        lines += [f"召回目标 {report['target_recall']:.0%}：至少找到 {report['target_positive_events']} / {p} 个高风险事件。", "",
                  "| 排序方式 | 需复核事件数 | 找到的高风险事件数 | 漏报数 | 实际召回率 |",
                  "| --- | ---: | ---: | ---: | ---: |",
                  f"| 输入文件 baseline_score | {base['reviews']} | {base['true_positives']} | {base['missed_events']} | {base['recall']:.1%} |",
                  f"| 模型 alert_score | {model['reviews']} | {model['true_positives']} | {model['missed_events']} | {model['recall']:.1%} |", ""]
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
    return lines + [""]


def _markdown(report):
    title = "验证集复核量分析（开发结果）" if report["split"] == "validation" else "留出集复核量分析"
    lines = [f"# {title}", "", f"输入：`{report['source_file']}`，共 {report['input_events']} 个事件。", "",
             "目标为 `final_tier == 'Critical'`；直接使用 A 输出的最终等级标签，不按 Pc 阈值或输入距离重新推导真值。", "",
             "主结果只评估 has_future=True；全部事件另列为补充，避免把标签已知的事件混入主结果。两种排序直接使用文件中的 baseline_score 和 alert_score。", ""]
    context = report.get("provenance")
    if context:
        lines += [f"模型 `{context['variant']}`，保存预测版本 `{context['prediction_model_version']}`；当前 bundle 版本 `{context['current_bundle_model_version']}`。", "",
                  f"**留出集共看了 {context['holdout_look_count']} 次**；本次仅重算保存预测的复核量，不训练或重新预测，不增加一次模型评估。", ""]
        lines += [f"- {warning}" for warning in context["warnings"]] + [""]
    lines += _scope_markdown(report) + _scope_markdown(report["supplementary_all"])
    boot = report["bootstrap"]
    provenance_command = (f" --metrics \"{context['metrics_file']}\" --model-metadata \"{context['metadata_file']}\""
                          f" --holdout-look-count {context['holdout_look_count']} --source-commit {context['source_commit']}"
                          if context else "")
    lines += ["这是对已知标签的事后排序评估；不能用它承诺未标注数据也达到相同召回率。验证集结果用于开发，最终效果需在模型和规则固定后用留出集确认。", "",
              "不根据这份结果重抽事件切分、选择更好看的种子或调整留出集。脚本不训练模型、不读取原始数据和其他预测文件。", "",
              f"输入文件 SHA256：`{report['source_sha256']}`。", "",
              "## 复现命令", "", "在项目根目录、安装了 pandas、NumPy、Matplotlib 的 Python 环境中运行：", "",
              "```powershell",
              f"python evaluate_workload.py --predictions \"{report['source_file']}\" --split {report['split']} --target-recall {report['target_recall']} --bootstrap {boot['iterations']} --seed {boot['seed']}{provenance_command}",
              "```", "",
              "JSON 顶层指标是主结果，supplementary_all 为补充结果；读取 `_curve.csv` 必须按 scope=future/all 分组，不能把两组曲线连成一条。`_curve.png` 分两栏展示。验证集产物必须标注为开发结果。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--split", choices=["validation", "holdout"])
    parser.add_argument("--target-recall", type=float, default=0.9)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", help="JSON 输出；同时生成同名 .md、_curve.csv 和 _curve.png")
    parser.add_argument("--metrics", help="与预测配套的已保存指标 JSON；连同 --model-metadata 记录来源")
    parser.add_argument("--model-metadata", help="当前 bundle metadata JSON；只读取，不加载模型")
    parser.add_argument("--holdout-look-count", type=int)
    parser.add_argument("--source-commit", help="保存预测及模型材料所在的固定 Git 提交号")
    args = parser.parse_args()
    source = Path(args.predictions)
    known = {"validation_predictions.csv": "validation", "holdout_predictions.csv": "holdout"}
    split = args.split or known.get(source.name)
    if split is None:
        parser.error("非标准预测文件名需用 --split 明确 validation 或 holdout")
    if source.name in known and split != known[source.name]:
        parser.error("--split 与标准预测文件名不符，不能把验证集标为留出集")
    context = None
    if any(value is not None for value in (args.metrics, args.model_metadata, args.holdout_look_count, args.source_commit)):
        if split != "holdout" or not all((args.metrics, args.model_metadata, args.holdout_look_count, args.source_commit)):
            parser.error("留出来源需同时提供 --metrics、--model-metadata、--holdout-look-count、--source-commit")
        context = saved_prediction_context(args.metrics, args.model_metadata, args.holdout_look_count, args.source_commit)
    output = Path(args.output) if args.output else Path(f"reports/workload_{split}.json")
    if output.suffix.lower() != ".json":
        parser.error("--output must end in .json")
    curves_path = output.with_name(output.stem + "_curve.csv")
    image_path = output.with_name(output.stem + "_curve.png")
    inputs = {source.resolve()} | {Path(path).resolve() for path in (args.metrics, args.model_metadata) if path}
    if inputs & {output.resolve(), output.with_suffix(".md").resolve(), curves_path.resolve(), image_path.resolve()}:
        parser.error("输出不能覆盖输入预测文件")
    frame = pd.read_csv(source)
    report, curves = evaluate_workload(frame, args.target_recall, args.bootstrap, args.seed)
    report.update(split=split, purpose="development_only" if split == "validation" else "final_evaluation",
                  source_file=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    if context:
        report.update(provenance=context, purpose="saved_prediction_reanalysis")
        metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
        for scope, result in (("future", report), ("all", report["supplementary_all"])):
            expected = metrics["tiers"][scope]
            if result["events"] != expected["events"] or result["positive_events"] != expected["critical"]["positive_events"]:
                raise ValueError("Saved metrics cohort does not match predictions")
            if args.target_recall == 0.9:
                for method in ("baseline", "model"):
                    if result[method]["reviews"] != expected["critical"]["review_for_90_recall"][method]:
                        raise ValueError("Saved metrics review count does not match predictions")
    output.parent.mkdir(parents=True, exist_ok=True)
    save_curve(report, curves, image_path)
    curves.to_csv(curves_path, index=False)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(_markdown(report), encoding="utf-8")
    print(f"Primary has_future=True: {report['events']} events; Critical: {report['positive_events']}; target recall: {args.target_recall:.0%}")
    print(f"Reviews: baseline={report['baseline']['reviews']}, model={report['model']['reviews']}")
    print(f"Report: {output}; curve: {image_path}")


if __name__ == "__main__":
    main()
