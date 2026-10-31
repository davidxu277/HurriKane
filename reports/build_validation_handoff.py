"""从已生成的官方口径验证报告构建 A/B 交接包，不训练或读取留出集。"""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
HANDOFF = REPORTS / "handoff"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def build():
    source = REPORTS / "validation_predictions.csv"
    report = json.loads((REPORTS / "workload_validation.json").read_text(encoding="utf-8"))
    metadata = json.loads((ROOT / "models/metadata.json").read_text(encoding="utf-8"))
    if (report.get("target") != "final_tier" or report.get("scope") != "future"
            or report.get("split") != "validation" or report.get("purpose") != "development_only"
            or report["source_sha256"] != sha256(source)):
        raise ValueError("报告口径或预测文件哈希不符，请先重新运行 evaluate_workload.py")
    raw = pd.read_csv(source)
    # 显式解析布尔值，拒绝把字符串 False 当作真。
    future = raw.has_future.astype(str).str.strip().str.lower().map({"true": True, "false": False})
    if future.isna().any():
        raise ValueError("has_future must be True/False")
    frame = raw.loc[future].copy()
    for score, prefix in [("baseline_score", "baseline"), ("alert_score", "model")]:
        ordered = frame.sort_values([score, "event_id"], ascending=[False, True], kind="stable")
        ranks = dict(zip(ordered.event_id, range(1, len(ordered) + 1)))
        frame[f"{prefix}_rank"] = frame.event_id.map(ranks)
        frame[f"in_{prefix}_review_budget"] = frame[f"{prefix}_rank"].le(report[prefix]["reviews"])
    frame["rank_delta_model_minus_baseline"] = frame.model_rank - frame.baseline_rank
    critical = frame.loc[frame.final_tier.eq("Critical")].copy()
    assert len(frame) == report["events"] and len(critical) == report["positive_events"]
    for prefix in ("baseline", "model"):
        assert int(critical[f"in_{prefix}_review_budget"].sum()) == report[prefix]["true_positives"]
    missed = critical.loc[~critical.in_model_review_budget].sort_values("model_rank", ascending=False)
    critical = critical.sort_values(["rank_delta_model_minus_baseline", "event_id"], ascending=[False, True])
    both_missed = set(missed.event_id) == set(critical.loc[~critical.in_baseline_review_budget, "event_id"])
    a_dir, b_dir = HANDOFF / "A", HANDOFF / "B"
    a_dir.mkdir(parents=True, exist_ok=True)
    b_dir.mkdir(parents=True, exist_ok=True)
    critical.to_csv(a_dir / "critical_rank_review.csv", index=False, encoding="utf-8-sig")
    missed.to_csv(a_dir / "critical_outside_model_budget.csv", index=False, encoding="utf-8-sig")
    frame.sort_values("event_id").to_csv(a_dir / "all_future_event_ranks.csv", index=False, encoding="utf-8-sig")
    shared = ["workload_validation.md", "workload_validation.json", "workload_validation_curve.csv", "workload_validation_curve.png"]
    for directory in (a_dir, b_dir):
        for name in shared:
            shutil.copy2(REPORTS / name, directory / name)
        shutil.copy2(ROOT / "models/metadata.json", directory / "metadata.json")
    shutil.copy2(source, a_dir / source.name)
    # 一张真实验证回放卡片；不把排名遗漏说成固定分类阈值的漏报。
    candidate = missed.iloc[0]
    card = {
        "type": "validation_replay", "scope": "has_future == True",
        "event_id": int(candidate.event_id), "current_tier": candidate.current_tier,
        "predicted_final_tier": candidate.predicted_final_tier, "true_final_tier": candidate.final_tier,
        "baseline_rank": int(candidate.baseline_rank), "model_rank": int(candidate.model_rank),
        "model_review_budget": report["model"]["reviews"],
        "description": "该事件未进入本次目标召回预算的复核队列；这是验证回放案例，需 A 确认失败原因。",
    }
    write_json(a_dir / "candidate_failure_case.json", card)
    write_json(b_dir / "candidate_failure_case.json", card)
    base, model, boot = report["baseline"], report["model"], report["bootstrap"]
    reduction = report["review_reduction_fraction"]
    change_word = "减少" if reduction >= 0 else "增加"
    review_difference = abs(model["reviews"] - base["reviews"])
    conclusion = "本次点估计略有减负" if reduction > 0 else "本次未观察到减负"
    a_text = f"""# 给 A：新版验证集评估交接

本包为验证集开发结果，目标 `final_tier == Critical`，主范围 `has_future == True`。
配套最终方案：`{metadata.get('variant')}`，模型版本 `{metadata.get('model_version')}`。
本次仅重新分析已保存预测，没有训练、改变模型/阈值、重抽切分或运行留出评估。

## 已完成的核对

- 上传预测共 {report['input_events']} 个事件；主范围 {report['events']} 个，Critical {report['positive_events']} 个。
- 达到至少 {report['target_recall']:.0%} 召回需找到 {report['target_positive_events']} 个，即实际召回 {model['recall']:.2%}。
- 基线复核 {base['reviews']} 个、模型 {model['reviews']} 个；两者均找到 {model['true_positives']} 个，剩余 {model['missed_events']} 个。
- 复核量减少比例 {report['review_reduction_fraction']:.2%}（负数表示增加）；bootstrap 95% 区间 [{boot['ci95_lower']:.2%}, {boot['ci95_upper']:.2%}]，跨 0。
- 两种排序在各自预算外的 Critical 集合是否相同：{both_missed}。这不代表两种排序整体相同。
- 不将旧版 Pc 报告的 56/119 作为新版结果，也不把距离单独算成模型功劳；真值直接取 A 输出的 final_tier。

## 请 A 做什么

1. 看 `critical_rank_review.csv`：按 `rank_delta_model_minus_baseline` 降序，正数表示模型比基线排得更后。检查哪些事件受影响，给出有证据的原因。
2. 看 `critical_outside_model_budget.csv`：这些是真实 Critical，但未进入模型前 {model['reviews']} 个事件的复核队列。它不是按分类阈值定义的漏报清单。
3. 同时看 `all_future_event_ranks.csv` 中靠前的非 Critical；本次模型复核量较基线{change_word} {review_difference} 个事件，找到的 Critical 数相同。
4. 核对评估输入、标签、消息截止时刻及模型版本；当前远程方案已标记为最终方案，本轮请完成误差解释和案例确认，不为改变报告数字而调整模型。
5. 若团队后续确实变更模型，交付更新的 validation_predictions.csv、metadata 和改进前后对比，让 C 用同一脚本重算。冻结后再由 A 交付留出预测，不能按留出表现调整模型。
6. 确认一个失败案例供 B 展示。候选事件 {card['event_id']}：当前 {card['current_tier']}、模型预测 {card['predicted_final_tier']}、真实最终 {card['true_final_tier']}，模型排第 {card['model_rank']}，基线第 {card['baseline_rank']}。这只是验证回放，尚无物理原因结论。

## 文件说明

- validation_predictions.csv：本次分析使用的完整预测输入。
- critical_rank_review.csv：主范围全部 Critical 的排名、原始分数及前述预算标记。
- all_future_event_ranks.csv：主范围全部事件，供检查前排误警和排名变化。
- critical_outside_model_budget.csv：模型目标复核预算之外的 Critical。
- workload_validation.*：报告和机器可读结果；带 _curve 的文件为曲线数据及图片。
- metadata.json：交接时仓库内的模型元数据副本。
- candidate_failure_case.json：待 A 确认的验证回放案例。
- manifest.json：预测文件、交付文件与代码版本的哈希记录。

复现：项目根目录运行 `python evaluate_workload.py --predictions reports/validation_predictions.csv --split validation --bootstrap 1000 --seed 42`。
"""
    (a_dir / "README_A.md").write_text(a_text, encoding="utf-8")
    b_text = f"""# 给 B：新版验证集评估页面交接

标题必须标注“验证集开发结果”。当前目标为 final_tier=Critical，主结果仅取 has_future=True。
本包是离线评估产物，不需要在页面训练或运行 bootstrap。
配套最终方案：`{metadata.get('variant')}`，模型版本 `{metadata.get('model_version')}`。

## 本次应展示的主结果

| 指标 | 值 |
| --- | ---: |
| 主范围事件数 | {report['events']} |
| 真实最终 Critical | {report['positive_events']} |
| 目标召回率 | {report['target_recall']:.0%} |
| 实际找到 / 总 Critical | {model['true_positives']} / {report['positive_events']} |
| 实际召回率 | {model['recall']:.2%} |
| 基线复核量 | {base['reviews']} |
| 模型复核量 | {model['reviews']} |
| 复核量{change_word} | {abs(reduction):.2%} |
| 减少比例的 95% 区间 | [{boot['ci95_lower']:.2%}, {boot['ci95_upper']:.2%}] |

本次页面结论写：“达到至少 {report['target_recall']:.0%} 的验证集 Critical 召回时，基线复核 {base['reviews']} 个、模型复核 {model['reviews']} 个；{conclusion}，区间跨 0，尚不能确认稳定改善。”

## 接入方法

1. 将本包四个 workload_validation 产物放入页面项目的 reports/，文件名保持一致。先核对 manifest 中的哈希，避免混用旧版。
2. 读取 JSON，要求 schema_version=2、target=final_tier、scope=future、purpose=development_only。
3. 顶层 baseline/model 等字段是主结果。supplementary_all 才是全部事件补充结果，不能替换顶部主结果。
4. 曲线 CSV 含 scope=future/all 两组，先筛选再画；PNG 已经分成主结果和补充结果两栏。
5. 复核量减少比例为负时，显示“增加”，不要把负数写成提升；区间跨 0 时提示不确定性。
6. candidate_failure_case.json 标为“验证集历史回放，待 A 确认原因”；最终展示内容请等 A 确认。不要把真值当作实时输入证据。

## 可复制的 Streamlit 评估区

以下代码放在现有 app.py 中，不替换已接好的 analyze_csv 主流程。示例假设 app.py 位于项目根目录。

```python
import json
from pathlib import Path

import pandas as pd
import streamlit as st

report_dir = Path(__file__).resolve().parent / "reports"
metrics = json.loads((report_dir / "workload_validation.json").read_text(encoding="utf-8"))
if (metrics.get("schema_version") != 2 or metrics.get("target") != "final_tier"
        or metrics.get("scope") != "future" or metrics.get("purpose") != "development_only"):
    st.error("评估报告版本或口径不符，请向 C 获取新版交接包。")
else:
    st.subheader("验证集开发结果")
    st.caption("主结果：有后续消息事件；高风险定义为最终等级 Critical。")
    c1, c2, c3 = st.columns(3)
    c1.metric("基线复核事件数", metrics["baseline"]["reviews"])
    c2.metric("模型复核事件数", metrics["model"]["reviews"])
    reduction = metrics["review_reduction_fraction"]
    change = "不可评估" if reduction is None else (
        f"增加 {{-reduction:.1%}}" if reduction < 0 else f"减少 {{reduction:.1%}}"
    )
    c3.metric("复核量变化", change)
    found = metrics["model"]["true_positives"]
    total = metrics["positive_events"]
    st.write(f"主范围 {{metrics['events']}} 个事件；找到 {{found}} / {{total}} 个 Critical。")
    interval = metrics["bootstrap"]
    lo, hi = interval["ci95_lower"], interval["ci95_upper"]
    if lo is not None and hi is not None:
        st.write(f"减少比例的 bootstrap 95% 区间：[{{lo:.1%}}, {{hi:.1%}}]")
        if lo <= 0 <= hi:
            st.info("区间跨 0，尚不能确认模型稳定减少或增加了复核量。")
    data = pd.read_csv(report_dir / "workload_validation_curve.csv")
    primary = data.loc[data["scope"].eq("future")]
    st.line_chart(primary, x="reviewed_events", y=["baseline_recall", "model_recall"])
    with st.expander("全部事件补充结果（包括标签已在输入中的事件）"):
        st.json(metrics["supplementary_all"])
        st.image(str(report_dir / "workload_validation_curve.png"))
```

这段示例按主范围绘制交互曲线；需要精确阶梯形状时，直接展示随包 PNG，或用 Plotly 设置 step/line_shape='hv'。CSV 的召回率为 0—1，不是百分数字符串。

## 交接边界

- 当前报告不是留出集效果。模型固定后的正式效果，等待 A 的 holdout_predictions.csv，由 C 用同一方法再分析。
- 全部事件有 {report['supplementary_all']['events']} 个，其中 {report['excluded_no_future_events']} 个没有后续消息；这些事件的最终标签已在输入中，所以只能补充展示。
- 服务层上传分析结果和这里的验证评估是两个数据源；不要用当前上传文件的事件数替换验证集事件数。
- manifest.json 标识这批输入与文件版本；metadata.json 提供交接时的模型版本。
"""
    (b_dir / "README_B.md").write_text(b_text, encoding="utf-8")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    base_manifest = {
        "target": "final_tier", "split": "validation", "purpose": "development_only",
        "primary_scope": "has_future == True", "positive_definition": "final_tier == 'Critical'",
        "source_prediction_sha256": sha256(source), "analysis_code_sha256": sha256(ROOT / "evaluate_workload.py"),
        "source_commit": commit, "accompanying_model_version": metadata.get("model_version"),
        "bootstrap_iterations": 1000, "seed": 42,
        "primary_events": report["events"], "primary_critical_events": report["positive_events"],
        "note": "模型版本来自当前配套 metadata；输入身份由预测文件 SHA256 确认。未运行训练或留出评估。",
    }
    for directory, audience in [(a_dir, "A"), (b_dir, "B")]:
        files = [path for path in directory.iterdir() if path.is_file() and path.name != "manifest.json"]
        manifest = {**base_manifest, "audience": audience,
                    "files": {path.name: sha256(path) for path in sorted(files)}}
        write_json(directory / "manifest.json", manifest)
        archive = HANDOFF / f"{audience}_validation_handoff.zip"
        with ZipFile(archive, "w", compression=ZIP_DEFLATED) as package:
            for path in sorted(directory.iterdir()):
                if path.is_file():
                    package.write(path, arcname=path.name)
    index = f"""# A/B 验证评估交接索引

已生成新版官方口径四份产物，并分别整理两个交接包。主结果：{report['events']} 个有后续消息的事件、{report['positive_events']} 个最终 Critical；基线 {base['reviews']} 个复核、模型 {model['reviews']} 个复核，同样找到 {model['true_positives']} 个。

- A：`A_validation_handoff.zip`，先读 `A/README_A.md`，内含全部 Critical 排名和预算外清单。
- B：`B_validation_handoff.zip`，先读 `B/README_B.md`，内含 JSON、两组曲线及页面接入示例。
- 旧版 Pc 56/119 报告已在 C 的本地 `reports/archive/` 留存，不用于新版模型成绩。
- 两包均为验证集开发结果；本次未运行训练、未读取或评估留出预测。
- 交接方式：通过 origin 和 upstream 的 main 分支共享；A/B 拉取后各自读取对应目录或压缩包。文件和包的完整性可按 manifest.json 核验。
"""
    (HANDOFF / "README.md").write_text(index, encoding="utf-8")
    print(f"A package: {HANDOFF / 'A_validation_handoff.zip'}")
    print(f"B package: {HANDOFF / 'B_validation_handoff.zip'}")
    print(f"Critical outside model budget: {len(missed)}; same missed set as baseline: {both_missed}")


if __name__ == "__main__":
    build()
