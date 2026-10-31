# 给 B：新版验证集评估页面交接

标题必须标注“验证集开发结果”。当前目标为 final_tier=Critical，主结果仅取 has_future=True。
本包是离线评估产物，不需要在页面训练或运行 bootstrap。
配套最终方案：`D_stale`，模型版本 `20261031-054344`。

## 本次应展示的主结果

| 指标 | 值 |
| --- | ---: |
| 主范围事件数 | 1333 |
| 真实最终 Critical | 88 |
| 目标召回率 | 90% |
| 实际找到 / 总 Critical | 80 / 88 |
| 实际召回率 | 90.91% |
| 基线复核量 | 113 |
| 模型复核量 | 111 |
| 复核量减少 | 1.77% |
| 减少比例的 95% 区间 | [-13.85%, 17.23%] |

本次页面结论写：“达到至少 90% 的验证集 Critical 召回时，基线复核 113 个、模型复核 111 个；本次点估计略有减负，区间跨 0，尚不能确认稳定改善。”

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
        f"增加 {-reduction:.1%}" if reduction < 0 else f"减少 {reduction:.1%}"
    )
    c3.metric("复核量变化", change)
    found = metrics["model"]["true_positives"]
    total = metrics["positive_events"]
    st.write(f"主范围 {metrics['events']} 个事件；找到 {found} / {total} 个 Critical。")
    interval = metrics["bootstrap"]
    lo, hi = interval["ci95_lower"], interval["ci95_upper"]
    if lo is not None and hi is not None:
        st.write(f"减少比例的 bootstrap 95% 区间：[{lo:.1%}, {hi:.1%}]")
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
- 全部事件有 1791 个，其中 458 个没有后续消息；这些事件的最终标签已在输入中，所以只能补充展示。
- 服务层上传分析结果和这里的验证评估是两个数据源；不要用当前上传文件的事件数替换验证集事件数。
- manifest.json 标识这批输入与文件版本；metadata.json 提供交接时的模型版本。
