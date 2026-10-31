# 模型 E：保存预测复核与提交材料

日期：2026-10-31（Asia/Shanghai）。代码与模型来源：main `1d52485`。
最终模型 **E**，37 个特征，其中 14 个物理特征；当前等级 Critical 兜底，不使用时间规则。
模型和切分已冻结，本次不训练、不调用 `evaluate_holdout.py`、不重新预测或选择模型。

**留出集共看了 3 次**：D_stale、E、E_stale。最终采用 E，详见 [final_model_summary.md](final_model_summary.md)。
本次仅重算已保存 E 预测的复核量与证据规则汇总，不是第四次模型评估，不能将结果描述为一次性未见数据验证。
多次查看造成选择偏差，bootstrap 区间也不消除这种偏差；不再用留出集调参。

## 数据与版本来源

- 唯一留出预测输入：`reports/holdout_second_look/holdout_predictions.csv`；配套指标为同目录 `holdout_metrics.json`。
- 保存预测版本：`20261031-063722`，variant E；当前 `models/metadata.json` 版本：`20261031-064613`，variant E。
- 两个版本号不同，当前材料没有保存预测时的模型二进制哈希来证明与当前 bundle 完全相同。本次按用户指定的第二次预测做分析，单独记录两种版本，不重新推理来消除差异。
- 新 `workload_holdout.*` 与 `evidence_holdout.json` 已替换过时的 D_stale 汇总。旧统计可从 main `1c1833f` 的 Git 历史查看；D_stale 的主结果 4.8% 减少量不再用于最终 E 材料。
- 事件切分、全部原始预测、三个已保存模型及 metadata、`ml.py`、`train.py`、`validation.py` 均未修改。

## 复核量

真值为 `final_tier == Critical`。直接使用保存的 `baseline_score` 与 `alert_score`，按分数降序、event_id 升序排序。
目标召回 90%；1,000 次配对事件 bootstrap，seed=42。主结果仅 `has_future=True`；全部事件作为补充。

| 范围 | 事件数 | Critical | 基线复核量 | E 复核量 | 实际召回 | 复核量减少 | bootstrap 95% 区间 |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| 主结果：有后续消息 | 1,335 | 89 | 125 | 125 | 81/89 = 91.01% | 0.00% | [-13.69%, 26.95%] |
| 补充：全部事件 | 1,792 | 125 | 153 | 150 | 113/125 = 90.40% | 1.96% | [-11.73%, 10.07%] |

两组区间均跨 0，不能宣称模型稳定降低复核量。主结果没有减少工作量。
主结果漏报 8 个，全部事件补充结果漏报 12 个。这是带真值的事后排序测量，不是部署召回保证。
以上复核数与 E 第二次查看时已保存的指标一致；没有更新这些原始指标。

离散等级 Critical Recall 与上述排序召回不同：按 A 已保存的结果，主结果基线 72/89=80.90%，E 为 73/89=82.02%；不能将 91.01% 排序召回当成模型直接分类召回。

## 证据不足与 NEEDS_REVIEW

使用保存的预测、截止时刻之前的原始历史输入和真实 sigma 调用既有 `rules.py`；按既有 `service.py` 传播输入 WARNING。
风险等级按最新已知 Pc 与距离计算，预测最终等级另列；质量状态和风险等级分别统计，不能把这张表当成 E 的最终等级混淆矩阵。

| 范围 | NEEDS_REVIEW（含警告） | 比例 | 仅规则复核数 | 大 sigma 且低 Pc | 保守估计跨级 | 有输入警告 | SAFE 且需复核 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 主结果：有后续消息 | 157/1,335 | 11.76% | 129 | 129 | 0 | 40 | 85 |
| 补充：全部留出事件 | 265/1,792 | 14.79% | 223 | 222 | 1 | 64 | 156 |
| 补充：无标签测试事件 | 256/2,167 | 11.81% | 162 | 162 | 0 | 107 | 133 |

原因重叠，不能相加。NEEDS_REVIEW 是质量/证据标记；排序预算 125 是另一种统计，不能直接相加或替换。
无标签测试仅统计质量比例，不报告准确率或召回；其保存预测与当前二进制的对应关系未通过新推理验证。

冻结阈值仍为 `0.10816974945035213`。来源是固定训练分区的 6,224 个有后续消息事件中，最终标签 y<-6 的 6,160 个有限 sigma_ratio 的 90 分位；没有使用留出集校准。
运行时按最新已知 Pc<1e-6 触发，训练选样按最终 Pc，两者不是相同口径；阈值保留原值，不宣称已针对实时低 Pc 人群重新校准。
真实留出事件缺失 sigma 为 0；8 个合成规则边界案例通过，包括阈值等号、刚高于阈值、缺失 sigma、缺失 Pc、缺失距离、零距离和保守估计跨级。

## 复现

以下命令仅复用保存结果。原始 CSV 路径可换成本机位置，不能运行训练或留出预测命令。

```powershell
python evaluate_workload.py --predictions reports/holdout_second_look/holdout_predictions.csv --split holdout --metrics reports/holdout_second_look/holdout_metrics.json --model-metadata models/metadata.json --holdout-look-count 3 --source-commit 1d52485 --target-recall 0.9 --bootstrap 1000 --seed 42 --output reports/workload_holdout.json
python evaluate_evidence.py --train Data_raw/train_data.csv --test Data_raw/test_data.csv --predictions reports/holdout_second_look/holdout_predictions.csv --metrics reports/holdout_second_look/holdout_metrics.json --workload reports/workload_holdout.json --holdout-look-count 3 --source-commit 1d52485 --output reports/evidence_holdout.json
```

本机运行使用 `C:/Users/user/anaconda3/python.exe`；CatBoost 依赖沿用 `D:/Autonomous-SG-2026/local_backups/review_deps`，未安装新依赖。
证据脚本会核对新复核量文件的预测 SHA256 与版本来源，拒绝把旧 D_stale 复核量混入 E 审计。
事件级证据明细仅作本地核查，提交材料使用汇总和曲线。原始数据不纳入交付包。

## 提交职责

用户已明确：**比赛提交由队友负责，本助手不再检查或操作比赛门户。**
本助手此前未执行过比赛平台有效提交，没有平台回执；历史 GitHub 报告 PR 和本地备份不能当作比赛有效提交。
团队的实际提交状态由负责提交的队友维护，本报告不替其确认或更新收件状态。
给队友的材料说明见 [submission_plan.md](submission_plan.md)。这些分析更新随代码交付；GitHub 发布不代表比赛平台已有效收件。
