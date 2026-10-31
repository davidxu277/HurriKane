# 留出集正式复核与首次 GitHub 提交

日期：2026-10-31（Asia/Shanghai）。基于 `davidxu277/HurriKane` 的 main `77e1518`。
冻结版本：`D_stale`，模型版本 `20261031-054344`。

本次直接使用 main 上已有 `holdout_predictions.csv` 和 `holdout_metrics.json`，
没有调用 `evaluate_holdout.py`、重新预测留出集、训练、调整模型、阈值或事件切分。
“普通事件”为有后续消息事件，即 `has_future=True`；无后续消息事件的最终标签已在输入中，只作为附带结果。

## 正式复核量

目标为 `final_tier == Critical`，召回目标 90%，分数按降序、同分按 event_id 升序。
沿用 `evaluate_workload.py` 默认 1,000 次配对事件 bootstrap，种子 42。

| 范围 | 事件数 | 最终 Critical | 基线复核量 | 模型复核量 | 实际召回 | 复核量减少比例 | bootstrap 95% 区间 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 主结果：有后续消息 | 1,335 | 89 | 125 | 119 | 81/89 = 91.01% | 4.80% | [-31.38%, 26.23%] |
| 附带：全部事件 | 1,792 | 125 | 153 | 154 | 113/125 = 90.40% | -0.65%（增加） | [-15.72%, 13.02%] |

两组区间均跨 0，不能据此宣称模型稳定降低工作量。两组复核数与 main 的
`holdout_metrics.json` 完全一致。主结果漏报 8 个，全部事件附带结果漏报 12 个。
这属于带真值的事后排序测量，不是无标签部署数据上的召回保证。

详细产物：`workload_holdout.json`、`workload_holdout.md`、
`workload_holdout_curve.csv`、`workload_holdout_curve.png`。

## 证据不足与 NEEDS_REVIEW

使用已有预测和所选历史输入中的真实 sigma，调用原 `rules.py`；
把截止时刻之前的历史输入 WARNING 按原 `service.py` 的事件传播逻辑计入。
不使用后续消息的 WARNING 判断实时质量。风险等级和质量状态分别统计。

冻结规则：最新已知 Pc < 1e-6 且 sigma_ratio > 0.10816974945035213 时需要复核；
sigma 缺失、其他证据缺失及保守估计跨级等也会触发复核。sigma_ratio 是无量纲量。

| 范围 | NEEDS_REVIEW（含输入警告） | 比例 | 仅规则 NEEDS_REVIEW | 大 sigma 且低 Pc | 保守估计跨级 | 有输入警告 | SAFE 且需复核 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 主结果：有后续消息 | 159/1,335 | 11.91% | 131（9.81%） | 129 | 2 | 40 | 87 |
| 附带：全部留出事件 | 267/1,792 | 14.90% | 225（12.56%） | 222 | 3 | 64 | 158 |
| 附带：无标签测试事件 | 256/2,167 | 11.81% | 162（7.48%） | 162 | 0 | 107 | 133 |

原因可以重叠，不能相加得到总复核数。以上比例以事件去重，分母不是消息条数。
`NEEDS_REVIEW` 是质量/证据标记；复核量评估中的 119 个是达到指定召回的排序预算，
两者定义不同，不能直接相加或替换。SAFE 且需要复核的事件不能只显示安全标识。
无标签测试数据仅统计质量比例，不报告准确率或召回率。

真实留出集缺失 sigma_ratio 为 0 个；另外使用合成案例验证缺失证据处理。
8 个审计边界案例全部通过：ratio 恰好等于阈值、刚高于阈值、Pc 恰好为 1e-6、
缺失 sigma、缺失 Pc、缺失距离、零距离、保守估计跨级。
缺失 sigma 不填 0；证据不足保留风险等级并标记 NEEDS_REVIEW。

### 阈值来源核实

现有 `train.py` 在固定训练分区的 6,224 个有后续消息事件中，按最终标签
`y < -6` 选择 6,160 个有限 sigma_ratio，90 分位数为 **0.10816974945035213**，
与冻结 metadata 精确复现一致。计算没有使用验证集或留出集。

运行时证据规则按最新已知 Pc < 1e-6 触发；训练选样按最终 Pc，二者不是相同口径。
若只在同一训练分区按最新已知 Pc 筛选，对应 5,998 个有限值，90 分位数为
0.09865259095238627。此值仅用于说明来源差异，**未替换冻结阈值**。
因此本次确认的是既有规则正确执行及来源可复现，不能将其描述为已按实时低 Pc 人群重新校准。

## 验证与复现

以下现有测试共 **126 项通过**：

```powershell
python -m pytest tests/test_workload.py tests/test_rules_tier.py tests/test_validation.py tests/test_service.py -q
```

正式分析命令（Python 环境需有 pandas、NumPy、Matplotlib、CatBoost；测试需 pytest）：

```powershell
python evaluate_workload.py --predictions reports/holdout_predictions.csv --split holdout --target-recall 0.9 --bootstrap 1000 --seed 42
python evaluate_evidence.py --train Data_raw/train_data.csv --test Data_raw/test_data.csv
```

本机审计使用 `C:/Users/user/anaconda3/python.exe`，CatBoost 1.2.10 仅安装到
`local_backups/review_deps`，执行时以 PYTHONPATH 引入；pytest 临时目录指定到工作区。
环境依赖与原始数据不包含在报告提交中。

审计脚本验证事件切分互斥、留出 ID 精确匹配、标签及输入证据一致、规则触发、
等级保持和 main 指标一致；审计前后对三份模型、metadata、切分清单、既有预测、
规则及实现文件计算 SHA256，全部未变。哈希清单见 `evidence_holdout.json`。

事件级明细 `evidence_holdout_events.csv`、`evidence_test_events.csv` 仅保留在本地，
可用上述审计脚本重新生成，不随 GitHub 提交发布。
GitHub 提供汇总、边界案例和阈值来源：`evidence_holdout.json`。

## GitHub 提交入口

用户已明确首次提交指 **GitHub 复核报告**。
目标仓库：[davidxu277/HurriKane](https://github.com/davidxu277/HurriKane)，目标分支 `main`。
通过 `codex/holdout-review-20261031` 报告分支推送并创建 Pull Request，保留 main 供审阅合并。
当前 GitHub 身份 `stephen-118`，已核实目标仓库 WRITE 权限。

提交只包含正式汇总报告、统计曲线、汇总审计结果及可复现审计脚本。
本地已有的服务说明、校验说明、归档和原始数据未纳入本次提交。
