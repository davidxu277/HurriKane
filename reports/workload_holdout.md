# 留出集复核量分析

输入：`reports\holdout_second_look\holdout_predictions.csv`，共 1792 个事件。

目标为 `final_tier == 'Critical'`；直接使用 A 输出的最终等级标签，不按 Pc 阈值或输入距离重新推导真值。

主结果只评估 has_future=True；全部事件另列为补充，避免把标签已知的事件混入主结果。两种排序直接使用文件中的 baseline_score 和 alert_score。

模型 `E`，保存预测版本 `20261031-063722`；当前 bundle 版本 `20261031-064613`。

**留出集共看了 3 次**；本次仅重算保存预测的复核量，不训练或重新预测，不增加一次模型评估。

- 留出集共看了 3 次；多次查看后的结果可能偏乐观，不是未见数据上的独立验证。
- 保存预测的 model_version 与当前模型 metadata 不同；只分析指定预测，不重新推理，不宣称已证明当前模型二进制与该预测一致。

## 主结果：有后续消息的事件

范围：`has_future == True`；1335 个事件，89 个最终 Critical 事件。

召回目标 90%：至少找到 81 / 89 个高风险事件。

| 排序方式 | 需复核事件数 | 找到的高风险事件数 | 漏报数 | 实际召回率 |
| --- | ---: | ---: | ---: | ---: |
| 输入文件 baseline_score | 125 | 81 | 8 | 91.0% |
| 模型 alert_score | 125 | 81 | 8 | 91.0% |

模型相对基线的复核量**减少 0.0%**。

减少比例定义：`(基线复核量 - 模型复核量) / 基线复核量`；负数代表模型增加工作量。

减少比例的 bootstrap 95% 区间：**[-13.7%, 26.9%]**。

使用配对事件有放回抽样 1000 次，种子 42；有效 1000 次，无正例 0 次。两种排序使用同一批抽样，重复抽到的事件不去重。

区间跨过 0，不能据此断言模型稳定减少或增加了复核量。


## 补充结果：全部事件

范围：`all supplied events`；1792 个事件，125 个最终 Critical 事件。

召回目标 90%：至少找到 113 / 125 个高风险事件。

| 排序方式 | 需复核事件数 | 找到的高风险事件数 | 漏报数 | 实际召回率 |
| --- | ---: | ---: | ---: | ---: |
| 输入文件 baseline_score | 153 | 113 | 12 | 90.4% |
| 模型 alert_score | 150 | 113 | 12 | 90.4% |

模型相对基线的复核量**减少 2.0%**。

减少比例定义：`(基线复核量 - 模型复核量) / 基线复核量`；负数代表模型增加工作量。

减少比例的 bootstrap 95% 区间：**[-11.7%, 10.1%]**。

使用配对事件有放回抽样 1000 次，种子 42；有效 1000 次，无正例 0 次。两种排序使用同一批抽样，重复抽到的事件不去重。

区间跨过 0，不能据此断言模型稳定减少或增加了复核量。

- 全部事件包含无后续消息的事件，其最终标签本身已在输入中；只能作为补充结果。

这是对已知标签的事后排序评估；不能用它承诺未标注数据也达到相同召回率。验证集结果用于开发，最终效果需在模型和规则固定后用留出集确认。

不根据这份结果重抽事件切分、选择更好看的种子或调整留出集。脚本不训练模型、不读取原始数据和其他预测文件。

输入文件 SHA256：`42aeba6b2628f2bc3b54bb9c171954212ddde8fc51e6bcd3ed5fd273169f462b`。

## 复现命令

在项目根目录、安装了 pandas、NumPy、Matplotlib 的 Python 环境中运行：

```powershell
python evaluate_workload.py --predictions "reports\holdout_second_look\holdout_predictions.csv" --split holdout --target-recall 0.9 --bootstrap 1000 --seed 42 --metrics "reports\holdout_second_look\holdout_metrics.json" --model-metadata "models\metadata.json" --holdout-look-count 3 --source-commit 1d52485
```

JSON 顶层指标是主结果，supplementary_all 为补充结果；读取 `_curve.csv` 必须按 scope=future/all 分组，不能把两组曲线连成一条。`_curve.png` 分两栏展示。验证集产物必须标注为开发结果。
