# HurriKane — 卫星碰撞风险评估

Autonomous SG 2026 · Airbus Challenge #1 的 Python 实现。项目根据同一交会事件的历史消息，预测最终风险等级和 `log10(Pc)`，并提供输入校验、当前风险分级、证据不足提示和复核工作量分析。

本文描述当前仓库代码及随附模型，不再使用早期六小时开发计划。当前模型为 **`E`**，版本 **`20261031-064613`**，前端为 **ORBITGUARD** Streamlit 仪表盘。

> **模型、阈值和事件切分已冻结，不再训练或评估留出集。** 留出集共查看 3 次，最终采用 E，详见 [最终模型记录](reports/final_model_summary.md)。比赛材料由负责提交的队友交接，见 [材料交接说明](reports/submission_plan.md)；GitHub 更新不代表比赛平台有效收件。

## 1. 当前实现

- `validation.py`：检查原始消息，区分 ERROR 与 WARNING，保留问题位置和原因。
- `ml.py`、`train.py`：构造事件样本，训练 CatBoost 模型，保存模型、验证预测和评估指标。
- `service.py`：提供 `analyze_csv()`，串联校验、缓存模型推理、风险规则、历史消息和事件汇总。
- `rules.py`：给出当前风险等级、触发原因及 `NEEDS_REVIEW` 标记。
- `app.py`、`gui_*.py`：ORBITGUARD 页面，支持模拟演示、真实 CSV 分析、筛选、事件详情、风险历史、R/T/N 示意图和下载。
- 评估脚本：保存预测的复核量曲线、配对 bootstrap 区间、证据规则审计及候选模型比较。历史训练、留出脚本保留用于追溯。

页面依赖见 [`requirements-gui.txt`](requirements-gui.txt)，前端接入细节见 [GUI 开发说明](GUI_DEVELOPMENT.md)。也可通过 Python 服务接口直接进行批量分析。

## 2. 环境与数据

建议使用 Python 3.11 或 3.12，在仓库根目录执行命令。

Windows PowerShell：

```powershell
git clone https://github.com/stephen-118/Autonomous-SG-2026.git
Set-Location Autonomous-SG-2026
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-gui.txt
python -m pip install scikit-learn matplotlib pytest "pandas>=2.2,<2.4"
```

macOS / Linux：

```bash
git clone https://github.com/stephen-118/Autonomous-SG-2026.git
cd Autonomous-SG-2026
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-gui.txt
python -m pip install scikit-learn matplotlib pytest "pandas>=2.2,<2.4"
```

第二条安装命令补齐训练/评估及测试依赖，并将 pandas 设为 2.2 或 2.3，以支持证据审计中的 `groupby.apply(include_groups=False)`。依赖文件提供版本范围，尚未锁定全部依赖。

安装后启动页面：

```bash
python -m streamlit run app.py
```

在侧边栏选择数据源：

1. **Simulation**：展示明确标注的模拟数据，不加载真实服务或模型，用于页面演示。
2. **Backend CSV**：上传原始消息 CSV，点击 **Analyze CSV** 调用真实服务。模型从页面所在目录的 `models/` 加载，不会训练。

页面包含 Mission Overview 和 Collision Events，可查看当前等级、预测最终等级、质量提示、真实输入历史及 R/T/N 示意图，并下载处理结果和问题清单。没有选择数据源时显示空状态。切换文件会使旧分析失效，筛选或翻页不会重复运行推理。

将主办方原始数据放在以下位置；数据需自行准备：

```text
Data_raw/
  train_data.csv
  test_data.csv
```

本文命令使用实际目录 `Data_raw/`。旧报告和模型 metadata 中的 `data/raw/` 是训练时记录的历史路径，不必为推理修改 metadata。

输入 CSV 至少需要以下列；随附 `final` 模型也需要历史 `risk`：

```text
event_id, time_to_tca, miss_distance, relative_speed,
relative_position_r, relative_position_t, relative_position_n,
c_object_type, c_sigma_r, t_sigma_r, max_risk_estimate, F10, AP, risk
```

| 字段 | 含义 / 单位 |
| --- | --- |
| `event_id` | 事件标识，仅用于分组，不作为模型特征 |
| `time_to_tca` | 距最近接近时刻的天数；数值越小，消息越新 |
| `risk` | 历史消息已知的 `log10(Pc)`，不是原始概率；应 ≤ 0 |
| `miss_distance`、`relative_position_r/t/n`、`c_sigma_r/t_sigma_r` | 米 |
| `relative_speed` | 米/秒 |

E 的物理特征还会读取 `c_sigma_t`、`t_sigma_t`、`c_sigma_n`、`t_sigma_n`，建议保留原始 CSV 的完整字段。这四列缺失时相应输入按 NaN 处理；不需上传由代码派生的特征列。

必需列缺失会导致文件级 ERROR。除事件 ID、时间及 final 的历史 risk 外，允许的缺失值保留为 NaN 并记录 WARNING，不填成 0。数值不可解析、无穷值、负距离和负 sigma 等会报 ERROR。

## 3. 使用现有模型分析 CSV

`models/` 已包含 `estimate.cbm`、`conservative.cbm`、`classifier.cbm` 和 `metadata.json`，使用它们推理不需要重新训练。

在项目根目录运行以下 Python 代码：

```python
from pathlib import Path

import pandas as pd

from service import analyze_csv

raw = pd.read_csv(
    "Data_raw/test_data.csv",
    low_memory=False,
    dtype={"event_id": str},
)
result = analyze_csv(raw, model_dir="models")

print(result["summary"])
print(result["events"][
    ["event_id", "risk_level", "predicted_final_tier",
     "data_quality_status", "prediction_status"]
].head())

output_dir = Path("reports/inference")
output_dir.mkdir(parents=True, exist_ok=True)
for name in ("records", "events", "history", "issues"):
    result[name].to_csv(
        output_dir / f"{name}.csv", index=False, encoding="utf-8-sig"
    )
```

这是 Python 代码示例，不是 PowerShell 命令；可保存为本地脚本后用 `python <脚本名>.py` 执行。以上导出目录与已有训练、留出报告分开。

服务返回六项内容：

| key | 类型 | 用途 |
| --- | --- | --- |
| `records` | DataFrame | 推理及规则结果；final 模式一事件一行，含未预测占位记录 |
| `events` | DataFrame | 事件表，按当前等级、`alert_score` 降序排列 |
| `history` | DataFrame | 实际参与推理的合法历史消息；`risk` 保留 log10 量纲 |
| `issues` | DataFrame | 校验及服务问题，含行位置、事件、代码和原因 |
| `summary` | dict | 行数、事件数、处理状态、等级计数和 24 小时覆盖 |
| `metadata` | dict | 模型配置副本，附规则版本和服务版本 |

模型在同一进程内按目录及文件修改时间、大小缓存。服务不训练模型，也不自动保存报告。缺少或损坏模型会抛出异常。

## 4. 任务口径与模型

### 样本与标签

当前 `train.py` 固定 `TASK = 'final'`，没有 `--task` 参数。

1. 每个事件只使用 `time_to_tca >= horizon_days` 的消息，默认截止为 TCA 前 **2 天**。
2. 取这些消息中最新的一条作为样本主记录，变化量、消息数和历史统计全部来自该输入窗口。
3. 标签取同事件全部消息中最后一条的 `risk` 和 `miss_distance`，按阈值生成 `final_tier`；代码没有“标签必须在 TCA 前 1 天以内”的条件。
4. `has_future=True` 表示事件存在 `time_to_tca < horizon_days` 的后续消息。无后续消息时，最终标签已在输入中，因此主要评估只使用有后续消息的事件。

`ml.py` 和服务层保留 `current` 模式支持，但随附模型及训练 CLI 均为 final。不能通过改参数名称或手动修改 metadata 将现有模型切换成 current。

### 特征与三个模型

E 使用 **37 个特征**：12 个基础特征、5 个 final 特征、6 个历史统计特征及 14 个物理派生特征。物理特征包含各轴合成 sigma、标准化相对位置、距离阈值裕量、趋势外推距离和固定随机样本构造的距离范围比例。完整名称见 `ml.py` 的特征常量；类别特征是 `c_object_type` 和 `current_tier`。

| 文件 | 模型 | 输出含义 |
| --- | --- | --- |
| `estimate.cbm` | CatBoost RMSE 回归 | 最终 `log10(Pc)` 点估计，**不是中位数模型** |
| `conservative.cbm` | CatBoost 分位数回归 | 保守估计；比较 alpha=0.8/0.9/0.95，随附模型选用 0.9 |
| `classifier.cbm` | CatBoost 三分类 | 最终 Safe / Elevated / Critical 的分类分数 |

回归训练目标下限为 -10；回归误差评估也使用同样截断。`conservative_log10_pc` 通过取最大值保证不低于点估计；该分位数输出不是经过校准的置信上界。

默认方案 **E** 使用物理特征、已选定的分类器参数和当前等级兜底，**没有时间兜底规则**。当前 `critical_threshold` 与 `stale_days` 均为 null：最终分类先取最高分等级，再保证当前已为 Critical 的事件不会被降级。

`alert_score` 仍使用当前等级对应的训练 Critical 比例作为下限，并加入很小的分类分数用于排序；E 的离散最终等级不通过该排序分数阈值判定。分类器参数为 depth=7、learning_rate=0.0652、l2_leaf_reg=6.7、rsm=0.76，调参方案最多训练 300 次迭代。RMSE 与分位数回归模型使用各自的回归设置。

### 三种等级与概率字段

| 字段 | 含义 |
| --- | --- |
| `current_tier` | 最新合格输入消息按已知 risk 和距离计算的等级 |
| `predicted_final_tier` | 分类模型和当前 Critical 不降级规则给出的最终等级预测 |
| `risk_level` | 服务规则给出的当前等级，包含 UNKNOWN，使用大写名称 |
| `tier_pc` / `tier_log10_pc` | 当前分级依据；final 使用最新已知 risk |
| `predicted_pc` / `predicted_log10_pc` | 回归模型预测的最终风险 |
| `conservative_pc` | 保守回归估计 |
| `p_safe/p_elevated/p_critical` | 三分类模型输出，不是物理碰撞概率 |
| `alert_score` | 用于事件排序的分数，不是碰撞概率 |
| `baseline_score` | 当前等级优先、同级距离越近越靠前的基线排序分数 |

尤其要区分 `risk_level` 与 `predicted_final_tier`：当前规则分级不使用最终风险回归估计替代历史观测。

## 5. 当前风险规则与数据质量

`rules.py` 按以下优先级使用分级用 Pc 和距离：

| 当前等级 | 条件 |
| --- | --- |
| CRITICAL | Pc ≥ 1e-4，或距离 < 1000 m |
| ELEVATED | 不满足 CRITICAL，且（Pc ≥ 1e-6 或距离 ≤ 5000 m） |
| SAFE | Pc < 1e-6，且距离 > 5000 m |
| UNKNOWN | 现有证据不足以确定上述等级 |

风险等级与质量状态分开。输入警告、缺失证据、保守估计使等级上升等情况可触发 `NEEDS_REVIEW`，SAFE 事件也可能需要复核。

证据比例为：

```text
sigma_ratio = sqrt(c_sigma_r² + t_sigma_r²) / miss_distance
```

当最新已知 Pc < 1e-6 且该比例大于训练阈值时标记复核。随附阈值为 **0.10816974945035213**；缺失 sigma 不能视为 0。阈值实际来自有后续消息的训练事件中最终标签 `y < -6` 的有限比例值的 90 分位数，与运行时按最新已知 Pc 触发的口径存在差异，审计说明见 [留出复核报告](reports/holdout_review_submission.md)。

服务层对同事件任意校验 ERROR 暂停该事件整体推理；文件级 ERROR 阻止整批推理。输入窗口内重复的事件消息时间也会阻止该事件预测。可识别但无法预测的事件保留 `UNKNOWN / NEEDS_REVIEW`，并标记 `prediction_status=NOT_PREDICTED`。

24 小时统计条件为 `0 < time_to_tca_hours <= 24`，只代表各事件最新输入消息报告的窗口，没有统一观测时刻或完整覆盖保证。默认 final 输入至少为 48 小时，因此正常预测记录不覆盖该窗口；`INSUFFICIENT` 应显示为“覆盖不足”。页面另有七天高风险展示计数：去重统计 `0 <= time_to_tca_hours <= 168` 的 CRITICAL/ELEVATED 事件，与服务的 24 小时统计分别展示，不代表模型预测时间范围。

## 6. 训练设计与冻结状态

事件按 70% / 15% / 15% 切为训练、验证和留出，seed=42；类别数量足够时按最终等级和是否存在后续消息分层。已有 `split_events.json` 会被复用并检查事件集合一致性。

| 分区 | 全部事件 | 有后续消息 |
| --- | ---: | ---: |
| 训练 | 8,359 | 6,224 |
| 验证 | 1,791 | 1,333 |
| 留出 | 1,792 | 1,335 |

默认 E 的三个生产模型使用训练分区中有后续消息的事件训练。代码保留 `train.py`、`experiments/` 和 `evaluate_holdout.py` 供检查训练方法与历史实验，日常运行只需加载已有模型。

**不再运行训练、重新切分或留出预测命令。** 留出集历史查看依次为 D_stale、E、E_stale，共 3 次；最终采用 E，第三次的 E_stale 未采用。多次查看可能造成选择偏差，不能将保存结果表述为一次独立的未见数据验证。

## 7. 已保存结果与评估工具

以下数字引用 E 的 [第二次保存指标](reports/holdout_second_look/holdout_metrics.json) 和 [当前复核量报告](reports/workload_holdout.md)，并非本次重新训练或评估所得。根目录 `reports/holdout_metrics.json` 和 `holdout_predictions.csv` 是第一次 D_stale 的历史结果，不能混入最终 E 的材料。

主要范围为 `has_future=True`：1,335 个事件，其中 89 个最终 Critical。

| 指标 | 当前等级 / 排序基线 | E |
| --- | ---: | ---: |
| 最终三分类准确率 | 91.61% | 92.51% |
| 三类宏平均 F1（保存报告约值） | 0.853 | 0.861 |
| 固定等级预测的 Critical Recall | 72/89 = 80.90% | 73/89 = 82.02% |
| 排序达到至少 90% Recall 所需复核数 | 125 | 125 |
| 该排序预算下实际发现的 Critical | 81/89 = 91.01% | 81/89 = 91.01% |

主范围复核量减少 **0%**，1,000 次配对事件 bootstrap 的 95% 区间为 **[-13.69%, 26.95%]**。全部 1,792 个事件的补充结果为 153 → 150 次复核，减少约 1.96%，区间为 [-11.73%, 10.07%]。两组区间均跨 0，不能宣称稳定降低工作量；全部事件宏平均 F1 约 0.866，也低于基线的 0.890。

保存 E 预测的版本为 **`20261031-063722`**，当前模型 metadata 为 **`20261031-064613`**。现有材料没有证明这两份模型二进制完全一致；不要将保存预测的指标说成对当前 bundle 新推理测得的成绩。详情见 [最终模型记录](reports/final_model_summary.md)。

排序预算是带真值的事后测量，与固定等级阈值的召回、服务中的 NEEDS_REVIEW 数量含义不同。测试 CSV 没有最终真值，不能据此报告最终预测准确率。

仅需重现验证集复核量报告时，可在新目录输出：

```bash
python evaluate_workload.py --predictions reports/validation_predictions.csv --split validation --target-recall 0.9 --bootstrap 1000 --seed 42 --output reports/reproduced/workload_validation.json
```

其他工具用途：

| 脚本 | 行为 |
| --- | --- |
| `evaluate_workload.py` | 从保存的预测计算复核量，输出 JSON、Markdown、曲线 CSV 和 PNG |
| `evaluate_evidence.py` | 核对保存预测、输入证据、阈值来源和规则；不训练或调用模型预测 |
| `submission_review.py` | 比较首版与候选的已保存预测，输出替换资格检查，不替换模型或提交 |

`evaluate_evidence.py` 默认读取 E 的第二次保存预测，可通过 `--predictions`、`--metrics`、`--models`、`--split`、`--workload` 等参数指定匹配的材料，路径相对于运行目录。只分析保存预测的复现命令见 [复核报告](reports/holdout_review_submission.md)，包括版本来源和三次查看记录；不要重新运行留出预测。

`submission_review.py` 保留历史替换检查：同事件、同真值、同基线条件下，宏平均 F1 不低于首版模型且 Critical Recall 不低于基线才具备替换资格。当前最终选择已冻结。

## 8. 目录与测试

```text
Autonomous-SG-2026/
  Data_raw/                 # 自行准备的原始 CSV
  models/                   # 三个 CatBoost 模型及 metadata
  reports/                  # 切分、预测、指标、审计和提交流程
  tests/                    # ML、训练、校验、规则、服务、评估与候选检查
  app.py                    # Streamlit 入口
  gui_backend.py            # 延迟调用服务的上传适配器
  gui_data.py                # 前端契约、排序与展示统计
  gui_charts.py              # Plotly 图表
  gui_mock.py                # 模拟演示数据
  gui_style.py               # 页面样式
  requirements-gui.txt
  GUI_DEVELOPMENT.md
  analysis_provenance.py     # 保存预测的来源记录
  experiments/               # 历史交叉验证与调参脚本
  tests_gui/                 # 页面及真实服务集成测试
  ml.py
  train.py
  validation.py
  rules.py
  service.py
  evaluate_holdout.py
  evaluate_workload.py
  evaluate_evidence.py
  submission_review.py
  README.md
```

安装依赖后运行现有测试：

```bash
python -m pytest tests -q
python -m unittest discover -s tests_gui -v
```

训练流程测试使用合成数据及临时模型目录；GUI 集成测试使用合成消息，不读取真实留出标签。更详细的前端说明见 [GUI 开发说明](GUI_DEVELOPMENT.md)，输入校验接口见 [`validation.py`](validation.py)，服务接口见 [`service.py`](service.py)。

## 9. 使用边界

本项目预测数据集提供的风险指标与最终等级，用于事件排序和人工复核，不执行轨道仿真、机动规划或自动避碰。R/T/N 表示相对位置，不是重建轨道。分类分数和回归分位数不应展示为经过物理验证的碰撞概率或置信保证。

当前 E 不使用消息时间兜底规则，全部事件指标包含没有后续消息的事件，且结果来自多次查看留出集后的选择；展示时须说明范围和版本。数据校验通过只表示可处理，`summary.status == "ok"` 不表示所有事件安全。Simulation 是模拟演示，真实 CSV 分析也不代表实时遥测。比赛材料及正式收件状态由负责提交的队友维护，见 [材料交接说明](reports/submission_plan.md)。
