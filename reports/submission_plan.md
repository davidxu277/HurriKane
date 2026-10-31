# 首版提交与候选模型复核流程

更新：2026-10-31，Asia/Shanghai。**尚未向比赛平台提交。经用户授权，本轮流程文档、替换检查和测试同步到 origin/upstream 的 `codex/submission-workflow` 分支；首版 ZIP 备份保留本地。**
本文件是当前流程；旧 `holdout_review_submission.md` 的 GitHub PR 记录是历史报告交付，不是比赛有效提交或时间分回执。

## 已核实的主办方规则

来源：[BoardingPass Autonomous SG 2026](https://boardingpass.work/experience/challenges/autonomous-sg)，Overview 与 Airbus 页签，2026-10-31 查看。
用户提供的 HOW TO SUBMIT 截图同时说明每桌只提交一个项目。

- 一个桌号对应一个项目，必须填写桌号；不要创建多个参赛项目。
- Airbus 最新补充说明：Time discipline 按首次有效提交时间计，后续重新提交不会重置该时间。因此可更新/重提同一项目，但不要把“一个项目”误读为可为同桌创建多个项目。
- 截止为 2026-10-31 16:15（新加坡和北京时间均为 UTC+8）。时间纪律占 5%，还包括三分钟内完成展示。
- 正式入口是 BoardingPass；GitHub push、PR 和本地 ZIP 均不等于比赛有效提交。
- 登录后提交表单的具体字段、上传格式、大小限制及更新按钮未核实。不能把内部预测 CSV 或本地备份 ZIP 宣称为主办方规定的提交格式。

## 怎么提交首版

1. 打开 [提交平台](https://boardingpass.work/experience)，使用 Gmail、LinkedIn 或学生账号登录。
2. 进入 Autonomous SG 2026，选择 Airbus，以及 Challenge 1：Satellite collision avoidance。
3. 在该桌号的唯一项目内，按登录后的表单填写所有必填内容并包含桌号。建议提前备好项目说明、仓库/演示链接、运行说明、基线对比和一个真实失败案例；这些是备料建议，具体必填项以表单为准。
4. 首版使用下述冻结 main 的现有模型，不等待 A 的新版，不临时训练、不更换模型。演示 GUI 时使用 Backend CSV 和真实输入，Simulation Data 必须标注为模拟。
5. 用户决定正式提交时，检查表单、链接和材料后提交；确认平台成功状态，保存有效提交时间、项目 ID、回执或截图。只有平台确认后才能说“首次有效提交完成”。

## 首版冻结版本

- 上轮已用 `git ls-remote` 核实 origin/main 与 upstream/main 均为 `1c1833fb951ea9d5ceaafd16a9897bfbc3ba6037`；已只读 fetch 对象，没有切换或合并分支。本次按该提交封存，不追随 A 后续改动。
- 现有模型：`D_stale`，版本 `20261031-054344`。保留此模型现有行为，包括原来的时间规则；“不用时间规则”是 A 候选方案的要求，不能提前应用于首版。
- 本地备份：`local_backups/first_submission/main-1c1833f.zip`，原样封存该 main 的已跟踪文件。ZIP 是复现备份，尚未验证为正式平台所需格式。
- `local_backups/first_submission/manifest.json` 记录提交号、ZIP 校验值、保护文件 SHA256 及“未正式提交”状态。备份操作不调用训练、推理或评估。

## A 开发期间

不改 `ml.py`、`train.py`、`validation.py`；不覆盖或重新生成 `models/`、已有 holdout 预测、指标、复核量和证据文件，不重抽事件切分。
A 用 TDD 实现“新版 + 兜底”、14 个物理特征和调参，并去掉候选方案时间规则。这些由 A 完成，本轮不代改。
不因发现远程有新提交就自行开始第二次查看；等待用户明确通知 A 已推送。

## 通知后的第二次查看留出集（现在不运行）

1. 固定 A 的提交号、模型版本/哈希、参数和相同的 `split_events.json`。保留首次预测与审计文件，在独立的候选快照目录运行，不覆盖本工作区的首次结果。
2. 如果 A 已完成第二次留出预测，直接使用其保存结果，不再做第三次预测。若尚未生成，先核对责任分工，只在候选快照中生成该次结果；不要删除旧预测绕过脚本的一次性限制。
3. 全部新报告明确标注 **第二次查看留出集**，记录候选提交号、模型哈希、输入哈希及已查看两次这一事实；不能再称其为未见过的独立最终验证，也不能继续用本次留出结果调参。
4. 沿用现有主口径 `has_future=True`，全部事件另列补充。三类平均 F1 指 Safe/Elevated/Critical 的宏平均 F1；Critical Recall 使用模型离散等级，对照同事件最新已知等级 `current_tier` 基线，不用排序达到 90% 的召回替代。
5. 同事件、同标签、同基线条件下，**候选宏平均 F1 ≥ 首版模型宏平均 F1，且候选 Critical Recall ≥ 基线 Critical Recall**，两项同时成立才具备替换资格。不满足任一项或证据不完整则保留首版，不自动替换。
6. 使用新增 `submission_review.py` 比较两份已保存预测；它只计算检查结果，不训练、不预测、不替换模型、不提交。示例路径需替换成实际快照路径，输出必须是新文件：

   ```powershell
   python submission_review.py --existing <首版快照>/reports/holdout_predictions.csv --candidate <候选快照>/reports/holdout_predictions.csv --output reports/second_review/replacement_gate.json
   ```

7. 在候选快照根目录重跑 `evaluate_workload.py` 和 `evaluate_evidence.py`。前者沿用 Critical 真值、90% 目标、1000 次配对 bootstrap、seed=42；后者读取该快照内相匹配的模型 metadata、预测、切分与原始数据，核对质量规则与阈值来源。当前证据脚本的目录写死，必须在独立快照内运行，不能只传一个新输出路径就误读旧输入。
8. 复核量、区间、NEEDS_REVIEW 及失败案例照实报告。若新方案不再满足原证据脚本的来源断言，应明确解释并修正审计依据，不能为通过断言而改模型或改数字。

## 本轮验证

新增替换检查只用合成数据测试，不读取真实留出数据：

```powershell
python -B -m unittest discover -s tests -p test_submission_review.py -v
```

本轮不计算新的真实留出指标，不替换任何模型，不触发正式提交。
