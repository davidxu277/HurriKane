# 给 A：新版验证集评估交接

本包为验证集开发结果，目标 `final_tier == Critical`，主范围 `has_future == True`。
配套最终方案：`D_stale`，模型版本 `20261031-054344`。
本次仅重新分析已保存预测，没有训练、改变模型/阈值、重抽切分或运行留出评估。

## 已完成的核对

- 上传预测共 1791 个事件；主范围 1333 个，Critical 88 个。
- 达到至少 90% 召回需找到 80 个，即实际召回 90.91%。
- 基线复核 113 个、模型 111 个；两者均找到 80 个，剩余 8 个。
- 复核量减少比例 1.77%（负数表示增加）；bootstrap 95% 区间 [-13.85%, 17.23%]，跨 0。
- 两种排序在各自预算外的 Critical 集合是否相同：True。这不代表两种排序整体相同。
- 不将旧版 Pc 报告的 56/119 作为新版结果，也不把距离单独算成模型功劳；真值直接取 A 输出的 final_tier。

## 请 A 做什么

1. 看 `critical_rank_review.csv`：按 `rank_delta_model_minus_baseline` 降序，正数表示模型比基线排得更后。检查哪些事件受影响，给出有证据的原因。
2. 看 `critical_outside_model_budget.csv`：这些是真实 Critical，但未进入模型前 111 个事件的复核队列。它不是按分类阈值定义的漏报清单。
3. 同时看 `all_future_event_ranks.csv` 中靠前的非 Critical；本次模型复核量较基线减少 2 个事件，找到的 Critical 数相同。
4. 核对评估输入、标签、消息截止时刻及模型版本；当前远程方案已标记为最终方案，本轮请完成误差解释和案例确认，不为改变报告数字而调整模型。
5. 若团队后续确实变更模型，交付更新的 validation_predictions.csv、metadata 和改进前后对比，让 C 用同一脚本重算。冻结后再由 A 交付留出预测，不能按留出表现调整模型。
6. 确认一个失败案例供 B 展示。候选事件 2958：当前 Safe、模型预测 Safe、真实最终 Critical，模型排第 409，基线第 1143。这只是验证回放，尚无物理原因结论。

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
