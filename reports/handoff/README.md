# A/B 验证评估交接索引

已生成新版官方口径四份产物，并分别整理两个交接包。主结果：1333 个有后续消息的事件、88 个最终 Critical；基线 113 个复核、模型 111 个复核，同样找到 80 个。

- A：`A_validation_handoff.zip`，先读 `A/README_A.md`，内含全部 Critical 排名和预算外清单。
- B：`B_validation_handoff.zip`，先读 `B/README_B.md`，内含 JSON、两组曲线及页面接入示例。
- 旧版 Pc 56/119 报告已在 C 的本地 `reports/archive/` 留存，不用于新版模型成绩。
- 两包均为验证集开发结果；本次未运行训练、未读取或评估留出预测。
- 交接方式：通过 origin 和 upstream 的 main 分支共享；A/B 拉取后各自读取对应目录或压缩包。文件和包的完整性可按 manifest.json 核验。
