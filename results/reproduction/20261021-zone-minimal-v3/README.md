# 20261021-zone-minimal-v3 轻量证据

全部25模型、6900评估回合；2026-10-23审计通过。方法未在主环境证实提升，详见 [结果报告](../../../docs/EXPERIMENT_RESULTS.md)。

- summary.json/csv：按训练种子汇总的全部结果及配对差。
- runs/：逐模型 evaluation-minimal.json（含全部回合）、config、metrics、progress。
- specs/：环境/任务/回合/重置规范。
- manifest/state/events：原始计划和调度完成记录。
- audit.json：51阶段核验、检查点SHA/实际步数、CPU加载检查、导入模型原配置及差异。
- SHA256SUMS.json：轻量证据文件哈希清单。

不包含 status.pt、环境资源和完整冻结源码。完整模型只保留在服务器 experiments/，GitHub不是模型异地备份。记录中的绝对路径是原始溯源路径，其他机器需显式适配；manifest不是可移植的一键启动文件。
