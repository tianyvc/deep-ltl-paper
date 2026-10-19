# 整理后验证报告

生成时间：2026-10-16T04:13:08.907186-04:00

## 实际结果

| 检查 | 结果 |
|---|---|
| Python syntax scan | 通过；错误数 0 |
| GNN embedding | 通过（exit 0） |
| GNN initialization and forward | 通过（exit 0） |
| Rabinizer direct executable | 通过（exit 0） |
| Training CLI | 通过（exit 0） |
| Big FlatWorld launcher CLI | 通过（exit 0） |
| Reward regression | 通过（exit 0） |
| FlatWorld-v0 PPO 8 steps | 通过（exit 0） |
| LetterEnv-v0 PPO 8 steps | 通过（exit 0） |
| LaTeX paper.tex | 未通过（exit 1） |
| LaTeX paper-new.tex | 未通过（exit 1） |

## 论文阻塞原因

- LaTeX paper.tex：! LaTeX Error: File `icml2025.sty' not found.；! Emergency stop.；!  ==> Fatal error occurred, no output PDF file produced!
- LaTeX paper-new.tex：LaTeX Warning: File `zoneenv_gFblue_yellow.png' not found on input line 316.；! Package pdftex.def Error: File `zoneenv_gFblue_yellow.png' not found: using d；!  ==> Fatal error occurred, no output PDF file produced!

## 运行环境与边界

所有 Python 运行检查在远端 Linux 执行。默认 Python 3.10 的 torch 2.6+cu124 需临时 CUDA 搜索路径，Gym plugin 还需 glfw。此次仅对子进程设置：

- LD_LIBRARY_PATH：来自 /home/dell/anaconda3/envs/ib-edl/lib/python3.10/site-packages/nvidia/*/lib。
- PYTHONPATH：src 加 /home/dell/anaconda3/envs/wty_LTL/lib/python3.10/site-packages。
- MUJOCO_GL=egl。

没有安装新依赖或修改共享环境。该跨环境组合用于本次排障检查，不是推荐安装方案；应后续建立锁定的独立环境。PyG 的旧 torch-scatter/cluster/sparse/spline 扩展存在 ABI 警告并被禁用，部分计算使用回退路径。Letter 还有观测空间范围警告。

整理前 GNN embedding/训练 CLI 已成功，Rabinizer 直接调用因执行位失败，8 步 FlatWorld 在日志阶段因零长度回合除零失败。整理后这两个失败已消除。初始输出保存在 baseline_validation.json 和 baseline_smoke.json。

FlatWorld 与 Letter 各 8 步验证覆盖环境构造、模型前向、rollout、PPO update 和日志；不能证明长训稳定、增强收益、所有环境兼容。ARPS/ADR 无完整回合时可为 NaN。没有真实 checkpoint 评估、Zone 完整运行、多种子性能实验或 CUDA 性能验证。论文完整 BibTeX 和多轮引用检查因前置编译失败没有完成。

## 原始记录

完整命令、退出码、耗时和 stdout/stderr 见 [validation.json](validation.json)。该文件保留环境警告及论文报错，避免只记录“通过”。

## 资源完整性与导航

核对了 1,704 个受保护文件的 SHA-256（已有结果、评估公式、Rabinizer 和 Safety Gymnasium）：没有内容差异。当前导航/状态文档链接没有缺失，shell 脚本 bash -n 检查没有报错。机器记录见 [integrity.json](integrity.json)。历史档案原文链接不计入当前入口断链判定。
