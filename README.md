# HiLTL / DeepLTL：LTL 与强化学习研究工作区

在 DeepLTL（Jackermeier & Abate, ICLR 2025）基础上研究自动机语义图、进展条件双目标价值、历史追踪及 Options。仓库保留训练实现、真实实验数据、论文工作稿和复现工具。

**2026-10-23：精简重跑已完成，51/51 阶段、25 模型、6,900 评估回合均通过独立归档审计。主环境 ZoneEnv 尚未证实方法提升。** `hv_track` 三种子主任务 macro SR 为 0%，内部 `gnn_track` 为 9.11%，`baseline` 为 27.78%。FlatWorld 与 Letter 单种子相对 `gnn_track` 有 +19.33/+4.67 pp 局部信号，但存在违反率或循环指标恶化，不代表稳定优势。

## 核心入口

- [详细实验结果与论文证据审查](docs/EXPERIMENT_RESULTS.md)：全部配置、任务、消融、统计限制和下一轮最小诊断路径。
- [可追溯轻量证据](results/reproduction/20261021-zone-minimal-v3/)：原始回合、训练日志、配置、汇总与检查点哈希；大模型仅保留服务器。
- [项目现状](docs/PROJECT_STATUS.md)、[验证路线](docs/CODE_VALIDATION_ROADMAP.md)、[运行与监视手册](docs/LONG_RUNNING_TASKS.md)。
- [论文导航](paper/README.md)：网站长稿、修订短稿、原稿与差异审查。
- [维护历史](docs/maintenance/RECORDS.md)：历史命令与验证记录，旧计划不代表当前任务。

## 方法与执行入口

统一研究入口为 `src/train/train_paper.py`，当前协议 `hiltl-product-v3`。原序列课程 PPO 保留在 `src/train/train_ppo.py` 及 `run_*.py`，两种协议不能混作同一实验。

|配置|含义|
|---|---|
|baseline|DeepSets/GRU 序列参考架构，统一产品协议下运行|
|gnn_single / gnn_track|匹配图编码器单 critic，对照无历史/有历史|
|hv_only / hv_track|双目标 critic，无历史/有历史|
|msgnn_only / msgnn_track|语义图增强，无历史/有历史|
|full_no_options / full_all|完整图与双头/历史组合，后者启用真实 Options/SMDP|

HV 的短头采用 TD 目标，长头采用折扣 MC 回报代理，融合价值参与 PPO；长头不是校准后的 LTL 满足概率。固定进展门控、guard 词元编码、真实 LDBA 转移、不可变历史与 Options 高低层更新均有实现。实现存在不等于已证实实验收益；循环任务报告有限时域接受访问代理。

`baseline` 并非官方 DeepLTL 完整课程与原预算的复现。历史 LTL2Action/GCRL-LTL 数值引用自原论文，未在本仓库按本轮协议重跑，不能与新实验合并宣传公平 SOTA 优势。

## 环境与快速检查

本工作区为 CloudFolder 挂载目录。所有项目运行在 SSH 服务器 conda `wty_LTL`，不在本机或 base 环境执行，不擅自变更共享依赖：

```powershell
cf.exe here
cf.exe shell
```

进入远端后：

```bash
conda activate wty_LTL
export PYTHONNOUSERSITE=1
export PYTHONPATH=src:src/envs/zones/safety-gymnasium
export MUJOCO_GL=egl
export PYTHONHASHSEED=0
python tests/test_paper_v3.py
python scripts/audit_paper_results.py
```

本地查看完成状态：

```powershell
.\scripts\check-training.ps1 -SampleSeconds 0
```

使用 `cf.exe` 避免 PowerShell 的 `cf.ps1` 吞掉 `--`。一次性命令使用 `cf.exe run -- /home/dell/anaconda3/bin/conda run --no-capture-output -n wty_LTL env PYTHONNOUSERSITE=1 <cmd>`。

已用环境包括 Python 3.10、torch 2.5.1+cu124、PyG 2.6.1、Gymnasium 0.28.1、MuJoCo 2.3.3、NumPy 1.23.5；Safety Gymnasium 使用仓库内定制源码。requirements.txt 不是完整锁定环境，pygame 精确版本约束仍有冲突，无头训练通过不代表全部渲染接口兼容。

短时训练示例（不会启动大队列，输出需新目录）：

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src python src/train/train_paper.py \
  --env FlatWorld-v0 --task-set configurations/flatworld_paper_v3.json \
  --configuration hv_track --device cpu --seed 1 \
  --steps 32 --rollout-steps 16 --max-episode-steps 16 \
  --batch-size 16 --epochs 1 --output experiments/manual-smoke-new
```

这仅验证训练链路，不能作性能证据。本轮已结束，不要重启旧195/31模型大队列。下一步先做小规模失败诊断。

## 目录与数据规则

|目录|用途|
|---|---|
|src/|模型、PPO、LTL/自动机、环境、训练与评估|
|configurations/|训练/验证/测试任务清单与协议配置|
|scripts/、tests/|队列、监视、结果审计/统计与方法检查|
|results/reproduction/|本轮可发布的真实轻量证据|
|results/、results_infinite/|原有历史结果，来源与本轮分开|
|experiments/|服务器完整模型、冻结源码与原始作业日志；Git 忽略|
|overleaf/、paper/|论文工作稿、网站长稿、原论文、历史与资源|
|docs/|当前结果、现状、验证路线、运行手册与历史记录|
|rabinizer4/、src/envs/zones/safety-gymnasium/|必要第三方工具/资源，保留许可证|

网站长稿 `overleaf/TLT/` 保持原样；`overleaf/TLT__Copy_/` 是经授权修订的工作稿，原短稿保存在 `paper/archive/pre-revision/`。本轮未改论文源码，后续应将新证据与论文实验段同步。旧图的硬编码/模拟数据不充当新结果。

保留 LICENSE 与第三方许可。GitHub 不含完整模型二进制；请单独异地备份服务器 experiments/。共享服务器不保存个人 GitHub 凭据，发布使用个人 Windows 的 Git/GCM HTTPS 授权。
