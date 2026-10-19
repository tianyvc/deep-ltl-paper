# HiLTL / DeepLTL：LTL 与强化学习研究工作区

本项目在 DeepLTL（Mathias Jackermeier、Alessandro Abate，ICLR 2025）基础上研究语义增强图网络、长短期价值融合和层次决策。工作区同时保存可执行训练代码、探索模块、历史基线结果、论文草稿及修改记录。

**当前可信状态：基础 PPO + 序列编码 + GNN 路径可进行小规模运行验证；完整 HiLTL/LongShort 方法尚未形成经过端到端验证的实现。现存结果不能证明论文中所有性能提升数字。** 具体证据、缺口和验证边界见[项目现状详解](docs/PROJECT_STATUS.md)及[验证报告](docs/maintenance/VALIDATION.md)。

## 文档入口

- [项目现状详解](docs/PROJECT_STATUS.md)：研究目标、逐模块实现、配置、数据、论文版本、已知问题、后续工作。
- [文件与代码清单](docs/FILE_INVENTORY.md)：源文件职责线索、类/函数、文档与资源分布、CSV 表头与行数。
- [论文导航](paper/README.md)：当前候选稿、模板、历史版本、图表和编译限制。
- [文档导航与历史汇编](docs/README.md)。
- [整理记录与恢复方法](docs/maintenance/CLEANUP.md)及[完整操作清单](docs/maintenance/cleanup-manifest.json)。

## 目录

```text
src/                    模型、PPO、环境、LTL/自动机、采样、预处理、评估与日志
run_*.py                常用环境训练启动器（从仓库根目录执行）
rabinizer4/              LTL → LDBA 转换工具，含第三方源码、运行库和文档
eval_datasets/          LTL 评估公式集
results/               已有有限任务基线 CSV、历史训练日志
results_infinite/      已有无限任务的有限时域评估 CSV
scripts/               数据录入、分析、评估命令生成与微基准
tests/                 可用 GNN 检查；legacy/ 保存旧接口与占位脚本
paper/                 两份 HiLTL 候选稿、参考文献、原论文、图片和投稿材料
docs/                  当前状态、完整文件清单、技术/实验/论文修改历史
.maintenance/          本地恢复备份（忽略于 Git，不应发布）
```

## 环境与安装

推荐 **Linux + Python 3.10 + Java**。日志使用 `fcntl`，原生 Windows 运行不是当前已验证路径；CloudFolder 用户应通过 `cf run -- …` 或 `cf sh -- …` 在远端执行。

`requirements.txt` 仅列出部分 Python 依赖，**不是完整锁定环境**。还需要相互兼容的 PyTorch、PyTorch Geometric、Gymnasium、MuJoCo/Safety Gymnasium 等；Zone 环境包含本地第三方 Safety Gymnasium 源码和模型资源，不要删除。依赖版本与 CUDA/扩展 ABI 必须匹配，不建议直接覆盖共享环境。

在已配置好这些依赖的独立环境中，从仓库根目录执行：

```bash
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD/src"
chmod +x rabinizer4/bin/*
java -version
python src/train/train_ppo.py --help
```

如需使用仓库内 Safety Gymnasium，请先阅读 `src/envs/zones/safety-gymnasium/` 自身安装说明。无显示器服务器可按运行环境设置 `MUJOCO_GL=egl`。本次检查使用的临时服务器依赖组合及 ABI 警告详见验证报告；它不是推荐的可移植环境配置。

## 最小训练检查

```bash
PYTHONPATH=src python src/train/train_ppo.py \
  --env FlatWorld-v0 --model_config FlatWorld-v0 --curriculum FlatWorld-v0 \
  --name smoke --seed 1 --device cpu --num_procs 1 \
  --steps_per_process 8 --batch_size 8 --epochs 1 --num_steps 8 \
  --no-save --no-log_csv --no-log_wandb
```

这是运行连通性检查，不产生有统计意义的性能结果。尚无完整回合时 ARPS/ADR 可为 NaN；`--no-save` 仍可能写入词表目录。

常用训练入口（默认训练预算较大，请先检查参数）：

```bash
PYTHONPATH=src python run_flatworld.py --name flatworld_run --seed 1 --device cpu
PYTHONPATH=src python run_letter.py --name letter_run --seed 1 --device cpu
PYTHONPATH=src python run_zones.py --name zones_run --seed 1 --device cpu
PYTHONPATH=src python run_flatworld_big.py --name big_run --seed 1 --device cpu --size 8
```

大地图尺寸支持 8、12、16、20。训练状态通常写入 `experiments/ppo/<env>/<name>/<seed>/`，课程由 `src/sequence/samplers/curricula.py` 定义，正式模型配置位于 `src/config/model_config.py`。不要将归档配置当作有效配置入口。重新使用同一实验名/种子会尝试恢复状态，应明确区分新实验和续训。

训练 CLI 提供 `--enable_hierarchical_value`、`--enable_multiscale_gnn`、`--enable_automaton_tracking`、`--enable_options` 等探索开关；**开关存在不等于对应方法已接入 PPO 损失和 rollout**。请先阅读现状详解，勿直接据此命名完整方法实验。

## 检查与评估

```bash
PYTHONPATH=src python tests/test_gnn_embedding_quick.py
PYTHONPATH=src python tests/test_gnn_fix.py
rabinizer4/bin/ltl2ldba -i 'F a' -p -d -e
PYTHONPATH=src python src/evaluation/simulate.py --help
```

实际策略评估需要匹配的 checkpoint 和词表；整理前仓库没有随附模型检查点。请使用 `simulate.py --help` 中真实支持的参数；历史批量命令生成器包含不受支持的 LongShort 参数，运行前需修正。Letter/FlatWorld 使用确定性策略还是采样策略应明确记录，不能混用统计口径。

`tests/legacy/` 不是通过的回归测试集合：包含旧 API、缺失模型依赖、捕获异常后继续和 TODO 占位。不能用其输出宣称算法验证成功。微基准 `scripts/benchmarks/benchmark_gnn_optimized.py` 也不替代任务效果实验。

## 结果与论文

现存 CSV 主要包含 DeepLTL、GCRL-LTL、LTL2Action 等基线；没有完整七组增强消融实验。绘图脚本中的硬编码/模拟数据只能用于示意，不能作为实验结果。历史训练日志记录一次 FlatWorld GNN 训练，不构成多种子对照证据。

- `paper/paper.tex`：ICML 风格较完整候选稿，缺少 `icml2025.sty`。
- `paper/paper-new.tex`：IEEE conference 风格后续候选稿，已修正参考文献文件名。
- `paper/templates/IEEE_journal_example.tex`：IEEE 示例模板，非完成的期刊稿。
- 图片统一存放 `paper/figures/`；原论文 PDF 保留在 `paper/`。
- 最新稿身份仍需作者确认；整理没有替作者选择投稿终稿，也未改写论文实验结论。

详细编译结果、缺图、参考文献核验状态和可复现性工作见[论文导航](paper/README.md)。

## 来源与许可

保留原项目 `LICENSE` 以及 Rabinizer、Safety Gymnasium 等第三方各自的许可文件。原方法的 DeepSets/RNN 序列编码与本项目增加的 GNN 应清楚区分；引用和论文陈述应以原论文及可追溯实验为依据。
