# HiLTL / DeepLTL：LTL 与强化学习研究工作区

本项目在 DeepLTL（Mathias Jackermeier、Alessandro Abate，ICLR 2025）基础上研究语义增强图网络、长短期价值融合和层次决策。工作区同时保存可执行训练代码、探索模块、历史基线结果、论文草稿及修改记录。

**当前版本：hiltl-product-v2（2026-10-20）。七个内部配置均有实际训练入口，包含真实 LDBA 语义图、双价值与门控、自动机历史、Options、多公式训练和 held-out zero-shot 评估；另有不含 GNN 的 DeepSets/GRU 课程基线。已开展短训、梯度、续训和评估验证，尚未完成百万步、多种子、500 回合的正式性能复现。** 实现选择和验证证据见[项目现状](docs/PROJECT_STATUS.md)与[执行记录](docs/maintenance/RECORDS.md)。

外部对照 LTL2Action、GCRL-LTL 的现存数值按用户确认引用自原论文，本阶段保留原数据，未接入官方实现或在本仓库重新运行；不能将其标成与本次新实验同一运行协议下的结果。网站终稿保持原样，内部有冲突的定义不通过改写实验数据来解决。

## 文档入口

- [项目现状](docs/PROJECT_STATUS.md)：代码、数据、论文的实际状态与证据边界。
- [代码核实与优化验证路线](docs/CODE_VALIDATION_ROADMAP.md)：方法合同、已知缺口、分阶段检查和验收标准。
- [长时间任务手册](docs/LONG_RUNNING_TASKS.md)：后台训练、GPU 选择、状态/日志、停止与续训、500 回合评估及输出。
- [论文入口](paper/README.md)：网站长短终稿、差异报告、旧稿与图表资源。
- [维护与验证记录](docs/maintenance/RECORDS.md)：历史清理、实际检查结果和恢复方法。

历史 Markdown 已收敛至 `docs/archive/document-history.zip`，成员清单保留原路径和 SHA-256；第三方文档与网站终稿保持不变。

## 目录

```text
src/                    模型、PPO、环境、LTL/自动机、采样、预处理、评估与日志
run_*.py                常用环境训练启动器（从仓库根目录执行）
rabinizer4/              LTL → LDBA 转换工具，含第三方源码、运行库和文档
eval_datasets/          LTL 评估公式集
results/               已有有限任务基线 CSV、历史训练日志
results_infinite/      已有无限任务的有限时域评估 CSV
scripts/               后台任务管理、按训练种子的统计汇总、分析与微基准
configurations/        三环境的 train / held_out 任务清单（不改写旧评估数据）
tests/                 GNN、方法语义、Options/SMDP、任务隔离和基线回归；legacy/ 为历史脚本
overleaf/              两个网站终稿原始快照（禁止改动内部文件）
paper/                 论文统一导航、对比报告、旧稿、原论文、图片和投稿材料
docs/                  当前现状、代码验证路线、维护记录与历史文档包
.maintenance/          本地恢复备份（忽略于 Git，不应发布）
```

## 环境与安装

推荐 **Linux + Python 3.10 + Java**。日志使用 `fcntl`，原生 Windows 运行不是当前已验证路径；CloudFolder 用户应通过 `cf run -- …` 或 `cf sh -- …` 在远端执行。

`requirements.txt` 仅列出部分 Python 依赖，**不是完整锁定环境**。还需要相互兼容的 PyTorch、PyTorch Geometric、Gymnasium、MuJoCo/Safety Gymnasium 等；Zone 环境包含本地第三方 Safety Gymnasium 源码和模型资源，不要删除。依赖版本与 CUDA/扩展 ABI 必须匹配，不建议直接覆盖共享环境。

本工作区所有项目命令使用远端已有 `wty_LTL` 环境，不擅自安装或变更依赖。先 `cf shell`，再 `conda activate wty_LTL`；之后从仓库根目录执行下列命令。后续训练、检查命令也均以该环境为前提：

```bash
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD/src"
java -version
python src/train/train_ppo.py --help
```

如需使用仓库内 Safety Gymnasium，请先阅读 `src/envs/zones/safety-gymnasium/` 自身安装说明。无显示器服务器可按运行环境设置 `MUJOCO_GL=egl`。本轮经用户授权仅补齐 wty_LTL，当前实测 torch2.5.1+cu124 / PyG2.6.1 / Gymnasium0.28.1 / MuJoCo2.3.3 / NumPy1.23.5；定制 Safety Gymnasium 必须来自仓库 editable 源码。pygame2.6.1 与其精确版本约束2.1.0仍冲突，三环境无渲染短训通过不代表所有渲染兼容；安装清单和失败记录见维护记录。一次性命令使用 `cf run -- conda run -n wty_LTL <cmd>`；若 conda 不在 PATH，本机可用 `/home/dell/anaconda3/bin/conda` 绝对路径调用同一环境。

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

## 七配置与方法入口

统一入口是 `src/train/train_paper.py`。`baseline` 使用原有 DeepSets/GRU 组件和 value-guided reach/avoid 序列搜索；其余配置使用真实乘积 LDBA 图。各配置共享环境奖励、任务清单和预算定义。

| configuration | HV | MSGNN | Track | Options |
|---|---|---|---|---|
| baseline | — | — | — | — |
| hv_only | ✓ | — | — | — |
| msgnn_only | — | ✓ | — | — |
| hv_track | ✓ | — | ✓ | — |
| msgnn_track | — | ✓ | ✓ | — |
| full_no_options | ✓ | ✓ | ✓ | — |
| full_all | ✓ | ✓ | ✓ | ✓ |

- SE-GNN：节点4D、边6D语义特征，三层64维/四头GATv2，残差/LayerNorm、图内softmax池化；独立守卫词元/GRU通路与进展边消息值保留命题身份，避免同构公式不可辨。
- HV：short/long隐藏层256/128，实际融合value进入PPO，分别计算TD/MC辅助损失。长期头是折扣回报代理，未约束成满足概率。
- 门控：默认 `--gate fixed` 为论文固定sigmoid式；`learned` 实现64隐藏单元MLP变体，`constant` 为.5消融。各变体单独记录，不能混作同一协议。
- Track：环境转移推进q、不可变历史快照、历史LSTM；forward和minibatch重放不推进自动机。
- Options：从真实非陷阱转移选目标，目标条件子策略持续执行至命中/状态改变/超时/回合结束；内在奖励子策略PPO与按持续时间折扣的高层SMDP PPO均实际更新。默认 `--option-max-steps 50`。终稿未给全训练细节，这是显式工程协议。
- 多公式与zero-shot：只采样task-set中的train；held_out仅评估，保存逐回合seed、公式、检查点哈希、训练步数和权重未变检查。拒绝文本重复及编译图指纹重叠；图指纹不是完整语言等价判定。

先用短时命令检查；output必须使用新目录：

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src python src/train/train_paper.py \
  --env FlatWorld-v0 --task-set configurations/flatworld_tasks.json \
  --configuration full_all --device cpu --seed 1 \
  --steps 32 --rollout-steps 16 --max-episode-steps 16 \
  --option-max-steps 4 --batch-size 16 --epochs 1 \
  --output experiments/my_v2_smoke
```

训练完成后保持相同训练参数，追加 `--evaluate 2 --eval-split held_out`，即可对清单中的每个held-out公式评估2回合；续训追加 `--resume --steps <累计目标>`。`--formula 'F blue'` 可替代task-set运行单公式，但同公式评估不能称zero-shot。Letter/Zone清单分别为 `configurations/letter_tasks.json` / `configurations/zones_tasks.json`。

v2与v1检查点不兼容，源码实现哈希也参与恢复校验，不允许缺键后静默随机初始化。参数、输出、后台和正式预算见[长任务手册](docs/LONG_RUNNING_TASKS.md)。

原序列课程中的无GNN基线使用 `src/train/train_ppo.py --architecture deepltl`，其他基础参数沿用最小训练检查；默认 `sequence_gnn` 保留旧基础结构。保存的architecture决定评估器如何构建模型，跨架构恢复会拒绝。该路径保留DeepLTL参考架构，未证明与原论文发布版本逐位一致；统一乘积入口的baseline也不等同原始课程序列训练协议。

旧 `--enable_*` 增强开关仍明确拒绝运行，避免把已知不完整的历史wrapper当作新方法。旧Options占位骨架保存在 `docs/archive/legacy-option-skeleton.zip`，有效实现为 `src/model/paper_options.py`。

## 检查与评估

```bash
PYTHONPATH=src python tests/test_gnn_embedding_quick.py
PYTHONPATH=src python tests/test_gnn_fix.py
PYTHONPATH=src python tests/test_paper_hiltl.py
PYTHONPATH=src python tests/test_paper_protocol.py
rabinizer4/bin/ltl2ldba -i 'F a' -p -d -e
PYTHONPATH=src python src/evaluation/simulate.py --help
```

实际策略评估需要匹配的 checkpoint 和词表；整理前仓库没有随附模型检查点。请使用 `simulate.py --help` 中真实支持的参数；历史批量命令生成器包含不受支持的 LongShort 参数，运行前需修正。Letter/FlatWorld 使用确定性策略还是采样策略应明确记录，不能混用统计口径。

`tests/legacy/` 不是通过的回归测试集合：包含旧 API、缺失模型依赖、捕获异常后继续和 TODO 占位。不能用其输出宣称算法验证成功。微基准 `scripts/benchmarks/benchmark_gnn_optimized.py` 也不替代任务效果实验。

## 结果与论文

现存 CSV 包含历史 DeepLTL、GCRL-LTL、LTL2Action 等基线；其中外部对照按用户确认为引用原论文数值，不是本仓库新运行。七配置现在均有运行入口和短时验证，但尚无完整预算的正式消融结果。绘图脚本中的硬编码/模拟数据只能用于示意，不能作为实验结果。历史训练日志记录一次 FlatWorld GNN 训练，不构成多种子对照证据。

- 网站长版：[`overleaf/TLT/manuscript.tex`](overleaf/TLT/manuscript.tex)，IEEE journal 排版。
- 网站短版：[`overleaf/TLT__Copy_/manuscript.tex`](overleaf/TLT__Copy_/manuscript.tex)，IEEE conference 排版。
- 两份均为用户指定的网站终稿，保留原路径和全部内容；旧候选稿与旧文献库已移到 `paper/archive/pre-overleaf/`。
- [长短终稿详细差异](paper/review/FINAL_VERSION_COMPARISON.md)：短版保留全部实验表数值，压缩算法和讨论，澄清门控符号与 Options 开关。
- [论文与代码对齐依据](docs/CODE_VALIDATION_ROADMAP.md)：后续工程整理的术语、配置、指标和实现缺口。
- 原样完整编译存在 natbib 模式问题；只在隔离副本使用数字引用后，两版分别为 13 页和 9 页。详见[本轮验证](docs/maintenance/RECORDS.md)。

图片生成器及历史资源仍在 `paper/`，与网站快照的同名图片不一定内容相同。论文完整导航见 [paper/README.md](paper/README.md)。

## 来源与许可

保留原项目 `LICENSE` 以及 Rabinizer、Safety Gymnasium 等第三方各自的许可文件。原方法的 DeepSets/RNN 序列编码与本项目增加的 GNN 应清楚区分；引用和论文陈述应以原论文及可追溯实验为依据。

## 按训练种子汇总评估

`scripts/summarize_paper_runs.py`读取各运行的`evaluation-*.json`及相邻config，先在每个训练种子内部平均回合，再以训练种子为独立单位计算均值、标准差和t区间，并输出与baseline的配对SR绝对差。默认要求种子1–5、每公式500回合；少种子/少回合仅供检查，不等于论文验证。n=1不输出置信区间，无限任务不输出严格满足SR。

```bash
python scripts/summarize_paper_runs.py \
  experiments/paper/run_seed1/evaluation-held_out-10001-500-det.json \
  experiments/paper/run_seed2/evaluation-held_out-10001-500-det.json \
  --seeds 1 2 --episodes 500 --output experiments/paper/two_seed_summary.json
```

工具拒绝重复训练种子、缺失种子、评估回合数不符和协议/任务清单不匹配；不会把引用的外部对照CSV混入新实验置信区间。正式比较仍须核对实际步数、地图、重置种子与相同协议。
