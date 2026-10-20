# 长时间训练、评估与后台任务手册

更新：2026-10-20。全部命令在远端仓库根目录、conda `wty_LTL` 中运行。本轮只执行短时工程验证，未开展百万步实验。一次真实后台停止测试将目标设为1M，但32步/2更新后即主动停止并检查保存，未遗留运行进程。证据在 `docs/maintenance/validation-runs/20261020/`，实际过程见 [RECORDS.md](maintenance/RECORDS.md)。

## 1. 当前可执行任务与边界

当前协议为`hiltl-product-v2`，旧v1检查点不能用于恢复。七配置`baseline/hv_only/msgnn_only/hv_track/msgnn_track/full_no_options/full_all`均有实际入口；full_all已接入Options子策略与高层SMDP更新。`--task-set configurations/flatworld_tasks.json`支持多公式训练和隔离的held_out评估，Letter/Zone有各自任务清单。无GNN序列课程基线可用`train_ppo.py --architecture deepltl`，与统一乘积协议baseline区别见README。

| 任务 | 入口与建议预算 | 产出/限制 |
|---|---|---|
| 固定公式pilot | train_paper，先10k再1M | 排查长回合、显存、奖励/梯度退化；不当作zero-shot |
| 多公式七配置正式训练 | 同任务清单×配置×种子1–5，每运行1M | 每个运行一个output；完整预算本轮未跑 |
| Options对照 | full_all对full_no_options，相同其余参数 | 显式记录option-max-steps、内在奖励与元策略统计；原稿未给齐训练细节 |
| zero-shot评估 | `--evaluate 500 --eval-split held_out` | 每held-out公式500回合/训练种子；不更新权重、不接触训练采样 |
| 训练分布评估 | `--eval-split train` | 与held-out输出分开，不混称迁移 |
| 门控消融 | --gate fixed / learned / constant | 分别训练，不能用不同gate加载同一检查点 |
| 多种子统计 | scripts/summarize_paper_runs.py | 独立单位为训练种子；n=1无CI，引用外部结果不参与新置信区间 |
| 外部LTL2Action/GCRL-LTL | 本阶段不接入官方代码，按用户要求保留原论文数据 | 标成引用数据，不写成本仓库同协议重跑 |

本轮代码落地不能替代正式性能实验。概率/MC、skip边、SCC解释及论文状态数等定义仍有记录在案的差异。可执行工程选择见README和记录；网站终稿不改动。

## 2. 远端环境和 GPU

本机 PowerShell：

```powershell
cf here
cf shell
```

远端交互 shell：

```bash
cd /home/dell/workspace/Tianyvc/deep-ltl-paper
conda activate wty_LTL
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD/src"
export MUJOCO_GL=egl
export OMP_NUM_THREADS=2
python -c 'import sys,torch,safety_gymnasium;print(sys.executable);print(torch.__version__,torch.__file__);print(safety_gymnasium.__file__);print(torch.cuda.is_available())'
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv
```

实测服务器有 4× NVIDIA RTX 4090，各 24,564 MiB，驱动 560.35.03；驱动显示支持 CUDA 12.6，实际 PyTorch 2.5.1+cu124 自带 CUDA 12.4 运行库。初始快照：

| GPU | 已用 MiB | 空闲 MiB | 利用率 |
|---|---:|---:|---:|
| 0 | 17915 | 6197 | 9% |
| 1 | 17551 | 6561 | 23% |
| 2 | 17473 | 6639 | 9% |
| 3 | 17657 | 6455 | 13% |

每卡已有其他 Python 与 Ollama 工作负载。快照不构成资源预留；开训前重新检查并遵守服务器分配规则，不结束他人进程。短时 GPU 检查使用物理卡 2，不能据此假设长训仍可用。小图/小模型 GPU 可能比 CPU 慢，优先 pilot 实测。

本轮仅补齐 wty_LTL；训练必须禁用用户级 `.local`。安装后清单为 `validation-runs/20261020/pip-freeze.log`，不是跨机器完整锁文件；Safety-Gymnasium 是本仓库 editable 源码，不能用 PyPI 上游版本替代。`pip check` 仍报告 pygame 2.6.1 与该包要求 2.1.0 不符；三环境无渲染短训通过，未证明历史图形界面/所有 pygame API 兼容。没有更改共享用户包或其他 conda 环境。

## 3. 可复制的后台训练命令

后台工具 `scripts/long_task.py` 使用当前环境 Python 启动独立会话，SSH 断开后可继续；不需要 tmux/nohup。它记录精确 argv、路径、环境覆盖项、PID 和 `/proc` 启动身份、每 2 秒心跳、退出码及日志，不持久化整个环境/凭证。仅支持远端 Linux wty_LTL。后台任务会继续占用资源，启动前应完成第 2 节检查。

先运行一个固定公式 pilot，`--gpu 2` 将物理卡 2 映射为子进程的 `cuda:0`；任务与模型输出目录都必须使用新名字：

```bash
python scripts/long_task.py start \
  --job-dir experiments/jobs/flat-hvtrack-seed1-pilot --gpu 2 -- \
  src/train/train_paper.py \
  --env FlatWorld-v0 --formula 'F blue & F green' \
  --configuration hv_track --seed 1 --device cuda:0 \
  --steps 10000 --rollout-steps 512 --max-episode-steps 500 \
  --batch-size 64 --epochs 4 --gamma .998 --lr .0003 \
  --output experiments/paper/flat-hvtrack-seed1
```

CPU pilot 将 `--device cuda:0` 改为 `--device cpu`，并去掉 `--gpu 2`。Letter 使用 `--env LetterEnv-v0 --formula 'F a & F b' --max-episode-steps 75`；Zone 使用 `--env PointLtl2-v0 --formula 'F blue & F green' --max-episode-steps 1000`。公式命题须来自环境，不能将颜色名原样传入 Letter。更换环境也必须更换 output 和 job-dir。

pilot 完成并检查资源和学习信号后，可以保留所有方法参数，将目标改为 `--steps 1000000 --resume`，使用新的 job-dir，继续同一模型输出。也可新 output 从零训练 1M。不要将上述短时验证用的 horizon16 / epoch1 参数用于论文正式实验。

多配置、多种子推荐逐个启动，先确认单任务显存和吞吐，再安排并发；每个组合一个 output、每次进程一个 job-dir。不要一次无上限启动 5×多种子×多环境。

如不进入交互 shell，本机可将同一个启动命令用 `cf run -- /home/dell/anaconda3/bin/conda run --no-capture-output -n wty_LTL python scripts/long_task.py ...` 路由到远端；涉及引号和重定向时优先使用 cf shell，避免 Windows/SSH 多层引号混淆。

## 4. 如何判断运行、完成和失败

```bash
python scripts/long_task.py status --job-dir experiments/jobs/flat-hvtrack-seed1-pilot
tail -n 20 experiments/jobs/flat-hvtrack-seed1-pilot/output.log
tail -f experiments/paper/flat-hvtrack-seed1/metrics.jsonl
cat experiments/paper/flat-hvtrack-seed1/progress.json
nvidia-smi
```

`tail -f` 用 Ctrl-C 结束查看，不会停止后台训练。

- `running` 且 supervisor_alive=true：监督进程身份匹配。心跳更新仅证明监督进程存活，训练可能在收集长回合；结合日志、CPU/GPU 与进度文件判断是否卡住。
- `starting`：训练可能尚未导入依赖或第一次更新尚未结束；检查 supervisor.log、output.log。
- `completed` 且 exit_code=0：子进程正常返回。训练还必须检查 progress.state=completed、steps≥target_steps、status.pt存在、损失/梯度有限。后台“completed”本身不证明训练目标达到或实验有效。
- `failed`：查看 exit_code/error 和 output.log；OOM、缺包、参数错误、已有输出等均不能写成训练成功。
- `interrupted`：主动停止。退出码可能为 0（训练器保存后正常返回）或负信号码；仍须检查 checkpoint 与 progress。
- `stale_or_crashed`：监督进程消失但未写最终状态，可能 kill -9、重启或异常。不能继续依赖旧 PID；先检查子进程是否仍活着及输出，再决定恢复。

工具通过 PID 加启动时间防止 PID 复用误报；若监督进程本身被强杀，子训练可能继续存活，需要通过 status 中 child_pid、`ps -fp <PID>` 和 `/proc/<PID>/cmdline` 人工确认归属。不要 `pkill python`，不要按旧 PID 盲目杀进程。

## 5. 安全停止与续训

```bash
python scripts/long_task.py stop --job-dir experiments/jobs/flat-hvtrack-seed1-pilot
python scripts/long_task.py status --job-dir experiments/jobs/flat-hvtrack-seed1-pilot
```

stop 只向身份匹配的本任务监督进程发送 SIGTERM，再转发给训练子进程。新训练器在当前完整回合收集和参数更新结束后保存；可能需要等一个 horizon 和一次 PPO update，不是立即终止。checkpoint 每次 update 原子替换；最近一次未完成 update 在进程强杀后会丢失。不要同时对同一 output 运行训练、续训或评估。

续训示例（先确认旧任务结束）：

```bash
python scripts/long_task.py start \
  --job-dir experiments/jobs/flat-hvtrack-seed1-million --gpu 2 -- \
  src/train/train_paper.py \
  --env FlatWorld-v0 --formula 'F blue & F green' \
  --configuration hv_track --seed 1 --device cuda:0 \
  --steps 1000000 --rollout-steps 512 --max-episode-steps 500 \
  --batch-size 64 --epochs 4 --gamma .998 --lr .0003 \
  --resume --output experiments/paper/flat-hvtrack-seed1
```

`--steps` 表示累计目标，不是额外增加的步数。恢复要求 protocol、env、formula、configuration、max_episode_steps、gamma、lr、seed、batch_size、epochs、rollout_steps、gate、option_max_steps、propositions、task_manifest、implementation_sha256 相同；device 与目标预算可改。完整加载模型和 optimizer、随机状态，不允许静默随机初始化；只读取自己生成的可信 checkpoint。恢复从新回合开始，未保存 MuJoCo/环境物理状态，不能承诺轨迹与不中断运行逐位一致。

## 6. 评估也是独立任务

训练结束后，保持同一组训练配置参数，增加 `--evaluate 500`：

```bash
python scripts/long_task.py start \
  --job-dir experiments/jobs/flat-hvtrack-seed1-eval --gpu 2 -- \
  src/train/train_paper.py \
  --env FlatWorld-v0 --formula 'F blue & F green' \
  --configuration hv_track --seed 1 --device cuda:0 \
  --rollout-steps 512 --max-episode-steps 500 \
  --batch-size 64 --epochs 4 --gamma .998 --lr .0003 \
  --evaluate 500 --eval-seed 10001 --deterministic \
  --output experiments/paper/flat-hvtrack-seed1
```

单公式默认train评估输出 `evaluation-train-10001-500-det.json`，task-set默认held_out输出 `evaluation-held_out-10001-500-det.json`，包含每回合 seed、实际步数、奖励和折扣回报、success、violation、accepting visits、checkpoint SHA-256、protocol 和有限/无限标记。随机策略用 `--no-deterministic`，输出后缀 sample；同名评估存在时拒绝覆盖，可改 eval-seed，避免误删旧证据。确定性与随机结果不能混算。

有限任务 success_rate 与无限任务 accepting visits 是不同量；对 `G F ...` 等无限任务，只得到有限时域访问代理，success_rate 字段为 null，不据此推断严格无限满足概率。对同一个固定公式换一个评估重置种子仍不是zero-shot。v2可用独立task-set train/held_out执行真实跨公式评估；默认held_out须明确报告，不能把清单外公式随意混入。

## 7. 输出与预算解释

```text
experiments/jobs/<job>/
  job.json          精确命令、cwd、环境覆盖项和创建时间
  launch.json       监督 PID 与启动身份
  status.json       心跳、子 PID、完成/失败/中断、退出码
  output.log        训练/评估 stdout + stderr
  supervisor.log    后台工具本身的异常
experiments/paper/<run>/
  config.json       参数、protocol_notes、实际自动机节点边与语义特征
  status.pt         模型、optimizer、累计 steps/updates、配置及随机状态
  metrics.jsonl     每次更新 PPO/双头损失、梯度、实际采样量、运行耗时
  episodes.jsonl    每回合奖励序列与成功/违反/接受次数
  progress.json     running / completed / interrupted 及目标和实际步数
  evaluation-*.json 逐回合评估与 checkpoint 身份
```

新协议每个 rollout 收集完整有限时域回合，horizon 截断按该有限时域任务结束处理并 bootstrap=0；这不同于无限任务 continuing bootstrap。累计实际 steps 可超过目标，超出量取决于最后一批完整回合，必须用记录的 steps 归一比较，不能只报告 CLI 预算。epoch/minibatch 不是环境步数。

`elapsed_seconds` 从当前进程训练循环开始计时，不含导入/建图，也不累加以前续训的耗时。吞吐估计应使用 pilot 两个稳态更新之间的 Δsteps/Δelapsed，再用 remaining_steps/steps_per_second 估算；完整耗时还包含启动、转换、保存和评估。horizon16 smoke 耗时不能线性外推 horizon500/1000 或不同图大小。历史重编码、GPU低利用、MuJoCo环境开销和 JSON/checkpoint IO 都值得 profiling。

experiments 默认 Git 忽略；重要模型与结果需另行备份并记录配置/代码哈希。本手册未自动发布或上传任何训练产物。

## 8. 多公式 / Options / zero-shot 的后台命令

以下参数为正式预算示例，先将steps降到10000检查资源和学习信号，再决定完整训练；本轮不自动启动长训。

```bash
python scripts/long_task.py start \
  --job-dir experiments/jobs/flat-v2-fullall-seed1 --gpu 2 -- \
  src/train/train_paper.py \
  --env FlatWorld-v0 --task-set configurations/flatworld_tasks.json \
  --configuration full_all --gate fixed --option-max-steps 50 \
  --seed 1 --device cuda:0 --steps 1000000 \
  --rollout-steps 512 --max-episode-steps 500 --batch-size 64 --epochs 4 \
  --gamma .998 --lr .0003 --output experiments/paper/flat-v2-fullall-seed1
```

训练完成后复用相同参数，把训练命令中的steps保留，增加`--evaluate 500 --eval-split held_out`并换新的job-dir，例如`flat-v2-fullall-seed1-heldout`。output保持模型目录，不加resume也会严格加载评估。每个held-out公式都评估500回合；3个公式意味着1500回合/训练种子。`--eval-formula '<清单内完整字符串>'`可只评估所选公式。清单/源实现哈希变化时拒绝旧checkpoint，应新开实验，不能忽略不匹配。

`--configuration`换baseline/hv_only等其余六组时也要换output/job-dir。训练种子2–5各自新output；评估重置种子与训练种子分开记录。held-out评估不更新参数，包含weights_unchanged、checkpoint_steps和模型hash；恢复后生成的新评估不要覆盖旧文件。

Options每条真实转移可作为目标技能，元控制器输出大小随当前图候选集合变化，未采用固定状态数one-hot，因此可处理held-out图大小。非陷阱进展/接受/epsilon转移优先，排除有进展时的非接受原地循环。子策略按内在回报分段更新，并共享任务PPO信号；元策略每次option决策使用实际duration的gamma^duration折扣。每回合的option边、起止步、持续时间和终止原因写入episodes.jsonl，meta_loss、meta_gradient_norm、option_decisions、option_mean_duration写入metrics.jsonl。超时不是自动机成功，不能计入SR。

## 9. 统计与正式实验验收

`python scripts/summarize_paper_runs.py <evaluation文件...> --seeds 1 2 3 4 5 --episodes 500 --output <新文件>`汇总真实逐回合数据。每组必须有相同五个独立训练种子；输入既可包含多个configuration，也可包含多个formula。输出每组seed均值、样本标准差、95%t区间，与baseline相同训练种子的SR绝对差配对区间。无限任务SR为null，报告accepting visits等有限时域代理。

引用原论文的外部对照保留在旧结果中，不能作为新运行种子或参与这份统计的置信区间。实际环境步数记录在每份评估的checkpoint_steps，避免续训后用最新progress冒充旧模型评估预算。

正式验收仍需：长horizon的奖励非退化/接受频率与参数学习、长训稳定性、原始参考版本一致性、地图和seed公平性、更大图的性能profiling、CUDA与pygame版本冻结，以及论文定义冲突裁定。代码已有实现不等于复现了论文数值；不能为接近报告值而重写旧CSV或使用示意图数据。
