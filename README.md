# HiLTL：以 TLT 长稿为主线的 DeepLTL 扩展

本仓库在 [DeepLTL](https://github.com/mathiasj33/deep-ltl) 的序列课程、DeepSets/GRU 策略和 PPO 基础上，实现自动机语义图、多尺度图池化、进展门控双价值头与自动机历史追踪。主环境为 ZoneEnv，同时保留 LetterEnv、FlatWorld。

当前阶段聚焦**实现对应与可运行性**。论文中的成功率及增益留待后续实验验证。方法对应关系、实现选择和未覆盖内容统一见 [阶段总结](docs/STAGE_SUMMARY.md)，机器可读检查结果见 [verification.json](docs/verification.json)。

## 保留的结构

|路径|用途|
|---|---|
|`src/train/train_tlt.py`|唯一维护的训练入口，沿用原 DeepLTL 序列课程|
|`src/model/`|原序列策略、LDBA 图特征、TLT 图网络与价值头|
|`src/torch_ac/`|原 PPO/GAE 与双目标辅助训练|
|`src/envs/`、`src/ltl/`、`src/sequence/`|三个环境、逻辑/自动机、原课程及序列搜索|
|`scripts/evaluate_tlt.py`|冻结模型、固定公式的有限/循环任务评估|
|`configurations/paper_tasks.json`|长稿三类有限任务及循环诊断模板|
|`tests/`|实现合同、论文映射、短训和离线官方基线核对|
|`overleaf/TLT/`、`overleaf/TLT__Copy_/`|长稿和短稿，本次整理未改动|
|`paper/DeepLTL-ICLR2025.pdf`|原论文及其附录|
|`eval_datasets/`|原评估任务与世界布局资源，未改动|
|`results/reference/`|保留的历史数据与精简来源记录，不能当作新性能验证|
|`rabinizer4/bin/ltl2ldba`、`rabinizer4/lib/owl.jar`|实际使用的公式转换工具，需 Java|

旧产品训练入口、旧矩阵调度器、过期脚本、重复论文图、历史文档和源码快照已经移除。旧代码可从 Git 历史查看；本地训练输出和缓存不纳入 Git。

## 运行环境

项目位于 CloudFolder 挂载盘，运行使用 SSH 服务器的 conda `wty_LTL`，不要在本机或 base 环境运行项目。

```powershell
cf.exe here
cf.exe shell
```

进入远端后，在仓库根目录：

```bash
conda activate wty_LTL
export PYTHONNOUSERSITE=1
export PYTHONPATH=src:src/envs/zones/safety-gymnasium
export MUJOCO_GL=egl
export PYTHONHASHSEED=0
```

`requirements.txt`记录已验证环境的主要运行依赖，本次整理没有安装或修改环境。Python 为3.10，主要版本为 torch2.5.1、PyG2.6.1、Gymnasium0.28.1、MuJoCo2.3.3、NumPy1.23.5。Safety-Gymnasium 使用仓库内定制源码与资源；其原安装元数据中的pygame2.1.0与现用2.6.1不同，不要对现有环境直接重装。完整的空环境安装尚未重新验证。

使用`cf.exe`可以避免PowerShell中的同名脚本吞掉`--`。本地一次性运行示例：

```powershell
cf.exe run -- /home/dell/anaconda3/bin/conda run --no-capture-output -n wty_LTL env PYTHONNOUSERSITE=1 PYTHONPATH=src:src/envs/zones/safety-gymnasium MUJOCO_GL=egl PYTHONHASHSEED=0 python tests/test_tlt_curriculum.py
```

## 快速检查

以下命令在上述远端环境执行；输出文件和目录使用新名称：

```bash
python tests/test_tlt_curriculum.py
python tests/test_paper_mapping.py --output experiments/checks/paper-mapping.json
python scripts/verify_tlt_upstream.py --output experiments/checks/upstream.json
python tests/test_runtime.py --output experiments/runtime-new \
  --report experiments/checks/runtime.json
```

前三项核实奖励/课程、图语义、门控、梯度与原模型兼容性。最后一项约分钟级，覆盖三个环境、六个配置共九组短训，验证更新、保存和有限参数；它不评估方法增益。

## 训练与配置

```bash
python src/train/train_tlt.py --env PointLtl2-v0 --configuration hv_track \
  --seed 1 --device cuda:0 --steps 4096 --num-procs 2 \
  --rollout-steps 1024 --epochs 2 --batch-size 64 \
  --output experiments/zone-hv-track-new
```

这是短时集成预算，原课程和1000步回合上限保持不变。支持`baseline`、`hv_only`、`msgnn_only`、`hv_track`、`msgnn_track`、`full_no_options`。**`full_all/Options`未纳入当前维护协议**，原因见阶段总结。Letter、FlatWorld分别指定`LetterEnv-v0`、`FlatWorld-v0`，其原回合上限为75、500。

`--steps`、`--output`必须显式提供，新运行目录必须不存在。未覆盖的Zone超参数保持原训练默认：16环境、rollout4096/环境、batch2048、10epoch、lr0.0003、discount0.998。请显式缩小短测参数，避免误用长预算。

原序列模型用`--warm-start /path/status.pth`及同目录`vocab.pkl`迁移；严格检查参数形状。已删除的产品路径`status.pt`不是兼容模型。续训使用同配置、同源码的`--resume`，`--steps`是累计目标；环境中的未完成回合会重启。

每次训练输出`config.json`、`vocab.pkl`、`status.pth`、`metrics.jsonl`、`progress.json`。完整自有训练源码哈希与课程采样状态、优化器、随机状态写入检查点。

## 评估与监视

```bash
python scripts/evaluate_tlt.py --checkpoint experiments/zone-hv-track-new/status.pth \
  --output experiments/zone-hv-track-new/eval-finite \
  --episodes 10 --max-seconds 300 --device cuda:0
```

默认读取任务配置中对应环境的三类有限公式；`--task-group recurrent`选择循环诊断任务，`--formulas 'F blue'`可显式指定公式。Zone默认确定性，Letter/FlatWorld默认采样。输出`episodes.json`和`summary.json`，记录模型SHA、评估脚本SHA、公式、种子、回合是否完整、成功、逻辑违例、物理回报与接受访问次数。循环任务的有限访问记录不能解释为无限时域满足概率；预算中断回合不计入成功率分母。

需要后台运行时，在远端已激活的环境中：

```bash
nohup python src/train/train_tlt.py --env PointLtl2-v0 --configuration hv_track \
  --steps 4096 --num-procs 2 --rollout-steps 1024 --epochs 2 --batch-size 64 \
  --device cuda:0 --output experiments/zone-background-new \
  > zone-background.log 2>&1 &
```

本地PowerShell查看状态：

```powershell
cf.exe run -- /home/dell/anaconda3/bin/conda run --no-capture-output -n wty_LTL cat experiments/zone-background-new/progress.json
```

`steps/updates`连续增加表示更新推进；`completed`表示达到步数目标或课程结束；`budget_stopped`表示更新间的时间预算耗尽；`failed`或日志Traceback需要排查。第一轮更新前可能尚无progress；旧JSON不能证明进程仍活跃，应结合`tail -n 20 zone-background.log`和`nvidia-smi`查看。没有完整回合时`success_rate=null`。训练结束、程序可运行与论文效果成立应分别判断。
