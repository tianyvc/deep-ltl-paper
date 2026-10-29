# HiLTL

HiLTL 在 [DeepLTL](https://github.com/mathiasj33/deep-ltl) 的序列课程与 PPO 策略上加入自动机语义图、多层图池化、进展门控双价值头及逻辑历史编码，支持 ZoneEnv、LetterEnv 和 FlatWorld。

论文位于 [`overleaf/HiLTL-final/`](overleaf/HiLTL-final/)。仓库提供研究代码与验证证据，当前实现尚未重新复现终稿性能数值；覆盖边界见[方法与实现](docs/METHOD_AND_IMPLEMENTATION.md)。

## 目录

|路径|内容|
|---|---|
|`src/`|训练、模型、PPO、课程、逻辑与环境|
|`configurations/`、`eval_datasets/`|配置、任务模板与原评估资源|
|`scripts/`|评估、轻量对照、搜索诊断和上游兼容检查|
|`tests/`|回归、模型合同、自动机映射及微型训练|
|`overleaf/HiLTL-final/`|作者确认的 LaTeX、图片、文献与 PDF|
|`results/reference/`|保留来源与哈希的历史 CSV|
|`results/validation/`、`results/submission/`|验证、先导实验、终稿编译和投稿要求来源|
|`paper/`、`rabinizer4/`|DeepLTL 参考论文与 Java 转换工具|

## 环境

本工作区通过 CloudFolder 访问服务器，项目命令统一在远端 conda `wty_LTL` 执行：

```powershell
cf.exe here
cf.exe shell
```

进入远端仓库根目录后：

```bash
conda activate wty_LTL
export PYTHONNOUSERSITE=1
export PYTHONPATH=src:src/envs/zones/safety-gymnasium
export MUJOCO_GL=egl
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=2
```

主要版本见 [`requirements.txt`](requirements.txt)，需要 Java；论文编译需要 LaTeX/BibTeX。该文件是现有环境的主要依赖清单，完整全新安装尚未验证。定制 Safety-Gymnasium 源码、运行资源和许可证均在 `src/envs/zones/safety-gymnasium/`，请保留。

## 训练与评估

以下为短时示例，输出目录须使用新名称：

```bash
python src/train/train_tlt.py --env PointLtl2-v0 --configuration hv_track \
  --seed 1 --device cuda:0 --steps 4096 --num-procs 2 \
  --rollout-steps 1024 --epochs 2 --batch-size 64 \
  --output experiments/zone-hv-track-new

python scripts/evaluate_tlt.py \
  --checkpoint experiments/zone-hv-track-new/status.pth \
  --output experiments/zone-hv-track-new/eval-finite \
  --episodes 10 --max-seconds 300 --device cuda:0
```

配置为 `baseline`、`hv_only`、`msgnn_only`、`hv_track`、`msgnn_track`、`full_no_options`。其他环境名为 `LetterEnv-v0`、`FlatWorld-v0`。`--steps`、`--output` 必须显式提供；Zone其他默认参数面向长训练，短测请显式缩小预算。

训练输出 `config.json`、`vocab.pkl`、`status.pth`、`metrics.jsonl`、`progress.json`。原序列模型迁移用 `--warm-start /path/status.pth`，保留同目录 `vocab.pkl`；同配置续训用 `--resume`，步数为累计目标，环境中途轨迹会重启。评估输出 `episodes.json`、`summary.json`。`--task-group recurrent` 的有限窗口结果不能解释为无限时域满足概率。

## 检查与论文编译

在上述远端环境执行：

```bash
python tests/test_curriculum_transition.py
python tests/test_tlt_curriculum.py
python tests/test_paper_mapping.py --output experiments/checks/mapping.json
python scripts/verify_tlt_upstream.py --output experiments/checks/upstream.json
python tests/test_runtime.py --output experiments/runtime-new \
  --report experiments/checks/runtime.json
```

复制终稿到构建目录，避免产生原稿残留：

```bash
mkdir -p experiments/paper-build-new
cp overleaf/HiLTL-final/LTL/* experiments/paper-build-new/
cd experiments/paper-build-new
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
bibtex manuscript
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
```

## 文档

- [方法、配置与实现差异](docs/METHOD_AND_IMPLEMENTATION.md)
- [验证与先导实验](docs/VALIDATION.md)
- [终稿评估、投稿要求与目标建议](docs/SUBMISSION_GUIDE.md)
- [目录维护、归档恢复与发布方式](docs/WORKSPACE.md)

项目许可见 [LICENSE](LICENSE)，第三方组件沿用各自许可证。训练输出、缓存和历史档案不进入活动仓库。
