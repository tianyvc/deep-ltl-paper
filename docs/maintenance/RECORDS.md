# 维护与验证记录

本文件区分历史检查、网站终稿整合和本轮 Markdown 清洗。旧验证环境与结论只适用于当时记录；代码后续修改需重新验证。具体算法执行路线见 [CODE_VALIDATION_ROADMAP.md](../CODE_VALIDATION_ROADMAP.md)。

## 1. 本轮 Markdown 清洗（2026-10-20，北京时间）

保留 7 份项目自有 Markdown（含 AGENTS），第三方文档不清理。清洗前完整自有 Markdown 快照保存到 [历史包](../archive/document-history.zip) 和 [成员清单](../archive/document-history-manifest.json)。旧导航、旧 inventory、历史稿和重复说明从当前文件树移除；内容可从包内按原路径取回。

旧实现对齐说明的完整方法合同、七配置矩阵和缺口表已合并进代码验证路线；路线新增具体 oracle、阶段门槛、梯度/恢复/统计/性能检查。本现状去除已失效的“终稿尚未确认”等判断。原清洗说明、两轮验证说明的全文收敛于本文后续历史章节。

验证在远端 wty_LTL 执行；未运行 RL 测试、训练或性能复现。检查范围包括受保护文件集合/哈希、历史包内容、文档本地链接、旧 Markdown 路径残留和变更空白检查。最终结果见 [md-cleanup-validation.json](md-cleanup-validation.json)；本轮迁移对应关系见 [md-cleanup-manifest.json](md-cleanup-manifest.json)。不改用户 AGENTS/.vscode，不提交或推送。

## 2. 首轮工作区清理（历史原文）

以下内容中的“本次”“原仓库没有提交”等指首轮清理时的状态，不代表当前 Git 状态。原文中档案 MD 已进入上述历史包，恢复时按清单查原路径。

# 工作区整理与恢复

### 范围

本次以保留研究资产和运行路径为原则：src、第三方依赖、评估公式及结果 CSV 保持原路径；整理文档、论文图片、测试和杂项。未提交、推送或发布。

原仓库没有任何 Git 提交，所有现有文件均为未跟踪状态，不能依赖 Git 恢复。清理前先创建：

- `.maintenance/before-cleanup.tar.gz`：2,934 个原始文件，约 47.2 MB。
- `.maintenance/original-inventory.json`：原路径、字节数、SHA-256。
- 备份排除了 .git、缓存、临时检查文件和检查产生的 experiments；不是 Git 历史备份。

以上备份已加入 .gitignore，仅供本地恢复。重要研究资产仍建议另做独立存储备份。

### 整理内容

- 删除 __MACOSX、.DS_Store、过期 build、硬编码编辑器配置、引用缺失脚本的 test_speed.sh、空且未引用的 graph_builder.py。
- GNN/LongShort 技术文档、实验计划及四份根目录论文指南分别汇编；原文完整保留，历史失效链接不冒充当前可用入口。
- 不同论文草稿全部保留在 docs/archive/papers；旧配置集中到 docs/archive/config。
- 论文图片统一到 paper/figures，内容一致才去重；保留不同变体。
- 生成器集中到 paper/scripts；投稿材料归 submission，IEEE 空示例归 templates。
- 可运行 GNN 检查归 tests，旧接口及占位脚本归 tests/legacy；微基准归 scripts/benchmarks。
- parallel.log 有真实训练记录，保存为 results/logs/flatworld_gnn_test_seed5.log。
- 对移动脚本与论文资源路径作配套更新。

逐项操作见 [cleanup-manifest.json](cleanup-manifest.json)，图片映射见 [figure-map.json](figure-map.json)。原始文件内容可从备份取回。

### 必要代码修复

- 无正长度回合时 ARPS/ADR 返回 NaN；跳过零长度初始化占位，防止日志除零，不将未完成回合伪装成有效样本。
- checkpoint 失败分支使用已有 info 方法输出 WARNING，避免调用不存在的 warning。
- Rabinizer bin 恢复执行位。
- run_* 使用 sys.executable 并传播子进程失败。
- 大地图启动器新增 size 参数，仅接受 8/12/16/20，环境 ID 与注册保持一致。
- 主模型、PPO 优化公式和已有 CSV 未作算法性修改。

### 恢复方法

建议先在仓库外的独立目录解压并核对清单，再选择恢复文件，避免直接覆盖当前成果。例如 Linux：

```bash
mkdir -p /tmp/deep-ltl-before-cleanup
tar -xzf .maintenance/before-cleanup.tar.gz -C /tmp/deep-ltl-before-cleanup
```

结合清理清单恢复所需原始路径。恢复前保留当前新增文档；不要将解压操作当作自动安全合并。隐藏备份不属于论文附件或发布内容。

## 3. 首轮代码连通性验证（历史原文）

此处采用过跨 conda 的临时路径，不能当作当前 wty_LTL 验证。原始机器记录保留在 [validation.json](validation.json)、[baseline_validation.json](baseline_validation.json)、[baseline_smoke.json](baseline_smoke.json) 和 [integrity.json](integrity.json)。

# 整理后验证报告

生成时间：2026-10-16T04:13:08.907186-04:00

### 实际结果

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

### 论文阻塞原因

- LaTeX paper.tex：! LaTeX Error: File `icml2025.sty' not found.；! Emergency stop.；!  ==> Fatal error occurred, no output PDF file produced!
- LaTeX paper-new.tex：LaTeX Warning: File `zoneenv_gFblue_yellow.png' not found on input line 316.；! Package pdftex.def Error: File `zoneenv_gFblue_yellow.png' not found: using d；!  ==> Fatal error occurred, no output PDF file produced!

### 运行环境与边界

所有 Python 运行检查在远端 Linux 执行。默认 Python 3.10 的 torch 2.6+cu124 需临时 CUDA 搜索路径，Gym plugin 还需 glfw。此次仅对子进程设置：

- LD_LIBRARY_PATH：来自 /home/dell/anaconda3/envs/ib-edl/lib/python3.10/site-packages/nvidia/*/lib。
- PYTHONPATH：src 加 /home/dell/anaconda3/envs/wty_LTL/lib/python3.10/site-packages。
- MUJOCO_GL=egl。

没有安装新依赖或修改共享环境。该跨环境组合用于本次排障检查，不是推荐安装方案；应后续建立锁定的独立环境。PyG 的旧 torch-scatter/cluster/sparse/spline 扩展存在 ABI 警告并被禁用，部分计算使用回退路径。Letter 还有观测空间范围警告。

整理前 GNN embedding/训练 CLI 已成功，Rabinizer 直接调用因执行位失败，8 步 FlatWorld 在日志阶段因零长度回合除零失败。整理后这两个失败已消除。初始输出保存在 baseline_validation.json 和 baseline_smoke.json。

FlatWorld 与 Letter 各 8 步验证覆盖环境构造、模型前向、rollout、PPO update 和日志；不能证明长训稳定、增强收益、所有环境兼容。ARPS/ADR 无完整回合时可为 NaN。没有真实 checkpoint 评估、Zone 完整运行、多种子性能实验或 CUDA 性能验证。论文完整 BibTeX 和多轮引用检查因前置编译失败没有完成。

### 原始记录

完整命令、退出码、耗时和 stdout/stderr 见 [validation.json](validation.json)。该文件保留环境警告及论文报错，避免只记录“通过”。

### 资源完整性与导航

核对了 1,704 个受保护文件的 SHA-256（已有结果、评估公式、Rabinizer 和 Safety Gymnasium）：没有内容差异。当前导航/状态文档链接没有缺失，shell 脚本 bash -n 检查没有报错。机器记录见 [integrity.json](integrity.json)。历史档案原文链接不计入当前入口断链判定。

## 4. 网站长短终稿整合（历史原文）

对应机器报告为 [paper/review/validation.json](../../paper/review/validation.json)，哈希、原始 diff、source anchors 与旧稿迁移清单仍在 paper/review。本文合并前的原 Markdown 同样保存在历史包中。

# 论文整合验证记录

### 环境与方法

所有项目脚本和编译由 CloudFolder 路由到远端 `/home/dell/workspace/Tianyvc/deep-ltl-paper`，通过已有 conda 环境 `wty_LTL` 执行；Python 为 `/home/dell/anaconda3/envs/wty_LTL/bin/python`。未安装、升级或降级依赖。

最初 `cf run -- conda run -n wty_LTL ...` 因非交互 PATH 不含 conda 返回 `conda: not found`，已报告。之后显式调用 `/home/dell/anaconda3/bin/conda run -n wty_LTL ...`，没有切换到 base 或本地环境。

```powershell
cf run -- /home/dell/anaconda3/bin/conda run -n wty_LTL python .maintenance/paper_build.py
cf run -- /home/dell/anaconda3/bin/conda run -n wty_LTL python .maintenance/paper_build_compat.py
cf run -- /home/dell/anaconda3/bin/conda run -n wty_LTL python .maintenance/paper_integrate.py
```

上列为本轮本机维护脚本，保存在 Git 忽略的 `.maintenance/`，不是承诺随仓库分发的公共工具。持久化审阅依据包含 `long-short.diff`、保护哈希、整合清单和 `validation.json`。最终结构检查由单独维护脚本完成，见机器报告。

### 构建结果必须区分原样与兼容副本

| 场景 | 长版 | 短版 |
|---|---|---|
| 原样副本首轮 pdflatex | exit 0 | exit 0 |
| 原样副本 BibTeX | exit 0 | exit 0 |
| 原样副本后续 pdflatex | exit 1，natbib 模式冲突 | exit 1，natbib 模式冲突 |
| numbers 兼容副本完整 4 步 | 全部 exit 0 | 全部 exit 0 |
| 兼容副本最终 PDF 页数 | 13 | 9 |

4 步为 `pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex`、`bibtex manuscript`、再两次同样的 pdflatex。原样后续错误为 `Package natbib Error: Bibliography not compatible with author-year citations.`。兼容检查唯一源码调整为副本中的 `\usepackage{natbib}` → `\usepackage[numbers]{natbib}`。终稿原件没有被这样修改。

兼容副本最终仍有 `h` 浮动体自动改成 `ht` 的排版警告；本轮不是出版校样审查，没有逐像素确认每页浮动体位置。检查了 pdftotext 可提取文本、图文与表格的源码引用，未替代人工终稿校样。

两张 `zoneenv_*.png.png` 本机可由 graphicx 解析，原样首轮没有因此缺图。不把历史 `paper/` 缺图结论套用到这两套新资源。

### 原稿、代码与结果保护

- 两个终稿文件夹合计 24 文件，先保存 SHA-256，再于整合结束核对完整路径集合及内容；基准为 [protected-sources.sha256](../../paper/review/protected-sources.sha256)。
- 三个旧稿文件迁移前后 SHA-256 相同，没有因归档改写其资源路径；旧稿隔离编译方法见档案 README。
- `src/`、`results/`、`results_infinite/`、`eval_datasets/` 的文件哈希在整合前后核对；没有算法、训练集或结果变更。
- 用户已有 AGENTS.md 修改和 `.vscode/` 内容保持原状；本轮不提交、推送。

### 恢复与持久记录

整合前 `paper/` 和被更新导航的备份在本机维护区 `.maintenance/overleaf-review/before-paper-integration.tar.gz`（Git 忽略）。恢复时应先检查并选择性提取，避免覆盖后续用户修改。旧稿本身已入档，持久迁移关系见 [integration-manifest.json](../../paper/review/integration-manifest.json)，不会只靠维护区找回论文内容。

本轮不运行 RL 训练或性能重现实验，不声称仓库代码已与终稿完全对齐。引用条目的真实来源、原始实验日志、完整满足概率论证及论文内部待核对问题见另外两份审阅文档。
# 2026-10-20 远端运行与论文方法核实

本节是本文件最新执行记录。此前“未运行训练/未修改依赖”等陈述仅描述各自历史维护轮次，不适用于本节。当前结论是：三环境基础训练可短时运行；新增固定公式协议的五配置已能进行参数更新和评估；完整七组论文复现、原始 DeepLTL 对照与 zero-shot/Options 仍未完成。

## 执行范围与保护

通过 CloudFolder 在 `/home/dell/workspace/Tianyvc/deep-ltl-paper` 执行，所有项目 Python 使用 `/home/dell/anaconda3/envs/wty_LTL/bin/python`。基础 Git HEAD 为 `682994f5da8789c5c602925b52dcc6b8c42ecac4`，包含未提交工作，不能仅凭 HEAD 重建本轮代码。没有启动第二 coding agent，没有提交/推送，没有启动百万步长训。既有文档清洗及用户 AGENTS/.vscode 修改未撤销。

受保护集合是网站终稿两个目录、results、results_infinite、eval_datasets；执行前后共 148 文件路径与 SHA-256 一致，见 `validation-runs/20261020/protected-after.json`。其中两个终稿共 24 文件没有修改或写编译产物。本轮不对论文报告数字或旧 CSV 作回填。

## 1. GPU 与依赖排障过程

1. 加载上下级 AGENTS、cf here 确认远端。通过 `cf run -- /home/dell/anaconda3/bin/conda run --no-capture-output -n wty_LTL python ...` 执行；不使用本地 Python 或远端 base。
2. 远端 nvidia-smi 实测 4× RTX4090，每卡 24564 MiB、驱动560.35.03；每卡他人 Python 约16.6GB、Ollama约0.9GB，初始空闲约6.2–6.6GB。仅物理 GPU2 进行了小张量和短训，没有终止他人进程。原始 UUID/PID/占用见 environment.json；这不是资源预留。
3. 最初导入命中 ~/.local 的 torch2.6，缺 CUDA 运行库；禁用 user site 后发现 wty_LTL 自身无完整训练依赖，初次短检失败。已明确向用户说明并获得“仅补齐 wty_LTL”授权。
4. 安装 torch2.5.1+cu124、torch-geometric2.6.1、gymnasium0.28.1、mujoco2.3.3、numpy1.23.5 等。首次 numpy1.26.4 解析与 gymnasium-robotics 冲突失败，随后使用仓库兼容版本成功；torch 所需 sympy 改为1.13.1。未改 ~/.local 或其他 conda 环境。
5. Zone首次失败是导入上游 safety-gymnasium，无 PointLtl2 注册。用本环境 `python -m pip install --no-deps --force-reinstall -e src/envs/zones/safety-gymnasium` 指向仓库定制1.2.0源码，注册和短训恢复。安装过程见 install-local-safety.log，路径证据见 safety-package-source.json。
6. 禁用用户包后 torch位于 wty_LTL，cuda可用、4卡可见，cuda:2 张量与短训成功。依赖实测清单在 pip-freeze.log，安装前 conda清单在 conda-packages.json。**pip check仍 exit1：定制safety-gymnasium要求pygame2.1.0，当前2.6.1；未假称依赖完全一致，未擅自安装系统编译依赖。** 无渲染三环境通过不能证明完整渲染兼容。

训练子进程统一 `PYTHONNOUSERSITE=1 PYTHONPATH=src MUJOCO_GL=egl OMP_NUM_THREADS=2`。上游 Adroit reward版本警告和Letter observation范围警告保留在日志中；未涉及这些Adroit任务，不把警告抹除或等同当前LTL结果失败。

## 2. 修复与实现决策

| 文件 | 已执行变更 | 理由与影响 |
|---|---|---|
| src/model/gnn_net.py | 图内softmax归一化注意力，再scatter sum | 原gate直接相乘不是论文attention pooling；会改变旧模型数值，旧checkpoint结果需重评估 |
| src/model/discrete_epsilon_distribution.py | log_prob先clone动作 | 避免PPO计算污染保存的epsilon动作，独立测试确认输入不变 |
| src/sequence/samplers/curriculum.py | 无完成任务统计/课程已完成时直接返回 | Zone首批无完整任务时原np.min空集合崩溃；不把无数据当成功或失败 |
| src/ltl/automata/ldba.py | 接受SCC去重、无接受返回False、接受sink所有转移均接受才可有限终止 | 原GF blue一个状态、含接受和非接受self-loop仍被判finite；会在首次接受后错误结束。新判据为保守充分条件，非完整co-safe判定算法 |
| src/train/train_ppo.py | 旧增强开关明确fail-fast | 静态检查旧增强默认绕过HV、链式图、forward推进tracker与PPO合同不一致；防止长训产出后误命名完整方法，基础无增强入口仍可用 |
| src/model/paper_hiltl.py | 新真实LDBA图、当前q、不可变历史、GATv2/多尺度池化、双头和固定门控 | 独立显式协议 hiltl-product-v1，保留历史代码以便审查，不把探索类当论文实现 |
| src/train/train_paper.py | 完整回合rollout、TD/GAE/MC、实际融合value PPO、双头辅助loss、梯度/有限性、严格恢复及逐回合评估 | 新实现可短训，但只是固定公式协议，非论文全复现 |
| scripts/long_task.py | 后台启动、身份匹配status/stop、心跳、退出码与日志 | 使用当前wty_LTL解释器，无shell拼接，不保存全环境/凭证；实测成功/失败/停止 |
| tests/test_paper_hiltl.py | 独立手算oracle与真实环境边界检查 | 覆盖图语义/目标/梯度/动作不可变/融合/恢复/有限无限判定 |

### 新协议的明确边界

- 图来自真实LDBA转移，不加人工反边/skip；节点4D、边6D、GATv2三层64维四头，LayerNorm残差与每层图内attention池化。多尺度开关控制3层concat或仅末层。
- q在环境真实step后推进，reset初始化；snapshot复制features，history为不可变tuple；模型forward仅读取，打乱minibatch不重新推进状态。
- 使用transition-based accepting目标节点作为accepting node；SCC bit解释为cyclic SCC；不可达distance映射1、零分母保护。这些是显式操作定义，不声称解决原文全部歧义。
- 沿无权最短距离的真实一步转移无法减少超过1，因此不虚造skip边；四类one-hot保留第2槽但当前不可激活。原稿的skip规则需要重定义或删改，不能静默伪造边。
- β采用短版固定sigmoid式κ3/θ.5；不是另一处MLP门控。长期头不接环境特征，用current/proximity与可选历史，监督为完整有限时域MC折扣回报，**不是[0,1]满足概率**。
- short/long头hidden256/128，short辅助loss1、long .3、fusedvalue .5；PPOclip .2、entropy .01、Adam3e-4、γ.998默认；完整回合GAE .95。历史对actor的接入、无Track仍观察q、统一MLP环境encoder均属于当前工程协议，未证明与原始消融/LetterCNN完全等价。
- horizon当有限时域任务定义，zero bootstrap；无限性质仅评估这个horizon内接受访问代理。每批完整回合使实际steps可超过目标，不可只按CLI预算比较。
- history LSTM重新编码全历史，尚未做性能profiling或截断优化；无并行产品环境训练。本轮短horizon16不证明500/1000回合性能。
- 提供五种配置，不提供baseline/full_all；固定公式不能直接改公式恢复评估，防止把不存在的迁移协议标成zero-shot。

## 3. 实际检查与结果

所有命令argv和退出码均保存在机器JSON及同名日志，不依赖Git忽略的维护脚本。测试脚本是正式仓库入口，短训产物在Git忽略的 `experiments/runtime_validation_20261020/`。

| 检查 | 实测结果 | 证据 |
|---|---|---|
| 原GNN embedding / fix 两脚本、train_ppo help | exit0 | isolated-environment.json |
| 原Flat、Letter CPU各64steps，两次PPO更新 | exit0；未完成回合时ARPS/ADR为NaN | original-smoke.json与两log |
| 原Zone初次 | exit1，课程np.min空集合 | extended-checks.json、zone_original.log |
| 原Zone修复后64steps两更新 | exit0，loss/gradient有限 | regressions.json、zone_original_fixed.log |
| 新Flat/Letter CPU，Flat GPU2，各32steps两更新 | exit0，双头/GNN/history/actor梯度非零 | paper-smoke.json，注意其中旧Zone失败仍如实保留 |
| 新Zone定制包修复后32steps两更新 | exit0 | extended-checks.json、zone_local.log |
| 另外四配置各两更新 | exit0；完整回合导致实际32/36/37/47steps | extended-checks.json与配置日志 |
| 从Flat32steps恢复至48steps/update3 | exit0，严格加载参数与optimizer | resume.log |
| finite任务3回合评估 | exit0，每回合seed/指标/checkpoint hash | evaluation.log及运行产物 |
| 不匹配公式恢复/已有输出重建 | 按预期拒绝exit1，无静默覆盖 | reject_mismatched_formula.log、reject_existing_output.log |
| 旧增强开关防误用 | 按预期NotImplementedError | legacy_guard_fixed.log；首轮少传必填num_steps，parser exit2，不算guard已通过 |
| 独立语义与回归测试 | 最终结果见final-semantic-regressions.json，新增GF判定与horizon验证 | finite_semantic_final.log |
| 无限任务 | 原先运行暴露finite=True错误；修复后以新output重训重评估，旧证据未覆盖 | infinite_fixed_train/evaluation.log |
| 后台工具 | 正常exit0、失败exit7、SIGTERM中断、状态身份查询均通过 | background-manager-tests.json |
| 源码语法 | 首轮136个非第三方Python文件AST通过，最终复核另存 | syntax.json及final-audit.json |
| 保护哈希 | 148文件集合与内容一致 | protected-after.json |
| git diff --check | exit0；不表示全部工作区改动属于本轮 | git-diff-check.log |

这里每项只是相应范围的通过，不能写“全项目所有方法/实验均验证通过”。legacy占位脚本、所有渲染/地图/epsilon轨迹、多并行、全部评估组合、长训稳定性、统计显著性与论文所有数据尚未验证。GPU只测试Flat小模型；三环境×五配置×CPU/GPU的完整笛卡尔组合未跑。

## 4. 论文公式与当前转换结果

使用仓库Rabinizer调用和现有LDBAWrapper FlatWorld命题剪枝/补sink，不人为固定状态数。具体表见 formula-automata.json：φ1/φ2/φ3原始分别4/8/4，Flat剪枝补全后4/8/5，原稿分别写5/12/18。剪枝的合法命题组合、补sink和converter版本都会影响计数，但目前没有证据解释稿件数字；不能用这三个数字当实现断言。G F blue原始1state并不代表finite，其接受与非接受self-loop已用于修复判定。

## 5. 后续长任务与执行顺序

见 [LONG_RUNNING_TASKS.md](../LONG_RUNNING_TASKS.md)，包含GPU检查、10k pilot、1M训练、500回合评估、状态查询、停止、严格续训、输出和异常判断。当前未启动任何长期任务。

优先级：先冻结论文冲突与基线定义 → 验证长horizon学习信号/资源/吞吐 → 冻结环境与seed/地图/任务协议 → 固定公式多种子对比 → 实现和验证universal/zero-shot及Options → 七配置正式实验。对原始论文的完整复现结论仍然开放，不能以短训通过替代。


### 本轮收尾结果

- `final-semantic-regressions.json` 全部5项按预期通过：12项独立语义检查、修复后无限任务短训与3回合评估、公式转换审计、8色大地图主入口短训。G F blue现在finite=False。
- 大地图启动器 `run_flatworld_big.py` 不接受训练预算等透传参数，原尝试parser exit2，未启动其默认15M长训；改用 `train_ppo.py --env FlatWorld-big-8-v0 --model_config FlatWorld-v0 --curriculum FlatWorld-big-v0 ... --num_steps 32` 完成两更新。默认启动器属于长任务，不能直接套用短检参数。
- 维护脚本首次动态compile读到UTF-8 BOM失败，改用utf-8-sig读取后重新执行；这是维护脚本编码问题，未把失败计为训练通过。
- `supervised-training-stop.json` 实测真实训练器后台stop：配置目标1M仅用于有界中断验证，实际运行32steps/2updates即停止，supervisor interrupted、child exit0、progress interrupted且status.pt存在；并未开展百万步实验。所有本轮后台测试均已结束。
- `final-audit.json` 保存136个非第三方Python源码哈希，AST错误0、git diff --check exit0、148个保护文件完全一致。小型config/progress/metrics/episodes/evaluation JSON已从experiments复制到证据目录，检查点仍保留在experiments，不把Git忽略目录当唯一结果记录。
- `pip-check.log` exit1是已知pygame精确版本冲突，不能因runner将它设为预期退出码就写成依赖检查通过。首次extended-checks中的Zone与guard命令失败保留，修复后证据分别在regressions与final-semantic-regressions，不抹除失败历史。


# 2026-10-20：补齐实现、最终验收与发布准备（hiltl-product-v2）

本节更新前述 v1 的历史限制。用户要求未落地方法先实现再提交；另确认 LTL2Action / GCRL-LTL 对照数值引用优化前原论文，本阶段不接官方实现。引用数据保留原样，不将其称为本仓库重跑或与新运行同协议的实验。

## 实际落地

- 七配置全部接统一乘积训练：baseline、hv_only、msgnn_only、hv_track、msgnn_track、full_no_options、full_all。baseline 用原 DeepSets/GRU、固定排序命题词表和 value-guided reach/avoid 序列，PPO保存采样时选定序列；不含 GNN。原课程入口增加 architecture=deepltl，严格加载并拒绝架构错配。参考架构与原发布版本逐位复现是两回事。
- 语义图保留真实 LDBA，节点4D/边6D、三层64维四头GATv2及多尺度池化；新增 guard GRU、词元均值旁路及进展边消息值。HV、历史、fixed/learned/constant门控均进入真实采样/优化。
- Options 不再是只有类和开关：按真实候选边选目标，目标条件低层策略、guard命中/epsilon/状态改变/超时终止、势差内在奖励、option critic、高层 SMDP PPO 均有实际更新。高层使用 gamma^duration 与 lambda^duration；逐回合保存区间、持续时间和终止原因，逐更新保存meta loss/梯度。旧未反向传播的骨架原字节归档至 docs/archive/legacy-option-skeleton.zip。
- 三环境任务清单严格分 train/held_out，训练只采样train。文本和编译后 guard/accept 图指纹拒绝重叠，修复状态编号敏感问题；WL结构摘要不构成完整语言等价证明。held-out评估不更新模型，输出逐回合、任务/模型哈希、评估实际训练步数。
- checkpoint验证实现源码SHA、门控、Options horizon、任务清单与训练参数；不允许静默缺键或改方法后复用旧实验。恢复新回合，不承诺环境轨迹逐位连续。统计先种子内平均，再以独立训练种子计算std/t区间及配对SR绝对差；n=1区间为null。

## 失败、修复与证据范围

独立 guard 可辨性检查先暴露：只把guard用于注意力时归一化会抵消区别；加入消息值后长DNF的单一GRU又遗忘部分命题。最终加入词元均值旁路，并以同构不同命题图验证可辨。任务隔离测试发现状态编号影响摘要，改用带标签转移节点的WL指纹后，F green & F blue 与 F blue & F green 的重叠被拒绝。这些失败驱动实现修复，不算作成功实验。

早期 implementation/matrix.json 在最终guard/源码哈希冻结前运行，保留为过程记录；正式短时验收以 final-matrix.json 为准。它调用真实parser/main，在同一远端wty_LTL解释器中执行34项，34项按预期通过：七配置各训练/held-out/恢复21项，full_all train评估1项，3项误用拒绝，Letter/Zone full_all训练和held-out共4项，learned/constant门控2项，第二训练种子训练/评估2项，CUDA2 full_all短训1项。训练预算16步、恢复24步、horizon8，仅作工程验收；自动机成功可能提前结束，实际步数见产物。

最终独立子进程重新运行 test_paper_hiltl.py（12项）和 test_paper_protocol.py（8项），均exit0。协议检查覆盖统计手算、SMDP持续时间折扣、Options终止、可变候选/pure batch、guard可辨、编译等价交集拒绝、实际高低层参数更新与基线序列可重放。原 test_gnn_embedding_quick.py / test_gnn_fix.py 均exit0。baseline/full_all另用新进程严格恢复到32步，验证跨进程词表和checkpoint兼容。原课程参考架构的训练/恢复/评估/错架构拒绝证据在早矩阵中，后续变化不涉及该路径。

统计工具真实读取七配置评估输出和第二训练种子，单种子与双种子汇总exit0；重复训练种子、缺少种子、错误回合数均按预期exit1。统计产物仅是2回合/horizon8的小样本验收，不是论文性能或正式CI证据。评估记录的checkpoint_steps保持16，即使随后恢复训练，也不误用新progress冒充旧模型预算。

新 full_all 经后台工具启动，在产生checkpoint后主动停止：监督进程interrupted且已结束、status.pt存在，实际步数见下列机器记录；命令中的1M只为有界停止验收，未执行完整长训。启动/状态/stop精确argv和输出保存在 background-v2.json。没有留下本轮后台训练。

141个非第三方Python文件AST无错误；两个终稿、旧结果与eval_datasets共148文件集合和SHA完全一致。小型config/progress/metrics/episodes/evaluation产物已复制到 implementation/artifacts，checkpoint仍留在忽略的experiments。待发布内容未发现所检查的GitHub凭据/私钥模式，最大单文件为已跟踪的racing_track.obj（57,529,789字节），无达到GitHub100MB限制的文件。

## 尚未完成及下一步

正式1M×五种子、每种子每公式500回合、长horizon学习有效性/稳定性、外部方法本仓库重跑与完整语义等价去重均未完成。pygame2.6.1与定制包精确2.1.0约束冲突仍在；无渲染短训不能替代完整渲染兼容验证。原稿skip/MC概率解释/SCC/门控歧义和自动机状态数差异没有通过修改网站终稿或数据解决，操作定义见README和验证路线。先pilot测吞吐/显存/奖励与梯度，再冻结正式协议；执行方式见LONG_RUNNING_TASKS.md。

README、PROJECT_STATUS、CODE_VALIDATION_ROADMAP与LONG_RUNNING_TASKS已同步，不新增平行摘要。远端Git提交使用 tianyvc <1038095778@qq.com> 的单次命令作者配置，不改服务器全局作者。个人电脑此前已获用户授权登录GitHub；通过增量bundle传送提交，再由个人电脑HTTPS推送，不在共享服务器写入GitHub凭据。发布结果以远端分支SHA核对为准。

后台有界验收最终progress（真实记录）：

```json
{
  "state": "interrupted",
  "steps": 16,
  "updates": 2,
  "target_steps": 1000000,
  "pid": 2382244
}
```

发布暂存检查补充：完整 `git diff --cached --check` 报exit2，全部来自首次纳入Git的网站终稿原有空白及逐字差异文件 paper/review/long-short.diff。遵守终稿不改和原始差异保真，不清理这些空白；排除上述3份保真文件后的暂存格式检查通过。完整输出另存 release-full-whitespace-check.json，不能把全范围exit2写成exit0。本机.vscode设置保留在工作树且不提交。


# 2026-10-21：论文投稿成熟度审查与正式重跑准备

用户要求先按较高通用会议标准审查并反馈论文缺口，完善后再按GPU资源并行后台重跑。本次重新全文审阅长短稿与v2关键路径，提取仓库内原始DeepLTL PDF；审查详见paper/review/FINAL_VERSION_COMPARISON.md第6节，正式重跑门槛同步到CODE_VALIDATION_ROADMAP.md。

确认GCRL-LTL引用错配、DeepLTL架构定位错误、原论文15M与新稿1M预算及任务不一致；q-only概率/MC混用、skip不可能、静态attention的进度主张、无限SR、baseline编码器混杂等需要先修订。现存结果不改、不判其产生方式；外部引用按用户决定不接官方实现，但不得写成当前同协议重跑。

所有远端检查使用cf run → /home/dell/anaconda3/bin/conda run -n wty_LTL，PYTHONNOUSERSITE=1；未安装依赖。GPU快照4×4090空闲5372/5844/5922/5518MiB，已有其他任务；不结束他人进程，不作资源预留。资源、精确命令、环境、引用键检查、网站快照哈希及原文锚点位于validation-runs/20261021。两稿无缺失交叉引用label/citation key；没有重新编译或完成最新文献联网核验。

本次只更新已有审查/验证/执行记录和机器证据，不修改源码、终稿、旧结果，不提交推送；正式训练/批量后台队列尚未启动。等待论文与实验合同完善后，先pilot定资源/吞吐，再真正启动并提供批次、任务ID和监视入口，避免把未冻结协议的运行冒充论文实验。


## 2026-10-21：用户授权Copy短稿修订（八点落实）

本次用户明确要求在Copy短版修改；此前“两个网站快照均不改”的要求仅在本次授权范围内被更新，长版仍未改。修改对象为overleaf/TLT__Copy_/manuscript.tex与ref.bib，原文件在paper/archive/pre-revision/short-manuscript-20261021.zip以原字节保留，含12资源SHA清单；其余10个资源未删未改。

短稿围绕dual-target critic重写：TD/MC及融合为近似架构，长期头是有限时域回报代理；完整写明guard GRU/mean/message values、距离/SCC/接受转移、epsilon、PPO、Options内在奖励和SMDP。补conditional fusion error bound并明确不证明收益；移除无效skip、精确概率分解和“静态attention随q改变”等主张。Outer TimeLimit计epsilon决策，已从env_utils包装顺序核对后写入正文。

改正DeepLTL/外部方法定位，Qiu2023和XuFekri2024依据本地原论文参考文献补入，Brody改为ICLR2022；85条旧文献收敛为实际引用14条，原文献库归档。没有完成最新文献完整联网核验。删去13张旧性能/重复表与相应提升、CI、难度趋势及旧图引用；2张新表仅记录诊断任务与工程checkcase，1张原生LaTeX图描述依赖，未编造性能结果。

正式同编码器单critic/history-only等对照、独立validation、扩大公式集及正式预算尚未执行，论文按待完成计划表述；短稿修订不能认定研究证据和顶会录用条件已经满足。代码及旧结果本次未改，无长训、无Git提交推送。README/论文入口/现状/路线与既有审查文档同步，不新增平行MD。

实际构建环境：cf run → /home/dell/anaconda3/bin/conda run -n wty_LTL env PYTHONNOUSERSITE=1 python维护脚本；LaTeX在.maintenance/short-revision-20261021-build隔离目录运行。pdflatex → bibtex → pdflatex → pdflatex均exit0，pdfinfo/pdftotext均exit0；6页（含参考文献），最终无缺失引用/标签/Overfull，有少量Underfull提示。初次构建引用未定义提示属于BibTeX及多轮编译前的正常中间状态，最终log无此提示。6页逐页渲染并检查整体排版，后续少量符号修订又完整重编译。

证据与PDF位于validation-runs/20261021/short-revision；998个长版/源码/历史结果等文件哈希一致，短版仅tex/bib变化，归档字节核对通过。首次维护命令过长未启动，改为分段准备后应用；没有把失败命令计入编译通过。

最终格式检查发现短稿EOF多余空行，清理后重新完整编译和复核；该格式失败未计为通过，未影响算法或论文内容。


## 2026-10-21：论文实验重跑计划、v3及真实后台批次

用户要求详细计划并按SSH资源后台开跑。本轮所有计算经cf run→/home/dell/anaconda3/bin/conda run -n wty_LTL，禁用户级包；cf here成功，读取上下级AGENTS。没有安装依赖、提交推送、改长稿或旧结果，没有启动第二个agent。初始本地沙箱CreateProcessWithLogonW 267无法启动，使用已获工具批准的沙箱外PowerShell访问挂载目录及cf后可运行。

先检查4×RTX4090占用和CPU/mem/disk；每卡已有他人任务，CPU load约61。仅启用物理卡2/1各一个worker，启动准入空闲4096MiB，torch/OMP2线程；不停止他人进程。资源原始证据resource-audit.json及launch-gpu.txt归档。

v3必要改动：同图encoder的gnn_single/gnn_track，容量匹配gnn_capacity/gnn_track_capacity（critic宽290/277，参数差<1%）；单critic+history输入history，包含msgnn_track。新增validation split并兼容旧train/held_out JSON，训练仅采样train；每环境固定10/3/9任务，含6/9/12访问长顺序，实际最大13状态。原始/剪枝图指纹跨split检查、环境reset/编译通过，未声称语言等价判定或所有地图物理可实现性证明。每100k实际更新存checkpoint，增加参数量/CUDA峰值记录，扩充指定配对基线统计工具。

原guard协议测试首次失败：剪枝后label由set枚举，GRU输入在跨进程顺序变化，且冗长析取稀释命题差异。修复为符号布尔规范化后tokenize，不放宽原测试。修复前日志在pre-canonical-guard，修复后12+8语义/协议、5项公平/容量/validation检查通过，PYTHONHASHSEED=1/2原协议再次通过。后台统一HASHSEED=0，PYTHONPATH显式指向冻结src和定制Safety-Gymnasium。15项实际新配置训练/validation/跨进程resume与checkpoint保存通过。学习曲线实际评估成功，初始维护断言错误假设精确8/16/24步；完整episode超预算，改以torch checkpoint实际steps核对，不改训练/评估实现，保留原失败日志。

新增paper_campaign.py：有界worker、显存准入、依赖、flock、原子状态/心跳、逐任务日志、产物验收、暂停、重启不重复及显式retry。调度8项成功/失败/等待/暂停/恢复/互斥/重复ID拒绝检查通过。analyze_paper_campaign统计oracle5项通过，预先固定种子级宏均值/配对差/HV×history交互，有限SR与循环访问次数分开；不混历史引用数据。checkpoint curve仅用validation，不以test选checkpoint。

批次experiments/reproduction/20261021-v3：195×1M训练，9个pilot，585项final评估，195项曲线，9项summary，共993任务。源码1744文件约119MB冻结，记录未提交工作区、git基准3a96a70、diff、依赖和每文件SHA/任务HOA/逐项argv。模型/结果不自动上Git，需独立备份。完整计划整合现有LONG_RUNNING_TASKS.md，未新增平行说明MD。

初始后台scheduler-01成功启动，9pilot均完成（1560–2008实际步，2–3update）。启动核查发现训练与train评估ID相撞；当时两项正式训练正在运行、评估尚未运行。立即pause，明确对本批次两worker保存停止；约21.6万/19.9万步checkpoint保留，原attempt日志保留。补重复ID拒绝回归，完全暂停后旧manifest/state/控制源码归档control-revisions/01-id-collision，修正eval-train ID和依赖，993唯一且DAG无环。仅控制修订02，训练/模型/任务源码未变；更新控制源码哈希与provenance，不伪装成初始从未改动。

scheduler-02后台恢复成功（监督PID1222943，调度PID1222945），旧9pilot不重跑。UTC05:49（北京时间13:49）真实核查：9completed/2running/982pending；gnn_single物理卡2实际218880步，hv_only物理卡1实际201952步，均从原checkpoint加载、严格实现哈希通过、新update/loss有限，CUDA进程约520MiB。两份job.json含--resume及source cwd，原中断error历史字段保留，当前state=running。真实证据running-check.json、runtime-audit.json、launch-02.json。

pilot稳态：baseline约79–85步/s；hv_track约15–36；full_all约8.6–24，条件估计单次百万步约3–32小时，不含评估和共享负载变化。全批次约142万评估回合，可能数周或更久，未承诺确定ETA。pilot零成功也保留，不以pilot结果挑seed/test或承诺最终收益。

尚未完成：全部正式预算、多种子性能、held-out/学习曲线统计、环境可实现性诊断、最终论文结果。summary训练耗时当前为最后进程elapsed，resume运行累计成本需按attempt另核算。原稿外部对照仍为引用数据，不接官方实现、不写同协议SOTA。所有监控/暂停/恢复/完成判断/归档文件解释已给于统一手册。格式检查曾发现两处CR/行尾空白，清理后重查。


## 2026-10-21：后台运行巡检、总耗时估算与本地只读监视

用户要求检查是否正常、整批ETA和最简单本地监视方法。本轮重新加载上下级AGENTS并cf here；远端wty_LTL只读检查确认scheduler-02存活、心跳更新，9completed/2running/982pending，无当前failed/blocked。新增scripts/check_paper_campaign.py（主工作区工具，不加入或修改已冻结source），10秒采样训练steps/update、PID身份、CPU时间、指标新鲜度/有限性、近期吞吐、剩余训练时间；两次实际执行成功并保存输出。不启动/停止/重试任何队列任务。

北京时间14:46两项373902/351202步，10秒增加553/627步；14:50再次查询，一项10秒未完成update但CPU仍工作、metrics_age13秒，另一项增加615步，不能将短窗口无更新当死锁。之后累计达到396907/374960步，近期吞吐49–54步/s。部分loss/裁剪前梯度明显波动（hv_only近100更新loss峰值约4767、gnn_single约127），当前有限且继续更新，不隐瞒为稳定收敛；运行健康与研究有效性分开，后续仍需正式评估。

为补评估成本，在GPU0检查显存headroom后，顺序对已完成pilot的baseline/hv_track/full_all及三环境，各取最长顺序与GF公式一回合，共18回合只读推理计时；冻结source/cwd导入，strict加载pilot，不读活跃checkpoint、不更改任何训练数据。计时全完成无后台遗留，不是正式性能证据。以195次训练、每模型7300评估回合、两个连续worker粗外推：训练2351 worker-hours，评估2722，总约106天；有利/较慢情景约68/182天。11个非reference非Options配置以hv_track近似，只有两个特选公式和一个reset seed，故不是统计置信区间/可靠截止承诺。推荐3–4个月中心估计、2–6个月规划，正式评估出现后再校准。共享资源/停机/存活时长/图大小会改变结果。

LONG_RUNNING_TASKS第13节补一条本地cf run监视命令、每字段解释、OK/CHECK_REQUIRED/WAIT/ALL_COMPLETED及真正全部完成条件。巡检证据monitor-20261021包含健康输出、状态快照、推理时间、ETA假设、loss波动。没有改训练方法、预算、并发、manifest或冻结哈希，没有依赖变更、提交推送。


## 2026-10-21：按数天预算收敛为 ZoneEnv 主环境精简实证并后台启动

用户拒绝数月训练预算，要求小而精、对齐草稿、ZoneEnv为主，Flat/Letter只检验扩展。本轮没有提交推送、安装依赖或修改网站长稿/历史结果；所有计算通过cf run及远端wty_LTL，禁用户级包，无第二个agent。

先暂停原195模型大队列，对本项目两个训练worker保存停止，保留所有attempt与最终约43万/41万步模型。按预定最早>=200k规则提取Flat gnn_single step-200532与hv_only step-200263，独立复制并strict CPU resume校验（零新增更新），不按评估收益挑checkpoint。原大批次保留PAUSE，不再自动恢复。

中途31模型/63阶段合同20261021-minimal-v3在用户进一步明确Zone为主之前已启动两项Flat评估；收到补充后立即pause停止新派发，两项短评估自然结束。原31队列最终paused，4completed/59pending，无新正式训练派发。旧原控制及数据保留，并写SUPERSEDED_BY。此前最小评估器真实gnn_single/baseline/full_all冻结推理、拒覆、过期deadline拒派发、31矩阵统计/负效应/单种子无CI共7项验收通过；初次测试fixture相对task_set路径失败保留日志，换实际绝对路径严格检查点通过，不计失败为成功。

读取本地原DeepLTL全文提取（.maintenance/deepltl-original-review.txt），核对附录F.2表2/3、F.3/F.4及G.3：原Zone若干任务使用magenta，当前训练10公式不覆盖它，不能以此预设未训语义泛化。原文使用15M课程及不同超参数，本轮参考任务定义，不称官方复刻或混比较旧SR。

最终合同20261021-zone-minimal-v3：Zone四因子gnn_single/hv_only/gnn_track/hv_track各3seed，加baseline3seed=15；Zone msgnn_only/msgnn_track/full_no_options/full_all各1seed=4；Flat四因子各1seed=4；Letter gnn_track/hv_track各1seed=2。共25模型×200k，23新训/约460万步；含复用两训练的总预算约500万。51阶段=25train+25eval+1summary。主环境核心优先，GPU2/1各1worker，>=4096MiB显存准入、OMP/torch2线程。

任务：三环境草稿phi1/phi2/phi3及GF循环各50回合；Zone额外三段顺序reachability与原DeepLTL表2 phi7原式各50回合，共6900回合。phi1在train内，其他任务按文本/原始及剪枝图标seen；有限任务主表只聚合草稿三项，补充任务与循环分别报告，不让改变任务权重影响宏平均。不预设正结果，单种子不输出训练CI，不用有限循环轨迹估计严格无限满足概率。取消百万步/五种子、完整门控/容量、长公式压力/密集曲线与外部官方接入，因而不声称排除了容量影响、全组件稳定收益或SOTA。

新增tests/test_paper_zone_summary.py：25模型合成聚合（合成fixture不是科研数据）、Zone三种子配对负效应保留、单种子无CI及CSV/逐任务比较通过。泛化summarize_paper_minimal.py，以manifest.main_environment选择主环境，并以primary_task_ids固定主表公式；添加独立逐任务配对统计。evaluate_paper_minimal.py仅将回合数备注改为从spec读取，算法五源与大批次逐字节一致，训练实现哈希不变。

真实Zone六任务冻结推理（旧hv_track pilot，每任务1回合，不作性能证据）通过：phi1实际4states/8edges、phi2 8/20、phi3 5/11、GF 2/4、三段顺序4/7、原phi7 6/12；仅phi1标已见，循环finite=False，其余finite=True。冻结51项唯一ID、全部源码/spec SHA检查通过，git diff --check exit0。既有controller8项及方法检查本次无需无关重复；没有更改它们的训练五源。

后台启动时间UTC 2026-10-21T07:50:33.993910+00:00；监督PID 1589541，调度PID 1589542。两旧批次确认paused且调度PID不存活后才启动；两份已完成Flat评估经spec/checkpoint SHA和逐回合产物验收复用，4阶段直接完成。启动后Zone hv_track seed1在物理GPU2、gnn_track seed1在GPU1，初检4completed/2running/45pending，failed/blocked=0，两个child及supervisor存活、CPU活动/心跳正常。初检尚无completed-update指标，仅计启动健康，后续正式更新见runtime/health证据，不把进程存在冒称收敛。

统一LONG_RUNNING_TASKS、README、PROJECT_STATUS、CODE_VALIDATION_ROADMAP；不新增平行计划MD。计划数天、约2–5天并预留一周，但整批ETA未由新正式吞吐验证；五天截止2026-10-26T07:41:38 UTC（北京时间15:41:38），停止新派发、让当前任务结束，未完成项必须披露，不自动扩大预算。ACTIVE.json切换最终合同，本地监视命令不变：cf run -- /home/dell/anaconda3/bin/conda run --no-capture-output -n wty_LTL env PYTHONNOUSERSITE=1 python scripts/check_paper_campaign.py。全部完成必须51completed、其余0、ALL_COMPLETED且summary JSON/CSV齐备。

机器证据：validation-runs/20261021/zone-minimal/的manifest/commands/imports、summary-acceptance.json、acceptance.json、zone-task-smoke、launch.json、initial-health.txt与runtime-check.json。正式科研结论须等真实结果/区间和失败项审查后形成，未声称已经满足投稿实证。

归档脚本复用时因已完成的文档替换片段不存在而退出（ValueError）；这是维护脚本重复运行缺陷，不是训练任务失败。已将一次性文档替换从只读复核脚本移除，保留closeout-initial-nonidempotent.log，再执行最终归档检查。

启动后真实PPO进度复核（UTC 2026-10-21T08:02:56.636199+00:00）：[{"run": "zones-hv_track-s1", "steps": 15640, "updates": 16, "loss": -0.010685997156542726, "gradient_norm": 0.5056424166541547}, {"run": "zones-gnn_track-s1", "steps": 15372, "updates": 15, "loss": -0.11004203205811791, "gradient_norm": 0.515424182638526}]；两项训练均已有多次更新、损失/梯度有限、failed=0。此前无update属于启动阶段，并非本轮最终进度。15:57巡检近期吞吐27–29产品步/s，单项训练剩余约2小时（不含评估/资源波动）；整批预计仍按数天规划，不把单项训练时间当完整实验ETA。runtime-check.json/closeout.log保存最终快照。


## 2026-10-21：PowerShell cf分隔符报错修复与监视字段说明

用户在PowerShell执行cf run --报Expected --。Get-Command cf -All确认cf.ps1优先于cf.exe，读取包装器后明确其依赖args中的字面--，当前PowerShell调用未保留此分隔符。显式调用CloudFolder cf.exe的原命令成功；没有修改Program Files安装、profile、SSH或训练队列。新增scripts/check-training.ps1简短入口，显式native exe并固定远端wty_LTL/PYTHONNOUSERSITE，支持SampleSeconds且透传远端退出码；本机真实执行成功。

主工作区check_paper_campaign.py新增整批训练步数/剩余步数（每模型按预算封顶、含复用检查点）、minimal评估回合/剩余回合、北京时间派发截止与预算剩余小时、整批ETA未校准说明；评估有基于实际回合吞吐的粗剩余时间，完成时验证summary及本批次CSV存在。不修改冻结source、manifest或训练逻辑，不暂停/重启模型。

16:21实际入口验证正常：4completed/2running/45pending，2/25模型完成；训练494759/5000000，评估400/6900。hv_track44640步、45update，gnn_track50119步、50update，10秒均增1000步，近期25.70/30.17步每秒，单项剩余训练1.68/1.38小时。loss有波动但有限，不能据此称已稳定收敛。整批仍按数天规划，不把两项当前ETA当全部训练评估ETA。

更新现有LONG_RUNNING_TASKS的命令及逐字段说明，README加最简入口；不新增平行MD。实际远端wty_LTL监视校验51任务/5M预算/6900回合通过、git diff --check通过。证据validation-runs/20261021/monitor-cli-fix/health.txt与diff-check.txt。无安装依赖、提交推送。


## 2026-10-23：精简重跑完成、结果审计与工作区整合

51/51阶段完成，25模型，6900评估回合。audit_paper_results.py在远端wty_LTL通过：冻结源码、checkpoint/spec哈希、模型/优化器张量、逐回合九指标与JSON/CSV重生成。实际检查点步数5,011,670，封顶预算5,000,000；两份导入Flat模型原始配置和差异明确记录。主环境未证明方法提升，详见EXPERIMENT_RESULTS.md；保留全部负结果，未修改论文源文件。

README、PROJECT_STATUS、CODE_VALIDATION_ROADMAP和LONG_RUNNING_TASKS整合为当前事实；paper导航同步阶段说明，监视脚本完成状态ETA改为0h。轻量证据在results/reproduction/20261021-zone-minimal-v3，完整模型/快照只在服务器experiments，未上传大二进制。

本轮删除一次性维护脚本、重复bundle和可重建构建缓存，详细删除清单及SHA见cleanup-20261023.json。受保护overleaf全部文件及完整完成批次文件逐字节复核不变；旧大队列源码作为导入模型依赖保留。未启动新训练。

本轮补充回归：tests/test_paper_v3.py 5项、test_paper_campaign.py 8项、test_paper_campaign_analysis.py 5项、test_paper_zone_summary.py 3项均通过（共21项）。完整stdout/stderr归档到轻量证据verification-tests.json。

补充真实推理回归tests/test_paper_minimal.py通过7项（3种模型冻结权重/拒绝覆盖、派发截止、31模型统计oracle及负结果/单种子边界），累计本轮28项检查通过。文档中的旧大队列manifest、展开命令清单及中间状态副本已删除，原始服务器experiments记录保留；本轮manifest只在轻量证据发布一份。历史条目提及的这些已退役副本路径不再作当前操作入口。

最终核对：共删除204个冗余文件、323,486,033字节；2,676个Overleaf/完成批次文件SHA不变。研究证据配置.gitattributes -text防止跨平台自动换行破坏哈希。git diff --check对原始nvidia-smi空格输出保留例外，其余源码/文档通过（允许CRLF）。新增暂存文件无常见GitHub凭据/私钥模式且均低于50MiB；.vscode不提交。GitHub推送采用个人Windows Git/GCM HTTPS，服务器不接收个人凭据。
