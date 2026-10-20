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
