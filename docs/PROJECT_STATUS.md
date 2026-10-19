# 项目现状详解

> 本文依据此次工作区静态阅读、文件盘点和实际命令检查编写。它记录当前可观察实现，而不是论文理想方案。整理前后命令原始输出见 maintenance；文档历史中的“完成”和性能声明不能替代实测。检查日期以 validation.json 及文件记录为准。

## 1. 项目定位与总体判断

这是一个围绕 LTL 约束强化学习的研究工作区，同时承载四种资产：DeepLTL 基础实现、本地增强模块、实验数据/工具、论文演进材料。整理前它们混合在根目录及 docs 中，容易将提案、代码和结果混为一谈。

原 DeepLTL 通过自动机规划得到 reach/avoid 目标序列，使用 DeepSets 与循环网络学习通用序列条件策略。本项目增加 GNN，进一步探索 Semantic-Enhanced GNN、长短期价值融合、多尺度表示、自动机追踪和 Options。HiLTL 与 DeepLTL-LS 是不同阶段稿件中出现的方法名称，尚需作者统一最终命名。

目前基础训练链路存在、模型可构建，GNN 实际进入主模型嵌入。LongShort 模块并非全部进入主训练计算和损失。已有结果主要是历史基线表，当前材料没有形成完整增强方法的可复现证据闭环。论文候选稿已较完整，但完整投稿就绪尚不能确认。

## 2. 整理范围与资产保护

整理前原目录约 140 MB，src 约 104 MB，其中大量为第三方 Safety Gymnasium assets；Rabinizer 约 15 MB。它们不是无用体积，应保留许可证、模型与运行库。仅删除系统元数据、过期构建、空文件和失效入口等明确杂项。

全部不同论文版本保留；技术/实验/论文修改文档无损汇编。已有 CSV 不重写、不重新估计数据。原 parallel.log 是训练证据，归档为 results/logs/flatworld_gnn_test_seed5.log。原始 DeepLTL PDF 不删除。

整理前没有 Git 提交且全部文件未跟踪，故先创建逐文件校验清单和压缩备份。具体移动/删除见 maintenance/cleanup-manifest.json，恢复方法见 maintenance/CLEANUP.md。

## 3. 主流程与模块关系

训练主链：

```text
train_ppo.py
  → ExperimentConfig / PPOConfig / model_configs
  → curriculum + CurriculumSampler
  → make_env(sequence=True) + SequenceWrapper
  → vocabulary + preprocess_obss
  → EnvNet + LTLNet + sequence GNN
  → actor distribution / critic value
  → PPO collect_experiences + GAE + minibatch update
  → logger / ModelStore / checkpoint
```

评估链按指定 LTL 公式构造 LDBA，搜索 reach/avoid 序列，以策略执行并记录成功、违反、回报或 accepting visits。训练的课程序列采样与评估的完整公式转换不能简单视为相同调用路径。

### 3.1 配置和启动器

正式 Python 配置是 src/config/ 包，包括 ExperimentConfig、PPOConfig、ModelConfig、SetNetConfig 和 model_configs。原 src/config.py 被同名包遮蔽，已归档；根 config 下的 Python/YAML 为探索草图，不被训练主程序加载。

正式模型配置包含 zones、letter、flatworld 等场景。当前 GNN hidden=64、3 层、4 heads、dropout=0.1，边特征 6 维和节点状态特征 4 维均启用。历史“默认关闭增强特征”的说明已过时。

Zone 环境编码维度为 64、LTL set 16、LTL 输出 32；Letter 使用卷积环境编码、set 32、LTL 64；FlatWorld 环境编码 16、set 16、LTL 32。具体代码优先于本文摘要。

run_flatworld、run_letter、run_zones 默认训练预算通常为 15M steps；不同环境 rollout、batch、discount 不同，不可混用超参数结果。FlatWorld 默认 rollout 4096/batch 2048/epochs 10/gamma .98；Letter 128/256/8/.94；Zones rollout 4096/gamma .998。

启动器仍要求在仓库根目录运行并设置 PYTHONPATH=src，因为 utils 在解析脚本时就会导入。现已使用 sys.executable 运行子进程并检查退出码。大地图原先引用不存在的 FlatWorld-big-v0 环境 ID，现按 size 选择已注册的 big-8/12/16/20；课程键依然使用已有 FlatWorld-big-v0。

ExperimentConfig 提供 save_dir，但 file_utils 的主路径仍硬编码 experiments；不要认为任意 save_dir 都会生效。

### 3.2 环境与包装器

envs.make_env 按环境类型创建基础环境。sequence=True 使用课程目标的 SequenceWrapper；其他路径结合 LTLWrapper/LDBAWrapper。另有时间限制和 truncation 包装。

FlatWorld-v0 是二维任务环境，LetterEnv 是字母地图任务，PointLtl2 是 Safety Gymnasium Zone 系列。默认限制分别约 FlatWorld 500、Letter 75、Zone 1000 步；以包装器当前参数为准。

Zone 第三方源码和 assets 保持原位置。图形、物理库和系统依赖较重，FlatWorld 短训通过不能替代 Zone 运行验证。

### 3.3 LTL、自动机与搜索

ltl/automata 实现 Rabinizer 转换调用、HOA 解析、LDBA、SCC 等；boolean logic 表达命题约束；search 模块选择可达/避让目标序列；sequence 提供课程和任务采样。

ltl2ldba 通过相对路径 rabinizer4/bin/ltl2ldba 调用工具并使用 joblib 缓存，必须从正确工作目录执行。Java 运行库已随工具保留。整理前执行位丢失导致直接调用 PermissionError，现已恢复 bin 执行权限。

自动机状态、accepting 标记、SCC 与序列目标在研究设计上相关，但这些语义元信息是否到达主模型须逐层验证，不能只凭结构定义确认。

### 3.4 预处理与序列语义

LDBASequence 支持元信息字典，但切片返回普通 tuple/list，丢失元信息。SequenceWrapper 生成剩余目标时切片，所以常见训练路径未必保留完整自动机信息。

preprocess_obss 输出 features、seq、epsilon_mask；当前不提供 tracker 所需 propositions。BatchedReachAvoidSequences 保留某些原始序列，但 minibatch __getitem__ 返回 tensor tuple，可能丢 sequences。token 顺序反转与 metadata 没有同步反转，语义对齐存在风险。

ExhaustiveSearch.Path.to_sequence 创建 LDBASequence，但没有填充所有预期 state_info。这些问题意味着节点特征配置启用不等于训练中有效获得真实自动机语义。

### 3.5 主模型和 GNN

Model 将环境嵌入、LTL 序列嵌入、GNN 嵌入拼接，分别供 actor 和 critic。主图构造器 _build_sequence_graph 使用批量 mask/prefix 构图，当前实际边是相邻序列节点的双向链。

边向量预留 forward/backward/skip/epsilon 类别、accepting 和 distance 等，共 6 维；节点特征为 accepting、in_scc、is_initial、归一化位置，共 4 维。功能位存在不代表所有边类型都在主训练生成。

gnn_net 使用 GATv2Conv、残差/投影、LayerNorm/ReLU 与 gated sum pooling。SimpleGlobalAttention 直接用 gate(x)*x 汇总，没有执行 softmax；论文若称“归一化注意力池化”应核对。torch_scatter 不可用时有 index_add 回退。

辅助 process_reach_avoid_to_graph 支持额外 skip 边，但不是主模型构图路径。不能以辅助示例的边证明主 PPO 已使用完整 LDBA 图。空序列、混合空序列及 metadata 缺失的维度行为仍需专门测试。

### 3.6 PPO 与训练状态

src/torch_ac 实现 rollout、GAE、PPO minibatch 更新；主接口是 model(obs) → (dist,value)，优化器使用主 PPO 损失。train_ppo 构建模型后可装增强包装器，再加载训练状态。

恢复文件通常包含 model_state、optimizer_state、num_steps、num_updates、curriculum_stage、num_eval_steps，另有 vocab 和 LTLNet 权重/阶段评估权重。复现实验需要完整配套，不只单个 pth。

原 checkpoint 异常分支调用不存在的 TextLogger.warning，现用现有 info 输出 WARNING。strict=False 仍可能静默允许 missing/unexpected keys，日志“成功”不代表所有权重完整载入，后续应报告键差异。

初次 rollout 尚无回合结束时，日志缓存含零长度占位，原 reward_utils 会除零。现过滤零长度；无有效样本返回 NaN，保留“没有观测”的含义。这只修复日志计算，不改变 PPO 更新公式。

### 3.7 LongShort：设计、实现和主路径的距离

LongShortConfig 类的默认开关与 CLI 默认值不同：类默认 HV/MS/AT 打开，CLI 默认关闭。训练发现任一开关后可建 LDBA；没有公式时会回退 F a，与环境命题可能不一致。

LongShortEnhancedModel 的默认 forward(return_multiscale=False) 直接调用 base_model。PPO 正好走该接口，所以层次价值、自动机追踪和 compute_loss 不会自动进入主优化。多尺度 GNN 如被替换可能影响基础嵌入，但多尺度辅助损失和详细输出没有因此接通。

详细 forward 假定 base_output 是 dict 并调用 .get，而基础模型返回 tuple；存在接口错误。环境维度推断默认 32 也可能与实际不符。tracker 没有通过默认 forward 完整 reset/step，观测中也缺所需命题。options 开关尚未实例化完整 Options 执行器。

旧 GNN 的 gnn_layers 与迁移代码检查的 convs 不一致，迁移日志不能证明拷贝成功；评估以 state_dict key 中 multiscale 猜结构也不可靠。应保存明确模型配置和版本元数据。

hierarchical_value.py 包含短期环境价值、长期自动机/公式价值与 alpha 融合设计；AutomatonStateTracker 使用 SCC/BFS 等特征。MultiTimescaleLoss 的短序列可能产生空张量 MSE，done.any 可能对整个 batch 过早停止长期回报。

multiscale_gnn 默认只返回最终尺度 pooling，return_details 才取短/中表示；底部 demo 还使用旧接口。hierarchical_gnn、hierarchical_options、training/hierarchical_trainer、graph_processor/ltl_graph 为探索分支，未形成主 PPO 一致训练路径。层次 GAT heads concat 与 norm shape 也有潜在不兼容。

因此当前不能声称完整 HV+MSGNN+AT+Options 已训练成功。本轮不为“整理目录”悄悄重写这些算法，避免改变研究对象而无法证明等价。

### 3.8 评估、分析与工具

simulate.py 是实际策略评估入口，支持 env/exp/seed/num_episodes/formula/finite/render/deterministic 等参数；不支持历史生成器附加的 LongShort CLI 参数。运行前必须以 --help 为准。

确定性策略默认打开；Letter/FlatWorld 的采样评估需显式选择对应参数并在实验记录中注明。有限任务统计 success/violation/ADR/成功步数；无限任务 accepting visits 为有限截断观测量，不直接构成无限满足证明。

scripts/batch_evaluation.py、generate_eval_commands.py 主要是命令组织/交互导入；quick_data_entry.py 手工录表；quick_analysis 和 generate_paper_figures 使用另一套配置/status 数据格式，不能把现存原基线 CSV 直接当增强消融输入。历史提及的 evaluate_all_configs.py/fast.py 不存在。

绘图生成器含硬编码/模拟性能和融合权重；这些是示意资产。重新生成不应替代真实采集日志。微基准测量网络调用耗时，不证明任务成功率或收敛提升。

## 4. 数据、结果与实验可追溯性

现存表的文件名、表头、行数详见 FILE_INVENTORY 的 CSV 节；以下是整理前盘点：

| 文件 | 数据行数 | 说明 |
|---|---:|---|
| results/FlatWorld-v0.csv | 75 | DeepLTL/GCRL-LTL/LTL2Action 等 |
| results/LetterEnv-v0.csv | 75 | 同类基线 |
| results/PointLtl2-v0.csv | 178 | 另含 RAD/RADd/Transformer |
| results/PointLtl2-v0_results_paper.csv | 0 | 仅表头 |
| results/generalisation.csv | 30 | 泛化历史结果 |
| results/safety.csv | 20 | 安全历史结果 |
| results_infinite/FlatWorld-v0.csv | 24 | accepting visits |
| results_infinite/LetterEnv-v0.csv | 20 | accepting visits |
| results_infinite/PointLtl2-v0.csv | 60 | accepting visits |

有限任务常见字段 method/task/seed/success_rate/mean_steps，无限任务包含 accepting_visits。不同任务集、方法和种子数不能直接按总行数推算独立重复实验数量。

现存 CSV 的完整生成溯源尚未恢复，不能将其重新标为本轮新实验。也没有证据表明其中每条结果均由当前增强代码生成。七配置增强消融的实测数据、原始逐回合输出、置信区间和统计检验缺失。

历史训练日志记录 FlatWorld gnn_test、seed 5、16 环境、累计 15,007,744 步，历时约 16:50:34。它支持“有一次实际训练”的判断，不支持完整增强收益与跨种子显著性；日志末尾还存在扩展 ABI 和空间警告。

整理前无随附 experiments checkpoint。现有 eval_datasets 是任务定义，不是策略结果；短训练仅作连通检查，不能补足论文证据。

## 5. 论文与文档演进

原始 DeepLTL PDF 为 29 页。早期 DeepLTL-LS 的中英文完整稿、短稿、Final 命名稿和两份 docs TeX 都保留在 archive/papers。它们内容不完全相同，“Final”文件名不能作为当前终稿判据。

当前 paper.tex 是 910 行左右的 HiLTL ICML 风格稿；paper-new.tex 是约 587 行 IEEE conference 后续稿。原 paper-journal.tex 是 bare_jrnl 示例，已按模板重新归档命名，不能作为完成期刊稿介绍。

references.bib 是共享文献库。paper-new 原 bibliography{ref} 与实际文件不匹配，本次修正为 references。未逐条在线核验文献，不宣称文献真实性审计完成。

图片来自 docs 与 paper，现集中 figures。相同 SHA 图去重，不同图保留；论文 graphicspath 配套更新。缺失两张 Zone 环境截图，未从其他图冒充替代。ICML 样式未随附。完整编译状态见验证报告。

历史技术/实验文档曾写 3.2×、+3–8%、20–35%、25–40% 等，新稿写 up to +6.4%。这些数字没有在当前增强对照原始数据中得到追溯，本总结不背书。图表、摘要、方法和统计必须在投稿前统一。

## 6. 环境与验证限制

环境依赖不完整锁定；requirements 未列全 torch/PyG/Gymnasium/物理引擎。服务器默认 Python 的 torch 2.6+cu124 缺 CUDA 搜索路径，Gym plugin 曾因 glfw 导入失败。不同 conda 环境分别有可用依赖，但无一个直接确认是完整项目环境。

本次通过对子进程临时设置 CUDA lib 与另一环境 site-packages 进行检查，没有安装依赖、没有改共享环境。PyG 检测到旧 torch-scatter/cluster/sparse/spline 与当前 torch ABI 不符并禁用；部分主代码可回退。该组合只用于此次验证，不是发布的推荐锁定环境。

初始 GNN embedding 成功、CLI 成功、shell 调 Rabinizer 成功，直接调用曾因执行位失败。初始 FlatWorld 8 步在日志阶段因除零失败；修复后结果以 validation.json 为准。不能把“模型已完成一次 PPO 更新”扩大为长期稳定性或结果复现。

Zone 完整训练、真实 checkpoint 评估、全部 legacy 测试、CUDA 性能、多种子长训练均没有因此得到验证。完整论文编译也受依赖/资源限制。

### 6.1 Docker 与发布边界

Dockerfile 固定了原 PyTorch 镜像摘要，安装本地 Safety Gymnasium 后再安装 requirements，并从外部网址重新下载 Rabinizer。构建需要联网，基础镜像与当前 GNN 依赖是否匹配未验证，本轮没有构建镜像。新增 .dockerignore 排除本地恢复备份、Git、训练输出与缓存，避免 COPY . 将这些内容打包。SciPy 是分析脚本额外依赖，requirements 也未显式列出。

## 7. 当前风险与建议顺序

| 优先级 | 工作 | 验收条件 |
|---|---|---|
| P0 | 锁定可复现独立环境 | 新环境按文档安装，Flat/Letter/Zone reset 和短 PPO 全通过 |
| P0 | 修正 LongShort 与 PPO 接口 | 明确 forward、价值输出、rollout 字段与损失；梯度到达预期参数 |
| P0 | 贯通自动机 metadata | 搜索→包装→切片→预处理→minibatch→图节点顺序一致 |
| P0 | 明确论文方法和基线 | 区分原 DeepLTL、当前 GNN、各探索组件，删除无证据完成声明 |
| P1 | checkpoint 配置和版本管理 | 保存模型结构、依赖、公式/任务集、seed，报告缺失和额外键 |
| P1 | 执行受控消融 | 同预算同任务集多种子，保存逐回合日志和统计脚本 |
| P1 | 修正评估命令生成器 | 每条生成命令通过实际 CLI 校验并自动收集结果 |
| P1 | 完整论文编译和文献核验 | 补齐真实缺图、合法样式、参考文献与交叉引用 |
| P2 | 清理探索分支旧 API | 更新 legacy 测试、短序列/空图/done mask 边界测试 |
| P2 | 正式版本管理 | 作者审阅后建立首个提交及独立备份，避免未跟踪资产丢失 |

本轮主要目标是工作区整理和事实文档化，不把上述研究开发待办伪装为已经完成。后续实现应先选定真实方法，再基于完整数据重写实验章节。

## 8. 完整性与索引

FILE_INVENTORY 自动枚举项目自有 Python 文件的类/函数、文档行数、CSV 表头/行数、各目录资源数量；第三方代码与 assets 保留原样并按目录统计。清理前 SHA 清单记录所有纳入备份的文件，可用于追溯未逐字展开的第三方材料和二进制资源。

这不是声称逐行理解全部第三方源码或逐像素审计所有资源。对项目自有主训练、模型、LTL/预处理、评估、配置、论文与历史指南进行了重点阅读；静态判断与实际测试分开报告。若某模块未运行，不以文件存在代替运行验证。

整理形成的当前导航与历史档案分别承担事实入口和研究追溯；历史原文保持原貌，因此阅读时应结合已知问题而非照搬命令。
