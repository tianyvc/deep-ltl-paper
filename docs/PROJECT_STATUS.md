# 当前项目现状

更新：2026-10-20（北京时间）。这是当前事实入口，后续验证路径仅在 [CODE_VALIDATION_ROADMAP.md](CODE_VALIDATION_ROADMAP.md) 维护。完整历史记录与旧文档见本文末尾；历史“完成”、旧文件名中的 Final 和论文报告值不作为已复现结论。

## 1. 总体判断与证据边界

项目基于 DeepLTL 的 DeepSets/RNN reach/avoid 序列条件策略，增加 GNN 并探索 HiLTL 的语义图、双价值、自动机追踪和 Options。HiLTL 为两个网站终稿的标题名称，DeepLTL-LS 保留作历史名称。

当前有两条训练协议：原序列课程 PPO（可选 sequence_gnn / deepltl 架构），以及 `src/train/train_paper.py` 的 hiltl-product-v2 多公式乘积自动机协议。v2 已实现 baseline、hv_only、msgnn_only、hv_track、msgnn_track、full_no_options、full_all 七配置。baseline 使用不含 GNN 的原 DeepSets/GRU 组件、固定排序命题词表与 value-guided reach/avoid 搜索；这是统一乘积协议中的参考架构，不是原始发布版本逐位复现。

v2 的真实 LDBA 图包含 guard 词元 GRU、词元均值旁路和进展边消息值，防止同构但不同命题的任务被结构编码混淆。双头、固定/学习/常数门控、不可变自动机历史进入 PPO。full_all 实际执行可变候选 Options，包含子目标 guard/epsilon/超时终止、势差内在奖励、低层条件策略和独立 SMDP 高层 PPO（折扣 gamma^duration）。旧无反向传播的 Options 草图已归档到 `docs/archive/legacy-option-skeleton.zip`。

三环境 train/held_out 清单在 configurations/，训练采样仅访问 train；文本及转换后带 guard/accept 标签的图指纹检查交集。图指纹用状态编号无关的 WL 结构摘要，是保守结构检查，不是完整 LTL 语言等价判定。评估冻结权重，记录每公式每回合指标、checkpoint/任务哈希及评估所用实际训练步数。统计工具先在各独立训练种子内聚合，再计算跨种子均值、标准差、t 区间及配对 SR 差；单种子不输出虚构 CI。

最终冻结源码的34项矩阵通过，涵盖七配置训练/held-out/恢复、误用拒绝、Letter/Zone full_all、门控、第二训练种子与 CUDA2 短训。独立语义和协议测试为12+8项；完整记录见 [维护记录](maintenance/RECORDS.md) 和其机器证据。此前注意力归一化、动作原地修改、空课程统计、GF有限判定亦已修复。长训练、多种子正式预算、500回合正式评估尚未完成，不能声称论文数字已复现。

外部对照 LTL2Action / GCRL-LTL 的数据按用户确认引用自优化前的原论文，本阶段不接入官方代码。保留原始数据并明确引用来源，不与本次新训练合并生成同协议 CI。数学歧义（skip、MC 概率解释、SCC、门控及自动机状态数）使用 README 中明确的工程约定，网站终稿不改。

依赖变更得到用户授权且仅限 wty_LTL；禁用用户级包后torch CUDA及三环境可用。pygame精确版本约束冲突仍在，不声称环境完全锁定或全部接口兼容。终稿、旧结果及评估数据148文件哈希一致。

## 2. 当前目录与使用规则

- `src/`、`src/torch_ac/`：模型、环境、LTL/序列、预处理、训练、评估与 PPO。
- `rabinizer4/` 与仓库内 Safety Gymnasium：必要第三方源码和资源，保留许可证和运行库。
- `results/`、`results_infinite/`、`eval_datasets/`：已有结果与任务定义，不以文档清洗重写它们。
- `overleaf/TLT`、`overleaf/TLT__Copy_`：用户指定的网站长短终稿，目录内文件不改动、不写编译产物。
- `paper/`：论文导航、差异报告、旧稿、图表生成器、投稿材料与原始论文；参见 [论文入口](../paper/README.md)。
- `docs/archive/document-history.zip`：本轮清洗前的项目自有 Markdown 完整快照，避免把旧草稿混进当前入口。
- `.maintenance/`：Git 忽略的本地备份与维护工具，不能作为唯一持久研究档案。

所有项目命令经 cf 在远端 `wty_LTL` 执行；不混入其他环境路径、不擅自修改依赖，不在远端启动第二个 coding agent。具体执行与验收要求见验证路线。

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

gnn_net 使用 GATv2Conv、残差/投影、LayerNorm/ReLU 与 gated sum pooling。SimpleGlobalAttention 已修复为每张图内 softmax 归一化，再加权汇总；该修复会改变旧权重的输出，旧数值需重新评估。torch_scatter 不可用时有 index_add 回退。

辅助 process_reach_avoid_to_graph 支持额外 skip 边，但不是主模型构图路径。不能以辅助示例的边证明主 PPO 已使用完整 LDBA 图。空序列、混合空序列及 metadata 缺失的维度行为仍需专门测试。

### 3.6 PPO 与训练状态

src/torch_ac 实现 rollout、GAE、PPO minibatch 更新；主接口是 model(obs) → (dist,value)，优化器使用主 PPO 损失。train_ppo 构建模型后可装增强包装器，再加载训练状态。

恢复文件通常包含 model_state、optimizer_state、num_steps、num_updates、curriculum_stage、num_eval_steps，另有 vocab 和 LTLNet 权重/阶段评估权重。复现实验需要完整配套，不只单个 pth。

原课程入口现保存 architecture，按明确架构构建与恢复，模型严格加载；架构不匹配直接拒绝，不再异常后从随机权重继续。新协议额外核对实现源码哈希和完整训练/任务合同。

初次 rollout 尚无回合结束时，日志缓存含零长度占位，原 reward_utils 会除零。现过滤零长度；无有效样本返回 NaN，保留“没有观测”的含义。这只修复日志计算，不改变 PPO 更新公式。

### 3.7 旧 LongShort 探索路径的历史诊断（非 v2）

LongShortConfig 类的默认开关与 CLI 默认值不同：类默认 HV/MS/AT 打开，CLI 默认关闭。历史训练曾在发现任一开关后构造 LDBA，并在无公式时回退 F a；当前旧增强开关已拒绝启动。

LongShortEnhancedModel 的默认 forward(return_multiscale=False) 直接调用 base_model。PPO 正好走该接口，所以层次价值、自动机追踪和 compute_loss 不会自动进入主优化。多尺度 GNN 如被替换可能影响基础嵌入，但多尺度辅助损失和详细输出没有因此接通。

详细 forward 假定 base_output 是 dict 并调用 .get，而基础模型返回 tuple；存在接口错误。环境维度推断默认 32 也可能与实际不符。tracker 没有通过默认 forward 完整 reset/step，观测中也缺所需命题。options 开关尚未实例化完整 Options 执行器。

旧 GNN 的 gnn_layers 与迁移代码检查的 convs 不一致，迁移日志不能证明拷贝成功；评估以 state_dict key 中 multiscale 猜结构也不可靠。应保存明确模型配置和版本元数据。

hierarchical_value.py 包含短期环境价值、长期自动机/公式价值与 alpha 融合设计；AutomatonStateTracker 使用 SCC/BFS 等特征。MultiTimescaleLoss 的短序列可能产生空张量 MSE，done.any 可能对整个 batch 过早停止长期回报。

multiscale_gnn 默认只返回最终尺度 pooling，return_details 才取短/中表示；底部 demo 还使用旧接口。hierarchical_gnn、hierarchical_options、training/hierarchical_trainer、graph_processor/ltl_graph 为探索分支，未形成主 PPO 一致训练路径。层次 GAT heads concat 与 norm shape 也有潜在不兼容。

以上是旧探索分支的诊断，不是 v2 的执行路径。v2 的 HV+MSGNN+Track+Options 已有真实梯度和短训证据；完整预算的学习稳定性和性能结论仍待验证。hierarchical_options.py 目前仅导出 v2 helper，旧骨架可在归档中审查。

### 3.8 评估、分析与工具

simulate.py 是实际策略评估入口，支持 env/exp/seed/num_episodes/formula/finite/render/deterministic 等参数；不支持历史生成器附加的 LongShort CLI 参数。运行前必须以 --help 为准。

确定性策略默认打开；Letter/FlatWorld 的采样评估需显式选择对应参数并在实验记录中注明。有限任务统计 success/violation/ADR/成功步数；无限任务 accepting visits 为有限截断观测量，不直接构成无限满足证明。

scripts/batch_evaluation.py、generate_eval_commands.py 主要是命令组织/交互导入；quick_data_entry.py 手工录表；quick_analysis 和 generate_paper_figures 使用另一套配置/status 数据格式，不能把现存原基线 CSV 直接当增强消融输入。历史提及的 evaluate_all_configs.py/fast.py 不存在。

绘图生成器含硬编码/模拟性能和融合权重；这些是示意资产。重新生成不应替代真实采集日志。微基准测量网络调用耗时，不证明任务成功率或收敛提升。

## 4. 数据、结果与实验可追溯性

现存表的文件名、表头、行数详见 历史压缩包内 `docs/FILE_INVENTORY.md` 的 CSV 节；以下是整理前盘点：

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

现存 CSV 的完整生成溯源尚未恢复，不能将其重新标为本轮新实验。也没有证据表明其中每条结果均由当前增强代码生成。七配置已保存小预算工程验证、逐回合输出和统计工具验收记录；正式预算的消融数据与性能置信区间仍缺失。

历史训练日志记录 FlatWorld gnn_test、seed 5、16 环境、累计 15,007,744 步，历时约 16:50:34。它支持“有一次实际训练”的判断，不支持完整增强收益与跨种子显著性；日志末尾还存在扩展 ABI 和空间警告。

整理前无随附 experiments checkpoint。现有 eval_datasets 是任务定义，不是策略结果；短训练仅作连通检查，不能补足论文证据。


## 5. 当前论文与实验口径

当前权威网站快照为 `overleaf/TLT/manuscript.tex`（长，journal）和 `overleaf/TLT__Copy_/manuscript.tex`（短，conference），两个目录各 12 文件，只有主 tex 不同。短版保留 13 个表环境中的实验数值，压缩完整伪代码与展望，补充 κ、有限距离、Opt 配置列与复杂度说明；兼容副本长 13 页、短 9 页。

原样完整构建在 BibTeX 后遇 natbib 数字引用/author-year 模式冲突；副本启用 `[numbers]` 后完整通过，原件未修复。两张 `.png.png` 环境图本机 graphicx 能解析，不再把旧稿的缺图判断套用到网站终稿。

旧 `paper.tex`、`paper-new.tex`、`references.bib` 已原字节迁至 `paper/archive/pre-overleaf/`；旧图和新快照同名不一定同内容，不能覆盖。详细数学/图注/数值疑点保留在 [长短版差异报告](../paper/review/FINAL_VERSION_COMPARISON.md)；方法合同、代码缺口、七组配置、验证顺序仅在 [代码验证路线](CODE_VALIDATION_ROADMAP.md) 维护。

## 6. 已知运行与工程限制

`requirements.txt` 不是完整环境锁。旧验证曾对子进程借用其他 conda 的 CUDA 库和 site-packages，并出现 PyG 扩展 ABI 警告；该历史结果不是当前 `wty_LTL` 的安装方案或验证结论。下次应先独立核实当前环境，不自动安装或升级依赖。

历史 Flat/Letter 各 8 步 smoke 只覆盖非常有限的连通性，不代表 Zone、CUDA、完整 checkpoint、长训练或增强收益验证。初次无完整 episode 时 ARPS/ADR 可为 NaN，不能据此编造有效指标。真实训练日志一次 15M FlatWorld GNN 运行不等于多种子消融。

Dockerfile 的固定原镜像、GNN 依赖和外部下载链尚未验证；没有本轮构建结论。save_dir、默认课程和评估参数等工程限制见上文及验证路线，后续先修实际调用链再优化。

## 7. 文档收敛、历史访问与维护规则

当前自有 Markdown 保留 8 份（含 AGENTS 和新增长任务手册）：根 README、协作指令、本现状、代码验证路线、paper README、论文差异报告和 maintenance RECORDS。第三方 Safety Gymnasium 的说明、许可证相关材料保持原样。

清洗前所有自有 Markdown 按原路径和原字节保存在 [document-history.zip](archive/document-history.zip)，逐文件 SHA-256 与归档哈希见 [清单](archive/document-history-manifest.json)。其中包括旧的三类历史汇编、8 份中英文稿、旧 FILE_INVENTORY、原对齐说明及清洗前现状；这些历史内容不是当前有效运行指令。机器清单/JSON、tex 旧稿、图片和论文证据文件保持原位置。

可用 Python 标准库 zipfile 在隔离目录查看或提取该包；先核对清单，不直接覆盖当前文件。为避免文档再次膨胀，现状事实更新本文，待验证项和决定更新路线，论文差异更新既有差异报告，实际验证追加机器记录和汇总，不为每个阶段新增一份摘要。

本轮变更与完整性检查见 [维护记录](maintenance/RECORDS.md)。用户已授权核验后提交推送，实际发布状态以维护记录和 Git 远端为准。
