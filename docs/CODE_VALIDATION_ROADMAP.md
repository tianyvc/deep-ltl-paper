# 代码核实、优化与验证路线

更新：2026-10-20。当前统一协议为 hiltl-product-v2，七配置、DeepSets/GRU 参考基线、多公式训练、独立 held-out zero-shot、Options/SMDP 和种子统计工具均已落地；最终34项矩阵及12+8项方法检查已有短时证据。正式论文性能尚未复现。最新结果见 [维护记录](maintenance/RECORDS.md)，长任务命令见 [操作手册](LONG_RUNNING_TASKS.md)。

当前执行入口为 `src/train/train_paper.py`，模型位于 `paper_hiltl.py` / `deepltl_reference.py`，任务合同位于 `src/train/paper_tasks.py`，Options 位于 `src/model/paper_options.py`。原序列课程使用 `train_ppo.py --architecture deepltl` 可单独运行。下面保留的方法合同与历史诊断用于追踪修复依据，**第5节旧分支缺口不代表 v2 仍缺少这些实现**。

接下来依次进行：长 horizon 小规模 pilot，测吞吐/显存/梯度与学习信号；冻结训练清单、评估策略、gamma、horizon 和预算；七配置×1–5种子正式训练；每种子每公式500回合 train/held_out 评估；按训练种子聚合 CI 与配对差；最后 profile 并逐项优化计算。外部对照本阶段沿用原论文引用数据，不接官方训练、不混入新运行统计。只要修改方法源码导致实现哈希变化，就使用新输出重新训练，不绕过严格恢复。

已明确的工程定义：cyclic SCC；接受转移目标作为接受节点；不可达距离为1；真实边不虚造 skip；固定 sigmoid3/.5 为默认、learned/constant 为单独消融；长期头监督有限时域 MC 回报代理而非严格满足概率；完整回合 horizon 零 bootstrap。Options 的内在奖励、子目标终止、gamma^duration 和高层 PPO 均见 README 及源码。语义图含 guard GRU/词元均值与进展边消息值。图指纹排除已知编译等价交集，但不保证完整语言等价。checkpoint 恢复新回合，不保证中断前后逐位连续轨迹。自动机4/8/4（剪枝补全4/8/5）与原稿5/12/18差异尚未解释；网站终稿不改。

阅读顺序：先读方法合同和缺口，再按阶段执行。状态区分为“静态发现”“待实测”“验证通过”；文件存在、开关打开、日志写完成，都不等于通过。不得把论文报告值当成程序必须跑出的验收目标。

## 1. 文档地位与边界

权威论文快照为 `overleaf/TLT` 与 `overleaf/TLT__Copy_`，共同方法是后续工程对齐的主要依据。短版澄清项与长版独有细节一起作为候选规范；冲突处列为待决策事项，不能擅自任选。本文保留静态合同并记录后续落地范围；不更改网站终稿或已有实验数据，不将工程定义当作原稿歧义已解决。

## 2. 研究目标与共同方法

未知动力学 MDP 与 LDBA 组成乘积过程，策略接收环境状态 s 和逻辑进度 q，目标是提高 LTL 满足表现。有限时间 success rate、折扣奖励、接受状态访问次数分别对应不同统计量，不能混称严格的无限满足概率。

### 2.1 SE-GNN 的论文规格

| 对象 | 共同描述 | 实现前必须确认 |
|---|---|---|
| 图 | 任务 LDBA 的状态和带语义转移 | 不能以公式序列邻接链替代完整自动机 |
| 节点 | 4D：accepting、SCC、initial、normalized distance to F | SCC 定义、不可达和零分母处理 |
| 边 | 6D：fwd/bwd/skip/epsilon 四类 one-hot、target accepting、normalized target distance | 互斥优先级、self-loop/trap、skip 的可实现定义 |
| 网络 | 3 层 GATv2，hidden 64，4 heads，LayerNorm 与残差 | 不只核对配置值，还要核对实际输入和调用 |
| 全图池化 | softmax attention pooling | 权重应在每张图节点集合上归一化 |
| 多尺度短期特征 | 拼接 3 层 pooled embeddings 得 z_short | 保持不同图、batch、rollout 时间轴一致 |
| 当前节点特征 | z_GNN(q) | q 由真实环境转移决定，不能用位置编号替代 |

### 2.2 双价值与融合

- 短期头输入：环境特征、z_short 和当前自动机节点特征；监督为短期 TD 类目标。
- 长期头输入：proximity、自动机历史 LSTM、当前节点特征；论文明确区分其与环境特征的依赖。到达/接受代理与完整回合 MC 的目标定义还需统一。
- 融合：`V_total=(1-β)V_short+βV_long`。短版固定式为 `β=σ(κ*(1-d/d_max-θ))`，给定 `κ=3,θ=.5`；正文 MLP/附录 beta_net 的另一说法暂不视作已裁定规范。
- PPO 使用实际融合 value，并对 short/long 分头计算辅助目标；不能只让独立类返回一个字典就认定训练链路接通。
- 长版补充 short/long hidden 256/128、loss 权重 1.0/.3 等，后续需以明确实验 preset 固化，避免从短版省略推断不存在。

### 2.3 自动机追踪的时序约束

每个并行环境独立保存 q，episode reset 时初始化，在每个真实环境 transition 后用下一状态的命题标签更新。rollout 必须保存对应 q/图元数据，PPO 打乱 minibatch 时只读取历史存储状态，不能在 forward 中再次推进自动机。terminated 与 truncated、epsilon actions、重置以及历史 LSTM 状态须有一致约定。

## 3. 七组实验配置合同

这里根据短版 Opt 列明确组件关系。七个名称现在均由 v2 CLI 实现，正式预算结果仍待采集。

| 配置 | HV | MSGNN | Track | Options |
|---|---|---|---|---|
| baseline | 否 | 否 | 否 | 否 |
| hv_only | 是 | 否 | 否 | 否 |
| msgnn_only | 否 | 是 | 否 | 否 |
| hv_track | 是 | 否 | 是 | 否 |
| msgnn_track | 否 | 是 | 是 | 否 |
| full_no_options | 是 | 是 | 是 | 否 |
| full_all | 是 | 是 | 是 | 是 |

baseline 必须明确是否原始 DeepLTL。原方法的 DeepSets/RNN 序列表示不能被悄悄替换成启用图语义的配置，再继续冠以原基线名称。HV 无 tracking 时如何获取 q/proximity 的合法替代，是上述消融可解释性的一部分，需要实验协议补足。

论文最佳平均配置为 **hv_track (.993)**，不是 full_all (.963)，full_no_options 为 .929，略低于 baseline .933。因此后续目标不是把组件全部打开就宣称最佳，MSGNN 在较小自动机上的额外复杂度与未来更大图价值需分别评估。

## 4. 任务与评估口径

| 项目 | 论文描述 | 需要保留的复现证据 |
|---|---|---|
| 环境 | PointLtl2 / Letter / Flat，有限与无限任务 | 实际 Gym ID、版本、地图、奖励与终止规则 |
| φ1 | F blue ∧ F green；声称 5 states | converter 命令、版本、原图和状态计数 |
| φ2 | F blue ∧ F green ∧ F yellow；声称 12 states | 同上 |
| φ3 | (!blue U green) ∧ F yellow；声称 18 states | 同上，检查 held-out 重叠 |
| 训练 | 约 1M steps，batch64，lr3e-4，Adam，γ_RL .998 | 完整命令、配置快照、预算计数口径 |
| 种子 | 1–5 | 每种子 checkpoint、日志与环境随机性 |
| 评估 | 500 episodes | 每种子还是总计、确定性/采样策略、时间上限 |
| 零样本 | blue/green/yellow 文法组合 | train/validation/test 明确划分、公式去重和同义处理 |
| 区间 | 95% CI | 聚合单位、方法、置信区间与显著性分析脚本 |

百分比统一同时保存原始 SR、百分点差和相对差；跨环境、跨公式和跨种子平均顺序必须固定。有限时域 infinite-task 评估中接受访问指标只能作为代理，不自动等价于无限时域语义满足。

## 5. 旧探索分支的历史关键差距（已由 v2 新路径处理）

以下是补齐前的静态发现，定位属历史版本。旧增强入口已 fail-fast；v2 不调用这些有问题的 wrapper。表内关于正式实验数据缺失仍成立；实现链路按本文件顶部及运行记录判断。

| 优先级 | 论文要求 | 当前观察 | 主要位置 |
|---|---|---|---|
| P0 | rollout 与 PPO 实际使用增强值 | 默认 forward 直接返回基础输出，增强分支需 return_multiscale=True；辅助损失未形成主 PPO 的完整链路 | `src/model/longshort_integration.py:160,179,222` |
| P0 | 合法模型输出接口 | 增强分支对 base_output 使用 `.get('ltl_embedding')`，而基础模型接口为 tuple，存在不匹配 | 同文件 207 |
| P0 | 每个真实 transition 更新 q | tracker.step 放在增强 forward 内，缺少安全的 rollout 时序合同，训练重放可能重复推进 | 同文件 193–198；rollout/采样路径 |
| P0 | 完整 LDBA 图及状态语义 | 当前主图路径主要由序列构造双向邻接链；位置归一化不能代表到 F 的真实距离 | `src/model/model.py` 及图构造/预处理路径 |
| P1 | 长期历史 LSTM 与双头输入 | tracker 主要是 4D 结构特征，没有论文历史 LSTM；short 主要用环境输入、long 用 automaton+formula，维度与合同不同 | `src/model/hierarchical_value.py:26,158` |
| P1 | 明确固定门控或学习门控 | 当前 alpha 来自 weight_net，不等于固定距离 κ/θ 函数 | 同文件 224 |
| P1 | 分头 TD/MC | 当前 compute_loss 使用 v_total 扩成全序列，再交多时间尺度损失，与独立 short/long 监督不同 | integration 255–269 |
| P1 | softmax 图池化和元数据一致 | 主路径池化门控与论文归一化注意力需对齐，切片/反序/minibatch 的图元数据传递需补齐 | model/ltl 图编码、预处理及 PPO batch |
| P1 | 真正原始 baseline | 正式三环境配置已启用 GNN node/edge 特征，关闭 HV 不等于恢复原 DeepLTL | `src/config/model_config.py` |
| P2 | 论文预算可复现 | 现有启动器/训练默认预算、batch 等与论文 1M/batch64 不一致 | `run_*.py` 与 train CLI |
| P2 | 完整七组结果证据 | 现有 CSV 主要为外部/原始基线，缺可追溯七组增强多种子结果 | `results/`、`results_infinite/` |

配置中的 64/3/4、node4/edge6 数值匹配只能证明声明接近，不证明实际图语义、损失或数据流符合论文。


## 6. 执行总表与阶段门槛

| 阶段 | 优先级 | 要解决的问题 | 通过之后才做什么 |
|---|---|---|---|
| A 环境与可恢复基线 | P0 | 当前 wty_LTL 能否独立运行，baseline 是否清楚 | 建立可信失败基准 |
| B 方法定义与数据合同 | P0 | 论文歧义、图结构、标签和时序是否有唯一约定 | 编写有语义的验证 |
| C 自动机、图和批处理 | P0 | 真实 q、图特征、metadata 是否贯通 | 接入价值和训练 |
| D 双价值与 PPO | P0 | 目标正确且实际参与 rollout、GAE 和梯度更新 | 跑最小 hv_track |
| E checkpoint、环境和评估 | P1 | 能否持续训练、正确恢复和可重复评估 | 固化实验 preset |
| F 受控实验与统计 | P1 | 七组配置、泛化和提升是否有原始证据 | 讨论性能结论 |
| G 计算优化 | P2 | 哪些瓶颈值得优化且不改变算法语义 | 扩展大图/多环境 |

阶段失败先定位原因，不把多个结构改动和加速改动同时合入。原始 baseline、当前序列 GNN、论文完整 LDBA 图方法须单独命名；二者互换属于研究方法变化，不是目录整理或无行为变化的重构。

## 7. A：环境与基线核实

**入口**：`src/train/train_ppo.py`、`src/config/`、`run_*.py`、`src/envs/__init__.py`、`rabinizer4/bin/ltl2ldba`。

1. 记录 Git revision、未提交差异、Python 可执行文件、依赖版本、CUDA/驱动信息、环境变量名及必要运行值；不记录凭据。仅使用远端 `wty_LTL`，不混入其他 conda 的 site-packages。
2. 核实 torch/PyG 的实际执行路径、扩展 ABI 警告及是否回退。缺依赖或环境失败时记录阻塞，需改变依赖时先征求确认。
3. 用已有 GNN 检查、训练/评估 `--help`、Rabinizer 小公式建立当前行为基准，分别保存退出码与日志。已有脚本可能覆盖范围窄或捕获异常，应阅读并确认失败会导致非零退出。
4. 用最小 FlatWorld、Letter smoke 验证基础接口；8 步只作为启动检查，另需覆盖至少两个 rollout、真实 episode 结束/reset 与非空 PPO minibatch。Zone 单独核实物理/渲染/观测接口；Flat 通过不能代替 Zone。

已有检查入口（具体执行范围和结果见机器记录）：

```powershell
cf run -- /home/dell/anaconda3/bin/conda run -n wty_LTL env PYTHONPATH=src python tests/test_gnn_embedding_quick.py
cf run -- /home/dell/anaconda3/bin/conda run -n wty_LTL env PYTHONPATH=src python tests/test_gnn_fix.py
cf run -- /home/dell/anaconda3/bin/conda run -n wty_LTL env PYTHONPATH=src python src/train/train_ppo.py --help
cf run -- /home/dell/anaconda3/bin/conda run -n wty_LTL env PYTHONPATH=src python src/evaluation/simulate.py --help
cf run -- /home/dell/anaconda3/bin/conda run -n wty_LTL rabinizer4/bin/ltl2ldba -i "F a" -p -d -e
```

从仓库根目录运行；普通终端先 `cf shell` 再 `conda activate wty_LTL`。绝对 conda 路径是当前服务器路径，不是可移植依赖锁。命令包含具体脚本时，应先确认其输出目录和副作用。`tests/legacy/` 不作为整体通过的回归集。

**验收**：上述环境记录可解释；已有检查真正失败即报错；每个声称支持的环境可 reset、step 并完成更新。不能将历史跨环境临时排障记录算作当前 wty_LTL 验证。

## 8. B：先确认方法定义

将以下决定写入本文件的决策记录，再编码和扩展测试：

| 待确认项 | 核实方法 | 产物与通过标准 |
|---|---|---|
| SCC 位的含义 | 用非平凡/accepting SCC 等候选定义手算小图 | 一种明确语义；特征不因所有节点属于 SCC 而恒为 1 |
| 距离、trap、d_max | 反向 BFS 与手算图对照，包括 F 为空和全距零 | 不可达、零分母、终止态都有限且定义明确 |
| forward/skip/epsilon | 检查无权最短距离下的距离不等式和真实转换图 | 类别可产生、优先级明确；不能保留永远不可能出现的 skip 定义 |
| 固定门控或学习门控 | 对照长短正文、设置和附录 beta_net | 唯一公式和输入；固定式应满足 [.1824,.8176] 数值范围 |
| V_long 的估计对象 | 区分到达 F、接受次数、MC 回报与无限满足概率 | 输出范围、监督标签、截断和理论解释一致 |
| HV 无 tracking 的消融 | 明确 q/距离/历史来自何处 | 不泄露标签，不偷用被关闭组件，配置变化可解释 |
| 500 episodes 与 CI | 明确每种子还是总数、跨种子聚合和随机策略 | 一种完整评估协议，不重复计算回合造成伪重复 |
| held-out φ3 | 比较训练、调参、评估公式及转换结果 | 排除明确重叠；语法不同不自动等于语义不同 |

**验收**：没有影响实现或统计含义的隐含默认。尚未裁定的配置标记不可复现，不凭个人猜测填成论文既定方法。终稿疑点保留在 [论文差异报告](../paper/review/FINAL_VERSION_COMPARISON.md)，网站目录保持原样。

## 9. C：自动机、图与元数据贯通

**入口**：`src/ltl/automata/ldba_sequence.py`、`src/envs/seq_wrapper.py`、`src/envs/ldba_wrapper.py`、`src/preprocessing/preprocessing.py`、`src/preprocessing/batched_sequences.py`、`src/model/model.py`、`src/model/gnn_net.py`、`src/model/hierarchical_value.py`。

### 9.1 自动机追踪

用人工可枚举轨迹构造独立 oracle，逐步核对 `(q_t, L(s_{t+1}), q_{t+1})`。覆盖无命题、多命题、违反、接受、epsilon、自环、不可达，以及并行环境不同时间 reset。区分 terminated、truncated 与训练 bootstrap 语义。

**通过标准**：逐步 q 与 oracle 一致；每个真实 transition 只更新一次；仅 reset 对应环境；同一 observation 被多次 forward、多个 PPO epochs 重放或打乱 minibatch 时，tracker 和历史状态不变化。LSTM history 必须与时间轴和 reset mask 一致。

### 9.2 图和特征

从转换器产物构建完整 LDBA，保存图与转换参数，对照手算节点/边特征。核实 edge type one-hot、目标接受标记、距离和当前 q→节点索引。针对原始状态号不连续、孤立节点、多 SCC、空序列和混合长度 batch 验证。

**通过标准**：边与转换器图一致；特征值与独立定义一致；没有图间串边或非预期 padding 节点；池化 softmax 在每张图内和为 1；节点置换并同步边索引后图级输出在约定浮点容差内一致。单独评估跨图 padding 和空图的输出维度。

### 9.3 元数据与批处理

给每张图/节点/环境/时间位置标识，沿“搜索 → sequence 切片 → wrapper → 反序 → preprocessing → rollout → minibatch”检查。不以 shape 正确代替语义对齐。

**通过标准**：独立前向与组合 batch、拆分 batch 的输出在固定随机性下相符；反序同步重排 metadata；切片仍能定位真实 q；sample id 的手工映射与模型读取一致。不得用零特征回退掩盖本应存在的自动机信息。

## 10. D：双价值、回报目标与 PPO 接通

**入口**：`src/model/longshort_integration.py`、`src/model/hierarchical_value.py`、`src/model/multiscale_gnn.py`、`src/torch_ac/algos/base.py`、`src/torch_ac/algos/ppo.py`、`src/train/train_ppo.py`。

旧探索分支历史调用点（v2 使用 train_paper 独立采集/优化，不经过该 wrapper）：base 的 rollout 与 bootstrap、ppo 的 minibatch update 都调用 `self.model(...)`；包装器默认 forward 在第 179 行直接返回 base 输出。应沿这些实际调用点验证增强值，而不是只运行 `return_multiscale=True` 的示例。

1. **统一输出接口**：明确 actor distribution、V_total、V_short、V_long、β 与辅助表示的取得方式，避免 tuple/dict 混用；batch=1 和不同环境特征维度都覆盖。
2. **回报手算对照**：给定短小 reward/done 序列，独立计算 TD、MC、GAE 与 bootstrap。覆盖同一 batch 不同 done、无 done、最后一步终止、时间截断、不足 n-step、长度 0/1；不能被 batch 中任意一个 done 截断其他样本。
3. **rollout 一致性**：buffer 保存当时对应 q/history/value；GAE、bootstrap、value clipping 和更新前向使用一致的 V_total 定义。PPO 不能用基础 value 收集、用另一种 value 更新却不明确算法变更。
4. **真实梯度与更新**：在非退化合成任务和真实小 rollout 上记录 loss 分项、梯度范数及参数更新。检查 short/long head、GNN、actor 各自有预期依赖；关闭组件后对应损失不进入优化，相关参数不更新。冻结指定分支检查另一分支，避免所有层共享后无法解释梯度来源。
5. **目标隔离**：bootstrap/MC target 的 detach 合同明确，value target 不通过同一个估计网络形成不当梯度；short 和 long 分头监督，不能将一个 V_total 复制为整段序列冒充多尺度学习。
6. **基线回归**：增强全部关闭后，明确恢复的基线与独立基线在同权重、eval 模式、同输入下输出一致；同随机种子的一次更新在约定容差内一致。

**验收**：小例子数值与独立 oracle 一致，loss/梯度有限，无空张量均值；增强路径确实参与更新且 inference 无状态推进副作用。梯度验证需避免恰巧饱和或零优势样本，单次零梯度不直接判实现错误。最小 hv_track 连通后再增加 MSGNN 和 Options。

Options 另查 `src/model/hierarchical_options.py`、`src/training/hierarchical_trainer.py` 与主循环是否真的连通；核实 option 初始化、选择、终止、执行动作和相应奖励/损失。若未接通，full_all 不能以“开关存在”验收。

## 11. E：持久化、环境兼容和评估链

1. **checkpoint**：保存模型结构版本、配置、词表、优化器、训练步、课程、随机状态及需持久化的历史状态合同；在新进程加载。检查 missing/unexpected keys，不用 strict=False 的无报错认定完整恢复。
2. **恢复对照**：固定确定性条件，对比连续两次更新与“更新一次→保存→重启→再更新”的参数、步数、课程和统计；若环境状态无法序列化，应明确恢复边界，不宣称轨迹完全一致。
3. **全环境**：Flat/Letter/Zone 分别检查 observation shape/range、propositions、action distribution、reset、time limit、render/headless 与一次完整 update；CUDA 在实际计划使用时独立检查。记录回退路径和警告影响。
4. **评估 CLI**：生成命令逐条与真实 `simulate.py --help` 核对；失败返回非零，缺 checkpoint/词表直接报错，不能自动降级成随机策略还写成功记录。
5. **指标 oracle**：构造已知成功/违反/超时轨迹核对 success rate、步数、奖励、accepting visits 分母；完成评估后每回合有原始记录并能聚合复算。

**验收**：新进程加载后 eval 输出可对照；配置不兼容被报告；评估不会因 CLI 参数失效或异常被吞而产生伪结果。首次 smoke 无完整回合时指标可缺失/NaN，但不能把它当有效成功率。

## 12. F：论文 preset、消融、泛化与统计

先建立 baseline、hv_track 的小预算运行作为工程试验，检查日志与统计链，再扩大到完整七配置。小预算的排序不构成论文结论。正式预算由论文合同和明确决定的协议控制，不能为了匹配论文数值反复选择种子、预算或任务。

- 七配置使用一致训练任务、训练预算、种子集合、评估策略与时间上限；每项开关实际控制的模块由 D 阶段证据证明。
- 保存论文 preset，而不是随意覆盖已有启动器默认值；记录“环境交互步数、并行进程步数、优化更新次数”的换算。
- 固化 train/validation/test 清单及哈希，检查 φ3 重叠。测试集不用于门控、损失权重或早停调参；语义去重能力不足时明确局限。
- 逐种子 checkpoint 与逐回合日志必须齐全；跨公式/环境/种子的平均顺序、失败运行处理、CI 方法提前指定。以独立种子为单位分析训练随机性，不能把同策略的所有回合当独立训练重复。
- 同时输出 SR、百分点差、相对差；区间和显著性针对哪种效应量须明确。保存无四舍五入中间表，再生成最终表和图。
- 外部论文抄录基线、仓库旧 CSV、本次实测结果分别标记来源。硬编码或模拟图不进入实验结论。

**验收**：任一图表格能由原始日志和固定分析版本重新生成；每个主张可定位到独立实验记录。不要求得到 .993 或 +6.4%，要求诚实报告结果、方差和限制。无限任务仅报告实际有限时域观测指标，理论满足保证需要单独论证。

## 13. G：在正确性稳定后优化计算

先固定一批真实轨迹与不同大小的自动机，记录基准，再逐项优化。每项有修改前后版本和数值/语义回归，避免把减少数据或预算当加速。

| 候选优化 | 先测什么 | 允许的优化方式 | 必须保持的条件 |
|---|---|---|---|
| LTL 转换、SCC、BFS | 重复调用次数、总时间、缓存命中 | 以公式/工具版本/选项为键缓存静态图 | 图与特征定义不变，缓存失效键完整 |
| 图构建和 batch | Python 构图耗时、GPU 等待、padding 比例 | 缓存静态拓扑、向量化 batch、减少重复搬运 | q/边/metadata 不变、无跨图串联 |
| GNN 前向与反向 | operator profile、峰值内存、图大小 | 批处理与传输优化；混合精度另行数值评估 | 参数更新后不可复用过期 embedding，梯度不丢失 |
| rollout 与环境 | env step、模型等待、进程开销 | 调整并行数、数据传输和日志频率 | 总交互预算、随机性与终止语义可对照 |
| checkpoint 和日志 | IO 时间、大小、丢失恢复成本 | 降低重复序列化，分离轻量日志与完整快照 | 可恢复状态完整，不删除研究证据 |

测量采用预热、重复、同硬件同负载、固定 workload；GPU 计时需同步，报告中位数/离散度、吞吐、峰值显存/内存及端到端占比。端到端时间包括转换、环境、训练、日志和评估；单层 microbenchmark 不能等同总训练加速。

**验收**：至少有可复算的性能基准与正确性回归，未劣化必要精度/稳定性；引入 mixed precision、采样改变或结构裁剪时单独评估学习效果。若瓶颈尚未测出，不先重写图网络或安装新依赖。

## 14. 执行记录与停止条件

每阶段留一份机器记录，建议路径 `docs/maintenance/validation-runs/<run-id>/`（本轮已创建并保存短时验收证据）：

- revision、工作树差异摘要、环境、精确命令、退出码、耗时；
- case id、输入或输入哈希、预期依据、实际值、容差、结论与日志路径；
- 配置、任务集合、转换器版本、种子、checkpoint/原始日志哈希；
- loss/梯度、指标聚合、性能 profile 按该阶段需求附上。

依赖失败、oracle 不明确、数据混用、梯度未接通、checkpoint 静默缺键、held-out 泄漏时停止扩展相应实验并登记原因。A–E 是完整运行和正确性的门槛，F 才产生研究效果证据，G 才讨论计算加速。不要在这些阶段之外另生多个摘要 Markdown；将决定和阶段状态更新到本文件，原始日志保持机器格式。
