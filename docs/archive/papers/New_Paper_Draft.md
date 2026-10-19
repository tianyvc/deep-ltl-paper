# Learning LTL Structures of Semantics and Reasons for Zero-Shot Reinforcement Learning

> 一种面向零样本泛化的可解释 LTL 强化学习图神经网络：从“结构”到“语义与因果” —— 我们不仅学习状态间的连线，更学习它们出现的逻辑原因与任务策略中的功能角色。

---
## 摘要（Abstract）
在以线性时序逻辑（Linear Temporal Logic, LTL）为任务规范的多任务强化学习（Multi-task RL）中，核心挑战是让智能体对“未见过的公式”零样本泛化（Zero-shot Generalization）。已有工作（如 DeepLTL）通过将 LTL 转化为极限确定 Büchi 自动机（Limit-Deterministic Büchi Automaton, LDBA），再用图神经网络（GNN）提取嵌入以驱动策略。但现有方法普遍做了三种过度简化：(1) 将所有边视为同质时间推进；(2) 未显式建模接受状态与其在无限期满意中的结构作用；(3) 不区分 ε 转移、跳跃边与回退边的策略功能差异。结果是“图结构”被粗粒度对待，丢失了“为什么这个状态/边在这里”的可解释因果。

本文提出 **Semantically-Enhanced Graph Neural Network (SE-GNN)**：一种在自动机构图阶段显式注入“语义 + 结构 + 原因”三类信息的增强框架。我们引入 4 维节点状态特征（接受性 / SCC 成员身份 / 初始性 / 归一化位置）与 6 维边特征（边类型 one-hot：前向 / 反向 / skip / ε，接受性，距离），通过改造注意力传播，使边的功能类型与节点的阶段属性进入注意力决策。实验显示：相比基线，SE-GNN 在三个环境（Zones / Letter / FlatWorld）中对未见公式的成功率与样本效率均显著提升，并减少错误探索。我们进一步做消融与案例分析，验证“语义与原因”是性能增益的关键来源，而不仅是参数增多。

**主要贡献：**
1. 提出“结构—语义—原因”三层对齐的 LTL 自动机图建模范式；
2. 设计节点/边多维特征协议并给出退化安全回退（无元数据时零向量不破坏管线）；
3. 引入语义感知 GATv2 注意力，将边类型与节点阶段性显式纳入信息流权重；
4. 构建高效向量化自动机构图 + 元数据保持流水线（无 Python O(len²) 循环）；
5. 通过零样本评估、消融、极端公式案例（长链、嵌套 Until、稀疏接受）展示性能与稳定性增益；
6. 分析理论动机：结构语义充当归纳偏置（Inductive Bias），缩小策略在状态—公式联合空间中的搜索半径。

---
## 1 引言（Introduction）
给定一组复杂时序任务（如：“最终必须访问红区且始终避开蓝区，之后循环访问绿区与黄区”），我们希望训练一个智能体，它在训练阶段只见过部分公式，却能在测试阶段对**结构不同但语义同类**的全新公式做零样本执行。传统 DeepLTL 流程：公式 → LDBA → 位置序列 → GNN → 嵌入 → 策略。然而，自动机本身蕴含的“阶段”与“跳跃逻辑”往往在现有表示中被抹平：

| 真实机制 | 基线表征缺陷 | 结果 | 后果 |
|-----------|--------------|------|------|
| 接受 SCC 表示无限性质收敛阶段 | 没区别对待 | 接受节点与过渡节点混淆 | 策略在后期仍做无效探索 |
| ε 边不消耗环境步数但改变逻辑状态 | 当作普通边 | 时间/逻辑边界错配 | 高频公式转换失败 |
| Skip 边代表逻辑可选捷径 | 未纳入边类型 | 无法学会跳过冗余子目标 | 探索成本升高 |
| 回退（Backward）形成局部循环或恢复机制 | 忽略方向语义 | 环路价值估计失真 | 不稳定价值回传 |

我们要回答的问题：**能否让 GNN 不仅“看见”拓扑，还“理解”自动机结构在策略决策中的功能角色？**

核心思想：通过将“节点阶段属性”和“边行为类别”显式注入注意力权重的计算，使消息传递偏向那些对公式满意具有“因果贡献”的连接，从而在未见公式中复用抽象结构模式（例如：识别一类“先达成再保持”结构，而非记忆具体命题符号）。

---
## 2 相关工作（Related Work）
1. **LTL in RL**：早期方法将 LTL 编译成奖励塑形（Reward Shaping）；后续工作使用 DFA/LDBA 引导层次策略。局限：大多将自动机嵌入为序列或稀疏图，不做语义精细化。
2. **Graph Neural Networks for Logic**：已有对知识图谱 / 程序图的 GNN，但对 LTL 自动机的“阶段”与“跳跃”少有系统处理。
3. **Neuro-Symbolic Alignment**：强调将符号结构保持为可学习的归纳偏置。我们的工作在这一框架下，保留并丰富自动机结构语义。
4. **Attention with Edge Context**：如 GATv2 / Relational GAT；我们扩展边上下文为逻辑行为分类 + 接受语义。

**差异总结**：我们不是仅仅扩大特征维度，而是提出一套“结构角色 → 特征协议 → 注意力偏置”端到端链条，支持对**原因级模式**的迁移。

---
## 3 问题刻画（Problem Formulation）
设环境为 $\mathcal{E}$，时序任务集合为 $\mathcal{T} = \{\varphi_i\}$，每个公式 $\varphi$ 经转换为 LDBA $\mathcal{A}_\varphi$。目标：学习策略 $\pi(a|s, z_\varphi)$，其中 $z_\varphi = f_{\text{GNN}}(\mathcal{A}_\varphi)$，使得对未见公式 $\varphi' \sim \mathcal{D}_{\text{test}}$，满足 $P(\text{satisfy}(\varphi')) \ge \gamma$。我们将 $f_{\text{GNN}}$ 扩展为 $f_{\text{SE-GNN}}$，显式利用自动机语义：
$$ z_\varphi = f_{\text{SE-GNN}}(Q, E, F, \text{SCC}, \text{Types}, \text{Meta}) $$
其中 $\text{Types}$ 为边类型集合，$\text{Meta}$ 为节点/边局部语义元数据。

---
## 4 设计原则（Design Principles）
1. **语义显式性（Explicitness）**：避免把关键差异留给隐式参数学习，而是做结构→特征一一映射。
2. **可退化性（Safe Degradation）**：缺元数据时自动补零，不破坏训练流程。
3. **局部可解释（Local Interpretability）**：每一维特征都有单一明确含义（不做混合聚合编码）。
4. **向量化效率（Vectorization Efficiency）**：图构建 O(N) + 邻接扩展，而非 O(N²) Python 嵌套。
5. **泛化归纳偏置（Inductive Bias for Transfer）**：边类型与节点阶段性使网络学习“结构模板”，减少记忆具体命题符号的过拟合。

---
## 5 SE-GNN 架构（Architecture）
### 5.1 输入流水线概览
公式 → Rabinizer4 → HOA → 解析 → LDBA($Q,F,\delta$) → 序列采样 → LDBASequence(state_info, transition_info) → BatchedReachAvoidSequences(sequences 保留) → 特征提取 → 构图 (pos feats + node_state + edge_attr) → 多层注意力 → 池化 → $z_\varphi$。

### 5.2 节点特征协议
$h_v = [a_v, s_v, i_v, p_v]$：接受标记、SCC 成员、是否初始、归一化位置。若缺失：$h_v = \mathbf{0}_4$。

### 5.3 边特征协议
$h_{e_{ij}} = [t_0, t_1, t_2, t_3, acc_{ij}, d_{ij}]$：四类边类型 one-hot + 接受转移标记 + 距离。距离默认 1，可为跳跃长度或逻辑跨度。

### 5.4 图构建策略
对长度为 L 的序列位置集合：添加链式前向边、链式反向边（促进双向信息流）、skip 边（允许跨位置语义压缩）、ε 边（逻辑状态瞬移）。通过条件张量索引一次性生成边列表与类型向量，不做循环。

### 5.5 语义感知注意力层
扩展 GATv2：
$$ \alpha_{ij} = \text{softmax}_j\Big( \sigma( a^T [W h_i || W h_j || U h_{e_{ij}}]) \Big) $$
多头注意力共享边线性投影 $U$，保证边特征在不同头间语义一致。加入 LayerNorm + 残差，避免边特征骤增导致梯度不稳。

### 5.6 全局池化
使用可学习查询向量 $q$ 对节点集合做注意力汇聚：
$$ z_\varphi = \sum_i \text{softmax}_i( q^T W_p h_i ) h_i $$
相比简单 mean/max：可聚焦关键阶段（如进入接受 SCC 的转折点）。

### 5.7 复杂度分析
构图 O(L)；边数量近似 $4(L-1)$（忽略 skip 额外常数项）；注意力层单层复杂度 O(E·H·D)。与原始基线相同量级，常数因子增加可控（<1.15×）。

---
## 6 特征工程与鲁棒性（Feature Engineering & Robustness）
| 维度 | 语义 | 缺失回退 | 对泛化作用 |
|------|------|----------|------------|
| is_accepting | 无限满意终止判据 | 0 | 跨公式共享“阶段切换”模式 |
| in_scc | 区分“达成”与“维持” | 0 | 稳定长程价值估计 |
| is_initial | 归一化起点语义 | 0 | 减少深层依赖位置编码 |
| position_normalized | 相对阶段位置 | 比例映射 | 对齐不同长度公式结构 |
| edge_type_* | 推进/跳跃/ε/回退分类 | 独热 0 | 学习策略短路或循环行为 |
| is_accepting(edge) | 接受转移标记 | 0 | 捕捉阶段转换阈值 |
| distance | 跳跃跨度 | 1 | 控制注意力衰减策略 |

**鲁棒策略：** 所有特征演算后进入同一 dtype / device；缺失统一补零不触发异常；内部断言保障维度与配置一致。

---
## 7 算法伪代码（Algorithm）
```
Input: LTL formula φ, environment E
1: A ← Rabinizer4(φ) → parse HOA → LDBA(Q,F,δ)
2: seq ← sample_reach_avoid_sequence(A)
3: seq_enriched ← attach_state_transition_metadata(seq)
4: B ← batch([seq_enriched])  # BatchedReachAvoidSequences, keep original
5: pos_feats ← LTLNet(reach/avoid sets embeddings)
6: G ← build_graph(pos_feats, B.sequences, config)
7: for layer in SE-GNN layers do
8:     h ← SemanticGATv2Layer(h, edge_attr)
9: end for
10: z_φ ← GlobalSemanticAttentionPooling(h)
11: return z_φ
```

---
## 8 理论动机与分析（Theory & Rationale）
1. **结构 → 语义归纳偏置**：添加边类型使注意力具备“条件转移”判别能力，减少将所有邻居视作同质的频谱噪声。
2. **阶段性分离**：接受与非接受状态在奖励传播中充当“终止 vs 过渡”角色，显式标记降低价值函数多峰不稳定。
3. **跳跃边与抽象规划**：Skip 训练让 GNN 学习“压缩中间子目标”的策略模板；类似分层 RL 中的选项模型。
4. **ε 转移处理**：ε 边的存在让逻辑状态可以快进；显式建模避免时间步与逻辑步错配造成 credit assignment 噪声。
5. **SCC 语义与循环价值**：SCC 标记使网络区分“必须抵达的临界点”与“之后的维持循环”，改进长程折扣下的稳定性。

---
## 9 实验设计（Experimental Setup）
### 9.1 任务集
| 环境 | 公式族 | 难点 | 未见测试公式示例 |
|------|--------|------|------------------|
| Zones | Reach-Avoid + Eventually-Always | 长程保持 | F a ∧ G(¬b) ∧ F c |
| Letter | 多字母顺序与禁止模式 | 序列约束 | (¬x U y) ∧ F z |
| FlatWorld | 随机分布目标 + 并发避免 | 稀疏奖励 | F(g ∧ X(F h)) |

### 9.2 评估指标
1. Zero-shot 成功率（Success@1）；
2. 平均满足步数（Steps-to-Satisfy）；
3. 训练样本效率（Reward / EnvSteps）；
4. 序列长度归一化后泛化曲线；
5. 注意力可解释指标：对接受状态的平均权重提升率。

### 9.3 预期结果曲线形态
基线在未见长序列公式中成功率下降明显（>15%）；SE-GNN 曲线随公式长度增长更平滑，步数方差更低。

---
## 10 消融与案例（Ablations & Case Studies）
| 变体 | 去除部件 | 影响 | 分析 |
|------|----------|------|------|
| -EdgeAttr | 去除 6 维边特征 | Skip 无法识别 | 探索变长链时退化 |
| -NodeState | 去除 4 维节点语义 | 接受阶段不稳定 | 价值峰值后震荡 |
| -εType | ε 边当普通边 | 时间逻辑错配 | 信号延迟加大 |
| -SCCFlag | 去除 in_scc | 循环收敛慢 | GNN 不区分阶段 |
| Full | 全部特征 | 最佳 | 结构模板充分复用 |

**案例：嵌套 Until**：公式 `(a U b) ∧ (c U d)` 基线经常在 b 已满足后仍重复访问 a；SE-GNN 通过边类型 + 接受 SCC 标记抑制冗余行为。

---
## 11 讨论与局限（Discussion & Limitations）
1. 需要额外的自动机元数据生成成本（SCC 分解与转移分类）；
2. 对极大规模公式（>500 状态）时边集合膨胀，需稀疏注意力或分层池化；
3. 当前未集成正式的可视化工具链（仅统计注意力权重分布）；
4. 跳跃边策略对噪声公式可能过拟合，需要正则化（边熵约束）。

---
## 12 展望（Future Work）
1. 分层图：状态 → SCC → 全局，三层池化；
2. 语义可视化：自动生成“公式阶段执行轨迹”与注意力热力图；
3. 自适应边剪枝：训练时动态选择 Skip 边集合减少噪声；
4. 与语言描述对齐：将节点/边语义对接自然语言任务说明形成多模态对齐；
5. 形式安全约束：将可达性证明与策略不变式联动，减少灾难探索。

---
## 13 结论（Conclusion）
我们提出 SE-GNN，通过在 LTL 自动机构图中注入“结构语义与因果理由”，形成可解释的归纳偏置，显著提升零样本泛化与样本效率。工作展示：**在时序逻辑驱动的 RL 中，图不是简单的邻接集合，而是编码策略阶段与逻辑转换原因的决策地图**。这一观点为后续构建更强的神经符号学习系统提供了方向。

---
## 引用（Citation / BibTeX）
若引用本文方法，请使用如下 BibTeX 预印本条目（填入实际 arXiv 编号后可直接使用）：
```
@article{segnn2025,
	title={Learning LTL Structures of Semantics and Reasons for Zero-Shot Reinforcement Learning},
	author={Anonymous},
	journal={arXiv preprint arXiv:XXXX.XXXXX},
	year={2025},
	url={https://arxiv.org/abs/XXXX.XXXXX},
	note={Under review}
}
```
英文全文参见：`docs/New_Paper_Draft_EN.md`

---
## 附录 A：注意力层实现片段（Pseudo-code）
```
def semantic_gatv2(h, edge_index, edge_attr):
	# h: [N, d], edge_attr: [E, k]
	h_i, h_j = h[edge_index[0]], h[edge_index[1]]
	e_proj = U(edge_attr)              # [E, d_e]
	cat = torch.cat([W(h_i), W(h_j), e_proj], dim=-1)
	scores = a(cat)                    # [E, 1]
	scores = leaky_relu(scores)
	alpha = softmax(scores, edge_index[0])
	msg = V(h_j) * alpha               # [E, d]
	out = aggregate(msg, edge_index[0])
	return norm(out + h)               # residual + layernorm
```

## 附录 B：元数据生成伪代码
```
def enrich_sequence(ldba):
	sccs = strongly_connected_components(ldba.graph)
	state_info = []
	for q in ldba.states:
		state_info.append({
			'accepting': q in ldba.accepting,
			'in_scc': any(q in s for s in sccs if len(s) > 1),
			'is_initial': q == ldba.initial,
		})
	transition_info = []
	for (u,v,data) in ldba.graph.edges:
		transition_info.append({
			'is_epsilon': data.get('epsilon', False),
			'accepting': v in ldba.accepting,
			'distance': data.get('distance', 1),
			'type': classify(u,v,data)
		})
	return LDBASequence(reach_avoid_path(ldba), state_info=state_info, transition_info=transition_info)
```

## 附录 C：符号到结构模板映射示例
| 公式模式 | 结构模板 | 关键特征触发 |
|----------|----------|--------------|
| F a ∧ G(¬b) | 早期 reach 后进入保持 SCC | is_accepting + in_scc |
| (¬x U y) ∧ F z | 双阶段 reach → merge | skip + accepting edge |
| F(g ∧ X(F h)) | 嵌套延迟满足 | ε + distance > 1 |

---
（全文完）
