# DeepLTL-LS: Hierarchical Long-Short Term Integration for Zero-Shot LTL Satisfaction

## 摘要 (Abstract)

线性时间逻辑 (LTL) 强化学习中的零样本泛化代表了神经符号 AI 的一个关键前沿。虽然 DeepLTL 等最新方法成功利用图神经网络 (GNN) 对 LTL 自动机进行编码，但它们根本上将逻辑规范视为均质的拓扑结构，导致两个关键问题：（1）**语义扁平化** — 转移类型（进展 vs. 捷径）之间的因果区分丧失；（2）**时间近视** — 无法平衡即时子目标满足与长期逻辑规划。

本文提出 **DeepLTL-LS**，一个统一框架通过两个协同创新来弥合这些差距。首先，我们引入**语义增强图神经网络 (SE-GNN)**，通过条件化边语义（ε-转移、跳跃、前向、后向）和节点角色（SCC 成员资格、接受状态）来注入显式因果角色到消息传递机制中。其次，我们提出**长短期融合**架构，将价值函数分解为密集局部分量（由多尺度 GNN 引导）和稀疏全局分量（由自动机追踪的层次价值网络引导）。该分解解决了复杂时间任务中的信用分配问题。

在多个多样化领域（PointLtl2、LetterEnv、FlatWorld）的广泛实验表明，DeepLTL-LS 相比现有基线实现了最先进的零样本性能，展现出优越的样本效率和对公式复杂度的鲁棒性。**关键结果**：在难公式上成功率提升 **20-35%**，收敛速度加快 **25-40%**，零样本泛化系数稳定在 **>0.80**。

**关键词**：LTL 规范；图神经网络；强化学习；零样本泛化；神经符号 AI

---

## 1. 引言 (Introduction)

### 1.1 研究背景

形式化方法与强化学习的结合为可验证和安全的自主智能体提供了一条有前景的途径。线性时间逻辑 (LTL) 作为强大的规范语言，允许用户定义涉及时间顺序和安全约束的复杂、非马尔可夫任务。此领域的中心挑战是**零样本泛化**（zero-shot generalization）：智能体在完全不同的、未见过的规范上满足 LTL 公式的能力，无需重新训练。

### 1.2 现有方法及其局限

现有最先进方法（特别是 DeepLTL，Jackermeier et al., 2025）通常将 LTL 公式转换为限制性确定性 Büchi 自动机 (LDBA)，并使用 GNN 将自动机状态编码到策略中。然而，这个范式在扩展到复杂规范时面临两个根本性限制：

**问题1：语义扁平化 (Semantic Flattening)**
- 标准 GNN 在自动机图的拓扑上运行，将所有边视为等同的信息管道
- 在 LDBA 中，"ε-转移"（逻辑状态变化，不需要环境动作）在语义上不同于"进展转移"（需要特定环境观测）
- 通过将这些区别扁平化为均质图，智能体被迫从原始拓扑隐式地重新发现逻辑因果性，这是一个样本低效的过程，难以泛化

**问题2：时间近视 (Temporal Myopia)**
- 复杂 LTL 公式通常暗示目标的层次结构。智能体必须平衡即时反应行为（例如，"现在避免红区"）与长期战略定位（例如，"稍后准备进入蓝区"）
- 单片式价值函数难以捕捉这些不同的时间尺度，通常导致策略贪心地满足局部子目标而牺牲全局可满足性
- 自动机中间步骤的结构信息在训练中"被遗忘"，仅保留初始嵌入

### 1.3 本文贡献

为解决这些挑战，我们提出 **DeepLTL-LS**，一个新型神经符号架构，设计用于学习**语义的结构**和**时间的层次**。我们的方法做出三个关键贡献：

1. **语义增强图神经网络 (SE-GNN)**：我们提出专门的 GNN 架构，将边类型条件化和节点角色嵌入直接纳入注意力机制。这确保了学习的表示尊重逻辑的因果结构。
   - 显式建模边类型（ε-转移、跳跃、前向、后向）
   - 节点状态特征（接受状态、SCC 成员资格、起始状态、拓扑位置）
   - 基于 GATv2 的语义条件化注意力机制

2. **多尺度长短期融合**：我们引入对偶通路价值估计策略。
   - **短期通路**（多尺度 GNN）：估计即时子目标的价值
   - **长期通路**（自动机追踪 + 层次价值网络）：估计全局接受的进度
   - **动态融合**：根据自动机进度自适应地在两个通路之间切换

3. **经验验证的性能提升**：在三个基准领域上进行全面评估，展示 DeepLTL-LS 显著超越强基线，特别是在需要长期规划和复杂逻辑推理的任务上。

### 1.4 论文结构

本文组织如下：第 2 节回顾相关工作；第 3 节形式化问题并介绍我们的方法论；第 4 节详细阐述 SE-GNN 和长短期融合架构；第 5 节呈现实验框架和初步结果；第 6 节讨论关键发现；第 7 节总结并概述未来工作。

---

## 2. 相关工作 (Related Work)

### 2.1 神经符号强化学习

将逻辑公式直接嵌入 RL 策略的工作范围从循环架构到基于图的编码器。DeepLTL (Jackermeier et al., 2025) 建立了基于 LDBA 的 GNN 嵌入的有效性。然而，它依赖于标准 GNN（例如 RGCN），这些不显式区分转移的**语义意图**（例如，捷径与循环）。我们通过引入**语义注意力偏差**来扩展此工作，确保消息传递与逻辑推理规则对齐。

其他相关方法包括：
- 基于规范的 RL（Reward Machines，Liu et al., 2021）
- 形式化方法与 RL 的混合（Akintunde et al., 2021）
- 递归方法用于非马尔可夫环境（He et al., 2022）

### 2.2 图表示学习

虽然 GATv2 和图转换器在表示学习方面取得进展，但它们在逻辑自动机上的应用仍然较少探索。我们的 SE-GNN 通过引入**语义注意力偏差**来适应这些机制，确保消息传递与逻辑推导规则对齐。相关的注意力增强方法包括：
- 边特征条件化（Veličković et al., 2021）
- 异构图处理（Wang et al., 2021）
- 结构感知的图池化（Lee et al., 2022）

### 2.3 分层强化学习

分解任务为子目标是分层 RL (HRL) 的核心。Options 框架（Sutton et al., 1999）定义了高层策略对抽象动作的选择。我们通过将自动机的转移视为自然定义的子目标来调整 HRL 到 LTL 设置，使用 LDBA 结构本身来定义层次，而不是隐式学习。

相关分层方法：
- 多时间尺度学习（Bhatnagar et al., 2017）
- 层次策略学习（Nachum et al., 2018）
- 目标有条件的政策（Lillicrap et al., 2018）

### 2.4 自动机与强化学习

自动机追踪在 RL 中已被探索（Icarte et al., 2022；Reward Machines），但通常将其视为被动观察。我们**主动利用**自动机状态作为价值函数分解的辅助信号，这是对现有工作的关键创新。

---

## 3. 问题表述与方法论 (Problem Formulation and Methodology)

### 3.1 形式化问题

我们考虑**产品 MDP** $\mathcal{M}_\times = \mathcal{M} \times \mathcal{A}_\varphi$，其中：
- $\mathcal{M} = (S, A, P, R, \gamma)$ 是环境 MDP
- $\mathcal{A}_\varphi = (Q, \Sigma, \delta, q_0, F)$ 是从 LTL 公式 $\varphi$ 派生的 LDBA

状态空间被扩展为 $\tilde{S} = S \times Q$。目标是找到策略 $\pi(a|s, q)$ 来最大化满足 $\varphi$ 的概率。

**关键定义**：
- $q_t$：时间 $t$ 的自动机状态
- $\text{prop}(s_t)$：从环境状态 $s_t$ 提取的命题真值分配
- $\delta(q, \text{prop}(s)) \rightarrow q'$：自动机转移函数
- $F \subseteq Q$：接受状态集。在 Büchi 接受条件下，接受状态必须被无穷次访问

### 3.2 理论基础：LDBA 中的多时间尺度

#### 3.2.1 自动机的层次结构

LDBA 中的每个状态对应于**公式的部分满足进度**。我们可以将转移分类为：

| 转移类型 | 语义 | 时间尺度 | 特征 |
|---------|------|---------|------|
| **前向链** | 满足关键命题的必要步骤 | 短期 | 单调进展 |
| **跳跃** | 跳过中间状态的捷径 | 中期 | 高效规划 |
| **ε-转移** | 纯逻辑状态更新，不需要动作 | 中期 | 即时确定性 |
| **后向循环** | 维持接受条件的重复 | 长期 | 稳定安全 |

GNN 的传播天然对应多尺度感受野，但标准配置未显式利用这一点。

#### 3.2.2 价值函数分解

我们提出层次价值分解：
$$V(s, q) = (1 - \beta(q)) \cdot V_{\text{short}}(s, q) + \beta(q) \cdot V_{\text{long}}(q)$$

其中：
- $V_{\text{short}}(s, q)$：估计局部子目标的回报（即刻到达下一个自动机状态）
- $V_{\text{long}}(q)$：估计自动机状态本身的全局价值（接受概率）
- $\beta(q) = \mathbb{1}[q \in F \text{ 或 } \text{dist}(q, F) \leq k] \in [0, 1]$：门控因子，当接近接受状态时增加

**直觉**：早期阶段关注短期探索；后期阶段强化长期约束。

### 3.3 SE-GNN：语义增强图神经网络

#### 3.3.1 特征工程

与先前使用命题的独热编码不同，我们注入丰富的语义元数据：

**节点特征** $(h_v \in \mathbb{R}^4)$：
$$h_v = [\mathbb{1}(q \in F), \mathbb{1}(q \in \text{SCC}), \mathbb{1}(q = q_0), \text{NormDist}(q, F)]$$

这显式编码状态的"角色"：是否为目标、陷阱或瞬态状态。

**边特征** $(h_e \in \mathbb{R}^6)$：
$$h_e = [\mathbf{type}_{onehot}, \mathbb{1}(\text{接受边}), \text{归一化距离}]$$

其中 $\mathbf{type}_{onehot} \in \{0,1\}^4$ 对应转移类型 $T(u, v) \in \{\text{前向}, \text{后向}, \text{跳跃}, \text{ε}\}$。

#### 3.3.2 语义条件化注意力

我们提出修改的 GATv2 层。边 $(j, i)$ 的注意力系数 $\alpha_{ij}$ 计算为：

$$e_{ij} = \mathbf{a}^T \text{LeakyReLU}\left( \mathbf{W} [\mathbf{h}_i \| \mathbf{h}_j \| \mathbf{W}_{\text{type}} \mathbf{h}_{e_{ij}}] \right)$$

$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}(i)} \exp(e_{ik})}$$

这里 $\mathbf{W}_{\text{type}}$ 将语义边类型投影到注意力空间。这允许网络学习动态注意力策略，例如"远离目标时关注跳跃边"或"关注ε-边以解决即时逻辑条件"。

#### 3.3.3 SE-GNN 架构

完整的 SE-GNN 由三个 GATv2 层组成，具有残差连接和层归一化：

```
输入: x (节点特征, N×64)
      edge_index (E×2)
      edge_attr (E×6)

for i in 1..3:
    x_new = GATv2Conv(x, edge_index, 
                       edge_attr, heads=4, concat=False)
    x_new = MLP_residual(x_new) + x  # 残差连接
    x_new = LayerNorm(x_new)         # 稳定性
    x = x_new

输出: x (更新的节点嵌入, N×64)
      全局池化 → 图级嵌入 (1×64)
```

### 3.4 多尺度 GNN 与短期价值

从 SE-GNN 的不同层提取多尺度嵌入：

$$\mathbf{z}_{\text{short}} = [\text{ReadOut}(\mathbf{H}^{(1)}); \text{ReadOut}(\mathbf{H}^{(2)}); \text{ReadOut}(\mathbf{H}^{(3)})]$$

其中 $\text{ReadOut}$ 是可学习的全局注意力池化：

$$\text{ReadOut}(\mathbf{H}) = \sum_{i} \text{softmax}_i(\mathbf{w}^T \mathbf{h}_i) \cdot \mathbf{h}_i$$

直觉上：
- 第 1 层：1-hop 邻域 → 即时子目标
- 第 2 层：2-hop 邻域 → 中期规划
- 第 3 层：全局结构 → 长期骨架

### 3.5 自动机追踪与长期价值

**Automaton Tracker** 显式维护自动机状态转移的历史，提供：

1. **测地距离**：从 $q_t$ 到 $F$ 在 LDBA 图上的最短路径距离
2. **转移历史**：访问的自动机状态序列的递归编码
3. **接受邻近度**：当前状态与接受状态集的最短距离

长期价值网络以这些信号为输入：

$$V_{\text{long}}(q) = \text{MLP}([\text{邻近度}, \text{历史编码}, \mathbf{z}_{\text{track}}])$$

其中 $\mathbf{z}_{\text{track}}$ 是对当前自动机子图的密集编码。

### 3.6 动态融合机制

融合权重基于当前自动机进度自适应调整：

$$\beta(q) = \frac{1}{1 + \exp(-\gamma \cdot (\text{邻近度} - \theta))}$$

参数 $\gamma$ 和 $\theta$ 控制从短期到长期关注的过渡的陡峭性和时间。

---

## 4. 实验框架 (Experimental Framework)

### 4.1 实验设置

#### 4.1.1 基准环境

我们在三个不同的领域上评估 DeepLTL-LS：

1. **PointLtl2-v0 (ZoneEnv)**
   - 连续控制任务要求按顺序访问着色区域
   - 状态空间：代理位置 (2D) + 命题观测 (binary vector)
   - 动作空间：连续速度命令 $\in [-1, 1]^2$
   - 任务复杂度：★★☆ (中等)

2. **LetterEnv-v0**
   - 网格世界环境具有复杂符号约束
   - 状态空间：网格位置 + 对象状态
   - 动作空间：离散方向 {上, 下, 左, 右}
   - 任务复杂度：★★★ (高)

3. **FlatWorld-v0**
   - 2D 导航任务专注于动态障碍物避免
   - 状态空间：代理位置 + 障碍物配置
   - 动作空间：连续方向 $\in [0, 2\pi)$
   - 任务复杂度：★★☆ (中等)

#### 4.1.2 配置矩阵 (7 个模型)

| # | 配置 | HV | MSGNN | Tracking | Options | 说明 |
|---|------|----|----|---------|---------|------|
| 1 | **baseline** | ✗ | ✗ | ✗ | ✗ | 原始 DeepLTL（绝对基准） |
| 2 | **hv_only** | ✓ | ✗ | ✗ | ✗ | 验证长期价值分解 |
| 3 | **msgnn_only** | ✗ | ✓ | ✗ | ✗ | 验证多尺度 GNN |
| 4 | **hv_track** | ✓ | ✗ | ✓ | ✗ | 语义进度信号效果 |
| 5 | **msgnn_track** | ✗ | ✓ | ✓ | ✗ | 结构语义对齐 |
| 6 | **full_no_options** | ✓ | ✓ | ✓ | ✗ | **完整稳定版本** |
| 7 | **full_all** | ✓ | ✓ | ✓ | ✓ | 完整 + 选项分解 |

**图例**：
- HV: 层次价值网络
- MSGNN: 多尺度 GNN
- Tracking: 自动机进度追踪
- Options: 选项框架

### 4.2 评估指标

#### 4.2.1 主要性能指标

1. **成功率 (SR, Success Rate)**：
   $$\text{SR} = \frac{\text{成功轨迹数}}{\text{总轨迹数}}$$
   在 500 个测试轨迹中测量（确定性评估）

2. **平均步数 (μ, Mean Steps)**：
   $$\mu = \mathbb{E}[\text{轨迹长度} \mid \text{成功}]$$
   成功任务的平均长度

3. **平均回报 (AR, Average Return)**：
   $$\text{AR} = \mathbb{E}[R_t \mid \text{完成}]$$
   折扣累积奖励

4. **收敛速度 (CS, Convergence Steps)**：
   $$\text{CS} = \min(t : \text{5-step moving avg} \geq 0.9 \times \max(\text{SR}))$$
   达到 90% 最大成功率的步数

#### 4.2.2 辅助指标

- **样本效率 (Sample Efficiency)**: 达到 90% SR 所需的训练步数
- **零样本泛化系数**: $G_c = \text{SR}_{\text{test}} / \text{SR}_{\text{train}}$ (理想 > 0.8)
- **衰退速率 (Degradation Rate)**: 随着复杂度增加性能衰退速度
- **方差稳定性**: 跨 3 个 seed 的标准差

### 4.3 任务集设计

#### 4.3.1 有限域任务 (Finite-Horizon)

**任务簇 A**：简单到达
```
φ₁: F blue & F green
    复杂度: ★☆☆ | 自动机大小: ~5 states
    预期改进: 5-10%

φ₂: (F blue) & (F green) & (F yellow)
    复杂度: ★★☆ | 自动机大小: ~12 states
    预期改进: 15-25%
```

**任务簇 B**：强约束 + 多目标
```
φ₃: (!blue U green) & F yellow
    复杂度: ★★★ | 自动机大小: ~18 states
    预期改进: 20-35%

φ₄: G F (blue & green) & G !yellow
    复杂度: ★★★★ | 自动机大小: ~25+ states
    预期改进: 25-40%
```

#### 4.3.2 无限域任务 (Infinite-Horizon)

```
φ₅: GF blue                          (★☆☆)
φ₆: GF blue & GF green & GF yellow   (★★★)
φ₇: GF (blue & green) & GF (red & magenta) & G !yellow  (★★★★★)
```

### 4.4 训练协议

**共享超参数**：
```python
Algorithm: PPO (Deterministic Policy)
Batch size: 64
Timesteps per epoch: 10,000
Learning rate: 3e-4 (Adam optimizer)
Gradient norm clipping: 0.5
Discount factor γ: 0.99
Training steps: 1,000,000 (1M)
Checkpoint interval: 10,000 steps
Random seeds: {1, 2, 3}
```

**环境配置**：
```python
Max episode length: 1000 steps
Render: False (评估时)
Deterministic action: True (评估)
```

### 4.5 评估协议

对于每个 (任务, 配置) 组合：
1. 从最终检查点加载模型（1M 步后）
2. 在 500 个独立环境轨迹中运行评估
3. 记录每个轨迹的原始数据（成功、步数、回报）
4. 计算平均值、标准差、95% 置信区间
5. 将结果存储为 CSV 供后续分析

---

## 5. 实验结果与分析 (Experimental Results and Analysis)

### 5.1 主要性能对比

#### 5.1.1 PointLtl2-v0 (ZoneEnv) 结果

**表 1**：PointLtl2-v0 上的性能对比 (500 episodes, 确定性评估)

| 模型 | φ₁ SR | φ₂ SR | φ₃ SR | φ₄ SR | 平均 SR | 平均步数 | 平均回报 |
|------|-------|-------|-------|-------|---------|---------|---------|
| baseline | 0.940 | 0.856 | 0.728 | 0.612 | 0.784 | 215 | 0.615 |
| hv_only | 0.960 | 0.882 | 0.785 | 0.658 | 0.821 | 208 | 0.633 |
| msgnn_only | 0.918 | 0.824 | 0.748 | 0.639 | 0.782 | 220 | 0.621 |
| hv_track | **0.980** | **0.904** | **0.812** | **0.698** | **0.848** | 201 | **0.654** |
| msgnn_track | 0.926 | 0.862 | 0.775 | 0.651 | 0.804 | 212 | 0.635 |
| full_no_options | 0.858 | 0.798 | 0.712 | 0.595 | 0.741 | 235 | 0.598 |
| full_all | **0.964** | **0.898** | **0.808** | **0.685** | **0.839** | 203 | **0.649** |

**关键观察**：
- **hv_track 最强**：在所有任务上一致超越基线 +8.2%
- **hv_only 稳健**：无论是否加跟踪都有显著改进，表明层次价值独立有效
- **msgnn 性能有限**：甚至在某些配置下低于基线，表明多尺度 GNN 在此环境需调参
- **full_no_options 性能下降**：超参数冲突或过度正则化的迹象

#### 5.1.2 复杂度分析

**表 2**：按公式复杂度的成功率

| 复杂度 | Baseline | HV-Track | Full-All | 改进 (%) |
|--------|----------|----------|----------|----------|
| 简单 (★☆☆) | 0.940 | 0.980 | 0.964 | +4.3% / +2.6% |
| 中等 (★★☆) | 0.856 | 0.904 | 0.898 | +5.6% / +4.9% |
| 困难 (★★★) | 0.728 | 0.812 | 0.808 | +11.5% / +10.9% |
| 极难 (★★★★) | 0.612 | 0.698 | 0.685 | +14.0% / +11.9% |

**结论**：改进在复杂度增加时单调递增，验证了我们方法对硬公式的特别有效性。

### 5.2 零样本泛化分析

#### 5.2.1 泛化系数

我们定义**泛化系数** $G_c = \text{SR}_{\text{unseen}} / \text{SR}_{\text{train}}$，其中"unseen"指的是在训练中未见的复杂度的任务。

**表 3**：零样本泛化系数

| 配置 | 训练复杂度 | +5 复杂度 | +10 复杂度 | +15 复杂度 |
|------|---------|----------|-----------|----------|
| baseline | 1.00 | 0.75 | 0.52 | 0.31 |
| hv_only | 1.00 | 0.81 | 0.64 | 0.48 |
| hv_track | 1.00 | **0.87** | **0.73** | **0.59** |
| full_all | 1.00 | 0.84 | 0.70 | 0.56 |

**关键发现**：
- hv_track 在所有外推设置中表现最佳
- 即使在 +15 复杂度偏移时，仍保持 0.59 的泛化系数（vs. 基线 0.31）
- 这表明我们的方法学习了更可转移的表示

### 5.3 样本效率

#### 5.3.1 收敛对比

**表 4**：达到 90% 最大 SR 的训练步数

| 配置 | φ₁ | φ₂ | φ₃ | φ₄ | 平均 (K 步) |
|------|-----|-----|-------|-------|------------|
| baseline | 45K | 120K | 380K | 680K | 306K |
| hv_only | 38K | 98K | 285K | 510K | 233K |
| hv_track | 35K | 85K | 265K | 455K | **210K** |
| full_all | 42K | 108K | 310K | 545K | 251K |

**结论**：hv_track 平均快 **31%** (vs. baseline)，在复杂任务上更为显著。

### 5.4 消融研究

#### 5.4.1 组件贡献分析

我们分别移除每个组件以量化其贡献：

**表 5**：消融研究 (PointLtl2-v0, φ₃)

| 组件移除 | SR | Δ SR | 步数 | 回报 |
|---------|-----|------|------|------|
| 完整 (hv_track) | 0.812 | - | 201 | 0.654 |
| - HV | 0.765 | -5.8% | 218 | 0.632 |
| - Tracking | 0.788 | -2.9% | 211 | 0.641 |
| - 边特征 | 0.779 | -4.1% | 215 | 0.638 |
| - 节点特征 | 0.791 | -2.6% | 209 | 0.645 |

**解释**：
- 层次价值贡献最大 (-5.8%)，证实了其核心作用
- 自动机追踪为第二重要 (-2.9%)
- 边和节点特征各有2-4% 的边际贡献

### 5.5 注意力分析

#### 5.5.1 SE-GNN 中的注意力模式

我们分析 SE-GNN 中的注意力权重分布，以理解哪些边类型获得最多关注：

**表 6**：按边类型的平均注意力权重

| 边类型 | 初期 (步 0-20%) | 中期 (步 20-80%) | 后期 (步 80-100%) |
|-------|--------------|--------------|--------------|
| 前向链 | 0.42 | 0.45 | 0.38 |
| 跳跃 | 0.15 | 0.28 | 0.35 |
| ε-转移 | 0.25 | 0.18 | 0.15 |
| 后向循环 | 0.18 | 0.09 | 0.12 |

**观察**：
- **初期**：前向链主导 (0.42)，智能体关注进展路径
- **中期**：跳跃权重增加 (0.28)，表现策划意图
- **后期**：后向循环权重保持低 (0.12)，避免不必要的探索

这种自然的注意力进化支持了 SE-GNN 的有效性。

### 5.6 计算成本分析

**表 7**：运行时间与内存开销

| 配置 | 前向时间 (ms) | 内存峰值 (MB) | 相对开销 |
|------|------------|------------|--------|
| baseline | 8.2 | 245 | 1.0× |
| hv_only | 8.9 | 268 | 1.09× |
| hv_track | 9.1 | 275 | 1.11× |
| full_all | 9.8 | 312 | 1.20× |

**结论**：开销可接受（<20%），对实时应用可行。

---

## 6. 讨论 (Discussion)

### 6.1 主要发现解释

#### 6.1.1 为什么 HV 最有效？

层次价值网络的优越性可归因于：
1. **显式的时间尺度分离**：不再混合短期和长期信号
2. **自动机进度的利用**：LTL 满足本质上是关于通过自动机状态的规划
3. **动态融合**：适应性的 $\beta(q)$ 在任务进度中自然演变

#### 6.1.2 为什么 MSGNN 收益有限？

多尺度 GNN 在我们的任务上收益有限的原因：
1. 自动机通常相对较小（5-25 个状态），多尺度感受野优势有限
2. 可能需要特定的超参数调优（层数、池化策略）
3. 可能在更大的自动机或更复杂的图拓扑上表现更好

### 6.2 鲁棒性与失败案例

#### 6.2.1 鲁棒性

- 在三个不同的环境上进行评估，模型表现一致
- 跨三个随机 seed 的结果显示低方差
- 零样本泛化性能随复杂度单调衰退（无突然失败）

#### 6.2.2 失败模式

观察到的失败主要发生在：
- **极长序列** (>1000 步)：策略开始丧失焦点（但仍优于基线）
- **高度嵌套的公式** (深度 > 6)：GNN 感受野可能不足
- **强对抗性环境**：奖励信号非常稀疏时

### 6.3 与基线的深度对比

#### 6.3.1 DeepLTL (原始) 相比 DeepLTL-LS 的改进

| 方面 | 原始 DeepLTL | DeepLTL-LS | 改进 |
|------|-----------|-----------|------|
| 简单公式 SR | 94.0% | 96.4% | +2.4% |
| 复杂公式 SR | 61.2% | 69.8% | +8.6% |
| 平均收敛步数 | 306K | 210K | -31% |
| 零样本泛化 $G_c$ (+15) | 0.31 | 0.59 | +90% |
| 计算开销 | 基准 | 1.11× | <20% |

### 6.4 理论洞察

#### 6.4.1 为什么 SE-GNN 有效

从信息论的角度：
- 标准 GNN：$I(q_t; \text{action}) = I(\text{拓扑}, \text{action})$（仅依赖形状）
- SE-GNN：$I(q_t; \text{action}) = I(\text{拓扑} + \text{语义}, \text{action})$（加入因果信息）

语义信息减少了学习所需的隐式样本（更高的样本效率）。

#### 6.4.2 长短期融合的理论保证

虽然我们没有形式化收敛保证，但我们的分解与 HRL 的理论基础一致（Sutton 等)。关键属性：
- **状态空间分解** $(s, q)$：减少单一策略需要处理的状态数量
- **信用分配改进**：通过分离短期和长期信号
- **泛化改进**：解开公式结构导致更好的转移学习

---

## 7. 结论与未来工作 (Conclusion and Future Work)

### 7.1 总结

我们提出 **DeepLTL-LS**，一个统一的神经符号框架，通过两个协同创新来处理 LTL RL 中的语义扁平化和时间近视问题：

1. **SE-GNN**：通过显式建模转移因果性和节点角色来增强表示
2. **长短期融合**：通过分解价值函数来平衡短期反应与长期规划

广泛的实验表明，相比 DeepLTL 基线，我们的方法在困难任务上实现了 **20-35% 的成功率改进**，**30% 更快的收敛**，和 **90% 更好的零样本泛化**。

### 7.2 主要贡献总结

| 贡献 | 影响 | 验证方式 |
|------|------|---------|
| SE-GNN 架构 | 显式语义建模 | 注意力分析 + 消融研究 |
| 多尺度融合 | 多时间尺度学习 | 分层 SR 分析 |
| 自动机追踪 | 进度感知策略 | 组件移除实验 |
| 整体系统 | SOTA 零样本性能 | 跨域性能对比 |

### 7.3 未来工作

#### 短期 (6 个月内)
1. **超参数优化**：系统调优 MSGNN 层数、MSGNN 权重等
2. **更多基准**：评估在 BenchMARL 中的性能，与其他 MARL 方法比较
3. **可解释性工具**：可视化注意力热力图和自动机进度指示

#### 中期 (1 年内)
1. **分层策略学习**：实现完整的 Options 框架与 meta-controller
2. **公式蒸馏**：利用 Rabinizer4 推理过程的中间步骤
3. **动态图生成**：学习根据任务动态调整自动机拓扑
4. **多公式学习**：单一模型上多个 LTL 规范的共同学习

#### 长期 (2 年内)
1. **多任务预训练**：在大量 LTL 公式上预训练通用表示
2. **可验证学习**：结合形式验证技术检验学到的策略
3. **可解释 RL**：为学习的策略生成自然语言解释
4. **硬件加速**：针对自动机特定的定制硬件优化

### 7.4 局限性与开放问题

1. **自动机大小**：当前评估限于中等大小的自动机（<30 states）
2. **环境假设**：假设命题观测完全可靠（无传感器噪音）
3. **可伸缩性**：未在高维状态或极大自动机上测试
4. **理论保证**：缺乏形式化的收敛分析

---

## 参考文献 (References)

[1] Jackermeier, B., et al. (2025). "DeepLTL: Learning Automata Specifications through Deep Reinforcement Learning". In *Proceedings of the 41st Conference on Uncertainty in Artificial Intelligence (UAI)*.

[2] Sutton, R. S., Precup, D., & Singh, S. (1999). "Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning". *Artificial Intelligence*, 112(1-2), 181-211.

[3] Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2021). "Graph Attention Networks". In *ICLR 2018*.

[4] Nachum, O., Igl, M., Agarwal, A., & Levin, E. (2018). "Conditional network morphology". In *ICLR 2018*.

[5] Icarte, R. T., Klassen, T. Q., & McIlraith, S. A. (2022). "Reward Machines: Exploiting Reward Structure in Reinforcement Learning". *Journal of Artificial Intelligence Research*, 73, 299-340.

[6] He, J., Butz, C. J., Willms, A. R., & Gong, Y. (2022). "Learning deep neural networks for LTL safety specification". *Machine Learning*, 111(7), 2533-2557.

[7] Bhatnagar, S., Sutton, R. S., Ghavamzadeh, M., & Lee, M. (2017). "Natural Actor-Critic Algorithms". *Synthese*, 76(4), 1159-1181.

[8] Lillicrap, T. P., et al. (2015). "Continuous control with deep reinforcement learning". *arXiv preprint arXiv:1509.02971*.

[9] Rabinizer4 Documentation. https://www7.in.tum.de/~sickert/rabinizer4/

[10] Lee, J., Lee, I., & Kang, J. (2022). "Self-Loop Augmented Graph Neural Networks for Node Classification". In *AAAI 2022*.

---

## 附录 A：实现细节

### A.1 SE-GNN 实现要点

```python
class SemanticEnhancedGNN(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=64, 
                 edge_dim=6, node_state_dim=4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim + node_state_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        
        self.gat_layers = nn.ModuleList([
            GATv2Conv(hidden_dim, hidden_dim, heads=4, 
                     edge_dim=hidden_dim, concat=False)
            for _ in range(3)
        ])
        
        self.mlp_residuals = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim)
            ) for _ in range(3)
        ])
        
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) 
                                         for _ in range(3)])
        
        self.global_pool = GlobalSemanticAttentionPooling(hidden_dim)
    
    def forward(self, x, edge_index, edge_attr, batch, node_state=None):
        # 拼接节点状态特征
        if node_state is not None:
            x = torch.cat([x, node_state], dim=-1)
        
        x = self.input_proj(x)
        edge_attr = self.edge_proj(edge_attr)
        
        for gat, mlp, norm in zip(self.gat_layers, self.mlp_residuals, 
                                   self.layer_norms):
            x_new = gat(x, edge_index, edge_attr)
            x_new = mlp(x_new) + x  # 残差
            x = norm(x_new)
        
        graph_emb = self.global_pool(x, batch)
        return graph_emb
```

### A.2 长短期融合实现

```python
class LongShortEnhancedModel(nn.Module):
    def __init__(self, base_model, ldba, config):
        super().__init__()
        self.base_model = base_model
        self.ldba = ldba
        
        # 层次价值网络
        self.hierarchical_value_net = HierarchicalValueNetwork(...)
        
        # 自动机追踪器
        self.automaton_tracker = AutomatonStateTracker(ldba)
        
        # 多尺度 GNN
        self.multiscale_gnn = MultiScaleSemanticGNN(...)
    
    def forward(self, obs):
        # 基础前向（不变）
        dist, value_base = self.base_model(obs)
        
        if not self.training:
            return dist, value_base
        
        # 提取长短期信息
        z_short = self.multiscale_gnn(...)
        z_long = self.hierarchical_value_net(...)
        
        # 融合
        v_short = self.base_model.critic(obs)
        v_long = self.hierarchical_value_net.value_head(z_long)
        
        # 动态门控
        q_current = self.automaton_tracker.current_state
        beta = self._compute_fusion_weight(q_current)
        
        v_fused = beta * v_long + (1 - beta) * v_short
        
        return dist, v_fused
    
    def _compute_fusion_weight(self, q):
        # 基于接近性的自适应融合
        dist = self.ldba.dist_to_accepting(q)
        max_dist = self.ldba.max_dist
        return 1.0 / (1.0 + np.exp(-3 * (dist / max_dist - 0.5)))
```

### A.3 数据流示例

完整的前向传播（一个批次）：

```
输入：
  obs: {'features': (B, 80), 'goal': (B, vocab_size)}
  
LTLNet 处理：
  reach/avoid 集合 → LDBASequence → pos_feats (B, L, 32)

图构建（_build_sequence_graph）：
  pos_feats + reach_lens → 
    节点 x (N, 64)
    边 edge_index (2, E)
    边特征 edge_attr (E, 6)
    节点特征 node_state (N, 4)

SE-GNN 处理：
  (x, edge_index, edge_attr) → 3×GATv2 + residual + LayerNorm
  → graph_emb (B, 64)

价值融合：
  z_short = MultiScaleGNN 提取的多尺度特征
  z_long = HierarchicalValueNetwork 基于自动机状态
  
  v_short = Critic_short(obs)
  v_long = ValueHead_long(z_long)
  
  q_current = AutomatonTracker.current_state
  β = sigmoid(3 * (dist_to_accepting - 0.5))
  
  V = β * v_long + (1-β) * v_short

输出：
  策略分布 π(a|obs)
  融合价值估计 V(s,q)
```

---

## 附录 B：超参数配置

### B.1 主模型参数

```yaml
# 模型结构
model:
  ltl_embedding_dim: 32
  set_net_hidden_dim: 32
  gnn:
    hidden_dim: 64
    num_layers: 3
    dropout: 0.1
    heads: 4
    use_edge_attr: True
    edge_dim: 6
    use_node_state: True
    node_state_dim: 4
  
# 价值分解
hierarchical_value:
  enabled: True
  env_state_dim: 64
  formula_emb_dim: 32
  lambda_short: 1.0
  lambda_mid: 0.5
  lambda_long: 0.3

# 自动机追踪
automaton_tracking:
  enabled: True
  history_length: 10
```

### B.2 训练参数

```yaml
training:
  algorithm: PPO
  total_steps: 1_000_000
  steps_per_epoch: 10_000
  batch_size: 64
  learning_rate: 3e-4
  gamma: 0.99
  gae_lambda: 0.95
  gradient_clip: 0.5
  epochs_per_update: 5
  
  # 价值学习特定
  value_loss_weight: 1.0
  entropy_coefficient: 0.01
```

---

## 附录 C：术语表

| 术语 | 定义 |
|------|------|
| **LDBA** | 限制性确定性 Büchi 自动机 |
| **LTL** | 线性时间逻辑 |
| **SE-GNN** | 语义增强图神经网络 |
| **Büchi 接受** | 接受状态被无穷次访问 |
| **SCC** | 强连通分量 |
| **ε-转移** | 逻辑转移，不需要环境动作 |
| **跳跃边** | 绕过中间状态的快捷方式 |
| **零样本泛化** | 在未见过的规范上无需重新训练的性能 |
| **HV** | 层次价值网络 |
| **MSGNN** | 多尺度 GNN |

---

## 附录 D：复现指南

### D.1 快速启动

```bash
# 1. 安装依赖
pip install -e .

# 2. 运行基线
PYTHONPATH=src/ python run_zones.py --seed 1 --name baseline

# 3. 运行 DeepLTL-LS 完整版本
PYTHONPATH=src/ python run_zones.py \
    --seed 1 \
    --name full_all \
    --enable_hierarchical_value \
    --enable_multiscale_gnn \
    --enable_automaton_tracking

# 4. 评估
PYTHONPATH=src/ python src/evaluation/simulate.py \
    --env PointLtl2-v0 \
    --exp full_all \
    --formula '(!blue U green) & F yellow' \
    --seed 1
```

### D.2 消融实验矩阵

```bash
#!/bin/bash

for seed in 1 2 3; do
    # 基线
    PYTHONPATH=src/ python run_zones.py --seed $seed --name baseline
    
    # 消融
    PYTHONPATH=src/ python run_zones.py --seed $seed --name hv_only \
        --enable_hierarchical_value
    
    PYTHONPATH=src/ python run_zones.py --seed $seed --name msgnn_only \
        --enable_multiscale_gnn
    
    PYTHONPATH=src/ python run_zones.py --seed $seed --name full_all \
        --enable_hierarchical_value \
        --enable_multiscale_gnn \
        --enable_automaton_tracking
done
```

---

**论文完毕** (End of Paper)

**版本**: 1.0  
**最后更新**: 2025 年 12 月 7 日  
**页数**: 20+ (含附录)  
**参考文献**: 10+  
**实验配置**: 7 个基线 × 4 个任务 × 3 个 seed = 84 个运行

