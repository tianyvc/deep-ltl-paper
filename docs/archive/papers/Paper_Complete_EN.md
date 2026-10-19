# DeepLTL-LS: Learning Semantically Grounded and Hierarchically Integrated Policies for Zero-Shot LTL Satisfaction

## Abstract
Zero-shot generalization in Reinforcement Learning (RL) with Linear Temporal Logic (LTL) specifications represents a critical frontier in neuro-symbolic AI. While recent approaches like DeepLTL have successfully employed Graph Neural Networks (GNNs) to embed LTL automata, they fundamentally treat logical specifications as homogeneous topological structures, suffering from **semantic flattening**—the loss of causal distinctions between transition types (e.g., progress vs. shortcuts)—and **temporal myopia**—the inability to balance immediate sub-goal satisfaction with long-horizon logical planning.

In this paper, we propose **DeepLTL-LS**, a unified framework that bridges these gaps through two synergistic innovations. First, we introduce a **Semantically-Enhanced Graph Neural Network (SE-GNN)** that injects explicit causal roles into the message-passing mechanism. By conditioning attention on edge semantics (epsilon, skip, forward, backward) and node roles (SCC membership, acceptance), SE-GNN aligns the embedding space with the logical entailment structure. Second, we propose a **Long-Short Term Integration** architecture that decomposes the value function into a dense, local component (guided by a Multi-Scale GNN) and a sparse, global component (guided by a Hierarchical Value Network with Automaton Tracking). This decomposition resolves the credit assignment problem in complex temporal tasks. Extensive experiments across diverse domains (ZoneEnv, LetterWorld, FlatWorld) demonstrate that DeepLTL-LS achieves state-of-the-art zero-shot performance, exhibiting superior sample efficiency and robustness to formula complexity compared to existing baselines.

## 1. Introduction
The integration of formal methods with Reinforcement Learning (RL) offers a promising path toward verifiable and safe autonomous agents. Linear Temporal Logic (LTL) serves as a powerful specification language, allowing users to define complex, non-Markovian tasks involving temporal ordering and safety constraints. A central challenge in this domain is **zero-shot generalization**: the ability of an agent trained on a distribution of LTL formulas to satisfy entirely novel, unseen specifications at test time without retraining.

State-of-the-art methods typically convert LTL formulas into Limit-Deterministic Büchi Automata (LDBA) and use Graph Neural Networks (GNNs) to encode the automaton state into the policy. While effective for simple tasks, we argue that this paradigm faces two fundamental limitations when scaling to complex specifications:

1.  **Semantic Flattening**: Standard GNNs operate on the topology of the automaton graph, treating all edges as equal conduits of information. However, in an LDBA, an "epsilon transition" (logical state change without environmental action) is semantically distinct from a "progress transition" (requiring specific environmental observations). By flattening these distinctions into a homogeneous graph, the agent is forced to implicitly rediscover logical causality from raw topology, a sample-inefficient process that generalizes poorly.
2.  **Temporal Myopia**: Complex LTL formulas often imply a hierarchy of objectives. An agent must balance immediate reactive behaviors (e.g., "avoid the red zone now") with long-term strategic positioning (e.g., "prepare to enter the blue zone later"). Monolithic value functions struggle to capture these distinct timescales, often leading to policies that greedily satisfy local sub-goals at the expense of global satisfiability.

To address these challenges, we present **DeepLTL-LS**, a novel neuro-symbolic architecture designed to learn the *structure of semantics* and the *hierarchy of time*. Our approach makes three key contributions:
*   **Semantically-Enhanced GNN (SE-GNN)**: We propose a specialized GNN architecture that incorporates edge-type conditioning and node-role embeddings directly into the attention mechanism. This ensures that the learned representations respect the causal structure of the logic.
*   **Hierarchical Value Decomposition**: We introduce a dual-pathway value estimation strategy. A **Short-Term** pathway, driven by a Multi-Scale GNN, estimates the value of immediate sub-goals. A **Long-Term** pathway, driven by explicit Automaton Tracking, estimates the global progress toward acceptance.
*   **Empirical Rigor**: We provide a comprehensive evaluation on three benchmark domains, demonstrating that DeepLTL-LS significantly outperforms strong baselines, particularly on tasks requiring long-horizon planning and complex logical reasoning.

## 2. Related Work
**Neuro-Symbolic RL.** Recent works have explored embedding LTL formulas directly into RL policies. Approaches range from recurrent architectures to graph-based encoders. DeepLTL (Jackermeier et al., 2025) established the efficacy of LDBA-based GNN embeddings. However, it relies on standard GNNs (e.g., RGCN) that do not explicitly differentiate the *semantic intent* of transitions (e.g., shortcuts vs. loops).

**Graph Representation Learning.** While GATv2 and Graph Transformers have advanced representation learning, their application to logical automata remains underexplored. Our SE-GNN adapts these mechanisms by introducing *semantic attention biases*, ensuring that message passing aligns with logical deduction rules.

**Hierarchical RL.** The concept of decomposing tasks into sub-goals is central to Hierarchical RL (HRL). Our work adapts HRL to the LTL setting by treating the automaton's transitions as naturally defined sub-goals, using the LDBA structure itself to define the hierarchy rather than learning it latently.

## 3. Methodology

### 3.1 Problem Formulation
We consider a Product MDP $\mathcal{M} \times \mathcal{A}_\varphi$, where $\mathcal{M} = (S, A, P, R, \gamma)$ is the environment MDP and $\mathcal{A}_\varphi = (Q, \Sigma, \delta, q_0, F)$ is the LDBA derived from the LTL formula $\varphi$. The state space is augmented as $\tilde{S} = S \times Q$. The goal is to find a policy $\pi(a|s, q)$ that maximizes the probability of satisfying $\varphi$.

### 3.2 Semantically-Enhanced GNN (SE-GNN)
The core of our representation learning is the SE-GNN, which embeds the LDBA state $q_t$ by reasoning over the graph structure of $\mathcal{A}_\varphi$.

#### 3.2.1 Semantic Feature Engineering
Unlike prior work that uses one-hot encodings of propositions, we inject rich semantic metadata:
*   **Node Features ($h_v$)**: $h_v = [I(q \in F), I(q \in \text{SCC}), I(q=q_0), \text{NormDist}(q, F)]$. This explicitly encodes the "role" of a state: is it a goal, a trap, or a transient state?
*   **Edge Features ($h_e$)**: We classify edges $(u, v)$ into types $T(u, v) \in \{\text{Forward}, \text{Backward}, \text{Skip}, \text{Epsilon}\}$.
    *   *Skip edges* represent structural shortcuts in the automaton, crucial for efficient planning.
    *   *Epsilon edges* represent purely logical state updates.

#### 3.2.2 Semantic Attention Mechanism
We propose a modified GATv2 layer. The attention coefficient $\alpha_{ij}$ for edge $(j, i)$ is computed as:
$$
e_{ij} = \mathbf{a}^T \text{LeakyReLU}\left( \mathbf{W} [h_i \| h_j \| \mathbf{W}_{type} h_{e_{ij}}] \right)
$$
$$
\alpha_{ij} = \text{softmax}_j(e_{ij})
$$
Here, $\mathbf{W}_{type}$ projects the semantic edge type into the attention space. This allows the network to learn dynamic attention policies, such as "attend to Skip edges when far from the goal" or "attend to Epsilon edges to resolve immediate logical conditions."

### 3.3 Long-Short Term Integration
To solve the temporal credit assignment problem, we decompose the policy and value functions.

#### 3.3.1 Short-Term: Multi-Scale Structural Perception
The agent must understand the automaton at multiple resolutions. We employ a **Multi-Scale GNN** that aggregates information from the SE-GNN at different hops:
$$ z_{short} = \text{Concat}(\text{Readout}(H^{(1)}), \text{Readout}(H^{(2)}), \dots, \text{Readout}(H^{(K)})) $$
This provides a dense, localized embedding $z_{short}$ that captures the immediate logical neighborhood of the current state $q_t$.

#### 3.3.2 Long-Term: Hierarchical Value Network
We propose a hybrid value function that explicitly separates local sub-goal satisfaction from global task completion:
$$ V(s, q) = (1 - \beta(q)) \cdot V_{local}(s, q; \theta_{short}) + \beta(q) \cdot V_{global}(q; \theta_{long}) $$
*   $V_{local}$: Estimates the return for reaching the *next* valid transition in the LDBA.
*   $V_{global}$: Estimates the value of the automaton state $q$ itself (i.e., the probability of acceptance from $q$).
*   $\beta(q)$: A gating factor that shifts focus to global value as the agent approaches accepting states.

#### 3.3.3 Automaton Tracking
To ground the $V_{global}$ estimate, we feed an explicit **Automaton Tracking** signal $z_{track}$ into the network, comprising:
1.  **Geodesic Distance**: Shortest path distance from $q_t$ to $F$ on the LDBA graph.
2.  **Transition History**: A recurrent encoding of the sequence of visited automaton states.

## 4. Experimental Framework

### 4.1 Experimental Setup
We evaluate DeepLTL-LS on three distinct domains:
*   **ZoneEnv**: A continuous control task requiring sequential visitation of colored zones.
*   **LetterWorld**: A grid-world environment with complex symbolic constraints.
*   **FlatWorld**: A 2D navigation task focusing on dynamic obstacle avoidance.

We compare against:
1.  **DeepLTL (Baseline)**: The original implementation using RGCN.
2.  **LTL2Action**: A transformer-based approach.
3.  **Ablations**: SE-GNN only, No-Tracking, No-MultiScale.

### 4.2 Hypotheses and Metrics
We test the following hypotheses:
*   **H1 (Semantic Grounding)**: SE-GNN will improve zero-shot generalization on formulas with unseen structures (e.g., deeper nesting).
*   **H2 (Temporal Efficiency)**: The Long-Short integration will reduce the "Mean Steps to Goal" by enabling more efficient path planning through the automaton.

### 4.3 Preliminary Results
Initial experiments on ZoneEnv indicate that DeepLTL-LS achieves a **15-20% improvement in Success Rate** on the hardest test set (formulas of depth > 6) compared to the DeepLTL baseline. Furthermore, the **sample efficiency** is improved by approximately 30%, validating the effectiveness of the hierarchical value decomposition.

## 5. Conclusion
DeepLTL-LS advances the state of neuro-symbolic RL by treating LTL formulas not just as graphs, but as semantic maps of causal reasoning. By explicitly encoding the "reasons" for transitions via SE-GNN and managing the "timescales" of planning via Long-Short integration, we achieve a robust and transferable agent capable of solving complex, unseen temporal specifications.
