# Learning LTL Structures of Semantics and Reasons for Zero-Shot Reinforcement Learning

> A semantically grounded graph neural framework for zero-shot generalization over temporal logic formulae. We learn not only the structure of the automaton but the reasons it exists.

---
## Abstract
Zero-shot generalization in Reinforcement Learning (RL) with Linear Temporal Logic (LTL) specifications demands an embedding that captures logical causality rather than surface topology. Prior approaches (e.g., DeepLTL) convert each formula into a Limit-Deterministic Büchi Automaton (LDBA) and apply a GNN over a flattened transition graph; this erases distinctions between progress vs. epsilon transitions, accepting vs. transient states, and structural shortcuts (skip edges). We propose a Semantically-Enhanced Graph Neural Network (SE-GNN) that injects explicit node-state and edge-type semantics into message passing. Nodes receive a 4-D role encoding (accepting, SCC membership, initiality, normalized position); edges receive a 6-D causal descriptor (type one-hot: forward/backward/skip/epsilon, accepting transition flag, distance). A semantic-aware GATv2 extends attention with edge attributes, prioritizing logically causal transitions. Experiments across Zones, Letter, and FlatWorld show improved zero-shot success, faster convergence, and reduced spurious exploration. Ablations confirm that performance gains stem from semantic structure, not model size. We argue explicit semantic factoring creates a transferable inductive bias for neuro-symbolic RL.

## 1. Introduction
Zero-shot LTL RL asks: can an agent satisfy previously unseen temporal formulas after training on related ones? A central failure mode of baseline embeddings is collapsing heterogeneous automaton dynamics into homogeneous graphs. Our thesis: preserving the *reasons* for transitions—epsilon acceleration, skip shortcuts, accepting state stabilization—enables a structural abstraction that transfers.

## 2. Related Work
- LTL for RL: shaping vs. automaton-guided planners; lack semantic role factoring.
- GNNs for logic/programs: seldom encode temporal stage roles explicitly.
- Neuro-symbolic alignment: we enrich symbolic traces instead of flattening.
- Edge-aware attention: we extend GATv2 with logical behavior types.

## 3. Problem Formulation
Given LTL tasks \(\varphi_i\), convert each to LDBA \(\mathcal{A}_\varphi\). Seek policy \(\pi(a|s,z_\varphi)\) where \(z_\varphi = f_{SE-GNN}(Q,E,F,\text{SCC},\text{Types},\text{Meta})\). Objective: maximize satisfaction probability on unseen \(\varphi'\).

## 4. Design Principles
1. Explicit semantics. 2. Safe degradation (zero-fill). 3. Per-dimension interpretability. 4. Vectorized O(L) construction. 5. Inductive bias via structural roles.

## 5. Architecture Overview
Pipeline: formula → Rabinizer4 → HOA → LDBA → sampled reach-avoid sequence → enriched LDBASequence(state/transition info) → BatchedReachAvoidSequences (preserve originals) → feature extraction → graph (pos feats + node_state + edge_attr) → multi-layer semantic GATv2 → global attention pooling → embedding.

### 5.1 Node Features
\(h_v = [accepting, in\_SCC, is\_initial, position\_norm]\); zero fallback.

### 5.2 Edge Features
\(h_e = [onehot(type), accepting\_trans, distance]\); types: forward, backward, skip, epsilon.

### 5.3 Semantic Attention
\(\alpha_{ij} = softmax_j( \sigma( a^T [W h_i || W h_j || U h_{e_{ij}}]))\). Multi-head shares edge projection; residual + layer normalization stabilize gradients.

### 5.4 Global Pooling
Learned query attention focuses on phase transitions (entering accepting SCC) rather than uniform averaging.

### 5.5 Complexity
Graph build O(L); edges ≈ 4(L-1)+|skip|; computational overhead <15% vs baseline.

## 6. Feature Robustness
Each dimension has singular semantics; zero-fallback ensures no crashes if metadata unavailable (training proceeds in compatibility mode).

## 7. Algorithm (Pseudo-code)
```
Input: LTL formula φ
A ← Rabinizer4(φ) → parse → LDBA
seq ← sample_sequence(A)
seq_enriched ← enrich(seq)  # state_info, transition_info
B ← batch([seq_enriched])
pos_feats ← LTLNet(reach/avoid embeddings)
G ← build_graph(pos_feats, B.sequences)
for layer in layers:
    h ← semantic_gatv2(h, edge_index, edge_attr)
z_φ ← attention_pool(h)
return z_φ
```

## 8. Theoretical Insights
- Edge-type conditioning reduces message ambiguity.
- SCC role clarifies long-horizon value propagation.
- Skip edges teach abstraction akin to hierarchical options.
- Epsilon modeling aligns logical and temporal steps, easing credit assignment.

## 9. Experimental Setup
Tasks across varied formula length & nesting. Metrics: zero-shot success, steps-to-satisfy, sample efficiency, attention semantic concentration.

## 10. Ablations
Removing edge attributes, node roles, epsilon typing, or SCC flags degrades different axes (shortcut utilization, stability, time alignment).

## 11. Discussion & Limitations
Metadata generation overhead; scaling to >500 states may require sparse or hierarchical pooling; attention interpretability tools pending.

## 12. Future Work
Hierarchical SCC pooling, semantic visualization dashboards, adaptive skip pruning, multimodal alignment with natural language, formal safety constraints.

## 13. Conclusion
SE-GNN reframes automata graphs as causal reasoning maps. Explicit semantic structuring yields transferable embeddings and strengthens zero-shot LTL RL. Structure alone is not enough; structure + reasons unlock abstraction.

## Appendix A: Semantic GATv2 Layer
```
def semantic_gatv2(h, edge_index, edge_attr):
    hi, hj = h[edge_index[0]], h[edge_index[1]]
    e_proj = U(edge_attr)
    cat = torch.cat([W(hi), W(hj), e_proj], dim=-1)
    scores = leaky_relu(a(cat))
    alpha = softmax(scores, edge_index[0])
    msg = V(hj) * alpha
    out = aggregate(msg, edge_index[0])
    return layer_norm(out + h)
```

## Appendix B: Metadata Enrichment
```
def enrich(ldba):
    sccs = strongly_connected_components(ldba.graph)
    state_info = [{
        'accepting': q in ldba.accepting,
        'in_scc': any(q in s and len(s)>1 for s in sccs),
        'is_initial': q == ldba.initial,
    } for q in ldba.states]
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

## BibTeX (Preprint Proposal)
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

For experiments and reproduction details see `SE_GNN_Experiments.md`.

---
(End of English draft)
