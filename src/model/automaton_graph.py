"""Semantic features of actual LDBA transitions for the TLT encoder."""
from collections import deque
from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class AutomatonGraph:
    nodes: torch.Tensor
    edges: torch.Tensor
    edge_features: torch.Tensor
    state_ids: tuple
    distance: dict
    initial_state: int
    guard_tokens: tuple
    progress_edges: tuple

    @classmethod
    def from_ldba(cls, ldba):
        ldba.compute_sccs() if not ldba.state_to_scc else None
        states = tuple(sorted(ldba.state_to_transitions))
        mapping = {state: index for index, state in enumerate(states)}
        transitions = [edge for state in states for edge in ldba.state_to_transitions[state]]
        accepting = {edge.target for edge in transitions if edge.accepting}
        distances = {state: float('inf') for state in states}
        queue = deque(accepting)
        for state in accepting:
            distances[state] = 0
        while queue:
            target = queue.popleft()
            for edge in ldba.state_to_incoming_transitions[target]:
                if distances[edge.source] > distances[target] + 1:
                    distances[edge.source] = distances[target] + 1
                    queue.append(edge.source)
        maximum = max((value for value in distances.values() if np.isfinite(value)), default=0)
        denominator = max(maximum, 1)
        normalized = {state: min(distances[state], denominator) / denominator for state in states}
        nodes = []
        for state in states:
            scc = ldba.state_to_scc.get(state)
            cyclic = scc is not None and (len(scc.states) > 1 or any(edge.target == state for edge in ldba.state_to_transitions[state]))
            nodes.append([float(state in accepting), float(cyclic), float(state == ldba.initial_state), normalized[state]])
        edges, features, guards = [], [], []
        import re
        from utils import simplify, to_sympy, sympy_to_str
        vocabulary = {p: i + 8 for i, p in enumerate(ldba.propositions)}
        vocabulary.update({"!": 1, "&": 2, "|": 3, "(": 4, ")": 5, "t": 6, "f": 7, "True": 6, "False": 7})
        for edge in transitions:
            # Environment pruning enumerates unordered assignments. Canonicalize
            # equivalent Boolean guards before sequence encoding.
            guard_label = sympy_to_str(simplify(to_sympy(edge.label))) if not edge.is_epsilon() else None
            guards.append(tuple(vocabulary[token] for token in re.findall(r"[A-Za-z_][A-Za-z_0-9]*|[!&|()]", guard_label)) if not edge.is_epsilon() else (0,))
            source, target = edge.source, edge.target
            if edge.is_epsilon():
                category = 3
            elif distances[target] < distances[source]:
                category = 0
            else:
                category = 1
            one_hot = [float(index == category) for index in range(4)]
            edges.append([mapping[source], mapping[target]])
            features.append(one_hot + [float(target in accepting), normalized[target]])
        return cls(torch.tensor(nodes, dtype=torch.float32),
                   torch.tensor(edges, dtype=torch.long).t().contiguous().reshape(2, -1),
                   torch.tensor(features, dtype=torch.float32).reshape(-1, 6), states, distances, ldba.initial_state, tuple(guards), tuple(edge.accepting or edge.is_epsilon() or distances[edge.target] < distances[edge.source] for edge in transitions))
