"""Topology-initiated, target-conditioned options with explicit time limits.

The paper does not specify an option learner. This operational protocol uses
PPO over primitive actions and PPO at SMDP decision boundaries. No option ID
is tied to a global automaton-state count, allowing unseen graph sizes.
"""
from collections import deque

import numpy as np
import torch


def available_options(observation):
    ldba, state = observation['ldba'], observation['ldba_state']
    edges = [edge for q in sorted(ldba.state_to_transitions) for edge in ldba.state_to_transitions[q]]
    candidates = []
    for index, edge in enumerate(edges):
        scc = ldba.state_to_scc[edge.target]
        if edge.source == state and not (scc.bottom and not scc.accepting):
            candidates.append(index)
    useful = [index for index in candidates if edges[index].target != state or edges[index].accepting or edges[index].is_epsilon()]
    candidates = useful or candidates
    if not candidates:
        raise ValueError('No non-trap outgoing option at a live product state')
    return candidates


def distance_to(ldba, target):
    distances = {target: 0}
    queue = deque([target])
    while queue:
        q = queue.popleft()
        for edge in ldba.state_to_incoming_transitions[q]:
            if edge.source not in distances:
                distances[edge.source] = distances[q] + 1
                queue.append(edge.source)
    return distances


def option_transition(previous, following, info, edge_index, age, max_steps):
    """Pure termination/intrinsic-reward oracle; no learner state is mutated."""
    ldba = previous['ldba']
    edges = [edge for q in sorted(ldba.state_to_transitions) for edge in ldba.state_to_transitions[q]]
    edge = edges[edge_index]
    old, new = previous['ldba_state'], following['ldba_state']
    if edge.is_epsilon():
        reached = bool(info.get('took_epsilon')) and old == edge.source and new == edge.target
    else:
        from ltl.logic import Assignment
        assignment = Assignment({p: p in following['propositions'] for p in ldba.propositions}).to_frozen()
        reached = not info.get('took_epsilon', False) and old == edge.source and new == edge.target and assignment in edge.valid_assignments
    distances = distance_to(ldba, edge.target)
    cap = max(len(ldba.state_to_transitions), 1)
    potential_before = -min(distances.get(old, cap), cap) / cap
    potential_after = -min(distances.get(new, cap), cap) / cap
    intrinsic = float(reached) + .1 * (potential_after - potential_before)
    reason = 'target' if reached else ('state_changed' if new != edge.source else ('timeout' if age >= max_steps else None))
    return intrinsic, reason


def smdp_targets(rewards, durations, values, gamma, gae_lambda=.95):
    """rewards contain within-option discounted extrinsic returns."""
    rewards = torch.as_tensor(rewards, device=values.device, dtype=values.dtype)
    durations = torch.as_tensor(durations, device=values.device, dtype=values.dtype)
    if len(rewards) != len(values) or not torch.all(durations > 0):
        raise ValueError('SMDP segments must be nonempty and aligned')
    following = torch.cat([values[1:], values.new_zeros(1)])
    discounts = gamma ** durations
    td = rewards + discounts * following
    advantages = torch.zeros_like(values)
    running = values.new_zeros(())
    for index in reversed(range(len(values))):
        running = td[index] - values[index] + discounts[index] * gae_lambda ** durations[index] * running
        advantages[index] = running
    return td.detach(), (values + advantages).detach(), advantages.detach()
