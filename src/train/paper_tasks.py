"""Auditable task splits. Graph equality rejects syntactic and compiled overlap.

Compiled graph equality is a leakage safeguard, not a language-equivalence
proof. Both raw and environment-pruned automata are fingerprinted.
"""
import hashlib
import json
import re
from pathlib import Path

from ltl.automata import ltl2ldba
from ltl.samplers import FixedSampler
from envs import make_env


def graph_fingerprint(ldba):
    import networkx as nx
    from utils import simplify, to_sympy
    graph = nx.DiGraph()
    for state in ldba.state_to_transitions:
        graph.add_node(('state', state), label='initial' if state == ldba.initial_state else 'state')
    serial = 0
    for state in sorted(ldba.state_to_transitions):
        for edge in ldba.state_to_transitions[state]:
            guard = 'epsilon' if edge.label is None else str(simplify(to_sympy(edge.label)))
            node = ('transition', serial)
            serial += 1
            graph.add_node(node, label=json.dumps([guard, bool(edge.accepting)]))
            graph.add_edge(('state', state), node)
            graph.add_edge(node, ('state', edge.target))
    structure = nx.weisfeiler_lehman_graph_hash(graph, node_attr='label', iterations=max(3, ldba.num_states))
    payload = dict(propositions=ldba.propositions, states=ldba.num_states, transitions=serial, structure=structure)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def normalize_formula(formula):
    return re.sub(r'\s+', '', formula)


def load_tasks(path, formula, env_name, horizon):
    if path is None:
        source = dict(train=[formula], held_out=[])
    else:
        source = json.loads(Path(path).read_text())
    if set(source) not in ({'train', 'held_out'}, {'train', 'validation', 'held_out'}) or not source['train']:
        raise ValueError('Task JSON must contain a nonempty train list, a held_out list, and optional validation list')
    splits = ['train'] + (['validation'] if 'validation' in source else []) + ['held_out']
    records = []
    seen_text, seen_raw, seen_pruned = {}, {}, {}
    for split in splits:
        if not isinstance(source[split], list):
            raise ValueError('Task split must be a JSON list')
        for formula in source[split]:
            if not isinstance(formula, str) or not formula.strip():
                raise ValueError('Tasks must be nonempty LTL strings')
            env = make_env(env_name, FixedSampler.partial(formula), max_steps=horizon)
            try:
                obs = env.reset(seed=0)
                raw = ltl2ldba(formula, obs['ldba'].propositions, simplify_labels=False)
                fingerprint = graph_fingerprint(obs['ldba'])
                raw_fingerprint = graph_fingerprint(raw)
                key = normalize_formula(formula)
                if key in seen_text or raw_fingerprint in seen_raw or fingerprint in seen_pruned:
                    raise ValueError('Duplicate or compiled-overlapping task in split: ' + formula)
                seen_text[key] = split
                seen_raw[raw_fingerprint] = split
                seen_pruned[fingerprint] = split
                records.append(dict(split=split, formula=formula, raw_sha256=raw_fingerprint,
                                    pruned_sha256=fingerprint, states=obs['ldba'].num_states,
                                    edges=obs['ldba'].num_transitions, finite=obs['ldba'].is_finite_specification()))
            finally:
                env.close()
    payload = {split: list(source[split]) for split in splits}
    payload['automata'] = records
    payload['sha256'] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload


def training_sampler(formulas):
    import random
    def factory(propositions):
        return lambda: random.choice(formulas)
    return factory
