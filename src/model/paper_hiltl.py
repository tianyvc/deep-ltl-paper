from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch_geometric.nn import GATv2Conv
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

from model.policy import ContinuousActor, DiscreteActor


PROTOCOL_VERSION = 'hiltl-product-v3'
CONFIGURATIONS = {
    'baseline': (False, False, False),
    'gnn_single': (False, False, False),
    'gnn_track': (False, False, True),
    'gnn_capacity': (False, False, False),
    'gnn_track_capacity': (False, False, True),
    'hv_only': (True, False, False),
    'msgnn_only': (False, True, False),
    'hv_track': (True, False, True),
    'msgnn_track': (False, True, True),
    'full_no_options': (True, True, True),
    'full_all': (True, True, True),
}


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


class ProductEpisode(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.history = ()

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.history = (obs['ldba_state'],)
        return self.snapshot(obs)

    def step(self, action):
        from ltl.automata import LDBASequence
        took_epsilon = bool((np.asarray(action) == LDBASequence.EPSILON).all())
        obs, _, done, info = self.env.step(np.asarray(action))
        info = dict(info, took_epsilon=took_epsilon)
        self.history = self.history + (obs['ldba_state'],)
        reward = float(info.get('accepting', False)) - float(info.get('violation', False))
        return self.snapshot(obs), reward, done, dict(info)

    def snapshot(self, obs):
        return dict(obs, features=np.array(obs['features'], copy=True), history=self.history)


def discounted_returns(rewards, gamma):
    rewards = torch.as_tensor(rewards, dtype=torch.float32)
    returns = torch.empty_like(rewards)
    running = rewards.new_zeros(())
    for index in reversed(range(len(rewards))):
        running = rewards[index] + gamma * running
        returns[index] = running
    return returns


class PaperHiLTL(nn.Module):
    def __init__(self, observation_shape, action_space, configuration='hv_track', hidden=64, propositions=(), gate='fixed'):
        super().__init__()
        self.hv, self.multiscale, self.tracking = CONFIGURATIONS[configuration]
        self.configuration = configuration
        self.propositions = tuple(sorted(propositions))
        self.options = configuration == "full_all"
        self.gate_type = gate
        self.guard_embedding = nn.Embedding(8 + len(self.propositions), hidden)
        self.guard_encoder = nn.GRU(hidden, hidden, batch_first=True)
        self.edge_encoder = nn.Linear(6 + 2 * hidden, hidden)
        self.guard_value = nn.Sequential(nn.Linear(hidden, hidden), nn.LeakyReLU(.2))
        self.beta_net = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 1)) if gate == "learned" else None
        self.env_encoder = nn.Sequential(nn.Flatten(), nn.Linear(int(np.prod(observation_shape)), 128), nn.ReLU(), nn.Linear(128, hidden), nn.ReLU())
        self.node_encoder = nn.Linear(4, hidden)
        self.convs = nn.ModuleList([GATv2Conv(hidden, hidden, heads=4, concat=False, edge_dim=hidden, add_self_loops=False) for _ in range(3)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(3)])
        self.transforms = nn.ModuleList([nn.Sequential(nn.Linear(hidden, hidden), nn.LeakyReLU(.2)) for _ in range(3)])
        self.pool_gates = nn.ModuleList([nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1)) for _ in range(3)])
        self.history_encoder = nn.LSTM(hidden, hidden, batch_first=True) if self.tracking else None
        graph_size = hidden * (3 if self.multiscale else 1)
        actor_size = hidden + graph_size + hidden + (hidden if self.tracking else 0)
        if self.options:
            self.option_encoder = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU())
            self.meta_controller = nn.Sequential(nn.Linear(actor_size + hidden, 128), nn.ReLU(), nn.Linear(128, 1))
            self.option_critic = nn.Sequential(nn.Linear(actor_size + hidden, 128), nn.ReLU(), nn.Linear(128, 1))
            actor_size += hidden
        if isinstance(action_space, gym.spaces.Discrete):
            self.actor = DiscreteActor(action_space.n, [actor_size, 64, 64], nn.ReLU)
        else:
            self.actor = ContinuousActor(action_space.shape[0], [actor_size, 64, 64], nn.Tanh)
        self.single_history = self.tracking and not self.hv
        critic_width = {'gnn_capacity': 290, 'gnn_track_capacity': 277}.get(configuration, 256)
        self.short_head = nn.Sequential(nn.Linear(hidden + graph_size + hidden + (hidden if self.single_history else 0), critic_width), nn.ReLU(), nn.Linear(critic_width, critic_width), nn.ReLU(), nn.Linear(critic_width, 1))
        self.long_head = nn.Sequential(nn.Linear(1 + hidden + (hidden if self.tracking else 0), 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1)) if self.hv else None
        self.graph_cache = {}

    def graph(self, ldba):
        if self.propositions and tuple(ldba.propositions) != self.propositions:
            raise ValueError('Automaton proposition vocabulary differs from model vocabulary')
        key = ldba
        if key not in self.graph_cache:
            self.graph_cache[key] = AutomatonGraph.from_ldba(ldba)
        return self.graph_cache[key]

    def representations(self, observations):
        device = next(self.parameters()).device
        graphs = [self.graph(obs['ldba']) for obs in observations]
        graph_embeddings = {}
        for graph in {id(graph): graph for graph in graphs}.values():
            hidden = self.node_encoder(graph.nodes.to(device))
            tokens = [torch.tensor(t, device=device, dtype=torch.long) for t in graph.guard_tokens]
            packed_guards = pack_padded_sequence(pad_sequence([self.guard_embedding(t) for t in tokens], batch_first=True), [len(t) for t in tokens], batch_first=True, enforce_sorted=False)
            _, guard_hidden = self.guard_encoder(packed_guards)
            lexical = torch.stack([self.guard_embedding(token).mean(0) for token in tokens])
            edges = self.edge_encoder(torch.cat([graph.edge_features.to(device), guard_hidden[-1], lexical], dim=-1))
            active = torch.tensor(graph.progress_edges, dtype=torch.bool, device=device)
            context = hidden.new_zeros(hidden.shape)
            counts = hidden.new_zeros((len(hidden), 1))
            sources = graph.edges[0].to(device)[active]
            context.index_add_(0, sources, self.guard_value(edges[active]))
            counts.index_add_(0, sources, hidden.new_ones((len(sources), 1)))
            hidden = hidden + context / counts.clamp_min(1)
            layers = []
            for conv, norm, transform, gate in zip(self.convs, self.norms, self.transforms, self.pool_gates):
                hidden = norm(hidden + transform(conv(hidden, graph.edges.to(device), edges)))
                weights = torch.softmax(gate(hidden), dim=0)
                layers.append((weights * hidden).sum(0))
            graph_embeddings[id(graph)] = (hidden, torch.cat(layers) if self.multiscale else layers[-1], edges)
        node_values, pooled_values, histories, proximities, epsilon_masks = [], [], [], [], []
        for obs, graph in zip(observations, graphs):
            mapping = {state: index for index, state in enumerate(graph.state_ids)}
            nodes, pooled, _ = graph_embeddings[id(graph)]
            state_index = mapping[obs['ldba_state']]
            node_values.append(nodes[state_index])
            pooled_values.append(pooled)
            proximities.append(1 - graph.nodes[state_index, 3].item())
            if self.tracking:
                histories.append(nodes[[mapping[state] for state in obs['history']]])
            epsilon_masks.append(any(edge.is_epsilon() for edge in obs['ldba'].state_to_transitions[obs['ldba_state']]))
        current = torch.stack(node_values)
        pooled = torch.stack(pooled_values)
        env_features = self.env_encoder(torch.as_tensor(np.stack([obs['features'] for obs in observations]), dtype=torch.float32, device=device))
        short_input = torch.cat([env_features, pooled, current], dim=-1)
        history = None
        if self.tracking:
            packed = pack_padded_sequence(pad_sequence(histories, batch_first=True), [len(values) for values in histories], batch_first=True, enforce_sorted=False)
            _, (last, _) = self.history_encoder(packed)
            history = last[-1]
        actor_input = torch.cat([short_input, history], dim=-1) if self.tracking else short_input
        return actor_input, short_input, current, history, proximities, epsilon_masks, graphs, graph_embeddings

    def option_representation(self, observation, edge_index, graph, encoded):
        nodes, _, edges = encoded[id(graph)]
        target = int(graph.edges[1, edge_index])
        return self.option_encoder(torch.cat([nodes[target], edges[edge_index]], dim=-1))

    def option_distribution(self, observation):
        from model.paper_options import available_options
        if not self.options:
            raise ValueError('Options only exist in full_all')
        actor_input, _, _, _, _, _, graphs, encoded = self.representations([observation])
        candidates = available_options(observation)
        option_vectors = torch.stack([self.option_representation(observation, index, graphs[0], encoded) for index in candidates])
        logits = self.meta_controller(torch.cat([actor_input.expand(len(candidates), -1), option_vectors], dim=-1)).squeeze(-1)
        return torch.distributions.Categorical(logits=logits), candidates

    def forward(self, observations):
        actor_input, short_input, current, history, proximities, epsilon_masks, graphs, encoded = self.representations(observations)
        device = actor_input.device
        option_values = None
        if self.options:
            option_vectors = torch.stack([self.option_representation(obs, obs['option_edge'], graph, encoded) for obs, graph in zip(observations, graphs)])
            actor_input = torch.cat([actor_input, option_vectors], dim=-1)
            option_values = self.option_critic(actor_input).squeeze(-1)
        distribution = self.actor(actor_input)
        distribution.set_epsilon_mask(torch.tensor(epsilon_masks, dtype=torch.bool, device=device))
        critic_input = torch.cat([short_input, history], dim=-1) if self.single_history else short_input
        short = self.short_head(critic_input).squeeze(-1)
        proximity = torch.tensor(proximities, dtype=torch.float32, device=device).unsqueeze(1)
        beta = (self.beta_net(proximity).squeeze(-1).sigmoid() if self.beta_net is not None
                else torch.full_like(proximity.squeeze(1), .5) if self.gate_type == "constant"
                else torch.sigmoid(3 * (proximity.squeeze(1) - .5)))
        if self.hv:
            long_input = torch.cat([proximity, current, history], dim=-1) if self.tracking else torch.cat([proximity, current], dim=-1)
            long = self.long_head(long_input).squeeze(-1)
            total = (1 - beta) * short + beta * long
        else:
            long = torch.zeros_like(short)
            total = short
        return distribution, {'total': total, 'short': short, 'long': long, 'beta': beta, 'option': option_values}
