"""DeepLTL sequence-policy reference using the repository's original components.

The curriculum entry trains reach/avoid sequences directly. The product adapter
uses the original value-guided sequence search for formula evaluation/training.
This preserves the DeepSets + GRU architecture, without claiming bit-identical
reproduction of an upstream release or of the paper's reported experiments.
"""
import gymnasium as gym
import torch
from torch import nn

from model.ltl.ltl_net import LTLNet
from model.policy import ContinuousActor, DiscreteActor
from preprocessing.vocab import VOCAB
from utils import torch_utils


class DeepLTLSequenceModel(nn.Module):
    def __init__(self, env, model_config):
        super().__init__()
        if len(VOCAB) <= 3:
            raise ValueError('Initialize assignment vocabulary before building DeepLTL')
        shape = env.observation_space['features'].shape
        self.env_net = model_config.env_net.build(shape) if model_config.env_net is not None else None
        env_size = self.env_net.embedding_size if self.env_net is not None else shape[0]
        embedding = nn.Embedding(len(VOCAB), model_config.ltl_embedding_dim, padding_idx=VOCAB['PAD'])
        self.ltl_net = LTLNet(embedding, model_config.set_net, model_config.num_rnn_layers)
        size = env_size + self.ltl_net.embedding_dim
        if isinstance(env.action_space, gym.spaces.Discrete):
            self.actor = DiscreteActor(env.action_space.n, [size, *model_config.actor.layers], model_config.actor.activation)
        else:
            self.actor = ContinuousActor(env.action_space.shape[0], [size, *model_config.actor.layers], model_config.actor.activation, model_config.actor.state_dependent_std)
        self.critic = torch_utils.make_mlp_layers([size, *model_config.critic.layers, 1], activation=model_config.critic.activation, final_layer_activation=False)
        self.recurrent = False

    def forward(self, observations):
        environment = self.env_net(observations.features) if self.env_net is not None else observations.features
        embedding = torch.cat([environment, self.ltl_net(observations.seq)], dim=-1)
        distribution = self.actor(embedding)
        distribution.set_epsilon_mask(observations.epsilon_mask)
        return distribution, self.critic(embedding).squeeze(-1)


class PaperDeepLTL(nn.Module):
    """Adapter stores selected sequences in the rollout for PPO replay consistency."""
    hv = False
    options = False
    configuration = 'baseline'

    def __init__(self, env, model_config):
        super().__init__()
        import preprocessing
        from preprocessing.vocab import reset_vocab
        from sequence.search.exhaustive_search import ExhaustiveSearch
        reset_vocab()
        preprocessing.init_vocab(sorted(env.get_possible_assignments(), key=lambda assignment: tuple(sorted(assignment.items()))))
        self.propositions = tuple(sorted(env.get_propositions()))
        self.reference = DeepLTLSequenceModel(env, model_config)
        self.search = ExhaustiveSearch(self.reference, self.propositions, num_loops=2)
        self.sequence_cache = {}

    def graph(self, ldba):
        from model.paper_hiltl import AutomatonGraph
        return AutomatonGraph.from_ldba(ldba)

    def prepare_observation(self, obs):
        import preprocessing
        from ltl.automata import LDBASequence
        key = (obs['ldba'], obs['ldba_state'])
        if key not in self.sequence_cache:
            self.sequence_cache[key] = self.search.all_sequences(obs['ldba'], obs['ldba_state'], obs=None, num_loops=2)
        candidates = self.sequence_cache[key]
        if not candidates or any(len(candidate) == 0 for candidate in candidates):
            raise ValueError('Sequence baseline requires a nonempty accepting path at a live state')
        device = next(self.parameters()).device
        batch = preprocessing.preprocess_obss([dict(obs, goal=seq) for seq in candidates], self.propositions, device=device)
        with torch.no_grad():
            _, values = self.reference(batch)
            selected = int(values.argmax())
        return dict(obs, baseline_sequence=candidates[selected])

    def forward(self, observations):
        import preprocessing
        device = next(self.parameters()).device
        batch = preprocessing.preprocess_obss([dict(obs, goal=obs['baseline_sequence']) for obs in observations], self.propositions, device=device)
        distribution, value = self.reference(batch)
        return distribution, dict(total=value, short=value, long=torch.zeros_like(value), beta=torch.zeros_like(value), option=None)
