"""Original DeepSets/GRU sequence policy with strict upstream checkpoint shapes."""
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
