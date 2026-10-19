from abc import abstractmethod, ABC
from typing import Type, Optional

from dataclasses import dataclass

from torch import nn

from model.env import ConvEnvNet, StandardEnvNet
from model.ltl.set_network import SetNetwork


class AbstractModelConfig(ABC):
    @abstractmethod
    def build(self, input_shape: tuple[int, ...]) -> nn.Module:
        pass


@dataclass
class StandardNetConfig(AbstractModelConfig):
    layers: list[int]
    activation: Optional[Type[nn.Module]]

    def build(self, input_shape: tuple[int,]) -> nn.Module:
        return StandardEnvNet(input_shape[0], self.layers, self.activation)


@dataclass
class ConvNetConfig(AbstractModelConfig):
    channels: list[int]
    kernel_size: tuple[int, int]
    activation: Type[nn.Module]

    def build(self, input_shape: tuple[int, int, int]) -> nn.Module:
        return ConvEnvNet(input_shape, self.channels, self.kernel_size, self.activation)


@dataclass
class SetNetConfig(AbstractModelConfig):
    layers: list[int]
    activation: Type[nn.Module]

    def build(self, input_shape: int) -> nn.Module:
        return SetNetwork(input_shape, self.layers, self.activation)


@dataclass
class ActorConfig:
    layers: list[int]
    activation: Optional[Type[nn.Module]] | dict[str, Type[nn.Module]]
    state_dependent_std: bool = False


@dataclass
class GNNConfig:
    hidden_dim: int = 64
    num_layers: int = 3
    dropout: float = 0.1
    heads: int = 4
    use_edge_attr: bool = False
    edge_dim: int = 6
    use_node_state: bool = False
    node_state_dim: int = 4


@dataclass
class ModelConfig:
    actor: ActorConfig
    critic: StandardNetConfig
    ltl_embedding_dim: int
    num_rnn_layers: int
    env_net: Optional[AbstractModelConfig]
    set_net: SetNetConfig
    gnn: Optional[GNNConfig] = None


zones = ModelConfig(
    actor=ActorConfig(
        layers=[64, 64, 64],
        activation=dict(
            hidden=nn.ReLU,
            output=nn.Tanh
        ),
        state_dependent_std=True
    ),
    critic=StandardNetConfig(
        layers=[64, 64],
        activation=nn.Tanh
    ),
    ltl_embedding_dim=16,
    num_rnn_layers=1,
    env_net=StandardNetConfig(
        layers=[128, 64],
        activation=nn.Tanh
    ),
    set_net=SetNetConfig(
        layers=[32, 16],
        activation=nn.ReLU
    ),
    gnn=GNNConfig(
        hidden_dim=64,
        num_layers=3,
        dropout=0.1,
        heads=4,
        use_edge_attr=True,      # ✅ 启用边特征
        edge_dim=6,
        use_node_state=True,     # ✅ 启用节点状态特征
        node_state_dim=4
    )
)

letter = ModelConfig(
    actor=ActorConfig(
        layers=[64, 64, 64],
        activation=nn.ReLU,
    ),
    critic=StandardNetConfig(
        layers=[64, 64],
        activation=nn.Tanh
    ),
    ltl_embedding_dim=32,
    num_rnn_layers=1,
    env_net=ConvNetConfig(
        channels=[16, 32, 64],
        kernel_size=(2, 2),
        activation=nn.ReLU
    ),
    set_net=SetNetConfig(
        layers=[32, 32],
        activation=nn.ReLU
    ),
    gnn=GNNConfig(
        hidden_dim=64,
        num_layers=3,
        dropout=0.1,
        heads=4,
        use_edge_attr=True,      # ✅ 启用边特征
        edge_dim=6,
        use_node_state=True,     # ✅ 启用节点状态特征
        node_state_dim=4
    )
)

flatworld = ModelConfig(
    actor=ActorConfig(
        layers=[64, 64, 64],
        activation=nn.ReLU,
    ),
    critic=StandardNetConfig(
        layers=[64, 64],
        activation=nn.ReLU
    ),
    ltl_embedding_dim=16,
    num_rnn_layers=1,
    env_net=StandardNetConfig(
        layers=[16, 16],
        activation=nn.ReLU
    ),
    set_net=SetNetConfig(
        layers=[32, 16],
        activation=nn.ReLU
    ),
    gnn=GNNConfig(
        hidden_dim=64,
        num_layers=3,
        dropout=0.1,
        heads=4,
        use_edge_attr=True,      # ✅ 启用边特征
        edge_dim=6,
        use_node_state=True,     # ✅ 启用节点状态特征
        node_state_dim=4
    )
)
