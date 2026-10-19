from dataclasses import dataclass
from typing import Optional, List, Union, Dict, Type
import torch.nn as nn

@dataclass
class GNNConfig:
    hidden_dim: int = 64
    num_layers: int = 3
    dropout: float = 0.1
    heads: int = 4
    # 增强GNN配置
    use_edge_attr: bool = False  # 是否使用边特征（类型、接受转移等）
    edge_dim: int = 6  # 边特征维度：[edge_type_onehot(4), is_accepting, distance]
    use_node_state: bool = False  # 是否使用节点状态特征
    node_state_dim: int = 4  # 节点状态特征维度：[is_accepting, in_scc, is_initial, position_normalized]

@dataclass
class NetworkConfig:
    layers: List[int]
    activation: Optional[Union[nn.Module, Dict[str, Type[nn.Module]]]] = None

@dataclass
class ModelConfig:
    ltl_embedding_dim: int = 32
    set_net: NetworkConfig = None
    num_rnn_layers: int = 2
    actor: NetworkConfig = None
    critic: NetworkConfig = None
    env_net: Optional[NetworkConfig] = None
    gnn: GNNConfig = GNNConfig()  # 添加GNN配置

@dataclass
class PPOConfig:
    steps_per_process: int = 4096
    batch_size: int = 2048
    epochs: int = 10
    lr: float = 0.0003
    clip_eps: float = 0.2
    discount: float = 0.998
    gae_lambda: float = 0.95
    entropy_coef: float = 0.003
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5
    optim_eps: float = 1e-8

# 预定义的模型配置
model_configs = {
    'PointLtl2-v0': ModelConfig(
        ltl_embedding_dim=32,
        set_net=NetworkConfig(
            layers=[32, 64, 64],
            activation=nn.ReLU
        ),
        num_rnn_layers=2,
        actor=NetworkConfig(
            layers=[64, 64],
            activation=nn.Tanh
        ),
        critic=NetworkConfig(
            layers=[64, 64],
            activation=nn.Tanh
        ),
        gnn=GNNConfig(
            hidden_dim=64,
            num_layers=3,
            dropout=0.1,
            heads=4
        )
    )
}