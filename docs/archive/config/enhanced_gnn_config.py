# Enhanced GNN Configuration with Automaton Structure
# This config enables:
# 1. Edge features (chain, skip, epsilon, accepting transitions)
# 2. Node state features (accepting states, SCC info, position)

from config import ModelConfig, GNNConfig, NetworkConfig
import torch.nn as nn

# Enhanced GNN configuration for PointLtl2-v0
enhanced_gnn_config = ModelConfig(
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
        heads=4,
        # 启用增强功能
        use_edge_attr=True,
        edge_dim=6,  # [edge_type_onehot(4), is_accepting, distance]
        use_node_state=True,
        node_state_dim=4  # [is_accepting, in_scc, is_initial, position_normalized]
    )
)

# Baseline GNN configuration (for comparison)
baseline_gnn_config = ModelConfig(
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
        heads=4,
        # 不启用增强功能（仅链式边）
        use_edge_attr=False,
        edge_dim=6,
        use_node_state=False,
        node_state_dim=4
    )
)
