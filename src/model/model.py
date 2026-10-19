from typing import Any, Optional

import gymnasium
import torch
import torch.nn as nn

from config import ModelConfig
from model.ltl.ltl_net import LTLNet
from model.gnn_net import GNNNet
from preprocessing.batched_sequences import BatchedReachAvoidSequences
from model.mixed_distribution import MixedDistribution
from preprocessing.vocab import VOCAB
from model.policy import ContinuousActor
from model.policy import DiscreteActor
from utils import torch_utils
from utils.torch_utils import get_number_of_params


class Model(nn.Module):
    def __init__(self,
                 actor: nn.Module,
                 critic: nn.Module,
                 ltl_net: nn.Module,
                 gnn_net: nn.Module,
                 env_net: Optional[nn.Module],
                 ):
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.ltl_net = ltl_net
        self.gnn_net = gnn_net
        self.env_net = env_net
        self.recurrent = False
        
        # 缓存上一次的图结构以避免重复构建
        self._last_graph_cache = None
    
    def _build_sequence_graph(self, pos_feats, reach_lens, sequences=None, use_automaton_structure=False):
        """
        高度向量化的图构建函数：将序列特征(batch, max_len, feat_dim) 按实际长度 reach_lens
        构造成每个序列一条链式双向图（i <-> i+1）。
        
        增强功能：
        - 支持从LDBASequence提取自动机状态和转移信息
        - 支持接受状态节点特征标记
        - 支持边类型特征（链式、跳跃、epsilon、接受转移等）
        
        相比旧实现：
        - 去除 Python for 循环中的张量写入（除去 prefix 计算外完全使用批量布尔掩码与向量化）
        - 减少 GPU 上的小 kernel 启动次数，避免 16 次循环写入导致的 ~15ms 延迟
        - 复杂度 O(batch_size * max_len)；边与节点一次性构建
        
        Args:
            pos_feats: 位置特征 (batch, max_len, feat_dim)
            reach_lens: 每个序列的实际长度 (batch,)
            sequences: 可选的LDBASequence列表，用于提取状态和转移信息
            use_automaton_structure: 是否使用自动机结构信息（接受状态、边类型）
        
        Returns:
            x: 节点特征 (total_nodes, feat_dim)
            edge_index: 边索引 (2, num_edges)
            batch_tensor: 批次索引 (total_nodes,)
            edge_attr: 边特征 (num_edges, edge_feat_dim) [如果use_automaton_structure=True]
            node_state_feat: 节点状态特征 (total_nodes, node_state_dim) [如果use_automaton_structure=True]
        """
        batch_size, max_len, feat_dim = pos_feats.shape
        device = pos_feats.device

        # 过滤非法长度（防御性，避免下游错误）
        # 统一设备：有的 dataloader / 预处理可能在 CPU 保存 lengths
        if reach_lens.device != pos_feats.device:
            reach_lens = reach_lens.to(pos_feats.device, non_blocking=True)
        reach_lens = torch.clamp(reach_lens, min=0, max=max_len)

        # 1) 构建节点：利用布尔掩码一次性 gather
        # mask[b, t] = True  iff t < reach_lens[b]
        time_ids = torch.arange(max_len, device=device).unsqueeze(0)  # (1, max_len)
        mask = time_ids < reach_lens.unsqueeze(1)                     # (batch, max_len)
        total_nodes = int(mask.sum().item())
        if total_nodes == 0:
            x = torch.zeros((1, feat_dim), device=device)
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            batch_tensor = torch.zeros(1, dtype=torch.long, device=device)
            if use_automaton_structure:
                return x, edge_index, batch_tensor, None, None
            return x, edge_index, batch_tensor

        # 按 (b, t) 行主序拍平选择有效节点
        x = pos_feats[mask]  # (total_nodes, feat_dim)

        # batch_tensor：repeat_interleave 按各序列长度展开
        batch_tensor = torch.repeat_interleave(torch.arange(batch_size, device=device), reach_lens, output_size=total_nodes)
        
        # 提取节点状态特征（接受状态标记等）
        node_state_feat = None
        if use_automaton_structure and sequences is not None:
            node_state_feat = self._extract_node_state_features(sequences, mask, total_nodes, device)

        # 2) 构建边（双向链）
        lengths_minus_one = torch.clamp(reach_lens - 1, min=0)  # 每个序列的 forward 边数量
        total_forward = int(lengths_minus_one.sum().item())
        if total_forward == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            if use_automaton_structure:
                return x, edge_index, batch_tensor, None, node_state_feat
            return x, edge_index, batch_tensor

        # prefix offsets: 每个序列在展平节点数组中的起始索引
        # prefix[b] = sum_{k<b} reach_lens[k]
        prefix = torch.cumsum(reach_lens, dim=0) - reach_lens  # (batch,)

        # 构造所有可能相对位置 0..max_len-2 与有效掩码 (seq_len-1)
        if max_len > 1:
            rel_pos = torch.arange(max_len - 1, device=device)  # (max_len-1,)
            # valid_edge_mask[b, r] = r < (reach_lens[b]-1)
            valid_edge_mask = rel_pos.unsqueeze(0) < lengths_minus_one.unsqueeze(1)  # (batch, max_len-1)
            # 展开后按行主序提取 (sequence_index, relative_pos)
            if valid_edge_mask.any():
                seq_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, rel_pos.size(0))[valid_edge_mask]  # (total_forward,)
                rel_indices = rel_pos.unsqueeze(0).expand(batch_size, -1)[valid_edge_mask]                                    # (total_forward,)
                # sources = prefix[seq] + rel_indices
                sources = prefix[seq_indices] + rel_indices
                destinations = sources + 1
                # 拼接双向边
                edge_index = torch.stack([
                    torch.cat([sources, destinations], dim=0),
                    torch.cat([destinations, sources], dim=0)
                ], dim=0)
                
                # 构建边特征（边类型、是否接受转移等）
                edge_attr = None
                if use_automaton_structure and sequences is not None:
                    edge_attr = self._extract_edge_features(sequences, seq_indices, rel_indices, device)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
                edge_attr = None
        else:
            # max_len == 1 不会产生边
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            edge_attr = None

        if use_automaton_structure:
            return x, edge_index, batch_tensor, edge_attr, node_state_feat
        return x, edge_index, batch_tensor
    
    def _extract_node_state_features(self, sequences, mask, total_nodes, device):
        """从LDBASequence提取节点状态特征（接受状态、SCC信息等）"""
        # 节点状态特征: [is_accepting, in_scc, is_initial, position_normalized]
        node_features = []
        for seq_idx, seq in enumerate(sequences):
            if hasattr(seq, 'state_info') and seq.state_info:
                for pos_idx, state_info in enumerate(seq.state_info):
                    if mask[seq_idx, pos_idx]:
                        is_accepting = float(state_info.get('accepting', 0))
                        in_scc = float(state_info.get('in_scc', 0))
                        is_initial = float(state_info.get('is_initial', 0))
                        pos_norm = pos_idx / max(len(seq.state_info), 1)
                        node_features.append([is_accepting, in_scc, is_initial, pos_norm])
            else:
                # 没有状态信息时使用默认值
                for pos_idx in range(int(mask[seq_idx].sum())):
                    pos_norm = pos_idx / max(int(mask[seq_idx].sum()), 1)
                    node_features.append([0.0, 0.0, 0.0, pos_norm])
        
        if node_features:
            return torch.tensor(node_features, dtype=torch.float32, device=device)
        return None
    
    def _extract_edge_features(self, sequences, seq_indices, rel_indices, device):
        """从LDBASequence提取边特征（边类型、是否接受转移等）"""
        # 边特征: [edge_type_onehot(4维), is_accepting, distance]
        # edge_type: 0=chain_forward, 1=chain_backward, 2=skip, 3=epsilon
        edge_features = []
        total_edges = len(seq_indices)
        
        for i in range(total_edges):
            seq_idx = int(seq_indices[i])
            rel_idx = int(rel_indices[i])
            seq = sequences[seq_idx]
            
            # 默认链式前向边
            edge_type = [1.0, 0.0, 0.0, 0.0]  # chain_forward
            is_accepting = 0.0
            distance = 1.0
            
            if hasattr(seq, 'transition_info') and seq.transition_info and rel_idx < len(seq.transition_info):
                trans_info = seq.transition_info[rel_idx]
                is_accepting = float(trans_info.get('accepting', 0))
                if trans_info.get('is_epsilon', False):
                    edge_type = [0.0, 0.0, 0.0, 1.0]  # epsilon
                distance = float(trans_info.get('distance', 1))
            
            edge_features.append(edge_type + [is_accepting, distance])
        
        # 双向边：前向边 + 反向边
        forward_features = torch.tensor(edge_features, dtype=torch.float32, device=device)
        # 反向边特征：edge_type改为chain_backward
        backward_features = forward_features.clone()
        backward_features[:, 0] = 0.0  # 不是chain_forward
        backward_features[:, 1] = 1.0  # 是chain_backward
        
        return torch.cat([forward_features, backward_features], dim=0)

    def compute_embedding(self, obs):
        env_embedding = self.env_net(obs.features) if self.env_net is not None else obs.features
        ltl_embedding = self.ltl_net(obs.seq)
        # 为每个批次条目构建LTL图并通过GNN得到图级表示
        # obs.seq 可能是 BatchedReachAvoidSequences（完整batch）或在 minibatch 中为
        # ((reach_lens, reach_data),(avoid_lens, avoid_data)) 的 tuple 形式
        try:
            if isinstance(obs.seq, BatchedReachAvoidSequences):
                (reach_lens, reach_data), (avoid_lens, avoid_data) = obs.seq.all()
            else:
                # 处理 exps minibatch 中的 tuple 形式
                (reach_lens, reach_data), (avoid_lens, avoid_data) = obs.seq
        except Exception:
            reach_lens = reach_data = avoid_lens = avoid_data = None

        # 如果成功读取到 reach/avoid 数据，则构建图并通过GNN得到表示
        if reach_lens is not None:
            # reach_data/avoid_data: (batch, max_len, max_set_size)
            # 使用 LTLNet 的 embedding 和 set_network 生成每个位置的表示
            # 统一设备（可能出现 lengths 在 CPU, data 在 GPU 的情况）
            device = env_embedding.device
            if reach_data.device != device:
                reach_data = reach_data.to(device, non_blocking=True)
            if avoid_data.device != device:
                avoid_data = avoid_data.to(device, non_blocking=True)
            if reach_lens.device != device:
                reach_lens = reach_lens.to(device, non_blocking=True)

            reach_emb = self.ltl_net.embedding(reach_data)  # (batch, max_len, max_set_size, emb_dim)
            reach_pos_emb = self.ltl_net.set_network(reach_emb)
            # 如果 set_network 返回了每个集合元素的表示（4D），对集合维度求平均以得到每个位置的表示
            if reach_pos_emb.dim() == 4:
                reach_pos_emb = reach_pos_emb.mean(dim=2)
            avoid_emb = self.ltl_net.embedding(avoid_data)
            avoid_pos_emb = self.ltl_net.set_network(avoid_emb)
            if avoid_pos_emb.dim() == 4:
                avoid_pos_emb = avoid_pos_emb.mean(dim=2)
            
            # 每个位置的特征为 reach+avoid 的拼接
            pos_feats = torch.cat([reach_pos_emb, avoid_pos_emb], dim=-1)  # (batch, max_len, ltl_pos_dim)

            # 检查是否需要使用自动机结构信息
            use_edge_attr = hasattr(self.gnn_net, 'use_edge_attr') and self.gnn_net.use_edge_attr
            use_node_state = hasattr(self.gnn_net, 'node_state_dim') and self.gnn_net.node_state_dim > 0
            sequences = getattr(obs.seq, 'sequences', None) if hasattr(obs.seq, 'sequences') else None
            
            # 使用增强的图构建函数（只有在有 sequences 且启用了增强功能时才提取特征）
            use_automaton_structure = (use_edge_attr or use_node_state) and sequences is not None
            
            if use_automaton_structure:
                x, edge_index, batch_tensor, edge_attr, node_state_feat = self._build_sequence_graph(
                    pos_feats, reach_lens, sequences, use_automaton_structure=True
                )
                # 通过GNN得到图级表示（使用边特征和节点状态特征）
                gnn_embedding = self.gnn_net(x, edge_index, edge_attr=edge_attr, 
                                             node_state_feat=node_state_feat, batch=batch_tensor)
            else:
                # 不使用自动机结构信息（向后兼容或无可用序列）
                # 注意：如果配置了 node_state_dim 但没有序列，需要提供零填充特征
                x, edge_index, batch_tensor = self._build_sequence_graph(pos_feats, reach_lens)
                
                # 如果 GNN 期望节点状态特征但没有可用序列，创建零填充特征
                if use_node_state:
                    node_state_feat = torch.zeros(x.size(0), self.gnn_net.node_state_dim, 
                                                  device=x.device, dtype=x.dtype)
                else:
                    node_state_feat = None
                
                gnn_embedding = self.gnn_net(x, edge_index, edge_attr=None, 
                                            node_state_feat=node_state_feat, batch=batch_tensor)
        else:
            # 如果无法从 obs.seq 中取出 LTL 数据，则使用零向量
            batch_size = env_embedding.size(0)
            gnn_embedding = torch.zeros(
                batch_size, 
                self.gnn_net.output_dim, 
                device=env_embedding.device
            )

        # 检查并对齐batch维度
        try:
            batch_dims = {
                'env': env_embedding.size(0),
                'ltl': ltl_embedding.size(0),
                'gnn': gnn_embedding.size(0)
            }
            
            # 如果维度不一致，尝试广播
            max_batch = max(batch_dims.values())
            if not all(d == max_batch for d in batch_dims.values()):
                if batch_dims['ltl'] == 1 and max_batch > 1:
                    ltl_embedding = ltl_embedding.expand(max_batch, -1)
                if batch_dims['gnn'] == 1 and max_batch > 1:
                    gnn_embedding = gnn_embedding.expand(max_batch, -1)
                if batch_dims['env'] == 1 and max_batch > 1:
                    env_embedding = env_embedding.expand(max_batch, -1)
                    
        except Exception as e:
            dims = {
                'env': getattr(env_embedding, 'shape', None),
                'ltl': getattr(ltl_embedding, 'shape', None),
                'gnn': getattr(gnn_embedding, 'shape', None)
            }
            raise RuntimeError(f"Failed to align batch dimensions: {dims}") from e

        return torch.cat([env_embedding, ltl_embedding, gnn_embedding], dim=1)

    def forward(self, obs):
        embedding = self.compute_embedding(obs)
        dist = self.actor(embedding)
        dist.set_epsilon_mask(obs.epsilon_mask)
        value = self.critic(embedding).squeeze(1)
        return dist, value


def build_model(
        env: gymnasium.Env,
        training_status: dict[str, Any],
        model_config: ModelConfig,
) -> Model:
    if len(VOCAB) <= 3:
        raise ValueError('VOCAB not initialized')
    obs_shape = env.observation_space['features'].shape
    print(f'Observation shape: {obs_shape}')
    action_space = env.action_space
    action_dim = action_space.n if isinstance(action_space, gymnasium.spaces.Discrete) else action_space.shape[0]
    if model_config.env_net is not None:
        env_net = model_config.env_net.build(obs_shape)
        env_embedding_dim = env_net.embedding_size
    else:
        assert len(obs_shape) == 1
        env_net = None
        env_embedding_dim = obs_shape[0]

    embedding = nn.Embedding(len(VOCAB), model_config.ltl_embedding_dim, padding_idx=VOCAB['PAD'])
    ltl_net = LTLNet(embedding, model_config.set_net, model_config.num_rnn_layers)
    print(f'Num LTLNet params: {get_number_of_params(ltl_net)}')
    print(f'LTL embedding dim: {ltl_net.embedding_dim}')
    print(f'Set network embedding size: {ltl_net.set_network.embedding_size}')

    # 创建GNN网络（支持边特征和节点状态特征）
    gnn_config = model_config.gnn if hasattr(model_config, 'gnn') else None
    gnn_input_dim = 2 * ltl_net.set_network.embedding_size  # reach + avoid set embeddings concatenated
    
    if gnn_config:
        gnn_net = GNNNet(
            input_dim=gnn_input_dim,
            hidden_dim=gnn_config.hidden_dim,
            num_layers=gnn_config.num_layers,
            dropout=gnn_config.dropout,
            heads=gnn_config.heads,
            edge_dim=gnn_config.edge_dim if gnn_config.use_edge_attr else None,
            use_edge_attr=gnn_config.use_edge_attr,
            node_state_dim=gnn_config.node_state_dim if gnn_config.use_node_state else 0
        )
    else:
        # 默认配置（向后兼容）
        gnn_net = GNNNet(
            input_dim=gnn_input_dim,
            hidden_dim=64,
            num_layers=3,
            dropout=0.1,
            heads=4,
            edge_dim=None,
            use_edge_attr=False,
            node_state_dim=0
        )
    print(f'GNN input dim: {gnn_input_dim}')
    if gnn_config and gnn_config.use_edge_attr:
        print(f'GNN edge features: ENABLED (dim={gnn_config.edge_dim})')
    if gnn_config and gnn_config.use_node_state:
        print(f'GNN node state features: ENABLED (dim={gnn_config.node_state_dim})')
    
    # 计算总的嵌入维度（环境 + LTL + GNN）
    total_embedding_dim = env_embedding_dim + ltl_net.embedding_dim + gnn_net.output_dim

    if isinstance(env.action_space, gymnasium.spaces.Discrete):
        actor = DiscreteActor(action_dim=action_dim,
                              layers=[total_embedding_dim, *model_config.actor.layers],
                              activation=model_config.actor.activation)
    else:
        actor = ContinuousActor(action_dim=action_dim,
                                layers=[total_embedding_dim, *model_config.actor.layers],
                                activation=model_config.actor.activation,
                                state_dependent_std=model_config.actor.state_dependent_std)

    critic = torch_utils.make_mlp_layers([total_embedding_dim, *model_config.critic.layers, 1],
                                         activation=model_config.critic.activation,
                                         final_layer_activation=False)

    model = Model(actor, critic, ltl_net, gnn_net, env_net)

    # NOTE: 不在这里加载 state_dict，由调用者决定何时加载
    # 这样可以先增强模型（如 LongShortEnhancedModel）再加载权重
    return model
