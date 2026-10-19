"""
多尺度 GNN：从不同层提取短期/中期/长期语义特征

架构：
- Layer 0: 1-hop 邻域 → 短期子目标
- Layer 1: 2-hop 邻域 → 中期规划
- Layer 2: 全局结构 → 长期策略
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Batch
try:  # 尝试一次性导入，可用则用于矢量化池化
    from torch_scatter import scatter_softmax, scatter_add  # type: ignore
except Exception:  # 包含 ImportError/OSError 等所有失败情况
    scatter_softmax = None
    scatter_add = None
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class MultiScaleEmbeddings:
    """多尺度嵌入容器"""
    short_term: torch.Tensor   # [batch, dim] Layer 0 输出
    mid_term: torch.Tensor     # [batch, dim] Layer 1 输出
    long_term: torch.Tensor    # [batch, dim] Layer 2 输出
    attention_weights: List[torch.Tensor]  # 每层的注意力权重


# 移除独立的 SemanticGATv2Layer，直接使用更高效的 GATv2Conv


class MultiScaleSemanticGNN(nn.Module):
    """多尺度语义 GNN（兼容旧 `GNNNet` 接口）

    与原模型的集成需求：
    - 旧模型调用签名：forward(x, edge_index, edge_attr=None, node_state_feat=None, batch=None)
    - 需要属性：output_dim, use_edge_attr, node_state_dim
    - 权重迁移代码期望存在 `convs`（此处提供别名指向 `gat_layers`）

    当前实现：
    - 多层 GATv2 + 注意力池化得到每层图嵌入
    - 默认返回最终长期（long_term）嵌入作为图级表示（维度 = hidden_dim）
    - 可通过 `return_details=True` 访问全部多尺度信息
    - 将最近一次前向的多尺度结果缓存到 `self.last_multiscale`，供外部增强模块使用
    """
    def __init__(
        self,
        input_dim: int = 32,
        hidden_dim: int = 64,
        edge_dim: int = 6,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
        use_edge_attr: bool = True,
        node_state_dim: int = 0
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.use_edge_attr = use_edge_attr and edge_dim is not None and edge_dim > 0
        self.node_state_dim = node_state_dim

        total_input_dim = input_dim + (node_state_dim if node_state_dim > 0 else 0)
        self.input_proj = nn.Linear(total_input_dim, hidden_dim)

        # 边特征投影（一次性）
        if self.use_edge_attr and edge_dim > 0:
            self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        else:
            self.edge_proj = None

        # 多层 GATv2（复用高效设计，concat=False 避免后处理）
        self.gat_layers = nn.ModuleList([
            GATv2Conv(
                hidden_dim, hidden_dim, heads=heads, dropout=dropout,
                edge_dim=hidden_dim if self.use_edge_attr else None,
                add_self_loops=True, concat=False
            ) for _ in range(num_layers)
        ])
        self.convs = self.gat_layers  # 兼容别名

        # 层归一化（轻量）
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

        # 仅最后一层用注意力池化（长期），中间层用均值池化（短/中期）
        self.final_pooling = GlobalSemanticAttentionPooling(hidden_dim)

        # 移除价值头（训练时不直接使用，按需外部计算）

        self.output_dim = hidden_dim  # 供上层拼接使用
        self.last_multiscale: Optional[Dict[str, torch.Tensor]] = None

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        node_state_feat: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        return_details: bool = False
    ):
        # 节点状态特征拼接
        if node_state_feat is not None and self.node_state_dim > 0:
            x = torch.cat([x, node_state_feat], dim=-1)

        h = self.input_proj(x)
        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)

        # 边特征投影（一次性）
        edge_attr_proj = None
        if self.use_edge_attr and edge_attr is not None and self.edge_proj is not None:
            edge_attr_proj = self.edge_proj(edge_attr)

        # 逐层传播（保存中间表示）
        layer_node_features = []  # 每层的节点特征
        for i, (gat, norm) in enumerate(zip(self.gat_layers, self.layer_norms)):
            h_new = gat(h, edge_index, edge_attr=edge_attr_proj)
            h = norm(h_new + h)  # 残差 + 归一化
            layer_node_features.append(h)

        # 快速路径（默认）：只返回长期嵌入
        if not return_details:
            long_term, _ = self.final_pooling(h, batch)
            return long_term

        # 详细路径（按需）：构造多尺度嵌入
        # 短期：第1层均值池化（局部）
        if scatter_add is not None:
            short_term = scatter_add(layer_node_features[0], batch, dim=0)
            mid_term = scatter_add(layer_node_features[min(1, len(layer_node_features)-1)], batch, dim=0) if self.num_layers > 1 else short_term
            batch_size = short_term.size(0)
            counts = scatter_add(torch.ones_like(batch, dtype=torch.float32), batch, dim=0).unsqueeze(-1).clamp(min=1)
            short_term = short_term / counts
            mid_term = mid_term / counts
        else:
            # 退化：分组均值
            batch_ids = batch.unique(sorted=True)
            short_list = [layer_node_features[0][batch == b].mean(dim=0, keepdim=True) for b in batch_ids]
            short_term = torch.cat(short_list, dim=0)
            if self.num_layers > 1:
                mid_list = [layer_node_features[1][batch == b].mean(dim=0, keepdim=True) for b in batch_ids]
                mid_term = torch.cat(mid_list, dim=0)
            else:
                mid_term = short_term

        # 长期：最后一层注意力池化（全局）
        long_term, attn_weights = self.final_pooling(h, batch)

        embeddings = MultiScaleEmbeddings(
            short_term=short_term,
            mid_term=mid_term,
            long_term=long_term,
            attention_weights=[attn_weights]
        )

        # 缓存（外部按需计算价值）
        self.last_multiscale = {'embeddings': embeddings}
        return self.last_multiscale


class GlobalSemanticAttentionPooling(nn.Module):
    """
    可学习的全局注意力池化
    
    类似 Transformer 的 [CLS] token 机制
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, hidden_dim))
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** -0.5
    
    def forward(self, node_features, batch_idx):
        """全局注意力池化

        node_features: [num_nodes, hidden_dim]
        batch_idx:     [num_nodes] 所属图索引
        返回: pooled [batch_size, hidden_dim], attention_weights(占位)
        """
        keys = self.key_proj(node_features)
        values = self.value_proj(node_features)
        scores = (self.query @ keys.t()) * self.scale  # [1, num_nodes]
        scores = scores.squeeze(0)

        if scatter_softmax is not None and scatter_add is not None:
            # 使用 torch_scatter 矢量化分组 softmax + 加权求和
            attn = scatter_softmax(scores, batch_idx, dim=0)  # [num_nodes]
            weighted = values * attn.unsqueeze(-1)
            pooled = scatter_add(weighted, batch_idx, dim=0)
        else:
            # 无扩展库：分组处理，batch_size 通常较小（并行环境数）
            batch_ids = batch_idx.unique(sorted=True)
            pooled_list = []
            for b in batch_ids:
                mask = (batch_idx == b)
                local_scores = scores[mask]
                local_values = values[mask]
                attn_local = torch.softmax(local_scores, dim=0)
                emb = (attn_local.unsqueeze(-1) * local_values).sum(dim=0, keepdim=True)
                pooled_list.append(emb)
            pooled = torch.cat(pooled_list, dim=0)

        # 占位注意力（如需可后续加入具体权重）
        return pooled, []


class MultiTimescaleValueLoss(nn.Module):
    """
    多时间尺度价值损失（配合 MultiScaleSemanticGNN）
    
    L = Σ_i λ_i * L_i(V_layer_i, Return_timescale_i)
    """
    def __init__(
        self,
        lambdas: List[float] = [1.0, 0.5, 0.3],  # 短/中/长期权重
        timescales: List[int] = [1, 5, None]     # 对应的时间尺度
    ):
        super().__init__()
        self.lambdas = lambdas
        self.timescales = timescales
        assert len(lambdas) == len(timescales)
    
    def forward(
        self,
        layer_values: torch.Tensor,  # [layers, batch, 1]
        rewards: torch.Tensor,       # [batch, seq_len]
        dones: torch.Tensor,         # [batch, seq_len]
        gamma: float = 0.99
    ) -> Dict[str, torch.Tensor]:
        """
        计算每层对应时间尺度的价值损失
        """
        num_layers = layer_values.shape[0]
        losses = []
        
        for i in range(num_layers):
            v_pred = layer_values[i].squeeze(-1)  # [batch]
            timescale = self.timescales[i]
            
            if timescale is None:
                # 长期：完整 episode return
                returns = self._compute_mc_return(rewards, dones, gamma)
            else:
                # 短/中期：n-step return
                returns = self._compute_nstep_return(rewards, dones, timescale, gamma)
            
            # MSE 损失
            loss = nn.functional.mse_loss(v_pred, returns.detach())
            losses.append(self.lambdas[i] * loss)
        
        total_loss = sum(losses)
        
        return {
            'total_loss': total_loss,
            'layer_losses': losses
        }
    
    def _compute_nstep_return(self, rewards, dones, n, gamma):
        """n-step return"""
        batch_size, seq_len = rewards.shape
        returns = torch.zeros(batch_size, device=rewards.device)
        
        for i in range(batch_size):
            ret = 0
            for t in range(min(n, seq_len)):
                ret += (gamma ** t) * rewards[i, t]
                if dones[i, t]:
                    break
            returns[i] = ret
        
        return returns
    
    def _compute_mc_return(self, rewards, dones, gamma):
        """Monte Carlo return"""
        batch_size, seq_len = rewards.shape
        returns = torch.zeros(batch_size, device=rewards.device)
        
        for i in range(batch_size):
            ret = 0
            for t in range(seq_len):
                ret += (gamma ** t) * rewards[i, t]
                if dones[i, t]:
                    break
            returns[i] = ret
        
        return returns


# 使用示例与测试
if __name__ == "__main__":
    from torch_geometric.data import Data, Batch
    
    # 创建模拟图数据
    def create_mock_graph(num_nodes=10):
        x = torch.randn(num_nodes, 32)  # 节点特征
        edge_index = torch.randint(0, num_nodes, (2, 20))  # 边
        edge_attr = torch.randn(20, 6)  # 边特征
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    # 批量图
    batch_size = 8
    graphs = [create_mock_graph() for _ in range(batch_size)]
    batch = Batch.from_data_list(graphs)
    
    # 初始化模型
    model = MultiScaleSemanticGNN(
        input_dim=32,
        hidden_dim=64,
        edge_dim=6,
        num_layers=3
    )
    
    # 前向传播
    output = model(batch, return_all_layers=True)
    
    print("=== MultiScale GNN Output ===")
    print(f"Short-term embedding: {output['embeddings'].short_term.shape}")
    print(f"Mid-term embedding: {output['embeddings'].mid_term.shape}")
    print(f"Long-term embedding: {output['embeddings'].long_term.shape}")
    print(f"Layer values shape: {output['values'].shape}")
    print(f"Number of attention weight tensors: {len(output['attention_weights'])}")
    
    # 测试损失函数
    rewards = torch.randn(batch_size, 50)
    dones = torch.zeros(batch_size, 50)
    
    loss_fn = MultiTimescaleValueLoss()
    losses = loss_fn(output['values'], rewards, dones)
    
    print("\n=== Loss Components ===")
    print(f"Total loss: {losses['total_loss'].item():.4f}")
    for i, l in enumerate(losses['layer_losses']):
        print(f"Layer {i} loss: {l.item():.4f}")
