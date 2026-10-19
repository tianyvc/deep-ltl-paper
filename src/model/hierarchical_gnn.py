import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric.data import Data, Batch

class HierarchicalGNNNet(nn.Module):
    """分层GNN网络：同时在LDBA状态图和序列图上进行处理"""
    
    def __init__(self, input_dim, hidden_dim=64, num_layers=3, dropout=0.1, heads=4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # LDBA图处理部分
        self.ldba_gnn = nn.ModuleList([
            gnn.GATv2Conv(
                hidden_dim,
                hidden_dim,
                heads=heads,
                dropout=dropout,
                edge_dim=4,  # 边类型维度：初始、接受、sink、普通
                add_self_loops=False
            ) for _ in range(num_layers)
        ])
        
        # 序列图处理部分
        self.seq_gnn = nn.ModuleList([
            gnn.GATv2Conv(
                hidden_dim,
                hidden_dim,
                heads=heads,
                dropout=dropout,
                edge_dim=5,  # 边类型维度：顺序、自环、长距离、语义相似、跨层
                add_self_loops=False
            ) for _ in range(num_layers)
        ])
        
        # 层归一化
        self.ldba_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.seq_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # 跨层注意力
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 特征转换
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.ldba_proj = nn.Linear(4, hidden_dim)  # LDBA状态特征投影
        
        # 全局池化
        gate_nn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 1)
        )
        
        self.global_attention = gnn.GlobalAttention(
            gate_nn=gate_nn,
            nn=nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
        )
        
        self.output_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.output_dim = hidden_dim
        
    def forward(self, seq_data, ldba_data):
        """
        前向传播
        Args:
            seq_data: (x, edge_index, edge_attr, batch) - 序列图数据
            ldba_data: (x, edge_index, edge_attr, batch) - LDBA图数据
        """
        # 1. 处理LDBA图
        ldba_x = self.ldba_proj(ldba_data.x)
        x_ldba = ldba_x
        for gnn_layer, norm in zip(self.ldba_gnn, self.ldba_norms):
            # GNN传播
            out = gnn_layer(x_ldba, ldba_data.edge_index, edge_attr=ldba_data.edge_attr)
            if out.dim() == 3:  # 多头注意力输出
                out = out.mean(dim=1)
            # 残差连接和归一化
            x_ldba = norm(out + x_ldba)
            x_ldba = torch.relu(x_ldba)
            
        # 2. 处理序列图
        seq_x = self.input_proj(seq_data.x)
        x_seq = seq_x
        for gnn_layer, norm in zip(self.seq_gnn, self.seq_norms):
            # GNN传播
            out = gnn_layer(x_seq, seq_data.edge_index, edge_attr=seq_data.edge_attr)
            if out.dim() == 3:
                out = out.mean(dim=1)
            # 残差连接和归一化
            x_seq = norm(out + x_seq)
            x_seq = torch.relu(x_seq)
            
        # 3. 跨层信息交互
        # 准备注意力的query、key、value
        x_seq_reshape = x_seq.view(-1, 1, self.hidden_dim)  # [num_seq_nodes, 1, hidden]
        x_ldba_reshape = x_ldba.view(-1, 1, self.hidden_dim)  # [num_ldba_nodes, 1, hidden]
        
        # 双向cross-attention
        seq2ldba, _ = self.cross_attention(x_seq_reshape, x_ldba_reshape, x_ldba_reshape)
        ldba2seq, _ = self.cross_attention(x_ldba_reshape, x_seq_reshape, x_seq_reshape)
        
        # 合并cross-attention结果
        x_seq = x_seq + seq2ldba.view_as(x_seq)
        x_ldba = x_ldba + ldba2seq.view_as(x_ldba)
        
        # 4. 全局池化
        seq_graph = self.global_attention(x_seq, batch=seq_data.batch)
        ldba_graph = self.global_attention(x_ldba, batch=ldba_data.batch)
        
        # 5. 组合两个层次的特征
        combined = torch.cat([seq_graph, ldba_graph], dim=-1)
        output = self.output_proj(combined)
        
        return output