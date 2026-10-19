import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric.data import Data
try:
    from torch_scatter import scatter_add
except Exception:  # pragma: no cover
    # 退化实现（无 torch_scatter 时使用 index_add_）
    def scatter_add(src, index, dim=0):
        assert dim == 0, "Fallback scatter_add 仅支持 dim=0"
        num = int(index.max().item()) + 1 if index.numel() > 0 else 0
        out = src.new_zeros((num, *src.shape[1:]))
        if index.numel() > 0:
            out.index_add_(0, index, src)
        return out


class SimpleGlobalAttention(nn.Module):
    """自定义全局注意力池化，避免 PyG 版本差异导致的接口不兼容。
    公式与 GlobalAttention 等价：score = gate_nn(x); 输出为 sum(score * x) 按 batch 聚合。
    """
    def __init__(self, gate_nn: nn.Module):
        super().__init__()
        self.gate_nn = gate_nn

    def forward(self, x: torch.Tensor, batch: torch.Tensor):
        # gate_nn 内部已包含非线性，输出形状 (N,1)
        scores = self.gate_nn(x)  # (N,1)
        weighted = x * scores  # 广播 (N, feat)
        return scatter_add(weighted, batch, dim=0)

class GNNNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=3, dropout=0.1, heads=4, 
                 edge_dim=None, use_edge_attr=False, node_state_dim=0):
        """
        增强的GNN网络，支持：
        1. 边特征（类型：链式、跳跃、epsilon、接受转移等）
        2. 节点状态特征（接受状态标记、SCC信息等）
        
        Args:
            input_dim: 输入节点特征维度
            hidden_dim: 隐藏层维度
            num_layers: GNN层数
            dropout: Dropout率
            heads: 注意力头数
            edge_dim: 边特征维度（原始维度）
            use_edge_attr: 是否使用边特征
            node_state_dim: 节点状态特征维度（接受状态、SCC等）
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.node_state_dim = node_state_dim
        self.use_edge_attr = use_edge_attr and edge_dim is not None and edge_dim > 0
        
        # 输入投影层（结合位置特征和状态特征）
        total_input_dim = input_dim + node_state_dim
        self.input_proj = nn.Linear(total_input_dim, hidden_dim)
        
        # 边特征转换层（仅在需要使用边特征时创建）
        if self.use_edge_attr and edge_dim is not None:
            self.edge_proj = nn.Linear(int(edge_dim), hidden_dim)
        else:
            self.edge_proj = None
        
        # GNN层
        self.gnn_layers = nn.ModuleList([
            gnn.GATv2Conv(
                hidden_dim,
                hidden_dim,
                heads=heads,
                dropout=dropout,
                edge_dim=hidden_dim if self.use_edge_attr else None,
                add_self_loops=True,   # 让GAT自动添加自环
                concat=False           # 直接在层内聚合多头，避免后处理均值
            ) for _ in range(num_layers)
        ])
        
        # Layer Normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) 
            for _ in range(num_layers)
        ])
        
        # 跳跃连接层
        self.skip_connections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])
        
        # 全局注意力池化 - 使用自定义实现避免不同版本的参数签名问题
        self.global_attention = SimpleGlobalAttention(
            gate_nn=nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            )
        )
        
        self.output_dim = hidden_dim
        
    def forward(self, x, edge_index, edge_attr=None, node_state_feat=None, batch=None):
        """
        前向传播
        Args:
            x: 节点特征 (num_nodes, input_dim)
            edge_index: 边的连接关系 (2, num_edges)
            edge_attr: 可选的边特征 (num_edges, edge_dim)
            node_state_feat: 可选的节点状态特征 (num_nodes, node_state_dim) 
                           包含接受状态标记、SCC信息等
            batch: 批处理向量，指示节点属于哪个图 (num_nodes,)
        """
        # 拼接位置特征和状态特征
        if node_state_feat is not None and self.node_state_dim > 0:
            x = torch.cat([x, node_state_feat], dim=-1)
        
        # 输入特征投影
        x = self.input_proj(x)
        
        # 处理边特征（只在初始化时配置了边特征支持时）
        if self.use_edge_attr and edge_attr is not None and self.edge_proj is not None:
            edge_attr = self.edge_proj(edge_attr)
            
        # GNN处理 - 使用预先配置好的参数避免条件判断
        for gat, skip, norm in zip(self.gnn_layers, self.skip_connections, self.layer_norms):
            # 跳跃连接
            identity = skip(x)
            
            # 应用GAT层
            if self.use_edge_attr and edge_attr is not None:
                gat_out = gat(x, edge_index, edge_attr=edge_attr)
            else:
                gat_out = gat(x, edge_index)
                
            # 当 concat=False 时，GATv2Conv 已在层内聚合多头，无需手动均值
                
            # 残差连接 + 归一化 + 激活
            x = torch.relu(norm(gat_out + identity))
        
        # 全局池化
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        return self.global_attention(x, batch)

def process_reach_avoid_to_graph(reach_data, avoid_data, reach_lens=None,
                                 *, build_edge_attr=False, include_self_loops=False, max_skip=2):
    """
    将reach/avoid序列对转换为图结构
    
    Args:
        reach_data: reach集合的特征 (batch, max_len, feat_dim) 或 (batch, max_len)
        avoid_data: avoid集合的特征 (batch, max_len, feat_dim) 或 (batch, max_len)
        reach_lens: 每个序列的实际长度 (batch,)
    
    Returns:
        Data: PyTorch Geometric的Data对象
    """
    batch_size = reach_data.size(0)
    max_len = reach_data.size(1)
    
    # 处理序列长度
    if reach_lens is None:
        reach_lens = torch.full((batch_size,), max_len, device=reach_data.device)
    
    # 收集节点和边的信息
    nodes = []
    edges = []
    edge_attrs = []  # 可选：边类型特征
    batch_vec = []
    node_offset = 0
    
    for b in range(batch_size):
        # 使用 int() 显式转换，避免 .item() 返回 float 触发类型检查告警
        seq_len = int(reach_lens[b].item())
        if seq_len == 0:
            # 空序列：放置一个占位节点（维度与正常拼接后保持一致）
            nodes.append(torch.zeros(1, reach_data.size(-1) * 2, device=reach_data.device))
            batch_vec.append(torch.full((1,), b, dtype=torch.long, device=reach_data.device))
            node_offset += 1
            continue

        # 添加实际序列位置的节点 (concat reach + avoid)
        batch_reach = reach_data[b, :seq_len]
        batch_avoid = avoid_data[b, :seq_len]
        nodes.append(torch.cat([batch_reach, batch_avoid], dim=-1))
        batch_vec.append(torch.full((seq_len,), b, dtype=torch.long, device=reach_data.device))

        # 构建边
        max_k = 2 if max_skip is None else int(max_skip)
        for i in range(seq_len):
            # 1. 顺序边（与下一个位置，双向）
            if i < seq_len - 1:
                edges.append([node_offset + i, node_offset + i + 1])
                edges.append([node_offset + i + 1, node_offset + i])
                if build_edge_attr:
                    edge_attrs.extend([[0], [0]])

            # 2. 可选自环（一般可由 GATv2Conv 自动添加）
            if include_self_loops:
                edges.append([node_offset + i, node_offset + i])
                if build_edge_attr:
                    edge_attrs.append([3])

            # 3. 跨步长连接（i -> i+k, 双向）
            if max_k > 1:
                upper = min(max_k + 1, seq_len - i)
                for k in range(2, upper):
                    edges.append([node_offset + i, node_offset + i + k])
                    edges.append([node_offset + i + k, node_offset + i])
                    if build_edge_attr:
                        edge_attrs.extend([[4], [4]])

        node_offset += seq_len
    
    # 合并所有节点和边信息
    x = torch.cat(nodes, dim=0) if nodes else torch.empty((0, reach_data.size(-1)*2), device=reach_data.device)
    edge_index = torch.tensor(edges, dtype=torch.long, device=reach_data.device).t() if edges else torch.empty((2, 0), dtype=torch.long, device=reach_data.device)
    edge_attr = (torch.tensor(edge_attrs, dtype=torch.float, device=reach_data.device)
                 if build_edge_attr and len(edge_attrs) > 0 else None)
    batch_tensor = torch.cat(batch_vec, dim=0) if batch_vec else torch.empty((0,), dtype=torch.long, device=reach_data.device)
    
    return Data(
        x=x, 
        edge_index=edge_index,
        edge_attr=edge_attr,
        batch=batch_tensor
    )