#!/usr/bin/env python3
"""测试GNN相关代码的修复"""

import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from model.gnn_net import GNNNet

def test_gnn_initialization():
    """测试GNN初始化"""
    print("测试 GNN 初始化...")
    
    # 测试不使用边特征的情况
    gnn = GNNNet(
        input_dim=128,
        hidden_dim=64,
        num_layers=3,
        dropout=0.1,
        heads=4,
        edge_dim=None,
        use_edge_attr=False
    )
    print(f"✓ GNN (无边特征) 初始化成功")
    print(f"  - Input dim: {gnn.input_dim}")
    print(f"  - Hidden dim: {gnn.hidden_dim}")
    print(f"  - Output dim: {gnn.output_dim}")
    print(f"  - Use edge attr: {gnn.use_edge_attr}")
    
    return gnn

def test_gnn_forward():
    """测试GNN前向传播"""
    print("\n测试 GNN 前向传播...")
    
    gnn = GNNNet(
        input_dim=128,
        hidden_dim=64,
        num_layers=3,
        use_edge_attr=False
    )
    
    # 创建测试数据
    num_nodes = 10
    num_edges = 18
    batch_size = 2
    
    x = torch.randn(num_nodes, 128)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    batch = torch.cat([
        torch.zeros(5, dtype=torch.long),
        torch.ones(5, dtype=torch.long)
    ])
    
    # 前向传播（不使用边特征）
    try:
        output = gnn(x, edge_index, edge_attr=None, batch=batch)
        print(f"✓ 前向传播成功（无边特征）")
        print(f"  - 输入形状: {x.shape}")
        print(f"  - 输出形状: {output.shape}")
        print(f"  - 期望输出形状: ({batch_size}, {gnn.output_dim})")
        
        if output.shape == (batch_size, gnn.output_dim):
            print("  ✓ 输出形状正确")
        else:
            print(f"  ✗ 输出形状不正确: {output.shape}")
            
    except Exception as e:
        print(f"✗ 前向传播失败: {e}")
        raise

def test_simple_sequence():
    """测试简单序列的图构建"""
    print("\n测试简单序列的图构建...")
    
    # 模拟 reach/avoid 特征
    batch_size = 2
    max_len = 5
    feat_dim = 64
    
    reach_pos_emb = torch.randn(batch_size, max_len, feat_dim)
    avoid_pos_emb = torch.randn(batch_size, max_len, feat_dim)
    reach_lens = torch.tensor([5, 3])
    
    # 构建图
    pos_feats = torch.cat([reach_pos_emb, avoid_pos_emb], dim=-1)
    
    nodes = []
    edge_index_list = []
    batch_vec = []
    node_offset = 0
    
    for b in range(batch_size):
        seq_len = int(reach_lens[b].item())
        nodes.append(pos_feats[b, :seq_len, :])
        batch_vec.append(torch.full((seq_len,), b, dtype=torch.long))
        
        # 构建顺序边
        for i in range(seq_len - 1):
            edge_index_list.append([node_offset + i, node_offset + i + 1])
            edge_index_list.append([node_offset + i + 1, node_offset + i])
            
        node_offset += seq_len
    
    x = torch.cat(nodes, dim=0)
    batch_tensor = torch.cat(batch_vec, dim=0)
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t() if edge_index_list else torch.empty((2, 0), dtype=torch.long)
    
    print(f"✓ 图构建成功")
    print(f"  - 节点数: {x.shape[0]}")
    print(f"  - 节点特征维度: {x.shape[1]}")
    print(f"  - 边数: {edge_index.shape[1]}")
    print(f"  - Batch分配: {batch_tensor}")
    
    # 通过GNN
    gnn = GNNNet(input_dim=feat_dim*2, hidden_dim=64, num_layers=2, use_edge_attr=False)
    output = gnn(x, edge_index, edge_attr=None, batch=batch_tensor)
    
    print(f"✓ GNN处理成功")
    print(f"  - 输出形状: {output.shape}")
    print(f"  - 期望形状: ({batch_size}, {gnn.output_dim})")
    
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("GNN 代码测试")
    print("=" * 60)
    
    try:
        test_gnn_initialization()
        test_gnn_forward()
        test_simple_sequence()
        
        print("\n" + "=" * 60)
        print("✓ 所有测试通过!")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"✗ 测试失败: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        sys.exit(1)
