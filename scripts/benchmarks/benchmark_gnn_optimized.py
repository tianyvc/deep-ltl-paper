#!/usr/bin/env python3
"""测试GNN优化后的性能"""

import torch
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from model.gnn_net import GNNNet

def benchmark_gnn(num_iterations=100):
    """基准测试GNN性能"""
    print("=" * 60)
    print("GNN 性能基准测试")
    print("=" * 60)
    
    # 设置参数
    batch_size = 16
    num_nodes_per_graph = 10
    total_nodes = batch_size * num_nodes_per_graph
    feat_dim = 128
    hidden_dim = 64
    
    # 创建GNN模型
    gnn = GNNNet(
        input_dim=feat_dim,
        hidden_dim=hidden_dim,
        num_layers=3,
        dropout=0.1,
        heads=4,
        use_edge_attr=False
    )
    
    if torch.cuda.is_available():
        gnn = gnn.cuda()
        device = torch.device('cuda')
        print(f"✓ 使用 GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("✓ 使用 CPU")
    
    # 生成测试数据
    x = torch.randn(total_nodes, feat_dim, device=device)
    
    # 生成边索引（序列图结构）
    edges = []
    for b in range(batch_size):
        offset = b * num_nodes_per_graph
        for i in range(num_nodes_per_graph - 1):
            edges.append([offset + i, offset + i + 1])
            edges.append([offset + i + 1, offset + i])
    
    edge_index = torch.tensor(edges, dtype=torch.long, device=device).t()
    
    # 生成batch索引
    batch = torch.repeat_interleave(
        torch.arange(batch_size, device=device),
        num_nodes_per_graph
    )
    
    print(f"\n测试配置:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Nodes per graph: {num_nodes_per_graph}")
    print(f"  - Total nodes: {total_nodes}")
    print(f"  - Feature dim: {feat_dim}")
    print(f"  - Hidden dim: {hidden_dim}")
    print(f"  - Num edges: {edge_index.shape[1]}")
    print(f"  - Iterations: {num_iterations}")
    
    # 预热
    print("\n预热中...")
    with torch.no_grad():
        for _ in range(10):
            _ = gnn(x, edge_index, batch=batch)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # 基准测试
    print("\n开始基准测试...")
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_iterations):
            output = gnn(x, edge_index, batch=batch)
            
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    elapsed = time.time() - start_time
    
    # 输出结果
    print("\n" + "=" * 60)
    print("测试结果:")
    print("=" * 60)
    print(f"  总时间: {elapsed:.4f} 秒")
    print(f"  平均时间: {elapsed/num_iterations*1000:.2f} 毫秒/次")
    print(f"  吞吐量: {num_iterations/elapsed:.2f} 次/秒")
    print(f"  输出形状: {output.shape}")
    print("=" * 60)
    
    # 内存使用
    if torch.cuda.is_available():
        memory = torch.cuda.max_memory_allocated() / 1024**2
        print(f"\nGPU 内存使用: {memory:.2f} MB")
    
    return elapsed / num_iterations

def test_graph_building():
    """测试图构建性能"""
    print("\n" + "=" * 60)
    print("图构建性能测试")
    print("=" * 60)
    
    batch_size = 16
    max_len = 20
    feat_dim = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 模拟位置特征
    pos_feats = torch.randn(batch_size, max_len, feat_dim, device=device)
    reach_lens = torch.randint(5, max_len, (batch_size,), device=device)
    
    print(f"\n测试配置:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Max length: {max_len}")
    print(f"  - Feature dim: {feat_dim}")
    
    # 导入优化的图构建函数（需要从model.py提取）
    def build_sequence_graph_loop(pos_feats, reach_lens):
        batch_size, max_len, feat_dim = pos_feats.shape
        device = pos_feats.device
        
        total_nodes = int(reach_lens.sum().item())
        if total_nodes == 0:
            x = torch.zeros((1, feat_dim), device=device)
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            batch_tensor = torch.zeros(1, dtype=torch.long, device=device)
            return x, edge_index, batch_tensor
        
        x = torch.zeros((total_nodes, feat_dim), device=device)
        batch_tensor = torch.zeros(total_nodes, dtype=torch.long, device=device)
        
        num_edges = sum(max(0, int(reach_lens[b].item()) - 1) * 2 for b in range(batch_size))
        edge_index = torch.zeros((2, num_edges), dtype=torch.long, device=device) if num_edges > 0 else torch.empty((2, 0), dtype=torch.long, device=device)
        
        node_idx = 0
        edge_idx = 0
        
        for b in range(batch_size):
            seq_len = int(reach_lens[b].item())
            if seq_len == 0:
                continue
            
            x[node_idx:node_idx + seq_len] = pos_feats[b, :seq_len]
            batch_tensor[node_idx:node_idx + seq_len] = b
            
            if seq_len > 1:
                num_forward = seq_len - 1
                src = torch.arange(node_idx, node_idx + num_forward, device=device)
                dst = torch.arange(node_idx + 1, node_idx + seq_len, device=device)
                
                edge_index[0, edge_idx:edge_idx + num_forward] = src
                edge_index[1, edge_idx:edge_idx + num_forward] = dst
                edge_idx += num_forward
                
                edge_index[0, edge_idx:edge_idx + num_forward] = dst
                edge_index[1, edge_idx:edge_idx + num_forward] = src
                edge_idx += num_forward
            
            node_idx += seq_len
        
        return x, edge_index, batch_tensor
    
    # 预热
        for _ in range(10):
            _ = build_sequence_graph_loop(pos_feats, reach_lens)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # 测试
    num_iterations = 1000
    start_time = time.time()
    
    for _ in range(num_iterations):
        x, edge_index, batch_tensor = build_sequence_graph_loop(pos_feats, reach_lens)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    elapsed = time.time() - start_time
    
    print(f"\n测试结果:")
    print(f"  总时间: {elapsed:.4f} 秒")
    print(f"  平均时间: {elapsed/num_iterations*1000:.2f} 毫秒/次")
    print(f"  吞吐量: {num_iterations/elapsed:.2f} 次/秒")
    print(f"  节点数: {x.shape[0]}")
    print(f"  边数: {edge_index.shape[1]}")
    print("=" * 60)

    # ================= 新的向量化实现 =================
    print("\n向量化图构建性能测试 (新实现)")
    print("-" * 60)

    def build_sequence_graph_vectorized(pos_feats, reach_lens):
        batch_size, max_len, feat_dim = pos_feats.shape
        device = pos_feats.device
        reach_lens = torch.clamp(reach_lens, min=0, max=max_len)
        time_ids = torch.arange(max_len, device=device).unsqueeze(0)
        mask = time_ids < reach_lens.unsqueeze(1)
        total_nodes = int(mask.sum().item())
        if total_nodes == 0:
            x = torch.zeros((1, feat_dim), device=device)
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            batch_tensor = torch.zeros(1, dtype=torch.long, device=device)
            return x, edge_index, batch_tensor
        x = pos_feats[mask]
        batch_tensor = torch.repeat_interleave(torch.arange(batch_size, device=device), reach_lens, output_size=total_nodes)
        lengths_minus_one = torch.clamp(reach_lens - 1, min=0)
        total_forward = int(lengths_minus_one.sum().item())
        if total_forward == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            return x, edge_index, batch_tensor
        prefix = torch.cumsum(reach_lens, dim=0) - reach_lens
        if max_len > 1:
            rel_pos = torch.arange(max_len - 1, device=device)
            valid_edge_mask = rel_pos.unsqueeze(0) < lengths_minus_one.unsqueeze(1)
            if valid_edge_mask.any():
                seq_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, rel_pos.size(0))[valid_edge_mask]
                rel_indices = rel_pos.unsqueeze(0).expand(batch_size, -1)[valid_edge_mask]
                sources = prefix[seq_indices] + rel_indices
                destinations = sources + 1
                edge_index = torch.stack([
                    torch.cat([sources, destinations], dim=0),
                    torch.cat([destinations, sources], dim=0)
                ], dim=0)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        return x, edge_index, batch_tensor

    # 预热
    for _ in range(10):
        _ = build_sequence_graph_vectorized(pos_feats, reach_lens)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.time()
    for _ in range(num_iterations):
        x_v, edge_index_v, batch_tensor_v = build_sequence_graph_vectorized(pos_feats, reach_lens)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_vec = time.time() - start_time
    print(f"  总时间: {elapsed_vec:.4f} 秒")
    print(f"  平均时间: {elapsed_vec/num_iterations*1000:.2f} 毫秒/次")
    print(f"  吞吐量: {num_iterations/elapsed_vec:.2f} 次/秒")
    print(f"  节点数: {x_v.shape[0]}")
    print(f"  边数: {edge_index_v.shape[1]}")
    speedup = (elapsed / elapsed_vec) if elapsed_vec > 0 else float('inf')
    print(f"  相对加速比(向量化/循环): {speedup:.2f}x")
    print("=" * 60)

if __name__ == "__main__":
    try:
        # 测试GNN性能
        avg_time = benchmark_gnn(num_iterations=100)
        
        # 测试图构建性能
        test_graph_building()
        
        print("\n✓ 所有测试完成!")
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
