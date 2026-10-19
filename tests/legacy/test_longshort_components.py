#!/usr/bin/env python3
"""
长短期RL集成测试脚本
快速验证各组件是否正常工作
"""

import torch
import sys
sys.path.insert(0, 'src')

print("="*60)
print("长短期RL组件测试")
print("="*60)

# 测试1: 导入检查
print("\n[1/5] 测试导入...")
try:
    from ltl.automata import ltl2ldba
    from model.longshort_integration import (
        LongShortConfig,
        LongShortEnhancedModel,
        enhance_model_with_multiscale
    )
    from model.hierarchical_value import AutomatonStateTracker, HierarchicalValueNetwork
    from model.multiscale_gnn import MultiScaleSemanticGNN
    print("✓ 所有模块导入成功")
except ImportError as e:
    print(f"✗ 导入失败: {e}")
    sys.exit(1)

# 测试2: LDBA 构建
print("\n[2/5] 测试LDBA构建...")
try:
    formula = "F a & G(b -> F c)"
    ldba = ltl2ldba(formula)
    print(f"✓ LDBA构建成功")
    print(f"  - 公式: {formula}")
    print(f"  - 状态数: {ldba.num_states}")
    print(f"  - 转移数: {ldba.num_transitions}")
    print(f"  - 命题: {ldba.propositions}")
except Exception as e:
    print(f"✗ LDBA构建失败: {e}")
    sys.exit(1)

# 测试3: 自动机追踪器
print("\n[3/5] 测试自动机追踪器...")
try:
    tracker = AutomatonStateTracker(ldba, device='cpu')
    tracker.reset(batch_size=4)
    
    # 模拟命题输入
    props = torch.zeros(4, len(ldba.propositions))
    props[0, 0] = 1.0  # 第一个样本满足第一个命题
    
    next_states, is_accepting = tracker.step(props)
    features = tracker.get_state_features()
    
    print("✓ 自动机追踪器测试通过")
    print(f"  - 初始状态: {ldba.initial_state}")
    print(f"  - 下一状态: {next_states.tolist()}")
    print(f"  - 状态特征形状: {features.shape}")
except Exception as e:
    print(f"✗ 自动机追踪器测试失败: {e}")
    import traceback
    traceback.print_exc()

# 测试4: 层次化价值网络
print("\n[4/5] 测试层次化价值网络...")
try:
    value_net = HierarchicalValueNetwork(
        env_state_dim=32,
        formula_emb_dim=64
    )
    
    # 模拟输入
    env_state = torch.randn(4, 32)
    automaton_features = torch.randn(4, 4)
    formula_emb = torch.randn(4, 64)
    
    output = value_net(env_state, automaton_features, formula_emb)
    
    print("✓ 层次化价值网络测试通过")
    print(f"  - 总价值形状: {output['v_total'].shape}")
    print(f"  - 短期价值形状: {output['v_short'].shape}")
    print(f"  - 长期价值形状: {output['v_long'].shape}")
    print(f"  - 融合权重: {output['alpha'].mean().item():.3f}")
except Exception as e:
    print(f"✗ 层次化价值网络测试失败: {e}")
    import traceback
    traceback.print_exc()

# 测试5: 多尺度GNN
print("\n[5/5] 测试多尺度GNN...")
try:
    from torch_geometric.data import Data, Batch
    
    gnn = MultiScaleSemanticGNN(
        input_dim=32,
        hidden_dim=64,
        edge_dim=6,
        num_layers=3
    )
    
    # 创建简单图批次
    graphs = []
    for _ in range(4):
        x = torch.randn(5, 32)  # 5个节点
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        edge_attr = torch.randn(4, 6)
        graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))
    
    batch = Batch.from_data_list(graphs)
    output = gnn(batch)
    
    print("✓ 多尺度GNN测试通过")
    print(f"  - 短期嵌入形状: {output['embeddings'].short_term.shape}")
    print(f"  - 中期嵌入形状: {output['embeddings'].mid_term.shape}")
    print(f"  - 长期嵌入形状: {output['embeddings'].long_term.shape}")
    print(f"  - 注意力权重层数: {len(output['attention_weights'])}")
except Exception as e:
    print(f"✗ 多尺度GNN测试失败: {e}")
    import traceback
    traceback.print_exc()

# 测试6: 配置创建
print("\n[6/6] 测试配置系统...")
try:
    config = LongShortConfig(
        enable_hierarchical_value=True,
        enable_multiscale_gnn=True,
        enable_automaton_tracking=True,
        lambda_short=1.0,
        lambda_mid=0.5,
        lambda_long=0.3
    )
    print("✓ 配置系统测试通过")
    print(f"  - 层次价值: {config.enable_hierarchical_value}")
    print(f"  - 多尺度GNN: {config.enable_multiscale_gnn}")
    print(f"  - 自动机追踪: {config.enable_automaton_tracking}")
    print(f"  - 损失权重: λ_s={config.lambda_short}, λ_m={config.lambda_mid}, λ_l={config.lambda_long}")
except Exception as e:
    print(f"✗ 配置系统测试失败: {e}")

print("\n" + "="*60)
print("所有测试完成！")
print("="*60)
print("\n如果所有测试均显示 ✓，说明长短期RL组件已正确安装。")
print("可以开始训练实验：")
print("  PYTHONPATH=src/ python run_zones.py --device cuda --name test --seed 1 --enable_hierarchical_value")
print("")
