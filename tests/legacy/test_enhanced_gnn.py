#!/usr/bin/env python3
"""
增强GNN性能对比测试脚本

对比三种配置：
1. 无GNN（仅RNN/Transformer）
2. 基础GNN（仅链式边）
3. 增强GNN（边特征 + 节点状态特征）

运行方式：
    python test_enhanced_gnn.py --env PointLtl2-v0 --formula "(!blue U green) & F yellow" --seeds 5
"""

import argparse
import sys
import os
import torch
import numpy as np
from pathlib import Path

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def run_evaluation(config_type, env_name, formula, num_seeds, deterministic=True):
    """运行评估并返回结果"""
    from evaluation.simulate import evaluate_policy
    
    results = []
    for seed in range(num_seeds):
        print(f"\n{'='*60}")
        print(f"运行 {config_type} - Seed {seed+1}/{num_seeds}")
        print(f"{'='*60}")
        
        # 这里需要根据config_type加载不同的模型
        # TODO: 实现模型加载逻辑
        result = {
            'seed': seed,
            'success_rate': 0.0,
            'avg_reward': 0.0,
            'avg_steps': 0.0
        }
        results.append(result)
    
    return results

def compare_results(baseline_results, enhanced_results):
    """对比基线和增强版本的结果"""
    print(f"\n{'='*80}")
    print("性能对比结果")
    print(f"{'='*80}")
    
    metrics = ['success_rate', 'avg_reward', 'avg_steps']
    
    for metric in metrics:
        baseline_vals = [r[metric] for r in baseline_results]
        enhanced_vals = [r[metric] for r in enhanced_results]
        
        baseline_mean = np.mean(baseline_vals)
        baseline_std = np.std(baseline_vals)
        enhanced_mean = np.mean(enhanced_vals)
        enhanced_std = np.std(enhanced_vals)
        
        improvement = ((enhanced_mean - baseline_mean) / baseline_mean * 100) if baseline_mean != 0 else 0
        
        print(f"\n{metric.upper()}:")
        print(f"  基础GNN: {baseline_mean:.4f} ± {baseline_std:.4f}")
        print(f"  增强GNN: {enhanced_mean:.4f} ± {enhanced_std:.4f}")
        print(f"  提升: {improvement:+.2f}%")
    
    # 统计显著性检验
    from scipy import stats
    for metric in metrics:
        baseline_vals = [r[metric] for r in baseline_results]
        enhanced_vals = [r[metric] for r in enhanced_results]
        
        t_stat, p_value = stats.ttest_ind(baseline_vals, enhanced_vals)
        print(f"\n{metric} t-test: t={t_stat:.4f}, p={p_value:.4f} {'*' if p_value < 0.05 else ''}")

def main():
    parser = argparse.ArgumentParser(description='增强GNN性能对比测试')
    parser.add_argument('--env', type=str, default='PointLtl2-v0', help='环境名称')
    parser.add_argument('--formula', type=str, default='(!blue U green) & F yellow', help='LTL公式')
    parser.add_argument('--seeds', type=int, default=5, help='随机种子数量')
    parser.add_argument('--deterministic', action='store_true', help='确定性评估')
    
    args = parser.parse_args()
    
    print("="*80)
    print("增强GNN性能对比测试")
    print("="*80)
    print(f"环境: {args.env}")
    print(f"公式: {args.formula}")
    print(f"种子数: {args.seeds}")
    print(f"确定性: {args.deterministic}")
    
    # 1. 运行基础GNN（仅链式边）
    print(f"\n{'#'*80}")
    print("阶段 1: 评估基础GNN（仅链式边）")
    print(f"{'#'*80}")
    baseline_results = run_evaluation('baseline', args.env, args.formula, args.seeds, args.deterministic)
    
    # 2. 运行增强GNN（边特征 + 节点状态）
    print(f"\n{'#'*80}")
    print("阶段 2: 评估增强GNN（边特征 + 节点状态）")
    print(f"{'#'*80}")
    enhanced_results = run_evaluation('enhanced', args.env, args.formula, args.seeds, args.deterministic)
    
    # 3. 对比结果
    compare_results(baseline_results, enhanced_results)
    
    print(f"\n{'='*80}")
    print("测试完成")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()
