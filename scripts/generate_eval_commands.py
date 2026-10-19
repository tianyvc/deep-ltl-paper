#!/usr/bin/env python3
"""
生成批量评估命令列表
支持快速生成第一层和第三层的所有评估命令
"""

import os
from pathlib import Path

CONFIGS = {
    'baseline': '--no-flags',
    'hv_only': '--enable_hierarchical_value',
    'hv_track': '--enable_hierarchical_value --enable_automaton_tracking',
    'msgnn_only': '--enable_multiscale_gnn',
    'msgnn_track': '--enable_multiscale_gnn --enable_automaton_tracking',
    'full_no_options': '--enable_hierarchical_value --enable_multiscale_gnn --enable_automaton_tracking',
    'full_all': '--enable_hierarchical_value --enable_multiscale_gnn --enable_automaton_tracking --enable_options',
}

# 第一层：所有配置 × 有限时域
TIER1_TASKS = {
    'PointLtl2-v0': {
        'finite': [
            '(!blue U green) & F yellow',
            'F (blue & F (green & F yellow))',
            '!(red | magenta) U (blue & F yellow)',
        ]
    },
    'LetterEnv-v0': {
        'finite': [
            'F (a & (!b U c)) & F d',
            '(F d) & (!f U (d & F b))',
            'F ((a | c | j) & F b) & F (c & F d) & F k',
        ]
    },
    'FlatWorld-v0': {
        'finite': [
            'F ((red & magenta) & F ((blue & green) & F yellow))',
            'F (orange & (!red U magenta))',
            '(!red U (green & blue & aqua)) & F (orange & (F (red & magenta)))',
        ]
    },
}

# 第三层：关键配置 × 无限时域
TIER3_TASKS = {
    'PointLtl2-v0': {
        'infinite': [
            'G F (blue & green) & G F red',
            'G F blue & G F green & G F yellow & G !magenta',
        ]
    },
    'LetterEnv-v0': {
        'infinite': [
            'G F (e & (!a U f))',
            'G F a & G F b & G F c & G F d & G (!e & !f)',
        ]
    },
    'FlatWorld-v0': {
        'infinite': [
            'G F (blue & green) & G F (red & magenta)',
            'G F (aqua & blue) & G F red & G F yellow & G !green',
        ]
    },
}

def generate_commands(tier='tier1', seed=1):
    """
    生成命令列表
    tier: 'tier1' 或 'tier3'
    seed: 种子编号
    """
    
    tasks = TIER1_TASKS if tier == 'tier1' else TIER3_TASKS
    configs_to_test = list(CONFIGS.keys())  # tier1 测试所有配置
    if tier == 'tier3':
        configs_to_test = ['baseline', 'hv_track']  # tier3 只测试两个
    
    commands = []
    count = 0
    
    for env, formulas_dict in tasks.items():
        horizon = list(formulas_dict.keys())[0]  # 'finite' 或 'infinite'
        horizon_flag = '--finite' if horizon == 'finite' else '--no-finite'
        
        # 为 PointLtl2-v0 设置 deterministic
        det_flag = '--deterministic' if env == 'PointLtl2-v0' else '--no-deterministic'
        
        for formula in formulas_dict[horizon]:
            for config in configs_to_test:
                count += 1
                flags = CONFIGS[config]
                flags_str = '' if flags == '--no-flags' else flags
                
                cmd = (
                    f"PYTHONPATH=src/ python src/evaluation/simulate.py "
                    f"--env {env} --exp test --seed {seed} "
                    f"--formula '{formula}' {horizon_flag} {det_flag} {flags_str}"
                ).strip()
                
                commands.append({
                    'num': count,
                    'env': env,
                    'config': config,
                    'horizon': horizon,
                    'formula': formula,
                    'cmd': cmd
                })
    
    return commands

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='生成批量评估命令')
    parser.add_argument('--tier', choices=['tier1', 'tier3'], default='tier1',
                       help='选择数据收集层级（tier1=有限时域全配置, tier3=无限时域关键配置）')
    parser.add_argument('--seed', type=int, default=1, help='种子编号')
    parser.add_argument('--format', choices=['shell', 'csv', 'markdown'], default='shell',
                       help='输出格式')
    args = parser.parse_args()
    
    commands = generate_commands(tier=args.tier, seed=args.seed)
    
    if args.format == 'shell':
        print("#!/bin/bash")
        print("# 自动生成的批量评估脚本")
        print(f"# 时间层级: {args.tier} (共 {len(commands)} 条命令)")
        print(f"# 种子: {args.seed}\n")
        
        current_env = None
        for item in commands:
            if item['env'] != current_env:
                current_env = item['env']
                print(f"\n# ============ {current_env} ============")
            
            print(f"\n# {item['num']}. [{item['config']}] {item['horizon']}")
            print(f"# Formula: {item['formula'][:60]}...")
            print(item['cmd'])
    
    elif args.format == 'csv':
        import csv
        output_file = f"commands_{args.tier}_seed{args.seed}.csv"
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['num', 'env', 'config', 'horizon', 'formula', 'cmd'])
            writer.writeheader()
            writer.writerows(commands)
        print(f"✓ 已保存命令列表到 {output_file}")
    
    elif args.format == 'markdown':
        print(f"# {args.tier.upper()} 评估命令列表\n")
        print(f"**总计**: {len(commands)} 条命令  \n")
        
        current_env = None
        for item in commands:
            if item['env'] != current_env:
                current_env = item['env']
                print(f"\n## {current_env}\n")
                print("| # | Config | Formula | Command |\n|---|--------|---------|----------|\n")
            
            formula_short = item['formula'][:40] + "..." if len(item['formula']) > 40 else item['formula']
            print(f"| {item['num']} | {item['config']} | {formula_short} | `{item['cmd'][:50]}...` |")

if __name__ == '__main__':
    main()
