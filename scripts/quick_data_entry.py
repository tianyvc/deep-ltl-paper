#!/usr/bin/env python3
"""
快速数据录入和可视化脚本

使用方法：
    # 步骤1：运行测试命令并记录结果
    python scripts/quick_data_entry.py --record
    
    # 步骤2：生成图表
    python scripts/quick_data_entry.py --visualize
"""

import argparse
import csv
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 配置
CSV_FILE = 'results/PointLtl2-v0_results_paper.csv'
CONFIGS = ['baseline', 'hv_only', 'msgnn_only', 'hv_track', 'msgnn_track', 'full_no_options', 'full_all']
CONFIG_LABELS = {
    'baseline': 'Baseline',
    'hv_only': 'HV Only',
    'msgnn_only': 'MSGNN Only',
    'hv_track': 'HV+Track',
    'msgnn_track': 'MSGNN+Track',
    'full_no_options': 'Full (No Options)',
    'full_all': 'Full (All)'
}

# 任务定义（与evaluate_all_configs.py保持一致）
TASKS = {
    'finite': [
        ('F blue & F green', 'finite', '简单'),
        ('(F blue) & (F green) & (F yellow)', 'finite', '中等'),
        ('(!blue U green) & F yellow', 'finite', '困难'),
        ('GF (blue & green) & G !yellow', 'finite', '极困难'),
    ],
    'infinite': [
        ('GF blue', 'infinite', '简单'),
        ('GF blue & GF green & GF yellow', 'infinite', '困难'),
        ('GF (blue & green) & GF (red & magenta) & G !yellow', 'infinite', '极困难'),
    ]
}


def record_data():
    """交互式数据录入"""
    print("\n" + "="*70)
    print("📊 快速数据录入模式")
    print("="*70)
    
    # 确保目录存在
    os.makedirs(os.path.dirname(CSV_FILE), exist_ok=True)
    
    # 检查文件是否存在
    file_exists = os.path.exists(CSV_FILE)
    
    print(f"\n数据将保存到: {CSV_FILE}")
    print(f"文件状态: {'已存在' if file_exists else '将新建'}")
    
    if file_exists:
        print("\n当前已有数据：")
        with open(CSV_FILE, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            print(f"  总记录数: {len(rows)}")
            configs_done = set(row['config'] for row in rows)
            print(f"  已完成配置: {', '.join(sorted(configs_done))}")
    
    print("\n" + "-"*70)
    print("请准备测试命令，格式：")
    print("  PYTHONPATH=src/ python src/evaluation/simulate.py --env PointLtl2-v0 \\")
    print("    --exp CONFIG --seed 1 --formula 'FORMULA' --finite --deterministic")
    print("-"*70)
    
    # 创建或打开CSV文件（添加status列以兼容分析脚本）
    fieldnames = ['config', 'task', 'task_type', 'difficulty', 'seed',
                  'success_rate', 'violation_rate', 'avg_return', 'avg_steps', 'status']
    
    with open(CSV_FILE, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        while True:
            print("\n" + "="*70)
            print("输入 'q' 退出 | 输入 's' 查看状态 | 或继续录入数据")
            print("="*70)
            
            config = input("\n配置名称 (baseline/hv_only/...): ").strip()
            
            if config.lower() == 'q':
                print("\n✅ 数据录入完成！")
                break
            
            if config.lower() == 's':
                show_status()
                continue
            
            if config not in CONFIGS:
                print(f"⚠️  配置 '{config}' 不在标准列表中")
                continue
            
            # 让用户选择任务
            print("\n请选择任务类型:")
            print("  1. Finite (有限) - 4个任务")
            print("  2. Infinite (无限) - 3个任务")
            
            task_type_choice = input("选择 (1/2): ").strip()
            
            if task_type_choice == '1':
                task_category = 'finite'
            elif task_type_choice == '2':
                task_category = 'infinite'
            else:
                print("❌ 无效选择")
                continue
            
            # 显示可用任务
            print(f"\n{task_category.upper()} 任务列表:")
            for idx, (formula, ttype, diff) in enumerate(TASKS[task_category], 1):
                print(f"  {idx}. [{diff}] {formula}")
            
            task_idx = input(f"\n选择任务 (1-{len(TASKS[task_category])}): ").strip()
            
            try:
                task_idx = int(task_idx) - 1
                if task_idx < 0 or task_idx >= len(TASKS[task_category]):
                    print("❌ 任务编号超出范围")
                    continue
                
                task, task_type, difficulty = TASKS[task_category][task_idx]
            except ValueError:
                print("❌ 请输入数字")
                continue
            
            # 种子选择
            seed_input = input("\n种子 (默认1): ").strip()
            seed = int(seed_input) if seed_input else 1
            
            print(f"\n当前测试：")
            print(f"  配置: {config}")
            print(f"  任务: {task}")
            print(f"  类型: {task_type}")
            print(f"  难度: {difficulty}")
            print(f"  种子: {seed}")
            
            # 根据任务类型生成命令
            if task_type == 'finite':
                finite_flag = '--finite'
                print(f"\n📋 运行命令（有限时域）：")
            else:
                finite_flag = '--no-finite'
                print(f"\n📋 运行命令（无限时域）：")
            
            print(f"PYTHONPATH=src/ python src/evaluation/simulate.py --env PointLtl2-v0 --exp {config} --seed {seed} --formula '{task}' {finite_flag} --deterministic")
            
            if task_type == 'finite':
                result_input = input("\n粘贴结果 (4个值, 格式: 0.858,0.094,0.602,180.737): ").strip()
            else:
                result_input = input("\n粘贴结果 (1个值, 格式: 0.523): ").strip()
            
            # 处理可能的前缀
            if ':' in result_input:
                result_input = result_input.split(':', 1)[1].strip()
            
            try:
                parts = result_input.split(',')
                
                # 支持有限和无限时域两种输出格式
                if len(parts) == 4:
                    # 有限时域格式: success_rate,violation_rate,avg_return,avg_steps
                    success_rate = float(parts[0])
                    violation_rate = float(parts[1])
                    avg_return = float(parts[2])
                    avg_steps = float(parts[3])
                    
                    print("\n✅ 解析成功（有限时域）：")
                    print(f"  成功率: {success_rate:.3f} ({success_rate*100:.1f}%)")
                    print(f"  违反率: {violation_rate:.3f} ({violation_rate*100:.1f}%)")
                    print(f"  平均回报: {avg_return:.3f}")
                    print(f"  平均步数: {avg_steps:.1f}")
                    
                elif len(parts) == 1:
                    # 无限时域格式: average_visits (单个值)
                    avg_visits = float(parts[0])
                    success_rate = avg_visits  # 重用字段
                    violation_rate = 0.0
                    avg_return = avg_visits
                    avg_steps = 0.0
                    
                    print("\n✅ 解析成功（无限时域）：")
                    print(f"  平均访问数: {avg_visits:.3f}")
                    
                else:
                    print(f"❌ 错误：需要1个或4个值，得到{len(parts)}个")
                    continue
                
                confirm = input("\n确认保存? (y/n, 默认y): ").strip().lower()
                if confirm in ['', 'y', 'yes']:
                    row = {
                        'config': config,
                        'task': task,
                        'task_type': task_type,
                        'difficulty': difficulty,
                        'seed': seed,
                        'success_rate': success_rate,
                        'violation_rate': violation_rate,
                        'avg_return': avg_return,
                        'avg_steps': avg_steps,
                        'status': 'success'  # 兼容分析脚本
                    }
                    writer.writerow(row)
                    csvfile.flush()  # 立即写入磁盘
                    print("✅ 已保存！")
                else:
                    print("❌ 已取消")
                
            except ValueError as e:
                print(f"❌ 解析错误: {e}")
                continue


def show_status():
    """显示当前数据收集状态"""
    if not os.path.exists(CSV_FILE):
        print("\n⚠️  还没有数据文件")
        return
    
    with open(CSV_FILE, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print("\n" + "="*70)
    print("📊 数据收集状态")
    print("="*70)
    print(f"总记录数: {len(rows)}")
    
    configs_done = {}
    for row in rows:
        config = row['config']
        configs_done[config] = configs_done.get(config, 0) + 1
    
    print(f"\n各配置完成情况:")
    for config in CONFIGS:
        status = "✅" if config in configs_done else "❌"
        count = configs_done.get(config, 0)
        print(f"  {status} {config:20s} : {count} 条记录")
    
    print("\n需要完成的配置:")
    remaining = [c for c in CONFIGS if c not in configs_done]
    if remaining:
        for i, config in enumerate(remaining, 1):
            print(f"  {i}. {config}")
    else:
        print("  🎉 所有配置已完成！")


def visualize_data():
    """生成可视化图表"""
    if not os.path.exists(CSV_FILE):
        print(f"\n❌ 数据文件不存在: {CSV_FILE}")
        print("请先运行 --record 模式录入数据")
        return
    
    # 读取数据
    with open(CSV_FILE, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    if not rows:
        print("\n❌ CSV文件为空")
        return
    
    print(f"\n✅ 读取到 {len(rows)} 条记录")
    
    # 按配置分组
    data_by_config = {}
    for row in rows:
        config = row['config']
        if config not in data_by_config:
            data_by_config[config] = []
        data_by_config[config].append(row)
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('DeepLTL-LS Performance Comparison\nTask: (!blue U green) & F yellow', 
                 fontsize=16, fontweight='bold')
    
    # 准备数据
    configs_present = [c for c in CONFIGS if c in data_by_config]
    
    success_rates = []
    violation_rates = []
    avg_returns = []
    avg_steps = []
    
    for config in configs_present:
        config_data = data_by_config[config]
        success_rates.append(np.mean([float(r['success_rate']) for r in config_data]))
        violation_rates.append(np.mean([float(r['violation_rate']) for r in config_data]))
        avg_returns.append(np.mean([float(r['avg_return']) for r in config_data]))
        avg_steps.append(np.mean([float(r['avg_steps']) for r in config_data]))
    
    labels = [CONFIG_LABELS.get(c, c) for c in configs_present]
    x = np.arange(len(labels))
    
    # 1. 成功率
    ax1 = axes[0, 0]
    bars1 = ax1.bar(x, success_rates, color='steelblue', alpha=0.8)
    ax1.set_ylabel('Success Rate', fontsize=12)
    ax1.set_title('Success Rate Comparison', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    ax1.set_ylim([0, 1.0])
    ax1.grid(axis='y', alpha=0.3)
    # 添加数值标签
    for bar, val in zip(bars1, success_rates):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 2. 违反率
    ax2 = axes[0, 1]
    bars2 = ax2.bar(x, violation_rates, color='coral', alpha=0.8)
    ax2.set_ylabel('Violation Rate', fontsize=12)
    ax2.set_title('Violation Rate Comparison', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    ax2.set_ylim([0, max(violation_rates) * 1.2 if violation_rates else 0.2])
    ax2.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars2, violation_rates):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 3. 平均回报
    ax3 = axes[1, 0]
    bars3 = ax3.bar(x, avg_returns, color='seagreen', alpha=0.8)
    ax3.set_ylabel('Average Discounted Return', fontsize=12)
    ax3.set_title('Average Return Comparison', fontsize=13, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, rotation=45, ha='right')
    ax3.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars3, avg_returns):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 4. 平均步数
    ax4 = axes[1, 1]
    bars4 = ax4.bar(x, avg_steps, color='mediumpurple', alpha=0.8)
    ax4.set_ylabel('Average Steps', fontsize=12)
    ax4.set_title('Average Steps Comparison', fontsize=13, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels, rotation=45, ha='right')
    ax4.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars4, avg_steps):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # 保存图表
    output_dir = 'results/figures'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'performance_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ 图表已保存到: {output_file}")
    
    # 显示图表
    plt.show()
    
    # 打印统计表
    print("\n" + "="*80)
    print("📊 性能统计表")
    print("="*80)
    print(f"{'Config':<20} {'Success Rate':>12} {'Violation':>10} {'Avg Return':>11} {'Avg Steps':>10}")
    print("-"*80)
    for config, sr, vr, ar, steps in zip(configs_present, success_rates, violation_rates, avg_returns, avg_steps):
        label = CONFIG_LABELS.get(config, config)
        print(f"{label:<20} {sr:>12.3f} {vr:>10.3f} {ar:>11.3f} {steps:>10.1f}")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='快速数据录入和可视化工具')
    parser.add_argument('--record', action='store_true', help='数据录入模式')
    parser.add_argument('--visualize', action='store_true', help='生成可视化')
    parser.add_argument('--status', action='store_true', help='查看状态')
    
    args = parser.parse_args()
    
    if args.record:
        record_data()
    elif args.visualize:
        visualize_data()
    elif args.status:
        show_status()
    else:
        print("请选择模式:")
        print("  --record      : 录入数据")
        print("  --visualize   : 生成图表")
        print("  --status      : 查看状态")


if __name__ == '__main__':
    main()
