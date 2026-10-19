#!/usr/bin/env python3
"""
自动化批量评估脚本
用于第一层（Tier 1）和第三层（Tier 3）数据收集
支持：环境选择 -> 模型选择 -> 公式选择 -> 自动生成命令 -> 结果导入 CSV
"""

import os
import sys
import csv
from datetime import datetime
from pathlib import Path

# 配置定义
CONFIGS = {
    'baseline': {'flags': '', 'name': 'Baseline'},
    'hv_only': {'flags': '--enable_hierarchical_value', 'name': 'HV-only'},
    'hv_track': {'flags': '--enable_hierarchical_value --enable_automaton_tracking', 'name': 'HV+Track'},
    'msgnn_only': {'flags': '--enable_multiscale_gnn', 'name': 'MSGNN-only'},
    'msgnn_track': {'flags': '--enable_multiscale_gnn --enable_automaton_tracking', 'name': 'MSGNN+Track'},
    'full_no_options': {'flags': '--enable_hierarchical_value --enable_multiscale_gnn --enable_automaton_tracking', 'name': 'Full (no options)'},
    'full_all': {'flags': '--enable_hierarchical_value --enable_multiscale_gnn --enable_automaton_tracking --enable_options', 'name': 'Full (all)'},
}

# 环境和公式定义
ENVIRONMENTS = {
    'PointLtl2-v0': {
        'finite': [
            ('(!blue U green) & F yellow', '(!blue U green) ∧ F yellow'),
            ('F (blue & F (green & F yellow))', 'F (blue ∧ F (green ∧ F yellow))'),
            ('!(red | magenta) U (blue & F yellow)', '¬(red ∨ magenta) U (blue ∧ F yellow)'),
        ],
        'infinite': [
            ('G F (blue & green) & G F red', 'GF (blue ∧ green) ∧ GF red'),
            ('G F blue & G F green & G F yellow & G !magenta', 'GF blue ∧ GF green ∧ GF yellow ∧ G ¬magenta'),
        ],
    },
    'LetterEnv-v0': {
        'finite': [
            ('F (a & (!b U c)) & F d', 'F (a ∧ (¬b U c)) ∧ F d'),
            ('(F d) & (!f U (d & F b))', '(F d) ∧ (¬f U (d ∧ F b))'),
            ('F ((a | c | j) & F b) & F (c & F d) & F k', 'F ((a ∨ c ∨ j) ∧ F b) ∧ F (c ∧ F d) ∧ F k'),
        ],
        'infinite': [
            ('G F (e & (!a U f))', 'GF (e ∧ (¬a U f))'),
            ('G F a & G F b & G F c & G F d & G (!e & !f)', 'GF a ∧ GF b ∧ GF c ∧ GF d ∧ G (¬e ∧ ¬f)'),
        ],
    },
    'FlatWorld-v0': {
        'finite': [
            ('F ((red & magenta) & F ((blue & green) & F yellow))', 'F ((red ∧ magenta) ∧ F ((blue ∧ green) ∧ F yellow))'),
            ('F (orange & (!red U magenta))', 'F (orange ∧ (¬red U magenta))'),
            ('(!red U (green & blue & aqua)) & F (orange & (F (red & magenta)))', '(¬red U (green ∧ blue ∧ aqua)) ∧ F (orange ∧ (F (red ∧ magenta)))'),
        ],
        'infinite': [
            ('G F (blue & green) & G F (red & magenta)', 'GF (blue ∧ green) ∧ GF (red ∧ magenta)'),
            ('G F (aqua & blue) & G F red & G F yellow & G !green', 'GF (aqua ∧ blue) ∧ GF red ∧ GF yellow ∧ G ¬green'),
        ],
    },
}

class BatchEvaluator:
    def __init__(self):
        self.workspace_dir = Path('/home/owl/deep-ltl-paper')
        self.results_dir = self.workspace_dir / 'evaluation_results'
        self.results_dir.mkdir(exist_ok=True)
        
    def create_csv_if_needed(self, filepath):
        """创建 CSV 文件（如果不存在）"""
        if not filepath.exists():
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'Environment', 'Config', 'Formula', 'Formula_Display', 
                                'Seed', 'Horizon', 'Success_Rate', 'Value', 'Avg_Discounted_Return', 
                                'Avg_Steps', 'Deterministic'])
    
    def save_result(self, env, config, formula, formula_display, seed, horizon, sr, value, adr, steps):
        """保存单个结果到 CSV"""
        csv_file = self.results_dir / f'{env}_{horizon}_results.csv'
        self.create_csv_if_needed(csv_file)
        
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            deterministic = 'Yes' if env == 'PointLtl2-v0' else 'No'
            writer.writerow([
                datetime.now().isoformat(),
                env,
                config,
                formula,
                formula_display,
                seed,
                horizon,
                sr,
                value,
                adr,
                steps,
                deterministic
            ])
        print(f"✓ 结果已保存到 {csv_file}")
    
    def print_command(self, env, config, formula, seed, finite, deterministic):
        """生成并打印评估命令"""
        config_obj = CONFIGS[config]
        
        # 判断是否需要 --deterministic 标志
        det_flag = '--deterministic' if deterministic else '--no-deterministic'
        horizon_flag = '--finite' if finite else '--no-finite'
        
        cmd = (
            f"PYTHONPATH=src/ python src/evaluation/simulate.py "
            f"--env {env} --exp test --seed {seed} "
            f"--formula '{formula}' {horizon_flag} {det_flag}"
        )
        
        return cmd
    
    def run_interactive_session(self):
        """交互式评估会话"""
        print("\n" + "="*70)
        print("DeepLTL 批量评估工具")
        print("="*70)
        
        # 步骤 1: 选择环境
        print("\n[步骤 1] 选择环境:")
        env_list = list(ENVIRONMENTS.keys())
        for i, env in enumerate(env_list, 1):
            print(f"  {i}. {env}")
        env_choice = input("请选择 (1-3): ").strip()
        env = env_list[int(env_choice) - 1]
        print(f"✓ 已选择环境: {env}")
        
        # 步骤 2: 选择时间视野 (有限/无限)
        print("\n[步骤 2] 选择时间视野:")
        print("  1. 有限时域 (Finite-horizon)")
        print("  2. 无限时域 (Infinite-horizon)")
        horizon_choice = input("请选择 (1-2): ").strip()
        finite = (horizon_choice == '1')
        horizon_type = 'finite' if finite else 'infinite'
        print(f"✓ 已选择: {horizon_type}")
        
        # 步骤 3: 选择公式
        formulas = ENVIRONMENTS[env][horizon_type]
        print(f"\n[步骤 3] 选择公式 ({horizon_type}):")
        for i, (formula, display) in enumerate(formulas, 1):
            print(f"  {i}. {display}")
        
        formula_choice = input("请选择 (1-{}): ".format(len(formulas))).strip()
        formula, formula_display = formulas[int(formula_choice) - 1]
        print(f"✓ 已选择公式: {formula_display}")
        
        # 步骤 4: 选择模型配置
        print("\n[步骤 4] 选择模型配置:")
        config_list = list(CONFIGS.keys())
        for i, cfg in enumerate(config_list, 1):
            print(f"  {i}. {CONFIGS[cfg]['name']}")
        
        config_choice = input("请选择 (1-{}): ".format(len(config_list))).strip()
        config = config_list[int(config_choice) - 1]
        print(f"✓ 已选择配置: {CONFIGS[config]['name']}")
        
        # 步骤 5: 选择种子
        seed = input("\n[步骤 5] 输入种子编号 (通常为 1): ").strip()
        print(f"✓ 已选择种子: {seed}")
        
        # 步骤 6: 确定是否使用 deterministic
        print("\n[步骤 6] Deterministic 模式:")
        if env == 'PointLtl2-v0':
            print("  (ZoneEnv 默认使用 deterministic=True)")
            deterministic = True
        else:
            print("  1. 是 (Yes)")
            print("  2. 否 (No)")
            det_choice = input("请选择 (1-2, 默认为否): ").strip()
            deterministic = (det_choice == '1')
        print(f"✓ Deterministic: {deterministic}")
        
        # 生成命令
        cmd = self.print_command(env, config, formula, seed, finite, deterministic)
        print("\n" + "="*70)
        print("[生成的命令]")
        print("="*70)
        print(f"\n{cmd}\n")
        
        print("请在终端执行上述命令，并记录输出结果。")
        print("输出格式应包含: S=0.xxx, V=0.xxx, ADR=0.xxx, AS=xxx\n")
        
        # 步骤 7: 输入结果
        print("="*70)
        print("[输入结果]")
        print("="*70)
        
        try:
            sr = float(input("Success Rate (S): "))
            value = float(input("Value (V): "))
            adr = float(input("Avg Discounted Return (ADR): "))
            steps = float(input("Avg Steps (AS): "))
            
            # 保存结果
            self.save_result(
                env=env,
                config=config,
                formula=formula,
                formula_display=formula_display,
                seed=seed,
                horizon=horizon_type,
                sr=sr,
                value=value,
                adr=adr,
                steps=steps
            )
            
            print("\n✓ 结果已成功保存！")
            
        except ValueError:
            print("\n✗ 输入格式错误，请重新输入数字值")
            return False
        
        return True
    
    def print_summary(self):
        """打印已保存结果的摘要"""
        print("\n" + "="*70)
        print("[已保存的结果摘要]")
        print("="*70)
        
        csv_files = list(self.results_dir.glob('*.csv'))
        if not csv_files:
            print("暂无已保存的结果")
            return
        
        for csv_file in sorted(csv_files):
            print(f"\n文件: {csv_file.name}")
            try:
                with open(csv_file, 'r') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    print(f"  共 {len(rows)} 条记录")
                    if rows:
                        # 按环境和配置分组显示
                        configs_seen = set()
                        for row in rows[-3:]:  # 显示最后 3 条
                            key = (row['Environment'], row['Config'])
                            if key not in configs_seen:
                                print(f"    {row['Config']:20} | SR={row['Success_Rate']:6} | AS={row['Avg_Steps']:8}")
                                configs_seen.add(key)
            except Exception as e:
                print(f"  读取失败: {e}")

def main():
    evaluator = BatchEvaluator()
    
    while True:
        print("\n" + "="*70)
        print("主菜单")
        print("="*70)
        print("1. 开始新的评估会话")
        print("2. 查看已保存结果摘要")
        print("3. 退出")
        
        choice = input("\n请选择 (1-3): ").strip()
        
        if choice == '1':
            evaluator.run_interactive_session()
            again = input("\n是否继续进行下一个评估？(y/n): ").strip().lower()
            if again != 'y':
                break
        elif choice == '2':
            evaluator.print_summary()
        elif choice == '3':
            print("\n再见！")
            break
        else:
            print("\n✗ 选择无效")

if __name__ == '__main__':
    main()
