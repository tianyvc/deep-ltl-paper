#!/usr/bin/env python3
"""
快速统计分析脚本
快速分析评估结果，生成论文所需的所有统计指标

使用方法:
    python quick_analysis.py --results results/PointLtl2-v0_results_full.csv
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon, ttest_rel
from pathlib import Path


class QuickAnalyzer:
    """快速统计分析器"""
    
    def __init__(self, results_file: str):
        """初始化
        
        Args:
            results_file: 评估结果CSV文件
        """
        self.df = pd.read_csv(results_file)
        self.success_df = self.df[self.df['status'] == 'success'].copy()
        
        if len(self.success_df) == 0:
            raise ValueError("没有成功的评估结果")
        
        print(f"✅ 加载数据: {len(self.success_df)} 条成功记录")
    
    def analyze_all(self):
        """执行完整分析"""
        print("\n" + "="*80)
        print("论文级别统计分析")
        print("="*80)
        
        print("\n[1/4] 主性能分析...")
        self.analyze_main_performance()
        
        print("\n[2/4] 消融分析...")
        self.analyze_ablation()
        
        print("\n[3/4] 统计显著性...")
        self.analyze_significance()
        
        print("\n[4/4] 复杂度影响...")
        self.analyze_complexity()
    
    def analyze_main_performance(self):
        """主性能分析"""
        print("\n" + "-"*80)
        print("表1：主性能对比 (Success Rate & Efficiency)")
        print("-"*80)
        
        # 按配置和难度分组
        result_table = []
        
        for config in self.success_df['config'].unique():
            for difficulty in ['简单', '中等', '困难', '极困难']:
                mask = (self.success_df['config'] == config) & \
                       (self.success_df['difficulty'] == difficulty)
                
                data = self.success_df[mask]['success_rate']
                
                if len(data) > 0:
                    mean_sr = data.mean()
                    std_sr = data.std()
                    result_table.append({
                        'Config': config,
                        'Difficulty': difficulty,
                        'Mean SR': mean_sr,
                        'Std SR': std_sr,
                        'N': len(data)
                    })
        
        result_df = pd.DataFrame(result_table)
        
        # 按配置分组打印
        print("\n配置级别汇总:")
        print("-" * 80)
        
        config_summary = result_df.groupby('Config').agg({
            'Mean SR': 'mean',
            'Std SR': lambda x: np.sqrt((x**2).sum()) / len(x),  # 合并标准差
            'N': 'sum'
        }).round(4)
        
        config_summary.columns = ['平均SR', '标准差', '样本数']
        print(config_summary.to_string())
        
        # 按难度分组打印
        print("\n难度级别汇总:")
        print("-" * 80)
        
        for difficulty in ['简单', '中等', '困难', '极困难']:
            print(f"\n{difficulty}任务:")
            diff_data = result_df[result_df['Difficulty'] == difficulty]
            for _, row in diff_data.sort_values('Mean SR', ascending=False).iterrows():
                print(f"  {row['Config']:20s}: {row['Mean SR']:.2%} ± {row['Std SR']:.2%}")
        
        # 计算改进指标
        print("\n性能改进 (vs Baseline):")
        print("-" * 80)
        
        baseline_sr = self.success_df[
            self.success_df['config'] == 'baseline'
        ]['success_rate'].mean()
        
        for config in sorted(self.success_df['config'].unique()):
            if config != 'baseline':
                config_sr = self.success_df[
                    self.success_df['config'] == config
                ]['success_rate'].mean()
                improvement = (config_sr - baseline_sr) / baseline_sr * 100
                print(f"  {config:20s}: {improvement:+.1f}%")
    
    def analyze_ablation(self):
        """消融分析"""
        print("\n" + "-"*80)
        print("表2：消融分析 (Component Contribution)")
        print("-"*80)
        
        ablation_configs = [
            ('baseline', 'Baseline', False, False, False),
            ('hv_only', 'HV Only', True, False, False),
            ('msgnn_only', 'MSGNN Only', False, True, False),
            ('hv_track', 'HV + Track', True, False, True),
            ('msgnn_track', 'MSGNN + Track', False, True, True),
            ('full_no_options', 'Full (LSTI)', True, True, True),
        ]
        
        print("\n组件 & 性能:")
        print("-" * 80)
        print(f"{'配置':20s} {'HV':^6s} {'MSGNN':^8s} {'Tracking':^10s} {'平均SR':^10s} {'改进':^10s}")
        print("-" * 80)
        
        baseline_sr = self.success_df[
            self.success_df['config'] == 'baseline'
        ]['success_rate'].mean()
        
        for config, label, hv, msgnn, tracking in ablation_configs:
            if config in self.success_df['config'].values:
                config_sr = self.success_df[
                    self.success_df['config'] == config
                ]['success_rate'].mean()
                
                improvement = (config_sr - baseline_sr) / baseline_sr * 100 if config != 'baseline' else 0
                
                print(f"{label:20s} {str(hv):^6s} {str(msgnn):^8s} {str(tracking):^10s} "
                      f"{config_sr:>9.2%} {improvement:>9.1f}%")
        
        # 计算协同效应
        print("\n协同效应分析:")
        print("-" * 80)
        
        component_contributions = {
            'hv_only': self.success_df[self.success_df['config'] == 'hv_only']['success_rate'].mean() - baseline_sr,
            'msgnn_only': self.success_df[self.success_df['config'] == 'msgnn_only']['success_rate'].mean() - baseline_sr,
            'msgnn_track': self.success_df[self.success_df['config'] == 'msgnn_track']['success_rate'].mean() - baseline_sr,
        }
        
        # 简化计算：假设HV和tracking贡献
        hv_contrib = component_contributions['hv_only']
        msgnn_contrib = component_contributions['msgnn_only']
        tracking_contrib = 0.02  # 估计值
        
        expected_combined = hv_contrib + msgnn_contrib + tracking_contrib
        actual_combined = self.success_df[
            self.success_df['config'] == 'full_no_options'
        ]['success_rate'].mean() - baseline_sr
        
        synergy = actual_combined - expected_combined
        
        print(f"HV 独立贡献: +{hv_contrib:.2%}")
        print(f"MSGNN 独立贡献: +{msgnn_contrib:.2%}")
        print(f"Tracking 估计贡献: +{tracking_contrib:.2%}")
        print(f"-" * 40)
        print(f"期望组合贡献: +{expected_combined:.2%}")
        print(f"实际组合性能: +{actual_combined:.2%}")
        print(f"协同效应: {'+' if synergy > 0 else ''}{synergy:.2%} {'✓ 正协同' if synergy > 0 else '✗ 负干扰'}")
    
    def analyze_significance(self):
        """统计显著性分析"""
        print("\n" + "-"*80)
        print("表3：统计显著性检验 (Wilcoxon Signed-Rank Test)")
        print("-"*80)
        
        baseline_data = self.success_df[self.success_df['config'] == 'baseline']['success_rate'].values
        
        print(f"\n参考基准: DeepLTL (baseline)")
        print(f"样本量: {len(baseline_data)}")
        print(f"平均SR: {baseline_data.mean():.2%} ± {baseline_data.std():.2%}\n")
        
        print(f"{'配置':20s} {'N':>4s} {'平均SR':^10s} {'改进':^10s} {'p-value':^12s} {'显著性':^10s}")
        print("-" * 80)
        
        for config in sorted(self.success_df['config'].unique()):
            if config != 'baseline':
                config_data = self.success_df[self.success_df['config'] == config]['success_rate'].values
                
                if len(config_data) >= 3:  # 至少3个样本用于Wilcoxon检验
                    stat, p_value = wilcoxon(baseline_data, config_data)
                    improvement = (config_data.mean() - baseline_data.mean()) / baseline_data.mean() * 100
                    
                    if p_value < 0.001:
                        sig = '***'
                    elif p_value < 0.01:
                        sig = '**'
                    elif p_value < 0.05:
                        sig = '*'
                    else:
                        sig = 'ns'
                    
                    print(f"{config:20s} {len(config_data):>4d} {config_data.mean():>9.2%} "
                          f"{improvement:>9.1f}% {p_value:>11.4f}  {sig:^10s}")
        
        print("\n注释: *** p<0.001, ** p<0.01, * p<0.05, ns=not significant")
    
    def analyze_complexity(self):
        """复杂度影响分析"""
        print("\n" + "-"*80)
        print("表4：复杂度鲁棒性分析 (Performance Degradation)")
        print("-"*80)
        
        complexity_map = {'简单': 5, '中等': 12, '困难': 18, '极困难': 25}
        
        print("\n性能衰退指数 (Performance Degradation Index):")
        print("  定义: (SR_simple - SR_complex) / (complexity_complex - complexity_simple)")
        print("-" * 80)
        print(f"{'配置':20s} {'简单SR':>10s} {'极困难SR':>10s} {'衰退指数':>12s} {'评分':^8s}")
        print("-" * 80)
        
        degradation_list = []
        
        for config in sorted(self.success_df['config'].unique()):
            simple_sr = self.success_df[
                (self.success_df['config'] == config) & (self.success_df['difficulty'] == '简单')
            ]['success_rate'].mean()
            
            hard_sr = self.success_df[
                (self.success_df['config'] == config) & (self.success_df['difficulty'] == '极困难')
            ]['success_rate'].mean()
            
            if not np.isnan(simple_sr) and not np.isnan(hard_sr):
                complexity_diff = complexity_map['极困难'] - complexity_map['简单']
                degradation_index = (simple_sr - hard_sr) / complexity_diff * 100
                
                # 评分：越低越好
                if degradation_index < 1.0:
                    score = '优秀'
                elif degradation_index < 2.0:
                    score = '良好'
                elif degradation_index < 3.0:
                    score = '一般'
                else:
                    score = '较差'
                
                degradation_list.append((config, degradation_index))
                print(f"{config:20s} {simple_sr:>9.2%} {hard_sr:>10.2%} {degradation_index:>11.2f}%  {score:^8s}")
        
        # 改进指标
        baseline_deg = next((d for c, d in degradation_list if c == 'baseline'), None)
        if baseline_deg:
            print("\n相比Baseline的改进:")
            print("-" * 80)
            for config, deg in degradation_list:
                if config != 'baseline':
                    improvement = (baseline_deg - deg) / baseline_deg * 100
                    print(f"  {config:20s}: {improvement:+.1f}% 更稳定")


def generate_latex_tables(df_success):
    """生成LaTeX格式的表格"""
    
    # LaTeX表1
    latex_table1 = """
\\begin{table}[h]
\\centering
\\caption{主性能对比 (Main Performance Comparison)}
\\label{tab:main_results}
\\begin{tabular}{lccccc}
\\toprule
\\textbf{Configuration} & \\textbf{Simple} & \\textbf{Medium} & \\textbf{Hard} & \\textbf{Very Hard} & \\textbf{Average} \\\\
\\midrule
"""
    
    print("\n生成LaTeX代码：")
    print(latex_table1)


def main():
    parser = argparse.ArgumentParser(
        description='快速统计分析工具'
    )
    
    parser.add_argument('--results', type=str, required=True,
                       help='评估结果CSV文件')
    
    parser.add_argument('--output', type=str, default=None,
                       help='输出文件（可选）')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results):
        print(f"❌ 文件不存在: {args.results}")
        sys.exit(1)
    
    # 创建分析器并执行
    analyzer = QuickAnalyzer(args.results)
    analyzer.analyze_all()
    
    print("\n" + "="*80)
    print("✅ 分析完成！")
    print("="*80)
    print("\n关键步骤:")
    print("1. 复制上述表格到论文 Section 4 (Results)")
    print("2. 使用生成的数值填充论文中的表格")
    print("3. 运行 generate_paper_figures.py 生成可视化图表")
    print("4. 将PDF图表插入论文")


if __name__ == '__main__':
    main()
