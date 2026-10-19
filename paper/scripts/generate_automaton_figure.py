#!/usr/bin/env python3
"""
生成自动机流程图，展示HiLTL方法的优化点
"""

from pathlib import Path
import os
os.chdir(Path(__file__).resolve().parents[1] / 'figures')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Wedge
import numpy as np

# 设置字体
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

def create_automaton_diagram():
    """创建展示优化的自动机流程图"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # 标题
    ax.text(7, 9.5, 'LDBA with HiLTL Enhancements', 
            fontsize=16, fontweight='bold', ha='center')
    ax.text(7, 9.0, 'Example: LTL Formula: (!blue U green) & F yellow', 
            fontsize=11, ha='center', style='italic', color='#555')
    
    # 定义状态位置
    states = {
        'q0': (2, 6.5),    # 初始状态
        'q1': (5, 7.5),    # 中间状态1
        'q2': (5, 5.5),    # 中间状态2
        'q3': (8, 6.5),    # 接近目标
        'q_acc': (11, 6.5), # 接受状态
        'q_trap': (2, 3.5), # 陷阱状态
    }
    
    # 定义状态属性
    state_info = {
        'q0': {'label': '$q_0$\n(Initial)', 'color': '#E8F4F8', 'accepting': False, 'trap': False},
        'q1': {'label': '$q_1$\n(¬blue)', 'color': '#FFF9E6', 'accepting': False, 'trap': False},
        'q2': {'label': '$q_2$\n(progress)', 'color': '#FFF9E6', 'accepting': False, 'trap': False},
        'q3': {'label': '$q_3$\n(green)', 'color': '#E8F8E8', 'accepting': False, 'trap': False},
        'q_acc': {'label': '$q_{acc}$\n(yellow)', 'color': '#FFE6E6', 'accepting': True, 'trap': False},
        'q_trap': {'label': '$q_{trap}$\n(blue)', 'color': '#F0F0F0', 'accepting': False, 'trap': True},
    }
    
    # 绘制状态节点
    for state_name, (x, y) in states.items():
        info = state_info[state_name]
        
        # 节点特征标注 (SE-GNN增强)
        if info['accepting']:
            # 接受状态 - 双圈
            outer_circle = Circle((x, y), 0.5, color=info['color'], ec='red', linewidth=3, zorder=3)
            inner_circle = Circle((x, y), 0.42, fill=False, ec='red', linewidth=2, zorder=4)
            ax.add_patch(outer_circle)
            ax.add_patch(inner_circle)
            # 添加特征标注
            ax.text(x, y-0.9, 'is_accepting=1\nin_scc=1', 
                   fontsize=7, ha='center', style='italic', color='red',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFE6E6', alpha=0.7))
        elif info['trap']:
            # 陷阱状态
            circle = Circle((x, y), 0.5, color=info['color'], ec='gray', linewidth=2.5, 
                          linestyle='--', zorder=3)
            ax.add_patch(circle)
            # 添加特征标注
            ax.text(x, y-0.9, 'trap_state=1\ndist_to_acc=∞', 
                   fontsize=7, ha='center', style='italic', color='gray',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='#F0F0F0', alpha=0.7))
        else:
            # 普通状态
            circle = Circle((x, y), 0.5, color=info['color'], ec='#333', linewidth=2, zorder=3)
            ax.add_patch(circle)
            if state_name == 'q0':
                # 初始状态特殊标记
                ax.text(x, y-0.9, 'is_initial=1\ndist_to_acc=3', 
                       fontsize=7, ha='center', style='italic', color='blue',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F4F8', alpha=0.7))
        
        # 状态标签
        ax.text(x, y, info['label'], ha='center', va='center', 
               fontsize=10, fontweight='bold', zorder=5)
    
    # 定义转移边及其类型
    transitions = [
        # (from, to, label, edge_type, curve)
        ('q0', 'q1', '¬blue', 'forward', 0.2),
        ('q0', 'q_trap', 'blue', 'trap', -0.3),
        ('q1', 'q2', '¬blue', 'forward', 0.15),
        ('q1', 'q3', 'green', 'skip', 0.25),  # 跳跃边
        ('q2', 'q3', 'green', 'forward', 0),
        ('q3', 'q_acc', 'yellow', 'accepting', 0),
        ('q_trap', 'q_trap', 'any', 'self', None),  # 自环
        ('q3', 'q3', '¬yellow', 'self', None),  # 自环
    ]
    
    edge_styles = {
        'forward': {'color': '#2E86AB', 'style': '-', 'width': 2.5, 'label': 'chain_fwd'},
        'skip': {'color': '#A23B72', 'style': '--', 'width': 3, 'label': 'skip'},
        'accepting': {'color': '#F18F01', 'style': '-', 'width': 3, 'label': 'to_accepting'},
        'trap': {'color': '#666', 'style': '-.', 'width': 2, 'label': 'to_trap'},
        'self': {'color': '#555', 'style': '-', 'width': 1.5, 'label': 'self_loop'},
    }
    
    # 绘制转移边
    for trans in transitions:
        from_state, to_state, label, edge_type, curve = trans
        x1, y1 = states[from_state]
        x2, y2 = states[to_state]
        style = edge_styles[edge_type]
        
        if edge_type == 'self':
            # 自环
            if from_state == 'q_trap':
                loop = Wedge((x1, y1-0.5), 0.4, 200, 340, width=0.05, 
                           facecolor='none', edgecolor=style['color'], 
                           linewidth=style['width'], linestyle=style['style'])
                ax.add_patch(loop)
                ax.text(x1, y1-1.1, label, fontsize=8, ha='center', 
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
            else:
                loop = Wedge((x2, y2+0.5), 0.4, 20, 160, width=0.05,
                           facecolor='none', edgecolor=style['color'],
                           linewidth=style['width'], linestyle=style['style'])
                ax.add_patch(loop)
                ax.text(x2, y2+1.1, label, fontsize=8, ha='center',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        else:
            # 普通边
            if curve is not None and curve != 0:
                arrow = FancyArrowPatch(
                    (x1+0.5, y1), (x2-0.5, y2),
                    arrowstyle='->', mutation_scale=20,
                    connectionstyle=f"arc3,rad={curve}",
                    color=style['color'], linewidth=style['width'],
                    linestyle=style['style'], zorder=2
                )
            else:
                arrow = FancyArrowPatch(
                    (x1+0.5, y1), (x2-0.5, y2),
                    arrowstyle='->', mutation_scale=20,
                    color=style['color'], linewidth=style['width'],
                    linestyle=style['style'], zorder=2
                )
            ax.add_patch(arrow)
            
            # 边标签
            mid_x, mid_y = (x1+x2)/2, (y1+y2)/2
            if curve:
                mid_y += curve * 0.8
            ax.text(mid_x, mid_y+0.3, label, fontsize=9, ha='center',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    # 添加优化标注框
    # 1. SE-GNN特征增强
    ax.add_patch(FancyBboxPatch((0.3, 0.3), 3.5, 1.8, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#E8F4F8', edgecolor='#2E86AB', 
                                linewidth=2, alpha=0.3, zorder=1))
    ax.text(2.05, 1.8, '① SE-GNN Enhancement', fontsize=10, fontweight='bold', 
           ha='center', color='#2E86AB')
    ax.text(2.05, 1.45, '• Edge features (6D):', fontsize=8, ha='center')
    ax.text(2.05, 1.15, '  type + semantic distance', fontsize=8, ha='center')
    ax.text(2.05, 0.85, '• Node features (4D):', fontsize=8, ha='center')
    ax.text(2.05, 0.55, '  accepting + SCC + distance', fontsize=8, ha='center')
    
    # 2. Multi-Scale GNN
    ax.add_patch(FancyBboxPatch((4.5, 0.3), 3.2, 1.8,
                                boxstyle="round,pad=0.1",
                                facecolor='#FFF9E6', edgecolor='#A23B72',
                                linewidth=2, alpha=0.3, zorder=1))
    ax.text(6.1, 1.8, '② Multi-Scale GNN', fontsize=10, fontweight='bold',
           ha='center', color='#A23B72')
    ax.text(6.1, 1.45, '• Layer 0: local (1-hop)', fontsize=8, ha='center')
    ax.text(6.1, 1.15, '• Layer 1: mid-range (2-hop)', fontsize=8, ha='center')
    ax.text(6.1, 0.85, '• Layer 2: global', fontsize=8, ha='center')
    ax.text(6.1, 0.55, '→ Short/Mid/Long embeddings', fontsize=8, ha='center')
    
    # 3. Hierarchical Value
    ax.add_patch(FancyBboxPatch((8.2, 0.3), 3.5, 1.8,
                                boxstyle="round,pad=0.1",
                                facecolor='#FFE6E6', edgecolor='#F18F01',
                                linewidth=2, alpha=0.3, zorder=1))
    ax.text(9.95, 1.8, '③ Hierarchical Value', fontsize=10, fontweight='bold',
           ha='center', color='#F18F01')
    ax.text(9.95, 1.45, '• V_short(s): env state', fontsize=8, ha='center')
    ax.text(9.95, 1.15, '• V_long(q): automaton state', fontsize=8, ha='center')
    ax.text(9.95, 0.85, '• α(q): dynamic fusion', fontsize=8, ha='center')
    ax.text(9.95, 0.55, 'V = α·V_long + (1-α)·V_short', fontsize=8, ha='center')
    
    # 添加图例
    legend_elements = [
        mpatches.Patch(facecolor='#E8F4F8', edgecolor='#333', label='Initial State'),
        mpatches.Patch(facecolor='#FFF9E6', edgecolor='#333', label='Intermediate State'),
        mpatches.Patch(facecolor='#E8F8E8', edgecolor='#333', label='Progress State'),
        mpatches.Patch(facecolor='#FFE6E6', edgecolor='red', linewidth=2, label='Accepting State (SCC)'),
        mpatches.Patch(facecolor='#F0F0F0', edgecolor='gray', linestyle='--', label='Trap State'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8, 
             framealpha=0.9, title='State Types')
    
    # 添加边类型图例
    edge_legend_y = 8.2
    ax.text(0.5, edge_legend_y, 'Edge Types:', fontsize=9, fontweight='bold')
    for i, (etype, style) in enumerate(edge_styles.items()):
        y = edge_legend_y - 0.4 - i*0.3
        ax.plot([0.5, 1.2], [y, y], color=style['color'], 
               linewidth=style['width'], linestyle=style['style'])
        ax.text(1.4, y, style['label'], fontsize=8, va='center')
    
    plt.tight_layout()
    return fig

if __name__ == '__main__':
    # 生成图片
    fig = create_automaton_diagram()
    
    # 保存
    output_base = 'figure_automaton_enhanced'
    fig.savefig(f'{output_base}.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(f'{output_base}.png', dpi=300, bbox_inches='tight')
    
    print(f"✓ Generated: {output_base}.pdf")
    print(f"✓ Generated: {output_base}.png")
    
    plt.show()
