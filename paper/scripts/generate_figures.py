#!/usr/bin/env python3
"""
Generate publication-quality figures for HiLTL paper
Creates: SE-GNN architecture, long-short term fusion, automaton hierarchy, fusion weight dynamics

Upgrades:
- Unified theme (fonts, line widths, colors) for top-tier publication quality
- Helper primitives for consistent boxes/arrows/labels
- Carefully tuned layouts to eliminate overlaps and uneven spacing
- Export both PNG (high DPI) and PDF (vector) assets
"""

import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import numpy as np
from matplotlib.patheffects import withStroke
import seaborn as sns
from matplotlib import rcParams

# -------------------- Global Theme & Helpers --------------------
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'figures'))

# Global style (publication-friendly)
rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.family": "DejaVu Sans",
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "legend.fontsize": 14,
    "figure.titlesize": 18,
    "lines.linewidth": 2.4,
    "axes.linewidth": 1.4,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "mathtext.default": "regular",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Seaborn palette (colorblind-friendly)
try:
    sns.set_theme(style="whitegrid", font_scale=1.0)
    palette = sns.color_palette("colorblind")
except Exception:
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf",
    ]

COLORS = {
    "accent1": palette[0],  # blue
    "accent2": palette[1],  # orange
    "accent3": palette[2],  # green
    "accent4": palette[3],  # red
    "accent5": palette[4],  # purple
    "accent6": palette[5],  # brown
    "muted": "#999999",
    "yellow": "#FFE66D",
    "teal": "#4ECDC4",
    "teal_light": "#95E1D3",
    "pink": "#F38181",
    "lavender": "#AA96DA",
    "peach": "#FFA07A",
    "wheat": "#F5DEB3",
    "lightyellow": "#FFF7CC",
}


def _ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_figure(fig, basename: str):
    """Save both PNG (high DPI) and PDF (vector) with tight bounding box."""
    _ensure_output_dir()
    png_path = os.path.join(OUTPUT_DIR, f"{basename}.png")
    pdf_path = os.path.join(OUTPUT_DIR, f"{basename}.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    print(f"✅ Saved: {os.path.basename(png_path)} and {os.path.basename(pdf_path)}")


def draw_box(ax, xy, w, h, text, fc, ec="black", lw=1.8, alpha=0.95,
             fontsize=10, weight="bold", color="black", boxstyle="round,pad=0.12"):
    """Draw a rounded box with centered text using axes coordinates.
    xy, w, h are in data coordinates of the axis.
    """
    rect = FancyBboxPatch(xy, w, h, boxstyle=boxstyle, edgecolor=ec,
                          facecolor=fc, linewidth=lw, alpha=alpha)
    ax.add_patch(rect)
    ax.text(xy[0] + w/2, xy[1] + h/2, text, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, color=color)
    return rect


def draw_arrow(ax, xy_from, xy_to, color="black", lw=2, style="->", ms=20, connectionstyle="arc3"):
    arrow = FancyArrowPatch(xy_from, xy_to, arrowstyle=style, mutation_scale=ms,
                            linewidth=lw, color=color, connectionstyle=connectionstyle)
    ax.add_patch(arrow)
    return arrow


def add_title(ax, text):
    """Simple panel title without background boxes."""
    ax.text(0.5, 1.02, text, fontsize=13, fontweight="bold", ha="center", va="bottom",
            transform=ax.transAxes)

# ==================== Figure 1: SE-GNN Architecture ====================
def figure_1_se_gnn_architecture():
    """SE-GNN architecture with clean, symmetric layout and no overlaps."""
    fig, axes = plt.subplots(1, 2, figsize=(16.5, 7.2), constrained_layout=True)

    # Left panel: Input features & semantics
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    add_title(ax, 'Input Features & Semantics')

    # Node features (left column)
    node_labels = ['Accept', 'SCC', 'Init', 'Distance']
    node_colors = [COLORS['pink'], COLORS['teal'], COLORS['accent1'], COLORS['peach']]
    y0 = 8.2
    dy = 1.2
    x_circle = 2.0
    r = 0.65
    for i, (label, color) in enumerate(zip(node_labels, node_colors)):
        y = y0 - i * dy
        circle = Circle((x_circle, y), r, facecolor=color, ec='black', linewidth=2.2)
        ax.add_patch(circle)
        ax.text(x_circle + 1.3, y, label, fontsize=18, va='center', fontweight='bold', color='#222222')

    ax.text(1.0, 2.5, 'Node Features\n$h_v \\in \\mathbb{R}^4$', fontsize=17, fontweight='bold', ha='left', color='#222222')

    # Edge features (right column)
    edge_labels = ['ε-Transition', 'Forward', 'Skip', 'Backward']
    edge_colors = [COLORS['teal_light'], COLORS['pink'], COLORS['lavender'], COLORS['accent5']]
    x_rect = 6.2
    w_rect, h_rect = 1.2, 0.85
    for i, (label, color) in enumerate(zip(edge_labels, edge_colors)):
        y = y0 - i * dy
        rect = Rectangle((x_rect, y - h_rect/2), w_rect, h_rect, facecolor=color, ec='black', linewidth=1.8)
        ax.add_patch(rect)
        ax.text(x_rect + w_rect + 0.7, y, label, fontsize=18, va='center', fontweight='bold', ha='left', color='#222222')

    ax.text(5.8, 2.5, 'Edge Features\n$h_e \\in \\mathbb{R}^6$', fontsize=17, fontweight='bold', ha='left', color='#222222')

    # Right panel: GNN processing pipeline
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    add_title(ax, 'SE-GNN Processing Pipeline')

    layer_names = ['GATv2 Layer 1\n(1-hop)', 'GATv2 Layer 2\n(2-hop)', 'GATv2 Layer 3\n(3-hop)', 'Readout Pool']
    layer_colors = [COLORS['yellow'], COLORS['teal_light'], COLORS['pink'], COLORS['lavender']]
    x_box, w_box, h_box = 2.2, 5.6, 1.3
    y_start, y_gap = 8.1, 1.9
    prev_center = None
    for i, (name, color) in enumerate(zip(layer_names, layer_colors)):
        y_box = y_start - i * y_gap
        draw_box(ax, (x_box, y_box - h_box/2), w_box, h_box, name, fc=color, fontsize=18)
        if prev_center is not None:
            draw_arrow(ax, (x_box + w_box/2, prev_center - h_box/2 - 0.15),
                       (x_box + w_box/2, y_box + h_box/2 + 0.15), color="#333", lw=2.8)
        prev_center = y_box

        ax.text(5.0, 0.7,
            r'$e_{ij} = a^\top\,\mathrm{LeakyReLU}(W_q h_i + W_k h_j + W_e h_e)$',
            fontsize=18, ha='center', style='italic', fontweight='bold', color='#222222')

    save_figure(fig, 'figure1_se_gnn')
    plt.close(fig)

# ==================== Figure 2: Long-Short Term Fusion ====================
def figure_2_long_short_fusion():
    """Long-short term fusion with symmetric routing and balanced spacing."""
    fig, ax = plt.subplots(figsize=(15.8, 8.2), constrained_layout=True)

    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis('off')

    # Input state
    draw_box(ax, (0.4, 3.6), 2.4, 1.8, '(s, q)', fc=COLORS['yellow'], fontsize=22)

    # Short-term branch (upper)
    draw_arrow(ax, (2.8, 4.6), (3.8, 6.2), color=COLORS['teal'], lw=3.2, ms=30)
    draw_box(ax, (3.8, 5.3), 3.8, 1.8, 'Multi-scale GNN\n(MSGNN)', fc=COLORS['teal'], fontsize=22)
    draw_arrow(ax, (5.7, 5.3), (5.7, 4.5), color=COLORS['teal'], lw=3.0, ms=27)
    draw_box(ax, (3.8, 3.8), 3.8, 1.8, 'V_short Critic\nNetwork', fc=COLORS['teal_light'], fontsize=22)

    # Long-term branch (lower)
    draw_arrow(ax, (2.8, 3.2), (3.8, 1.2), color=COLORS['pink'], lw=3.2, ms=30)
    draw_box(ax, (3.8, 0.9), 3.8, 1.8, 'Automaton Tracker\n(LSTM)', fc=COLORS['pink'], fontsize=22)
    draw_arrow(ax, (5.7, 2.4), (5.7, 3.2), color=COLORS['pink'], lw=3.0, ms=27)
    draw_box(ax, (3.8, 2.8), 3.8, 1.8, 'V_long Critic\nNetwork', fc="#FCBAD3", fontsize=22)

    # Fusion mechanism
    draw_box(ax, (7.4, 2.0), 3.8, 3.0, 'Dynamic Fusion\nβ(q)', fc=COLORS['lavender'], lw=3.4, fontsize=24)
    draw_arrow(ax, (6.9, 4.9), (7.4, 4.1), color=COLORS['teal'], lw=3.0)
    ax.text(7.0, 4.5, 'V_short', fontsize=18, color=COLORS['teal'], fontweight='bold', rotation=-18)
    draw_arrow(ax, (6.9, 3.4), (7.4, 3.4), color=COLORS['pink'], lw=3.0)
    ax.text(7.0, 3.7, 'V_long', fontsize=18, color=COLORS['pink'], fontweight='bold')

    # Output
    draw_arrow(ax, (11.0, 3.5), (12.8, 3.5), color="#333", lw=3.2, ms=31)
    draw_box(ax, (12.8, 2.9), 3.2, 1.8, 'V(s, q)', fc=COLORS['yellow'], fontsize=22)

    # Formula
    ax.text(8.8, 0.55,
        r'$V(s,q) = (1 - \beta(q))\,V_{short}(s,q) + \beta(q)\,V_{long}(q)$',
        fontsize=20, ha='center', fontweight='bold', style='italic', color='#222222')

    # Legend for β(q) behavior (top right, non-overlapping)
    ax.text(8.8, 8.3, 'β(q) behavior', fontsize=18, fontweight='bold', ha='center', color='#222222')
    ax.text(8.8, 7.6, '• Far: β ≈ 0.1 → short-term focus', fontsize=15, ha='center', color='#222222', fontweight='bold')
    ax.text(8.8, 7.0, '• Mid: β ≈ 0.5 → balanced', fontsize=15, ha='center', color='#222222', fontweight='bold')
    ax.text(8.8, 6.4, '• Near: β ≈ 0.9 → long-term focus', fontsize=15, ha='center', color='#222222', fontweight='bold')

    save_figure(fig, 'figure2_fusion')
    plt.close(fig)

# ==================== Figure 3: LDBA Transition Types ====================
def figure_3_ldba_transitions():
    """Four transition types in LDBA with consistent framing and spacing."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12.2), constrained_layout=True)
    
    transition_types = [
        {
            'name': 'ε-Transition\n(Pure Logic)',
            'desc': 'Logical state change\nNo environment action needed',
            'color': '#95E1D3',
            'example': 'q₁ →ε q₂'
        },
        {
            'name': 'Forward Chain\n(Progress)',
            'desc': 'Necessary steps\nSatisfying key propositions',
            'color': '#FFE66D',
            'example': 'q₁ →p q₂'
        },
        {
            'name': 'Skip Edge\n(Shortcut)',
            'desc': 'Bypass intermediates\nEfficient planning',
            'color': '#F38181',
            'example': 'q₁ →skip q₃'
        },
        {
            'name': 'Backward Loop\n(Acceptance)',
            'desc': 'Maintain acceptance\nStable safety',
            'color': '#AA96DA',
            'example': 'q_f ↻'
        }
    ]
    
    for idx, (ax, trans) in enumerate(zip(axes.flat, transition_types)):
        ax.set_xlim(-0.6, 2.2)
        ax.set_ylim(-0.8, 2.3)
        ax.axis('off')
        
        # Draw transition diagram
        if idx < 3:  # Forward transitions
            circle1 = Circle((0.3, 1.1), 0.38, color=COLORS['teal_light'], ec='black', linewidth=2.4)
            circle2 = Circle((1.8, 1.1), 0.38, color=COLORS['pink'], ec='black', linewidth=2.4)
            ax.add_patch(circle1)
            ax.add_patch(circle2)
            ax.text(0.3, 1.1, 'q₁', fontsize=30, ha='center', va='center', fontweight='bold')
            ax.text(1.8, 1.1, 'q₂', fontsize=30, ha='center', va='center', fontweight='bold')
            
            arrow = FancyArrowPatch((0.62, 1.1), (1.48, 1.1),
                                   arrowstyle='->', mutation_scale=32, linewidth=3.0,
                                   color=trans['color'])
            ax.add_patch(arrow)
        else:  # Self-loop
            circle = Circle((1.05, 1.1), 0.38, color=COLORS['accent3'], ec='black', linewidth=2.4)
            ax.add_patch(circle)
            ax.text(1.05, 1.1, 'q_f', fontsize=30, ha='center', va='center', fontweight='bold')
            
            # Self-loop
            from matplotlib.patches import Arc
            arc = Arc((1.05, 1.42), 0.85, 0.85, angle=0, theta1=45, theta2=315,
                     color=trans['color'], linewidth=3.0)
            ax.add_patch(arc)
        
        # Title and description
        ax.text(1.05, 1.95, trans['name'], fontsize=30, ha='center', fontweight='bold', color='#222222')
        ax.text(1.05, 0.15, trans['desc'], fontsize=30, ha='center', color='#222222', fontweight='bold')
        ax.text(1.05, -0.45, trans['example'], fontsize=30, ha='center', style='italic', fontweight='bold', color='#222222')
    
    save_figure(fig, 'figure3_ldba_transitions')
    plt.close(fig)

# ==================== Figure 4: Fusion Weight Dynamics ====================
def figure_4_fusion_weight_dynamics():
    """Fusion weight β(q) dynamics with clean shading and labels."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)
    
    # Left plot: Beta function
    proximity = np.linspace(0, 1, 200)
    gamma = 3.0
    theta = 0.5
    beta = 1 / (1 + np.exp(-gamma * (proximity - theta)))
    
    ax1.plot(proximity, beta, linewidth=3, color=COLORS['lavender'], label=r'$\beta(q)=\sigma(\gamma(\mathrm{prox}-\theta))$')
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax1.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Mark regions
    ax1.fill_between(proximity[proximity < 0.35], 0, 1, alpha=0.18, color=COLORS['teal'], label='Short-term focus')
    ax1.fill_between(proximity[(proximity >= 0.35) & (proximity <= 0.65)], 0, 1, alpha=0.18, color=COLORS['yellow'], label='Balanced')
    ax1.fill_between(proximity[proximity > 0.65], 0, 1, alpha=0.18, color=COLORS['pink'], label='Long-term focus')
    
    # Annotations
    ax1.annotate('Initial State\nβ≈0.1', xy=(0.15, 0.12), xytext=(0.1, 0.35),
                arrowprops=dict(arrowstyle='->', color='#4ECDC4', lw=2),
                fontsize=11, fontweight='bold', color='#4ECDC4')
    
    ax1.annotate('Mid-range\nβ≈0.5', xy=(0.5, 0.5), xytext=(0.5, 0.75),
                arrowprops=dict(arrowstyle='->', color='orange', lw=2),
                fontsize=11, fontweight='bold', color='orange')
    
    ax1.annotate('Near Acceptance\nβ≈0.9', xy=(0.85, 0.88), xytext=(0.75, 0.65),
                arrowprops=dict(arrowstyle='->', color='#F38181', lw=2),
                fontsize=11, fontweight='bold', color='#F38181')
    
    ax1.set_xlabel('Proximity to Accepting States', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Fusion Weight β(q)', fontsize=11, fontweight='bold')
    ax1.set_title('Fusion Weight as Function of Task Progress', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.05, 1.05)
    
    # Right plot: Value decomposition over trajectory
    steps = np.arange(0, 100)
    proximity_trajectory = 1 - np.exp(-steps / 30)  # Monotonically increasing
    beta_trajectory = 1 / (1 + np.exp(-gamma * (proximity_trajectory - theta)))
    
    v_short = 0.6 - 0.3 * (steps / 100)  # Decreasing
    v_long = 0.3 + 0.6 * (steps / 100)   # Increasing
    v_fused = (1 - beta_trajectory) * v_short + beta_trajectory * v_long

    ax2.plot(steps, v_short, linewidth=2.6, label=r'$V_{short}(s,q)$', color=COLORS['teal'], linestyle='--')
    ax2.plot(steps, v_long, linewidth=2.6, label=r'$V_{long}(q)$', color=COLORS['pink'], linestyle='--')
    ax2.plot(steps, v_fused, linewidth=3.2, label=r'$V_{fused}=(1-\beta)V_{short}+\beta V_{long}$',
            color=COLORS['lavender'])
    
    # Fill between
    ax2.fill_between(steps, v_short, v_fused, alpha=0.2, color=COLORS['teal'])
    ax2.fill_between(steps, v_fused, v_long, alpha=0.2, color=COLORS['pink'])
    
    ax2.set_xlabel('Task Progress (Steps)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Value Estimate', fontsize=11, fontweight='bold')
    ax2.set_title('Value Decomposition Over Trajectory', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='center right', fontsize=10)
    ax2.set_ylim(0, 1)
    
    save_figure(fig, 'figure4_fusion_weight')
    plt.close(fig)

# ==================== Figure 5: Performance Improvement ====================
def figure_5_performance_improvement():
    """Performance improvement visuals with refined labels and spacing."""
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 5.2), constrained_layout=True)
    
    # Data
    configs = ['baseline', 'hv_only', 'msgnn_only', 'hv_track', 'msgnn_track', 'full_all']
    improvements = [0, 4.5, 2.3, 6.4, 3.1, 3.2]
    colors_configs = [COLORS['muted'], COLORS['teal'], COLORS['yellow'], COLORS['lavender'], COLORS['pink'], COLORS['teal_light']]
    
    # Left: Configuration comparison
    ax = axes[0]
    bars = ax.bar(configs, improvements, color=colors_configs, edgecolor='black', linewidth=2, alpha=0.8)
    ax.axhline(y=0, color='black', linewidth=1)
    ax.set_ylabel('Success Rate Improvement (%)', fontsize=11, fontweight='bold')
    ax.set_title('Overall Performance by Configuration', fontsize=12, fontweight='bold')
    ax.set_ylim(-1, 8)
    ax.grid(axis='y', alpha=0.25)
    
    # Add value labels
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.2,
               f'+{imp}%' if imp > 0 else f'{imp}%',
               ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax.tick_params(axis='x', rotation=45)
    
    # Right: Task difficulty analysis
    ax = axes[1]
    tasks = ['φ₁\n(Easy)', 'φ₂\n(Medium)', 'φ₃\n(Hard)']
    baseline_sr = [0.988, 0.970, 0.940]
    hv_track_sr = [1.000, 0.998, 0.980]
    
    x = np.arange(len(tasks))
    width = 0.18
    
    bars1 = ax.bar(x - width/2, baseline_sr, width, label='Baseline', 
                   color='#999999', edgecolor='black', linewidth=1.5, alpha=0.8)
    bars2 = ax.bar(x + width/2, hv_track_sr, width, label='HV-Track',
                   color='#AA96DA', edgecolor='black', linewidth=1.5, alpha=0.8)
    
    ax.set_ylabel('Success Rate', fontsize=11, fontweight='bold')
    ax.set_title('HV-Track Performance by Task Difficulty', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.set_ylim(0.9, 1.01)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(axis='y', alpha=0.25)
    
    # Add improvement percentages
    improvements_by_task = [1.2, 2.9, 4.3]
    for i, imp in enumerate(improvements_by_task):
        ax.text(i, 1.005, f'+{imp}%', ha='center', va='bottom', 
               fontweight='bold', fontsize=9, color='#AA96DA')
    
    save_figure(fig, 'figure5_performance')
    plt.close(fig)

# ==================== Figure 6: Architecture Overview ====================
def figure_6_architecture_overview():
    """High-level architecture overview with balanced grid-like layout."""
    fig, ax = plt.subplots(figsize=(14, 7.8), constrained_layout=True)
    
    ax.set_xlim(-0.5, 13.5)
    ax.set_ylim(-0.5, 9)
    ax.axis('off')
    
    # Input layer
    draw_box(ax, (0.2, 7), 2, 1, 'Environment\nState (s)', fc=COLORS['yellow'])
    
    draw_box(ax, (3, 7), 2, 1, 'LDBA\nAutomaton (φ)', fc=COLORS['yellow'])
    
    # SE-GNN
    draw_box(ax, (0.5, 5), 2.5, 1.5, 'SE-GNN\n(3 layers)', fc=COLORS['teal'])
    
    arrow_to_segnn = FancyArrowPatch((1.2, 7), (1.75, 6.5),
                                    arrowstyle='->', mutation_scale=20, linewidth=2)
    ax.add_patch(arrow_to_segnn)
    
    # MSGNN
    draw_box(ax, (3.5, 5), 2.5, 1.5, 'Multi-scale\nExtractor', fc=COLORS['teal_light'])
    
    arrow_to_msgnn = FancyArrowPatch((4, 7), (4.75, 6.5),
                                    arrowstyle='->', mutation_scale=20, linewidth=2)
    ax.add_patch(arrow_to_msgnn)
    
    # Automaton Tracker
    draw_box(ax, (6.5, 5), 2.5, 1.5, 'Automaton\nTracker (LSTM)', fc=COLORS['pink'])
    
    arrow_to_tracker = FancyArrowPatch((4, 7), (7.75, 6.5),
                                      arrowstyle='->', mutation_scale=20, linewidth=2)
    ax.add_patch(arrow_to_tracker)
    
    # Value networks
    draw_box(ax, (1, 2.5), 2.2, 1.5, 'V_short\nCritic Network', fc=COLORS['teal'])
    
    arrow_vshort = FancyArrowPatch((2.5, 5), (2.1, 4),
                                  arrowstyle='->', mutation_scale=20, linewidth=2, color='#4ECDC4')
    ax.add_patch(arrow_vshort)
    
    draw_box(ax, (4, 2.5), 2.2, 1.5, 'V_long\nCritic Network', fc=COLORS['pink'])
    
    arrow_vlong = FancyArrowPatch((7, 5), (5.1, 4),
                                 arrowstyle='->', mutation_scale=20, linewidth=2, color='#F38181')
    ax.add_patch(arrow_vlong)
    
    # Fusion
    draw_box(ax, (7, 2.5), 2.5, 1.5, 'Dynamic\nFusion β(q)', fc=COLORS['lavender'], lw=2.6)
    
    arrow_f1 = FancyArrowPatch((3.2, 3.25), (7, 3.25),
                              arrowstyle='->', mutation_scale=20, linewidth=2)
    ax.add_patch(arrow_f1)
    
    arrow_f2 = FancyArrowPatch((6.2, 3.25), (7, 3.25),
                              arrowstyle='->', mutation_scale=20, linewidth=2)
    ax.add_patch(arrow_f2)
    
    # Policy and Action
    draw_box(ax, (10.5, 2.5), 2.5, 1.5, 'Value + Actor\nPolicy Network', fc=COLORS['yellow'])
    
    arrow_to_output = FancyArrowPatch((9.5, 3.25), (10.5, 3.25),
                                     arrowstyle='->', mutation_scale=25, linewidth=2.5)
    ax.add_patch(arrow_to_output)
    
    # Key differences box - adjusted positioning
    ax.text(6.5, 8.3, 'Key Innovations', fontsize=12, fontweight='bold', ha='left')
    
    innovations = [
        '✓ SE-GNN: Semantic edges & nodes',
        '✓ MSGNN: Multi-scale extraction',
        '✓ Tracking: Automaton state use',
        '✓ Fusion: Dynamic β(q) balance'
    ]
    
    for i, innovation in enumerate(innovations):
        ax.text(6.5, 7.85 - i*0.35, innovation, fontsize=9.5, family='monospace', ha='left')
    
    save_figure(fig, 'figure6_architecture')
    plt.close(fig)


def figure_6_architecture_compact():
    """Compact version: larger text, tighter layout, reduced whitespace."""
    fig, ax = plt.subplots(figsize=(13.2, 6.4), constrained_layout=True)

    ax.set_xlim(-0.2, 12.2)
    ax.set_ylim(0.2, 7.6)
    ax.axis('off')

    box_w, box_h = 2.5, 1.3
    font_main = 13

    # Input layer
    draw_box(ax, (0.2, 6.0), box_w, box_h, 'Environment\nState (s)', fc=COLORS['yellow'], fontsize=font_main)
    draw_box(ax, (3.0, 6.0), box_w, box_h, 'LDBA\nAutomaton (φ)', fc=COLORS['yellow'], fontsize=font_main)

    # SE-GNN and MSGNN
    draw_box(ax, (0.5, 4.35), 2.7, 1.4, 'SE-GNN\n(3 layers)', fc=COLORS['teal'], fontsize=font_main)
    ax.add_patch(FancyArrowPatch((1.5, 6.0), (1.85, 5.7), arrowstyle='->', mutation_scale=18, linewidth=2.2, color="#333"))

    draw_box(ax, (3.5, 4.35), 2.7, 1.4, 'Multi-scale\nExtractor (1/2/3-hop)', fc=COLORS['teal_light'], fontsize=font_main)
    ax.add_patch(FancyArrowPatch((4.1, 6.0), (4.85, 5.7), arrowstyle='->', mutation_scale=18, linewidth=2.2, color="#333"))

    # Automaton Tracker
    draw_box(ax, (6.4, 4.35), 2.7, 1.4, 'Automaton\nTracker (LSTM)\nproximity/dist/history', fc=COLORS['pink'], fontsize=font_main)
    ax.add_patch(FancyArrowPatch((4.1, 6.0), (7.6, 5.7), arrowstyle='->', mutation_scale=18, linewidth=2.2, color="#333"))

    # Value networks
    draw_box(ax, (0.9, 2.15), 2.35, 1.3, 'V_short\nCritic Network\n(dense rewards)', fc=COLORS['teal'], fontsize=font_main)
    ax.add_patch(FancyArrowPatch((2.1, 4.35), (1.95, 3.6), arrowstyle='->', mutation_scale=18, linewidth=2.2, color=COLORS['teal']))

    draw_box(ax, (3.9, 2.15), 2.35, 1.3, 'V_long\nCritic Network\n(sparse accept.)', fc=COLORS['pink'], fontsize=font_main)
    ax.add_patch(FancyArrowPatch((7.0, 4.35), (5.0, 3.6), arrowstyle='->', mutation_scale=18, linewidth=2.2, color=COLORS['pink']))

    # Fusion and policy
    draw_box(ax, (6.8, 2.15), 2.6, 1.3, 'Dynamic\nFusion β(q)\nweights V_short/V_long', fc=COLORS['lavender'], fontsize=font_main, lw=2.4)
    ax.add_patch(FancyArrowPatch((3.2, 2.8), (6.8, 2.8), arrowstyle='->', mutation_scale=20, linewidth=2.2, color="#333"))
    ax.add_patch(FancyArrowPatch((6.2, 2.8), (6.8, 2.8), arrowstyle='->', mutation_scale=20, linewidth=2.2, color="#333"))

    draw_box(ax, (9.5, 2.15), 2.4, 1.3, 'Value + Actor\nPolicy Network', fc=COLORS['yellow'], fontsize=font_main)
    ax.add_patch(FancyArrowPatch((9.3, 2.8), (9.5, 2.8), arrowstyle='->', mutation_scale=22, linewidth=2.4, color="#333"))

    # Key innovations (tight, larger font)
    ax.text(6.8, 6.95, 'Key Innovations', fontsize=13.5, fontweight='bold', ha='left')
    innovations = [
        '✓ SE-GNN: Semantic edges & nodes',
        '✓ MSGNN: Multi-scale extraction',
        '✓ Tracking: Automaton state use',
        '✓ Fusion: Dynamic β(q) balance'
    ]
    for i, innovation in enumerate(innovations):
        ax.text(6.8, 6.55 - i*0.35, innovation, fontsize=10.5, family='monospace', ha='left')

    save_figure(fig, 'figure6_architecture_compact')
    plt.close(fig)


def figure_6_architecture_flow():
    """Expressive flow version matching paper narrative with four stages."""
    fig, ax = plt.subplots(figsize=(13.5, 6.6), constrained_layout=True)

    ax.set_xlim(-0.2, 13.5)
    ax.set_ylim(-0.5, 7.8)
    ax.axis('off')

    box_w = 2.7
    box_h = 1.3
    font_main = 12

    # Stage 1: Environment & Automaton Interaction
    draw_box(ax, (0.2, 5.8), box_w, box_h, 'Env & Automaton\nInteraction\n(Spot/Rabinizer)', fc=COLORS['yellow'], fontsize=font_main)
    ax.text(1.55, 5.05, 'φ → A_φ, q_t, o_t', ha='center', fontsize=12, color="#222222")

    # Stage 2: SE-GNN Encoding
    draw_box(ax, (3.3, 5.8), box_w, box_h, 'SE-GNN Encoding\n(semantic edges/nodes)', fc=COLORS['teal'], fontsize=font_main)
    ax.text(4.65, 5.05, 'Nodes: accept/SCC/init/dist\nEdges: ε/forward/skip/back', ha='center', fontsize=12, color="#222222")

    # Stage 3: Dual-Pathway Value Decomposition
    draw_box(ax, (6.4, 6.2), box_w, box_h, 'Multi-scale\nEmbeddings (1/2/3-hop)', fc=COLORS['teal_light'], fontsize=font_main)
    draw_box(ax, (6.4, 4.6), box_w, box_h, 'Automaton\nTracking (LSTM)\nprox/dist/history', fc=COLORS['pink'], fontsize=font_main)
    draw_box(ax, (6.4, 3.0), box_w, box_h, 'V_short Critic\n(dense)', fc=COLORS['teal'], fontsize=font_main)
    draw_box(ax, (6.4, 1.4), box_w, box_h, 'V_long Critic\n(sparse)', fc='#FCBAD3', fontsize=font_main)

    # Stage 4: Dynamic Fusion & Policy
    draw_box(ax, (9.6, 4.4), box_w, 1.4, 'Dynamic Fusion β(q)\nweights V_short/V_long', fc=COLORS['lavender'], fontsize=font_main, lw=2.6)
    draw_box(ax, (9.6, 2.4), box_w, 1.4, 'Value + Actor\nPolicy Network', fc=COLORS['yellow'], fontsize=font_main)

    # Arrows for flow
    ax.add_patch(FancyArrowPatch((2.9, 6.45), (3.3, 6.45), arrowstyle='->', mutation_scale=22, linewidth=2.4, color="#333"))
    ax.add_patch(FancyArrowPatch((5.95, 6.45), (6.4, 6.45), arrowstyle='->', mutation_scale=22, linewidth=2.4, color="#333"))
    ax.add_patch(FancyArrowPatch((5.95, 6.45), (6.4, 5.3), arrowstyle='->', mutation_scale=22, linewidth=2.0, linestyle='--', color='#d62728'))

    # From embeddings/tracker to critics
    ax.add_patch(FancyArrowPatch((7.75, 6.2), (7.75, 4.6), arrowstyle='->', mutation_scale=18, linewidth=2.2, color=COLORS['teal']))
    ax.add_patch(FancyArrowPatch((7.75, 4.6), (7.75, 3.0), arrowstyle='->', mutation_scale=18, linewidth=2.2, color=COLORS['teal']))
    ax.add_patch(FancyArrowPatch((7.75, 4.6), (7.75, 1.4), arrowstyle='->', mutation_scale=18, linewidth=2.2, color=COLORS['pink']))

    # Fusion arrows
    ax.add_patch(FancyArrowPatch((8.1, 3.65), (9.6, 5.1), arrowstyle='->', mutation_scale=22, linewidth=2.4, color=COLORS['teal']))
    ax.text(8.8, 4.7, 'V_short', fontsize=12, color=COLORS['teal'], fontweight='bold')
    ax.add_patch(FancyArrowPatch((8.1, 2.05), (9.6, 5.1), arrowstyle='->', mutation_scale=22, linewidth=2.4, color=COLORS['pink']))
    ax.text(8.8, 3.4, 'V_long', fontsize=12, color=COLORS['pink'], fontweight='bold')

    # To policy
    ax.add_patch(FancyArrowPatch((11.25, 4.4), (11.25, 3.8), arrowstyle='->', mutation_scale=22, linewidth=2.4, color="#333"))
    ax.add_patch(FancyArrowPatch((11.25, 2.4), (11.25, 1.7), arrowstyle='->', mutation_scale=22, linewidth=2.4, linestyle='--', color='#7f7f7f'))

    # Legends / annotations
    ax.text(2.0, 4.0, 'Gradient flow (dashed)', fontsize=11.5, color='#b2182b', fontweight='bold')
    ax.text(2.0, 3.5, 'Forward inference (solid)', fontsize=11.5, color='#222222', fontweight='bold')
    ax.text(9.6, 6.8, 'Dynamic fusion:\nβ(q) balances\nshort vs long\nbased on proximity', fontsize=11.5, ha='left', color='#222222', fontweight='bold')
    ax.text(2.0, 2.5, 'Dual-pathway value decomposition:\n- Short-term: 1/2/3-hop embeddings\n- Long-term: proximity / distance / history', fontsize=11.5, ha='left', color='#222222', fontweight='bold')

    save_figure(fig, 'figure6_architecture_flow')
    plt.close(fig)

# ==================== Main ====================
if __name__ == '__main__':
    print("\n" + "="*60)
    print("Generating Publication-Quality Figures for HiLTL Paper")
    print("="*60 + "\n")
    
    figure_1_se_gnn_architecture()
    figure_2_long_short_fusion()
    figure_3_ldba_transitions()
    figure_4_fusion_weight_dynamics()
    figure_5_performance_improvement()
    figure_6_architecture_overview()
    figure_6_architecture_compact()
    figure_6_architecture_flow()
    
    print("\n" + "="*60)
    print("✅ All figures generated successfully (PNG + PDF)!")
    print("="*60)
    print("\nGenerated figures:")
    print("  1. figure1_se_gnn.(png|pdf) - SE-GNN Architecture")
    print("  2. figure2_fusion.(png|pdf) - Long-Short Term Fusion")
    print("  3. figure3_ldba_transitions.(png|pdf) - LDBA Transition Types")
    print("  4. figure4_fusion_weight.(png|pdf) - Fusion Weight Dynamics")
    print("  5. figure5_performance.(png|pdf) - Performance Improvements")
    print("  6. figure6_architecture.(png|pdf) - Original System Architecture")
    print("  7. figure6_architecture_compact.(png|pdf) - Compact tightened layout")
    print("  8. figure6_architecture_flow.(png|pdf) - Expressive narrative layout")
    print(f"\nLocations: {OUTPUT_DIR}/")
    print("="*60 + "\n")
