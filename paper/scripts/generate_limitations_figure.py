#!/usr/bin/env python3
"""
Generate the 'Two Core Limitations' figure for the paper.
This script creates a three-panel visualization showing:
(a) Standard LDBA with semantic information
(b) Semantic Flattening problem
(c) Temporal Myopia problem
"""

from pathlib import Path
import os
os.chdir(Path(__file__).resolve().parents[1] / 'figures')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D
import matplotlib as mpl
import numpy as np

# Set up the figure
mpl.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 9
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['savefig.pad_inches'] = 0.05

fig = plt.figure(figsize=(10.5, 3.4))
fig.patch.set_facecolor('white')

# Define colors
node_color = '#AFE9F0'     # Light blue
gray_node_color = '#D9D9D9' # Light gray
fwd_color = '#228B22'       # Forest green
skip_color = '#FF8C00'      # Dark orange
epsilon_color = '#C51F5D'   # Magenta-red (better contrast)
bwd_color = '#6F5AA5'       # Muted purple
gray_color = '#808080'      # Gray

# ===== PANEL (a): Standard LDBA with Semantic Information =====
ax1 = plt.subplot(1, 3, 1)
ax1.set_xlim(-0.5, 3.5)
ax1.set_ylim(-2.5, 1.5)
ax1.axis('off')

# Draw nodes
nodes_a = {
    'q0': (0.5, 0),
    'q1': (1.7, 0.8),
    'q2': (1.7, -0.8),
    'q3': (2.9, 0)
}

for name, (x, y) in nodes_a.items():
    circle = Circle((x, y), 0.25, facecolor=node_color, edgecolor='black', linewidth=1.5, zorder=3)
    ax1.add_patch(circle)
    ax1.text(x, y, f'${name}$', ha='center', va='center', fontsize=10, fontweight='bold', zorder=4)

# Draw arrows with semantic colors
# Forward: q0 -> q1
arrow1 = FancyArrowPatch(nodes_a['q0'], nodes_a['q1'], 
                         connectionstyle="arc3,rad=0.2", 
                         arrowstyle='->', mutation_scale=15, linewidth=2.5,
                         color=fwd_color, zorder=2)
ax1.add_patch(arrow1)
ax1.text(0.8, 0.6, 'fwd', fontsize=7, fontweight='bold', color=fwd_color)

# Skip: q0 -> q2
arrow2 = FancyArrowPatch(nodes_a['q0'], nodes_a['q2'],
                         connectionstyle="arc3,rad=-0.2",
                         arrowstyle='->', mutation_scale=15, linewidth=2.5,
                         color=skip_color, zorder=2)
ax1.add_patch(arrow2)
ax1.text(0.8, -0.6, 'skip', fontsize=7, fontweight='bold', color=skip_color)

# Epsilon: q2 -> q1
arrow3 = FancyArrowPatch(nodes_a['q2'], nodes_a['q1'],
                         connectionstyle="arc3,rad=0.3",
                         arrowstyle='->', mutation_scale=15, linewidth=2.5,
                         color=epsilon_color, zorder=2)
ax1.add_patch(arrow3)
ax1.text(2.1, 0, r'$\epsilon$', fontsize=8, fontweight='bold', color=epsilon_color)

# Backward: q1 -> q3
arrow4 = FancyArrowPatch(nodes_a['q1'], nodes_a['q3'],
                         arrowstyle='->', mutation_scale=15, linewidth=2.5,
                         color=bwd_color, zorder=2)
ax1.add_patch(arrow4)
ax1.text(2.3, 0.5, 'bwd', fontsize=7, fontweight='bold', color=bwd_color)

# Self-loop on q3
theta = np.linspace(0.5, 2*np.pi - 0.5, 100)
loop_x = nodes_a['q3'][0] + 0.35 * np.cos(theta)
loop_y = nodes_a['q3'][1] + 0.5 + 0.35 * np.sin(theta)
ax1.plot(loop_x, loop_y, color=bwd_color, linewidth=2.5, zorder=2)
ax1.arrow(loop_x[-5], loop_y[-5], loop_x[-1]-loop_x[-5], loop_y[-1]-loop_y[-5],
          head_width=0.15, head_length=0.1, fc=bwd_color, ec=bwd_color, zorder=2)

# Title
ax1.text(0.02, 0.96, '(a) Standard LDBA', transform=ax1.transAxes, ha='left', fontsize=10, fontweight='bold')
ax1.text(0.02, 0.88, 'with Transition Semantics', transform=ax1.transAxes, ha='left', fontsize=8)

# ===== PANEL (b): Semantic Flattening =====
ax2 = plt.subplot(1, 3, 2)
ax2.set_xlim(-0.5, 3.5)
ax2.set_ylim(-2.5, 1.8)
ax2.axis('off')

# Draw nodes (grayed out)
nodes_b = {
    'q0': (0.5, 0.3),
    'q1': (1.7, 1.2),
    'q2': (1.7, -0.6),
    'q3': (2.9, 0.3)
}

for name, (x, y) in nodes_b.items():
    circle = Circle((x, y), 0.25, facecolor=gray_node_color, edgecolor='gray', linewidth=1.5, zorder=3)
    ax2.add_patch(circle)
    ax2.text(x, y, f'${name}$', ha='center', va='center', fontsize=10, fontweight='bold', color='gray', zorder=4)

# Draw arrows (all gray)
arrow1 = FancyArrowPatch(nodes_b['q0'], nodes_b['q1'],
                         connectionstyle="arc3,rad=0.2",
                         arrowstyle='->', mutation_scale=15, linewidth=2.5,
                         color=gray_color, zorder=2)
ax2.add_patch(arrow1)

arrow2 = FancyArrowPatch(nodes_b['q0'], nodes_b['q2'],
                         connectionstyle="arc3,rad=-0.2",
                         arrowstyle='->', mutation_scale=15, linewidth=2.5,
                         color=gray_color, zorder=2)
ax2.add_patch(arrow2)

arrow3 = FancyArrowPatch(nodes_b['q2'], nodes_b['q1'],
                         connectionstyle="arc3,rad=0.3",
                         arrowstyle='->', mutation_scale=15, linewidth=2.5,
                         color=gray_color, zorder=2)
ax2.add_patch(arrow3)

arrow4 = FancyArrowPatch(nodes_b['q1'], nodes_b['q3'],
                         arrowstyle='->', mutation_scale=15, linewidth=2.5,
                         color=gray_color, zorder=2)
ax2.add_patch(arrow4)

# Self-loop
theta = np.linspace(0.5, 2*np.pi - 0.5, 100)
loop_x = nodes_b['q3'][0] + 0.35 * np.cos(theta)
loop_y = nodes_b['q3'][1] + 0.5 + 0.35 * np.sin(theta)
ax2.plot(loop_x, loop_y, color=gray_color, linewidth=2.5, zorder=2)
ax2.arrow(loop_x[-5], loop_y[-5], loop_x[-1]-loop_x[-5], loop_y[-1]-loop_y[-5],
          head_width=0.15, head_length=0.1, fc=gray_color, ec=gray_color, zorder=2)

# Big red X
ax2.text(1.7, 0.3, '✗', ha='center', va='center', fontsize=40, color='red', fontweight='bold', zorder=5)

# Title
ax2.text(0.02, 0.96, '(b) Semantic Flattening', transform=ax2.transAxes, ha='left', fontsize=10, fontweight='bold')
ax2.text(0.02, 0.88, 'All transitions treated uniformly', transform=ax2.transAxes, ha='left', fontsize=8)

# ===== PANEL (c): Temporal Myopia =====
ax3 = plt.subplot(1, 3, 3)
ax3.set_xlim(-0.3, 4.2)
ax3.set_ylim(-2.2, 1.3)
ax3.axis('off')

# Title
ax3.text(0.02, 0.96, '(c) Temporal Myopia', transform=ax3.transAxes, ha='left', fontsize=10, fontweight='bold')

# Timeline
ax3.plot([0.5, 4], [0.3, 0.3], 'k-', linewidth=2, zorder=1)
ax3.text(0.15, 0.3, r'$t$', fontsize=9, va='center')

# Time windows with labels showing temporal scale
# Short-term (yellow) - 0-5 steps
rect1 = Rectangle((0.8, -0.05), 0.8, 0.7, facecolor='yellow', alpha=0.4, edgecolor='orange', linewidth=1.5)
ax3.add_patch(rect1)
ax3.text(1.2, 0.5, 'Short', ha='center', fontsize=8, fontweight='bold')
ax3.text(1.2, -0.35, '(0-5 steps)', ha='center', fontsize=7, style='italic')

# Mid-term (orange) - 5-20 steps
rect2 = Rectangle((1.8, -0.05), 1.2, 0.7, facecolor='orange', alpha=0.3, edgecolor='darkorange', linewidth=1.5)
ax3.add_patch(rect2)
ax3.text(2.4, 0.5, 'Mid-term', ha='center', fontsize=8, fontweight='bold')
ax3.text(2.4, -0.35, '(5-20 steps)', ha='center', fontsize=7, style='italic')

# Long-term (gray) - 20+ steps
rect3 = Rectangle((3.2, -0.05), 0.6, 0.7, facecolor='gray', alpha=0.4, edgecolor='darkgray', linewidth=1.5)
ax3.add_patch(rect3)
ax3.text(3.5, 0.5, 'Long', ha='center', fontsize=8, fontweight='bold', color='#333333')
ax3.text(3.5, -0.35, '(20+ steps)', ha='center', fontsize=7, style='italic', color='#333333')

# === Weight function curves ===
# Static embedding: flat red dashed line
ax3.plot([0.5, 4], [-1.0, -1.0], 'r--', linewidth=2.8, label='Static Embedding', zorder=2)

# Ideal adaptive weight: green curve that peaks at different times for different scales
t_vals = np.linspace(0.5, 4, 200)
# Create a smooth curve that emphasizes the adaptive nature
# It should dip and rise to show multi-scale optimization
w_adaptive = 0.5 - 0.35*np.cos(3*np.pi*(t_vals - 0.5)/(4-0.5)) + 0.05*np.sin(8*np.pi*(t_vals - 0.5)/(4-0.5))
ax3.plot(t_vals, -0.5 - 0.6*w_adaptive, color='#2ca02c', linewidth=2.8, label='Adaptive Weight (ideal)', zorder=2)

# Annotate the curves
ax3.text(0.3, -1.0, 'Static', ha='right', fontsize=7, color='red', fontweight='bold')
ax3.text(0.3, -0.75, 'Ideal', ha='right', fontsize=7, color='#2ca02c', fontweight='bold')

# Add annotation explaining the problem
ax3.annotate('Cannot optimize\nmultiple scales', xy=(2, -0.95), xytext=(1.2, -1.8),
             arrowprops=dict(arrowstyle='->', color='red', lw=1.2, linestyle='--'),
             ha='left', fontsize=7.5, color='red', bbox=dict(boxstyle='round,pad=0.3', 
             facecolor='white', edgecolor='red', alpha=0.8))

# Adjust layout
plt.tight_layout()

# Shared legend
legend_handles = [
    Line2D([0], [0], color=fwd_color, lw=3, label='Forward'),
    Line2D([0], [0], color=skip_color, lw=3, label='Skip'),
    Line2D([0], [0], color=epsilon_color, lw=3, label='Epsilon'),
    Line2D([0], [0], color=bwd_color, lw=3, label='Backward'),
]
fig.legend(handles=legend_handles, loc='lower center', ncol=4, frameon=False, fontsize=8)

# Save figure
output_file = 'figure_limitations.pdf'
plt.savefig(output_file, format='pdf', bbox_inches='tight', dpi=300)
print(f"Figure saved as: {output_file}")

# Also save as PNG for preview
output_png = 'figure_limitations.png'
plt.savefig(output_png, format='png', bbox_inches='tight', dpi=300)
print(f"PNG preview saved as: {output_png}")

plt.close()
