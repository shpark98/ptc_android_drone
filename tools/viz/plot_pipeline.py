#!/usr/bin/env python3
"""
Pipeline visualization utility.
Usage: python tools/viz/plot_pipeline.py [--output results/pipeline.png]
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import argparse

COLORS = {
    'flow': '#4CAF50',
    'motion': '#2196F3',
    'tri': '#FF9800',
    'seg': '#9C27B0',
    'fusion': '#F44336',
    'depth': '#607D8B',
    'scale': '#795548',
    'warp': '#00BCD4'
}

def plot_python_pipeline(ax, timing=None):
    """Draw Python sequential pipeline."""
    timing = timing or {
        'depth': 80, 'flow': 35, 'motion': 36,
        'tri': 45, 'seg': 280, 'fusion': 10, 'scale': 60
    }
    total = sum(timing.values())

    ax.set_xlim(0, 16)
    ax.set_ylim(0, 5)
    ax.axis('off')
    ax.set_title(f'Python Pipeline (Sequential) - Total: ~{total}ms', fontsize=20, fontweight='bold')

    boxes = [
        (0.2, 2.0, 1.9, 1.6, 'Depth', f"{timing['depth']}ms", COLORS['depth']),
        (2.3, 2.0, 1.9, 1.6, 'Flow', f"{timing['flow']}ms", COLORS['flow']),
        (4.4, 2.0, 1.9, 1.6, 'Motion', f"{timing['motion']}ms", COLORS['motion']),
        (6.5, 2.0, 1.9, 1.6, 'Tri', f"{timing['tri']}ms", COLORS['tri']),
        (8.6, 2.0, 2.2, 1.6, 'Segment', f"{timing['seg']}ms", COLORS['seg']),
        (11.0, 2.0, 1.9, 1.6, 'Fusion', f"{timing['fusion']}ms", COLORS['fusion']),
        (13.1, 2.0, 2.0, 1.6, 'Scale', f"{timing['scale']}ms", COLORS['scale']),
    ]

    for x, y, w, h, label, time, color in boxes:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                             facecolor=color, edgecolor='black', linewidth=3, alpha=0.9)
        ax.add_patch(box)
        ax.text(x + w/2, y + h*0.65, label, ha='center', va='center',
                fontsize=16, fontweight='bold', color='white')
        ax.text(x + w/2, y + h*0.25, time, ha='center', va='center',
                fontsize=18, fontweight='bold', color='white')

    arrow_style = dict(arrowstyle='->', color='black', lw=3)
    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + boxes[i][2]
        x2 = boxes[i+1][0]
        y = boxes[i][1] + boxes[i][3]/2
        ax.annotate('', xy=(x2, y), xytext=(x1, y), arrowprops=arrow_style)

    fps = 1000 / total
    ax.text(8, 0.5, f'FPS: ~{fps:.1f}', fontsize=18, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='black', lw=2))


def plot_cpp_pipeline(ax, timing=None):
    """Draw C++ parallel pipeline."""
    timing = timing or {
        'depth': 80, 'seg': 50, 'flow': 38, 'motion': 28,
        'tri': 4, 'warp': 2, 'fusion': 5, 'scale': 16
    }
    # Parallel: max(seg, flow) + motion + max(tri, warp) + fusion + scale
    total = timing['depth'] + max(timing['seg'], timing['flow']) + timing['motion'] + \
            max(timing['tri'], timing['warp']) + timing['fusion'] + timing['scale']

    ax.set_xlim(0, 16)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_title(f'C++ Pipeline (Parallel) - Total: ~{total}ms', fontsize=20, fontweight='bold')

    arrow_style = dict(arrowstyle='->', color='black', lw=3)

    # Phase 1: Parallel Seg + Flow
    ax.add_patch(mpatches.Rectangle((0.1, 2.3), 3.3, 2.8, fill=False, edgecolor='#666', linestyle='--', lw=2))
    ax.text(1.75, 5.0, 'Parallel', fontsize=14, fontweight='bold', ha='center', color='#666')

    box1 = FancyBboxPatch((0.3, 3.8), 2.9, 1.1, boxstyle="round,pad=0.05",
                          facecolor=COLORS['seg'], edgecolor='black', linewidth=3, alpha=0.9)
    ax.add_patch(box1)
    ax.text(1.75, 4.55, 'Segment', ha='center', fontsize=15, fontweight='bold', color='white')
    ax.text(1.75, 4.1, f"{timing['seg']}ms", ha='center', fontsize=16, fontweight='bold', color='white')

    box2 = FancyBboxPatch((0.3, 2.5), 2.9, 1.1, boxstyle="round,pad=0.05",
                          facecolor=COLORS['flow'], edgecolor='black', linewidth=3, alpha=0.9)
    ax.add_patch(box2)
    ax.text(1.75, 3.25, 'Flow', ha='center', fontsize=15, fontweight='bold', color='white')
    ax.text(1.75, 2.8, f"{timing['flow']}ms", ha='center', fontsize=16, fontweight='bold', color='white')

    # Phase 2: Motion
    box3 = FancyBboxPatch((3.7, 3.0), 2.4, 1.5, boxstyle="round,pad=0.05",
                          facecolor=COLORS['motion'], edgecolor='black', linewidth=3, alpha=0.9)
    ax.add_patch(box3)
    ax.text(4.9, 4.0, 'Motion', ha='center', fontsize=15, fontweight='bold', color='white')
    ax.text(4.9, 3.4, f"{timing['motion']}ms", ha='center', fontsize=16, fontweight='bold', color='white')
    ax.annotate('', xy=(3.7, 3.75), xytext=(3.4, 3.75), arrowprops=arrow_style)

    # Phase 3: Parallel Tri + Warp
    ax.add_patch(mpatches.Rectangle((6.3, 2.3), 3.3, 2.8, fill=False, edgecolor='#666', linestyle='--', lw=2))
    ax.text(7.95, 5.0, 'Parallel', fontsize=14, fontweight='bold', ha='center', color='#666')

    box4 = FancyBboxPatch((6.5, 3.8), 2.9, 1.1, boxstyle="round,pad=0.05",
                          facecolor=COLORS['tri'], edgecolor='black', linewidth=3, alpha=0.9)
    ax.add_patch(box4)
    ax.text(7.95, 4.55, 'Tri', ha='center', fontsize=15, fontweight='bold', color='white')
    ax.text(7.95, 4.1, f"{timing['tri']}ms", ha='center', fontsize=16, fontweight='bold', color='white')

    box5 = FancyBboxPatch((6.5, 2.5), 2.9, 1.1, boxstyle="round,pad=0.05",
                          facecolor=COLORS['warp'], edgecolor='black', linewidth=3, alpha=0.9)
    ax.add_patch(box5)
    ax.text(7.95, 3.25, 'Warp', ha='center', fontsize=15, fontweight='bold', color='white')
    ax.text(7.95, 2.8, f"{timing['warp']}ms", ha='center', fontsize=16, fontweight='bold', color='white')
    ax.annotate('', xy=(6.5, 3.75), xytext=(6.1, 3.75), arrowprops=arrow_style)

    # Phase 4: Fusion + Scale
    box6 = FancyBboxPatch((9.9, 3.5), 2.4, 1.3, boxstyle="round,pad=0.05",
                          facecolor=COLORS['fusion'], edgecolor='black', linewidth=3, alpha=0.9)
    ax.add_patch(box6)
    ax.text(11.1, 4.4, 'Fusion', ha='center', fontsize=15, fontweight='bold', color='white')
    ax.text(11.1, 3.85, f"{timing['fusion']}ms", ha='center', fontsize=16, fontweight='bold', color='white')

    box7 = FancyBboxPatch((12.6, 3.5), 2.4, 1.3, boxstyle="round,pad=0.05",
                          facecolor=COLORS['scale'], edgecolor='black', linewidth=3, alpha=0.9)
    ax.add_patch(box7)
    ax.text(13.8, 4.4, 'Scale', ha='center', fontsize=15, fontweight='bold', color='white')
    ax.text(13.8, 3.85, f"{timing['scale']}ms", ha='center', fontsize=16, fontweight='bold', color='white')

    ax.annotate('', xy=(9.9, 4.15), xytext=(9.6, 4.15), arrowprops=arrow_style)
    ax.annotate('', xy=(12.6, 4.15), xytext=(12.3, 4.15), arrowprops=arrow_style)

    # Depth (separate)
    box0 = FancyBboxPatch((0.3, 0.5), 2.9, 1.3, boxstyle="round,pad=0.05",
                          facecolor=COLORS['depth'], edgecolor='black', linewidth=3, alpha=0.9)
    ax.add_patch(box0)
    ax.text(1.75, 1.35, 'Depth', ha='center', fontsize=15, fontweight='bold', color='white')
    ax.text(1.75, 0.85, f"{timing['depth']}ms", ha='center', fontsize=16, fontweight='bold', color='white')

    fps = 1000 / total
    ax.text(11, 0.8, f'FPS: ~{fps:.1f}', fontsize=18, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen', edgecolor='black', lw=2))


def main():
    parser = argparse.ArgumentParser(description='Generate pipeline diagrams')
    parser.add_argument('--type', choices=['python', 'cpp', 'both'], default='both')
    parser.add_argument('--output', default='results/pipeline.png')
    args = parser.parse_args()

    import matplotlib
    matplotlib.use('Agg')

    if args.type == 'both':
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 11))
        plot_python_pipeline(ax1)
        plot_cpp_pipeline(ax2)
        plt.tight_layout()
        plt.savefig(args.output, dpi=150, bbox_inches='tight')
    elif args.type == 'python':
        fig, ax = plt.subplots(figsize=(16, 5))
        plot_python_pipeline(ax)
        plt.tight_layout()
        plt.savefig(args.output, dpi=150, bbox_inches='tight')
    else:
        fig, ax = plt.subplots(figsize=(16, 6))
        plot_cpp_pipeline(ax)
        plt.tight_layout()
        plt.savefig(args.output, dpi=150, bbox_inches='tight')

    print(f"Saved to {args.output}")


if __name__ == '__main__':
    main()
