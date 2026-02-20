"""Generate architecture and result figures for the RW-ROP report."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

# Japanese font setup
plt.rcParams['font.family'] = 'Yu Gothic'
plt.rcParams['axes.unicode_minus'] = False

FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def draw_box(ax, xy, w, h, text, color='#E3F2FD', edgecolor='#1565C0',
             fontsize=9, fontweight='normal', textcolor='black', alpha=1.0):
    """Draw a rounded box with centered text."""
    box = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.15",
                         facecolor=color, edgecolor=edgecolor, linewidth=1.5, alpha=alpha)
    ax.add_patch(box)
    cx, cy = xy[0] + w / 2, xy[1] + h / 2
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fontsize,
            fontweight=fontweight, color=textcolor, wrap=True)
    return (cx, cy)


def draw_arrow(ax, start, end, color='#333333', style='->', lw=1.5):
    """Draw an arrow between two points."""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw))


def fig1_architecture():
    """Figure 1: Full model architecture diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(7, 9.6, 'Figure 1: Model Architecture — clinical_v3',
            ha='center', va='center', fontsize=14, fontweight='bold')

    # ---- Input layer ----
    img_c = draw_box(ax, (1.0, 8.2), 3.5, 0.9,
                     'Input Image\n(512 x 512 RGB)',
                     color='#FFF3E0', edgecolor='#E65100', fontsize=10, fontweight='bold')
    clin_c = draw_box(ax, (9.5, 8.2), 3.5, 0.9,
                      'Clinical Features\n(Sex, GA, BW, PMA)',
                      color='#E8F5E9', edgecolor='#2E7D32', fontsize=10, fontweight='bold')

    # ---- Backbone / Encoder ----
    bb_c = draw_box(ax, (1.0, 6.5), 3.5, 1.0,
                    'EfficientNet-B0\n(ImageNet pretrained)\nGlobal Avg Pool',
                    color='#E3F2FD', edgecolor='#1565C0', fontsize=9, fontweight='bold')
    enc_c = draw_box(ax, (9.5, 6.5), 3.5, 1.0,
                     'Clinical Encoder\nLinear(4→64) → ReLU\n→ BN → Linear(64→32)',
                     color='#E8F5E9', edgecolor='#2E7D32', fontsize=9, fontweight='bold')

    draw_arrow(ax, (img_c[0], 8.2), (bb_c[0], 7.5))
    draw_arrow(ax, (clin_c[0], 8.2), (enc_c[0], 7.5))

    # ---- Feature dimensions ----
    ax.text(2.75, 6.15, '1,280-dim', ha='center', va='center',
            fontsize=9, color='#1565C0', fontstyle='italic')
    ax.text(11.25, 6.15, '32-dim', ha='center', va='center',
            fontsize=9, color='#2E7D32', fontstyle='italic')

    # ---- Concatenation ----
    concat_c = draw_box(ax, (4.5, 5.0), 5.0, 0.8,
                        'Concatenate → Fused Features (1,312-dim)',
                        color='#F3E5F5', edgecolor='#6A1B9A', fontsize=10, fontweight='bold')

    draw_arrow(ax, (2.75, 6.5), (concat_c[0] - 1.5, 5.8))
    draw_arrow(ax, (11.25, 6.5), (concat_c[0] + 1.5, 5.8))

    # ---- Task heads ----
    tasks = [
        ('Zone\n(I / II / III)', '#BBDEFB', '#1565C0'),
        ('Stage\n(0 / 1 / 2 / 3)', '#C8E6C9', '#2E7D32'),
        ('Plus\n(Norm / Pre / Plus)', '#FFE0B2', '#E65100'),
        ('AROP\n(No / Yes)', '#FFCDD2', '#C62828'),
        ('Treatment\n(No / Yes)', '#D1C4E9', '#4527A0'),
    ]

    head_y = 3.0
    head_w = 2.2
    head_h = 1.0
    start_x = 0.8
    gap = (14 - 2 * start_x - 5 * head_w) / 4

    for i, (label, bg, edge) in enumerate(tasks):
        x = start_x + i * (head_w + gap)
        hc = draw_box(ax, (x, head_y), head_w, head_h,
                      label, color=bg, edgecolor=edge, fontsize=9, fontweight='bold')
        draw_arrow(ax, (concat_c[0], 5.0), (hc[0], head_y + head_h))

    # ---- Annotations ----
    # Task weight annotations
    weights = ['w=1.0', 'w=1.0', 'w=1.0', 'w=1.5', 'w=1.5']
    for i, w in enumerate(weights):
        x = start_x + i * (head_w + gap) + head_w / 2
        ax.text(x, 2.7, w, ha='center', va='center', fontsize=8, color='gray')

    # Loss function box
    loss_c = draw_box(ax, (2.0, 1.2), 10.0, 0.9,
                      'Loss = (Weighted Cross-Entropy + Weighted Focal Loss) / 2\n'
                      '+ Label Smoothing (0.1) + MixUp (alpha=0.2)',
                      color='#FFF9C4', edgecolor='#F9A825', fontsize=9)

    # RW-ROP derivation annotation
    ax.annotate('RW-ROP = Plus OR Stage3 OR ZoneI',
                xy=(start_x + 2 * (head_w + gap) + head_w, head_y + 0.1),
                xytext=(10.5, 1.8),
                fontsize=8, color='#C62828', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#C62828', lw=1,
                                connectionstyle='arc3,rad=0.3'))

    # Regularization note for AROP
    ax.text(start_x + 3 * (head_w + gap) + head_w / 2, 2.3,
            '(implicit regularizer)', ha='center', va='center',
            fontsize=7, color='#C62828', fontstyle='italic')

    plt.tight_layout()
    out = FIGURES_DIR / 'fig1_architecture.png'
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {out}")


def fig2_arop_regularization():
    """Figure 2: AROP regularization effect — SD comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Left: Mean performance ---
    ax = axes[0]
    metrics = ['Zone\nKappa', 'Stage\nKappa', 'Plus\nKappa', 'Treatment\nAUC', 'Treatment\nSens']
    v3_means = [0.669, 0.776, 0.755, 0.975, 0.890]
    icrop_means = [0.593, 0.680, 0.661, 0.929, 0.841]
    v3_sds = [0.024, 0.023, 0.062, 0.025, 0.109]
    icrop_sds = [0.183, 0.202, 0.220, 0.110, 0.146]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(x - width/2, v3_means, width, yerr=v3_sds, capsize=4,
                   label='clinical_v3 (AROP included)', color='#1976D2', alpha=0.85)
    bars2 = ax.bar(x + width/2, icrop_means, width, yerr=icrop_sds, capsize=4,
                   label='icrop_treatment (AROP excluded)', color='#E53935', alpha=0.85)

    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('(a) Performance Comparison', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=9)
    ax.legend(fontsize=8, loc='lower left')
    ax.set_ylim(0.3, 1.1)
    ax.grid(axis='y', alpha=0.3)

    # --- Right: SD multiplier ---
    ax = axes[1]
    sd_ratios = [icrop_sds[i] / v3_sds[i] for i in range(len(metrics))]
    colors = ['#FF7043' if r > 5 else '#FFA726' if r > 3 else '#66BB6A' for r in sd_ratios]

    bars = ax.bar(x, sd_ratios, 0.6, color=colors, edgecolor='gray', linewidth=0.8)
    ax.axhline(y=1, color='black', linestyle='--', lw=1, alpha=0.5)

    for bar, ratio in zip(bars, sd_ratios):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                f'{ratio:.1f}x', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('SD Increase Ratio', fontsize=11)
    ax.set_title('(b) SD Increase by AROP Removal', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=9)
    ax.set_ylim(0, 11)
    ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Figure 2: Effect of AROP Task Removal on Model Stability',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    out = FIGURES_DIR / 'fig2_arop_regularization.png'
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {out}")


def fig3_threshold_summary():
    """Figure 3: Threshold optimization operating points."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # --- Treatment ---
    ax = axes[0]
    ops = ['Default\n(t=0.50)', 'Sens>=95%\n(t=0.50)', "Youden's J\n(t=0.47)"]
    sens = [0.890, 0.960, 0.963]
    spec = [0.955, 0.922, 0.933]
    npv = [0.991, 0.996, 0.997]

    x = np.arange(len(ops))
    width = 0.25

    ax.bar(x - width, sens, width, label='Sensitivity', color='#E53935', alpha=0.85)
    ax.bar(x, spec, width, label='Specificity', color='#1976D2', alpha=0.85)
    ax.bar(x + width, npv, width, label='NPV', color='#43A047', alpha=0.85)

    for i in range(len(ops)):
        ax.text(x[i] - width, sens[i] + 0.005, f'{sens[i]:.1%}', ha='center', va='bottom', fontsize=8)
        ax.text(x[i], spec[i] + 0.005, f'{spec[i]:.1%}', ha='center', va='bottom', fontsize=8)
        ax.text(x[i] + width, npv[i] + 0.005, f'{npv[i]:.1%}', ha='center', va='bottom', fontsize=8)

    ax.axhline(y=0.95, color='red', linestyle='--', lw=1, alpha=0.6, label='95% line')
    ax.set_title('(a) Treatment', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(ops, fontsize=9)
    ax.set_ylim(0.6, 1.05)
    ax.set_ylabel('Score', fontsize=11)
    ax.legend(fontsize=8, loc='lower left')
    ax.grid(axis='y', alpha=0.3)

    # --- RW-ROP ---
    ax = axes[1]
    ops = ['Default\n(t=0.50)', 'Sens>=95%\n(t=0.44)', "Youden's J\n(t=0.57)"]
    sens = [0.912, 0.952, 0.892]
    spec = [0.849, 0.769, 0.903]
    npv = [0.971, 0.982, 0.967]

    ax.bar(x - width, sens, width, label='Sensitivity', color='#E53935', alpha=0.85)
    ax.bar(x, spec, width, label='Specificity', color='#1976D2', alpha=0.85)
    ax.bar(x + width, npv, width, label='NPV', color='#43A047', alpha=0.85)

    for i in range(len(ops)):
        ax.text(x[i] - width, sens[i] + 0.005, f'{sens[i]:.1%}', ha='center', va='bottom', fontsize=8)
        ax.text(x[i], spec[i] + 0.005, f'{spec[i]:.1%}', ha='center', va='bottom', fontsize=8)
        ax.text(x[i] + width, npv[i] + 0.005, f'{npv[i]:.1%}', ha='center', va='bottom', fontsize=8)

    ax.axhline(y=0.95, color='red', linestyle='--', lw=1, alpha=0.6, label='95% line')
    ax.set_title('(b) RW-ROP', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(ops, fontsize=9)
    ax.set_ylim(0.6, 1.05)
    ax.set_ylabel('Score', fontsize=11)
    ax.legend(fontsize=8, loc='lower left')
    ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Figure 3: Threshold Optimization — Operating Points',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    out = FIGURES_DIR / 'fig3_threshold_optimization.png'
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    fig1_architecture()
    fig2_arop_regularization()
    fig3_threshold_summary()
    print("\nAll figures generated.")
