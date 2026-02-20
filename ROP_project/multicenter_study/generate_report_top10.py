"""
Top-10/Top-5 品質フィルタリング評価レポート生成 (PDF)
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec

# --- Japanese font ---
from matplotlib import font_manager
jp_fonts = [f.name for f in font_manager.fontManager.ttflist if 'Gothic' in f.name or 'Meiryo' in f.name]
if jp_fonts:
    matplotlib.rcParams['font.family'] = jp_fonts[0]
else:
    matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['axes.unicode_minus'] = False

OUTPUT_PATH = r"C:\Users\ykita\ROP_AI_project\ROP_project\multicenter_study\outputs_clinical_v3\top10_quality_filter_report.pdf"


def add_title_page(pdf):
    fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape
    fig.text(0.5, 0.62, "Top-K 画像品質フィルタリングによる\nROP分類性能への影響評価",
             ha='center', va='center', fontsize=24, fontweight='bold')
    fig.text(0.5, 0.42, "clinical_v3 モデル (5タスク + 臨床データ融合)\n"
             "C-rule3_thr0.80 (disc_edge_coverage >= 0.80 + 3特徴量スコアリング)",
             ha='center', va='center', fontsize=14, color='#444444')
    fig.text(0.5, 0.28, "2026-02-11", ha='center', va='center', fontsize=13, color='#666666')
    fig.text(0.5, 0.15, "multicenter_study / evaluate_top10_quality_filtered.ipynb",
             ha='center', va='center', fontsize=10, color='#888888', family='monospace')
    plt.axis('off')
    pdf.savefig(fig)
    plt.close(fig)


def add_overview_page(pdf):
    fig = plt.figure(figsize=(11.69, 8.27))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2.5], hspace=0.3,
                           left=0.08, right=0.92, top=0.92, bottom=0.06)

    # Title
    fig.text(0.5, 0.96, "1. 背景と方法", ha='center', fontsize=18, fontweight='bold')

    # Background text
    ax_text = fig.add_subplot(gs[0])
    ax_text.axis('off')
    text = (
        "背景: clinical_v3モデルは全Good+Fair画像 6,448枚 (347 videos) で評価されている。\n"
        "臨床現場では動画から高品質画像のみを選別してAI判定に使う運用が想定される。\n\n"
        "目的: 品質フィルタリング後のサブセット (Top-10, Top-5) で分類性能がどう変化するか検証。"
    )
    ax_text.text(0.02, 0.7, text, transform=ax_text.transAxes, fontsize=12,
                 verticalalignment='top', linespacing=1.6)

    # Method table
    ax_tbl = fig.add_subplot(gs[1])
    ax_tbl.axis('off')

    col_labels = ['項目', '内容']
    table_data = [
        ['対象データ', 'predictions.csv (6,448 images, 347 video_ids, 5-fold CV)'],
        ['品質特徴量', 'retina_ratio, mbss_Grad_p90, mbss_score, disc_edge_coverage_ratio'],
        ['選出アルゴリズム', 'C-rule3_thr0.80 (per video_id)'],
        ['Stage 1', 'disc_edge_coverage >= 0.80 → 0.4×retina + 0.4×grad + 0.2×mbss 降順'],
        ['Stage 2 (Fallback)', 'Stage1不足分を retina_ratio 降順で補完'],
        ['評価タスク', 'Zone (3cls), Stage (4cls), Plus (3cls), AROP (bin), Treatment (bin)'],
        ['派生指標', 'RW-ROP = Plus(2) OR Stage(3) OR Zone(0)'],
        ['閾値最適化', 'Default(0.5), Sensitivity>=95%, Youden\'s J'],
    ]

    tbl = ax_tbl.table(cellText=table_data, colLabels=col_labels, loc='center',
                       cellLoc='left', colWidths=[0.18, 0.75])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.6)
    for key, cell in tbl.get_celld().items():
        if key[0] == 0:
            cell.set_facecolor('#4472C4')
            cell.set_text_props(color='white', fontweight='bold')
        elif key[0] % 2 == 0:
            cell.set_facecolor('#D6E4F0')
        cell.set_edgecolor('#CCCCCC')

    pdf.savefig(fig)
    plt.close(fig)


def add_data_summary_page(pdf):
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.text(0.5, 0.96, "2. データ概要・選出結果", ha='center', fontsize=18, fontweight='bold')

    gs = gridspec.GridSpec(1, 2, wspace=0.3, left=0.08, right=0.92, top=0.88, bottom=0.15)

    # Left: selection summary table
    ax_l = fig.add_subplot(gs[0])
    ax_l.axis('off')
    col_labels = ['', 'All', 'Top-10', 'Top-5']
    data = [
        ['画像数', '6,448', '3,089', '1,650'],
        ['削減率', '-', '52.1%', '74.4%'],
        ['Video数', '347', '347', '347'],
        ['Stage1率', '-', '-', '97.0%'],
        ['Fallback使用', '-', '-', '13 videos'],
        ['画像不足(<K)', '-', '-', '33 videos'],
    ]
    tbl = ax_l.table(cellText=data, colLabels=col_labels, loc='center',
                     cellLoc='center', colWidths=[0.28, 0.2, 0.2, 0.2])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.8)
    for key, cell in tbl.get_celld().items():
        if key[0] == 0:
            cell.set_facecolor('#4472C4')
            cell.set_text_props(color='white', fontweight='bold')
        elif key[1] == 0:
            cell.set_facecolor('#E2EFDA')
            cell.set_text_props(fontweight='bold')
        elif key[0] % 2 == 0:
            cell.set_facecolor('#D6E4F0')
        cell.set_edgecolor('#CCCCCC')
    ax_l.set_title('選出サマリ', fontsize=13, fontweight='bold', pad=15)

    # Right: bar chart of image counts
    ax_r = fig.add_subplot(gs[1])
    conditions = ['All', 'Top-10', 'Top-5']
    counts = [6448, 3089, 1650]
    colors = ['#4472C4', '#ED7D31', '#70AD47']
    bars = ax_r.bar(conditions, counts, color=colors, width=0.6, edgecolor='white')
    for bar, count in zip(bars, counts):
        ax_r.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 80,
                  f'{count:,}', ha='center', fontsize=13, fontweight='bold')
    ax_r.set_ylabel('画像数', fontsize=12)
    ax_r.set_title('評価画像数の比較', fontsize=13, fontweight='bold')
    ax_r.set_ylim(0, 7500)
    ax_r.spines['top'].set_visible(False)
    ax_r.spines['right'].set_visible(False)

    pdf.savefig(fig)
    plt.close(fig)


def add_multiclass_results_page(pdf):
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.text(0.5, 0.96, "3. Multiclass タスク (Zone / Stage / Plus)", ha='center', fontsize=18, fontweight='bold')

    # Data
    tasks = ['Zone', 'Stage', 'Plus']
    metrics = ['Accuracy', 'Kappa', 'F1 macro']
    all_vals = [
        [0.7323, 0.6660, 0.7132],
        [0.7298, 0.7613, 0.5780],
        [0.8826, 0.7502, 0.5440],
    ]
    top10_vals = [
        [0.7339, 0.6613, 0.7110],
        [0.7193, 0.7669, 0.5668],
        [0.8792, 0.7332, 0.5268],
    ]
    top5_vals = [
        [0.7273, 0.6600, 0.7058],
        [0.7194, 0.7723, 0.5660],
        [0.8800, 0.7261, 0.5218],
    ]

    gs = gridspec.GridSpec(1, 3, wspace=0.35, left=0.08, right=0.95, top=0.86, bottom=0.35)

    colors = ['#4472C4', '#ED7D31', '#70AD47']
    for i, task in enumerate(tasks):
        ax = fig.add_subplot(gs[i])
        x = np.arange(len(metrics))
        w = 0.25
        ax.bar(x - w, all_vals[i], w, label='All', color=colors[0], alpha=0.85)
        ax.bar(x, top10_vals[i], w, label='Top-10', color=colors[1], alpha=0.85)
        ax.bar(x + w, top5_vals[i], w, label='Top-5', color=colors[2], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=9, rotation=15)
        ax.set_ylim(0.4, 0.95)
        ax.set_title(task, fontsize=14, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if i == 0:
            ax.legend(fontsize=9, loc='lower left')

    # Table below
    ax_tbl = fig.add_axes([0.06, 0.04, 0.88, 0.25])
    ax_tbl.axis('off')
    col_labels = ['Task', 'Metric', 'All (6448)', 'Top-10 (3089)', 'Top-5 (1650)', 'Δ(Top5-All)']
    rows = []
    for i, task in enumerate(tasks):
        for j, met in enumerate(metrics):
            a, t10, t5 = all_vals[i][j], top10_vals[i][j], top5_vals[i][j]
            d = t5 - a
            rows.append([task, met, f'{a:.4f}', f'{t10:.4f}', f'{t5:.4f}', f'{d:+.4f}'])
    tbl = ax_tbl.table(cellText=rows, colLabels=col_labels, loc='center',
                       cellLoc='center', colWidths=[0.1, 0.12, 0.15, 0.15, 0.15, 0.15])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.35)
    for key, cell in tbl.get_celld().items():
        if key[0] == 0:
            cell.set_facecolor('#4472C4')
            cell.set_text_props(color='white', fontweight='bold', fontsize=8)
        elif key[0] % 2 == 0:
            cell.set_facecolor('#D6E4F0')
        cell.set_edgecolor('#CCCCCC')
        # Highlight delta column
        if key[0] > 0 and key[1] == 5:
            val = float(rows[key[0]-1][5])
            if val < -0.02:
                cell.set_text_props(color='#C00000')
            elif val > 0.005:
                cell.set_text_props(color='#007030')

    pdf.savefig(fig)
    plt.close(fig)


def add_binary_results_page(pdf):
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.text(0.5, 0.96, "4. Binary タスク (Treatment / AROP / RW-ROP)", ha='center', fontsize=18, fontweight='bold')

    # Data: [All, Top10, Top5]
    data = {
        'Treatment': {
            'Sensitivity': [0.9171, 0.9254, 0.9363],
            'Specificity': [0.9526, 0.9474, 0.9431],
            'AUC': [0.9780, 0.9798, 0.9841],
            'F1': [0.7781, 0.7636, 0.7558],
            'PPV': [0.6757, 0.6500, 0.6336],
            'NPV': [0.9907, 0.9918, 0.9929],
        },
        'AROP': {
            'Sensitivity': [0.8596, 0.8889, 0.9333],
            'Specificity': [0.7749, 0.7696, 0.7607],
            'AUC': [0.9164, 0.9357, 0.9497],
            'F1': [0.2149, 0.1858, 0.1783],
            'PPV': [0.1228, 0.1038, 0.0986],
            'NPV': [0.9934, 0.9957, 0.9975],
        },
        'RW-ROP': {
            'Sensitivity': [0.8652, 0.8312, 0.8273],
            'Specificity': [0.9080, 0.9065, 0.8962],
            'AUC': [0.9536, 0.9379, 0.9311],
            'F1': [0.7894, 0.7741, 0.7643],
            'PPV': [0.7259, 0.7244, 0.7102],
            'NPV': [0.9599, 0.9478, 0.9441],
        }
    }

    gs = gridspec.GridSpec(1, 3, wspace=0.35, left=0.07, right=0.96, top=0.86, bottom=0.38)
    colors = ['#4472C4', '#ED7D31', '#70AD47']
    key_metrics = ['Sensitivity', 'Specificity', 'AUC', 'PPV', 'NPV']

    for i, task in enumerate(['Treatment', 'AROP', 'RW-ROP']):
        ax = fig.add_subplot(gs[i])
        x = np.arange(len(key_metrics))
        w = 0.25
        for j, (cond, color) in enumerate(zip(['All', 'Top-10', 'Top-5'], colors)):
            vals = [data[task][m][j] for m in key_metrics]
            offset = (j - 1) * w
            ax.bar(x + offset, vals, w, label=cond, color=color, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(key_metrics, fontsize=8, rotation=20)
        ax.set_ylim(0.0, 1.08)
        ax.set_title(task, fontsize=14, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if i == 0:
            ax.legend(fontsize=8)

    # Table
    ax_tbl = fig.add_axes([0.04, 0.02, 0.92, 0.32])
    ax_tbl.axis('off')
    col_labels = ['Task', 'Metric', 'All', 'Top-10', 'Top-5', 'Δ(Top5-All)']
    rows = []
    for task in ['Treatment', 'AROP', 'RW-ROP']:
        for met in ['Sensitivity', 'Specificity', 'AUC']:
            vals = data[task][met]
            d = vals[2] - vals[0]
            rows.append([task, met, f'{vals[0]:.4f}', f'{vals[1]:.4f}', f'{vals[2]:.4f}', f'{d:+.4f}'])
    tbl = ax_tbl.table(cellText=rows, colLabels=col_labels, loc='center',
                       cellLoc='center', colWidths=[0.12, 0.13, 0.13, 0.13, 0.13, 0.14])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)
    for key, cell in tbl.get_celld().items():
        if key[0] == 0:
            cell.set_facecolor('#4472C4')
            cell.set_text_props(color='white', fontweight='bold', fontsize=8)
        elif key[0] % 2 == 0:
            cell.set_facecolor('#D6E4F0')
        cell.set_edgecolor('#CCCCCC')
        if key[0] > 0 and key[1] == 5:
            val = float(rows[key[0]-1][5])
            if val < -0.01:
                cell.set_text_props(color='#C00000', fontweight='bold')
            elif val > 0.005:
                cell.set_text_props(color='#007030', fontweight='bold')

    pdf.savefig(fig)
    plt.close(fig)


def add_trend_page(pdf):
    """Sensitivity/AUC trend across All → Top-10 → Top-5"""
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.text(0.5, 0.96, "5. 画像削減に伴う Sensitivity / AUC の推移",
             ha='center', fontsize=18, fontweight='bold')

    gs = gridspec.GridSpec(1, 2, wspace=0.3, left=0.1, right=0.92, top=0.86, bottom=0.15)
    x_labels = ['All\n(6,448)', 'Top-10\n(3,089)', 'Top-5\n(1,650)']
    x = [0, 1, 2]

    # Sensitivity trend
    ax1 = fig.add_subplot(gs[0])
    tasks_sens = {
        'Treatment': [0.9171, 0.9254, 0.9363],
        'AROP': [0.8596, 0.8889, 0.9333],
        'RW-ROP': [0.8652, 0.8312, 0.8273],
    }
    markers = {'Treatment': 'o', 'AROP': 's', 'RW-ROP': '^'}
    colors = {'Treatment': '#4472C4', 'AROP': '#ED7D31', 'RW-ROP': '#C00000'}
    for task, vals in tasks_sens.items():
        ax1.plot(x, vals, marker=markers[task], color=colors[task],
                 linewidth=2.5, markersize=10, label=task)
        for xi, v in zip(x, vals):
            ax1.annotate(f'{v:.3f}', (xi, v), textcoords="offset points",
                        xytext=(0, 12), ha='center', fontsize=9, color=colors[task])
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels, fontsize=11)
    ax1.set_ylabel('Sensitivity', fontsize=13)
    ax1.set_title('Sensitivity', fontsize=14, fontweight='bold')
    ax1.set_ylim(0.78, 0.98)
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # AUC trend
    ax2 = fig.add_subplot(gs[1])
    tasks_auc = {
        'Treatment': [0.9780, 0.9798, 0.9841],
        'AROP': [0.9164, 0.9357, 0.9497],
        'RW-ROP': [0.9536, 0.9379, 0.9311],
    }
    for task, vals in tasks_auc.items():
        ax2.plot(x, vals, marker=markers[task], color=colors[task],
                 linewidth=2.5, markersize=10, label=task)
        for xi, v in zip(x, vals):
            ax2.annotate(f'{v:.3f}', (xi, v), textcoords="offset points",
                        xytext=(0, 12), ha='center', fontsize=9, color=colors[task])
    ax2.set_xticks(x)
    ax2.set_xticklabels(x_labels, fontsize=11)
    ax2.set_ylabel('AUC', fontsize=13)
    ax2.set_title('AUC-ROC', fontsize=14, fontweight='bold')
    ax2.set_ylim(0.90, 1.0)
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Annotation
    fig.text(0.5, 0.06,
             "Treatment / AROP: 画像を絞るほど Sensitivity・AUC ともに単調改善\n"
             "RW-ROP: 画像を絞るほど低下 → 成分別分析で原因を解明（次ページ）",
             ha='center', fontsize=12, style='italic', color='#333333',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF2CC', edgecolor='#D6B656'))

    pdf.savefig(fig)
    plt.close(fig)


def add_rwrop_analysis_page(pdf):
    """RW-ROP低下の原因分析"""
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.text(0.5, 0.96, "6. RW-ROP低下の原因分析", ha='center', fontsize=18, fontweight='bold')

    gs = gridspec.GridSpec(2, 2, wspace=0.35, hspace=0.45,
                           left=0.08, right=0.92, top=0.88, bottom=0.08)

    # Component sensitivity
    ax1 = fig.add_subplot(gs[0, 0])
    components = ['Zone I\n(zone=0)', 'Stage 3\n(stage=3)', 'Plus\n(plus=2)', 'RW-ROP\n(OR)']
    all_sens = [0.8173, 0.8381, 0.6838, 0.8652]
    top10_sens = [0.7890, 0.7913, 0.6453, 0.8312]
    top5_sens = [0.7816, 0.7941, 0.6154, 0.8273]

    x = np.arange(len(components))
    w = 0.25
    ax1.bar(x - w, all_sens, w, label='All', color='#4472C4', alpha=0.85)
    ax1.bar(x, top10_sens, w, label='Top-10', color='#ED7D31', alpha=0.85)
    ax1.bar(x + w, top5_sens, w, label='Top-5', color='#70AD47', alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(components, fontsize=9)
    ax1.set_ylabel('Sensitivity', fontsize=10)
    ax1.set_ylim(0.5, 0.95)
    ax1.set_title('RW-ROP 成分別 Sensitivity', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Delta chart
    ax2 = fig.add_subplot(gs[0, 1])
    comp_names = ['Zone I', 'Stage 3', 'Plus', 'RW-ROP']
    deltas = [0.7816-0.8173, 0.7941-0.8381, 0.6154-0.6838, 0.8273-0.8652]
    bar_colors = ['#C00000' if d < 0 else '#007030' for d in deltas]
    bars = ax2.barh(comp_names, deltas, color=bar_colors, alpha=0.8)
    for bar, d in zip(bars, deltas):
        ax2.text(bar.get_width() + (0.002 if d >= 0 else -0.002), bar.get_y() + bar.get_height()/2,
                 f'{d:+.3f}', ha='left' if d >= 0 else 'right', va='center', fontsize=11, fontweight='bold')
    ax2.axvline(0, color='black', linewidth=0.8)
    ax2.set_xlabel('Δ Sensitivity (Top-5 - All)', fontsize=10)
    ax2.set_title('成分別 Sensitivity 変化量', fontsize=12, fontweight='bold')
    ax2.set_xlim(-0.10, 0.02)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Miss pattern
    ax3 = fig.add_subplot(gs[1, 0])
    categories = ['Zone I\nonly', 'Stage 3\nonly', 'Plus\nonly', 'Multiple']
    all_miss = [97, 67, 1, 26]
    top5_miss = [34, 24, 0, 9]
    x = np.arange(len(categories))
    w = 0.35
    ax3.bar(x - w/2, all_miss, w, label='All (191 misses)', color='#4472C4', alpha=0.85)
    ax3.bar(x + w/2, top5_miss, w, label='Top-5 (67 misses)', color='#70AD47', alpha=0.85)
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories, fontsize=9)
    ax3.set_ylabel('Miss count', fontsize=10)
    ax3.set_title('RW-ROP False Negative 内訳', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=8)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    # Explanation text
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    explanation = (
        "RW-ROP低下の原因:\n\n"
        "1. RW-ROPはZone I / Stage 3 / Plus の\n"
        "   OR結合（派生指標）\n\n"
        "2. 3成分すべてのSensitivityが低下:\n"
        "   Zone I: -3.6pp\n"
        "   Stage 3: -4.4pp\n"
        "   Plus:   -6.8pp (最大)\n\n"
        "3. 原因: 品質スコアは一般的な画像鮮明度\n"
        "   を評価するが、病的所見の可視性とは\n"
        "   独立。Plus（血管拡張・蛇行）を示す\n"
        "   画像が品質上位に含まれにくい\n\n"
        "4. 対照的にTreatment/AROPは専用の\n"
        "   二値分類ヘッドで直接予測 → 高品質\n"
        "   画像で予測確度が向上"
    )
    ax4.text(0.05, 0.95, explanation, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', linespacing=1.4,
             bbox=dict(boxstyle='round,pad=0.6', facecolor='#FFF2CC', edgecolor='#D6B656'))

    pdf.savefig(fig)
    plt.close(fig)


def add_conclusion_page(pdf):
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.text(0.5, 0.96, "7. 結論", ha='center', fontsize=18, fontweight='bold')

    ax = fig.add_axes([0.08, 0.08, 0.84, 0.82])
    ax.axis('off')

    conclusions = (
        "主要な知見\n"
        "================================================\n\n"
        "1. 画像数74%削減 (Top-5) でも主要性能を維持\n"
        "   Zone/Stage/Plus の accuracy, kappa は Δ < 2.5pp\n\n"
        "2. Treatment / AROP は画像を絞るほど改善（単調増加）\n"
        "   Treatment AUC:  0.978 → 0.980 → 0.984 (全条件最高)\n"
        "   AROP Sensitivity: 0.860 → 0.889 → 0.933 (+7.4pp)\n"
        "   → 低品質画像がノイズとなり重症検出を妨げていた\n\n"
        "3. RW-ROP のみ低下 (Sensitivity -3.8pp, AUC -2.2pp)\n"
        "   原因: Zone I / Stage 3 / Plus の個別Sensitivityが低下\n"
        "   特にPlus (-6.8pp) の影響が大きい\n"
        "   品質スコアは病的所見の可視性と独立であるため\n\n"
        "4. 臨床運用上の結論\n"
        "   - Top-5~10枚で十分な診断性能が得られる\n"
        "   - スクリーニング (Treatment/AROP) はむしろ改善\n"
        "   - RW-ROP判定には閾値調整で対応可能\n"
        "     (Youden: sens=0.822, spec=0.915 at thr=0.61)\n\n"
        "================================================\n\n"
        "次のステップ\n\n"
        "  - 患者単位の多数決集約 (majority vote) による性能評価\n"
        "  - Plus検出を重視した品質スコアの改良検討\n"
        "  - 前向き検証データでの再現性確認"
    )

    ax.text(0.05, 0.92, conclusions, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', linespacing=1.5)

    pdf.savefig(fig)
    plt.close(fig)


def main():
    print(f"Generating PDF report: {OUTPUT_PATH}")
    with PdfPages(OUTPUT_PATH) as pdf:
        add_title_page(pdf)
        add_overview_page(pdf)
        add_data_summary_page(pdf)
        add_multiclass_results_page(pdf)
        add_binary_results_page(pdf)
        add_trend_page(pdf)
        add_rwrop_analysis_page(pdf)
        add_conclusion_page(pdf)
    print(f"Done! 8 pages saved to {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
