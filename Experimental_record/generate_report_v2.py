# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.family'] = 'Meiryo'
plt.rcParams['axes.unicode_minus'] = False

from fpdf import FPDF
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, cohen_kappa_score, f1_score
from collections import Counter

# Output directory for figures
FIG_DIR = "C:/Users/ykita/ROP_AI_project/Experimental_record/figures"
os.makedirs(FIG_DIR, exist_ok=True)

# Load prediction data
pred_good_only = pd.read_csv("C:/Users/ykita/ROP_AI_project/ROP_project/multicenter_study/outputs_quality_comparison/good_only/predictions.csv")
pred_good_fair = pd.read_csv("C:/Users/ykita/ROP_AI_project/ROP_project/multicenter_study/outputs_quality_comparison/good_fair/predictions.csv")

# ============ Generate Confusion Matrix Figures ============
def plot_confusion_matrix(y_true, y_pred, classes, title, filename, cmap='Blues'):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_norm, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Proportion', rotation=-90, va="bottom", fontsize=9)

    # Labels
    ax.set(xticks=np.arange(len(classes)),
           yticks=np.arange(len(classes)),
           xticklabels=classes,
           yticklabels=classes,
           ylabel='True Label',
           xlabel='Predicted Label')
    ax.set_title(title, fontsize=11, fontweight='bold')

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    thresh = 0.5
    for i in range(len(classes)):
        for j in range(len(classes)):
            color = "white" if cm_norm[i, j] > thresh else "black"
            text = f"{cm[i, j]}\n({cm_norm[i, j]:.2f})"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()

def plot_roc_curve(y_true, y_score, title, filename):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color='#2ecc71', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random')
    ax.fill_between(fpr, tpr, alpha=0.3, color='#2ecc71')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Specificity)')
    ax.set_ylabel('True Positive Rate (Sensitivity)')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()

    return roc_auc

def plot_comparison_bar(metrics_dict, metric_name, title, filename):
    """Bar chart comparing good_only vs good_fair"""
    fig, ax = plt.subplots(figsize=(6, 4))

    tasks = list(metrics_dict['good_only'].keys())
    x = np.arange(len(tasks))
    width = 0.35

    vals1 = [metrics_dict['good_only'][t] for t in tasks]
    vals2 = [metrics_dict['good_fair'][t] for t in tasks]

    bars1 = ax.bar(x - width/2, vals1, width, label='good_only', color='#3498db')
    bars2 = ax.bar(x + width/2, vals2, width, label='good_fair', color='#2ecc71')

    ax.set_ylabel(metric_name)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, axis='y', alpha=0.3)

    # Add value labels
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()

# Generate figures for each condition
for cond, pred_df in [('good_only', pred_good_only), ('good_fair', pred_good_fair)]:
    # Zone
    y_true = pred_df['zone_label'].values.astype(int)
    y_pred = pred_df['zone_pred'].values.astype(int)
    mask = y_true >= 0
    plot_confusion_matrix(y_true[mask], y_pred[mask],
                          ['Zone I', 'Zone II', 'Zone III'],
                          f'Zone Classification ({cond})',
                          f'cm_zone_{cond}.png')

    # Stage
    y_true = pred_df['stage_label'].values.astype(int)
    y_pred = pred_df['stage_pred'].values.astype(int)
    mask = y_true >= 0
    plot_confusion_matrix(y_true[mask], y_pred[mask],
                          ['Stage 0', 'Stage 1', 'Stage 2', 'Stage 3'],
                          f'Stage Classification ({cond})',
                          f'cm_stage_{cond}.png')

    # Aggressive ROP - ROC
    y_true = pred_df['aggressive_rop_label'].values.astype(int)
    y_pred = pred_df['aggressive_rop_pred'].values.astype(int)
    mask = y_true >= 0
    plot_roc_curve(y_true[mask], y_pred[mask],
                   f'Aggressive ROP ({cond})',
                   f'roc_arop_{cond}.png')

    # Aggressive ROP - CM
    plot_confusion_matrix(y_true[mask], y_pred[mask],
                          ['No', 'Yes'],
                          f'Aggressive ROP ({cond})',
                          f'cm_arop_{cond}.png', cmap='Oranges')

    # Treatment - ROC
    y_true = pred_df['treatment_label'].values.astype(int)
    y_pred = pred_df['treatment_pred'].values.astype(int)
    mask = y_true >= 0
    plot_roc_curve(y_true[mask], y_pred[mask],
                   f'Treatment ({cond})',
                   f'roc_treatment_{cond}.png')

    # Treatment - CM
    plot_confusion_matrix(y_true[mask], y_pred[mask],
                          ['No', 'Yes'],
                          f'Treatment ({cond})',
                          f'cm_treatment_{cond}.png', cmap='Greens')

# Generate comparison bar charts
acc_dict = {'good_only': {}, 'good_fair': {}}
kappa_dict = {'good_only': {}, 'good_fair': {}}

for cond, pred_df in [('good_only', pred_good_only), ('good_fair', pred_good_fair)]:
    for task in ['zone', 'stage', 'aggressive_rop', 'treatment']:
        y_true = pred_df[f'{task}_label'].values.astype(int)
        y_pred = pred_df[f'{task}_pred'].values.astype(int)
        mask = y_true >= 0
        acc_dict[cond][task] = accuracy_score(y_true[mask], y_pred[mask])
        kappa_dict[cond][task] = cohen_kappa_score(y_true[mask], y_pred[mask], weights='quadratic')

plot_comparison_bar(acc_dict, 'Accuracy', 'Accuracy Comparison: good_only vs good_fair', 'comparison_accuracy.png')
plot_comparison_bar(kappa_dict, 'QW Kappa', 'Quadratic Weighted Kappa Comparison', 'comparison_kappa.png')

# Case-level ensemble ROC curves
def case_level_roc(pred_df, task, title, filename):
    """Generate ROC curve for case-level ensemble using pos_ratio as score"""
    label_col = f'{task}_label'
    pred_col = f'{task}_pred'

    case_true = []
    case_scores = []

    for vid, grp in pred_df.groupby('video_id'):
        y_t = grp[label_col].values
        y_p = grp[pred_col].values
        m = y_t >= 0
        y_t, y_p = y_t[m], y_p[m]
        if len(y_t) == 0: continue
        true_label = int(y_t[0])
        pos_ratio = (y_p == 1).mean()
        case_true.append(true_label)
        case_scores.append(pos_ratio)

    case_true = np.array(case_true)
    case_scores = np.array(case_scores)

    fpr, tpr, _ = roc_curve(case_true, case_scores)
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, roc_auc

# Combined ROC for case-level ensemble
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for idx, task in enumerate(['aggressive_rop', 'treatment']):
    ax = axes[idx]
    task_title = 'Aggressive ROP' if task == 'aggressive_rop' else 'Treatment'

    for cond, pred_df, color in [('good_only', pred_good_only, '#3498db'),
                                   ('good_fair', pred_good_fair, '#2ecc71')]:
        fpr, tpr, roc_auc = case_level_roc(pred_df, task, '', '')
        ax.plot(fpr, tpr, color=color, lw=2, label=f'{cond} (AUC={roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'Case-level ROC: {task_title}', fontsize=11, fontweight='bold')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'roc_case_level.png'), dpi=150, bbox_inches='tight')
plt.close()

print("Figures generated successfully")

# ============ Generate PDF ============
class PDF(FPDF):
    def __init__(self):
        super().__init__()
        font_path = "C:/Windows/Fonts/meiryo.ttc"
        if os.path.exists(font_path):
            self.add_font("Meiryo", "", font_path)
            self.add_font("Meiryo", "B", "C:/Windows/Fonts/meiryob.ttc")
            self.jp_font = "Meiryo"
        else:
            self.jp_font = "Helvetica"

    def header(self):
        self.set_font(self.jp_font, "B", 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, "ROP AI Project - Experimental Report", new_x="LMARGIN", new_y="NEXT", align="R")
        self.line(10, 18, 200, 18)
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font(self.jp_font, "", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def section_title(self, title):
        self.set_font(self.jp_font, "B", 14)
        self.set_text_color(30, 80, 150)
        self.set_fill_color(240, 245, 255)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT", align="L", fill=True)
        self.ln(3)

    def subsection_title(self, title):
        self.set_font(self.jp_font, "B", 11)
        self.set_text_color(50, 50, 50)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT", align="L")
        self.ln(1)

    def body_text(self, text):
        self.set_font(self.jp_font, "", 10)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 6, text)
        self.ln(2)

    def add_table(self, headers, data, col_widths=None):
        self.set_font(self.jp_font, "B", 9)
        self.set_fill_color(70, 130, 180)
        self.set_text_color(255, 255, 255)
        if col_widths is None:
            col_widths = [190 / len(headers)] * len(headers)
        for i, header in enumerate(headers):
            self.cell(col_widths[i], 7, header, border=1, align="C", fill=True)
        self.ln()
        self.set_font(self.jp_font, "", 9)
        self.set_text_color(0, 0, 0)
        fill = False
        for row in data:
            self.set_fill_color(245, 250, 255) if fill else self.set_fill_color(255, 255, 255)
            for i, cell in enumerate(row):
                self.cell(col_widths[i], 6, str(cell), border=1, align="C", fill=True)
            self.ln()
            fill = not fill
        self.ln(3)

    def highlight_box(self, text, color="blue"):
        colors = {
            "blue": ((230, 240, 255), (30, 60, 120)),
            "green": ((230, 255, 230), (30, 100, 30)),
            "orange": ((255, 245, 230), (150, 80, 0))
        }
        fill_c, text_c = colors.get(color, colors["blue"])
        self.set_fill_color(*fill_c)
        self.set_text_color(*text_c)
        self.set_font(self.jp_font, "", 9)
        self.multi_cell(0, 6, text, border=1, fill=True)
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def add_image_row(self, img1, img2, w=90):
        x_start = self.get_x()
        y_start = self.get_y()
        self.image(img1, x=x_start, y=y_start, w=w)
        self.image(img2, x=x_start + w + 5, y=y_start, w=w)
        self.set_y(y_start + w * 0.8)

# Create PDF
pdf = PDF()
pdf.set_auto_page_break(auto=True, margin=15)

# ===== Page 1: Title =====
pdf.add_page()
pdf.ln(30)
pdf.set_font(pdf.jp_font, "B", 24)
pdf.set_text_color(30, 60, 120)
pdf.cell(0, 15, "実験記録レポート", new_x="LMARGIN", new_y="NEXT", align="C")
pdf.set_font(pdf.jp_font, "", 16)
pdf.set_text_color(80, 80, 80)
pdf.cell(0, 10, "2026年1月28日", new_x="LMARGIN", new_y="NEXT", align="C")
pdf.ln(15)

pdf.set_font(pdf.jp_font, "", 12)
pdf.set_text_color(0, 0, 0)
contents = [
    "1. ROP分類器 画質サブセット比較",
    "   - Zone / Stage 分類結果",
    "   - Aggressive ROP / Treatment 検出",
    "2. 症例レベルアンサンブル検討",
    "   - Multi-class / Binary タスク分析"
]
pdf.cell(0, 10, "【目次】", new_x="LMARGIN", new_y="NEXT", align="C")
for c in contents:
    pdf.cell(0, 7, c, new_x="LMARGIN", new_y="NEXT", align="C")

pdf.ln(20)
pdf.set_font(pdf.jp_font, "", 10)
pdf.set_text_color(100, 100, 100)
pdf.cell(0, 8, "ROP AI Project - Multicenter Study", new_x="LMARGIN", new_y="NEXT", align="C")

# ===== Page 2: Overview & Comparison =====
pdf.add_page()
pdf.section_title("1. ROP分類器 画質サブセット比較")

pdf.subsection_title("実験概要")
pdf.body_text("EfficientNet-B0ベースのマルチタスクROP分類器を、異なる画質サブセットで学習・評価。\n- good_only: Good画像のみ (n=3,453, 235症例)\n- good_fair: Good + Fair画像 (n=4,978, 259症例)\n5-fold交差検証、患者レベル層化分割を使用。")

pdf.subsection_title("全体比較")
pdf.image(os.path.join(FIG_DIR, 'comparison_accuracy.png'), w=90)
pdf.image(os.path.join(FIG_DIR, 'comparison_kappa.png'), x=105, y=pdf.get_y()-60, w=90)
pdf.ln(5)

# ===== Page 3: Zone =====
pdf.add_page()
pdf.subsection_title("Zone分類 結果")
pdf.add_table(
    ["指標", "good_only", "good_fair", "差分"],
    [
        ["Accuracy", "0.7118", "0.7137", "+0.002"],
        ["QW Kappa", "0.4530", "0.5024", "+0.049"],
        ["Macro F1", "0.6216", "0.6323", "+0.011"],
    ],
    [40, 50, 50, 50]
)

pdf.subsection_title("Confusion Matrix")
pdf.add_image_row(
    os.path.join(FIG_DIR, 'cm_zone_good_only.png'),
    os.path.join(FIG_DIR, 'cm_zone_good_fair.png'),
    w=90
)

pdf.subsection_title("クラス別性能 (good_fair)")
pdf.add_table(
    ["Class", "n", "Sensitivity", "Specificity", "PPV", "F1"],
    [
        ["Zone I", "651", "0.458", "0.980", "0.776", "0.576"],
        ["Zone II", "3,127", "0.860", "0.485", "0.738", "0.794"],
        ["Zone III", "1,200", "0.473", "0.898", "0.595", "0.527"],
    ],
    [30, 25, 35, 35, 30, 30]
)

# ===== Page 4: Stage =====
pdf.add_page()
pdf.subsection_title("Stage分類 結果")
pdf.add_table(
    ["指標", "good_only", "good_fair", "差分"],
    [
        ["Accuracy", "0.6432", "0.6531", "+0.010"],
        ["QW Kappa", "0.6497", "0.6674", "+0.018"],
        ["Macro F1", "0.6347", "0.6386", "+0.004"],
    ],
    [40, 50, 50, 50]
)

pdf.subsection_title("Confusion Matrix")
pdf.add_image_row(
    os.path.join(FIG_DIR, 'cm_stage_good_only.png'),
    os.path.join(FIG_DIR, 'cm_stage_good_fair.png'),
    w=90
)

pdf.subsection_title("クラス別性能 (good_fair)")
pdf.add_table(
    ["Class", "n", "Sensitivity", "Specificity", "PPV", "F1"],
    [
        ["Stage 0", "1,969", "0.749", "0.743", "0.657", "0.700"],
        ["Stage 1", "1,355", "0.555", "0.873", "0.622", "0.586"],
        ["Stage 2", "823", "0.447", "0.944", "0.614", "0.518"],
        ["Stage 3", "809", "0.795", "0.937", "0.711", "0.751"],
    ],
    [30, 25, 35, 35, 30, 30]
)

# ===== Page 5: Aggressive ROP =====
pdf.add_page()
pdf.subsection_title("Aggressive ROP 検出")
pdf.add_table(
    ["指標", "good_only", "good_fair", "差分"],
    [
        ["Accuracy", "0.9615", "0.9681", "+0.007"],
        ["QW Kappa", "0.3698", "0.5465", "+0.177"],
        ["AUC-ROC", "0.6386", "0.7202", "+0.082"],
        ["Sensitivity", "0.286", "0.447", "+0.161"],
    ],
    [40, 50, 50, 50]
)

pdf.highlight_box("Fair画像追加でAggressive ROPの検出力が大幅改善 (Kappa +0.18, AUC +0.08, Sens +0.16)", "green")

pdf.subsection_title("Confusion Matrix & ROC Curve")
pdf.add_image_row(
    os.path.join(FIG_DIR, 'cm_arop_good_only.png'),
    os.path.join(FIG_DIR, 'cm_arop_good_fair.png'),
    w=90
)
pdf.ln(5)
pdf.add_image_row(
    os.path.join(FIG_DIR, 'roc_arop_good_only.png'),
    os.path.join(FIG_DIR, 'roc_arop_good_fair.png'),
    w=90
)

# ===== Page 6: Treatment =====
pdf.add_page()
pdf.subsection_title("Treatment 予測")
pdf.add_table(
    ["指標", "good_only", "good_fair", "差分"],
    [
        ["Accuracy", "0.9401", "0.9413", "+0.001"],
        ["QW Kappa", "0.7052", "0.7050", "-0.000"],
        ["AUC-ROC", "0.8350", "0.8184", "-0.017"],
        ["Sensitivity", "0.696", "0.654", "-0.042"],
    ],
    [40, 50, 50, 50]
)

pdf.highlight_box("Treatmentはほぼ変化なし（やや低下傾向）", "orange")

pdf.subsection_title("Confusion Matrix & ROC Curve")
pdf.add_image_row(
    os.path.join(FIG_DIR, 'cm_treatment_good_only.png'),
    os.path.join(FIG_DIR, 'cm_treatment_good_fair.png'),
    w=90
)
pdf.ln(5)
pdf.add_image_row(
    os.path.join(FIG_DIR, 'roc_treatment_good_only.png'),
    os.path.join(FIG_DIR, 'roc_treatment_good_fair.png'),
    w=90
)

# ===== Page 7: Section 2 - Ensemble =====
pdf.add_page()
pdf.section_title("2. 症例レベルアンサンブル検討")

pdf.subsection_title("概要")
pdf.body_text("同一症例の全画像を多数決アンサンブルした場合の症例レベルメトリクスをシミュレーション。\n・症例あたり画像数: good_only 平均14.7枚、good_fair 平均19.2枚\n・アンサンブル方法: 多数決 (majority vote)")

pdf.subsection_title("Multi-classタスク: 改善")
pdf.add_table(
    ["Task", "Metric", "good_only (Img->Case)", "good_fair (Img->Case)"],
    [
        ["Zone", "Accuracy", "0.711->0.740 (+2.9%)", "0.714->0.753 (+3.9%)"],
        ["Zone", "Kappa", "0.453->0.458 (+0.5%)", "0.502->0.537 (+3.5%)"],
        ["Stage", "Accuracy", "0.643->0.654 (+1.1%)", "0.653->0.667 (+1.4%)"],
        ["Stage", "Kappa", "0.650->0.687 (+3.7%)", "0.667->0.719 (+5.2%)"],
    ],
    [30, 30, 65, 65]
)
pdf.highlight_box("多数決アンサンブルでZone/Stageとも改善。特にKappaの改善が顕著", "green")

pdf.subsection_title("Binaryタスク: 単純多数決では悪化")
pdf.add_table(
    ["Task", "Condition", "Image->Case Sens", "原因"],
    [
        ["Aggressive ROP", "good_only", "0.286->0.111", "陽性率中央値5.6%"],
        ["Aggressive ROP", "good_fair", "0.447->0.444", "陽性率中央値40%"],
        ["Treatment", "good_only", "0.696->0.645", "陽性率中央値80%"],
        ["Treatment", "good_fair", "0.654->0.636", "陽性率中央値75%"],
    ],
    [45, 35, 50, 60]
)
pdf.highlight_box("クラス不均衡により、陽性症例内でも陽性予測が少数派となり多数決で陰性に倒れる", "orange")

# ===== Page 8: Case-level ROC =====
pdf.add_page()
pdf.subsection_title("症例レベルROC曲線 (閾値最適化)")
pdf.body_text("陽性画像比率 (pos_ratio) をスコアとした症例レベルROC曲線:")
pdf.image(os.path.join(FIG_DIR, 'roc_case_level.png'), w=180)

pdf.subsection_title("閾値最適化結果")
pdf.add_table(
    ["Task", "Condition", "最適閾値", "Sens改善", "Case F1", "Case AUC"],
    [
        ["Aggressive ROP", "good_only", "0.05", "0.286->0.667", "0.791", "0.822"],
        ["Aggressive ROP", "good_fair", "0.15", "0.447->0.778", "0.863", "0.991"],
        ["Treatment", "good_only", "0.20", "0.696->0.806", "0.852", "0.916"],
        ["Treatment", "good_fair", "0.05", "0.654->0.909", "0.870", "0.939"],
    ],
    [40, 30, 25, 35, 30, 30]
)
pdf.highlight_box("good_fairのAggressive ROP Case-AUC 0.991 は非常に高い判別能力", "green")

# ===== Page 9: Conclusions =====
pdf.add_page()
pdf.section_title("結論・推奨")

pdf.subsection_title("画質サブセット比較")
pdf.body_text("1. Fair画像追加でAggressive ROPの検出力が大幅改善 (Kappa +0.18)\n2. Zone, StageはFair追加で微小な改善\n3. Treatmentはほぼ変化なし")

pdf.subsection_title("症例レベルアンサンブル")
pdf.body_text("1. Multi-classタスク (Zone, Stage): 多数決アンサンブルで安定して改善\n2. Binaryタスク: 閾値を下げたアンサンブル、またはsoft voting (確率平均) を推奨\n3. good_fair > good_only: 画像数が多いほどアンサンブル効果が大きい")

pdf.subsection_title("今後の課題")
pdf.body_text("1. Zone I/III, Stage 2の感度が0.45前後と低い\n2. logit/softmax確率を保存したsoft votingの実装\n3. 症例レベル評価の正式実装")

# Save
output_path = "C:/Users/ykita/ROP_AI_project/Experimental_record/20260128_report_v2.pdf"
pdf.output(output_path)
print(f"PDF saved to: {output_path}")
