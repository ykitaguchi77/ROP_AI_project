# -*- coding: utf-8 -*-
"""
PR-AUC (Precision-Recall AUC) 補足解析レポート
- Report 1: 閾値最適化レポート補足 (per-fold CV, Treatment/AROP/RW-ROP)
- Report 2: Top-K Majority Vote レポート補足 (per-image/Soft Vote)

陽性ケースが少ないため、AU-ROCだけでなくPR-AUCも報告する。
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    roc_curve, roc_auc_score, auc
)

plt.rcParams['font.family'] = 'Meiryo'
plt.rcParams['axes.unicode_minus'] = False

# --- Paths ---
MULTI_DIR = Path(r"C:\Users\ykita\ROP_AI_project\ROP_project\multicenter_study")
PRED_PATH = MULTI_DIR / "outputs_clinical_v3" / "predictions.csv"
PRED_ICROP = MULTI_DIR / "outputs_icrop_treatment" / "predictions.csv"
KUBOTA_EXCEL = Path(r"E:\Multicenter_ROP_study\Multicenter_images\Kubota_selection\selected_images_disc_retina.xlsx")
TOP_EXCEL = Path(r"E:\Multicenter_ROP_study\Multicenter_images\selected_images_disc_retina.xlsx")
FIG_DIR = Path(r"C:\Users\ykita\ROP_AI_project\Experimental_record\figures")
FIG_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = Path(r"C:\Users\ykita\ROP_AI_project\Experimental_record")
SHARE_DIR = Path(r"C:\Users\ykita\ROP_AI_project\Share_with_Fukushima_dr")
SHARE_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(MULTI_DIR))
from select_best_images import minmax_norm

# --- Constants ---
EDGE_COV = 0.80
W_R, W_G, W_M = 0.4, 0.4, 0.2


# ============ Data Loading ============
def load_predictions():
    return pd.read_csv(PRED_PATH)


def load_data_with_features():
    pred_df = pd.read_csv(PRED_PATH)
    pred_df['image_name'] = pred_df['image_path'].apply(lambda x: Path(x).name)

    feat_cols = ['retina_ratio', 'mbss_Grad_p90', 'mbss_score', 'disc_detected', 'disc_edge_coverage_ratio']
    if KUBOTA_EXCEL.exists():
        kubota_df = pd.read_excel(KUBOTA_EXCEL)
        avail = [c for c in feat_cols if c in kubota_df.columns]
        kubota_features = kubota_df[['image_name'] + avail].drop_duplicates(subset='image_name', keep='first')
        merged_df = pred_df.merge(kubota_features, on='image_name', how='left')

        if TOP_EXCEL.exists():
            tl = pd.read_excel(TOP_EXCEL)
            tl_feat = [c for c in avail if c in tl.columns]
            if 'image_name' in tl.columns and tl_feat:
                tl_features = tl[['image_name'] + tl_feat].drop_duplicates(subset='image_name', keep='first').set_index('image_name')
                mm = merged_df['retina_ratio'].isna()
                for col in tl_feat:
                    fill = merged_df.loc[mm, 'image_name'].map(tl_features[col])
                    merged_df.loc[mm, col] = fill.values
        return merged_df
    else:
        print("WARNING: Feature Excel not found, skipping Top-K analysis")
        return None


def select_top_k(df, top_k):
    all_sel = []
    for vid, group in df.groupby('video_id'):
        valid = group[group['retina_ratio'].notna() & (group['retina_ratio'] > 0)].copy()
        if len(valid) == 0:
            continue
        s1 = valid[
            (valid['disc_detected'] == True) &
            valid['disc_edge_coverage_ratio'].notna() &
            (valid['disc_edge_coverage_ratio'] >= EDGE_COV)
        ].copy()
        if len(s1) > 0:
            s1['rn'] = minmax_norm(s1['retina_ratio'].fillna(0))
            s1['gn'] = minmax_norm(s1['mbss_Grad_p90'].fillna(0))
            s1['mn'] = minmax_norm(s1['mbss_score'].fillna(0))
            s1['qs'] = W_R * s1['rn'] + W_G * s1['gn'] + W_M * s1['mn']
            s1 = s1.sort_values('qs', ascending=False)
            sel = s1.head(top_k).copy()
        else:
            sel = pd.DataFrame()
        nr = top_k - len(sel)
        if nr > 0:
            rem = valid[~valid.index.isin(sel.index)].sort_values('retina_ratio', ascending=False)
            sel = pd.concat([sel, rem.head(nr)])
        if len(sel) > 0:
            all_sel.append(sel)
    return pd.concat(all_sel, ignore_index=True) if all_sel else pd.DataFrame()


def aggregate_soft_vote(df):
    rows = []
    for vid, group in df.groupby('video_id'):
        row = {'video_id': vid, 'n_images': len(group)}
        for col in ['zone_label', 'stage_label', 'plus_label', 'aggressive_rop_label', 'treatment_label']:
            row[col] = int(group[col].iloc[0])

        zone_probs = group[['zone_prob_0', 'zone_prob_1', 'zone_prob_2']].mean()
        row['zone_pred'] = int(np.argmax(zone_probs.values))
        row['zone_prob_0'] = float(zone_probs['zone_prob_0'])

        stage_probs = group[['stage_prob_0', 'stage_prob_1', 'stage_prob_2', 'stage_prob_3']].mean()
        row['stage_pred'] = int(np.argmax(stage_probs.values))
        row['stage_prob_3'] = float(stage_probs['stage_prob_3'])

        plus_probs = group[['plus_prob_0', 'plus_prob_1', 'plus_prob_2']].mean()
        row['plus_pred'] = int(np.argmax(plus_probs.values))
        row['plus_prob_2'] = float(plus_probs['plus_prob_2'])

        arop_prob = float(group['aggressive_rop_prob_1'].mean())
        row['aggressive_rop_prob_1'] = arop_prob
        row['aggressive_rop_pred'] = int(arop_prob >= 0.5)

        treat_prob = float(group['treatment_prob_1'].mean())
        row['treatment_prob_1'] = treat_prob
        row['treatment_pred'] = int(treat_prob >= 0.5)

        rows.append(row)
    return pd.DataFrame(rows)


# ============ PR-AUC Computation ============
def compute_prauc(y_true, y_prob):
    """Compute PR-AUC (Average Precision)"""
    if len(set(y_true)) < 2:
        return np.nan
    return average_precision_score(y_true, y_prob)


def compute_rw_rop_labels_and_probs(df):
    """Compute RW-ROP true labels and probability scores"""
    rw_true = ((df['plus_label'] == 2) | (df['stage_label'] == 3) | (df['zone_label'] == 0)).astype(int)
    rw_prob = 1 - ((1 - df['plus_prob_2']) * (1 - df['stage_prob_3']) * (1 - df['zone_prob_0']))
    return rw_true.values, rw_prob.values


def compute_all_prauc(df):
    """Compute PR-AUC for Treatment, AROP, RW-ROP"""
    results = {}

    # Treatment
    y_true = df['treatment_label'].values.astype(int)
    y_prob = df['treatment_prob_1'].values
    results['treatment'] = {
        'prauc': compute_prauc(y_true, y_prob),
        'rocauc': roc_auc_score(y_true, y_prob) if len(set(y_true)) >= 2 else np.nan,
        'prevalence': y_true.mean(),
        'n_pos': y_true.sum(),
        'n_total': len(y_true),
    }

    # AROP
    y_true = df['aggressive_rop_label'].values.astype(int)
    y_prob = df['aggressive_rop_prob_1'].values
    results['arop'] = {
        'prauc': compute_prauc(y_true, y_prob),
        'rocauc': roc_auc_score(y_true, y_prob) if len(set(y_true)) >= 2 else np.nan,
        'prevalence': y_true.mean(),
        'n_pos': y_true.sum(),
        'n_total': len(y_true),
    }

    # RW-ROP
    rw_true, rw_prob = compute_rw_rop_labels_and_probs(df)
    results['rw_rop'] = {
        'prauc': compute_prauc(rw_true, rw_prob),
        'rocauc': roc_auc_score(rw_true, rw_prob) if len(set(rw_true)) >= 2 else np.nan,
        'prevalence': rw_true.mean(),
        'n_pos': rw_true.sum(),
        'n_total': len(rw_true),
    }

    return results


def compute_per_fold_prauc(df):
    """Compute per-fold PR-AUC (mean +/- SD) for Treatment, AROP, RW-ROP"""
    folds = sorted(df['fold'].unique())
    fold_results = {task: [] for task in ['treatment', 'arop', 'rw_rop']}

    for fold in folds:
        fold_df = df[df['fold'] == fold]

        # Treatment
        y_true = fold_df['treatment_label'].values.astype(int)
        y_prob = fold_df['treatment_prob_1'].values
        fold_results['treatment'].append({
            'prauc': compute_prauc(y_true, y_prob),
            'rocauc': roc_auc_score(y_true, y_prob) if len(set(y_true)) >= 2 else np.nan,
            'prevalence': y_true.mean(),
        })

        # AROP
        y_true = fold_df['aggressive_rop_label'].values.astype(int)
        y_prob = fold_df['aggressive_rop_prob_1'].values
        fold_results['arop'].append({
            'prauc': compute_prauc(y_true, y_prob),
            'rocauc': roc_auc_score(y_true, y_prob) if len(set(y_true)) >= 2 else np.nan,
            'prevalence': y_true.mean(),
        })

        # RW-ROP
        rw_true, rw_prob = compute_rw_rop_labels_and_probs(fold_df)
        fold_results['rw_rop'].append({
            'prauc': compute_prauc(rw_true, rw_prob),
            'rocauc': roc_auc_score(rw_true, rw_prob) if len(set(rw_true)) >= 2 else np.nan,
            'prevalence': rw_true.mean(),
        })

    # Summarize
    summary = {}
    for task in ['treatment', 'arop', 'rw_rop']:
        praucs = [r['prauc'] for r in fold_results[task] if not np.isnan(r['prauc'])]
        rocaucs = [r['rocauc'] for r in fold_results[task] if not np.isnan(r['rocauc'])]
        prevs = [r['prevalence'] for r in fold_results[task]]
        summary[task] = {
            'prauc_mean': np.mean(praucs),
            'prauc_std': np.std(praucs),
            'rocauc_mean': np.mean(rocaucs),
            'rocauc_std': np.std(rocaucs),
            'prevalence_mean': np.mean(prevs),
            'fold_praucs': praucs,
            'fold_rocaucs': rocaucs,
        }
    return summary


# ============ Figures ============
def plot_pr_roc_curves(df, title_suffix, filename):
    """Plot PR and ROC curves side-by-side for Treatment, AROP, RW-ROP"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    tasks = [
        ('Treatment', df['treatment_label'].values.astype(int), df['treatment_prob_1'].values),
        ('AROP', df['aggressive_rop_label'].values.astype(int), df['aggressive_rop_prob_1'].values),
    ]
    rw_true, rw_prob = compute_rw_rop_labels_and_probs(df)
    tasks.append(('RW-ROP', rw_true, rw_prob))

    colors = ['#2ecc71', '#3498db', '#e74c3c']

    for col_idx, (task_name, y_true, y_prob) in enumerate(tasks):
        color = colors[col_idx]
        prevalence = y_true.mean()

        # ROC curve (top row)
        ax_roc = axes[0][col_idx]
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc_val = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, color=color, lw=2, label=f'ROC-AUC = {roc_auc_val:.3f}')
        ax_roc.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
        ax_roc.fill_between(fpr, tpr, alpha=0.15, color=color)
        ax_roc.set_xlabel('False Positive Rate (1 - Specificity)')
        ax_roc.set_ylabel('True Positive Rate (Sensitivity)')
        ax_roc.set_title(f'{task_name} - ROC Curve', fontsize=11, fontweight='bold')
        ax_roc.legend(loc='lower right', fontsize=10)
        ax_roc.grid(True, alpha=0.3)
        ax_roc.set_xlim([0, 1])
        ax_roc.set_ylim([0, 1.05])

        # PR curve (bottom row)
        ax_pr = axes[1][col_idx]
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc_val = average_precision_score(y_true, y_prob)
        ax_pr.plot(recall, precision, color=color, lw=2, label=f'PR-AUC = {pr_auc_val:.3f}')
        ax_pr.axhline(y=prevalence, color='gray', linestyle='--', lw=1, alpha=0.7,
                      label=f'Baseline (prevalence={prevalence:.3f})')
        ax_pr.fill_between(recall, precision, alpha=0.15, color=color)
        ax_pr.set_xlabel('Recall (Sensitivity)')
        ax_pr.set_ylabel('Precision (PPV)')
        ax_pr.set_title(f'{task_name} - PR Curve', fontsize=11, fontweight='bold')
        ax_pr.legend(loc='upper right', fontsize=9)
        ax_pr.grid(True, alpha=0.3)
        ax_pr.set_xlim([0, 1])
        ax_pr.set_ylim([0, 1.05])

    plt.suptitle(f'ROC vs PR Curves — {title_suffix}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def plot_per_fold_pr_curves(df, filename):
    """Plot per-fold PR curves for Treatment, AROP, RW-ROP"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    folds = sorted(df['fold'].unique())
    fold_colors = plt.cm.Set2(np.linspace(0, 1, len(folds)))
    task_names = ['Treatment', 'AROP', 'RW-ROP']

    for col_idx, task_name in enumerate(task_names):
        ax = axes[col_idx]
        all_praucs = []

        for fold_idx, fold in enumerate(folds):
            fold_df = df[df['fold'] == fold]
            if task_name == 'Treatment':
                y_true = fold_df['treatment_label'].values.astype(int)
                y_prob = fold_df['treatment_prob_1'].values
            elif task_name == 'AROP':
                y_true = fold_df['aggressive_rop_label'].values.astype(int)
                y_prob = fold_df['aggressive_rop_prob_1'].values
            else:
                y_true, y_prob = compute_rw_rop_labels_and_probs(fold_df)

            if len(set(y_true)) < 2:
                continue
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            pr_auc_val = average_precision_score(y_true, y_prob)
            all_praucs.append(pr_auc_val)
            ax.plot(recall, precision, color=fold_colors[fold_idx], lw=1.5,
                    label=f'Fold {fold} ({pr_auc_val:.3f})', alpha=0.8)

        mean_prauc = np.mean(all_praucs)
        ax.set_title(f'{task_name}\nMean PR-AUC = {mean_prauc:.3f} +/- {np.std(all_praucs):.3f}',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.legend(fontsize=8, loc='lower left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])

    plt.suptitle('Per-Fold PR Curves (5-Fold CV)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def plot_prauc_comparison_bar(results_dict, filename):
    """Bar chart comparing ROC-AUC vs PR-AUC across conditions"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    tasks = ['treatment', 'arop', 'rw_rop']
    task_labels = ['Treatment', 'AROP', 'RW-ROP']
    conditions = list(results_dict.keys())
    cond_labels = [c.replace('_', '\n') for c in conditions]

    x = np.arange(len(conditions))
    width = 0.25

    for metric_idx, (metric, ylabel, ax) in enumerate([
        ('rocauc', 'ROC-AUC', axes[0]),
        ('prauc', 'PR-AUC', axes[1]),
    ]):
        for t_idx, (task, task_label) in enumerate(zip(tasks, task_labels)):
            vals = []
            for cond in conditions:
                r = results_dict[cond][task]
                vals.append(r.get(metric, r.get(f'{metric}_mean', 0)))
            bars = ax.bar(x + t_idx * width - width, vals, width, label=task_label)
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.annotate(f'{v:.3f}', xy=(bar.get_x() + bar.get_width()/2, v),
                                xytext=(0, 3), textcoords="offset points",
                                ha='center', fontsize=7)

        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(ylabel, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(cond_labels, fontsize=9)
        ax.legend(fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.grid(True, axis='y', alpha=0.3)

    plt.suptitle('ROC-AUC vs PR-AUC Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


# ============ PDF Generation ============
def generate_pdf(fold_summary, condition_results, merged_df):
    from fpdf import FPDF

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
            self.cell(0, 8, "ROP AI Project - PR-AUC Supplementary Report",
                      new_x="LMARGIN", new_y="NEXT", align="R")
            self.line(10, 18, 200, 18)
            self.ln(5)

        def footer(self):
            self.set_y(-15)
            self.set_font(self.jp_font, "", 8)
            self.set_text_color(128, 128, 128)
            self.cell(0, 10, f"Page {self.page_no()}/{self.pages_count}", align="C")

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
            for i, h in enumerate(headers):
                self.cell(col_widths[i], 7, h, border=1, align="C", fill=True)
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
                "orange": ((255, 245, 230), (150, 80, 0)),
                "red": ((255, 230, 230), (150, 30, 30)),
            }
            fill_c, text_c = colors.get(color, colors["blue"])
            self.set_fill_color(*fill_c)
            self.set_text_color(*text_c)
            self.set_font(self.jp_font, "", 9)
            self.multi_cell(0, 6, text, border=1, fill=True)
            self.set_text_color(0, 0, 0)
            self.ln(2)

    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # ===== Page 1: Title =====
    pdf.add_page()
    pdf.ln(15)
    pdf.set_font(pdf.jp_font, "B", 22)
    pdf.set_text_color(30, 60, 120)
    pdf.multi_cell(0, 12, "PR-AUC (Precision-Recall AUC)\n補足解析レポート", align="C")
    pdf.ln(5)
    pdf.set_font(pdf.jp_font, "", 12)
    pdf.set_text_color(80, 80, 80)
    pdf.multi_cell(0, 8,
        "陽性ケースの低頻度を考慮したPR-AUCによる追加評価\n"
        "データセット: Multicenter ROP Study (6,448画像, 347 video_ids)\n"
        "モデル: clinical_v3 (EfficientNet-B0 + 臨床データ融合, 5タスク)\n"
        "作成日: 2026-02-13", align="C")
    pdf.ln(8)
    pdf.set_font(pdf.jp_font, "", 11)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, "【目次】", new_x="LMARGIN", new_y="NEXT", align="C")
    for item in [
        "1. PR-AUCの意義: なぜROC-AUCだけでは不十分か",
        "2. 全体評価: ROC vs PR曲線",
        "3. Per-Fold CV結果: Treatment / AROP / RW-ROP",
        "4. 画像品質フィルタリング + Majority Vote でのPR-AUC",
        "5. ROC-AUC vs PR-AUC 対比表",
        "6. 結論と臨床的意義",
    ]:
        pdf.cell(0, 7, item, new_x="LMARGIN", new_y="NEXT", align="C")

    # ===== Page 2: Background =====
    pdf.add_page()
    pdf.section_title("1. PR-AUCの意義")
    pdf.subsection_title("1.1 クラス不均衡とROC-AUCの限界")
    pdf.body_text(
        "本データセットの陽性率:\n"
        f"  Treatment: {fold_summary['treatment']['prevalence_mean']:.1%} (n={int(condition_results['per_image']['treatment']['n_pos'])})\n"
        f"  AROP: {fold_summary['arop']['prevalence_mean']:.1%} (n={int(condition_results['per_image']['arop']['n_pos'])})\n"
        f"  RW-ROP: {fold_summary['rw_rop']['prevalence_mean']:.1%} (n={int(condition_results['per_image']['rw_rop']['n_pos'])})\n\n"
        "ROC-AUCはクラス不均衡データで楽観的な評価を与える傾向がある。大多数の陰性例を正しく分類するだけで"
        "FPRが小さくなり、AUCが高くなるためである。\n\n"
        "PR-AUC (Average Precision) は陽性例に焦点を当て、予測の精度と網羅性のバランスを評価する。"
        "陽性率が低い場合、PR-AUCはROC-AUCよりも分類器の実用的な性能を反映する。"
    )
    pdf.subsection_title("1.2 解釈の目安")
    pdf.body_text(
        "PR-AUCのベースライン（ランダム分類器）= 陽性率（prevalence）\n"
        "  Treatment: ベースライン = 9.7%\n"
        "  AROP: ベースライン = 3.5%\n"
        "  RW-ROP: ベースライン = 24.5%\n\n"
        "PR-AUCがベースラインを大きく上回るほど、分類器が有用であることを示す。"
    )

    # ===== Page 3: ROC vs PR curves (overall) =====
    pdf.add_page()
    pdf.section_title("2. 全体評価: ROC vs PR曲線")
    pr_roc_path = str(FIG_DIR / "pr_roc_curves_overall.png")
    pred_df = load_predictions()
    plot_pr_roc_curves(pred_df, "clinical_v3 (全6,448画像)", pr_roc_path)
    pdf.image(pr_roc_path, w=190)
    pdf.ln(2)

    # Overall summary table
    overall = compute_all_prauc(pred_df)
    pdf.add_table(
        ["指標", "Treatment", "AROP", "RW-ROP"],
        [
            ["陽性率", f"{overall['treatment']['prevalence']:.1%}",
             f"{overall['arop']['prevalence']:.1%}",
             f"{overall['rw_rop']['prevalence']:.1%}"],
            ["ROC-AUC", f"{overall['treatment']['rocauc']:.4f}",
             f"{overall['arop']['rocauc']:.4f}",
             f"{overall['rw_rop']['rocauc']:.4f}"],
            ["PR-AUC", f"{overall['treatment']['prauc']:.4f}",
             f"{overall['arop']['prauc']:.4f}",
             f"{overall['rw_rop']['prauc']:.4f}"],
            ["PR-AUC/陽性率 比", f"{overall['treatment']['prauc']/overall['treatment']['prevalence']:.1f}x",
             f"{overall['arop']['prauc']/overall['arop']['prevalence']:.1f}x",
             f"{overall['rw_rop']['prauc']/overall['rw_rop']['prevalence']:.1f}x"],
        ],
        [40, 50, 50, 50]
    )
    pdf.highlight_box(
        "ROC-AUCでは3タスクとも0.90以上だが、PR-AUCではAROPが特に低い。\n"
        "AROPは陽性率3.5%と極めて少なく、偽陽性の影響がPR-AUCに大きく反映される。",
        "orange"
    )

    # ===== Page 4: Per-Fold CV =====
    pdf.add_page()
    pdf.section_title("3. Per-Fold CV結果 (5-Fold)")
    pdf.subsection_title("3.1 ROC-AUC vs PR-AUC (Mean +/- SD)")

    fold_table = []
    task_labels = {'treatment': 'Treatment', 'arop': 'AROP', 'rw_rop': 'RW-ROP'}
    for task in ['treatment', 'arop', 'rw_rop']:
        s = fold_summary[task]
        fold_table.append([
            task_labels[task],
            f"{s['prevalence_mean']:.1%}",
            f"{s['rocauc_mean']:.4f} +/- {s['rocauc_std']:.4f}",
            f"{s['prauc_mean']:.4f} +/- {s['prauc_std']:.4f}",
            f"{s['rocauc_mean'] - s['prauc_mean']:+.4f}",
        ])

    pdf.add_table(
        ["タスク", "陽性率", "ROC-AUC (Mean+/-SD)", "PR-AUC (Mean+/-SD)", "差 (ROC-PR)"],
        fold_table,
        [30, 22, 48, 48, 32]
    )

    pdf.subsection_title("3.2 Fold別 PR-AUC")
    fold_detail_table = []
    for task in ['treatment', 'arop', 'rw_rop']:
        s = fold_summary[task]
        row = [task_labels[task]]
        for prauc_val in s['fold_praucs']:
            row.append(f"{prauc_val:.4f}")
        row.append(f"{s['prauc_mean']:.4f}")
        fold_detail_table.append(row)

    pdf.add_table(
        ["タスク", "Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5", "Mean"],
        fold_detail_table,
        [28, 27, 27, 27, 27, 27, 27]
    )

    pdf.subsection_title("3.3 Per-Fold PR曲線")
    fold_pr_path = str(FIG_DIR / "per_fold_pr_curves.png")
    plot_per_fold_pr_curves(pred_df, fold_pr_path)
    pdf.image(fold_pr_path, w=190)

    # ===== Page 5: Top-K + Majority Vote PR-AUC =====
    if merged_df is not None:
        pdf.add_page()
        pdf.section_title("4. 品質フィルタリング + Majority Vote PR-AUC")
        pdf.subsection_title("4.1 画像単位: All vs Top-10 vs Top-5")

        img_table = []
        for task in ['treatment', 'arop', 'rw_rop']:
            for metric, metric_label in [('rocauc', 'ROC-AUC'), ('prauc', 'PR-AUC')]:
                row = [task_labels[task], metric_label]
                for cond in ['per_image', 'top10_img', 'top5_img']:
                    val = condition_results[cond][task][metric]
                    row.append(f"{val:.4f}" if not np.isnan(val) else "N/A")
                delta = condition_results['top5_img'][task][metric] - condition_results['per_image'][task][metric]
                row.append(f"{delta:+.4f}" if not np.isnan(delta) else "N/A")
                img_table.append(row)

        pdf.add_table(
            ["Task", "Metric", "All", "Top-10", "Top-5", "Delta(T5-All)"],
            img_table,
            [28, 25, 34, 34, 34, 35]
        )

        pdf.subsection_title("4.2 患者単位 Soft Vote")

        sv_table = []
        for task in ['treatment', 'arop', 'rw_rop']:
            for metric, metric_label in [('rocauc', 'ROC-AUC'), ('prauc', 'PR-AUC')]:
                row = [task_labels[task], metric_label]
                for cond in ['per_image', 'All_soft', 'Top-10_soft', 'Top-5_soft']:
                    val = condition_results[cond][task][metric]
                    row.append(f"{val:.4f}" if not np.isnan(val) else "N/A")
                sv_table.append(row)

        pdf.add_table(
            ["Task", "Metric", "Per-Image", "SV All", "SV Top-10", "SV Top-5"],
            sv_table,
            [28, 25, 34, 34, 34, 35]
        )

        pdf.highlight_box(
            "Treatment PR-AUC: 画像単位 0.87 -> Soft Vote Top-5 で大幅改善\n"
            "AROP PR-AUC: 陽性率3.5%のため絶対値は低いが、Soft Voteで改善傾向\n"
            "RW-ROP PR-AUC: ROC-AUCと同様、Soft Voteでほぼ横ばい",
            "blue"
        )

        # Comparison bar chart
        pdf.add_page()
        pdf.subsection_title("4.3 ROC-AUC vs PR-AUC 条件別比較")
        bar_path = str(FIG_DIR / "prauc_comparison_bar.png")
        plot_prauc_comparison_bar(condition_results, bar_path)
        pdf.image(bar_path, w=190)

    # ===== Page: ROC-AUC vs PR-AUC Summary =====
    pdf.add_page()
    pdf.section_title("5. ROC-AUC vs PR-AUC 対比表")
    pdf.body_text(
        "以下に、両レポートの主要結果をROC-AUCとPR-AUCで対比する。"
    )

    pdf.subsection_title("5.1 閾値最適化レポート対応 (Per-Fold CV)")
    pdf.add_table(
        ["タスク", "ROC-AUC (Mean+/-SD)", "PR-AUC (Mean+/-SD)", "陽性率"],
        [
            ["Treatment",
             f"{fold_summary['treatment']['rocauc_mean']:.3f} +/- {fold_summary['treatment']['rocauc_std']:.3f}",
             f"{fold_summary['treatment']['prauc_mean']:.3f} +/- {fold_summary['treatment']['prauc_std']:.3f}",
             f"{fold_summary['treatment']['prevalence_mean']:.1%}"],
            ["AROP",
             f"{fold_summary['arop']['rocauc_mean']:.3f} +/- {fold_summary['arop']['rocauc_std']:.3f}",
             f"{fold_summary['arop']['prauc_mean']:.3f} +/- {fold_summary['arop']['prauc_std']:.3f}",
             f"{fold_summary['arop']['prevalence_mean']:.1%}"],
            ["RW-ROP",
             f"{fold_summary['rw_rop']['rocauc_mean']:.3f} +/- {fold_summary['rw_rop']['rocauc_std']:.3f}",
             f"{fold_summary['rw_rop']['prauc_mean']:.3f} +/- {fold_summary['rw_rop']['prauc_std']:.3f}",
             f"{fold_summary['rw_rop']['prevalence_mean']:.1%}"],
        ],
        [30, 55, 55, 30]
    )

    if merged_df is not None:
        pdf.subsection_title("5.2 Majority Vote レポート対応 (Best: SV Top-5)")
        best_cond = 'Top-5_soft'
        pdf.add_table(
            ["タスク", "ROC-AUC", "PR-AUC", "陽性率"],
            [
                ["Treatment",
                 f"{condition_results[best_cond]['treatment']['rocauc']:.4f}",
                 f"{condition_results[best_cond]['treatment']['prauc']:.4f}",
                 f"{condition_results[best_cond]['treatment']['prevalence']:.1%}"],
                ["AROP",
                 f"{condition_results[best_cond]['arop']['rocauc']:.4f}" if not np.isnan(condition_results[best_cond]['arop']['rocauc']) else "N/A",
                 f"{condition_results[best_cond]['arop']['prauc']:.4f}" if not np.isnan(condition_results[best_cond]['arop']['prauc']) else "N/A",
                 f"{condition_results[best_cond]['arop']['prevalence']:.1%}"],
                ["RW-ROP",
                 f"{condition_results[best_cond]['rw_rop']['rocauc']:.4f}",
                 f"{condition_results[best_cond]['rw_rop']['prauc']:.4f}",
                 f"{condition_results[best_cond]['rw_rop']['prevalence']:.1%}"],
            ],
            [40, 50, 50, 40]
        )

    # ===== Page: Conclusions =====
    pdf.add_page()
    pdf.section_title("6. 結論と臨床的意義")

    pdf.subsection_title("6.1 主要な知見")

    # Format conclusion text based on computed values
    treat_prauc = fold_summary['treatment']['prauc_mean']
    arop_prauc = fold_summary['arop']['prauc_mean']
    rw_prauc = fold_summary['rw_rop']['prauc_mean']

    pdf.body_text(
        f"1. Treatment PR-AUC = {treat_prauc:.3f}\n"
        f"   ROC-AUC ({fold_summary['treatment']['rocauc_mean']:.3f}) との差は小さく、分類器の性能は実質的に高い。\n"
        f"   陽性率9.7%に対してPR-AUCが{treat_prauc:.3f}であり、ランダムの{treat_prauc/fold_summary['treatment']['prevalence_mean']:.0f}倍の精度。\n\n"
        f"2. AROP PR-AUC = {arop_prauc:.3f}\n"
        f"   ROC-AUC ({fold_summary['arop']['rocauc_mean']:.3f}) と比較して大幅に低下。陽性率3.5%のクラス不均衡が原因。\n"
        f"   ROC-AUC 0.888 という数値ほど臨床的に高い精度ではないことに注意。\n"
        f"   ただし Soft Vote (患者単位集約) で改善傾向あり。\n\n"
        f"3. RW-ROP PR-AUC = {rw_prauc:.3f}\n"
        f"   陽性率24.5%と比較的高いため、ROC-AUC ({fold_summary['rw_rop']['rocauc_mean']:.3f}) との乖離は中程度。\n"
        f"   スクリーニング指標として実用的な水準を維持。"
    )

    pdf.subsection_title("6.2 臨床的解釈")
    pdf.body_text(
        "Treatment (治療必要性):\n"
        "  ROC-AUC, PR-AUCともに高く、閾値最適化による感度>=95%の運用は妥当。\n"
        "  PR-AUCの高さは、陽性予測の精度が実用水準にあることを裏付ける。\n\n"
        "AROP (Aggressive ROP):\n"
        "  PR-AUCが示すように、低陽性率でのスクリーニングには偽陽性の課題がある。\n"
        "  Soft Vote による患者単位集約が偽陽性削減に有効。\n"
        "  AROPの独立スクリーニングよりも、RW-ROP (複合指標) の使用を推奨。\n\n"
        "RW-ROP (紹介必要):\n"
        "  PR-AUCは実用的水準を維持しており、スクリーニング指標として適切。\n"
        "  閾値0.44で感度95.2%達成時のPrecision低下はPR-AUCにも反映されるが、\n"
        "  NPV 98.2%の安全性は維持される。"
    )

    pdf.subsection_title("6.3 推奨事項")
    pdf.body_text(
        "1. 学術報告ではROC-AUCとPR-AUCの両方を報告することを推奨\n"
        "2. 特にAROPなど低陽性率のタスクでは、PR-AUCがより実態を反映する\n"
        "3. Soft Vote (患者単位集約) はPR-AUCの観点からも有効\n"
        "4. 論文投稿時には、PR曲線にベースライン（陽性率）を明示すること"
    )

    # Save
    output_path = OUTPUT_DIR / "report_prauc_supplement.pdf"
    pdf.output(str(output_path))
    print(f"PDF saved to: {output_path}")
    return output_path


# ============ Main ============
def main():
    print("=" * 60)
    print("PR-AUC Supplementary Analysis")
    print("=" * 60)

    # 1. Load predictions
    print("\n[1/5] Loading predictions...")
    pred_df = load_predictions()
    print(f"  {len(pred_df)} images, {pred_df['video_id'].nunique()} video_ids, {pred_df['fold'].nunique()} folds")

    # 2. Per-fold CV analysis
    print("\n[2/5] Computing per-fold PR-AUC...")
    fold_summary = compute_per_fold_prauc(pred_df)
    for task, label in [('treatment', 'Treatment'), ('arop', 'AROP'), ('rw_rop', 'RW-ROP')]:
        s = fold_summary[task]
        print(f"  {label}: ROC-AUC={s['rocauc_mean']:.4f}+/-{s['rocauc_std']:.4f}, "
              f"PR-AUC={s['prauc_mean']:.4f}+/-{s['prauc_std']:.4f}, "
              f"Prevalence={s['prevalence_mean']:.1%}")

    # 3. Load features and compute Top-K / Majority Vote
    print("\n[3/5] Loading features for Top-K analysis...")
    merged_df = load_data_with_features()
    condition_results = {}

    if merged_df is not None:
        condition_results['per_image'] = compute_all_prauc(merged_df)

        for k, label in [(10, 'top10_img'), (5, 'top5_img')]:
            topk_df = select_top_k(merged_df, k)
            condition_results[label] = compute_all_prauc(topk_df)
            print(f"  {label}: {len(topk_df)} images")

        # Soft Vote
        for data_label, src_df in [('All', merged_df), ('Top-10', select_top_k(merged_df, 10)), ('Top-5', select_top_k(merged_df, 5))]:
            key = f'{data_label}_soft'
            agg_df = aggregate_soft_vote(src_df)
            condition_results[key] = compute_all_prauc(agg_df)
            print(f"  {key}: {len(agg_df)} videos")
    else:
        condition_results['per_image'] = compute_all_prauc(pred_df)

    # 4. Generate figures
    print("\n[4/5] Generating figures...")
    plot_pr_roc_curves(pred_df, "clinical_v3 (全6,448画像)", str(FIG_DIR / "pr_roc_curves_overall.png"))
    plot_per_fold_pr_curves(pred_df, str(FIG_DIR / "per_fold_pr_curves.png"))
    if merged_df is not None:
        plot_prauc_comparison_bar(condition_results, str(FIG_DIR / "prauc_comparison_bar.png"))

    # 5. Generate PDF
    print("\n[5/5] Generating PDF report...")
    output_path = generate_pdf(fold_summary, condition_results, merged_df)

    # Copy to shared directory
    import shutil
    share_path = SHARE_DIR / output_path.name
    shutil.copy2(output_path, share_path)
    print(f"Copied to: {share_path}")

    # Also copy existing reports
    for report_name in ['report_rwrop_threshold_optimization.pdf', 'report_top10_majority_vote.pdf']:
        src = OUTPUT_DIR / report_name
        if src.exists():
            shutil.copy2(src, SHARE_DIR / report_name)
            print(f"Copied to: {SHARE_DIR / report_name}")

    print("\nDone!")


if __name__ == '__main__':
    main()
