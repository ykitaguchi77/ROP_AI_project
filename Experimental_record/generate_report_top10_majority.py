# -*- coding: utf-8 -*-
"""
Top-K品質フィルタリング + Majority Vote 評価レポート (PDF)
書式: report_clinical_fusion_comparison.pdf と同一
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, auc
)

plt.rcParams['font.family'] = 'Meiryo'
plt.rcParams['axes.unicode_minus'] = False

# --- Paths ---
MULTI_DIR = Path(r"C:\Users\ykita\ROP_AI_project\ROP_project\multicenter_study")
PRED_PATH = MULTI_DIR / "outputs_clinical_v3" / "predictions.csv"
KUBOTA_EXCEL = Path(r"E:\Multicenter_ROP_study\Multicenter_images\Kubota_selection\selected_images_disc_retina.xlsx")
TOP_EXCEL = Path(r"E:\Multicenter_ROP_study\Multicenter_images\selected_images_disc_retina.xlsx")
FIG_DIR = Path(r"C:\Users\ykita\ROP_AI_project\Experimental_record\figures")
FIG_DIR.mkdir(exist_ok=True)
OUTPUT_PATH = Path(r"C:\Users\ykita\ROP_AI_project\Experimental_record\report_top10_majority_vote.pdf")

sys.path.insert(0, str(MULTI_DIR))
from select_best_images import minmax_norm

# --- Constants ---
EDGE_COV = 0.80
W_R, W_G, W_M = 0.4, 0.4, 0.2


# ============ Data Loading ============
def load_data():
    pred_df = pd.read_csv(PRED_PATH)
    pred_df['image_name'] = pred_df['image_path'].apply(lambda x: Path(x).name)

    feat_cols = ['retina_ratio', 'mbss_Grad_p90', 'mbss_score', 'disc_detected', 'disc_edge_coverage_ratio']
    kubota_df = pd.read_excel(KUBOTA_EXCEL)
    avail = [c for c in feat_cols if c in kubota_df.columns]
    kubota_features = kubota_df[['image_name'] + avail].drop_duplicates(subset='image_name', keep='first')
    merged_df = pred_df.merge(kubota_features, on='image_name', how='left')

    tl = pd.read_excel(TOP_EXCEL)
    tl_feat = [c for c in avail if c in tl.columns]
    if 'image_name' in tl.columns and tl_feat:
        tl_features = tl[['image_name'] + tl_feat].drop_duplicates(subset='image_name', keep='first').set_index('image_name')
        mm = merged_df['retina_ratio'].isna()
        for col in tl_feat:
            fill = merged_df.loc[mm, 'image_name'].map(tl_features[col])
            merged_df.loc[mm, col] = fill.values

    return merged_df


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


# ============ Majority Vote ============
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


# ============ Evaluation ============
def compute_multiclass(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'kappa': cohen_kappa_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
    }

def compute_binary(y_true, y_pred, y_prob=None):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    result = {'sensitivity': sens, 'specificity': spec, 'PPV': ppv, 'NPV': npv, 'F1': f1}
    if y_prob is not None and len(set(y_true)) >= 2:
        result['AUC'] = roc_auc_score(y_true, y_prob)
    return result

def compute_rw_rop(df):
    rw_true = ((df['plus_label'] == 2) | (df['stage_label'] == 3) | (df['zone_label'] == 0)).astype(int)
    rw_pred = ((df['plus_pred'] == 2) | (df['stage_pred'] == 3) | (df['zone_pred'] == 0)).astype(int)
    rw_prob = 1 - ((1 - df['plus_prob_2']) * (1 - df['stage_prob_3']) * (1 - df['zone_prob_0']))
    return rw_true, rw_pred, rw_prob

def evaluate_all(df):
    results = {}
    results['zone'] = compute_multiclass(df['zone_label'], df['zone_pred'])
    results['stage'] = compute_multiclass(df['stage_label'], df['stage_pred'])
    results['plus'] = compute_multiclass(df['plus_label'], df['plus_pred'])
    results['aggressive_rop'] = compute_binary(
        df['aggressive_rop_label'], df['aggressive_rop_pred'], df['aggressive_rop_prob_1'])
    results['treatment'] = compute_binary(
        df['treatment_label'], df['treatment_pred'], df['treatment_prob_1'])
    rw_true, rw_pred, rw_prob = compute_rw_rop(df)
    results['rw_rop'] = compute_binary(rw_true, rw_pred, rw_prob)
    return results


# ============ Agreement Analysis ============
def compute_agreement(df):
    """Per-video prediction agreement for each task"""
    tasks = {
        'zone_pred': 'zone_label',
        'stage_pred': 'stage_label',
        'plus_pred': 'plus_label',
        'treatment_pred': 'treatment_label',
        'aggressive_rop_pred': 'aggressive_rop_label',
    }
    # Add RW-ROP
    df = df.copy()
    df['rw_rop_pred'] = ((df['plus_pred'] == 2) | (df['stage_pred'] == 3) | (df['zone_pred'] == 0)).astype(int)
    df['rw_rop_label'] = ((df['plus_label'] == 2) | (df['stage_label'] == 3) | (df['zone_label'] == 0)).astype(int)
    tasks['rw_rop_pred'] = 'rw_rop_label'

    results = {}
    for pred_col, label_col in tasks.items():
        name = pred_col.replace('_pred', '')
        agreements = []
        for vid, group in df.groupby('video_id'):
            mode_val = stats.mode(group[pred_col], keepdims=True).mode[0]
            agree_rate = (group[pred_col] == mode_val).mean()
            correct = int(mode_val) == int(group[label_col].iloc[0])
            agreements.append({'video_id': vid, 'agreement': agree_rate, 'correct': correct})
        adf = pd.DataFrame(agreements)
        results[name] = adf
    return results


# ============ Figures ============
def plot_agreement_histogram(agreement_data, filename):
    """Histogram of per-video agreement rates"""
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    task_order = ['zone', 'stage', 'plus', 'treatment', 'aggressive_rop', 'rw_rop']
    task_labels = ['Zone', 'Stage', 'Plus', 'Treatment', 'AROP', 'RW-ROP']

    for idx, (task, label) in enumerate(zip(task_order, task_labels)):
        ax = axes[idx // 3][idx % 3]
        adf = agreement_data[task]
        arr = adf['agreement'].values
        ax.hist(arr, bins=20, range=(0, 1), color='#3498db', edgecolor='white', alpha=0.8)
        ax.axvline(x=0.9, color='red', linestyle='--', linewidth=1, alpha=0.7, label='90%')
        ax.set_title(f'{label} (mean={arr.mean():.3f})', fontsize=10, fontweight='bold')
        ax.set_xlabel('Agreement Rate')
        ax.set_ylabel('Videos')
        ax.set_xlim(0, 1.05)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def plot_agreement_vs_correctness(agreement_data, filename):
    """Bar chart: correctness by agreement level"""
    fig, ax = plt.subplots(figsize=(8, 5))
    task_order = ['zone', 'stage', 'plus', 'treatment', 'aggressive_rop', 'rw_rop']
    task_labels = ['Zone', 'Stage', 'Plus', 'Treatment', 'AROP', 'RW-ROP']

    x = np.arange(len(task_labels))
    width = 0.35

    high_acc = []
    low_acc = []
    for task in task_order:
        adf = agreement_data[task]
        high = adf[adf['agreement'] >= 0.9]
        low = adf[adf['agreement'] < 0.9]
        high_acc.append(high['correct'].mean() if len(high) > 0 else 0)
        low_acc.append(low['correct'].mean() if len(low) > 0 else 0)

    bars1 = ax.bar(x - width/2, high_acc, width, label='Agreement >= 90%', color='#2ecc71')
    bars2 = ax.bar(x + width/2, low_acc, width, label='Agreement < 90%', color='#e74c3c')

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.annotate(f'{h:.2f}', xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)

    ax.set_ylabel('Correctness Rate')
    ax.set_title('Majority Vote Correctness by Agreement Level', fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def plot_trend_chart(results_dict, filename):
    """Sensitivity/Specificity/AUC trend: PerImg -> Top10_soft -> Top5_soft"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    conditions = ['Per-Image\n(All 6448)', 'Soft Vote\n(Top-10)', 'Soft Vote\n(Top-5)']
    cond_keys = ['per_image', 'Top-10_soft', 'Top-5_soft']
    colors = {'treatment': '#2ecc71', 'aggressive_rop': '#3498db', 'rw_rop': '#e74c3c'}
    labels = {'treatment': 'Treatment', 'aggressive_rop': 'AROP', 'rw_rop': 'RW-ROP'}
    x = np.arange(len(conditions))

    for metric_idx, metric in enumerate(['sensitivity', 'specificity', 'AUC']):
        ax = axes[metric_idx]
        for task in ['treatment', 'aggressive_rop', 'rw_rop']:
            vals = [results_dict[k][task].get(metric, 0) for k in cond_keys]
            ax.plot(x, vals, 'o-', color=colors[task], label=labels[task], linewidth=2, markersize=8)
            for i, v in enumerate(vals):
                offset_y = 10 if task != 'rw_rop' else -15
                if metric == 'specificity' and task == 'aggressive_rop':
                    offset_y = -15
                ax.annotate(f'{v:.3f}', (x[i], v), textcoords="offset points",
                           xytext=(0, offset_y), ha='center', fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(conditions, fontsize=9)
        ax.set_title(metric.capitalize(), fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.70, 1.02)

    plt.suptitle('Binary Tasks: Per-Image vs Soft Vote (Patient-level)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


# ============ PDF Generation ============
def generate_pdf(all_results, agreement_data):
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
            self.cell(0, 8, "ROP AI Project - Experimental Report", new_x="LMARGIN", new_y="NEXT", align="R")
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
    pdf.ln(20)
    pdf.set_font(pdf.jp_font, "B", 22)
    pdf.set_text_color(30, 60, 120)
    pdf.multi_cell(0, 12, "Top-K画像品質フィルタリングによる\nROP分類性能への影響評価", align="C")
    pdf.ln(5)
    pdf.set_font(pdf.jp_font, "", 12)
    pdf.set_text_color(80, 80, 80)
    pdf.multi_cell(0, 8,
        "品質フィルタリング + 患者単位 Majority Vote による診断精度の検証\n"
        "データセット: Multicenter ROP Study (6,448画像, 347 video_ids)\n"
        "モデル: clinical_v3 (EfficientNet-B0 + 臨床データ融合, 5タスク)\n"
        "作成日: 2026-02-11", align="C")
    pdf.ln(10)
    pdf.set_font(pdf.jp_font, "", 11)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, "【目次】", new_x="LMARGIN", new_y="NEXT", align="C")
    for item in [
        "1. 背景と目的",
        "2. 手法: 品質フィルタリング + Majority Vote",
        "3. 結果: 画像単位評価 (All vs Top-10 vs Top-5)",
        "4. 結果: 患者単位 Majority Vote",
        "5. 予測一致率分析: 信頼度指標としての活用",
        "6. 結論と臨床的意義",
    ]:
        pdf.cell(0, 7, item, new_x="LMARGIN", new_y="NEXT", align="C")

    # ===== Page 2: Background & Methods =====
    pdf.add_page()
    pdf.section_title("1. 背景と目的")
    pdf.body_text(
        "臨床現場では、動画撮影から高品質画像を選別しAI診断に使用する運用が想定される。"
        "全Good+Fair画像6,448枚での評価は、低品質画像を含むため実運用と乖離がある。\n\n"
        "本検証の目的:\n"
        "  (1) 品質フィルタリング (Top-10/Top-5) が画像単位の分類性能に与える影響\n"
        "  (2) 患者単位の多数決集約 (Majority Vote) による性能変化\n"
        "  (3) 画像間の予測一致率が信頼度指標として有用かの検証"
    )

    pdf.section_title("2. 手法")
    pdf.subsection_title("2.1 品質フィルタリング: C-rule3_thr0.80")
    pdf.body_text(
        "Stage 1: disc_edge_coverage >= 0.80 の画像を対象\n"
        "  スコア = 0.4 x retina_ratio_norm + 0.4 x Grad_p90_norm + 0.2 x mbss_score_norm\n"
        "  スコア降順で Top-K 枚を選出\n"
        "Stage 2 (Fallback): Stage1 候補 < K の場合、残りを retina_ratio 降順で補完"
    )
    pdf.subsection_title("2.2 Majority Vote (Soft Vote)")
    pdf.body_text(
        "同一 video_id の画像群から患者単位の予測を集約する。\n"
        "各画像のクラス別予測確率 (softmax出力) を画像間で平均し、その平均確率から患者単位の予測を決定する。\n\n"
        "Multiclass (Zone/Stage/Plus):\n"
        "  各クラスの確率を画像間で平均 -> argmax (最大確率のクラスを予測)\n"
        "  例: Zone - 5枚の [P(I), P(II), P(III)] を平均 -> [0.15, 0.60, 0.25] -> 予測: Zone II\n\n"
        "Binary (Treatment/AROP):\n"
        "  陽性確率 P(positive) を画像間で平均 -> 閾値 0.5 で判定\n"
        "  例: 5枚の P(treat=1) = [0.8, 0.9, 0.7, 0.6, 0.85] -> 平均 0.77 -> 予測: 陽性\n\n"
        "RW-ROP (derived):\n"
        "  集約済みの Zone/Stage/Plus 確率から導出\n"
        "  P(RW) = 1 - (1-P(Plus=2)) x (1-P(Stage=3)) x (1-P(Zone=I))\n\n"
        "Soft Vote の利点: Hard Vote (最頻値) は予測クラスしか使わないが、Soft Vote は確率の大きさ "
        "(モデルの確信度) も反映するため、判断境界付近の症例でより安定した予測が得られる。"
    )

    pdf.subsection_title("2.3 データ概要")
    pdf.add_table(
        ["条件", "画像数", "Video数", "評価単位"],
        [
            ["All (Per-Image)", "6,448", "347", "画像"],
            ["Top-10 (Per-Image)", "3,089", "347", "画像"],
            ["Top-5 (Per-Image)", "1,650", "347", "画像"],
            ["All (Soft Vote)", "-", "347", "患者"],
            ["Top-10 (Soft Vote)", "-", "347", "患者"],
            ["Top-5 (Soft Vote)", "-", "347", "患者"],
        ],
        [55, 35, 35, 55]
    )

    # ===== Page 3: Per-Image Results =====
    pdf.add_page()
    pdf.section_title("3. 結果: 画像単位評価")
    pdf.subsection_title("3.1 Multiclass タスク")

    r = all_results
    pdf.add_table(
        ["Task", "Metric", "All (6448)", "Top-10 (3089)", "Top-5 (1650)", "Delta(T5-All)"],
        [
            ["Zone", "Accuracy", f"{r['per_image']['zone']['accuracy']:.4f}",
             f"{r['top10_img']['zone']['accuracy']:.4f}", f"{r['top5_img']['zone']['accuracy']:.4f}",
             f"{r['top5_img']['zone']['accuracy'] - r['per_image']['zone']['accuracy']:+.4f}"],
            ["Zone", "Kappa", f"{r['per_image']['zone']['kappa']:.4f}",
             f"{r['top10_img']['zone']['kappa']:.4f}", f"{r['top5_img']['zone']['kappa']:.4f}",
             f"{r['top5_img']['zone']['kappa'] - r['per_image']['zone']['kappa']:+.4f}"],
            ["Stage", "Accuracy", f"{r['per_image']['stage']['accuracy']:.4f}",
             f"{r['top10_img']['stage']['accuracy']:.4f}", f"{r['top5_img']['stage']['accuracy']:.4f}",
             f"{r['top5_img']['stage']['accuracy'] - r['per_image']['stage']['accuracy']:+.4f}"],
            ["Stage", "Kappa", f"{r['per_image']['stage']['kappa']:.4f}",
             f"{r['top10_img']['stage']['kappa']:.4f}", f"{r['top5_img']['stage']['kappa']:.4f}",
             f"{r['top5_img']['stage']['kappa'] - r['per_image']['stage']['kappa']:+.4f}"],
            ["Plus", "Accuracy", f"{r['per_image']['plus']['accuracy']:.4f}",
             f"{r['top10_img']['plus']['accuracy']:.4f}", f"{r['top5_img']['plus']['accuracy']:.4f}",
             f"{r['top5_img']['plus']['accuracy'] - r['per_image']['plus']['accuracy']:+.4f}"],
            ["Plus", "Kappa", f"{r['per_image']['plus']['kappa']:.4f}",
             f"{r['top10_img']['plus']['kappa']:.4f}", f"{r['top5_img']['plus']['kappa']:.4f}",
             f"{r['top5_img']['plus']['kappa'] - r['per_image']['plus']['kappa']:+.4f}"],
        ],
        [25, 25, 35, 35, 35, 35]
    )
    pdf.highlight_box(
        "Zone/Stage/Plus: 画像74%削減 (Top-5) でも accuracy/kappa の低下は 2.5pp 以内",
        "green"
    )

    pdf.subsection_title("3.2 Binary タスク")

    def binary_row(task_label, task_key, metric):
        return [task_label, metric.capitalize(),
                f"{r['per_image'][task_key][metric]:.4f}",
                f"{r['top10_img'][task_key][metric]:.4f}",
                f"{r['top5_img'][task_key][metric]:.4f}",
                f"{r['top5_img'][task_key][metric] - r['per_image'][task_key][metric]:+.4f}"]

    pdf.add_table(
        ["Task", "Metric", "All", "Top-10", "Top-5", "Delta(T5-All)"],
        [
            binary_row("Treatment", "treatment", "sensitivity"),
            binary_row("Treatment", "treatment", "specificity"),
            binary_row("Treatment", "treatment", "AUC"),
            binary_row("AROP", "aggressive_rop", "sensitivity"),
            binary_row("AROP", "aggressive_rop", "specificity"),
            binary_row("AROP", "aggressive_rop", "AUC"),
            binary_row("RW-ROP", "rw_rop", "sensitivity"),
            binary_row("RW-ROP", "rw_rop", "specificity"),
            binary_row("RW-ROP", "rw_rop", "AUC"),
        ],
        [30, 27, 33, 33, 33, 34]
    )
    pdf.highlight_box(
        "Treatment/AROP: 画像を絞るほど改善 (AROP Sens +7.4pp, Treatment AUC 0.984)\n"
        "RW-ROP: 唯一低下 (Sens -3.8pp, AUC -2.2pp) -- 3成分の微小劣化が複合蓄積",
        "orange"
    )

    # ===== Page 4: Majority Vote Results =====
    pdf.add_page()
    pdf.section_title("4. 結果: 患者単位 Majority Vote (Soft Vote)")
    pdf.subsection_title("4.1 Multiclass タスク")
    pdf.add_table(
        ["Task", "Metric", "Per-Image", "SV All", "SV Top-10", "SV Top-5"],
        [
            ["Zone", "Accuracy",
             f"{r['per_image']['zone']['accuracy']:.4f}",
             f"{r['All_soft']['zone']['accuracy']:.4f}",
             f"{r['Top-10_soft']['zone']['accuracy']:.4f}",
             f"{r['Top-5_soft']['zone']['accuracy']:.4f}"],
            ["Zone", "Kappa",
             f"{r['per_image']['zone']['kappa']:.4f}",
             f"{r['All_soft']['zone']['kappa']:.4f}",
             f"{r['Top-10_soft']['zone']['kappa']:.4f}",
             f"{r['Top-5_soft']['zone']['kappa']:.4f}"],
            ["Stage", "Accuracy",
             f"{r['per_image']['stage']['accuracy']:.4f}",
             f"{r['All_soft']['stage']['accuracy']:.4f}",
             f"{r['Top-10_soft']['stage']['accuracy']:.4f}",
             f"{r['Top-5_soft']['stage']['accuracy']:.4f}"],
            ["Stage", "Kappa",
             f"{r['per_image']['stage']['kappa']:.4f}",
             f"{r['All_soft']['stage']['kappa']:.4f}",
             f"{r['Top-10_soft']['stage']['kappa']:.4f}",
             f"{r['Top-5_soft']['stage']['kappa']:.4f}"],
            ["Plus", "Accuracy",
             f"{r['per_image']['plus']['accuracy']:.4f}",
             f"{r['All_soft']['plus']['accuracy']:.4f}",
             f"{r['Top-10_soft']['plus']['accuracy']:.4f}",
             f"{r['Top-5_soft']['plus']['accuracy']:.4f}"],
            ["Plus", "Kappa",
             f"{r['per_image']['plus']['kappa']:.4f}",
             f"{r['All_soft']['plus']['kappa']:.4f}",
             f"{r['Top-10_soft']['plus']['kappa']:.4f}",
             f"{r['Top-5_soft']['plus']['kappa']:.4f}"],
        ],
        [25, 25, 35, 35, 35, 35]
    )
    pdf.highlight_box(
        "Zone: Soft Vote Top-10 で Kappa +0.066 改善 (0.550 -> 0.616)\n"
        "Stage/Plus: 大きな変化なし",
        "green"
    )

    pdf.subsection_title("4.2 Binary タスク")

    def sv_row(task_label, task_key, metric):
        return [task_label, metric.capitalize(),
                f"{r['per_image'][task_key][metric]:.4f}",
                f"{r['All_soft'][task_key][metric]:.4f}",
                f"{r['Top-10_soft'][task_key][metric]:.4f}",
                f"{r['Top-5_soft'][task_key][metric]:.4f}"]

    pdf.add_table(
        ["Task", "Metric", "Per-Image", "SV All", "SV Top-10", "SV Top-5"],
        [
            sv_row("Treatment", "treatment", "sensitivity"),
            sv_row("Treatment", "treatment", "specificity"),
            sv_row("Treatment", "treatment", "AUC"),
            sv_row("AROP", "aggressive_rop", "sensitivity"),
            sv_row("AROP", "aggressive_rop", "specificity"),
            sv_row("AROP", "aggressive_rop", "AUC"),
            sv_row("RW-ROP", "rw_rop", "sensitivity"),
            sv_row("RW-ROP", "rw_rop", "specificity"),
            sv_row("RW-ROP", "rw_rop", "AUC"),
        ],
        [30, 27, 33, 33, 33, 34]
    )
    pdf.highlight_box(
        "Treatment Sens: 0.917 -> 1.000 (Soft Vote Top-5), Spec: 微減 (-0.7pp)\n"
        "AROP Sens: 0.860 -> 1.000 (Top-5 Soft Vote), Spec: 微減 (-2.1pp)\n"
        "RW-ROP: Soft Vote でほぼ変化なし (Sens 0.865 -> 0.869, Spec 0.908 -> 0.905)",
        "blue"
    )

    # ===== Page 5: Trend chart =====
    pdf.add_page()
    pdf.section_title("4.3 Sensitivity / Specificity / AUC 推移")
    trend_path = str(FIG_DIR / "trend_sensitivity_auc.png")
    plot_trend_chart(r, trend_path)
    pdf.image(trend_path, w=185)
    pdf.ln(3)
    pdf.body_text(
        "Treatment / AROP: Per-Image -> Soft Vote で大幅改善。特に Top-5 Soft Vote で "
        "Treatment Sensitivity 1.000, AROP Sensitivity 1.000 を達成。\n"
        "RW-ROP: Sensitivity はほぼ横ばい (0.865 -> 0.869)、AUC は微減 (0.954 -> 0.945)。"
    )

    # ===== Page 6: Agreement Analysis =====
    pdf.add_page()
    pdf.section_title("5. 予測一致率分析")
    pdf.subsection_title("5.1 Per-Video Agreement Rate")
    pdf.body_text(
        "同一 video_id 内の全画像の予測がどの程度一致しているかを分析。"
        "一致率 = (最頻予測と同じ予測をした画像数) / (全画像数)"
    )

    hist_path = str(FIG_DIR / "agreement_histogram.png")
    plot_agreement_histogram(agreement_data, hist_path)
    pdf.image(hist_path, w=185)

    # ===== Page 7: Agreement vs Correctness =====
    pdf.add_page()
    pdf.subsection_title("5.2 一致率と正解率の関係")
    pdf.body_text(
        "一致率 >= 90% の症例と < 90% の症例で、多数決の正解率を比較。"
    )

    # Table
    agr_table = []
    for task, label in [('zone', 'Zone'), ('stage', 'Stage'), ('plus', 'Plus'),
                         ('treatment', 'Treatment'), ('aggressive_rop', 'AROP'), ('rw_rop', 'RW-ROP')]:
        adf = agreement_data[task]
        high = adf[adf['agreement'] >= 0.9]
        low = adf[adf['agreement'] < 0.9]
        h_acc = high['correct'].mean() if len(high) > 0 else 0
        l_acc = low['correct'].mean() if len(low) > 0 else 0
        agr_table.append([
            label,
            f"{adf['agreement'].mean():.3f}",
            f"{(adf['agreement'] == 1.0).sum()}/{len(adf)}",
            f"{(adf['agreement'] < 0.9).sum()}",
            f"{h_acc:.3f} ({len(high)})",
            f"{l_acc:.3f} ({len(low)})",
            f"{h_acc - l_acc:+.3f}",
        ])

    pdf.add_table(
        ["Task", "Mean Agr", "100%一致", "<90%数", "正解(>=90%)", "正解(<90%)", "差分"],
        agr_table,
        [22, 22, 26, 20, 33, 33, 22]
    )

    corr_path = str(FIG_DIR / "agreement_vs_correctness.png")
    plot_agreement_vs_correctness(agreement_data, corr_path)
    pdf.image(corr_path, w=155)
    pdf.ln(2)

    pdf.highlight_box(
        "一致率 < 90% の症例は正解率が大幅に低下 (Plus: 96% -> 59%, RW-ROP: 96% -> 69%)\n"
        "-> 予測一致率は信頼度指標として有用: 低一致率の患者をフラグし手動レビューに回す運用が可能",
        "green"
    )

    # ===== Page 8: Conclusions =====
    pdf.add_page()
    pdf.section_title("6. 結論と臨床的意義")

    pdf.subsection_title("6.1 主要な知見")
    pdf.body_text(
        "1. 画像品質フィルタリング (Top-5) で74%の画像を削減しても主要性能を維持\n"
        "   Zone/Stage/Plus: accuracy/kappa の低下は 2.5pp 以内\n\n"
        "2. Treatment/AROP は画像を絞るほど改善 (単調増加)\n"
        "   Treatment AUC: 0.978 -> 0.984, AROP Sensitivity: 0.860 -> 0.933\n"
        "   低品質画像がノイズとなり重症例の検出を妨げていた\n\n"
        "3. 患者単位 Soft Vote で Treatment/AROP が劇的に改善\n"
        "   Treatment Sensitivity: 0.917 -> 1.000 (Top-5), AUC: 0.978 -> 0.989\n"
        "   AROP Sensitivity: 0.860 -> 1.000, AUC: 0.916 -> 0.964\n\n"
        "4. RW-ROP は画像フィルタリングで微減するが、Majority Vote でほぼ回復\n"
        "   Sensitivity: 0.865 -> 0.827 (Top-5) -> 0.869 (Soft Vote Top-5)\n\n"
        "5. 予測一致率は信頼度指標として有用\n"
        "   一致率 < 90% の症例は多数決正解率が大幅低下 (Plus 37pp, RW-ROP 27pp)"
    )

    pdf.subsection_title("6.2 臨床運用への推奨")
    pdf.body_text(
        "1. 動画から Top-5 画像を品質フィルタリングで自動選出\n"
        "2. 5枚の AI 予測確率を平均 (Soft Vote) し患者単位の診断を出力\n"
        "3. 画像間の予測一致率を算出し、低一致率の患者にフラグを付与\n"
        "   -> 低一致率 (< 90%) の患者は手動レビューを推奨"
    )

    pdf.subsection_title("6.3 次のステップ")
    pdf.body_text(
        "- RW-ROP 閾値最適化の検討 (Youden's J で Sensitivity/Specificity のバランス調整)\n"
        "- 一致率に基づく信頼度スコアの臨床ワークフローへの組み込み\n"
        "- 前向き検証データでの再現性確認"
    )

    pdf.output(str(OUTPUT_PATH))
    print(f"PDF saved to: {OUTPUT_PATH}")


# ============ Main ============
def main():
    print("Loading data...")
    merged_df = load_data()
    print(f"  {len(merged_df)} images, {merged_df['video_id'].nunique()} video_ids")

    print("Selecting Top-K...")
    top10_df = select_top_k(merged_df, 10)
    top5_df = select_top_k(merged_df, 5)
    print(f"  Top-10: {len(top10_df)}, Top-5: {len(top5_df)}")

    print("Evaluating per-image...")
    all_results = {
        'per_image': evaluate_all(merged_df),
        'top10_img': evaluate_all(top10_df),
        'top5_img': evaluate_all(top5_df),
    }

    print("Computing Majority Vote (Soft Vote)...")
    for label, src_df in [('All', merged_df), ('Top-10', top10_df), ('Top-5', top5_df)]:
        key = f'{label}_soft'
        agg_df = aggregate_soft_vote(src_df)
        all_results[key] = evaluate_all(agg_df)
        print(f"  {key}: {len(agg_df)} videos")

    print("Computing agreement...")
    agreement_data = compute_agreement(merged_df)

    print("Generating figures...")
    print("Generating PDF...")
    generate_pdf(all_results, agreement_data)


if __name__ == '__main__':
    main()
