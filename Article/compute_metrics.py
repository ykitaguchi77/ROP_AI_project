"""Compute all missing metrics for article.md update (clinical_v3_retrain)."""
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, f1_score,
    roc_auc_score, confusion_matrix
)
import json

# Load predictions
df = pd.read_csv("outputs_clinical_v3_retrain/predictions.csv")
print(f"Total rows: {len(df)}")
print(f"Unique videos: {df['video_id'].nunique()}")
print(f"Folds: {sorted(df['fold'].unique())}")

# ============================================================
# 1. CLASS DISTRIBUTIONS (Table 2)
# ============================================================
print("\n" + "="*60)
print("CLASS DISTRIBUTIONS")
print("="*60)

n = len(df)
for task, classes in [
    ("zone_label", {0: "Zone I", 1: "Zone II", 2: "Zone III"}),
    ("stage_label", {0: "Stage 0", 1: "Stage 1", 2: "Stage 2", 3: "Stage 3"}),
    ("plus_label", {0: "Normal", 1: "Pre-plus", 2: "Plus"}),
    ("aggressive_rop_label", {0: "No", 1: "Yes"}),
    ("treatment_label", {0: "Not indicated", 1: "Indicated"}),
]:
    print(f"\n{task}:")
    valid = df[task].dropna()
    unknown = n - len(valid)
    for val, name in classes.items():
        cnt = (valid == val).sum()
        pct = cnt / n * 100
        print(f"  {name}: {cnt} ({pct:.1f}%)")
    if unknown > 0:
        print(f"  Unknown: {unknown}")

# RW-ROP derived
df["rw_rop_label"] = ((df["zone_label"] == 0) |
                       (df["stage_label"] == 3) |
                       (df["plus_label"] == 2)).astype(int)
rw_yes = df["rw_rop_label"].sum()
rw_no = n - rw_yes
print(f"\nRW-ROP:")
print(f"  No: {rw_no} ({rw_no/n*100:.1f}%)")
print(f"  Yes: {rw_yes} ({rw_yes/n*100:.1f}%)")

# Video-level counts
vdf = df.groupby("video_id").first()
print(f"\nVideo-level counts:")
print(f"  Treatment Yes: {(vdf['treatment_label']==1).sum()}")
print(f"  A-ROP Yes: {(vdf['aggressive_rop_label']==1).sum()}")
vdf["rw_rop_label"] = ((vdf["zone_label"] == 0) |
                        (vdf["stage_label"] == 3) |
                        (vdf["plus_label"] == 2)).astype(int)
print(f"  RW-ROP Yes: {vdf['rw_rop_label'].sum()}")

# ============================================================
# 2. PER-FOLD METRICS (Table 3) - Treatment Specificity & RW-ROP
# ============================================================
print("\n" + "="*60)
print("PER-FOLD METRICS (Table 3)")
print("="*60)

# RW-ROP probability
df["rw_rop_prob"] = 1 - (1 - df["zone_prob_0"]) * (1 - df["stage_prob_3"]) * (1 - df["plus_prob_2"])
df["rw_rop_pred"] = (df["rw_rop_prob"] >= 0.5).astype(int)

for task in ["treatment", "rw_rop"]:
    print(f"\n--- {task} ---")
    sensitivities = []
    specificities = []
    ppvs = []
    npvs = []
    aucs = []

    for fold in sorted(df["fold"].unique()):
        fdf = df[df["fold"] == fold]

        if task == "treatment":
            y_true = fdf["treatment_label"].values
            y_pred = fdf["treatment_pred"].values
            y_prob = fdf["treatment_prob_1"].values
        else:
            y_true = fdf["rw_rop_label"].values
            y_pred = fdf["rw_rop_pred"].values
            y_prob = fdf["rw_rop_prob"].values

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else np.nan

        sensitivities.append(sens)
        specificities.append(spec)
        ppvs.append(ppv)
        npvs.append(npv)
        aucs.append(auc)

        print(f"  Fold {fold}: Sens={sens:.4f}, Spec={spec:.4f}, PPV={ppv:.4f}, NPV={npv:.4f}, AUC={auc:.4f}")

    print(f"  Mean: Sens={np.mean(sensitivities):.4f}±{np.std(sensitivities):.4f}")
    print(f"         Spec={np.mean(specificities):.4f}±{np.std(specificities):.4f}")
    print(f"         PPV={np.mean(ppvs):.4f}±{np.std(ppvs):.4f}")
    print(f"         NPV={np.mean(npvs):.4f}±{np.std(npvs):.4f}")
    aucs_valid = [a for a in aucs if not np.isnan(a)]
    print(f"         AUC={np.mean(aucs_valid):.4f}±{np.std(aucs_valid):.4f}")

# ============================================================
# 3. POOLED PER-IMAGE METRICS (Table 4 - Per-Image column)
# ============================================================
print("\n" + "="*60)
print("POOLED PER-IMAGE METRICS (Table 4)")
print("="*60)

# Multiclass
for task, n_classes in [("zone", 3), ("stage", 4), ("plus", 3)]:
    y_true = df[f"{task}_label"].dropna().astype(int)
    y_pred = df.loc[y_true.index, f"{task}_pred"].astype(int)

    acc = accuracy_score(y_true, y_pred)
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    f1m = f1_score(y_true, y_pred, average="macro")
    print(f"{task}: Acc={acc:.4f}, QWK={qwk:.4f}, F1_macro={f1m:.4f}")

# Binary - A-ROP
y_true_arop = df["aggressive_rop_label"].values
y_pred_arop = df["aggressive_rop_pred"].values
y_prob_arop = df["aggressive_rop_prob_1"].values

tn, fp, fn, tp = confusion_matrix(y_true_arop, y_pred_arop, labels=[0,1]).ravel()
sens_arop = tp / (tp + fn) if (tp + fn) > 0 else 0
spec_arop = tn / (tn + fp) if (tn + fp) > 0 else 0
auc_arop = roc_auc_score(y_true_arop, y_prob_arop)
print(f"A-ROP: Sens={sens_arop:.4f}, Spec={spec_arop:.4f}, AUC={auc_arop:.4f}")

# Binary - Treatment
y_true_tx = df["treatment_label"].values
y_pred_tx = df["treatment_pred"].values
y_prob_tx = df["treatment_prob_1"].values

tn, fp, fn, tp = confusion_matrix(y_true_tx, y_pred_tx, labels=[0,1]).ravel()
sens_tx = tp / (tp + fn)
spec_tx = tn / (tn + fp)
ppv_tx = tp / (tp + fp)
npv_tx = tn / (tn + fn)
auc_tx = roc_auc_score(y_true_tx, y_prob_tx)
print(f"Treatment: Sens={sens_tx:.4f}, Spec={spec_tx:.4f}, PPV={ppv_tx:.4f}, NPV={npv_tx:.4f}, AUC={auc_tx:.4f}")

# Binary - RW-ROP
y_true_rw = df["rw_rop_label"].values
y_pred_rw = df["rw_rop_pred"].values
y_prob_rw = df["rw_rop_prob"].values

tn, fp, fn, tp = confusion_matrix(y_true_rw, y_pred_rw, labels=[0,1]).ravel()
sens_rw = tp / (tp + fn)
spec_rw = tn / (tn + fp)
ppv_rw = tp / (tp + fp)
npv_rw = tn / (tn + fn)
auc_rw = roc_auc_score(y_true_rw, y_prob_rw)
print(f"RW-ROP: Sens={sens_rw:.4f}, Spec={spec_rw:.4f}, PPV={ppv_rw:.4f}, NPV={npv_rw:.4f}, AUC={auc_rw:.4f}")

# ============================================================
# 4. VIDEO-LEVEL THRESHOLD OPTIMIZATION (Top-5 soft voting)
# ============================================================
print("\n" + "="*60)
print("VIDEO-LEVEL THRESHOLD OPTIMIZATION")
print("="*60)

# Load top-10 selected images
top10_df = pd.read_csv("outputs_clinical_v3_retrain/top10_selected_images.csv")
print(f"Top-10 selected images: {len(top10_df)}")

# Merge predictions with top-10 selections
# Need to match on image_path
top10_merged = top10_df.merge(df, on="image_path", how="inner", suffixes=("_sel", ""))
print(f"Top-10 merged: {len(top10_merged)}")

# Top-5: take top 5 per video by rank
top5_merged = top10_merged.groupby("video_id").head(5)
print(f"Top-5: {len(top5_merged)}")

from sklearn.metrics import roc_curve

def compute_video_level_soft(frame_df, agg_col_label, agg_col_prob, threshold=0.5):
    """Aggregate per-image probs to video-level by mean, then apply threshold."""
    video_agg = frame_df.groupby("video_id").agg({
        agg_col_label: "first",
        agg_col_prob: "mean"
    }).reset_index()

    y_true = video_agg[agg_col_label].values
    y_prob = video_agg[agg_col_prob].values
    y_pred = (y_prob >= threshold).astype(int)

    return y_true, y_prob, y_pred

def find_thresholds(y_true, y_prob):
    """Find Youden and Sens>=95% thresholds."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    # Youden
    youden_idx = np.argmax(tpr - fpr)
    youden_thr = thresholds[youden_idx]

    # Sens >= 95%
    sens95_candidates = np.where(tpr >= 0.95)[0]
    if len(sens95_candidates) > 0:
        # Among those with sens >= 95%, find highest specificity
        best_idx = sens95_candidates[np.argmax(1 - fpr[sens95_candidates])]
        sens95_thr = thresholds[best_idx]
    else:
        sens95_thr = 0.0

    return youden_thr, sens95_thr, auc

def eval_at_threshold(y_true, y_prob, thr):
    y_pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    return sens, spec

# For each condition: All, Top-10, Top-5
for name, frame_data in [("All", df), ("Top-10", top10_merged), ("Top-5", top5_merged)]:
    print(f"\n=== {name} (soft voting) ===")

    for task, label_col, prob_col in [
        ("Treatment", "treatment_label", "treatment_prob_1"),
        ("RW-ROP", "rw_rop_label", "rw_rop_prob"),
    ]:
        y_true, y_prob, _ = compute_video_level_soft(frame_data, label_col, prob_col)
        youden_thr, sens95_thr, auc = find_thresholds(y_true, y_prob)

        default_sens, default_spec = eval_at_threshold(y_true, y_prob, 0.5)
        youden_sens, youden_spec = eval_at_threshold(y_true, y_prob, youden_thr)
        sens95_sens, sens95_spec = eval_at_threshold(y_true, y_prob, sens95_thr)

        print(f"\n  {task} (AUC={auc:.4f}):")
        print(f"    Default(0.5): Sens={default_sens:.4f}, Spec={default_spec:.4f}")
        print(f"    Youden(thr={youden_thr:.4f}): Sens={youden_sens:.4f}, Spec={youden_spec:.4f}")
        print(f"    Sens>=95%(thr={sens95_thr:.4f}): Sens={sens95_sens:.4f}, Spec={sens95_spec:.4f}")

# ============================================================
# 5. A-ROP VIDEO-LEVEL POOLED
# ============================================================
print("\n" + "="*60)
print("A-ROP VIDEO-LEVEL POOLED")
print("="*60)

for name, frame_data in [("Per-Image pooled", df), ("Top-10", top10_merged), ("Top-5", top5_merged)]:
    if name == "Per-Image pooled":
        # Just pooled per-image
        y_true = frame_data["aggressive_rop_label"].values
        y_pred = frame_data["aggressive_rop_pred"].values
        y_prob = frame_data["aggressive_rop_prob_1"].values
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        auc = roc_auc_score(y_true, y_prob)
        print(f"{name}: Sens={sens:.4f}, AUC={auc:.4f}")
    else:
        # Video-level soft voting
        y_true, y_prob, y_pred = compute_video_level_soft(
            frame_data, "aggressive_rop_label", "aggressive_rop_prob_1"
        )
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        auc = roc_auc_score(y_true, y_prob)
        print(f"{name} soft: Sens={sens:.4f}, AUC={auc:.4f}, n_pos={y_true.sum()}")

# ============================================================
# 6. UNKNOWN STAGE/PLUS COUNT
# ============================================================
print("\n" + "="*60)
print("UNKNOWN/MISSING LABELS")
print("="*60)

# Check for any special label values - stage has 4 classes (0-3), plus has 3 (0-2)
# In predictions.csv, labels should be int. Check if any have unusual values
for col in ["zone_label", "stage_label", "plus_label"]:
    unique_vals = sorted(df[col].unique())
    print(f"{col}: unique values = {unique_vals}, count = {len(df[col])}")

# Count stage==-1 or plus==-1 style unknowns
stage_unknown = df["stage_label"].isna().sum()
plus_unknown = df["plus_label"].isna().sum()
print(f"Stage unknown (NaN): {stage_unknown}")
print(f"Plus unknown (NaN): {plus_unknown}")

# Check for label -1
for col in ["stage_label", "plus_label"]:
    neg_count = (df[col] < 0).sum()
    if neg_count > 0:
        print(f"{col} has {neg_count} negative values")
