"""Compute video-level threshold optimization matching majority_vote pipeline."""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

df = pd.read_csv("outputs_clinical_v3_retrain/predictions.csv")
top10_df = pd.read_csv("outputs_clinical_v3_retrain/top10_selected_images.csv")
top10_merged = top10_df.merge(df, on="image_path", how="inner", suffixes=("_sel", ""))
top5_merged = top10_merged.groupby("video_id").head(5)

# RW-ROP label (video level, use first image's labels)
df["rw_rop_label"] = ((df["zone_label"] == 0) | (df["stage_label"] == 3) | (df["plus_label"] == 2)).astype(int)
top10_merged["rw_rop_label"] = ((top10_merged["zone_label"] == 0) | (top10_merged["stage_label"] == 3) | (top10_merged["plus_label"] == 2)).astype(int)
top5_merged = top10_merged.groupby("video_id").head(5)
top5_merged["rw_rop_label"] = ((top5_merged["zone_label"] == 0) | (top5_merged["stage_label"] == 3) | (top5_merged["plus_label"] == 2)).astype(int)

def eval_thr(y_true, y_prob, thr):
    y_pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    return sens, spec

def find_thresholds(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    youden_idx = np.argmax(tpr - fpr)
    youden_thr = thresholds[youden_idx]
    sens95_candidates = np.where(tpr >= 0.95)[0]
    if len(sens95_candidates) > 0:
        best_idx = sens95_candidates[np.argmax(1 - fpr[sens95_candidates])]
        sens95_thr = thresholds[best_idx]
    else:
        sens95_thr = 0.0
    return youden_thr, sens95_thr, auc

# Approach 1: Average task probs at video level, THEN derive RW-ROP
# (matches majority_vote_results.json)
def compute_rwrop_video_approach1(frame_df):
    """Average zone/stage/plus probs per video, then derive RW-ROP prob."""
    video_agg = frame_df.groupby("video_id").agg({
        "rw_rop_label": "first",
        "zone_prob_0": "mean",
        "stage_prob_3": "mean",
        "plus_prob_2": "mean",
    }).reset_index()
    video_agg["rw_rop_prob"] = 1 - (1 - video_agg["zone_prob_0"]) * (1 - video_agg["stage_prob_3"]) * (1 - video_agg["plus_prob_2"])
    return video_agg["rw_rop_label"].values, video_agg["rw_rop_prob"].values

# Approach 2: Derive RW-ROP per image, then average
def compute_rwrop_video_approach2(frame_df):
    """Derive RW-ROP prob per image, then average per video."""
    frame_df = frame_df.copy()
    frame_df["rw_rop_prob"] = 1 - (1 - frame_df["zone_prob_0"]) * (1 - frame_df["stage_prob_3"]) * (1 - frame_df["plus_prob_2"])
    video_agg = frame_df.groupby("video_id").agg({
        "rw_rop_label": "first",
        "rw_rop_prob": "mean",
    }).reset_index()
    return video_agg["rw_rop_label"].values, video_agg["rw_rop_prob"].values

# Test which approach matches JSON: Top-5_soft rw_rop: sens=0.8452, spec=0.9280
print("=== Approach comparison (Top-5 soft, RW-ROP, threshold=0.5) ===")
y1, p1 = compute_rwrop_video_approach1(top5_merged)
s1, sp1 = eval_thr(y1, p1, 0.5)
print(f"Approach 1 (avg probs then derive): Sens={s1:.4f}, Spec={sp1:.4f}")

y2, p2 = compute_rwrop_video_approach2(top5_merged)
s2, sp2 = eval_thr(y2, p2, 0.5)
print(f"Approach 2 (derive then avg probs): Sens={s2:.4f}, Spec={sp2:.4f}")
print(f"JSON target: Sens=0.8452, Spec=0.9280")

# Use the approach that matches, then compute thresholds
print("\n=== VIDEO-LEVEL THRESHOLD OPTIMIZATION (Approach 1) ===")

for name, frame_data in [("All", df), ("Top-10", top10_merged), ("Top-5", top5_merged)]:
    print(f"\n--- {name} soft voting ---")

    # Treatment
    tx_agg = frame_data.groupby("video_id").agg({
        "treatment_label": "first",
        "treatment_prob_1": "mean",
    }).reset_index()
    y_true_tx = tx_agg["treatment_label"].values
    y_prob_tx = tx_agg["treatment_prob_1"].values

    youden_thr, sens95_thr, auc = find_thresholds(y_true_tx, y_prob_tx)
    ds, dsp = eval_thr(y_true_tx, y_prob_tx, 0.5)
    ys, ysp = eval_thr(y_true_tx, y_prob_tx, youden_thr)
    ss, ssp = eval_thr(y_true_tx, y_prob_tx, sens95_thr)
    print(f"Treatment (AUC={auc:.4f}):")
    print(f"  Default(0.500): Sens={ds:.4f}, Spec={dsp:.4f}")
    print(f"  Youden({youden_thr:.4f}): Sens={ys:.4f}, Spec={ysp:.4f}")
    print(f"  Sens>=95%({sens95_thr:.4f}): Sens={ss:.4f}, Spec={ssp:.4f}")

    # RW-ROP (approach 1)
    y_true_rw, y_prob_rw = compute_rwrop_video_approach1(frame_data)
    youden_thr_rw, sens95_thr_rw, auc_rw = find_thresholds(y_true_rw, y_prob_rw)
    ds_rw, dsp_rw = eval_thr(y_true_rw, y_prob_rw, 0.5)
    ys_rw, ysp_rw = eval_thr(y_true_rw, y_prob_rw, youden_thr_rw)
    ss_rw, ssp_rw = eval_thr(y_true_rw, y_prob_rw, sens95_thr_rw)
    print(f"RW-ROP (AUC={auc_rw:.4f}):")
    print(f"  Default(0.500): Sens={ds_rw:.4f}, Spec={dsp_rw:.4f}")
    print(f"  Youden({youden_thr_rw:.4f}): Sens={ys_rw:.4f}, Spec={ysp_rw:.4f}")
    print(f"  Sens>=95%({sens95_thr_rw:.4f}): Sens={ss_rw:.4f}, Spec={ssp_rw:.4f}")

    # A-ROP
    arop_agg = frame_data.groupby("video_id").agg({
        "aggressive_rop_label": "first",
        "aggressive_rop_prob_1": "mean",
    }).reset_index()
    y_true_arop = arop_agg["aggressive_rop_label"].values
    y_prob_arop = arop_agg["aggressive_rop_prob_1"].values
    arop_auc = roc_auc_score(y_true_arop, y_prob_arop)
    arop_s, _ = eval_thr(y_true_arop, y_prob_arop, 0.5)
    print(f"A-ROP: Sens={arop_s:.4f}, AUC={arop_auc:.4f}")
