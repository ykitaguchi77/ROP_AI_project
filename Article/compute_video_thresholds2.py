"""Check if majority_vote_results.json uses hard-prediction-derived RW-ROP."""
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score

df = pd.read_csv("outputs_clinical_v3_retrain/predictions.csv")
top10_df = pd.read_csv("outputs_clinical_v3_retrain/top10_selected_images.csv")
top10_merged = top10_df.merge(df, on="image_path", how="inner", suffixes=("_sel", ""))
top5_merged = top10_merged.groupby("video_id").head(5).copy()

# RW-ROP label
for d in [df, top10_merged, top5_merged]:
    d["rw_rop_label"] = ((d["zone_label"] == 0) | (d["stage_label"] == 3) | (d["plus_label"] == 2)).astype(int)

def compute_video_rwrop_from_soft_multiclass(frame_data):
    """Average multiclass probs per video, argmax each, then derive RW-ROP binary."""
    video_agg = frame_data.groupby("video_id").agg({
        "rw_rop_label": "first",
        "zone_prob_0": "mean", "zone_prob_1": "mean", "zone_prob_2": "mean",
        "stage_prob_0": "mean", "stage_prob_1": "mean", "stage_prob_2": "mean", "stage_prob_3": "mean",
        "plus_prob_0": "mean", "plus_prob_1": "mean", "plus_prob_2": "mean",
    }).reset_index()

    zone_pred = video_agg[["zone_prob_0", "zone_prob_1", "zone_prob_2"]].values.argmax(axis=1)
    stage_pred = video_agg[["stage_prob_0", "stage_prob_1", "stage_prob_2", "stage_prob_3"]].values.argmax(axis=1)
    plus_pred = video_agg[["plus_prob_0", "plus_prob_1", "plus_prob_2"]].values.argmax(axis=1)

    rw_rop_pred = ((zone_pred == 0) | (stage_pred == 3) | (plus_pred == 2)).astype(int)
    y_true = video_agg["rw_rop_label"].values

    # For AUC, use probability
    video_agg["rw_rop_prob"] = 1 - (1 - video_agg["zone_prob_0"]) * (1 - video_agg["stage_prob_3"]) * (1 - video_agg["plus_prob_2"])

    return y_true, rw_rop_pred, video_agg["rw_rop_prob"].values

print("=== Approach 3: Soft-voted multiclass → derive RW-ROP binary ===")
for name, data in [("All", df), ("Top-10", top10_merged), ("Top-5", top5_merged)]:
    y_true, y_pred, y_prob = compute_video_rwrop_from_soft_multiclass(data)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    auc = roc_auc_score(y_true, y_prob)
    print(f"{name}_soft: Sens={sens:.4f}, Spec={spec:.4f}, AUC={auc:.4f}")
    print(f"  JSON target: Sens=", end="")
    if name == "All":
        print("0.8214, Spec=0.9356, AUC=0.9438")
    elif name == "Top-10":
        print("0.8452, Spec=0.9318, AUC=0.9469")
    else:
        print("0.8452, Spec=0.9280, AUC=0.9442")
