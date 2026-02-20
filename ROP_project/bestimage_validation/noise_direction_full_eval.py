"""
Noise Direction Full Evaluation
================================
1. Generate noise maps for ALL lens crops
2. Extract EfficientNet-B0 features from noise maps
3. Compute "noise direction" vector (analogous to quality_proj)
4. Evaluate as standalone feature and combined with Equal-3key
5. Full LOVO-CV on all 22 cases
"""

import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torchvision import models, transforms
import warnings
warnings.filterwarnings('ignore')

BASE = Path(r"C:\Users\ykita\ROP_AI_project\ROP_project\bestimage_validation")

# ============================================================
# 1. Load data
# ============================================================
meta = pd.read_csv(BASE / "lens_crops" / "metadata.csv")
merged = pd.read_csv(BASE / "validation_results" / "merged_with_disc.csv")

meta_valid = meta[meta["has_lens"] == True].reset_index(drop=True)
print(f"Valid lens crops: {len(meta_valid)}")
print(f"Cases: {meta_valid['case_id'].nunique()}")

# Merge human labels (drop is_human_top from meta to avoid _x/_y suffix)
meta_valid = meta_valid.drop(columns=["is_human_top"], errors="ignore")
meta_valid = meta_valid.merge(
    merged[["image_name", "case_id", "is_human_top",
            "mbss_Grad_p90", "disc_edge_coverage_ratio", "retina_ratio",
            "disc_detected"]].drop_duplicates(),
    on=["image_name", "case_id"],
    how="left"
)
meta_valid["is_human_top"] = meta_valid["is_human_top"].fillna(False)
print(f"Human top images in valid set: {meta_valid['is_human_top'].sum()}")

# ============================================================
# 2. Noise map generation functions
# ============================================================
def map_highlight(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float64)
    s = hsv[:, :, 1].astype(np.float64)
    highlight = np.clip((v / 255) * (1 - s / 255) * 255, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(highlight, (15, 15), 0)

def map_haze(gray):
    from scipy import ndimage
    local_max = ndimage.maximum_filter(gray.astype(np.float64), size=31)
    local_min = ndimage.minimum_filter(gray.astype(np.float64), size=31)
    contrast = local_max - local_min
    if contrast.max() > 0:
        return (255 - (contrast / contrast.max() * 255)).astype(np.uint8)
    return np.full_like(gray, 255)

def map_local_noise(gray):
    smoothed = cv2.GaussianBlur(gray.astype(np.float64), (5, 5), 0)
    noise = np.abs(gray.astype(np.float64) - smoothed)
    noise_sq = noise ** 2
    local_rms = np.sqrt(cv2.GaussianBlur(noise_sq, (31, 31), 0))
    if local_rms.max() > 0:
        return (local_rms / local_rms.max() * 255).astype(np.uint8)
    return np.zeros_like(gray)

def make_noise_map(img_bgr, gray):
    """3-channel: [Highlight, Haze, LocalNoise]"""
    return np.stack([map_highlight(img_bgr), map_haze(gray), map_local_noise(gray)], axis=-1)

# Also compute scalar noise features
def compute_noise_scalars(img_bgr, gray):
    hl = map_highlight(img_bgr)
    hz = map_haze(gray)
    ns = map_local_noise(gray)
    return {
        "highlight_mean": hl.mean(),
        "highlight_p90": np.percentile(hl, 90),
        "highlight_max_area": (hl > 200).mean(),
        "haze_mean": hz.mean(),
        "local_noise_mean": ns.mean(),
    }

# ============================================================
# 3. Extract CNN features for all noise maps
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

effnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
effnet.classifier = nn.Identity()
effnet = effnet.to(device).eval()

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

SIZE = 224
BATCH_SIZE = 32

print("\nExtracting noise map features for all images...")

all_features = []
all_scalars = []
valid_indices = []

batch_tensors = []
batch_indices = []

for idx, row in meta_valid.iterrows():
    img_bgr = cv2.imread(str(row["crop_path"]))
    if img_bgr is None:
        continue

    img_bgr = cv2.resize(img_bgr, (SIZE, SIZE))
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Noise map
    nmap = make_noise_map(img_bgr, gray)
    tensor = preprocess(nmap)
    batch_tensors.append(tensor)
    batch_indices.append(idx)

    # Scalar features
    scalars = compute_noise_scalars(img_bgr, gray)
    all_scalars.append(scalars)

    # Process in batches
    if len(batch_tensors) >= BATCH_SIZE:
        batch = torch.stack(batch_tensors).to(device)
        with torch.no_grad():
            feats = effnet(batch).cpu().numpy()
        all_features.append(feats)
        valid_indices.extend(batch_indices)
        batch_tensors = []
        batch_indices = []

        if len(valid_indices) % 1000 == 0:
            print(f"  {len(valid_indices)} images processed...")

# Last batch
if batch_tensors:
    batch = torch.stack(batch_tensors).to(device)
    with torch.no_grad():
        feats = effnet(batch).cpu().numpy()
    all_features.append(feats)
    valid_indices.extend(batch_indices)

noise_features = np.concatenate(all_features, axis=0)
print(f"Total: {len(valid_indices)} images, feature shape: {noise_features.shape}")

# Map features back to metadata
meta_feat = meta_valid.loc[valid_indices].reset_index(drop=True)
scalar_df = pd.DataFrame(all_scalars).reset_index(drop=True)
meta_feat = pd.concat([meta_feat, scalar_df], axis=1)

# ============================================================
# 4. Noise Direction Projection (analogous to quality_proj)
# ============================================================
print("\n" + "=" * 70)
print("LOVO-CV: Noise Direction Projection")
print("=" * 70)

# Load existing quality_proj features (RETFound-based)
retfound_data = np.load(BASE / "lens_crops" / "cnn_features_retfound.npz", allow_pickle=True)
retfound_features = retfound_data["features"]
retfound_names = retfound_data["image_names"]

# Build name-to-index map for noise features
noise_name_to_idx = {row["image_name"]: i for i, (_, row) in enumerate(meta_feat.iterrows())}

cases = sorted(meta_feat["case_id"].unique())
print(f"Cases: {len(cases)}")

results = []

for test_case in cases:
    # Split
    train_mask = meta_feat["case_id"] != test_case
    test_mask = meta_feat["case_id"] == test_case

    train_data = meta_feat[train_mask]
    test_data = meta_feat[test_mask]
    train_feats = noise_features[train_mask.values]
    test_feats = noise_features[test_mask.values]

    # Compute noise direction from training data
    # Direction = mean(human_best) - mean(all)
    train_best_mask = train_data["is_human_top"].values.astype(bool)

    if train_best_mask.sum() == 0:
        continue

    mean_best = train_feats[train_best_mask].mean(axis=0)
    mean_all = train_feats.mean(axis=0)
    noise_direction = mean_best - mean_all
    noise_direction = noise_direction / (np.linalg.norm(noise_direction) + 1e-8)

    # Project test images
    test_noise_proj = test_feats @ noise_direction

    # Also get existing features for test case
    test_merged = merged[merged["case_id"] == test_case].copy()

    # Match noise_proj to merged data
    test_noise_dict = {}
    for i, (_, row) in enumerate(test_data.iterrows()):
        test_noise_dict[row["image_name"]] = test_noise_proj[i]

    test_merged["noise_proj"] = test_merged["image_name"].map(test_noise_dict)

    # Also add scalar noise features
    test_scalar_dict = {}
    for _, row in test_data.iterrows():
        test_scalar_dict[row["image_name"]] = {
            "highlight_mean": row.get("highlight_mean", np.nan),
            "haze_mean": row.get("haze_mean", np.nan),
        }

    for col in ["highlight_mean", "haze_mean"]:
        test_merged[col] = test_merged["image_name"].map(
            lambda x: test_scalar_dict.get(x, {}).get(col, np.nan)
        )

    # Filter: disc_detected + retina_ratio > 0 + disc_edge_coverage >= 0.80
    valid = test_merged[
        (test_merged["disc_detected"] == True) &
        (test_merged["retina_ratio"] > 0) &
        (test_merged["disc_edge_coverage_ratio"] >= 0.80)
    ].copy()

    if len(valid) == 0:
        # Fallback: no filter
        valid = test_merged[test_merged["noise_proj"].notna()].copy()

    if len(valid) == 0:
        continue

    n_human = valid["is_human_top"].sum()

    # Normalize features within this case
    def safe_norm(s):
        std = s.std()
        if std < 1e-8:
            return s * 0
        return (s - s.mean()) / std

    # Score functions to evaluate
    scores = {}

    # 1. Noise direction projection only
    if valid["noise_proj"].notna().sum() > 0:
        scores["noise_proj_only"] = safe_norm(valid["noise_proj"].fillna(0))

    # 2. Highlight (lower = better, so negate)
    if valid["highlight_mean"].notna().sum() > 0:
        scores["highlight_neg"] = -safe_norm(valid["highlight_mean"].fillna(valid["highlight_mean"].mean()))

    # 3. Equal-3key baseline (Grad + edge_cov + quality_proj)
    # We need quality_proj - load from RETFound features
    # For now, use the existing features from merged
    grad_norm = safe_norm(valid["mbss_Grad_p90"].fillna(0))
    edge_norm = safe_norm(valid["disc_edge_coverage_ratio"].fillna(0))

    # Quality proj: compute from RETFound features
    retfound_name_to_idx = {n: i for i, n in enumerate(retfound_names)}
    train_merged_names = train_data["image_name"].values
    train_retfound_mask = [retfound_name_to_idx.get(n) for n in merged[merged["case_id"] != test_case]["image_name"]]
    train_retfound_mask = [i for i in train_retfound_mask if i is not None]

    # Compute quality direction from RETFound
    train_merged_all = merged[merged["case_id"] != test_case]
    train_best_names = train_merged_all[train_merged_all["is_human_top"] == True]["image_name"].values
    train_all_names = train_merged_all["image_name"].values

    best_indices = [retfound_name_to_idx[n] for n in train_best_names if n in retfound_name_to_idx]
    all_indices = [retfound_name_to_idx[n] for n in train_all_names if n in retfound_name_to_idx]

    if best_indices and all_indices:
        qdir = retfound_features[best_indices].mean(axis=0) - retfound_features[all_indices].mean(axis=0)
        qdir = qdir / (np.linalg.norm(qdir) + 1e-8)

        test_names = valid["image_name"].values
        test_qproj = []
        for n in test_names:
            if n in retfound_name_to_idx:
                test_qproj.append(retfound_features[retfound_name_to_idx[n]] @ qdir)
            else:
                test_qproj.append(0)
        qproj_norm = safe_norm(pd.Series(test_qproj))
    else:
        qproj_norm = pd.Series(np.zeros(len(valid)))

    # Equal-3key: (Grad + edge_cov + qproj) / 3
    scores["equal_3key"] = (grad_norm.values + edge_norm.values + qproj_norm.values) / 3

    # 4. Equal-3key + noise_proj (4 features)
    if "noise_proj_only" in scores:
        nproj_norm = safe_norm(valid["noise_proj"].fillna(0))
        scores["equal_4key_nproj"] = (grad_norm.values + edge_norm.values + qproj_norm.values + nproj_norm.values) / 4

    # 5. Equal-3key + highlight_neg (4 features)
    if "highlight_neg" in scores:
        hl_norm = -safe_norm(valid["highlight_mean"].fillna(valid["highlight_mean"].mean()))
        scores["equal_4key_highlight"] = (grad_norm.values + edge_norm.values + qproj_norm.values + hl_norm.values) / 4

    # 6. Equal-3key + noise_proj + highlight_neg (5 features)
    if "noise_proj_only" in scores and "highlight_neg" in scores:
        scores["equal_5key"] = (grad_norm.values + edge_norm.values + qproj_norm.values + nproj_norm.values + hl_norm.values) / 5

    # 7. Equal-3key - noise_penalty (noise as rejection)
    if "noise_proj_only" in scores:
        # Use noise features as penalty (subtract from quality)
        scores["3key_minus_noise"] = (grad_norm.values + edge_norm.values + qproj_norm.values) / 3 - 0.3 * (-nproj_norm.values)

    # Evaluate each scoring method
    for method, score_vals in scores.items():
        valid_copy = valid.copy()
        valid_copy["score"] = score_vals
        top5 = valid_copy.nlargest(5, "score")
        matches = top5["is_human_top"].sum()

        results.append({
            "test_case": test_case,
            "method": method,
            "n_candidates": len(valid),
            "n_human_top": n_human,
            "top5_matches": matches,
        })

# ============================================================
# 5. Aggregate results
# ============================================================
results_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("LOVO-CV Results: Top-5 Precision")
print("=" * 70)

methods = results_df["method"].unique()
for method in methods:
    mdf = results_df[results_df["method"] == method]
    total_matches = mdf["top5_matches"].sum()
    total_human = mdf["n_human_top"].sum()
    n_cases = len(mdf)
    # Top-5 Precision: matches / (5 * n_cases_with_human)
    cases_with_human = mdf[mdf["n_human_top"] > 0]
    prec = total_matches / (5 * len(cases_with_human)) * 100 if len(cases_with_human) > 0 else 0
    video_match = (mdf["top5_matches"] > 0).sum()

    print(f"\n  {method}:")
    print(f"    Top-5 matches: {total_matches}/{total_human} total human")
    print(f"    Top-5 Precision: {prec:.1f}%")
    print(f"    Video match: {video_match}/{n_cases} ({video_match/n_cases*100:.1f}%)")

# Per-case breakdown for key methods
print("\n" + "=" * 70)
print("Per-case Breakdown")
print("=" * 70)

key_methods = ["equal_3key", "equal_4key_nproj", "equal_4key_highlight", "equal_5key"]
key_methods = [m for m in key_methods if m in methods]

header = f"{'Case':>8} {'nHuman':>7}"
for m in key_methods:
    header += f" {m[:15]:>16}"
print(header)
print("-" * (16 + 17 * len(key_methods)))

for case_id in sorted(results_df["test_case"].unique()):
    case_results = results_df[results_df["test_case"] == case_id]
    n_human = case_results.iloc[0]["n_human_top"]
    row_str = f"{case_id:>8} {int(n_human):>7}"
    for m in key_methods:
        mr = case_results[case_results["method"] == m]
        if len(mr) > 0:
            matches = int(mr.iloc[0]["top5_matches"])
            row_str += f" {matches:>16}"
        else:
            row_str += f" {'N/A':>16}"
    print(row_str)

# Summary comparison
print("\n" + "=" * 70)
print("SUMMARY COMPARISON")
print("=" * 70)

for method in key_methods:
    mdf = results_df[results_df["method"] == method]
    cases_with_human = mdf[mdf["n_human_top"] > 0]
    total_matches = cases_with_human["top5_matches"].sum()
    prec = total_matches / (5 * len(cases_with_human)) * 100 if len(cases_with_human) > 0 else 0
    print(f"  {method:<30} Top-5 Prec: {prec:.1f}%  ({int(total_matches)} matches)")

print("\nReference: Previous Equal-3key = 55.5%")
print("Done!")
