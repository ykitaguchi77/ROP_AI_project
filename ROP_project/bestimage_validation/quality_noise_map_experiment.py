"""
Quality Map & Noise Map Experiment
===================================
Two hypotheses:
1. Local quality maps (sharpness spatial distribution) → better quality signal for CNN
2. Noise/artifact maps (highlight, haze, local anomaly) → capture rejection criteria

For each map type:
- Generate spatial maps from lens crops
- Extract CNN features (EfficientNet-B0 frozen)
- Compare within-patient similarity & quality discrimination
- Also compute scalar statistics and correlate with human selection
"""

import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. Data loading
# ============================================================
BASE = Path(r"C:\Users\ykita\ROP_AI_project\ROP_project\bestimage_validation")
meta = pd.read_csv(BASE / "lens_crops" / "metadata.csv")
merged = pd.read_csv(BASE / "validation_results" / "merged_with_disc.csv")

meta = meta[meta["has_lens"] == True].reset_index(drop=True)

# Sample: 5 cases, 50 images each
np.random.seed(42)
sample_cases = np.random.choice(meta["case_id"].unique(), size=5, replace=False)
sample = meta[meta["case_id"].isin(sample_cases)].groupby("case_id").head(50).reset_index(drop=True)
print(f"Sampled: {len(sample)} images from cases {sorted(sample_cases)}")

# ============================================================
# 2. Map generation functions
# ============================================================

# --- QUALITY MAPS (sharpness / focus) ---

def map_laplacian_variance(gray, ksize=31):
    """Local Laplacian variance map - measures local focus quality.
    High value = sharp, Low value = blurry."""
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    lap_sq = lap ** 2
    # Local mean of squared Laplacian = local variance of Laplacian
    local_var = cv2.GaussianBlur(lap_sq, (ksize, ksize), 0)
    # Normalize to 0-255
    local_var = np.sqrt(local_var)  # sqrt for better dynamic range
    if local_var.max() > 0:
        local_var = (local_var / local_var.max() * 255).astype(np.uint8)
    return local_var

def map_gradient_magnitude(gray, ksize=31):
    """Local gradient magnitude map - measures edge strength spatially."""
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sx**2 + sy**2)
    # Local average
    local_mag = cv2.GaussianBlur(mag, (ksize, ksize), 0)
    if local_mag.max() > 0:
        local_mag = (local_mag / local_mag.max() * 255).astype(np.uint8)
    return local_mag

def map_highfreq_energy(gray, ksize=31):
    """High-frequency energy map via bandpass filtering.
    Captures fine detail / texture richness."""
    # Remove low-freq (large blur)
    low = cv2.GaussianBlur(gray.astype(np.float64), (51, 51), 0)
    high = gray.astype(np.float64) - low
    energy = high ** 2
    # Local average
    local_energy = cv2.GaussianBlur(energy, (ksize, ksize), 0)
    local_energy = np.sqrt(local_energy)
    if local_energy.max() > 0:
        local_energy = (local_energy / local_energy.max() * 255).astype(np.uint8)
    return local_energy

def map_quality_composite(gray):
    """3-channel quality map: [LapVar, GradMag, HFEnergy]"""
    return np.stack([
        map_laplacian_variance(gray),
        map_gradient_magnitude(gray),
        map_highfreq_energy(gray),
    ], axis=-1)

# --- NOISE / ARTIFACT MAPS ---

def map_highlight_mask(img_bgr):
    """Highlight/specular reflection detection.
    Bright saturated regions that indicate light artifacts."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # High V, low S = specular highlight
    v = hsv[:, :, 2].astype(np.float64)
    s = hsv[:, :, 1].astype(np.float64)
    highlight = np.clip((v / 255) * (1 - s / 255) * 255, 0, 255).astype(np.uint8)
    # Smooth
    highlight = cv2.GaussianBlur(highlight, (15, 15), 0)
    return highlight

def map_local_noise(gray, ksize=5):
    """Local noise estimation via difference from smoothed version.
    Detects sensor noise, compression artifacts."""
    smoothed = cv2.GaussianBlur(gray.astype(np.float64), (ksize, ksize), 0)
    noise = np.abs(gray.astype(np.float64) - smoothed)
    # Local RMS noise
    noise_sq = noise ** 2
    local_rms = np.sqrt(cv2.GaussianBlur(noise_sq, (31, 31), 0))
    if local_rms.max() > 0:
        local_rms = (local_rms / local_rms.max() * 255).astype(np.uint8)
    return local_rms

def map_haze_indicator(gray):
    """Haze/fog detection via local contrast.
    Low local contrast = hazy/foggy region (e.g., media opacity)."""
    # Local max - min in a window
    ksize = 31
    local_max = ndimage.maximum_filter(gray.astype(np.float64), size=ksize)
    local_min = ndimage.minimum_filter(gray.astype(np.float64), size=ksize)
    local_contrast = local_max - local_min
    # Invert: high value = MORE haze (low contrast)
    if local_contrast.max() > 0:
        haze = 255 - (local_contrast / local_contrast.max() * 255).astype(np.uint8)
    else:
        haze = np.full_like(gray, 255)
    return haze

def map_saturation_anomaly(img_bgr):
    """Saturation anomaly map.
    Unusual color saturation can indicate artifacts or poor white balance."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    # Compute deviation from local mean
    local_mean = cv2.GaussianBlur(s.astype(np.float64), (31, 31), 0)
    anomaly = np.abs(s.astype(np.float64) - local_mean)
    if anomaly.max() > 0:
        anomaly = (anomaly / anomaly.max() * 255).astype(np.uint8)
    return anomaly

def map_noise_composite(img_bgr, gray):
    """3-channel noise/artifact map: [Highlight, Haze, LocalNoise]"""
    return np.stack([
        map_highlight_mask(img_bgr),
        map_haze_indicator(gray),
        map_local_noise(gray),
    ], axis=-1)

# ============================================================
# 3. Feature extraction
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

preprocess_rgb = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_batch(tensors):
    batch = torch.stack(tensors).to(device)
    with torch.no_grad():
        feats = effnet(batch)
    return feats.cpu().numpy()

# ============================================================
# 4. Process all images
# ============================================================
print("\nProcessing images...")

SIZE = 224
features = {
    "raw": {},
    "quality_map": {},
    "noise_map": {},
    "quality_noise_6ch": {},  # Will be split into two CNN passes
}

# Also collect scalar statistics for correlation analysis
scalar_stats = []

for case_id in sorted(sample_cases):
    case_imgs = sample[sample["case_id"] == case_id]
    tensors = {"raw": [], "quality_map": [], "noise_map": []}
    valid_rows = []

    for _, row in case_imgs.iterrows():
        img_bgr = cv2.imread(str(row["crop_path"]))
        if img_bgr is None:
            continue
        img_bgr = cv2.resize(img_bgr, (SIZE, SIZE))
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Raw RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tensors["raw"].append(preprocess_rgb(Image.fromarray(img_rgb)))

        # Quality map (3-ch)
        qmap = map_quality_composite(gray)
        tensors["quality_map"].append(preprocess(qmap))

        # Noise map (3-ch)
        nmap = map_noise_composite(img_bgr, gray)
        tensors["noise_map"].append(preprocess(nmap))

        # Scalar statistics for this image
        is_best = merged[(merged["case_id"] == case_id) &
                         (merged["image_name"] == row["image_name"])]["is_human_top"].values
        is_best = bool(is_best[0]) if len(is_best) > 0 else False

        # Quality scalars
        lap_var = map_laplacian_variance(gray)
        grad_mag = map_gradient_magnitude(gray)
        hf_energy = map_highfreq_energy(gray)

        # Noise scalars
        highlight = map_highlight_mask(img_bgr)
        haze = map_haze_indicator(gray)
        noise = map_local_noise(gray)

        scalar_stats.append({
            "case_id": case_id,
            "image_name": row["image_name"],
            "is_human_top": is_best,
            # Quality (higher = better)
            "lap_var_mean": lap_var.mean(),
            "lap_var_p90": np.percentile(lap_var, 90),
            "grad_mag_mean": grad_mag.mean(),
            "hf_energy_mean": hf_energy.mean(),
            # Noise (higher = worse)
            "highlight_mean": highlight.mean(),
            "highlight_max_area": (highlight > 200).mean(),  # fraction of extreme highlights
            "haze_mean": haze.mean(),
            "haze_high_area": (haze > 200).mean(),  # fraction of very hazy regions
            "noise_mean": noise.mean(),
        })
        valid_rows.append(row)

    if not tensors["raw"]:
        continue

    for key in ["raw", "quality_map", "noise_map"]:
        features[key][case_id] = extract_batch(tensors[key])

    # Quality + Noise concatenated features (1280 + 1280 = 2560 dim)
    features["quality_noise_6ch"][case_id] = np.concatenate(
        [features["quality_map"][case_id], features["noise_map"][case_id]], axis=1
    )

    print(f"  Case {case_id}: {len(tensors['raw'])} images")

scalar_df = pd.DataFrame(scalar_stats)

# ============================================================
# 5. Cosine similarity analysis
# ============================================================
def within_patient_sim(feat_dict):
    sims = []
    for case_id, feats in feat_dict.items():
        if len(feats) < 2:
            continue
        sim = cosine_similarity(feats)
        n = len(feats)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        sims.extend(sim[mask])
    return np.array(sims)

def between_patient_sim(feat_dict):
    cases = list(feat_dict.keys())
    sims = []
    for i in range(len(cases)):
        for j in range(i+1, len(cases)):
            fi, fj = feat_dict[cases[i]], feat_dict[cases[j]]
            idx_i = np.random.choice(len(fi), min(20, len(fi)), replace=False)
            idx_j = np.random.choice(len(fj), min(20, len(fj)), replace=False)
            sims.extend(cosine_similarity(fi[idx_i], fj[idx_j]).flatten())
    return np.array(sims)

def quality_discrimination(feat_dict, sample_df, merged_df):
    bb, bo, oo = [], [], []
    for case_id in sorted(feat_dict.keys()):
        case_meta = sample_df[sample_df["case_id"] == case_id].reset_index(drop=True)
        feats = feat_dict[case_id]
        n = len(feats)
        labels = []
        for _, row in case_meta.head(n).iterrows():
            is_b = merged_df[(merged_df["case_id"] == case_id) &
                             (merged_df["image_name"] == row["image_name"])]["is_human_top"].values
            labels.append(bool(is_b[0]) if len(is_b) > 0 else False)
        labels = np.array(labels)
        if labels.sum() == 0:
            continue
        sim = cosine_similarity(feats)
        for i in range(n):
            for j in range(i+1, n):
                s = sim[i, j]
                if labels[i] and labels[j]:
                    bb.append(s)
                elif labels[i] or labels[j]:
                    bo.append(s)
                else:
                    oo.append(s)
    return bb, bo, oo

print("\n" + "=" * 75)
print("PART 1: CNN Feature Similarity Analysis")
print("=" * 75)

for name, feat_dict in [("Raw RGB", features["raw"]),
                         ("Quality Map (LapVar+Grad+HF)", features["quality_map"]),
                         ("Noise Map (Highlight+Haze+Noise)", features["noise_map"]),
                         ("Quality+Noise concat (2560-dim)", features["quality_noise_6ch"])]:
    within = within_patient_sim(feat_dict)
    between = between_patient_sim(feat_dict)
    bb, bo, oo = quality_discrimination(feat_dict, sample, merged)

    gap_id = within.mean() - between.mean()
    gap_qual = np.mean(bb) - np.mean(bo) if bb else 0
    snr = gap_qual / gap_id if gap_id > 0 else 0

    print(f"\n--- {name} ---")
    print(f"  Within-patient sim:  {within.mean():.4f} (std={within.std():.4f})")
    print(f"  Between-patient sim: {between.mean():.4f}")
    print(f"  Patient ID gap:      {gap_id:.4f}")
    if bb:
        print(f"  Best-Best sim:       {np.mean(bb):.4f} (n={len(bb)})")
    print(f"  Best-Other sim:      {np.mean(bo):.4f} (n={len(bo)})")
    print(f"  Other-Other sim:     {np.mean(oo):.4f}")
    print(f"  Quality gap:         {gap_qual:.4f}")
    print(f"  SNR (quality/ID):    {snr:.2f}")

# ============================================================
# 6. Scalar feature correlation with human selection
# ============================================================
print("\n" + "=" * 75)
print("PART 2: Scalar Feature Correlation with Human Selection")
print("=" * 75)

# Per-case normalized (rank within case)
feat_cols = ["lap_var_mean", "lap_var_p90", "grad_mag_mean", "hf_energy_mean",
             "highlight_mean", "highlight_max_area", "haze_mean", "haze_high_area", "noise_mean"]

print(f"\n{'Feature':<22} {'Human Mean':>12} {'Other Mean':>12} {'Diff':>8} {'Direction':>10}")
print("-" * 70)

for col in feat_cols:
    human_vals = scalar_df[scalar_df["is_human_top"] == True][col]
    other_vals = scalar_df[scalar_df["is_human_top"] == False][col]
    h_mean = human_vals.mean()
    o_mean = other_vals.mean()
    diff = h_mean - o_mean
    # Direction interpretation
    if col in ["highlight_mean", "highlight_max_area", "haze_mean", "haze_high_area", "noise_mean"]:
        direction = "LESS noise" if diff < 0 else "MORE noise?!"
    else:
        direction = "SHARPER" if diff > 0 else "BLURRIER?!"
    print(f"{col:<22} {h_mean:>12.2f} {o_mean:>12.2f} {diff:>+8.2f} {direction:>10}")

# ============================================================
# 7. Per-case rank correlation (Spearman)
# ============================================================
from scipy.stats import spearmanr

print("\n" + "=" * 75)
print("PART 3: Per-case Spearman Rank Correlation (scalar feature vs human label)")
print("=" * 75)

print(f"\n{'Feature':<22}", end="")
for c in sorted(sample_cases):
    print(f" {c:>8}", end="")
print(f" {'Mean':>8}")
print("-" * (22 + 9 * (len(sample_cases) + 1)))

for col in feat_cols:
    corrs = []
    print(f"{col:<22}", end="")
    for case_id in sorted(sample_cases):
        case_data = scalar_df[scalar_df["case_id"] == case_id]
        if case_data["is_human_top"].sum() == 0:
            print(f" {'N/A':>8}", end="")
            continue
        rho, _ = spearmanr(case_data[col], case_data["is_human_top"].astype(float))
        corrs.append(rho)
        print(f" {rho:>8.3f}", end="")
    mean_corr = np.mean(corrs) if corrs else 0
    print(f" {mean_corr:>8.3f}")

# ============================================================
# 8. Combined quality-noise scoring (simple test)
# ============================================================
print("\n" + "=" * 75)
print("PART 4: Simple Scoring Test (Top-5 Precision per case)")
print("=" * 75)

# For each case, rank by scalar features and check overlap with human top
def top5_precision(case_data, score_col, ascending=False):
    """What fraction of predicted top-5 are actual human best?"""
    sorted_data = case_data.sort_values(score_col, ascending=ascending)
    top5 = sorted_data.head(5)
    return top5["is_human_top"].sum()  # matches in top-5

# Create composite scores
scalar_df["quality_score"] = (
    scalar_df.groupby("case_id")["lap_var_p90"].transform(lambda x: (x - x.mean()) / (x.std() + 1e-8)) +
    scalar_df.groupby("case_id")["grad_mag_mean"].transform(lambda x: (x - x.mean()) / (x.std() + 1e-8)) +
    scalar_df.groupby("case_id")["hf_energy_mean"].transform(lambda x: (x - x.mean()) / (x.std() + 1e-8))
) / 3

scalar_df["noise_score"] = (
    scalar_df.groupby("case_id")["highlight_mean"].transform(lambda x: (x - x.mean()) / (x.std() + 1e-8)) +
    scalar_df.groupby("case_id")["haze_mean"].transform(lambda x: (x - x.mean()) / (x.std() + 1e-8)) +
    scalar_df.groupby("case_id")["noise_mean"].transform(lambda x: (x - x.mean()) / (x.std() + 1e-8))
) / 3

# Combined: high quality + low noise
scalar_df["combined_score"] = scalar_df["quality_score"] - scalar_df["noise_score"]

score_configs = [
    ("lap_var_p90 (quality)", "lap_var_p90", False),
    ("highlight_mean (noise, asc)", "highlight_mean", True),
    ("haze_mean (noise, asc)", "haze_mean", True),
    ("quality_score (composite)", "quality_score", False),
    ("noise_score (low=good, asc)", "noise_score", True),
    ("combined (quality - noise)", "combined_score", False),
]

print(f"\n{'Method':<35}", end="")
for c in sorted(sample_cases):
    n_best = scalar_df[(scalar_df["case_id"] == c) & (scalar_df["is_human_top"])].shape[0]
    print(f" {c}({n_best})", end="")
print(f" {'Total':>8} {'Prec@5':>8}")
print("-" * (35 + 12 * len(sample_cases) + 18))

for method_name, col, asc in score_configs:
    total_match = 0
    total_best = 0
    print(f"{method_name:<35}", end="")
    for case_id in sorted(sample_cases):
        case_data = scalar_df[scalar_df["case_id"] == case_id]
        n_best = case_data["is_human_top"].sum()
        matches = top5_precision(case_data, col, ascending=asc)
        total_match += matches
        total_best += min(5, n_best)
        print(f" {int(matches):>8}", end="")
    prec = total_match / (5 * len(sample_cases)) * 100
    print(f" {int(total_match):>8} {prec:>7.1f}%")

print("\nDone!")
