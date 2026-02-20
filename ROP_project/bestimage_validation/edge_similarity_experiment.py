"""
Edge Detection Similarity Experiment
=====================================
Hypothesis: Edge-transformed images reduce within-patient cosine similarity,
making it possible for CNNs to learn quality differences instead of patient identity.

Compares:
- Raw lens crops → EfficientNet-B0 features → within/between-patient cosine sim
- Edge-transformed images → EfficientNet-B0 features → within/between-patient cosine sim
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
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. Data loading
# ============================================================
BASE = Path(r"C:\Users\ykita\ROP_AI_project\ROP_project\bestimage_validation")
meta = pd.read_csv(BASE / "lens_crops" / "metadata.csv")

# Filter: has_lens=True only (valid lens crops)
meta = meta[meta["has_lens"] == True].reset_index(drop=True)
print(f"Total valid lens crops: {len(meta)}")
print(f"Cases: {sorted(meta['case_id'].unique())}")
print(f"Images per case (mean): {meta.groupby('case_id').size().mean():.0f}")

# Sample: pick 5 cases, up to 50 images each for speed
np.random.seed(42)
sample_cases = np.random.choice(meta["case_id"].unique(), size=min(5, meta["case_id"].nunique()), replace=False)
sample = meta[meta["case_id"].isin(sample_cases)].groupby("case_id").head(50).reset_index(drop=True)
print(f"\nSampled: {len(sample)} images from cases {sorted(sample_cases)}")

# ============================================================
# 2. Edge transform functions
# ============================================================
def load_image_gray(path, size=224):
    img = cv2.imread(str(path))
    if img is None:
        return None
    img = cv2.resize(img, (size, size))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray

def edge_canny(gray):
    """Canny edge detection"""
    edges = cv2.Canny(gray, 50, 150)
    return edges

def edge_sobel(gray):
    """Sobel gradient magnitude"""
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sx**2 + sy**2)
    mag = np.clip(mag / mag.max() * 255, 0, 255).astype(np.uint8)
    return mag

def edge_laplacian(gray):
    """Laplacian (focus measure)"""
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    lap = np.abs(lap)
    lap = np.clip(lap / lap.max() * 255, 0, 255).astype(np.uint8)
    return lap

def edge_multichannel(gray):
    """3-channel edge map: [Canny, Sobel, Laplacian]"""
    c = edge_canny(gray)
    s = edge_sobel(gray)
    l = edge_laplacian(gray)
    return np.stack([c, s, l], axis=-1)  # H x W x 3

# ============================================================
# 3. Feature extraction with EfficientNet-B0
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load EfficientNet-B0 (frozen, feature extractor only)
effnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
effnet.classifier = nn.Identity()  # Output 1280-dim features
effnet = effnet.to(device).eval()

# Transforms
preprocess_rgb = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

preprocess_edge = transforms.Compose([
    transforms.ToTensor(),  # already 224x224
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features_batch(images_tensor):
    """Extract features from a batch of images"""
    with torch.no_grad():
        features = effnet(images_tensor.to(device))
    return features.cpu().numpy()

# ============================================================
# 4. Process all sampled images
# ============================================================
print("\nProcessing images...")

raw_features = {}    # case_id -> [features]
edge_features = {}   # case_id -> [features]  (multi-channel edge)
canny_features = {}  # case_id -> [features]  (canny only, stacked 3x)
sobel_features = {}  # case_id -> [features]  (sobel only, stacked 3x)

for case_id in sorted(sample_cases):
    case_imgs = sample[sample["case_id"] == case_id]
    raw_tensors = []
    edge_tensors = []
    canny_tensors = []
    sobel_tensors = []
    valid_count = 0

    for _, row in case_imgs.iterrows():
        result = load_image_gray(row["crop_path"])
        if result is None:
            continue
        img_bgr, gray = result

        # Raw RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        raw_tensors.append(preprocess_rgb(pil_img))

        # Multi-channel edge: [Canny, Sobel, Laplacian]
        edge_3ch = edge_multichannel(gray)
        edge_tensors.append(preprocess_edge(edge_3ch))

        # Canny only (stacked 3x)
        c = edge_canny(gray)
        canny_3ch = np.stack([c, c, c], axis=-1)
        canny_tensors.append(preprocess_edge(canny_3ch))

        # Sobel only (stacked 3x)
        s = edge_sobel(gray)
        sobel_3ch = np.stack([s, s, s], axis=-1)
        sobel_tensors.append(preprocess_edge(sobel_3ch))

        valid_count += 1

    if valid_count == 0:
        continue

    # Batch extract
    raw_batch = torch.stack(raw_tensors)
    edge_batch = torch.stack(edge_tensors)
    canny_batch = torch.stack(canny_tensors)
    sobel_batch = torch.stack(sobel_tensors)

    raw_features[case_id] = extract_features_batch(raw_batch)
    edge_features[case_id] = extract_features_batch(edge_batch)
    canny_features[case_id] = extract_features_batch(canny_batch)
    sobel_features[case_id] = extract_features_batch(sobel_batch)

    print(f"  Case {case_id}: {valid_count} images processed")

# ============================================================
# 5. Compute cosine similarities
# ============================================================
def compute_within_patient_sim(features_dict):
    """Mean cosine similarity within each patient"""
    sims = []
    for case_id, feats in features_dict.items():
        if len(feats) < 2:
            continue
        sim_matrix = cosine_similarity(feats)
        # Upper triangle (exclude diagonal)
        n = len(feats)
        for i in range(n):
            for j in range(i+1, n):
                sims.append(sim_matrix[i, j])
    return np.array(sims)

def compute_between_patient_sim(features_dict):
    """Mean cosine similarity between different patients"""
    cases = list(features_dict.keys())
    sims = []
    for i in range(len(cases)):
        for j in range(i+1, len(cases)):
            feats_i = features_dict[cases[i]]
            feats_j = features_dict[cases[j]]
            # Sample to limit computation
            idx_i = np.random.choice(len(feats_i), min(20, len(feats_i)), replace=False)
            idx_j = np.random.choice(len(feats_j), min(20, len(feats_j)), replace=False)
            cross_sim = cosine_similarity(feats_i[idx_i], feats_j[idx_j])
            sims.extend(cross_sim.flatten())
    return np.array(sims)

print("\n" + "="*70)
print("RESULTS: Within-patient vs Between-patient Cosine Similarity")
print("="*70)

for name, feat_dict in [("Raw RGB", raw_features),
                         ("Edge Multi-ch", edge_features),
                         ("Canny only", canny_features),
                         ("Sobel only", sobel_features)]:
    within = compute_within_patient_sim(feat_dict)
    between = compute_between_patient_sim(feat_dict)

    print(f"\n--- {name} ---")
    print(f"  Within-patient:  mean={within.mean():.4f}, std={within.std():.4f}, "
          f"min={within.min():.4f}, max={within.max():.4f}")
    print(f"  Between-patient: mean={between.mean():.4f}, std={between.std():.4f}, "
          f"min={between.min():.4f}, max={between.max():.4f}")
    print(f"  Gap (within - between): {within.mean() - between.mean():.4f}")
    print(f"  Discrimination ratio:   {(within.mean() - between.mean()) / within.std():.2f} sigma")

# ============================================================
# 6. Per-case breakdown
# ============================================================
print("\n" + "="*70)
print("Per-case within-patient similarity")
print("="*70)
print(f"{'Case':>8} {'Raw':>8} {'Edge':>8} {'Canny':>8} {'Sobel':>8} {'Drop(Raw-Edge)':>14}")
print("-" * 60)

for case_id in sorted(raw_features.keys()):
    raw_sim = cosine_similarity(raw_features[case_id])
    edge_sim = cosine_similarity(edge_features[case_id])
    canny_sim = cosine_similarity(canny_features[case_id])
    sobel_sim = cosine_similarity(sobel_features[case_id])

    n = len(raw_features[case_id])
    mask = np.triu(np.ones((n,n), dtype=bool), k=1)

    r = raw_sim[mask].mean()
    e = edge_sim[mask].mean()
    c = canny_sim[mask].mean()
    s = sobel_sim[mask].mean()

    print(f"{case_id:>8} {r:>8.4f} {e:>8.4f} {c:>8.4f} {s:>8.4f} {r-e:>14.4f}")

# ============================================================
# 7. Quality discrimination check
# ============================================================
print("\n" + "="*70)
print("Quality Discrimination: Human-best vs Others (within each case)")
print("="*70)

# Load human labels
merged = pd.read_csv(BASE / "validation_results" / "merged_with_disc.csv")

for name, feat_dict in [("Raw RGB", raw_features),
                         ("Edge Multi-ch", edge_features),
                         ("Sobel only", sobel_features)]:
    print(f"\n--- {name} ---")
    best_vs_best = []
    best_vs_other = []
    other_vs_other = []

    for case_id in sorted(feat_dict.keys()):
        case_meta = sample[sample["case_id"] == case_id].reset_index(drop=True)
        case_merged = merged[merged["case_id"] == case_id]

        # Get human_top labels for sampled images
        feats = feat_dict[case_id]
        n = len(feats)

        # Match image names
        labels = []
        for _, row in case_meta.iterrows():
            is_best = case_merged[case_merged["image_name"] == row["image_name"]]["is_human_top"].values
            if len(is_best) > 0 and is_best[0]:
                labels.append(True)
            else:
                labels.append(False)
        labels = np.array(labels[:n])

        if labels.sum() == 0 or labels.sum() == n:
            continue

        sim_matrix = cosine_similarity(feats)

        for i in range(n):
            for j in range(i+1, n):
                s = sim_matrix[i, j]
                if labels[i] and labels[j]:
                    best_vs_best.append(s)
                elif labels[i] or labels[j]:
                    best_vs_other.append(s)
                else:
                    other_vs_other.append(s)

    if best_vs_best:
        print(f"  Best vs Best:   {np.mean(best_vs_best):.4f} (n={len(best_vs_best)})")
    print(f"  Best vs Other:  {np.mean(best_vs_other):.4f} (n={len(best_vs_other)})")
    print(f"  Other vs Other: {np.mean(other_vs_other):.4f} (n={len(other_vs_other)})")
    if best_vs_best:
        print(f"  Quality gap (best_best - best_other): {np.mean(best_vs_best) - np.mean(best_vs_other):.4f}")

print("\nDone!")
