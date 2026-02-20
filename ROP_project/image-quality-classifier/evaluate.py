"""Evaluation script: confusion matrix, per-class metrics, ordinal accuracy.

Usage:
    python evaluate.py                  # Evaluate all folds
    python evaluate.py --fold 0         # Evaluate single fold
"""

import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader

from config import Config
from dataset import QualityDataset, collect_samples, create_folds, get_transforms, parse_institution
from model import create_model


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_labels)


def ordinal_accuracy(preds, labels, tolerance=1):
    """Fraction of predictions within `tolerance` classes of the true label."""
    return np.mean(np.abs(preds - labels) <= tolerance)


def plot_confusion_matrix(cm, class_names, title, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True",
        xlabel="Predicted",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def evaluate_fold(fold_idx: int, fold_data: dict, cfg: Config, device: torch.device):
    checkpoint_path = Path(cfg.output_dir) / f"fold{fold_idx}_best.pth"
    if not checkpoint_path.exists():
        print(f"  Fold {fold_idx}: checkpoint not found ({checkpoint_path})")
        return None

    # Load model
    model = create_model(cfg.model_name, cfg.num_classes, cfg.dropout).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])

    # Val dataset
    val_ds = QualityDataset(
        fold_data["val_paths"], fold_data["val_labels"],
        transform=get_transforms(is_train=False, img_size=cfg.img_size),
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )

    preds, labels = predict(model, val_loader, device)

    return {
        "preds": preds,
        "labels": labels,
        "val_paths": fold_data["val_paths"],
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Image Quality Classifier")
    parser.add_argument("--fold", type=str, default="all", help="Fold index or 'all'")
    parser.add_argument("--model_name", type=str, default=None, help="timm model name")
    parser.add_argument("--img_size", type=int, default=None, help="Input image size")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    cfg = Config()
    if args.model_name is not None:
        cfg.model_name = args.model_name
    if args.img_size is not None:
        cfg.img_size = args.img_size
    if args.output_dir is not None:
        cfg.output_dir = Path(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Creating folds...")
    folds = create_folds(cfg)

    if args.fold == "all":
        fold_indices = list(range(cfg.n_folds))
    else:
        fold_indices = [int(args.fold)]

    # Collect predictions across folds
    all_preds = []
    all_labels = []
    all_paths = []

    for fi in fold_indices:
        print(f"\nEvaluating fold {fi}...")
        result = evaluate_fold(fi, folds[fi], cfg, device)
        if result is None:
            continue

        preds, labels = result["preds"], result["labels"]
        all_preds.append(preds)
        all_labels.append(labels)
        all_paths.extend(result["val_paths"])

        # Per-fold confusion matrix
        cm = confusion_matrix(labels, preds, labels=list(range(cfg.num_classes)))
        plot_confusion_matrix(
            cm, cfg.class_names,
            f"Fold {fi} Confusion Matrix",
            Path(cfg.output_dir) / f"fold{fi}_confusion_matrix.png",
        )

        print(f"  Fold {fi} accuracy: {np.mean(preds == labels):.4f}")
        print(f"  Fold {fi} ordinal accuracy (+-1): {ordinal_accuracy(preds, labels):.4f}")

    if not all_preds:
        print("No folds evaluated.")
        return

    # Aggregate
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    print(f"\n{'='*60}")
    print("  Aggregated Results")
    print(f"{'='*60}")

    # Overall metrics
    acc = np.mean(all_preds == all_labels)
    ord_acc = ordinal_accuracy(all_preds, all_labels)
    wf1 = f1_score(all_labels, all_preds, average="weighted")

    print(f"  Overall Accuracy:       {acc:.4f}")
    print(f"  Ordinal Accuracy (+-1): {ord_acc:.4f}")
    print(f"  Weighted F1:            {wf1:.4f}")

    # Per-class report
    print("\n  Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=cfg.class_names, digits=4))

    # Aggregated confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(cfg.num_classes)))
    plot_confusion_matrix(
        cm, cfg.class_names,
        "Aggregated Confusion Matrix (all folds)",
        Path(cfg.output_dir) / "aggregated_confusion_matrix.png",
    )

    # Institution-level accuracy
    print("\n  Per-institution accuracy:")
    inst_list = [parse_institution(Path(p).name) for p in all_paths]
    df = pd.DataFrame({"institution": inst_list, "pred": all_preds, "label": all_labels})
    for inst, group in df.groupby("institution"):
        inst_acc = np.mean(group["pred"] == group["label"])
        inst_ord = ordinal_accuracy(group["pred"].values, group["label"].values)
        print(f"    {inst:5s}: acc={inst_acc:.4f}, ordinal_acc={inst_ord:.4f} (n={len(group)})")

    # Save per-image results
    results_df = pd.DataFrame({
        "image_path": [str(p) for p in all_paths],
        "institution": inst_list,
        "true_label": all_labels,
        "pred_label": all_preds,
        "true_name": [cfg.class_names[l] for l in all_labels],
        "pred_name": [cfg.class_names[p] for p in all_preds],
        "correct": all_preds == all_labels,
    })
    results_path = Path(cfg.output_dir) / "evaluation_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n  Per-image results saved: {results_path}")


if __name__ == "__main__":
    main()
