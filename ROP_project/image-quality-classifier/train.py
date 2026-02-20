"""Training script for Image Quality Classifier.

Usage:
    python train.py --fold 0          # Train single fold
    python train.py --fold all        # Train all 5 folds
    python train.py --fold 0 --epochs 10 --lr 5e-5
"""

import argparse
import csv
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from config import Config
from dataset import QualityDataset, create_folds, get_transforms
from model import create_model


def compute_class_weights(labels: list, num_classes: int) -> torch.Tensor:
    """Compute inverse-frequency class weights: w[c] = N / (C * N_c)."""
    counts = Counter(labels)
    n_total = len(labels)
    weights = []
    for c in range(num_classes):
        nc = counts.get(c, 1)
        weights.append(n_total / (num_classes * nc))
    return torch.tensor(weights, dtype=torch.float32)


def cosine_warmup_scheduler(optimizer, warmup_epochs, total_epochs, steps_per_epoch):
    """Cosine annealing with linear warmup."""
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def mixup_data(x, y, alpha=0.2):
    """Apply Mixup augmentation."""
    if alpha <= 0:
        return x, y, y, 0.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute Mixup loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(model, loader, criterion, optimizer, device, scheduler=None, mixup_alpha=0.0):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        # Mixup
        if mixup_alpha > 0:
            images, targets_a, targets_b, lam = mixup_data(images, labels, mixup_alpha)
            optimizer.zero_grad()
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def train_fold(fold_idx: int, fold_data: dict, cfg: Config, device: torch.device):
    print(f"\n{'='*60}")
    print(f"  Fold {fold_idx}")
    print(f"{'='*60}")

    # Datasets
    train_ds = QualityDataset(
        fold_data["train_paths"], fold_data["train_labels"],
        transform=get_transforms(is_train=True, img_size=cfg.img_size),
    )
    val_ds = QualityDataset(
        fold_data["val_paths"], fold_data["val_labels"],
        transform=get_transforms(is_train=False, img_size=cfg.img_size),
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )

    print(f"  Train: {len(train_ds)} images | Val: {len(val_ds)} images")
    train_counts = Counter(fold_data["train_labels"])
    val_counts = Counter(fold_data["val_labels"])
    for c, name in enumerate(cfg.class_names):
        print(f"    {name}: train={train_counts.get(c, 0)}, val={val_counts.get(c, 0)}")

    # Model
    model = create_model(cfg.model_name, cfg.num_classes, cfg.dropout).to(device)

    # Freeze backbone if configured
    if cfg.freeze_backbone:
        model.freeze_backbone(cfg.unfreeze_last_n_blocks)

    # Class weights & loss
    class_weights = compute_class_weights(fold_data["train_labels"], cfg.num_classes).to(device)
    print(f"  Class weights: {[f'{w:.3f}' for w in class_weights.tolist()]}")
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=cfg.label_smoothing)

    # Optimizer (only trainable params)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Scheduler with warmup
    steps_per_epoch = len(train_loader)
    scheduler = cosine_warmup_scheduler(optimizer, cfg.warmup_epochs, cfg.epochs, steps_per_epoch)

    print(f"  Regularization: label_smoothing={cfg.label_smoothing}, mixup_alpha={cfg.mixup_alpha}, "
          f"weight_decay={cfg.weight_decay}, dropout={cfg.dropout}")

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = -1
    log_rows = []

    checkpoint_path = Path(cfg.output_dir) / f"fold{fold_idx}_best.pth"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # Validation uses no label smoothing
    val_criterion = nn.CrossEntropyLoss(weight=class_weights)

    for epoch in range(cfg.epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            scheduler=scheduler, mixup_alpha=cfg.mixup_alpha,
        )
        val_loss, val_acc = validate(model, val_loader, val_criterion, device)

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"  Epoch {epoch+1:02d}/{cfg.epochs} "
            f"| train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"| val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"| lr={lr_now:.2e} | {elapsed:.1f}s",
            end="",
        )

        log_rows.append({
            "fold": fold_idx, "epoch": epoch + 1,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc, "lr": lr_now,
        })

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
            }, checkpoint_path)
            print(" *BEST*")
        else:
            patience_counter += 1
            print()
            if patience_counter >= cfg.patience:
                print(f"  Early stopping at epoch {epoch+1} (patience={cfg.patience})")
                break

    print(f"  Best epoch: {best_epoch}, val_loss: {best_val_loss:.4f}")

    # Save training log
    log_path = Path(cfg.output_dir) / f"fold{fold_idx}_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)

    return best_val_loss, log_rows[-1]["val_acc"] if log_rows else 0.0


def main():
    parser = argparse.ArgumentParser(description="Train Image Quality Classifier")
    parser.add_argument("--fold", type=str, default="all", help="Fold index (0-4) or 'all'")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--model_name", type=str, default=None, help="timm model name")
    parser.add_argument("--img_size", type=int, default=None, help="Input image size")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    cfg = Config()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.lr = args.lr
    if args.model_name is not None:
        cfg.model_name = args.model_name
    if args.img_size is not None:
        cfg.img_size = args.img_size
    if args.output_dir is not None:
        cfg.output_dir = Path(args.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Create folds
    print("Creating folds...")
    folds = create_folds(cfg)
    print(f"  {len(folds)} folds created")

    # Train
    if args.fold == "all":
        fold_indices = list(range(cfg.n_folds))
    else:
        fold_indices = [int(args.fold)]

    results = []
    for fi in fold_indices:
        val_loss, val_acc = train_fold(fi, folds[fi], cfg, device)
        results.append({"fold": fi, "val_loss": val_loss, "val_acc": val_acc})

    # Summary
    if len(results) > 1:
        print(f"\n{'='*60}")
        print("  Summary (all folds)")
        print(f"{'='*60}")
        losses = [r["val_loss"] for r in results]
        accs = [r["val_acc"] for r in results]
        for r in results:
            print(f"  Fold {r['fold']}: val_loss={r['val_loss']:.4f}, val_acc={r['val_acc']:.4f}")
        print(f"  Mean val_loss: {np.mean(losses):.4f} +/- {np.std(losses):.4f}")
        print(f"  Mean val_acc:  {np.mean(accs):.4f} +/- {np.std(accs):.4f}")


if __name__ == "__main__":
    main()
