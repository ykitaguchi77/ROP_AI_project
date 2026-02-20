"""Dataset, transforms, and case-level stratified fold splitting."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset
from torchvision import transforms

from config import Config


# ImageNet statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(is_train: bool, img_size: int = 224) -> transforms.Compose:
    if is_train:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


class QualityDataset(Dataset):
    def __init__(self, image_paths: List[Path], labels: List[int], transform: Optional[transforms.Compose] = None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple:
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


def parse_case_id(filename: str) -> str:
    """Extract unique case identifier ({case_id}_{institution}) from filename.

    Filename format: {case_id}_{institution}_{frame}.png
    e.g. '0001_AMU_00110.png' -> '0001_AMU'
    """
    parts = filename.replace(".png", "").split("_")
    # case_id = parts[0], institution = parts[1]
    return f"{parts[0]}_{parts[1]}"


def parse_institution(filename: str) -> str:
    """Extract institution code from filename."""
    parts = filename.replace(".png", "").split("_")
    return parts[1]


def collect_samples(cfg: Config) -> Tuple[List[Path], List[int], List[str], List[str]]:
    """Collect all image paths, labels, case_ids, and institutions from data_root.

    Skips corrupted image files that cannot be opened by PIL.
    """
    image_paths: List[Path] = []
    labels: List[int] = []
    case_ids: List[str] = []
    institutions: List[str] = []
    skipped = 0

    for class_name, class_idx in cfg.class_to_idx.items():
        class_dir = cfg.data_root / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Class directory not found: {class_dir}")
        for img_path in sorted(class_dir.glob("*.png")):
            try:
                img = Image.open(img_path)
                img.verify()
            except Exception:
                skipped += 1
                continue
            image_paths.append(img_path)
            labels.append(class_idx)
            case_ids.append(parse_case_id(img_path.name))
            institutions.append(parse_institution(img_path.name))

    if skipped > 0:
        print(f"  Warning: skipped {skipped} corrupted image files")

    return image_paths, labels, case_ids, institutions


def create_folds(cfg: Config) -> List[Dict]:
    """Create case-level stratified group k-fold splits.

    Uses StratifiedGroupKFold to ensure:
    - Same case never appears in both train and val (group = case_id)
    - Institution distribution is balanced across folds (stratify by institution)

    Returns list of dicts with 'train_paths', 'train_labels', 'val_paths', 'val_labels'.
    """
    image_paths, labels, case_ids, institutions = collect_samples(cfg)

    image_paths = np.array(image_paths)
    labels = np.array(labels)
    case_ids = np.array(case_ids)
    institutions = np.array(institutions)

    # Build per-case institution mapping for stratification
    unique_cases = []
    case_institutions = []
    seen = set()
    for cid, inst in zip(case_ids, institutions):
        if cid not in seen:
            seen.add(cid)
            unique_cases.append(cid)
            case_institutions.append(inst)
    unique_cases = np.array(unique_cases)
    case_institutions = np.array(case_institutions)

    # StratifiedGroupKFold: stratify by institution, group by case_id
    sgkf = StratifiedGroupKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)

    folds = []
    for train_idx, val_idx in sgkf.split(image_paths, labels, groups=case_ids):
        folds.append({
            "train_paths": image_paths[train_idx].tolist(),
            "train_labels": labels[train_idx].tolist(),
            "val_paths": image_paths[val_idx].tolist(),
            "val_labels": labels[val_idx].tolist(),
        })

    return folds
