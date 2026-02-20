"""Configuration for Image Quality Classifier."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class Config:
    # --- Paths ---
    data_root: Path = Path(r"E:\Multicenter_ROP_study\Multicenter_images\Kubota_selection")
    output_dir: Path = Path(r"checkpoints_b3")

    # --- Class mapping ---
    class_names: List[str] = field(default_factory=lambda: ["Good", "Fair", "Bad", "Worst"])
    class_to_idx: Dict[str, int] = field(
        default_factory=lambda: {"Good": 0, "Fair": 1, "Bad": 2, "Worst": 3}
    )

    # --- Hyperparameters ---
    img_size: int = 300
    batch_size: int = 16
    epochs: int = 30
    lr: float = 5e-5
    weight_decay: float = 1e-3
    dropout: float = 0.4
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.2

    # --- Backbone freeze ---
    freeze_backbone: bool = False
    unfreeze_last_n_blocks: int = 2

    # --- Warmup ---
    warmup_epochs: int = 3

    # --- CV ---
    n_folds: int = 5
    seed: int = 42

    # --- Early stopping ---
    patience: int = 10

    # --- Model ---
    model_name: str = "efficientnet_b3"
    num_classes: int = 4

    # --- Workers ---
    num_workers: int = 0
