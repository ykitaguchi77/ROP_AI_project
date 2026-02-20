"""
Configuration settings for the Pairwise Ranking Model.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Config:
    """Configuration for pairwise ranking model training."""

    # Paths
    base_dir: Path = Path(r'C:\Users\ykita\ROP_AI_project\ROP_project\bestimage_validation')
    results_dir: Path = field(default=None)
    human_excel_path: Path = field(default=None)
    merged_csv_path: Path = field(default=None)
    output_dir: Path = field(default=None)

    # Feature columns
    feature_columns: List[str] = field(default_factory=lambda: [
        'retina_ratio',
        'retina_area',
        'disc_detected',
        'disc_edge_coverage_ratio',
        'disc_area_ratio',
        'disc_center_dist_ratio',
        'disc_pos_ok',
        'mbss_L_multi',
        'mbss_HF_ratio',
        'mbss_Spec_centroid',
        'mbss_Grad_p90',
        'mbss_score',
        'disc_core_score',
        'disc_ring_score',
        'S_mean',
    ])

    # Rule-based score weights (for Top 100 filtering)
    rule_based_weights: dict = field(default_factory=lambda: {
        'retina_ratio': 0.4,
        'mbss_Grad_p90': 0.4,
        'mbss_score': 0.2,
    })

    # Data settings
    top_k_candidates: int = 100  # Top 100 images per video
    num_negatives_per_positive: int = 10  # Negative pairs per positive
    hard_negative_ratio: float = 0.5  # Ratio of hard negatives

    # Model architecture
    input_dim: int = 15
    hidden_dims: List[int] = field(default_factory=lambda: [64, 32, 16])
    dropout: float = 0.3

    # Training settings
    learning_rate: float = 1e-3
    margin: float = 1.0
    batch_size: int = 64
    num_epochs: int = 100
    early_stopping_patience: int = 10
    weight_decay: float = 1e-4

    # Evaluation settings
    top_k_eval: int = 5  # Evaluate Top-5 precision/recall

    # Random seed
    seed: int = 42

    def __post_init__(self):
        """Set derived paths after initialization."""
        if self.results_dir is None:
            self.results_dir = self.base_dir / 'validation_results'
        if self.human_excel_path is None:
            self.human_excel_path = self.base_dir / 'bestimage_human.xlsx'
        if self.merged_csv_path is None:
            self.merged_csv_path = self.results_dir / 'merged_with_disc.csv'
        if self.output_dir is None:
            self.output_dir = self.base_dir / 'ranking_model' / 'outputs'

    def get_feature_dim(self) -> int:
        """Return the number of input features."""
        return len(self.feature_columns)


# Default configuration instance
default_config = Config()
