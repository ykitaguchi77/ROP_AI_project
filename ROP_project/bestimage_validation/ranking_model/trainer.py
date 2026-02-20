"""
Training loop for Pairwise Ranker Model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
from tqdm import tqdm

from .config import Config, default_config
from .model import PairwiseRanker, create_model
from .dataset import ROPRankingDataset, PairwiseDataLoader


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait before stopping.
            min_delta: Minimum change to qualify as improvement.
            mode: 'min' for loss, 'max' for metrics like accuracy.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.should_stop = False

    def __call__(self, value: float) -> bool:
        """
        Check if training should stop.

        Args:
            value: Current metric value.

        Returns:
            True if training should stop.
        """
        if self.best_value is None:
            self.best_value = value
            return False

        if self.mode == 'min':
            improved = value < self.best_value - self.min_delta
        else:
            improved = value > self.best_value + self.min_delta

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True

        return False


class Trainer:
    """
    Trainer class for Pairwise Ranker.

    Handles training loop, validation, and checkpointing.
    """

    def __init__(
        self,
        model: PairwiseRanker = None,
        config: Config = None,
        device: str = None
    ):
        """
        Initialize trainer.

        Args:
            model: PairwiseRanker model. Creates new one if None.
            config: Configuration object.
            device: Device to use ('cuda' or 'cpu').
        """
        self.config = config or default_config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = model or create_model(self.config)
        self.model = self.model.to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )

        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }

    def train_epoch(
        self,
        train_loader: DataLoader,
        margin: float = None
    ) -> Tuple[float, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader.
            margin: Margin for ranking loss.

        Returns:
            Tuple of (average loss, pairwise accuracy).
        """
        margin = margin or self.config.margin
        self.model.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch in train_loader:
            pos_features = batch['pos_features'].to(self.device)
            neg_features = batch['neg_features'].to(self.device)

            self.optimizer.zero_grad()

            loss = self.model.compute_pairwise_loss(pos_features, neg_features, margin)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item() * pos_features.size(0)

            # Compute accuracy
            with torch.no_grad():
                pos_scores = self.model(pos_features)
                neg_scores = self.model(neg_features)
                correct = (pos_scores > neg_scores).sum().item()
                total_correct += correct
                total_samples += pos_features.size(0)

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        return avg_loss, accuracy

    def validate(
        self,
        val_loader: DataLoader,
        margin: float = None
    ) -> Tuple[float, float]:
        """
        Validate the model.

        Args:
            val_loader: Validation data loader.
            margin: Margin for ranking loss.

        Returns:
            Tuple of (average loss, pairwise accuracy).
        """
        margin = margin or self.config.margin
        self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                pos_features = batch['pos_features'].to(self.device)
                neg_features = batch['neg_features'].to(self.device)

                loss = self.model.compute_pairwise_loss(pos_features, neg_features, margin)
                total_loss += loss.item() * pos_features.size(0)

                pos_scores = self.model(pos_features)
                neg_scores = self.model(neg_features)
                correct = (pos_scores > neg_scores).sum().item()
                total_correct += correct
                total_samples += pos_features.size(0)

        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        accuracy = total_correct / total_samples if total_samples > 0 else 0

        return avg_loss, accuracy

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        num_epochs: int = None,
        early_stopping_patience: int = None,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the model.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader (optional).
            num_epochs: Number of epochs to train.
            early_stopping_patience: Patience for early stopping.
            verbose: Whether to print progress.

        Returns:
            Training history dictionary.
        """
        num_epochs = num_epochs or self.config.num_epochs
        patience = early_stopping_patience or self.config.early_stopping_patience

        early_stopper = EarlyStopping(patience=patience, mode='min')
        best_loss = float('inf')
        best_state = None

        epoch_iter = tqdm(range(num_epochs), desc='Training') if verbose else range(num_epochs)

        for epoch in epoch_iter:
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)

            # Validate
            if val_loader is not None:
                val_loss, val_acc = self.validate(val_loader)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                monitor_loss = val_loss
            else:
                monitor_loss = train_loss

            # Learning rate scheduling
            self.scheduler.step(monitor_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['lr'].append(current_lr)

            # Save best model
            if monitor_loss < best_loss:
                best_loss = monitor_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

            # Early stopping
            if early_stopper(monitor_loss):
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                break

            # Progress update
            if verbose:
                desc = f"Epoch {epoch + 1}: Loss={train_loss:.4f}, Acc={train_acc:.4f}"
                if val_loader is not None:
                    desc += f", Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}"
                if hasattr(epoch_iter, 'set_description'):
                    epoch_iter.set_description(desc)

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self.history

    def save_checkpoint(self, path: Path):
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': {
                'input_dim': self.config.input_dim,
                'hidden_dims': self.config.hidden_dims,
                'dropout': self.config.dropout,
            },
            'history': self.history,
        }

        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: Path):
        """
        Load model checkpoint.

        Args:
            path: Path to checkpoint file.
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'history' in checkpoint:
            self.history = checkpoint['history']

        print(f"Checkpoint loaded from {path}")


def run_lovo_cv(
    dataset: ROPRankingDataset,
    config: Config = None,
    verbose: bool = True,
    num_folds: int = None
) -> Dict[str, List]:
    """
    Run Leave-One-Video-Out cross-validation.

    Args:
        dataset: ROPRankingDataset instance.
        config: Configuration object.
        verbose: Whether to print progress.
        num_folds: Number of folds to run. If None, runs all folds.
                   Use num_folds=1 for quick pilot testing.

    Returns:
        Dictionary with CV results for each fold.
    """
    from .evaluator import Evaluator

    config = config or default_config
    data_loader = PairwiseDataLoader(dataset, config)

    results = {
        'case_id': [],
        'top5_precision': [],
        'top5_recall': [],
        'video_match': [],
        'ndcg_at_5': [],
        'mrr': [],
        'pairwise_acc': [],
    }

    # Select case IDs to evaluate
    case_ids_to_eval = dataset.case_ids
    if num_folds is not None and num_folds < len(case_ids_to_eval):
        case_ids_to_eval = case_ids_to_eval[:num_folds]
        if verbose:
            print(f"Pilot mode: Running {num_folds} fold(s) out of {len(dataset.case_ids)}")

    case_iter = tqdm(case_ids_to_eval, desc='LOVO CV') if verbose else case_ids_to_eval

    for test_case_id in case_iter:
        # Get train/test split
        train_loader, test_df = data_loader.get_lovo_split(test_case_id)

        # Create and train model
        model = create_model(config)
        trainer = Trainer(model, config)
        trainer.fit(train_loader, num_epochs=config.num_epochs, verbose=False)

        # Evaluate
        evaluator = Evaluator(model, config)
        metrics = evaluator.evaluate_case(test_df, test_case_id)

        # Store results
        results['case_id'].append(test_case_id)
        for key in ['top5_precision', 'top5_recall', 'video_match', 'ndcg_at_5', 'mrr']:
            results[key].append(metrics.get(key, 0))

        # Pairwise accuracy on test
        pairwise_acc = metrics.get('pairwise_accuracy', 0)
        results['pairwise_acc'].append(pairwise_acc)

        if verbose:
            desc = f"Case {test_case_id}: Prec={metrics.get('top5_precision', 0):.2f}"
            if hasattr(case_iter, 'set_description'):
                case_iter.set_description(desc)

    # Compute summary statistics
    if verbose:
        print("\n" + "=" * 60)
        if num_folds is not None and num_folds < len(dataset.case_ids):
            print(f"Pilot Test Results ({num_folds} fold(s))")
        else:
            print("LOVO Cross-Validation Results")
        print("=" * 60)
        print(f"Mean Top-5 Precision: {np.mean(results['top5_precision']):.4f}")
        print(f"Mean Top-5 Recall: {np.mean(results['top5_recall']):.4f}")
        print(f"Video Match Rate: {np.mean(results['video_match']):.4f}")
        print(f"Mean NDCG@5: {np.mean(results['ndcg_at_5']):.4f}")
        print(f"Mean MRR: {np.mean(results['mrr']):.4f}")
        print(f"Mean Pairwise Acc: {np.mean(results['pairwise_acc']):.4f}")

    return results


def run_pilot_test(
    dataset: ROPRankingDataset,
    config: Config = None,
    num_folds: int = 1,
    num_epochs: int = 20,
    verbose: bool = True
) -> Dict[str, List]:
    """
    Run a quick pilot test with reduced epochs and folds.

    This is useful for verifying the training pipeline works
    before running the full cross-validation.

    Args:
        dataset: ROPRankingDataset instance.
        config: Configuration object.
        num_folds: Number of folds to run (default: 1).
        num_epochs: Number of epochs per fold (default: 20).
        verbose: Whether to print progress.

    Returns:
        Dictionary with pilot test results.

    Example:
        >>> dataset = ROPRankingDataset(config)
        >>> dataset.load_data()
        >>> results = run_pilot_test(dataset, num_folds=1, num_epochs=20)
    """
    # Create a modified config with fewer epochs
    if config is None:
        config = default_config

    # Temporarily override num_epochs
    original_epochs = config.num_epochs
    config.num_epochs = num_epochs

    if verbose:
        print("=" * 60)
        print("PILOT TEST MODE")
        print("=" * 60)
        print(f"Folds: {num_folds}")
        print(f"Epochs per fold: {num_epochs}")
        print()

    try:
        results = run_lovo_cv(
            dataset,
            config=config,
            verbose=verbose,
            num_folds=num_folds
        )
    finally:
        # Restore original config
        config.num_epochs = original_epochs

    return results
