"""
Pairwise Ranker Model.

Feature-based MLP that outputs a ranking score for each image.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from .config import Config, default_config


class PairwiseRanker(nn.Module):
    """
    MLP-based pairwise ranking model.

    Architecture:
        Input(15) -> BatchNorm -> FC(64) -> ReLU -> Dropout
                  -> FC(32) -> ReLU -> Dropout
                  -> FC(16) -> ReLU
                  -> FC(1) -> Score
    """

    def __init__(
        self,
        input_dim: int = None,
        hidden_dims: List[int] = None,
        dropout: float = None,
        config: Config = None
    ):
        """
        Initialize the ranker model.

        Args:
            input_dim: Number of input features.
            hidden_dims: List of hidden layer dimensions.
            dropout: Dropout probability.
            config: Configuration object (overrides other args if provided).
        """
        super().__init__()

        config = config or default_config
        self.input_dim = input_dim or config.input_dim
        self.hidden_dims = hidden_dims or config.hidden_dims
        self.dropout_prob = dropout or config.dropout

        # Build network
        layers = []

        # Input batch normalization
        layers.append(nn.BatchNorm1d(self.input_dim))

        # Hidden layers
        prev_dim = self.input_dim
        for i, hidden_dim in enumerate(self.hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())

            # Dropout except for last hidden layer
            if i < len(self.hidden_dims) - 1:
                layers.append(nn.Dropout(self.dropout_prob))

            prev_dim = hidden_dim

        # Output layer (single score)
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute ranking scores.

        Args:
            x: Input features of shape (batch_size, input_dim).

        Returns:
            Ranking scores of shape (batch_size,).
        """
        return self.network(x).squeeze(-1)

    def compute_pairwise_loss(
        self,
        pos_features: torch.Tensor,
        neg_features: torch.Tensor,
        margin: float = 1.0
    ) -> torch.Tensor:
        """
        Compute margin ranking loss for pairwise samples.

        Args:
            pos_features: Positive sample features (batch_size, input_dim).
            neg_features: Negative sample features (batch_size, input_dim).
            margin: Margin for ranking loss.

        Returns:
            Scalar loss value.
        """
        pos_scores = self.forward(pos_features)
        neg_scores = self.forward(neg_features)

        # Margin ranking loss: max(0, margin - (pos_score - neg_score))
        # target = 1 means pos should be ranked higher than neg
        loss = F.margin_ranking_loss(
            pos_scores,
            neg_scores,
            target=torch.ones_like(pos_scores),
            margin=margin
        )

        return loss

    def compute_accuracy(
        self,
        pos_features: torch.Tensor,
        neg_features: torch.Tensor
    ) -> float:
        """
        Compute pairwise accuracy (fraction where pos > neg).

        Args:
            pos_features: Positive sample features.
            neg_features: Negative sample features.

        Returns:
            Accuracy as float.
        """
        with torch.no_grad():
            pos_scores = self.forward(pos_features)
            neg_scores = self.forward(neg_features)
            correct = (pos_scores > neg_scores).float().mean()
        return correct.item()

    def rank_images(
        self,
        features: torch.Tensor,
        return_scores: bool = False
    ) -> torch.Tensor:
        """
        Rank images by predicted scores.

        Args:
            features: Features for all images (num_images, input_dim).
            return_scores: If True, also return scores.

        Returns:
            Indices that would sort images from highest to lowest score.
            If return_scores=True, returns (indices, scores).
        """
        self.eval()
        with torch.no_grad():
            scores = self.forward(features)
            sorted_indices = torch.argsort(scores, descending=True)

        if return_scores:
            return sorted_indices, scores
        return sorted_indices

    def get_top_k(
        self,
        features: torch.Tensor,
        k: int = 5
    ) -> torch.Tensor:
        """
        Get indices of top-k ranked images.

        Args:
            features: Features for all images.
            k: Number of top images to return.

        Returns:
            Indices of top-k images.
        """
        sorted_indices = self.rank_images(features)
        return sorted_indices[:k]


class RankNetLoss(nn.Module):
    """
    RankNet loss function using binary cross-entropy.

    Alternative to margin ranking loss, uses sigmoid probability.
    """

    def __init__(self, sigma: float = 1.0):
        """
        Initialize RankNet loss.

        Args:
            sigma: Temperature parameter for sigmoid.
        """
        super().__init__()
        self.sigma = sigma

    def forward(
        self,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute RankNet loss.

        Args:
            pos_scores: Scores for positive samples.
            neg_scores: Scores for negative samples.

        Returns:
            Scalar loss value.
        """
        # P(i > j) = sigmoid(sigma * (s_i - s_j))
        diff = self.sigma * (pos_scores - neg_scores)
        # Target: P(pos > neg) = 1
        loss = F.binary_cross_entropy_with_logits(
            diff,
            torch.ones_like(diff)
        )
        return loss


class TripletRankingLoss(nn.Module):
    """
    Triplet loss for ranking.

    Uses anchor (positive), positive (another positive), negative triplets.
    """

    def __init__(self, margin: float = 1.0):
        """
        Initialize triplet loss.

        Args:
            margin: Margin for triplet loss.
        """
        super().__init__()
        self.margin = margin

    def forward(
        self,
        anchor_scores: torch.Tensor,
        positive_scores: torch.Tensor,
        negative_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute triplet loss.

        For ranking, we want anchor and positive to be similar,
        and both to be higher than negative.

        Args:
            anchor_scores: Scores for anchor samples.
            positive_scores: Scores for positive samples.
            negative_scores: Scores for negative samples.

        Returns:
            Scalar loss value.
        """
        # Triplet loss: max(0, margin - (pos - neg))
        # Here we use anchor as proxy for positive
        loss = F.triplet_margin_loss(
            anchor_scores.unsqueeze(-1),
            positive_scores.unsqueeze(-1),
            negative_scores.unsqueeze(-1),
            margin=self.margin
        )
        return loss


def create_model(config: Config = None) -> PairwiseRanker:
    """
    Factory function to create a PairwiseRanker model.

    Args:
        config: Configuration object.

    Returns:
        Initialized PairwiseRanker model.
    """
    config = config or default_config
    return PairwiseRanker(
        input_dim=config.input_dim,
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
        config=config
    )
