"""
Evaluation metrics for Pairwise Ranker Model.
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from .config import Config, default_config
from .model import PairwiseRanker


class Evaluator:
    """
    Evaluator class for ranking model.

    Computes metrics like Top-K precision, NDCG, MRR.
    """

    def __init__(
        self,
        model: PairwiseRanker,
        config: Config = None,
        device: str = None
    ):
        """
        Initialize evaluator.

        Args:
            model: Trained PairwiseRanker model.
            config: Configuration object.
            device: Device for inference.
        """
        self.model = model
        self.config = config or default_config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()

    def predict_scores(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict ranking scores for all images in DataFrame.

        Args:
            df: DataFrame with feature columns.

        Returns:
            Array of ranking scores.
        """
        features = self._extract_features(df)
        features_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            scores = self.model(features_tensor).cpu().numpy()

        return scores

    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract feature matrix from DataFrame."""
        features = []
        for col in self.config.feature_columns:
            if col in df.columns:
                values = df[col].values
                # Handle missing values
                if col in ['disc_detected', 'disc_pos_ok']:
                    values = np.where(pd.isna(values), 0, values).astype(float)
                else:
                    median_val = np.nanmedian(values) if np.any(~np.isnan(values)) else 0
                    values = np.where(pd.isna(values), median_val, values)
            else:
                values = np.zeros(len(df))
            features.append(values)

        return np.stack(features, axis=1).astype(np.float32)

    def get_top_k_predictions(
        self,
        df: pd.DataFrame,
        k: int = None
    ) -> List[str]:
        """
        Get top-k predicted images.

        Args:
            df: DataFrame with images and features.
            k: Number of top images to return.

        Returns:
            List of top-k image names.
        """
        k = k or self.config.top_k_eval
        scores = self.predict_scores(df)
        df_with_scores = df.copy()
        df_with_scores['predicted_score'] = scores

        top_k = df_with_scores.nlargest(k, 'predicted_score')
        return top_k['image_name'].tolist()

    def evaluate_case(
        self,
        df: pd.DataFrame,
        case_id: int,
        k: int = None
    ) -> Dict[str, float]:
        """
        Evaluate ranking performance on a single video case.

        Args:
            df: DataFrame with images, features, and is_human_top labels.
            case_id: Video case ID.
            k: Top-k for evaluation.

        Returns:
            Dictionary with evaluation metrics.
        """
        k = k or self.config.top_k_eval

        # Predict scores
        scores = self.predict_scores(df)
        df_eval = df.copy()
        df_eval['predicted_score'] = scores

        # Get AI top-k
        ai_top_k = df_eval.nlargest(k, 'predicted_score')['image_name'].tolist()

        # Get human selections
        human_top = df_eval[df_eval['is_human_top'] == True]['image_name'].tolist()

        # Compute metrics
        matches = len(set(ai_top_k) & set(human_top))

        metrics = {
            'case_id': case_id,
            'top5_precision': matches / k if k > 0 else 0,
            'top5_recall': matches / len(human_top) if len(human_top) > 0 else 0,
            'video_match': 1 if matches > 0 else 0,
            'num_matches': matches,
            'num_human_selections': len(human_top),
            'ai_top_k': ai_top_k,
            'human_top': human_top,
        }

        # NDCG@k
        metrics['ndcg_at_5'] = self._compute_ndcg(df_eval, k)

        # MRR
        metrics['mrr'] = self._compute_mrr(df_eval)

        # Pairwise accuracy
        metrics['pairwise_accuracy'] = self._compute_pairwise_accuracy(df_eval)

        return metrics

    def _compute_ndcg(self, df: pd.DataFrame, k: int) -> float:
        """
        Compute Normalized Discounted Cumulative Gain at k.

        Args:
            df: DataFrame with predicted_score and is_human_top.
            k: Cutoff position.

        Returns:
            NDCG@k score.
        """
        # Sort by predicted score
        sorted_df = df.sort_values('predicted_score', ascending=False)

        # Get relevance (1 if human-selected, 0 otherwise)
        relevance = sorted_df['is_human_top'].astype(int).values[:k]

        # DCG
        dcg = 0.0
        for i, rel in enumerate(relevance):
            dcg += (2 ** rel - 1) / np.log2(i + 2)

        # Ideal DCG (all positives at top)
        ideal_relevance = sorted(df['is_human_top'].astype(int).values, reverse=True)[:k]
        idcg = 0.0
        for i, rel in enumerate(ideal_relevance):
            idcg += (2 ** rel - 1) / np.log2(i + 2)

        return dcg / idcg if idcg > 0 else 0.0

    def _compute_mrr(self, df: pd.DataFrame) -> float:
        """
        Compute Mean Reciprocal Rank.

        Args:
            df: DataFrame with predicted_score and is_human_top.

        Returns:
            MRR score (reciprocal of rank of first relevant item).
        """
        sorted_df = df.sort_values('predicted_score', ascending=False).reset_index(drop=True)

        for rank, row in enumerate(sorted_df.itertuples(), 1):
            if row.is_human_top:
                return 1.0 / rank

        return 0.0

    def _compute_pairwise_accuracy(self, df: pd.DataFrame) -> float:
        """
        Compute pairwise accuracy on test data.

        Args:
            df: DataFrame with predicted_score and is_human_top.

        Returns:
            Fraction of pairs where positive is ranked higher than negative.
        """
        positives = df[df['is_human_top'] == True]
        negatives = df[df['is_human_top'] == False]

        if len(positives) == 0 or len(negatives) == 0:
            return 0.0

        correct = 0
        total = 0

        for _, pos_row in positives.iterrows():
            for _, neg_row in negatives.iterrows():
                if pos_row['predicted_score'] > neg_row['predicted_score']:
                    correct += 1
                total += 1

        return correct / total if total > 0 else 0.0

    def evaluate_all_cases(
        self,
        dataset,
        k: int = None
    ) -> Dict[str, List]:
        """
        Evaluate on all video cases.

        Args:
            dataset: ROPRankingDataset instance.
            k: Top-k for evaluation.

        Returns:
            Dictionary with results for all cases.
        """
        results = {
            'case_id': [],
            'top5_precision': [],
            'top5_recall': [],
            'video_match': [],
            'ndcg_at_5': [],
            'mrr': [],
            'pairwise_accuracy': [],
        }

        for case_id in dataset.case_ids:
            case_df = dataset.get_case_data(case_id)
            metrics = self.evaluate_case(case_df, case_id, k)

            results['case_id'].append(case_id)
            for key in ['top5_precision', 'top5_recall', 'video_match', 'ndcg_at_5', 'mrr', 'pairwise_accuracy']:
                results[key].append(metrics.get(key, 0))

        return results

    def print_summary(self, results: Dict[str, List]):
        """
        Print summary statistics of evaluation results.

        Args:
            results: Dictionary with evaluation results.
        """
        print("=" * 60)
        print("Evaluation Summary")
        print("=" * 60)

        print(f"Number of cases: {len(results['case_id'])}")
        print()

        metrics = [
            ('Top-5 Precision', 'top5_precision'),
            ('Top-5 Recall', 'top5_recall'),
            ('Video Match Rate', 'video_match'),
            ('NDCG@5', 'ndcg_at_5'),
            ('MRR', 'mrr'),
            ('Pairwise Accuracy', 'pairwise_accuracy'),
        ]

        for name, key in metrics:
            values = results[key]
            print(f"{name}:")
            print(f"  Mean: {np.mean(values):.4f}")
            print(f"  Std:  {np.std(values):.4f}")
            print(f"  Min:  {np.min(values):.4f}")
            print(f"  Max:  {np.max(values):.4f}")
            print()

    def compare_with_baseline(
        self,
        results: Dict[str, List],
        baseline_precision: float = 0.545
    ):
        """
        Compare results with rule-based baseline.

        Args:
            results: Dictionary with evaluation results.
            baseline_precision: Baseline Top-5 precision (default: 54.5%).
        """
        print("=" * 60)
        print("Comparison with Baseline")
        print("=" * 60)

        model_precision = np.mean(results['top5_precision'])
        improvement = (model_precision - baseline_precision) / baseline_precision * 100

        print(f"Baseline Top-5 Precision: {baseline_precision:.4f}")
        print(f"Model Top-5 Precision:    {model_precision:.4f}")
        print(f"Improvement:              {improvement:+.1f}%")
        print()

        if model_precision > baseline_precision:
            print("Model OUTPERFORMS baseline")
        elif model_precision < baseline_precision:
            print("Model UNDERPERFORMS baseline")
        else:
            print("Model MATCHES baseline")


def create_comparison_report(
    results: Dict[str, List],
    output_path: Path = None
) -> pd.DataFrame:
    """
    Create a detailed comparison report.

    Args:
        results: Dictionary with evaluation results.
        output_path: Optional path to save CSV.

    Returns:
        DataFrame with per-case results.
    """
    df = pd.DataFrame(results)

    # Add summary row
    summary = pd.DataFrame([{
        'case_id': 'MEAN',
        'top5_precision': df['top5_precision'].mean(),
        'top5_recall': df['top5_recall'].mean(),
        'video_match': df['video_match'].mean(),
        'ndcg_at_5': df['ndcg_at_5'].mean(),
        'mrr': df['mrr'].mean(),
        'pairwise_accuracy': df['pairwise_accuracy'].mean(),
    }])

    df = pd.concat([df, summary], ignore_index=True)

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Report saved to {output_path}")

    return df
