"""
Dataset classes for Pairwise Ranking Model.

Handles data loading, preprocessing, and pairwise sample generation.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from .config import Config, default_config


class ROPRankingDataset:
    """
    Dataset class for ROP best image ranking.

    Loads features from CSV, human selections from Excel, and generates
    pairwise training samples.
    """

    def __init__(self, config: Config = None):
        """
        Initialize the dataset.

        Args:
            config: Configuration object. Uses default_config if None.
        """
        self.config = config or default_config
        self.df = None
        self.human_selections = {}
        self.case_ids = []
        self.scalers = {}  # Per-case scalers

    def load_data(self) -> pd.DataFrame:
        """
        Load and merge feature data with human selections.

        Returns:
            DataFrame with all features and is_human_top labels.
        """
        # Load merged CSV with features
        self.df = pd.read_csv(self.config.merged_csv_path)

        # Load human selections from Excel
        self._load_human_selections()

        # Mark human selections in dataframe
        self._mark_human_selections()

        # Get unique case IDs
        self.case_ids = sorted(self.df['case_id'].unique().tolist())

        print(f"Loaded {len(self.df)} images from {len(self.case_ids)} videos")
        print(f"Human-selected images: {self.df['is_human_top'].sum()}")

        return self.df

    def _load_human_selections(self):
        """Load human selections from Excel file."""
        try:
            # Try reading from the Excel file in validation_results
            xlsx_path = self.config.results_dir / 'ベストショット一致率20260104.xlsx'
            if xlsx_path.exists():
                df_excel = pd.read_excel(xlsx_path, sheet_name=0)

                for case_id in df_excel['動画番号'].unique():
                    case_data = df_excel[df_excel['動画番号'] == case_id]
                    yf_images = []
                    hk_images = []

                    for _, row in case_data.iterrows():
                        if pd.notna(row.get('YF')):
                            frame_num = int(row['YF'])
                            yf_images.append(f'IMG_{case_id}_{frame_num:04d}.jpg')
                        if pd.notna(row.get('HK')):
                            frame_num = int(row['HK'])
                            hk_images.append(f'IMG_{case_id}_{frame_num:04d}.jpg')

                    self.human_selections[case_id] = {
                        'YF': yf_images[:5],
                        'HK': hk_images[:5],
                        'all': list(set(yf_images[:5] + hk_images[:5]))
                    }
            else:
                # Fallback: use bestimage_human.xlsx
                df_human = pd.read_excel(self.config.human_excel_path)
                # Parse based on actual file structure
                print(f"Using {self.config.human_excel_path}")

        except Exception as e:
            print(f"Warning: Could not load human selections: {e}")
            # Use is_human_top column from merged CSV if available
            if 'is_human_top' in self.df.columns:
                for case_id in self.df['case_id'].unique():
                    case_df = self.df[(self.df['case_id'] == case_id) & (self.df['is_human_top'] == True)]
                    self.human_selections[case_id] = {
                        'all': case_df['image_name'].tolist()
                    }

    def _mark_human_selections(self):
        """Mark human-selected images in the dataframe."""
        if 'is_human_top' not in self.df.columns:
            self.df['is_human_top'] = False

        for case_id, selections in self.human_selections.items():
            all_selected = selections.get('all', [])
            mask = (self.df['case_id'] == case_id) & (self.df['image_name'].isin(all_selected))
            self.df.loc[mask, 'is_human_top'] = True

    def preprocess_features(self, case_id: int = None) -> pd.DataFrame:
        """
        Preprocess features with per-case normalization.

        Args:
            case_id: If provided, preprocess only this case. Otherwise, all cases.

        Returns:
            DataFrame with preprocessed features.
        """
        if case_id is not None:
            case_ids = [case_id]
        else:
            case_ids = self.case_ids

        processed_dfs = []

        for cid in case_ids:
            case_df = self.df[self.df['case_id'] == cid].copy()

            # Handle missing values
            for col in self.config.feature_columns:
                if col in ['disc_detected', 'disc_pos_ok']:
                    case_df[col] = case_df[col].fillna(False).astype(float)
                elif col in case_df.columns:
                    median_val = case_df[col].median()
                    if pd.isna(median_val):
                        median_val = 0
                    case_df[col] = case_df[col].fillna(median_val)
                else:
                    case_df[col] = 0.0

            # Min-max normalization per case
            numeric_cols = [c for c in self.config.feature_columns
                          if c not in ['disc_detected', 'disc_pos_ok'] and c in case_df.columns]

            if numeric_cols:
                scaler = MinMaxScaler()
                # Only fit on valid data
                valid_mask = case_df[numeric_cols].notna().all(axis=1)
                if valid_mask.sum() > 0:
                    case_df.loc[valid_mask, numeric_cols] = scaler.fit_transform(
                        case_df.loc[valid_mask, numeric_cols]
                    )
                    self.scalers[cid] = scaler

            processed_dfs.append(case_df)

        return pd.concat(processed_dfs, ignore_index=True)

    def compute_rule_based_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute rule-based score for Top 100 filtering.

        Args:
            df: DataFrame with features (should be normalized).

        Returns:
            Series with rule-based scores.
        """
        weights = self.config.rule_based_weights
        score = pd.Series(0.0, index=df.index)

        for col, weight in weights.items():
            if col in df.columns:
                score += weight * df[col].fillna(0)

        return score

    def get_top_k_candidates(self, case_id: int, k: int = None) -> pd.DataFrame:
        """
        Get top-k candidates for a video using rule-based score.

        Args:
            case_id: Video case ID.
            k: Number of top candidates. Uses config.top_k_candidates if None.

        Returns:
            DataFrame with top-k candidates.
        """
        k = k or self.config.top_k_candidates

        case_df = self.df[self.df['case_id'] == case_id].copy()

        # Basic filter: disc_detected and retina_ratio > 0
        filtered = case_df[
            (case_df['disc_detected'] == True) &
            (case_df['retina_ratio'] > 0)
        ].copy()

        if len(filtered) == 0:
            # Fallback: just use retina_ratio > 0
            filtered = case_df[case_df['retina_ratio'] > 0].copy()

        if len(filtered) == 0:
            print(f"Warning: No valid candidates for case {case_id}")
            return case_df.head(k)

        # Preprocess for scoring
        processed = self.preprocess_features(case_id)
        processed = processed[processed.index.isin(filtered.index)]

        # Compute rule-based score
        processed['rule_score'] = self.compute_rule_based_score(processed)

        # Get top-k
        top_k = processed.nlargest(min(k, len(processed)), 'rule_score')

        return top_k

    def generate_pairwise_samples(
        self,
        case_ids: List[int] = None,
        return_tensors: bool = True
    ) -> Tuple[List[Dict], pd.DataFrame]:
        """
        Generate pairwise training samples.

        Args:
            case_ids: List of case IDs to include. Uses all if None.
            return_tensors: If True, convert features to tensors.

        Returns:
            Tuple of (list of pair dicts, DataFrame with all candidates).
        """
        case_ids = case_ids or self.case_ids
        all_pairs = []
        all_candidates = []

        for case_id in case_ids:
            # Get top-k candidates
            candidates = self.get_top_k_candidates(case_id)
            all_candidates.append(candidates)

            # Split positive and negative
            positives = candidates[candidates['is_human_top'] == True]
            negatives = candidates[candidates['is_human_top'] == False]

            if len(positives) == 0:
                print(f"Warning: No positive samples in top-{self.config.top_k_candidates} for case {case_id}")
                continue

            if len(negatives) == 0:
                print(f"Warning: No negative samples for case {case_id}")
                continue

            # Generate pairs
            num_negs = self.config.num_negatives_per_positive
            hard_ratio = self.config.hard_negative_ratio

            for _, pos_row in positives.iterrows():
                # Hard negatives: top by rule_score
                num_hard = int(num_negs * hard_ratio)
                num_random = num_negs - num_hard

                hard_negs = negatives.nlargest(
                    min(len(negatives) // 2, num_hard * 2), 'rule_score'
                )
                if len(hard_negs) > num_hard:
                    hard_negs = hard_negs.sample(num_hard)

                # Random negatives
                remaining = negatives[~negatives.index.isin(hard_negs.index)]
                if len(remaining) > num_random:
                    random_negs = remaining.sample(num_random)
                else:
                    random_negs = remaining

                selected_negs = pd.concat([hard_negs, random_negs])

                for _, neg_row in selected_negs.iterrows():
                    pair = {
                        'case_id': case_id,
                        'pos_image': pos_row['image_name'],
                        'neg_image': neg_row['image_name'],
                        'pos_features': self._get_features(pos_row),
                        'neg_features': self._get_features(neg_row),
                    }

                    if return_tensors:
                        pair['pos_features'] = torch.tensor(pair['pos_features'], dtype=torch.float32)
                        pair['neg_features'] = torch.tensor(pair['neg_features'], dtype=torch.float32)

                    all_pairs.append(pair)

        all_candidates_df = pd.concat(all_candidates, ignore_index=True) if all_candidates else pd.DataFrame()

        print(f"Generated {len(all_pairs)} pairwise samples from {len(case_ids)} videos")

        return all_pairs, all_candidates_df

    def _get_features(self, row: pd.Series) -> np.ndarray:
        """Extract feature vector from a row."""
        features = []
        for col in self.config.feature_columns:
            if col in row.index:
                val = row[col]
                if pd.isna(val):
                    val = 0.0
                features.append(float(val))
            else:
                features.append(0.0)
        return np.array(features, dtype=np.float32)

    def get_case_data(self, case_id: int) -> pd.DataFrame:
        """
        Get all candidate data for a specific case.

        Args:
            case_id: Video case ID.

        Returns:
            DataFrame with candidate features and labels.
        """
        return self.get_top_k_candidates(case_id)


class PairwiseBatchDataset(Dataset):
    """PyTorch Dataset for pairwise samples."""

    def __init__(self, pairs: List[Dict]):
        """
        Initialize dataset with pairs.

        Args:
            pairs: List of pair dictionaries with pos_features and neg_features.
        """
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pair = self.pairs[idx]
        return {
            'pos_features': pair['pos_features'],
            'neg_features': pair['neg_features'],
        }


class PairwiseDataLoader:
    """
    DataLoader factory for pairwise training.

    Supports leave-one-video-out cross-validation.
    """

    def __init__(self, dataset: ROPRankingDataset, config: Config = None):
        """
        Initialize with dataset.

        Args:
            dataset: ROPRankingDataset instance.
            config: Configuration object.
        """
        self.dataset = dataset
        self.config = config or default_config

    def get_lovo_split(self, test_case_id: int) -> Tuple[DataLoader, pd.DataFrame]:
        """
        Get train DataLoader and test data for leave-one-video-out CV.

        Args:
            test_case_id: Case ID to hold out for testing.

        Returns:
            Tuple of (train DataLoader, test DataFrame).
        """
        # Train cases
        train_case_ids = [cid for cid in self.dataset.case_ids if cid != test_case_id]

        # Generate training pairs
        train_pairs, _ = self.dataset.generate_pairwise_samples(train_case_ids)

        # Get test data
        test_df = self.dataset.get_case_data(test_case_id)

        # Create DataLoader
        train_dataset = PairwiseBatchDataset(train_pairs)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=False
        )

        return train_loader, test_df

    def get_all_data_loader(self) -> DataLoader:
        """
        Get DataLoader with all data for final training.

        Returns:
            DataLoader with all pairwise samples.
        """
        all_pairs, _ = self.dataset.generate_pairwise_samples()

        train_dataset = PairwiseBatchDataset(all_pairs)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=False
        )

        return train_loader
