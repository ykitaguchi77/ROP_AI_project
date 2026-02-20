"""
Pairwise Ranking Model for Best Fundus Image Selection

デバッグ機能付きのトレーニングスクリプト
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# =============================================================================
# 設定
# =============================================================================

@dataclass
class Config:
    """Configuration for pairwise ranking model training."""

    base_dir: Path = Path(r'C:\Users\ykita\ROP_AI_project\ROP_project\bestimage_validation')

    feature_columns: List[str] = field(default_factory=lambda: [
        'retina_ratio', 'retina_area', 'disc_detected',
        'disc_edge_coverage_ratio', 'disc_area_ratio', 'disc_center_dist_ratio',
        'disc_pos_ok', 'mbss_L_multi', 'mbss_HF_ratio', 'mbss_Spec_centroid',
        'mbss_Grad_p90', 'mbss_score', 'disc_core_score', 'disc_ring_score', 'S_mean',
    ])

    rule_based_weights: dict = field(default_factory=lambda: {
        'retina_ratio': 0.4, 'mbss_Grad_p90': 0.4, 'mbss_score': 0.2,
    })

    top_k_candidates: int = 100
    num_negatives_per_positive: int = 10
    hard_negative_ratio: float = 0.5

    input_dim: int = 15
    hidden_dims: List[int] = field(default_factory=lambda: [64, 32, 16])
    dropout: float = 0.3

    learning_rate: float = 1e-3
    margin: float = 1.0
    batch_size: int = 64
    num_epochs: int = 100
    early_stopping_patience: int = 10
    weight_decay: float = 1e-4

    top_k_eval: int = 5
    seed: int = 42

    def __post_init__(self):
        self.results_dir = self.base_dir / 'validation_results'
        self.merged_csv_path = self.results_dir / 'merged_with_disc.csv'
        self.output_dir = self.base_dir / 'ranking_model' / 'outputs'


# =============================================================================
# データセット
# =============================================================================

class ROPRankingDataset:
    """Dataset class for ROP best image ranking."""

    def __init__(self, config: Config):
        self.config = config
        self.df = None
        self.human_selections = {}
        self.case_ids = []

    def load_data(self, verbose: bool = True) -> pd.DataFrame:
        """Load and merge feature data with human selections."""
        if verbose:
            print("=" * 60)
            print("LOADING DATA")
            print("=" * 60)

        # Load CSV
        self.df = pd.read_csv(self.config.merged_csv_path)
        if verbose:
            print(f"Loaded CSV: {len(self.df)} rows")
            print(f"Columns: {list(self.df.columns)}")

        # Check existing is_human_top
        if 'is_human_top' in self.df.columns:
            existing_human_top = self.df['is_human_top'].sum()
            if verbose:
                print(f"Existing is_human_top in CSV: {existing_human_top}")

        # Load human selections from Excel
        self._load_human_selections(verbose)

        # Mark human selections
        self._mark_human_selections(verbose)

        # Get case IDs
        self.case_ids = sorted(self.df['case_id'].unique().tolist())

        if verbose:
            print(f"\nFinal dataset:")
            print(f"  Total images: {len(self.df)}")
            print(f"  Number of cases: {len(self.case_ids)}")
            print(f"  Human-selected images: {self.df['is_human_top'].sum()}")

        return self.df

    def _load_human_selections(self, verbose: bool = True):
        """Load human selections from Excel file."""
        xlsx_path = self.config.results_dir / 'ベストショット一致率20260104.xlsx'

        if verbose:
            print(f"\nLooking for human selections Excel: {xlsx_path}")
            print(f"  Exists: {xlsx_path.exists()}")

        if xlsx_path.exists():
            try:
                df_excel = pd.read_excel(xlsx_path, sheet_name=0)
                if verbose:
                    print(f"  Loaded Excel: {len(df_excel)} rows")
                    print(f"  Columns: {list(df_excel.columns)}")

                for case_id in df_excel['動画番号'].unique():
                    case_data = df_excel[df_excel['動画番号'] == case_id]
                    yf_images, hk_images = [], []

                    for _, row in case_data.iterrows():
                        if pd.notna(row.get('YF')):
                            yf_images.append(f'IMG_{case_id}_{int(row["YF"]):04d}.jpg')
                        if pd.notna(row.get('HK')):
                            hk_images.append(f'IMG_{case_id}_{int(row["HK"]):04d}.jpg')

                    self.human_selections[case_id] = {
                        'YF': yf_images[:5],
                        'HK': hk_images[:5],
                        'all': list(set(yf_images[:5] + hk_images[:5]))
                    }

                if verbose:
                    print(f"\n  Human selections loaded for {len(self.human_selections)} cases:")
                    for case_id, sel in list(self.human_selections.items())[:3]:
                        print(f"    Case {case_id}: {len(sel['all'])} images - {sel['all'][:2]}...")

            except Exception as e:
                print(f"  ERROR loading Excel: {e}")
        else:
            # Fallback to is_human_top column
            if verbose:
                print("  Excel not found, using is_human_top column from CSV")

            if 'is_human_top' in self.df.columns:
                for case_id in self.df['case_id'].unique():
                    case_df = self.df[(self.df['case_id'] == case_id) & (self.df['is_human_top'] == True)]
                    self.human_selections[case_id] = {'all': case_df['image_name'].tolist()}

    def _mark_human_selections(self, verbose: bool = True):
        """Mark human-selected images in the dataframe."""
        if 'is_human_top' not in self.df.columns:
            self.df['is_human_top'] = False
        else:
            # Reset to False first, then re-mark
            self.df['is_human_top'] = False

        total_marked = 0
        for case_id, selections in self.human_selections.items():
            all_selected = selections.get('all', [])
            mask = (self.df['case_id'] == case_id) & (self.df['image_name'].isin(all_selected))
            matched = mask.sum()
            self.df.loc[mask, 'is_human_top'] = True
            total_marked += matched

            if verbose and matched != len(all_selected):
                print(f"  WARNING: Case {case_id}: {matched}/{len(all_selected)} images found in dataset")

        if verbose:
            print(f"\nTotal human-selected images marked: {total_marked}")

    def check_top_k_coverage(self, k: int = 100, verbose: bool = True) -> Dict:
        """Check how many human selections are in Top-K candidates."""
        if verbose:
            print("\n" + "=" * 60)
            print(f"TOP-{k} COVERAGE CHECK")
            print("=" * 60)

        coverage_results = []

        for case_id in self.case_ids:
            # Get all human selections for this case
            human_images = self.human_selections.get(case_id, {}).get('all', [])
            n_human_total = len(human_images)

            # Get top-k candidates
            top_k_df = self.get_top_k_candidates(case_id, k=k, verbose=False)
            top_k_images = set(top_k_df['image_name'].tolist())

            # Count matches
            n_in_top_k = len(set(human_images) & top_k_images)
            coverage = n_in_top_k / n_human_total if n_human_total > 0 else 0

            # Check which human images are missing
            missing = set(human_images) - top_k_images

            coverage_results.append({
                'case_id': case_id,
                'n_human_total': n_human_total,
                'n_in_top_k': n_in_top_k,
                'coverage': coverage,
                'missing': list(missing),
                'top_k_size': len(top_k_df),
            })

            if verbose:
                status = "OK" if coverage == 1.0 else "MISSING"
                print(f"  Case {case_id}: {n_in_top_k}/{n_human_total} ({coverage*100:.0f}%) - {status}")
                if missing:
                    print(f"    Missing: {missing}")

        mean_coverage = np.mean([r['coverage'] for r in coverage_results])
        if verbose:
            print(f"\nMean coverage: {mean_coverage*100:.1f}%")

            # Show cases with low coverage
            low_coverage = [r for r in coverage_results if r['coverage'] < 1.0]
            if low_coverage:
                print(f"\nWARNING: {len(low_coverage)} cases have incomplete coverage!")

        return coverage_results

    def get_top_k_candidates(self, case_id: int, k: int = None, verbose: bool = False) -> pd.DataFrame:
        """Get top-k candidates for a video."""
        k = k or self.config.top_k_candidates
        case_df = self.df[self.df['case_id'] == case_id].copy()

        if verbose:
            print(f"\n  Case {case_id}: {len(case_df)} total images")

        # Basic filter: disc_detected and retina_ratio > 0
        filtered = case_df[(case_df['disc_detected'] == True) & (case_df['retina_ratio'] > 0)].copy()

        if verbose:
            print(f"    After disc_detected & retina_ratio > 0: {len(filtered)}")

        if len(filtered) == 0:
            # Fallback: just retina_ratio > 0
            filtered = case_df[case_df['retina_ratio'] > 0].copy()
            if verbose:
                print(f"    Fallback (retina_ratio > 0 only): {len(filtered)}")

        if len(filtered) == 0:
            if verbose:
                print(f"    WARNING: No valid candidates, returning first {k} images")
            return case_df.head(k)

        # Preprocess and normalize features
        for col in self.config.feature_columns:
            if col in ['disc_detected', 'disc_pos_ok']:
                filtered[col] = filtered[col].fillna(False).astype(float)
            elif col in filtered.columns:
                median_val = filtered[col].median()
                filtered[col] = filtered[col].fillna(median_val if pd.notna(median_val) else 0)
            else:
                filtered[col] = 0.0

        # Normalize
        numeric_cols = [c for c in self.config.feature_columns
                       if c not in ['disc_detected', 'disc_pos_ok'] and c in filtered.columns]
        if numeric_cols and len(filtered) > 1:
            scaler = MinMaxScaler()
            filtered[numeric_cols] = scaler.fit_transform(filtered[numeric_cols].fillna(0))

        # Compute rule-based score
        score = pd.Series(0.0, index=filtered.index)
        for col, weight in self.config.rule_based_weights.items():
            if col in filtered.columns:
                score += weight * filtered[col].fillna(0)
        filtered['rule_score'] = score

        # Get top-k
        result = filtered.nlargest(min(k, len(filtered)), 'rule_score')

        if verbose:
            print(f"    Returning top-{len(result)} candidates")

        return result

    def _get_features(self, row: pd.Series) -> np.ndarray:
        """Extract feature vector from a row."""
        features = []
        for col in self.config.feature_columns:
            val = row.get(col, 0.0)
            features.append(float(val) if pd.notna(val) else 0.0)
        return np.array(features, dtype=np.float32)

    def generate_pairwise_samples(self, case_ids: List[int] = None, verbose: bool = True) -> Tuple[List[Dict], pd.DataFrame]:
        """Generate pairwise training samples."""
        case_ids = case_ids or self.case_ids
        all_pairs, all_candidates = [], []

        stats = {'total_positives': 0, 'total_negatives': 0, 'cases_with_data': 0}

        for case_id in case_ids:
            candidates = self.get_top_k_candidates(case_id, verbose=False)
            all_candidates.append(candidates)

            positives = candidates[candidates['is_human_top'] == True]
            negatives = candidates[candidates['is_human_top'] == False]

            stats['total_positives'] += len(positives)
            stats['total_negatives'] += len(negatives)

            if len(positives) == 0 or len(negatives) == 0:
                if verbose:
                    print(f"  WARNING: Case {case_id}: {len(positives)} positives, {len(negatives)} negatives - SKIPPING")
                continue

            stats['cases_with_data'] += 1

            num_negs = self.config.num_negatives_per_positive
            hard_ratio = self.config.hard_negative_ratio

            for _, pos_row in positives.iterrows():
                num_hard = int(num_negs * hard_ratio)
                num_random = num_negs - num_hard

                hard_negs = negatives.nlargest(min(len(negatives) // 2, num_hard * 2), 'rule_score')
                if len(hard_negs) > num_hard:
                    hard_negs = hard_negs.sample(num_hard, random_state=self.config.seed)

                remaining = negatives[~negatives.index.isin(hard_negs.index)]
                if len(remaining) > 0:
                    random_negs = remaining.sample(min(num_random, len(remaining)), random_state=self.config.seed)
                else:
                    random_negs = pd.DataFrame()

                selected_negs = pd.concat([hard_negs, random_negs])

                for _, neg_row in selected_negs.iterrows():
                    all_pairs.append({
                        'case_id': case_id,
                        'pos_features': torch.tensor(self._get_features(pos_row), dtype=torch.float32),
                        'neg_features': torch.tensor(self._get_features(neg_row), dtype=torch.float32),
                    })

        all_candidates_df = pd.concat(all_candidates, ignore_index=True) if all_candidates else pd.DataFrame()

        if verbose:
            print(f"\nPairwise sample generation:")
            print(f"  Cases with data: {stats['cases_with_data']}/{len(case_ids)}")
            print(f"  Total positives in Top-K: {stats['total_positives']}")
            print(f"  Total negatives in Top-K: {stats['total_negatives']}")
            print(f"  Generated pairs: {len(all_pairs)}")

        return all_pairs, all_candidates_df


# =============================================================================
# モデル
# =============================================================================

class PairwiseRanker(nn.Module):
    """MLP-based pairwise ranking model."""

    def __init__(self, input_dim: int = 15, hidden_dims: List[int] = [64, 32, 16], dropout: float = 0.3):
        super().__init__()
        layers = [nn.BatchNorm1d(input_dim)]
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU()])
            if i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)

    def compute_pairwise_loss(self, pos_features: torch.Tensor, neg_features: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
        pos_scores = self.forward(pos_features)
        neg_scores = self.forward(neg_features)
        return F.margin_ranking_loss(pos_scores, neg_scores, target=torch.ones_like(pos_scores), margin=margin)


class PairwiseBatchDataset(Dataset):
    def __init__(self, pairs: List[Dict]):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {'pos_features': self.pairs[idx]['pos_features'], 'neg_features': self.pairs[idx]['neg_features']}


# =============================================================================
# 学習
# =============================================================================

class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_value = None

    def __call__(self, value: float) -> bool:
        if self.best_value is None:
            self.best_value = value
            return False
        if value < self.best_value - self.min_delta:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class Trainer:
    def __init__(self, model: PairwiseRanker, config: Config, device: str = None):
        self.config = config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)
        self.history = {'train_loss': [], 'train_acc': [], 'lr': []}

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        self.model.train()
        total_loss, total_correct, total_samples = 0.0, 0, 0

        for batch in train_loader:
            pos_features = batch['pos_features'].to(self.device)
            neg_features = batch['neg_features'].to(self.device)

            self.optimizer.zero_grad()
            loss = self.model.compute_pairwise_loss(pos_features, neg_features, self.config.margin)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item() * pos_features.size(0)
            with torch.no_grad():
                pos_scores = self.model(pos_features)
                neg_scores = self.model(neg_features)
                total_correct += (pos_scores > neg_scores).sum().item()
                total_samples += pos_features.size(0)

        return total_loss / total_samples, total_correct / total_samples

    def fit(self, train_loader: DataLoader, num_epochs: int = None, verbose: bool = True) -> Dict:
        num_epochs = num_epochs or self.config.num_epochs
        early_stopper = EarlyStopping(patience=self.config.early_stopping_patience)
        best_loss, best_state = float('inf'), None

        epoch_iter = tqdm(range(num_epochs), desc='Training') if verbose else range(num_epochs)

        for epoch in epoch_iter:
            train_loss, train_acc = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)

            self.scheduler.step(train_loss)
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])

            if train_loss < best_loss:
                best_loss = train_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

            if early_stopper(train_loss):
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                break

            if verbose and hasattr(epoch_iter, 'set_description'):
                epoch_iter.set_description(f"Loss={train_loss:.4f}, Acc={train_acc:.4f}")

        if best_state:
            self.model.load_state_dict(best_state)

        return self.history


# =============================================================================
# 評価
# =============================================================================

class Evaluator:
    def __init__(self, model: PairwiseRanker, config: Config, device: str = None):
        self.model = model
        self.config = config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()

    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract features from DataFrame (already normalized from get_top_k_candidates)."""
        features = []
        for col in self.config.feature_columns:
            if col in df.columns:
                values = df[col].values.astype(float)
                values = np.nan_to_num(values, nan=0.0)
            else:
                values = np.zeros(len(df))
            features.append(values)
        return np.stack(features, axis=1).astype(np.float32)

    def predict_scores(self, df: pd.DataFrame) -> np.ndarray:
        features = self._extract_features(df)
        features_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            scores = self.model(features_tensor).cpu().numpy()
        return scores

    def evaluate_case(self, df: pd.DataFrame, case_id: int, k: int = 5, debug: bool = False) -> Dict:
        scores = self.predict_scores(df)
        df_eval = df.copy()
        df_eval['predicted_score'] = scores

        ai_top_k = df_eval.nlargest(k, 'predicted_score')['image_name'].tolist()
        human_top = df_eval[df_eval['is_human_top'] == True]['image_name'].tolist()
        matches = len(set(ai_top_k) & set(human_top))

        # NDCG@k
        sorted_df = df_eval.sort_values('predicted_score', ascending=False)
        relevance = sorted_df['is_human_top'].astype(int).values[:k]
        dcg = sum((2 ** rel - 1) / np.log2(i + 2) for i, rel in enumerate(relevance))
        ideal_relevance = sorted(df_eval['is_human_top'].astype(int).values, reverse=True)[:k]
        idcg = sum((2 ** rel - 1) / np.log2(i + 2) for i, rel in enumerate(ideal_relevance))
        ndcg = dcg / idcg if idcg > 0 else 0.0

        # MRR
        mrr = 0.0
        first_human_rank = None
        for rank, row in enumerate(sorted_df.itertuples(), 1):
            if row.is_human_top:
                mrr = 1.0 / rank
                first_human_rank = rank
                break

        if debug:
            print(f"    [EVAL] Test case {case_id}:")
            print(f"      Total candidates: {len(df)}")
            print(f"      Human selections in candidates: {len(human_top)}")
            print(f"      Score range: [{scores.min():.4f}, {scores.max():.4f}]")
            print(f"      Human selections: {human_top[:3]}...")
            print(f"      AI Top-5: {ai_top_k}")
            print(f"      Matches: {matches}")
            print(f"      First human rank: {first_human_rank}")

            # Show human selection scores
            human_df = df_eval[df_eval['is_human_top'] == True]
            print(f"      Human selection scores: {human_df['predicted_score'].values[:5]}")
            print(f"      Human selection ranks: ", end="")
            for img in human_top[:3]:
                rank = sorted_df[sorted_df['image_name'] == img].index
                if len(rank) > 0:
                    actual_rank = (sorted_df['image_name'] == img).values.nonzero()[0][0] + 1
                    print(f"{actual_rank}, ", end="")
            print()

        return {
            'case_id': case_id,
            'top5_precision': matches / k,
            'top5_recall': matches / len(human_top) if human_top else 0,
            'video_match': 1 if matches > 0 else 0,
            'ndcg_at_5': ndcg,
            'mrr': mrr,
            'ai_top_k': ai_top_k,
            'human_top': human_top,
            'num_matches': matches,
            'n_human_in_candidates': len(human_top),
        }


# =============================================================================
# LOVO CV
# =============================================================================

def evaluate_rule_based_baseline(dataset: ROPRankingDataset, config: Config) -> Dict:
    """
    Evaluate rule-based scoring as a baseline (no ML model).

    This replicates the original 54.5% baseline to verify our pipeline is correct.
    """
    print("\nEvaluating rule-based baseline...")

    results = {
        'case_id': [], 'top5_precision': [], 'top5_recall': [],
        'video_match': [], 'n_human_in_candidates': [],
    }

    for case_id in dataset.case_ids:
        # Get candidates with rule-based score
        candidates = dataset.get_top_k_candidates(case_id, verbose=False)

        if len(candidates) == 0:
            continue

        # Get top-5 by rule_score (already computed in get_top_k_candidates)
        ai_top_5 = candidates.nlargest(5, 'rule_score')['image_name'].tolist()
        human_top = candidates[candidates['is_human_top'] == True]['image_name'].tolist()
        matches = len(set(ai_top_5) & set(human_top))

        results['case_id'].append(case_id)
        results['top5_precision'].append(matches / 5)
        results['top5_recall'].append(matches / len(human_top) if human_top else 0)
        results['video_match'].append(1 if matches > 0 else 0)
        results['n_human_in_candidates'].append(len(human_top))

        print(f"  Case {case_id}: Prec={matches/5:.2f}, Matches={matches}/{len(human_top)}")

    print("\n" + "-" * 40)
    print(f"RULE-BASED BASELINE RESULTS:")
    print(f"  Mean Top-5 Precision: {np.mean(results['top5_precision']):.4f}")
    print(f"  Mean Top-5 Recall: {np.mean(results['top5_recall']):.4f}")
    print(f"  Video Match Rate: {np.mean(results['video_match']):.4f}")
    print(f"  (Expected baseline: ~0.545)")

    return results


def run_lovo_cv_v2(dataset: ROPRankingDataset, config: Config, num_folds: int = None,
                   num_epochs: int = None, verbose: bool = True, debug: bool = False) -> Dict:
    """
    Run Leave-One-Video-Out cross-validation with GLOBAL normalization.

    Key difference from v1: Uses consistent normalization across train/test.
    """
    results = {
        'case_id': [], 'top5_precision': [], 'top5_recall': [],
        'video_match': [], 'ndcg_at_5': [], 'mrr': [],
        'n_human_in_candidates': [], 'n_train_pairs': [],
    }

    case_ids_to_eval = dataset.case_ids[:num_folds] if num_folds else dataset.case_ids
    epochs = num_epochs or config.num_epochs

    if verbose:
        print("\n" + "=" * 60)
        print(f"LOVO CV v2 (Global Normalization): {len(case_ids_to_eval)} folds, {epochs} epochs each")
        print("=" * 60)

    case_iter = tqdm(case_ids_to_eval, desc='LOVO CV') if verbose else case_ids_to_eval

    for test_case_id in case_iter:
        # Train/test split - get RAW data (no normalization yet)
        train_case_ids = [cid for cid in dataset.case_ids if cid != test_case_id]

        # Collect all training candidates (raw features)
        train_dfs = []
        for case_id in train_case_ids:
            case_df = dataset.df[dataset.df['case_id'] == case_id].copy()
            # Filter: disc_detected and retina_ratio > 0
            filtered = case_df[(case_df['disc_detected'] == True) & (case_df['retina_ratio'] > 0)].copy()
            if len(filtered) == 0:
                filtered = case_df[case_df['retina_ratio'] > 0].copy()
            if len(filtered) > 0:
                train_dfs.append(filtered)

        if len(train_dfs) == 0:
            if verbose:
                print(f"\n  WARNING: No training data for fold {test_case_id}, skipping")
            results['case_id'].append(test_case_id)
            for key in ['top5_precision', 'top5_recall', 'video_match', 'ndcg_at_5', 'mrr', 'n_human_in_candidates', 'n_train_pairs']:
                results[key].append(0)
            continue

        # Combine all training data
        train_all = pd.concat(train_dfs, ignore_index=True)

        # Get test case data (raw)
        test_case_df = dataset.df[dataset.df['case_id'] == test_case_id].copy()
        test_filtered = test_case_df[(test_case_df['disc_detected'] == True) & (test_case_df['retina_ratio'] > 0)].copy()
        if len(test_filtered) == 0:
            test_filtered = test_case_df[test_case_df['retina_ratio'] > 0].copy()

        # Prepare features
        numeric_cols = [c for c in config.feature_columns if c not in ['disc_detected', 'disc_pos_ok']]

        # Handle boolean columns
        for col in ['disc_detected', 'disc_pos_ok']:
            if col in train_all.columns:
                train_all[col] = train_all[col].fillna(False).astype(float)
            if col in test_filtered.columns:
                test_filtered[col] = test_filtered[col].fillna(False).astype(float)

        # Handle missing values in numeric columns
        for col in numeric_cols:
            if col in train_all.columns:
                median_val = train_all[col].median()
                train_all[col] = train_all[col].fillna(median_val if pd.notna(median_val) else 0)
            if col in test_filtered.columns:
                median_val = test_filtered[col].median()
                test_filtered[col] = test_filtered[col].fillna(median_val if pd.notna(median_val) else 0)

        # GLOBAL NORMALIZATION: fit on training, transform both train and test
        scaler = MinMaxScaler()
        valid_numeric = [c for c in numeric_cols if c in train_all.columns]
        if valid_numeric and len(train_all) > 1:
            scaler.fit(train_all[valid_numeric].fillna(0))
            train_all[valid_numeric] = scaler.transform(train_all[valid_numeric].fillna(0))
            if len(test_filtered) > 0:
                test_filtered[valid_numeric] = scaler.transform(test_filtered[valid_numeric].fillna(0))

        # Compute rule-based score for top-k selection
        for df in [train_all, test_filtered]:
            score = pd.Series(0.0, index=df.index)
            for col, weight in config.rule_based_weights.items():
                if col in df.columns:
                    score += weight * df[col].fillna(0)
            df['rule_score'] = score

        # Get top-k from each training case
        train_pairs = []
        for case_id in train_case_ids:
            case_data = train_all[train_all['case_id'] == case_id]
            if len(case_data) == 0:
                continue
            top_k = case_data.nlargest(min(config.top_k_candidates, len(case_data)), 'rule_score')
            positives = top_k[top_k['is_human_top'] == True]
            negatives = top_k[top_k['is_human_top'] == False]

            if len(positives) == 0 or len(negatives) == 0:
                continue

            for _, pos_row in positives.iterrows():
                num_negs = min(config.num_negatives_per_positive, len(negatives))
                selected_negs = negatives.sample(num_negs, random_state=config.seed)
                for _, neg_row in selected_negs.iterrows():
                    pos_feat = np.array([float(pos_row.get(c, 0) if pd.notna(pos_row.get(c, 0)) else 0)
                                        for c in config.feature_columns], dtype=np.float32)
                    neg_feat = np.array([float(neg_row.get(c, 0) if pd.notna(neg_row.get(c, 0)) else 0)
                                        for c in config.feature_columns], dtype=np.float32)
                    train_pairs.append({
                        'pos_features': torch.tensor(pos_feat, dtype=torch.float32),
                        'neg_features': torch.tensor(neg_feat, dtype=torch.float32),
                    })

        results['n_train_pairs'].append(len(train_pairs))

        if len(train_pairs) == 0:
            if verbose:
                print(f"\n  WARNING: No training pairs for fold {test_case_id}, skipping")
            results['case_id'].append(test_case_id)
            for key in ['top5_precision', 'top5_recall', 'video_match', 'ndcg_at_5', 'mrr', 'n_human_in_candidates']:
                results[key].append(0)
            continue

        # Get top-k test candidates
        test_top_k = test_filtered.nlargest(min(config.top_k_candidates, len(test_filtered)), 'rule_score')

        train_dataset = PairwiseBatchDataset(train_pairs)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

        # Train
        model = PairwiseRanker(config.input_dim, config.hidden_dims, config.dropout)
        trainer = Trainer(model, config)
        history = trainer.fit(train_loader, num_epochs=epochs, verbose=False)

        if debug:
            print(f"\n  [DEBUG] Case {test_case_id}:")
            print(f"    Train pairs: {len(train_pairs)}")
            print(f"    Final train loss: {history['train_loss'][-1]:.4f}")
            print(f"    Final train acc: {history['train_acc'][-1]:.4f}")

            # Check score distribution on training sample
            model.eval()
            with torch.no_grad():
                sample_pos = train_pairs[0]['pos_features'].unsqueeze(0)
                sample_neg = train_pairs[0]['neg_features'].unsqueeze(0)
                pos_score = model(sample_pos.to(trainer.device)).item()
                neg_score = model(sample_neg.to(trainer.device)).item()
                print(f"    Sample pos score: {pos_score:.4f}")
                print(f"    Sample neg score: {neg_score:.4f}")
                print(f"    Pos > Neg: {pos_score > neg_score}")

        # Evaluate using normalized test data
        evaluator = Evaluator(model, config)
        metrics = evaluator.evaluate_case(test_top_k, test_case_id, debug=debug)

        results['case_id'].append(test_case_id)
        for key in ['top5_precision', 'top5_recall', 'video_match', 'ndcg_at_5', 'mrr', 'n_human_in_candidates']:
            results[key].append(metrics.get(key, 0))

        if verbose and hasattr(case_iter, 'set_description'):
            case_iter.set_description(f"Case {test_case_id}: Prec={metrics['top5_precision']:.2f}, Human={metrics['n_human_in_candidates']}")

    if verbose:
        print("\n" + "=" * 60)
        print(f"LOVO CV v2 Results ({len(case_ids_to_eval)} folds)")
        print("=" * 60)
        print(f"Mean Top-5 Precision: {np.mean(results['top5_precision']):.4f}")
        print(f"Mean Top-5 Recall: {np.mean(results['top5_recall']):.4f}")
        print(f"Video Match Rate: {np.mean(results['video_match']):.4f}")
        print(f"Mean NDCG@5: {np.mean(results['ndcg_at_5']):.4f}")
        print(f"Mean MRR: {np.mean(results['mrr']):.4f}")
        print(f"Avg Human in Candidates: {np.mean(results['n_human_in_candidates']):.1f}")
        print(f"Avg Train Pairs: {np.mean(results['n_train_pairs']):.0f}")

    return results


def run_lovo_cv(dataset: ROPRankingDataset, config: Config, num_folds: int = None,
                num_epochs: int = None, verbose: bool = True, debug: bool = False) -> Dict:
    """Run Leave-One-Video-Out cross-validation."""
    results = {
        'case_id': [], 'top5_precision': [], 'top5_recall': [],
        'video_match': [], 'ndcg_at_5': [], 'mrr': [],
        'n_human_in_candidates': [], 'n_train_pairs': [],
    }

    case_ids_to_eval = dataset.case_ids[:num_folds] if num_folds else dataset.case_ids
    epochs = num_epochs or config.num_epochs

    if verbose:
        print("\n" + "=" * 60)
        print(f"LOVO CV: {len(case_ids_to_eval)} folds, {epochs} epochs each")
        print("=" * 60)

    case_iter = tqdm(case_ids_to_eval, desc='LOVO CV') if verbose else case_ids_to_eval

    for test_case_id in case_iter:
        # Train/test split
        train_case_ids = [cid for cid in dataset.case_ids if cid != test_case_id]
        train_pairs, _ = dataset.generate_pairwise_samples(train_case_ids, verbose=False)
        test_df = dataset.get_top_k_candidates(test_case_id, verbose=False)

        results['n_train_pairs'].append(len(train_pairs))

        if len(train_pairs) == 0:
            if verbose:
                print(f"\n  WARNING: No training pairs for fold {test_case_id}, skipping")
            results['case_id'].append(test_case_id)
            for key in ['top5_precision', 'top5_recall', 'video_match', 'ndcg_at_5', 'mrr', 'n_human_in_candidates']:
                results[key].append(0)
            continue

        train_dataset = PairwiseBatchDataset(train_pairs)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

        # Train
        model = PairwiseRanker(config.input_dim, config.hidden_dims, config.dropout)
        trainer = Trainer(model, config)
        history = trainer.fit(train_loader, num_epochs=epochs, verbose=False)

        if debug:
            print(f"\n  [DEBUG] Case {test_case_id}:")
            print(f"    Train pairs: {len(train_pairs)}")
            print(f"    Final train loss: {history['train_loss'][-1]:.4f}")
            print(f"    Final train acc: {history['train_acc'][-1]:.4f}")

            # Check score distribution
            model.eval()
            with torch.no_grad():
                sample_pos = train_pairs[0]['pos_features'].unsqueeze(0)
                sample_neg = train_pairs[0]['neg_features'].unsqueeze(0)
                pos_score = model(sample_pos.to(trainer.device)).item()
                neg_score = model(sample_neg.to(trainer.device)).item()
                print(f"    Sample pos score: {pos_score:.4f}")
                print(f"    Sample neg score: {neg_score:.4f}")
                print(f"    Pos > Neg: {pos_score > neg_score}")

        # Evaluate
        evaluator = Evaluator(model, config)
        metrics = evaluator.evaluate_case(test_df, test_case_id, debug=debug)

        results['case_id'].append(test_case_id)
        for key in ['top5_precision', 'top5_recall', 'video_match', 'ndcg_at_5', 'mrr', 'n_human_in_candidates']:
            results[key].append(metrics.get(key, 0))

        if verbose and hasattr(case_iter, 'set_description'):
            case_iter.set_description(f"Case {test_case_id}: Prec={metrics['top5_precision']:.2f}, Human={metrics['n_human_in_candidates']}")

    if verbose:
        print("\n" + "=" * 60)
        print(f"LOVO CV Results ({len(case_ids_to_eval)} folds)")
        print("=" * 60)
        print(f"Mean Top-5 Precision: {np.mean(results['top5_precision']):.4f}")
        print(f"Mean Top-5 Recall: {np.mean(results['top5_recall']):.4f}")
        print(f"Video Match Rate: {np.mean(results['video_match']):.4f}")
        print(f"Mean NDCG@5: {np.mean(results['ndcg_at_5']):.4f}")
        print(f"Mean MRR: {np.mean(results['mrr']):.4f}")
        print(f"Avg Human in Candidates: {np.mean(results['n_human_in_candidates']):.1f}")
        print(f"Avg Train Pairs: {np.mean(results['n_train_pairs']):.0f}")

    return results


# =============================================================================
# メイン
# =============================================================================

def main():
    print("=" * 60)
    print("PAIRWISE RANKING MODEL - DEBUG MODE")
    print("=" * 60)

    # Setup
    config = Config()
    print(f"\nConfig:")
    print(f"  Base dir: {config.base_dir}")
    print(f"  Top-K candidates: {config.top_k_candidates}")
    print(f"  Model: {config.input_dim} -> {config.hidden_dims} -> 1")

    # Load data
    dataset = ROPRankingDataset(config)
    df = dataset.load_data(verbose=True)

    # Check coverage
    coverage = dataset.check_top_k_coverage(k=config.top_k_candidates, verbose=True)

    # Check pairwise sample generation
    print("\n" + "=" * 60)
    print("PAIRWISE SAMPLE GENERATION TEST")
    print("=" * 60)
    sample_pairs, _ = dataset.generate_pairwise_samples(case_ids=dataset.case_ids[:3], verbose=True)

    if len(sample_pairs) > 0:
        print(f"\nSample pair check:")
        print(f"  pos_features shape: {sample_pairs[0]['pos_features'].shape}")
        print(f"  neg_features shape: {sample_pairs[0]['neg_features'].shape}")
        print(f"  pos_features sample: {sample_pairs[0]['pos_features'][:5]}")

        # Analyze feature differences between positive and negative
        print(f"\nFeature comparison (Pos vs Neg):")
        pos_features = torch.stack([p['pos_features'] for p in sample_pairs])
        neg_features = torch.stack([p['neg_features'] for p in sample_pairs])
        pos_mean = pos_features.mean(dim=0).numpy()
        neg_mean = neg_features.mean(dim=0).numpy()

        print(f"  {'Feature':<30} {'Pos Mean':>10} {'Neg Mean':>10} {'Diff':>10}")
        print(f"  {'-'*60}")
        for i, col in enumerate(config.feature_columns):
            diff = pos_mean[i] - neg_mean[i]
            marker = "***" if abs(diff) > 0.1 else ""
            print(f"  {col:<30} {pos_mean[i]:>10.4f} {neg_mean[i]:>10.4f} {diff:>10.4f} {marker}")

    # First, test the baseline rule-based scoring
    print("\n" + "=" * 60)
    print("BASELINE TEST: Rule-based scoring only (no ML)")
    print("=" * 60)
    baseline_results = evaluate_rule_based_baseline(dataset, config)

    # Run test with V2 (global normalization)
    print("\n" + "=" * 60)
    print("LOVO CV V2 (3 folds, 30 epochs) - GLOBAL NORMALIZATION")
    print("=" * 60)
    pilot_results = run_lovo_cv_v2(dataset, config, num_folds=3, num_epochs=30, verbose=True, debug=True)

    # Summary
    print("\n" + "=" * 60)
    print("DIAGNOSIS SUMMARY")
    print("=" * 60)

    mean_coverage = np.mean([c['coverage'] for c in coverage])
    print(f"\n1. Top-{config.top_k_candidates} Coverage: {mean_coverage*100:.1f}%")
    if mean_coverage < 0.8:
        print("   PROBLEM: Human selections are not in Top-K candidates!")
        print("   SOLUTION: Increase top_k_candidates or adjust filtering criteria")

    total_human = df['is_human_top'].sum()
    print(f"\n2. Human Selections: {total_human}")
    if total_human == 0:
        print("   PROBLEM: No human selections loaded!")
        print("   SOLUTION: Check Excel file path and format")

    if len(sample_pairs) == 0:
        print("\n3. Training Pairs: 0")
        print("   PROBLEM: No training pairs generated!")
    else:
        print(f"\n3. Training Pairs: {len(sample_pairs)} (from 3 cases)")

    print(f"\n4. Pilot Test Result:")
    print(f"   Top-5 Precision: {pilot_results['top5_precision'][0]:.4f}")
    print(f"   Baseline: 0.545")


if __name__ == "__main__":
    main()
