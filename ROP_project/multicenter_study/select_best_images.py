"""
ベスト画像選出モジュール
validate_images_disc.ipynbのアルゴリズムを移植

アルゴリズム:
1. 足切り: disc_edge_coverage_ratio >= 0.80 の画像のみを対象
2. スコアリング: score = 0.4 × retina_ratio_norm + 0.4 × mbss_Grad_p90_norm + 0.2 × mbss_score_norm
3. 補完: 足切りで目標数に満たない場合は retina_ratio のみでソートして補完
"""
import numpy as np
import pandas as pd
from typing import List
import shutil
from pathlib import Path


# -------------------- パラメータ --------------------
# 足切り閾値
EDGE_COVERAGE_CUTOFF = 0.80

# スコア重み
WEIGHT_RETINA_RATIO = 0.4
WEIGHT_MBSS_GRAD_P90 = 0.4
WEIGHT_MBSS_SCORE = 0.2


def minmax_norm(series: pd.Series) -> pd.Series:
    """Min-Max正規化（0-1）"""
    min_val = series.min()
    max_val = series.max()
    if max_val - min_val < 1e-8:
        return pd.Series([0.5] * len(series), index=series.index)
    return (series - min_val) / (max_val - min_val)


def select_best_images(
    df: pd.DataFrame,
    top_k: int = 10,
    need_k: int = 5
) -> pd.DataFrame:
    """
    品質評価結果からベスト画像を選出（disc_edge_coverage版）

    Args:
        df: 品質評価結果のDataFrame
        top_k: 最終出力数（デフォルト: 10）
        need_k: 最低必要数（デフォルト: 5）※互換性のため残すが使用しない

    Returns:
        ベスト画像のDataFrame（rank列を含む）

    Algorithm:
        Stage 1: disc_edge_coverage_ratio >= 0.80 で足切り
                 score = 0.4*retina_ratio_norm + 0.4*mbss_Grad_p90_norm + 0.2*mbss_score_norm
        Stage 2: 足りなければ retina_ratio のみでソートして補完
    """
    # -------------------- 有効データ抽出 --------------------
    # lens_detected=True かつ retina_ratio>0
    valid = df[(df['lens_detected'] == True) & (df['retina_ratio'] > 0)].copy()

    if len(valid) == 0:
        raise RuntimeError("lens_detected=True かつ retina_ratio>0 のデータがありません")

    print(f"有効データ: {len(valid)}件")

    # -------------------- カラム補完 --------------------
    if 'mbss_score' not in valid.columns:
        valid['mbss_score'] = np.nan
    if 'mbss_Grad_p90' not in valid.columns:
        valid['mbss_Grad_p90'] = np.nan
    if 'disc_edge_coverage_ratio' not in valid.columns:
        valid['disc_edge_coverage_ratio'] = np.nan
    if 'disc_detected' not in valid.columns:
        valid['disc_detected'] = False

    # -------------------- Stage 1: 足切り通過画像 --------------------
    print("\n===== Stage 1: disc_edge_coverage >= 0.80 の画像 =====")

    # 足切り条件: disc_detected=True かつ disc_edge_coverage_ratio >= 0.80
    stage1_candidates = valid[
        (valid['disc_detected'] == True) &
        (valid['disc_edge_coverage_ratio'].notna()) &
        (valid['disc_edge_coverage_ratio'] >= EDGE_COVERAGE_CUTOFF)
    ].copy()

    print(f"足切り通過: {len(stage1_candidates)}件")

    if len(stage1_candidates) > 0:
        # Min-Max正規化（欠損値は0として扱う）
        stage1_candidates['retina_ratio_norm'] = minmax_norm(stage1_candidates['retina_ratio'].fillna(0))
        stage1_candidates['mbss_Grad_p90_norm'] = minmax_norm(stage1_candidates['mbss_Grad_p90'].fillna(0))
        stage1_candidates['mbss_score_norm'] = minmax_norm(stage1_candidates['mbss_score'].fillna(0))

        # スコア計算
        stage1_candidates['score'] = (
            WEIGHT_RETINA_RATIO * stage1_candidates['retina_ratio_norm'] +
            WEIGHT_MBSS_GRAD_P90 * stage1_candidates['mbss_Grad_p90_norm'] +
            WEIGHT_MBSS_SCORE * stage1_candidates['mbss_score_norm']
        )

        # スコア順にソート（降順）
        stage1_candidates = stage1_candidates.sort_values(by='score', ascending=False)
        stage1_selected = stage1_candidates.head(top_k).copy()
        stage1_selected['selection_stage'] = 'Stage1_edge_cov>=0.80'
    else:
        stage1_selected = pd.DataFrame()

    print(f"Stage 1 選定: {len(stage1_selected)}件")

    # -------------------- Stage 2: 補完（retina_ratioのみ） --------------------
    print("\n===== Stage 2: 補完（retina_ratioのみ） =====")

    n_remaining = top_k - len(stage1_selected)

    if n_remaining > 0:
        # Stage1で選ばれなかった画像から補完
        remaining = valid[~valid.index.isin(stage1_selected.index)].copy()

        if len(remaining) > 0:
            # retina_ratio順にソート（降順）
            remaining = remaining.sort_values(by='retina_ratio', ascending=False)
            stage2_selected = remaining.head(n_remaining).copy()
            stage2_selected['selection_stage'] = 'Stage2_補完'
            print(f"Stage 2 選定: {len(stage2_selected)}件")
        else:
            stage2_selected = pd.DataFrame()
            print("Stage 2 選定: 0件（残り画像なし）")
    else:
        stage2_selected = pd.DataFrame()
        print("Stage 2 選定: 0件（Stage 1で十分）")

    # -------------------- 結果結合 --------------------
    final_top = pd.concat([stage1_selected, stage2_selected], ignore_index=False)
    final_top = final_top.reset_index(drop=True)
    final_top['rank'] = range(1, len(final_top) + 1)

    print(f"\n===== 最終結果: {len(final_top)}件 =====")
    print(f"  Stage 1 (edge_cov>=0.80): {len(stage1_selected)}件")
    print(f"  Stage 2 (補完): {len(stage2_selected)}件")

    # -------------------- 表示 --------------------
    print(f"\n=== Best Top{min(10, top_k)} ===")
    for _, row in final_top.head(min(10, top_k)).iterrows():
        score_str = f", score={row['score']:.3f}" if 'score' in row and pd.notna(row.get('score')) else ""
        edge_cov = row.get('disc_edge_coverage_ratio')
        edge_cov_str = f", edge_cov={edge_cov:.3f}" if pd.notna(edge_cov) else ""
        print(f"  {row['rank']:2d}. {row['image_name']} (retina={row['retina_ratio']:.1f}%{edge_cov_str}{score_str})")

    if top_k > 10:
        print(f"  ... (以下省略)")

    return final_top


def copy_best_images(
    best_df: pd.DataFrame,
    output_dir: str,
    source_column: str = 'image_path'
) -> List[str]:
    """
    ベスト画像を指定ディレクトリにコピー

    Args:
        best_df: ベスト画像のDataFrame
        output_dir: 出力ディレクトリ
        source_column: ソース画像パスの列名

    Returns:
        コピーされた画像ファイルのパスのリスト
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    copied_paths = []

    for _, row in best_df.iterrows():
        source_path = Path(row[source_column])
        if not source_path.exists():
            print(f"警告: ソース画像が見つかりません: {source_path}")
            continue

        # ファイル名をそのまま使用
        dest_path = output_path / source_path.name
        shutil.copy2(source_path, dest_path)
        copied_paths.append(str(dest_path))

    print(f"{len(copied_paths)}枚の画像をコピーしました: {output_dir}")
    return copied_paths
