import os
import pandas as pd
import numpy as np
import glob
from pathlib import Path
from datetime import datetime

def generate_best_images_list():
    base_dir = r"C:\Users\ykita\ROP_AI_project\ROP_project\bestimage_validation"
    # 出力先: bestimage_validation/documents 配下へ保存（ユーザー要望）
    output_path = os.path.join(base_dir, "documents", "bestimage_list.xlsx")
    
    # Get all per-case validation CSV files
    # NOTE:
    # - 同じ image_id のCSVが複数存在すると、二重に処理されてしまうことがあります。
    # - そのため、見つかったCSVを image_id ごとに集約し「最新の1つだけ」を採用します。
    validation_results_dir = os.path.join(base_dir, "validation_results")
    if os.path.isdir(validation_results_dir):
        csv_files_found = glob.glob(os.path.join(validation_results_dir, "validation_results_*.csv"))
    else:
        # Fallback: recursive search (older layouts)
        csv_files_found = glob.glob(os.path.join(base_dir, "**", "validation_results_*.csv"), recursive=True)

    # Exclude "validation_results_all.csv" if it exists
    csv_files_found = [f for f in csv_files_found if os.path.basename(f) != "validation_results_all.csv"]

    # Deduplicate by normalized path first (safety)
    seen = set()
    csv_files_found_unique = []
    for f in csv_files_found:
        norm = os.path.normcase(os.path.abspath(f))
        if norm in seen:
            continue
        seen.add(norm)
        csv_files_found_unique.append(f)

    # Deduplicate by image_id, keep the newest file per id
    csv_by_id = {}
    for f in csv_files_found_unique:
        try:
            df_head = pd.read_csv(f, nrows=1)
            if 'image_id' in df_head.columns and len(df_head) > 0:
                image_id = str(df_head.iloc[0]['image_id'])
            else:
                image_id = os.path.basename(f).replace('validation_results_', '').replace('.csv', '')
        except Exception:
            # If header read fails, fall back to filename-based id
            image_id = os.path.basename(f).replace('validation_results_', '').replace('.csv', '')

        mtime = os.path.getmtime(f)
        if image_id in csv_by_id:
            prev_f, prev_mtime = csv_by_id[image_id]
            if mtime > prev_mtime:
                print(f"[WARN] Duplicate CSV for image_id={image_id}. Using newer file:\n  old: {prev_f}\n  new: {f}")
                csv_by_id[image_id] = (f, mtime)
            else:
                print(f"[WARN] Duplicate CSV for image_id={image_id}. Keeping newer file:\n  keep: {prev_f}\n  skip: {f}")
        else:
            csv_by_id[image_id] = (f, mtime)

    csv_items = sorted([(image_id, f_m[0]) for image_id, f_m in csv_by_id.items()], key=lambda x: x[0])
    
    all_best_images = []
    
    print(f"Found {len(csv_files_found)} CSV files (unique files: {len(csv_files_found_unique)}, unique image_ids: {len(csv_items)}).")
    
    for image_id, csv_file in csv_items:
        try:
            df = pd.read_csv(csv_file)
            if len(df) == 0:
                continue
                
            print(f"Processing {image_id}...")
            
            # Basic validation
            if 'lens_detected' not in df.columns or 'retina_ratio' not in df.columns:
                print(f"  Skipping {image_id}: Missing required columns")
                continue
                
            df_valid = df[(df['lens_detected'] == True) & (df['retina_ratio'] > 0)].copy()
            
            if len(df_valid) == 0:
                print(f"  Skipping {image_id}: No valid lens/retina detected")
                continue
            
            # --- Filtering Logic (Adaptive Threshold) ---
            
            # Goal: Get at least 5 candidates (ideally 10) by relaxing retina_ratio threshold
            final_candidates = pd.DataFrame()
            
            # Use median of ALL valid images for this ID as the quality baseline
            disc_ring_median = None
            if 'disc_ring_score' in df_valid.columns and df_valid['disc_ring_score'].notna().sum() > 0:
                disc_ring_median = df_valid['disc_ring_score'].median()
            
            target_count = 10
            # Try relaxing threshold: 90%ile -> 80%ile -> ... -> 0%ile
            percentiles = [0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.0]
            
            for p in percentiles:
                thresh_ratio = df_valid['retina_ratio'].quantile(p)
                
                # 1. Filter by Retina Ratio
                df_candidates_step1 = df_valid[df_valid['retina_ratio'] >= thresh_ratio].copy()
                
                # 2. Filter by Disc Ring Score (if available) > Median of ALL valid images
                if disc_ring_median is not None:
                    # Note: We use the median of the WHOLE valid set as a fixed quality bar,
                    # instead of the median of the current subset which would degrade as we relax ratio.
                    df_candidates_step2 = df_candidates_step1[
                        (df_candidates_step1['disc_ring_score'].notna()) &
                        (df_candidates_step1['disc_ring_score'] >= disc_ring_median)
                    ].copy()
                else:
                    # If no disc scores available, just use ratio filtered set
                    df_candidates_step2 = df_candidates_step1
                
                # Check if we have enough
                if len(df_candidates_step2) >= target_count:
                    final_candidates = df_candidates_step2
                    print(f"  Found {len(final_candidates)} candidates at percentile {p:.1f} (thresh={thresh_ratio:.2f})")
                    break
                
                # If we reached 0.0 (all images) and still don't have enough, take what we have
                if p == 0.0:
                    final_candidates = df_candidates_step2
                    print(f"  Found {len(final_candidates)} candidates at percentile {p:.1f} (thresh={thresh_ratio:.2f}) - Max relaxed")
            
            if len(final_candidates) == 0:
                print(f"  No candidates found even after relaxing thresholds")
                continue
            
            # 3. Rank by sum of ranks (mbss_score + disc_core_score)
            # Ensure columns exist
            if 'mbss_score' not in final_candidates.columns:
                final_candidates['mbss_score'] = np.nan
            if 'disc_core_score' not in final_candidates.columns:
                final_candidates['disc_core_score'] = np.nan

            # 追加要件:
            # retina_ratio の閾値を緩めて候補を増やした場合でも、
            # 「より厳しい閾値（高いパーセンタイル）を満たす画像」を常に上位に来るようにする。
            # → 各画像が満たす最大パーセンタイル（retina_tier）を付与し、tier優先でソートする。
            try:
                thr_by_p = {p: float(df_valid['retina_ratio'].quantile(p)) for p in percentiles}
                final_candidates = final_candidates.copy()
                final_candidates['retina_tier'] = 0.0
                for p in sorted(percentiles):  # low -> high (high wins)
                    thr = thr_by_p[p]
                    final_candidates.loc[final_candidates['retina_ratio'] >= thr, 'retina_tier'] = float(p)
            except Exception:
                # tier計算に失敗しても既存ロジックで続行
                final_candidates = final_candidates.copy()
                final_candidates['retina_tier'] = 0.0
                
            # Create ranks (descending score = rank 1)
            # method='min' means ties get same rank
            final_candidates['mbss_rank'] = final_candidates['mbss_score'].rank(ascending=False, method='min', na_option='bottom')
            final_candidates['disc_core_rank'] = final_candidates['disc_core_score'].rank(ascending=False, method='min', na_option='bottom')
            
            final_candidates['rank_sum'] = final_candidates['mbss_rank'] + final_candidates['disc_core_rank']
            
            # Sort priority:
            # 1) retina_tier (desc): 厳しいretina_ratio閾値を満たすほど上位
            # 2) rank_sum (asc)
            # 3) mbss_score (desc) tie-break
            final_candidates = final_candidates.sort_values(
                by=['retina_tier', 'rank_sum', 'mbss_score'],
                ascending=[False, True, False]
            )
            
            # Take top 10
            top_10 = final_candidates.head(10)
            
            # Add rank column for output
            for rank, (idx, row) in enumerate(top_10.iterrows(), 1):
                res = {
                    'image_id': image_id,
                    'rank': rank,
                    'image_name': row['image_name'],
                    'retina_ratio': row.get('retina_ratio'),
                    'mbss_score': row.get('mbss_score'),
                    'disc_core_score': row.get('disc_core_score'),
                    'disc_ring_score': row.get('disc_ring_score'),
                    'rank_sum': row.get('rank_sum')
                }
                all_best_images.append(res)
                
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            import traceback
            traceback.print_exc()

    if all_best_images:
        df_out = pd.DataFrame(all_best_images)
        # Create output directory if not exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        try:
            df_out.to_excel(output_path, index=False)
            print(f"\nSaved best image list to: {output_path}")
            print(f"Total rows: {len(df_out)}")
        except PermissionError as e:
            # Excel等でファイルが開かれていると上書きできないため、別名で退避保存する
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            root, ext = os.path.splitext(output_path)
            alt_path = f"{root}_{ts}{ext}"
            df_out.to_excel(alt_path, index=False)
            print(f"\n[WARN] 出力先ファイルが使用中のため上書きできませんでした: {output_path}")
            print(f"       代替ファイルに保存しました: {alt_path}")
            print(f"       （元ファイルを閉じてから再実行すれば上書きできます）")
            print(f"Total rows: {len(df_out)}")
    else:
        print("No results found.")

if __name__ == "__main__":
    generate_best_images_list()

