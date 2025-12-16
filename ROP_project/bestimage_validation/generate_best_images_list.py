import os
import pandas as pd
import numpy as np
import glob
from pathlib import Path

def generate_best_images_list():
    base_dir = r"C:\Users\ykita\ROP_AI_project\ROP_project\bestimage_validation"
    output_path = r"C:\Users\ykita\ROP_AI_project\ROP_project\documents\bestimage_list.xlsx"
    
    # Get all validation result CSV files
    # The csv files are in subdirectories or in the current directory?
    # Based on previous execution, they were in base_dir/validation_results_*.csv
    # But LS shows they might be missing or moved?
    # Let's search recursively just in case, or check the path.
    # Actually, previous LS of bestimage_validation showed ONLY .py/.ipynb files.
    # Wait, where did the CSVs go?
    # Ah, the user might have moved them or I am misremembering.
    # Let's look at the LS output again.
    # It seems the CSVs are NOT in c:\Users\ykita\ROP_AI_project\ROP_project\bestimage_validation\
    # But in previous turns they were there.
    # Let me check "documents" folder or if they were deleted?
    # Wait, the LS result showed:
    # c:\Users\ykita\ROP_AI_project\ROP_project\bestimage_validation\
    #   - documents\
    #   - generate_best_images_list.py
    # ...
    # The CSVs are missing from the list!
    
    # However, the previous "Glob" command (few turns ago) found them in:
    # c:\Users\ykita\ROP_AI_project\ROP_project\bestimage_validation\
    # Maybe they are hidden or I am blind?
    # Let me use recursive glob.
    
    csv_files = glob.glob(os.path.join(base_dir, "**", "validation_results_*.csv"), recursive=True)
    # Exclude "validation_results_all.csv" if it exists
    csv_files = [f for f in csv_files if "validation_results_all.csv" not in f]
    
    all_best_images = []
    
    print(f"Found {len(csv_files)} CSV files.")
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if len(df) == 0:
                continue
                
            image_id = str(df.iloc[0]['image_id']) if 'image_id' in df.columns else os.path.basename(csv_file).replace('validation_results_', '').replace('.csv', '')
            
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
                
            # Create ranks (descending score = rank 1)
            # method='min' means ties get same rank
            final_candidates['mbss_rank'] = final_candidates['mbss_score'].rank(ascending=False, method='min', na_option='bottom')
            final_candidates['disc_core_rank'] = final_candidates['disc_core_score'].rank(ascending=False, method='min', na_option='bottom')
            
            final_candidates['rank_sum'] = final_candidates['mbss_rank'] + final_candidates['disc_core_rank']
            
            # Sort by rank_sum (ascending), then break ties with mbss_score (descending)
            final_candidates = final_candidates.sort_values(by=['rank_sum', 'mbss_score'], ascending=[True, False])
            
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
        
        df_out.to_excel(output_path, index=False)
        print(f"\nSaved best image list to: {output_path}")
        print(f"Total rows: {len(df_out)}")
    else:
        print("No results found.")

if __name__ == "__main__":
    generate_best_images_list()

