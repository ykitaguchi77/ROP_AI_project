"""RW-ROP低下の原因分析スクリプト"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from select_best_images import minmax_norm

# Load data
pred_df = pd.read_csv(Path(__file__).parent / "outputs_clinical_v3" / "predictions.csv")
pred_df['image_name'] = pred_df['image_path'].apply(lambda x: Path(x).name)

kubota_df = pd.read_excel(r"E:\Multicenter_ROP_study\Multicenter_images\Kubota_selection\selected_images_disc_retina.xlsx")
feat_cols = ['retina_ratio', 'mbss_Grad_p90', 'mbss_score', 'disc_detected', 'disc_edge_coverage_ratio']
avail = [c for c in feat_cols if c in kubota_df.columns]
kubota_features = kubota_df[['image_name'] + avail].drop_duplicates(subset='image_name', keep='first')
merged_df = pred_df.merge(kubota_features, on='image_name', how='left')

# Top-level Excel fill
tl = pd.read_excel(r"E:\Multicenter_ROP_study\Multicenter_images\selected_images_disc_retina.xlsx")
tl_feat = [c for c in avail if c in tl.columns]
if 'image_name' in tl.columns and tl_feat:
    tl_features = tl[['image_name'] + tl_feat].drop_duplicates(subset='image_name', keep='first').set_index('image_name')
    mm = merged_df['retina_ratio'].isna()
    for col in tl_feat:
        fill = merged_df.loc[mm, 'image_name'].map(tl_features[col])
        merged_df.loc[mm, col] = fill.values

EDGE_COV = 0.80
W_R, W_G, W_M = 0.4, 0.4, 0.2

def select_top_k(df, top_k):
    all_sel = []
    for vid, group in df.groupby('video_id'):
        valid = group[group['retina_ratio'].notna() & (group['retina_ratio'] > 0)].copy()
        if len(valid) == 0:
            continue
        s1 = valid[
            (valid['disc_detected'] == True) &
            valid['disc_edge_coverage_ratio'].notna() &
            (valid['disc_edge_coverage_ratio'] >= EDGE_COV)
        ].copy()
        if len(s1) > 0:
            s1['rn'] = minmax_norm(s1['retina_ratio'].fillna(0))
            s1['gn'] = minmax_norm(s1['mbss_Grad_p90'].fillna(0))
            s1['mn'] = minmax_norm(s1['mbss_score'].fillna(0))
            s1['qs'] = W_R * s1['rn'] + W_G * s1['gn'] + W_M * s1['mn']
            s1 = s1.sort_values('qs', ascending=False)
            sel = s1.head(top_k).copy()
        else:
            sel = pd.DataFrame()
        nr = top_k - len(sel)
        if nr > 0:
            rem = valid[~valid.index.isin(sel.index)].sort_values('retina_ratio', ascending=False)
            sel = pd.concat([sel, rem.head(nr)])
        if len(sel) > 0:
            all_sel.append(sel)
    return pd.concat(all_sel, ignore_index=True) if all_sel else pd.DataFrame()

top10_df = select_top_k(merged_df, 10)
top5_df = select_top_k(merged_df, 5)

# === Analysis ===
print("=== RW-ROP Component Sensitivity ===")
print(f"{'Condition':<18} {'All(6448)':>12} {'Top10(3089)':>12} {'Top5(1650)':>12} {'D(Top5-All)':>12}")
print("-" * 70)

for df_list, labels in [([merged_df, top10_df, top5_df], ['All', 'Top10', 'Top5'])]:
    pass

results = {}
for df, label in [(merged_df, 'All'), (top10_df, 'Top10'), (top5_df, 'Top5')]:
    zi_m = df['zone_label'] == 0
    s3_m = df['stage_label'] == 3
    pl_m = df['plus_label'] == 2
    rw_t = (zi_m | s3_m | pl_m).astype(int)
    rw_p = ((df['zone_pred'] == 0) | (df['stage_pred'] == 3) | (df['plus_pred'] == 2)).astype(int)
    results[label] = {
        'Zone I': ((df['zone_pred'] == 0) & zi_m).sum() / zi_m.sum() if zi_m.sum() > 0 else 0,
        'Stage 3': ((df['stage_pred'] == 3) & s3_m).sum() / s3_m.sum() if s3_m.sum() > 0 else 0,
        'Plus': ((df['plus_pred'] == 2) & pl_m).sum() / pl_m.sum() if pl_m.sum() > 0 else 0,
        'RW-ROP': ((rw_p == 1) & (rw_t == 1)).sum() / rw_t.sum() if rw_t.sum() > 0 else 0,
        'Zone I (n)': int(zi_m.sum()),
        'Stage 3 (n)': int(s3_m.sum()),
        'Plus (n)': int(pl_m.sum()),
        'RW-ROP (n)': int(rw_t.sum()),
    }

for cond in ['Zone I', 'Stage 3', 'Plus', 'RW-ROP']:
    a = results['All'][cond]
    t10 = results['Top10'][cond]
    t5 = results['Top5'][cond]
    n_a = results['All'][f'{cond} (n)']
    n_t5 = results['Top5'][f'{cond} (n)']
    delta = t5 - a
    print(f"{cond:<18} {a:>8.4f}({n_a:>4}) {t10:>8.4f}      {t5:>8.4f}({n_t5:>4}) {delta:>+10.4f}")

# Miss pattern analysis
print("\n=== RW-ROP Miss Pattern (False Negatives) ===")
for df, label in [(merged_df, 'All(6448)'), (top5_df, 'Top5(1650)')]:
    rw_t = ((df['plus_label'] == 2) | (df['stage_label'] == 3) | (df['zone_label'] == 0)).astype(int)
    rw_p = ((df['plus_pred'] == 2) | (df['stage_pred'] == 3) | (df['zone_pred'] == 0)).astype(int)
    miss = (rw_t == 1) & (rw_p == 0)
    miss_df = df[miss]
    n_miss = len(miss_df)

    only_zi = ((miss_df['zone_label'] == 0) & (miss_df['stage_label'] != 3) & (miss_df['plus_label'] != 2)).sum()
    only_s3 = ((miss_df['zone_label'] != 0) & (miss_df['stage_label'] == 3) & (miss_df['plus_label'] != 2)).sum()
    only_pl = ((miss_df['zone_label'] != 0) & (miss_df['stage_label'] != 3) & (miss_df['plus_label'] == 2)).sum()
    multi = n_miss - only_zi - only_s3 - only_pl

    print(f"\n{label}: {n_miss} misses / {rw_t.sum()} positives (miss rate={n_miss/rw_t.sum()*100:.1f}%)")
    print(f"  Zone I only missed:  {only_zi}")
    print(f"  Stage 3 only missed: {only_s3}")
    print(f"  Plus only missed:    {only_pl}")
    print(f"  Multiple conditions: {multi}")

# RW-ROP probability distribution comparison
print("\n=== RW-ROP Probability Distribution (positive cases) ===")
for df, label in [(merged_df, 'All'), (top5_df, 'Top5')]:
    rw_t = ((df['plus_label'] == 2) | (df['stage_label'] == 3) | (df['zone_label'] == 0)).astype(int)
    rw_prob = 1 - ((1 - df['plus_prob_2']) * (1 - df['stage_prob_3']) * (1 - df['zone_prob_0']))
    pos_probs = rw_prob[rw_t == 1]
    print(f"{label}: mean={pos_probs.mean():.4f}, median={pos_probs.median():.4f}, "
          f"<0.5={(pos_probs < 0.5).sum()}/{len(pos_probs)} ({(pos_probs < 0.5).sum()/len(pos_probs)*100:.1f}%)")
