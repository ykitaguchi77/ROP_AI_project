"""Per-video prediction agreement analysis"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

pred_df = pd.read_csv('outputs_clinical_v3/predictions.csv')
pred_df['image_name'] = pred_df['image_path'].apply(lambda x: Path(x).name)
print(f'Total: {len(pred_df)} images, {pred_df["video_id"].nunique()} video_ids')

# RW-ROP derived
pred_df['rw_rop_pred'] = (
    (pred_df['plus_pred'] == 2) | (pred_df['stage_pred'] == 3) | (pred_df['zone_pred'] == 0)
).astype(int)
pred_df['rw_rop_label'] = (
    (pred_df['plus_label'] == 2) | (pred_df['stage_label'] == 3) | (pred_df['zone_label'] == 0)
).astype(int)

tasks = {
    'zone_pred': ('Zone', 'zone_label'),
    'stage_pred': ('Stage', 'stage_label'),
    'plus_pred': ('Plus', 'plus_label'),
    'treatment_pred': ('Treatment', 'treatment_label'),
    'aggressive_rop_pred': ('AROP', 'aggressive_rop_label'),
    'rw_rop_pred': ('RW-ROP', 'rw_rop_label'),
}

# === Part 1: Agreement rates ===
print('\n=== Per-Video Prediction Agreement (All images, N=347 videos) ===')
print(f'{"Task":<12} {"Mean":>6} {"Median":>7} {"Min":>6} {"<80%":>6} {"<60%":>6} {"100%":>6}  (out of 347)')
print('-' * 65)

agreement_data = {}
for col, (name, label_col) in tasks.items():
    agreements = []
    for vid, group in pred_df.groupby('video_id'):
        mode_val = stats.mode(group[col], keepdims=True).mode[0]
        agree_rate = (group[col] == mode_val).mean()
        agreements.append({'video_id': vid, 'agreement': agree_rate, 'mode': mode_val,
                           'label': group[label_col].iloc[0], 'n_images': len(group)})
    adf = pd.DataFrame(agreements)
    agreement_data[name] = adf
    arr = adf['agreement'].values
    print(f'{name:<12} {arr.mean():>6.3f} {np.median(arr):>7.3f} {arr.min():>6.3f} '
          f'{(arr < 0.8).sum():>6} {(arr < 0.6).sum():>6} {(arr == 1.0).sum():>6}')

# === Part 2: Does disagreement correlate with misclassification? ===
print('\n=== Agreement vs Correctness ===')
print(f'{"Task":<12} {"Correct(agree>=90%)":>20} {"Correct(agree<90%)":>20} {"Diff":>8}')
print('-' * 65)

for name, adf in agreement_data.items():
    adf['correct'] = (adf['mode'] == adf['label']).astype(int)
    high = adf[adf['agreement'] >= 0.9]
    low = adf[adf['agreement'] < 0.9]
    h_acc = high['correct'].mean() if len(high) > 0 else 0
    l_acc = low['correct'].mean() if len(low) > 0 else 0
    diff = h_acc - l_acc
    print(f'{name:<12} {h_acc:>8.3f} ({len(high):>3})     {l_acc:>8.3f} ({len(low):>3})     {diff:>+7.3f}')

# === Part 3: Distribution of unique predictions per video ===
print('\n=== Number of Distinct Predicted Classes per Video ===')
print(f'{"Task":<12} {"1 class":>8} {"2 classes":>10} {"3+ classes":>11}')
print('-' * 45)

for col, (name, _) in tasks.items():
    nuniq = pred_df.groupby('video_id')[col].nunique()
    n1 = (nuniq == 1).sum()
    n2 = (nuniq == 2).sum()
    n3 = (nuniq >= 3).sum()
    print(f'{name:<12} {n1:>8} {n2:>10} {n3:>11}')

# === Part 4: Low-agreement videos detail ===
print('\n=== Low Agreement Videos (agreement < 70%, RW-ROP) ===')
rw_adf = agreement_data['RW-ROP']
low_rw = rw_adf[rw_adf['agreement'] < 0.70].sort_values('agreement')
if len(low_rw) > 0:
    print(f'{"video_id":<20} {"agreement":>10} {"mode":>6} {"label":>6} {"n_img":>6} {"correct":>8}')
    print('-' * 60)
    for _, row in low_rw.iterrows():
        print(f'{row["video_id"]:<20} {row["agreement"]:>10.3f} {int(row["mode"]):>6} {int(row["label"]):>6} '
              f'{int(row["n_images"]):>6} {"Y" if row["correct"] else "N":>8}')
else:
    print('None')

print(f'\nTotal low agreement (<70%): {len(low_rw)} videos')
