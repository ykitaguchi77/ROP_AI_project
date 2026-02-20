"""
RW-ROP False Negative 動画ID抽出スクリプト

2つのレポートに対応するFalse Negative (見逃し) video_idを特定する:
1. report_rwrop_threshold_optimization.pdf: 画像単位 + 閾値最適化
2. report_top10_majority_vote.pdf: 患者単位 Soft Vote (All / Top-10 / Top-5)

RW-ROP定義: Plus disease (Plus=2) OR Stage 3 (Stage=3) OR Zone I (Zone=0)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import sys

# === Paths ===
BASE = Path(r'C:\Users\ykita\ROP_AI_project\ROP_project\multicenter_study')
PRED_PATH = BASE / 'outputs_clinical_v3' / 'predictions.csv'
KUBOTA_EXCEL = Path(r'E:\Multicenter_ROP_study\Multicenter_images\Kubota_selection\selected_images_disc_retina.xlsx')
TOP_EXCEL = Path(r'E:\Multicenter_ROP_study\Multicenter_images\selected_images_disc_retina.xlsx')
OUTPUT_DIR = Path(r'C:\Users\ykita\ROP_AI_project\Share_with_Fukushima_dr')

sys.path.insert(0, str(BASE))
from select_best_images import minmax_norm

# === Constants ===
EDGE_COVERAGE_CUTOFF = 0.80
WEIGHT_RETINA = 0.4
WEIGHT_GRAD = 0.4
WEIGHT_MBSS = 0.2


def load_and_merge_data():
    pred_df = pd.read_csv(PRED_PATH)
    pred_df['image_name'] = pred_df['image_path'].apply(lambda x: Path(x).name)

    feat_cols = ['retina_ratio', 'mbss_Grad_p90', 'mbss_score', 'disc_detected', 'disc_edge_coverage_ratio']
    kubota_df = pd.read_excel(KUBOTA_EXCEL)
    avail = [c for c in feat_cols if c in kubota_df.columns]
    kubota_features = kubota_df[['image_name'] + avail].drop_duplicates(subset='image_name', keep='first')
    merged_df = pred_df.merge(kubota_features, on='image_name', how='left')

    tl = pd.read_excel(TOP_EXCEL)
    tl_feat = [c for c in avail if c in tl.columns]
    if 'image_name' in tl.columns and tl_feat:
        tl_features = tl[['image_name'] + tl_feat].drop_duplicates(subset='image_name', keep='first').set_index('image_name')
        mm = merged_df['retina_ratio'].isna()
        for col in tl_feat:
            fill = merged_df.loc[mm, 'image_name'].map(tl_features[col])
            merged_df.loc[mm, col] = fill.values

    return merged_df


def select_top_k_per_video(df, top_k=10, edge_cov_cutoff=0.80):
    all_selected = []
    for vid, group in df.groupby('video_id'):
        valid = group[group['retina_ratio'].notna() & (group['retina_ratio'] > 0)].copy()
        if len(valid) == 0:
            continue
        stage1 = valid[
            (valid['disc_detected'] == True) &
            valid['disc_edge_coverage_ratio'].notna() &
            (valid['disc_edge_coverage_ratio'] >= edge_cov_cutoff)
        ].copy()
        if len(stage1) > 0:
            stage1['retina_norm'] = minmax_norm(stage1['retina_ratio'].fillna(0))
            stage1['grad_norm'] = minmax_norm(stage1['mbss_Grad_p90'].fillna(0))
            stage1['mbss_norm'] = minmax_norm(stage1['mbss_score'].fillna(0))
            stage1['quality_score'] = (
                WEIGHT_RETINA * stage1['retina_norm'] +
                WEIGHT_GRAD * stage1['grad_norm'] +
                WEIGHT_MBSS * stage1['mbss_norm']
            )
            stage1 = stage1.sort_values('quality_score', ascending=False)
            sel = stage1.head(top_k).copy()
        else:
            sel = pd.DataFrame()
        nr = top_k - len(sel)
        if nr > 0:
            rem = valid[~valid.index.isin(sel.index)].sort_values('retina_ratio', ascending=False)
            sel = pd.concat([sel, rem.head(nr)])
        if len(sel) > 0:
            all_selected.append(sel)
    return pd.concat(all_selected, ignore_index=True) if all_selected else pd.DataFrame()


def aggregate_soft_vote(df):
    rows = []
    for vid, group in df.groupby('video_id'):
        row = {'video_id': vid, 'n_images': len(group)}
        row['zone_label'] = int(group['zone_label'].iloc[0])
        row['stage_label'] = int(group['stage_label'].iloc[0])
        row['plus_label'] = int(group['plus_label'].iloc[0])
        row['treatment_label'] = int(group['treatment_label'].iloc[0])

        zone_probs = group[['zone_prob_0', 'zone_prob_1', 'zone_prob_2']].mean()
        row['zone_pred'] = int(np.argmax(zone_probs.values))
        row['zone_prob_0'] = float(zone_probs['zone_prob_0'])

        stage_probs = group[['stage_prob_0', 'stage_prob_1', 'stage_prob_2', 'stage_prob_3']].mean()
        row['stage_pred'] = int(np.argmax(stage_probs.values))
        row['stage_prob_3'] = float(stage_probs['stage_prob_3'])

        plus_probs = group[['plus_prob_0', 'plus_prob_1', 'plus_prob_2']].mean()
        row['plus_pred'] = int(np.argmax(plus_probs.values))
        row['plus_prob_2'] = float(plus_probs['plus_prob_2'])

        treat_prob = float(group['treatment_prob_1'].mean())
        row['treatment_prob_1'] = treat_prob
        row['treatment_pred'] = int(treat_prob >= 0.5)

        rows.append(row)
    return pd.DataFrame(rows)


def compute_rw_rop(df):
    rw_true = ((df['plus_label'] == 2) | (df['stage_label'] == 3) | (df['zone_label'] == 0)).astype(int)
    rw_pred_hard = ((df['plus_pred'] == 2) | (df['stage_pred'] == 3) | (df['zone_pred'] == 0)).astype(int)
    rw_prob = 1 - ((1 - df['plus_prob_2']) * (1 - df['stage_prob_3']) * (1 - df['zone_prob_0']))
    return rw_true, rw_pred_hard, rw_prob


def get_rw_rop_components(row):
    """RW-ROP陽性の構成要素を返す"""
    components = []
    if row.get('zone_label', -1) == 0:
        components.append('Zone I')
    if row.get('stage_label', -1) == 3:
        components.append('Stage 3')
    if row.get('plus_label', -1) == 2:
        components.append('Plus')
    return ', '.join(components) if components else 'None'


def find_fn_video_ids(df, threshold=0.5, use_hard=False):
    """RW-ROP True Positive だが予測 Negative の video_id を特定"""
    rw_true, rw_pred_hard, rw_prob = compute_rw_rop(df)

    if use_hard:
        rw_pred = rw_pred_hard
    else:
        rw_pred = (rw_prob >= threshold).astype(int)

    # FN: true=1, pred=0
    fn_mask = (rw_true == 1) & (rw_pred == 0)
    fn_df = df[fn_mask].copy()
    fn_df['rw_rop_true'] = 1
    fn_df['rw_rop_pred'] = 0
    fn_df['rw_rop_prob'] = rw_prob[fn_mask].values
    fn_df['rw_rop_components'] = fn_df.apply(get_rw_rop_components, axis=1)

    total_positive = rw_true.sum()
    return fn_df, total_positive


def main():
    print('Loading data...')
    merged_df = load_and_merge_data()
    print(f'Loaded: {len(merged_df)} images, {merged_df["video_id"].nunique()} video_ids')

    # === Report 1: 閾値最適化レポート (画像単位→video単位集約) ===
    # 画像単位のRW-ROP判定をvideo単位に集約して報告
    # report内ではdefault(0.5), sens>=95%(0.443), Youden's J(0.569) の3閾値を報告

    print('\n' + '='*80)
    print('Report 1: RW-ROP閾値最適化レポート (report_rwrop_threshold_optimization.pdf)')
    print('  → 画像単位5-fold CV評価のFN動画')
    print('='*80)

    thresholds_report1 = {
        'Default (0.50)': 0.50,
        'Sens>=95% (0.443)': 0.443,
        "Youden's J (0.569)": 0.569,
    }

    report1_results = {}
    for name, thr in thresholds_report1.items():
        # 画像単位でRW-ROPを計算し、video_id単位でFNを集約
        rw_true, _, rw_prob = compute_rw_rop(merged_df)
        rw_pred = (rw_prob >= thr).astype(int)
        merged_df['_rw_true'] = rw_true.values
        merged_df['_rw_pred'] = rw_pred.values
        merged_df['_rw_prob'] = rw_prob.values

        # 各video_idで: true=RW-ROPなのに、全画像の過半数がnegativeと判定された場合をFNとする
        # → ただし、report1は画像単位なので、video_id単位でuniqueなFN video_idを列挙
        # 少なくとも1枚でもFNがある動画 = 見逃しリスクのある動画
        fn_images = merged_df[(merged_df['_rw_true'] == 1) & (merged_df['_rw_pred'] == 0)]
        fn_video_ids = fn_images['video_id'].unique()

        # 各FN動画の詳細
        fn_details = []
        for vid in sorted(fn_video_ids):
            vid_data = merged_df[merged_df['video_id'] == vid]
            vid_rw_true = vid_data['_rw_true'].iloc[0]
            n_total = len(vid_data)
            n_fn = ((vid_data['_rw_true'] == 1) & (vid_data['_rw_pred'] == 0)).sum()
            n_tp = ((vid_data['_rw_true'] == 1) & (vid_data['_rw_pred'] == 1)).sum()
            mean_prob = vid_data['_rw_prob'].mean()
            components = get_rw_rop_components(vid_data.iloc[0])
            fn_details.append({
                'video_id': vid,
                'rw_rop_components': components,
                'n_images': n_total,
                'n_FN_images': n_fn,
                'n_TP_images': n_tp,
                'FN_rate': n_fn / n_total,
                'mean_rw_rop_prob': mean_prob,
            })

        fn_details_df = pd.DataFrame(fn_details)
        total_pos_videos = merged_df[merged_df['_rw_true'] == 1]['video_id'].nunique()

        print(f'\n--- {name} (threshold={thr}) ---')
        print(f'  RW-ROP陽性動画数: {total_pos_videos}')
        print(f'  FN画像を含む動画数: {len(fn_video_ids)}')
        if len(fn_details_df) > 0:
            # 全画像がFNの動画（完全な見逃し）
            complete_fn = fn_details_df[fn_details_df['FN_rate'] == 1.0]
            print(f'  うち全画像FNの動画（完全見逃し）: {len(complete_fn)}')
            print(f'  FN動画ID一覧:')
            for _, row in fn_details_df.iterrows():
                marker = ' [完全見逃し]' if row['FN_rate'] == 1.0 else ''
                print(f'    {row["video_id"]}: {row["rw_rop_components"]} | '
                      f'FN={row["n_FN_images"]}/{row["n_images"]} | '
                      f'mean_P(RW)={row["mean_rw_rop_prob"]:.3f}{marker}')

        report1_results[name] = fn_details_df

    merged_df.drop(columns=['_rw_true', '_rw_pred', '_rw_prob'], inplace=True)

    # === Report 2: Top-10 Majority Vote レポート ===
    print('\n' + '='*80)
    print('Report 2: Top-10 Majority Vote レポート (report_top10_majority_vote.pdf)')
    print('  → 患者単位 Soft Vote のFN動画')
    print('='*80)

    top10_df = select_top_k_per_video(merged_df, top_k=10)
    top5_df = select_top_k_per_video(merged_df, top_k=5)
    print(f'Top-10: {len(top10_df)} images, Top-5: {len(top5_df)} images')

    sv_conditions = {
        'SV All': merged_df,
        'SV Top-10': top10_df,
        'SV Top-5': top5_df,
    }

    report2_results = {}
    for label, src_df in sv_conditions.items():
        agg_df = aggregate_soft_vote(src_df)
        rw_true, rw_pred_hard, rw_prob = compute_rw_rop(agg_df)
        rw_pred = rw_pred_hard  # Report 2 uses hard prediction (OR of component preds)
        fn_mask = (rw_true == 1) & (rw_pred == 0)
        fn_df = agg_df[fn_mask].copy()

        total_pos = rw_true.sum()
        fn_df['rw_rop_components'] = fn_df.apply(get_rw_rop_components, axis=1)
        fn_df['rw_rop_prob'] = rw_prob[fn_mask].values

        print(f'\n--- {label} (hard OR prediction, default threshold) ---')
        print(f'  RW-ROP陽性動画数: {total_pos}')
        print(f'  FN動画数: {len(fn_df)}')
        print(f'  Sensitivity: {1 - len(fn_df)/total_pos:.4f}')
        if len(fn_df) > 0:
            print(f'  FN動画ID一覧:')
            for _, row in fn_df.iterrows():
                zone_pred_str = f'Zone {"I" if row["zone_pred"]==0 else "II" if row["zone_pred"]==1 else "III"}'
                stage_pred_str = f'Stage {row["stage_pred"]}'
                plus_pred_str = f'Plus {"Normal" if row["plus_pred"]==0 else "PrePlus" if row["plus_pred"]==1 else "Plus"}'
                print(f'    {row["video_id"]}: True={row["rw_rop_components"]} | '
                      f'Pred: {zone_pred_str}, {stage_pred_str}, {plus_pred_str} | '
                      f'P(RW)={row["rw_rop_prob"]:.3f} | n_images={row["n_images"]}')

        report2_results[label] = fn_df

    # === 出力: CSV + テキストサマリ ===
    print('\n' + '='*80)
    print('出力ファイル生成中...')
    print('='*80)

    # --- Report 1 CSV ---
    all_report1_rows = []
    for name, df in report1_results.items():
        if len(df) > 0:
            df_out = df.copy()
            df_out['threshold_condition'] = name
            all_report1_rows.append(df_out)
    if all_report1_rows:
        report1_csv = pd.concat(all_report1_rows, ignore_index=True)
        # 全画像FN（完全見逃し）のみに絞った列を追加
        report1_csv['complete_miss'] = report1_csv['FN_rate'] == 1.0
        csv_path1 = OUTPUT_DIR / 'rwrop_false_negatives_threshold_optimization.csv'
        report1_csv.to_csv(csv_path1, index=False, encoding='utf-8-sig')
        print(f'  Report 1 CSV: {csv_path1}')

    # --- Report 2 CSV ---
    all_report2_rows = []
    for label, df in report2_results.items():
        if len(df) > 0:
            df_out = df[['video_id', 'n_images', 'zone_label', 'stage_label', 'plus_label',
                         'zone_pred', 'stage_pred', 'plus_pred',
                         'zone_prob_0', 'stage_prob_3', 'plus_prob_2',
                         'rw_rop_components', 'rw_rop_prob']].copy()
            df_out['condition'] = label
            all_report2_rows.append(df_out)
    if all_report2_rows:
        report2_csv = pd.concat(all_report2_rows, ignore_index=True)
        csv_path2 = OUTPUT_DIR / 'rwrop_false_negatives_majority_vote.csv'
        report2_csv.to_csv(csv_path2, index=False, encoding='utf-8-sig')
        print(f'  Report 2 CSV: {csv_path2}')

    # --- 統合テキストサマリ ---
    summary_lines = []
    summary_lines.append('='*80)
    summary_lines.append('RW-ROP (TR-ROP) False Negative 動画ID サマリ')
    summary_lines.append(f'作成日: 2026-02-13')
    summary_lines.append(f'データセット: Multicenter ROP Study (6,448画像, 347 video_ids)')
    summary_lines.append(f'モデル: clinical_v3 (EfficientNet-B0 + 臨床データ融合, 5タスク)')
    summary_lines.append('')
    summary_lines.append('RW-ROP定義: Plus disease OR Stage 3 OR Zone I')
    summary_lines.append('='*80)

    summary_lines.append('')
    summary_lines.append('■ Report 1: 閾値最適化レポート (report_rwrop_threshold_optimization.pdf)')
    summary_lines.append('  評価方法: 画像単位5-fold CV → video_id単位でFNを集約')
    summary_lines.append('')

    for name, df in report1_results.items():
        total_pos_videos = len(merged_df[((merged_df['plus_label'] == 2) |
                                           (merged_df['stage_label'] == 3) |
                                           (merged_df['zone_label'] == 0))]['video_id'].unique())
        complete_fn = df[df['FN_rate'] == 1.0] if len(df) > 0 else pd.DataFrame()
        summary_lines.append(f'  [{name}]')
        summary_lines.append(f'    RW-ROP陽性動画: {total_pos_videos}')
        summary_lines.append(f'    FN画像を含む動画: {len(df)}')
        summary_lines.append(f'    完全見逃し動画（全画像FN）: {len(complete_fn)}')
        if len(complete_fn) > 0:
            summary_lines.append(f'    完全見逃し動画ID: {", ".join(sorted(complete_fn["video_id"].values))}')
        if len(df) > 0:
            summary_lines.append(f'    全FN動画ID:')
            for _, row in df.sort_values('FN_rate', ascending=False).iterrows():
                marker = ' *** 完全見逃し ***' if row['FN_rate'] == 1.0 else ''
                summary_lines.append(
                    f'      {row["video_id"]}: {row["rw_rop_components"]} | '
                    f'FN={row["n_FN_images"]}/{row["n_images"]} ({row["FN_rate"]*100:.0f}%) | '
                    f'mean P(RW)={row["mean_rw_rop_prob"]:.3f}{marker}')
        summary_lines.append('')

    summary_lines.append('')
    summary_lines.append('■ Report 2: Top-10 Majority Vote レポート (report_top10_majority_vote.pdf)')
    summary_lines.append('  評価方法: 患者単位 Soft Vote → hard OR prediction (default threshold)')
    summary_lines.append('')

    for label, df in report2_results.items():
        rw_true_count = None
        # Recompute to get total positive count
        if label == 'SV All':
            agg = aggregate_soft_vote(merged_df)
        elif label == 'SV Top-10':
            agg = aggregate_soft_vote(top10_df)
        else:
            agg = aggregate_soft_vote(top5_df)
        rw_true, _, _ = compute_rw_rop(agg)
        total_pos = int(rw_true.sum())

        summary_lines.append(f'  [{label}]')
        summary_lines.append(f'    RW-ROP陽性動画: {total_pos}')
        summary_lines.append(f'    FN動画数: {len(df)}')
        summary_lines.append(f'    Sensitivity: {1 - len(df)/total_pos:.4f}' if total_pos > 0 else '    Sensitivity: N/A')
        if len(df) > 0:
            summary_lines.append(f'    FN動画ID:')
            for _, row in df.iterrows():
                zone_str = ['I', 'II', 'III'][row['zone_pred']]
                plus_str = ['Normal', 'PrePlus', 'Plus'][row['plus_pred']]
                summary_lines.append(
                    f'      {row["video_id"]}: True={row["rw_rop_components"]} | '
                    f'Pred: Zone {zone_str}, Stage {row["stage_pred"]}, {plus_str} | '
                    f'P(RW)={row["rw_rop_prob"]:.3f}')
        summary_lines.append('')

    # 両レポート共通のFN動画IDを特定
    summary_lines.append('')
    summary_lines.append('■ 両レポート共通のFN動画ID (Report 2 SV All, default threshold)')
    report1_default_fn = set(report1_results.get('Default (0.50)', pd.DataFrame()).get('video_id', []))
    report1_default_complete_fn = set()
    if len(report1_results.get('Default (0.50)', pd.DataFrame())) > 0:
        complete = report1_results['Default (0.50)']
        report1_default_complete_fn = set(complete[complete['FN_rate'] == 1.0]['video_id'])
    report2_sv_all_fn = set(report2_results.get('SV All', pd.DataFrame()).get('video_id', []))

    common_fn = report1_default_complete_fn & report2_sv_all_fn
    summary_lines.append(f'  Report 1 完全見逃し (default 0.5): {len(report1_default_complete_fn)} 動画')
    summary_lines.append(f'  Report 2 SV All FN: {len(report2_sv_all_fn)} 動画')
    summary_lines.append(f'  共通FN: {len(common_fn)} 動画')
    if common_fn:
        summary_lines.append(f'  共通FN動画ID: {", ".join(sorted(common_fn))}')

    summary_text = '\n'.join(summary_lines)
    txt_path = OUTPUT_DIR / 'rwrop_false_negatives_summary.txt'
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    print(f'  Summary TXT: {txt_path}')

    # Print summary
    print('\n' + summary_text)


if __name__ == '__main__':
    main()
