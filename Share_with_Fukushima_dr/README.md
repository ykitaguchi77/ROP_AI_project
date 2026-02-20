# Share_with_Fukushima_dr

福島先生との共有ドキュメント一覧

**作成日**: 2026-02-13
**対象モデル**: clinical_v3 (EfficientNet-B0 + 臨床データ融合, 5-fold CV)
**データセット**: Multicenter ROP Study (347動画, 6,448画像, 5施設)

---

## ファイル一覧

### 1. 学習動画リスト

| ファイル | 説明 |
|---------|------|
| `training_video_list_clinical_v3.xlsx` | clinical_v3モデルの学習に使用した全347動画のリスト |

**Excelシート構成:**
- **Summary**: モデル概要、施設別・Fold別の動画数集計
- **Video_List**: 全347動画一覧（Video_ID, Center, Fold, 画像数, Good/Fair数, GA, BW, PMA, Sex, 各ラベル）
- **Center_AMU / FU / KCMC / OWCH / YCH**: 施設別の動画リスト

| 施設 | 動画数 | 画像数 |
|------|--------|--------|
| AMU | 64 | - |
| FU | 90 | - |
| KCMC | 43 | - |
| OWCH | 109 | - |
| YCH | 41 | - |
| **合計** | **347** | **6,448** |

---

### 2. RW-ROP False Negative 解析

| ファイル | 説明 |
|---------|------|
| `rwrop_false_negatives_summary.md` | RW-ROP False Negative動画IDの詳細サマリ（閾値別・Majority Vote別） |
| `rwrop_false_negatives_threshold_optimization.csv` | Report 1（閾値最適化）のFN動画詳細CSV |
| `rwrop_false_negatives_majority_vote.csv` | Report 2（Majority Vote）のFN動画詳細CSV |
| `extract_rwrop_false_negatives.py` | FN動画抽出に使用したPythonスクリプト |

**主要な知見:**
- 両レポート共通の完全見逃し動画: **0217_FU**（Zone I + Stage 3, P(RW)=0.293）
- FNの主因: Zone I → Zone II の誤判定（11動画中7動画）
- P(RW)は0.28〜0.67の範囲で判断境界（0.5）付近に集中

---

### 3. Top-K画像品質フィルタリング解析

| ファイル | 説明 |
|---------|------|
| `top10_quality_filter_insufficient_images_report.md` | 画像不足動画（33 videos）の取り扱いに関する詳細レポート |

**主要な知見:**
- 画像不足の動画は除外されず、選出できた画像（最小1枚）のままで解析に使用
- 全347動画で少なくとも1枚以上の有効画像が存在（0枚の動画はなし）
- Top-5選出時: 33動画が5枚未満で解析、Top-10選出時: 74動画が10枚未満で解析
