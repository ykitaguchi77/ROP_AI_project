# 福嶋先生への回答 (2026-02-13)

**対象レポート**:
- `report_rwrop_threshold_optimization.pdf` (2026-02-06作成)
- `report_top10_majority_vote.pdf` (2026-02-11作成)
- `report_prauc_supplement.pdf` (2026-02-13作成)

**データセット**: Multicenter ROP Study (6,448画像, 347動画, 5施設)
**モデル**: clinical_v3 (EfficientNet-B0 + 臨床データ融合, 5タスク同時学習)

---

## 回答(a): 解析対象動画IDリストと除外動画について

### 概要

| 項目 | 動画数 |
|------|--------|
| 患者データ全体 | 409 |
| 解析に使用された動画 | **347** |
| 除外された動画 | **62** |

### 解析に使用された347動画のリスト

`training_video_list_clinical_v3.xlsx` に全347動画の詳細を記載しています。

#### 施設別内訳

| 施設 | 動画数 |
|------|--------|
| OWCH (母子) | 109 |
| FU (福岡) | 90 |
| AMU (愛知) | 64 |
| KCMC (神奈川) | 43 |
| YCH | 41 |
| **合計** | **347** |

#### 5-Fold CV 割当

| Fold | 動画数 |
|------|--------|
| Fold 1 | 71 |
| Fold 2 | 67 |
| Fold 3 | 66 |
| Fold 4 | 69 |
| Fold 5 | 74 |

### 除外された62動画の内訳

除外理由の詳細は `excluded_videos_from_analysis.csv` に記載しています。

| 除外理由 | 動画数 | 説明 |
|----------|--------|------|
| Bad/Worst only | 47 | トレーニング時点でGood/Fair品質の画像が0枚 |
| No Kubota selection | 15 | 久保田先生による画像選定データなし（OMPU 8件含む） |
| **合計** | **62** | |

> **補足**: 0098_OWCHは現在Fair画像が1枚存在しますが、この画像はトレーニングデータスナップショット (2026-01-27) より前の分類バッチ (2026-01-18) で追加されたもので、トレーニング時のデータに含まれていませんでした。同様に、他の9動画でも計43枚のFair画像がトレーニング後に追加されています。

### 除外動画のうち治療例 (treatment_needed = 1)

**7動画が治療例でしたが、いずれもBad/Worst品質のため除外されています。**

| video_id | 施設 | GA | BW | PMA | Zone | Stage | Plus | 除外理由 |
|----------|------|----|----|-----|------|-------|------|----------|
| 0016_AMU | 愛知 | 23 | 334 | 33 | 1 | 2 | 2 | Bad/Worst only |
| 0017_AMU | 愛知 | 23 | 334 | 33 | 1 | 2 | 2 | Bad/Worst only |
| 0220_FU | 福岡 | 26 | 859 | 37 | 2 | 3 | 2 | Bad/Worst only |
| 0221_FU | 福岡 | 26 | 859 | 37 | 2 | 3 | 2 | Bad/Worst only |
| 0261_YCH | YCH | 24 | 640 | 42 | 2 | 3 | 2 | Bad/Worst only |
| 0262_YCH | YCH | 24 | 640 | 42 | 2 | 3 | 2 | Bad/Worst only |
| 0274_YCH | YCH | 23 | 697 | 31 | 1 | 3 | 2 | Bad/Worst only |

> **注**: 0016/0017_AMU、0220/0221_FU、0261/0262_YCH はそれぞれ同一患者の異なる検査動画です。実質的には4患者分の治療例が除外されています。これらは撮影品質が低く、Good/Fair画像が得られなかったため解析対象外となりました。

---

## 回答(b): Treatment (TR-ROP) False Negative のIDリスト

### 福嶋先生の仮説

> 「実際、症例ベースで仮定すると治療例20人のうち1例見逃し、画像ベースでも治療画像の900枚中30-40画像が治療すべき画像を見逃したという計算なので同一症例でも正しく判定した画像と見逃した症例が混ざっているだけなのでは？と思っており、すでに実質感度100を達成しているのではないかと期待を持っています。」

### 結論: **福嶋先生の仮説は正しく、Soft Vote Top-5/Top-10 で Treatment Sensitivity = 1.000 を達成しています。**

### Treatment 画像単位の結果 (5-fold CV, default threshold 0.50)

- Treatment 陽性動画数: **33** (627画像)
- FN画像数: **52 / 627** (8.3%)
- FN画像を含む動画数: **10 / 33**
- **完全見逃し動画（全画像FN）: 0** — 全33動画で少なくとも1枚はTPが存在

| video_id | FN画像数/全画像数 | FN率 | mean P(Treatment) | 備考 |
|----------|-------------------|------|-------------------|------|
| 0034_KCMC | 13/22 | 59% | 0.490 | SV Allで唯一のFN |
| 0259_OWCH | 9/22 | 41% | 0.554 | SVでは正しく判定 |
| 0009_AMU | 12/30 | 40% | 0.625 | SVでは正しく判定 |
| 0058_KCMC | 5/15 | 33% | 0.632 | SVでは正しく判定 |
| 0080_YCH | 1/4 | 25% | 0.648 | SVでは正しく判定 |
| 0144_OWCH | 6/27 | 22% | 0.625 | SVでは正しく判定 |
| 0048_KCMC | 3/27 | 11% | 0.667 | SVでは正しく判定 |
| 0077_OWCH | 1/9 | 11% | 0.814 | SVでは正しく判定 |
| 0008_AMU | 1/30 | 3% | 0.822 | SVでは正しく判定 |
| 0014_AMU | 1/29 | 3% | 0.912 | SVでは正しく判定 |

### Treatment 患者単位 Soft Vote の結果

| 評価条件 | FN動画数 | Sensitivity | 備考 |
|----------|----------|-------------|------|
| Per-Image (All) | — | 0.9171 | 画像単位: 575/627 TP |
| **SV All** | **1/33** | **0.9697** | FN: 0034_KCMC (P=0.490) |
| **SV Top-10** | **0/33** | **1.0000** | FN動画なし |
| **SV Top-5** | **0/33** | **1.0000** | FN動画なし |

### 解釈

1. **画像単位では52枚のFNが存在するが、全10動画でTPとFNが混在**しており、完全見逃し（全画像FN）の動画は0です。
2. **Soft Vote All で唯一のFN動画は 0034_KCMC** (Stage 3, P(Treatment)=0.490)。22枚中13枚がFNですが、残り9枚のTPがあるため、平均確率は0.490と閾値ギリギリです。
3. **Top-5/Top-10 品質フィルタリング後のSoft Vote では Treatment Sensitivity = 1.000** を達成。低品質画像を除外することで、0034_KCMC の平均確率が閾値 0.5 を超えるようになったと考えられます。
4. **福嶋先生のご指摘通り、同一症例内でTP画像とFN画像が混在しているだけであり、患者レベルでは実質的に感度100%です。**

### 補足: RW-ROP (Plus OR Stage 3 OR Zone I) の False Negative

RW-ROPのFNリストは `rwrop_false_negatives_summary.md` および対応するCSVファイルに詳細を記載しています。

**要約** (Report 2: Majority Vote, SV All):
- RW-ROP陽性動画: 84
- FN動画: 11 (Sensitivity = 0.869)
- Zone I の見逃し (Zone II と誤判定) が最多 (7/11動画)
- 最も困難な症例: 0217_FU (Zone I + Stage 3, P(RW)=0.293)

> **Treatment と RW-ROP の感度差の理由**: Treatment (33動画) は RW-ROP (84動画) のサブセットです。RW-ROP FN 11動画のうち、Treatment陽性は0034_KCMCのみです。残り10動画は「治療不要だが紹介が必要」なケース（例: Zone I のみ）であり、臨床的影響度はTreatment FNより低いと言えます。

---

## 回答(c): PR-AUC (Precision-Recall AUC) の結果

### 概要

陽性ケースが少ない（Treatment 9.7%, AROP 3.5%, RW-ROP 24.5%）ため、ROC-AUCに加えてPR-AUCを算出しました。詳細は `report_prauc_supplement.pdf` (8ページ) に記載しています。

### Per-Fold CV 結果 (Mean +/- SD)

| タスク | 陽性率 | ROC-AUC | PR-AUC | 差 (ROC-PR) |
|--------|--------|---------|--------|-------------|
| Treatment | 9.8% | 0.975 +/- 0.025 | 0.860 +/- 0.125 | +0.116 |
| AROP | 3.6% | 0.888 +/- 0.131 | 0.614 +/- 0.261 | +0.273 |
| RW-ROP | 22.1% | 0.954 +/- 0.018 | 0.889 +/- 0.050 | +0.066 |

### 患者単位 Soft Vote PR-AUC

| Task | Per-Image | SV All | SV Top-10 | SV Top-5 |
|------|-----------|--------|-----------|----------|
| Treatment (PR-AUC) | 0.880 | **0.925** | 0.923 | 0.905 |
| AROP (PR-AUC) | 0.649 | 0.675 | 0.660 | 0.617 |
| RW-ROP (PR-AUC) | 0.892 | **0.898** | 0.891 | 0.885 |

### 解釈

1. **Treatment**: PR-AUC 0.860 はランダム分類器の陽性率 (0.098) の約9倍 → 実質的に高い性能。Soft Vote で 0.925 まで改善。
2. **AROP**: ROC-AUC (0.888) と PR-AUC (0.614) の乖離が最大。陽性率 3.5% の強いクラス不均衡が原因。ROC-AUC の数値ほど臨床的精度は高くない。
3. **RW-ROP**: PR-AUC 0.889 はスクリーニング指標として実用的な水準。

> **発表用グラフ**: `report_prauc_supplement.pdf` にROC曲線とPR曲線のグラフを含めています。必要に応じて高解像度の図を再出力できます。

---

## 回答(d): Top-5画像フィルタリングにおける画像不足33動画の詳細

### 結論

1. **画像不足の33動画は除外されず、選出できた画像（1〜4枚）のまま解析に使用されています。**
2. **該当画像が0枚の動画はありません。** 全347動画で少なくとも1枚以上の有効画像が存在。
3. **最小選出数は1枚。**

### Top-5 選出の内訳

| 項目 | 値 |
|------|-----|
| 全画像数 | 6,448 |
| Top-5選出後 | 1,650 |
| 対象動画数 | 347 / 347（全動画カバー） |
| Stage1のみで十分 | 301 (86.7%) |
| Stage2補完あり | 13 (3.7%) |
| **画像不足 (<5枚)** | **33 (9.5%)** |
| 特徴量なし (0枚) | **0** |
| 最小選出数 | **1枚** |
| 平均選出数 | 4.8枚 |

### 選出アルゴリズム

```
動画ごとの全Good+Fair画像
    ↓
[有効画像フィルタ] retina_ratio > 0 かつ 非NaN
    ↓
  有効画像 = 0枚 → スキップ ★今回は該当なし
  有効画像 ≥ 1枚 → 次のステージへ
    ↓
[Stage 1] disc_edge_coverage_ratio >= 0.80 の画像を
          0.4×retina + 0.4×grad + 0.2×mbss のスコアで降順ソート → 上位K枚
    ↓
  K枚に満たない場合
    ↓
[Stage 2 (Fallback)] 残りの有効画像を retina_ratio 降順で補完
    ↓
  それでもK枚に満たない場合 → 「画像不足」にカウント（ある分だけで解析）
```

### 臨床的留意点

- 画像枚数が少ない動画では、Soft Vote の信頼性が低下する可能性があります。
- ただし、Top-5でもTreatment Sensitivity = 1.000 を達成しており、画像不足による見逃しは発生していません。

> **詳細レポート**: `top10_quality_filter_insufficient_images_report.md` に選出アルゴリズムのコード詳細と、Top-10の結果も記載しています。

---

## 参照ファイル一覧

| ファイル名 | 内容 |
|-----------|------|
| `report_rwrop_threshold_optimization.pdf` | 閾値最適化レポート (2026-02-06) |
| `report_top10_majority_vote.pdf` | Top-K Majority Vote レポート (2026-02-11) |
| `report_prauc_supplement.pdf` | PR-AUC補足解析レポート (2026-02-13) |
| `training_video_list_clinical_v3.xlsx` | 解析対象347動画リスト |
| `excluded_videos_from_analysis.csv` | 除外62動画リスト（除外理由付き） |
| `rwrop_false_negatives_summary.md` | RW-ROP FN動画IDサマリ |
| `rwrop_false_negatives_threshold_optimization.csv` | Report 1 FN詳細CSV |
| `rwrop_false_negatives_majority_vote.csv` | Report 2 FN詳細CSV |
| `top10_quality_filter_insufficient_images_report.md` | 画像不足動画の詳細レポート |
