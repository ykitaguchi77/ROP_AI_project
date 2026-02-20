# RW-ROP (TR-ROP) False Negative 動画ID サマリ

- **作成日**: 2026-02-13
- **データセット**: Multicenter ROP Study (6,448画像, 347 video_ids)
- **モデル**: clinical_v3 (EfficientNet-B0 + 臨床データ融合, 5タスク)
- **RW-ROP定義**: Plus disease OR Stage 3 OR Zone I

---

## Report 1: 閾値最適化レポート (report_rwrop_threshold_optimization.pdf)

**評価方法**: 画像単位5-fold CV → video_id単位でFNを集約

### Default (閾値 0.50)

- RW-ROP陽性動画: **84**
- FN画像を含む動画: **23**
- 完全見逃し動画（全画像FN）: **1** (0217_FU)

| video_id | RW-ROP構成要素 | FN画像数/全画像数 | FN率 | mean P(RW) | 備考 |
|----------|--------------|----------------|------|------------|------|
| 0217_FU | Zone I, Stage 3 | 8/8 | 100% | 0.293 | **完全見逃し** |
| 0128_OWCH | Stage 3 | 13/14 | 93% | 0.423 | |
| 0400_KCMC | Zone I | 19/21 | 90% | 0.384 | |
| 0215_FU | Stage 3 | 3/4 | 75% | 0.452 | |
| 0268_OWCH | Stage 3 | 6/9 | 67% | 0.490 | |
| 0390_KCMC | Zone I | 16/25 | 64% | 0.452 | |
| 0168_OWCH | Zone I | 8/13 | 62% | 0.499 | |
| 0127_OWCH | Stage 3 | 6/13 | 46% | 0.523 | |
| 0204_AMU | Stage 3 | 10/24 | 42% | 0.562 | |
| 0214_FU | Zone I | 2/5 | 40% | 0.508 | |
| 0379_AMU | Zone I | 1/3 | 33% | 0.617 | |
| 0057_KCMC | Zone I | 7/22 | 32% | 0.581 | |
| 0304_YCH | Zone I | 5/16 | 31% | 0.645 | |
| 0050_KCMC | Zone I | 6/23 | 26% | 0.574 | |
| 0001_AMU | Stage 3 | 2/8 | 25% | 0.697 | |
| 0058_KCMC | Stage 3 | 3/15 | 20% | 0.788 | |
| 0399_KCMC | Zone I | 1/6 | 17% | 0.567 | |
| 0251_FU | Zone I | 2/18 | 11% | 0.657 | |
| 0034_KCMC | Stage 3 | 2/22 | 9% | 0.743 | |
| 0031_AMU | Stage 3 | 2/29 | 7% | 0.789 | |
| 0257_FU | Zone I | 1/17 | 6% | 0.778 | |
| 0013_AMU | Stage 3 | 1/18 | 6% | 0.739 | |
| 0048_KCMC | Stage 3, Plus | 1/27 | 4% | 0.790 | |

### Sens>=95% (閾値 0.443)

- RW-ROP陽性動画: **84**
- FN画像を含む動画: **17**
- 完全見逃し動画（全画像FN）: **1** (0217_FU)

| video_id | RW-ROP構成要素 | FN画像数/全画像数 | FN率 | mean P(RW) | 備考 |
|----------|--------------|----------------|------|------------|------|
| 0217_FU | Zone I, Stage 3 | 8/8 | 100% | 0.293 | **完全見逃し** |
| 0400_KCMC | Zone I | 19/21 | 90% | 0.384 | |
| 0215_FU | Stage 3 | 3/4 | 75% | 0.452 | |
| 0128_OWCH | Stage 3 | 9/14 | 64% | 0.423 | |
| 0390_KCMC | Zone I | 13/25 | 52% | 0.452 | |
| 0268_OWCH | Stage 3 | 4/9 | 44% | 0.490 | |
| 0379_AMU | Zone I | 1/3 | 33% | 0.617 | |
| 0304_YCH | Zone I | 5/16 | 31% | 0.645 | |
| 0001_AMU | Stage 3 | 2/8 | 25% | 0.697 | |
| 0204_AMU | Stage 3 | 5/24 | 21% | 0.562 | |
| 0057_KCMC | Zone I | 4/22 | 18% | 0.581 | |
| 0168_OWCH | Zone I | 2/13 | 15% | 0.499 | |
| 0127_OWCH | Stage 3 | 1/13 | 8% | 0.523 | |
| 0058_KCMC | Stage 3 | 1/15 | 7% | 0.788 | |
| 0013_AMU | Stage 3 | 1/18 | 6% | 0.739 | |
| 0048_KCMC | Stage 3, Plus | 1/27 | 4% | 0.790 | |
| 0031_AMU | Stage 3 | 1/29 | 3% | 0.789 | |

### Youden's J (閾値 0.569)

- RW-ROP陽性動画: **84**
- FN画像を含む動画: **33**
- 完全見逃し動画（全画像FN）: **4** (0012_AMU, 0128_OWCH, 0217_FU, 0400_KCMC)

| video_id | RW-ROP構成要素 | FN画像数/全画像数 | FN率 | mean P(RW) | 備考 |
|----------|--------------|----------------|------|------------|------|
| 0012_AMU | Stage 3 | 1/1 | 100% | 0.566 | **完全見逃し** |
| 0400_KCMC | Zone I | 21/21 | 100% | 0.384 | **完全見逃し** |
| 0217_FU | Zone I, Stage 3 | 8/8 | 100% | 0.293 | **完全見逃し** |
| 0128_OWCH | Stage 3 | 14/14 | 100% | 0.423 | **完全見逃し** |
| 0168_OWCH | Zone I | 11/13 | 85% | 0.499 | |
| 0214_FU | Zone I | 4/5 | 80% | 0.508 | |
| 0268_OWCH | Stage 3 | 7/9 | 78% | 0.490 | |
| 0127_OWCH | Stage 3 | 10/13 | 77% | 0.523 | |
| 0390_KCMC | Zone I | 19/25 | 76% | 0.452 | |
| 0215_FU | Stage 3 | 3/4 | 75% | 0.452 | |
| 0399_KCMC | Zone I | 4/6 | 67% | 0.567 | |
| 0050_KCMC | Zone I | 15/23 | 65% | 0.574 | |
| 0204_AMU | Stage 3 | 13/24 | 54% | 0.562 | |
| 0057_KCMC | Zone I | 10/22 | 45% | 0.581 | |
| 0404_KCMC | Zone I | 2/5 | 40% | 0.624 | |
| 0001_AMU | Stage 3 | 3/8 | 38% | 0.697 | |
| 0253_FU | Zone I | 1/3 | 33% | 0.776 | |
| 0379_AMU | Zone I | 1/3 | 33% | 0.617 | |
| 0304_YCH | Zone I | 5/16 | 31% | 0.645 | |
| 0080_YCH | Zone I, Stage 3 | 1/4 | 25% | 0.768 | |
| 0058_KCMC | Stage 3 | 3/15 | 20% | 0.788 | |
| 0034_KCMC | Stage 3 | 4/22 | 18% | 0.743 | |
| 0267_OWCH | Stage 3 | 1/6 | 17% | 0.867 | |
| 0251_FU | Zone I | 3/18 | 17% | 0.657 | |
| 0203_AMU | Stage 3 | 2/15 | 13% | 0.716 | |
| 0257_FU | Zone I | 2/17 | 12% | 0.778 | |
| 0393_KCMC | Stage 3 | 1/9 | 11% | 0.707 | |
| 0031_AMU | Stage 3 | 3/29 | 10% | 0.789 | |
| 0164_OWCH | Zone I | 1/13 | 8% | 0.650 | |
| 0048_KCMC | Stage 3, Plus | 2/27 | 7% | 0.790 | |
| 0144_OWCH | Zone I, Stage 3 | 2/27 | 7% | 0.753 | |
| 0013_AMU | Stage 3 | 1/18 | 6% | 0.739 | |
| 0010_AMU | Stage 3 | 1/26 | 4% | 0.749 | |

---

## Report 2: Top-10 Majority Vote レポート (report_top10_majority_vote.pdf)

**評価方法**: 患者単位 Soft Vote → hard OR prediction (default threshold)

### SV All（全画像使用）

- RW-ROP陽性動画: **84**
- FN動画数: **11**
- Sensitivity: **0.8690**

| video_id | True RW-ROP構成要素 | 予測 Zone | 予測 Stage | 予測 Plus | P(RW) |
|----------|-------------------|----------|----------|---------|-------|
| 0050_KCMC | Zone I | Zone II | Stage 1 | Normal | 0.578 |
| 0127_OWCH | Stage 3 | Zone II | Stage 2 | Normal | 0.521 |
| 0128_OWCH | Stage 3 | Zone II | Stage 1 | Normal | 0.423 |
| 0168_OWCH | Zone I | Zone II | Stage 1 | PrePlus | 0.496 |
| 0214_FU | Zone I | Zone II | Stage 2 | Normal | 0.508 |
| 0215_FU | Stage 3 | Zone II | Stage 2 | PrePlus | 0.458 |
| 0217_FU | Zone I, Stage 3 | Zone II | Stage 2 | Normal | 0.293 |
| 0268_OWCH | Stage 3 | Zone II | Stage 2 | Normal | 0.489 |
| 0304_YCH | Zone I | Zone II | Stage 1 | Normal | 0.671 |
| 0390_KCMC | Zone I | Zone II | Stage 1 | PrePlus | 0.454 |
| 0400_KCMC | Zone I | Zone III | Stage 0 | Normal | 0.384 |

### SV Top-10（Top-10品質フィルタリング）

- RW-ROP陽性動画: **84**
- FN動画数: **12**
- Sensitivity: **0.8571**

| video_id | True RW-ROP構成要素 | 予測 Zone | 予測 Stage | 予測 Plus | P(RW) |
|----------|-------------------|----------|----------|---------|-------|
| 0050_KCMC | Zone I | Zone II | Stage 1 | Normal | 0.522 |
| 0127_OWCH | Stage 3 | Zone II | Stage 2 | Normal | 0.515 |
| 0128_OWCH | Stage 3 | Zone II | Stage 0 | Normal | 0.438 |
| 0144_OWCH | Zone I, Stage 3 | Zone II | Stage 2 | PrePlus | 0.746 |
| 0168_OWCH | Zone I | Zone II | Stage 1 | PrePlus | 0.494 |
| 0204_AMU | Stage 3 | Zone II | Stage 2 | PrePlus | 0.513 |
| 0214_FU | Zone I | Zone II | Stage 2 | Normal | 0.508 |
| 0215_FU | Stage 3 | Zone II | Stage 2 | PrePlus | 0.458 |
| 0217_FU | Zone I, Stage 3 | Zone II | Stage 2 | Normal | 0.293 |
| 0268_OWCH | Stage 3 | Zone II | Stage 2 | Normal | 0.489 |
| 0390_KCMC | Zone I | Zone II | Stage 1 | Normal | 0.403 |
| 0400_KCMC | Zone I | Zone III | Stage 0 | Normal | 0.371 |

### SV Top-5（Top-5品質フィルタリング）

- RW-ROP陽性動画: **84**
- FN動画数: **11**
- Sensitivity: **0.8690**

| video_id | True RW-ROP構成要素 | 予測 Zone | 予測 Stage | 予測 Plus | P(RW) |
|----------|-------------------|----------|----------|---------|-------|
| 0050_KCMC | Zone I | Zone II | Stage 1 | Normal | 0.519 |
| 0127_OWCH | Stage 3 | Zone II | Stage 2 | Normal | 0.511 |
| 0128_OWCH | Stage 3 | Zone II | Stage 2 | Normal | 0.437 |
| 0168_OWCH | Zone I | Zone II | Stage 1 | PrePlus | 0.524 |
| 0204_AMU | Stage 3 | Zone II | Stage 2 | PrePlus | 0.480 |
| 0214_FU | Zone I | Zone II | Stage 2 | Normal | 0.508 |
| 0215_FU | Stage 3 | Zone II | Stage 2 | PrePlus | 0.458 |
| 0217_FU | Zone I, Stage 3 | Zone II | Stage 2 | Normal | 0.280 |
| 0268_OWCH | Stage 3 | Zone II | Stage 2 | PrePlus | 0.485 |
| 0390_KCMC | Zone I | Zone II | Stage 1 | Normal | 0.370 |
| 0400_KCMC | Zone I | Zone II | Stage 0 | Normal | 0.369 |

---

## 両レポート共通のFN動画ID

| 比較対象 | FN動画数 |
|---------|---------|
| Report 1 完全見逃し (default 0.5) | 1 動画 |
| Report 2 SV All FN | 11 動画 |
| **共通FN** | **1 動画: 0217_FU** |

**0217_FU** は両レポートで一貫してFalse Negativeとなっており、Zone I + Stage 3 にもかかわらず P(RW)=0.293 と非常に低い確率が出力されている最も困難な症例である。

---

## FNパターンの傾向

- **Zone I の誤判定**が最多（11動画中7動画）: Zone I → Zone II と誤判定されるケースが支配的
- **Stage 3 の誤判定**: Stage 3 → Stage 2 への1段階下方誤判定が多い
- P(RW) は 0.28〜0.67 の範囲で、判断境界（0.5）付近に集中
