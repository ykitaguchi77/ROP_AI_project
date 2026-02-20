# Best Image Selection Protocol（ベスト画像選定手順）

対象: `ROP_project/bestimage_validation/validate_images.ipynb` の選定ロジック（Cell 10 / Cell 12）

## 目的

- **Disc位置が適切**な画像を優先しつつ（`disc_pos_ok`）
- **網膜面積が大きい**画像を重視し（`retina_area`）
- **画質（ピント/解像感）**も考慮してランキング（MBSS + Disc周辺シャープネス + 彩度）

---

## 入力と出力

- **入力**
  - 画像フォルダ: `...\<case_id>\images\`
  - モデル:
    - Lens検出: RT-DETR
    - Retina/Disc/Macula セグメンテーション: YOLO-seg（`retina_masks=True`）

- **出力**
  - per-case CSV: `bestimage_validation/validation_results/validation_results_<case_id>.csv`
  - per-case Best xlsx: `bestimage_validation/best_images_<case_id>.xlsx`（Cell 10）
  - 集計 Best xlsx: `D:\ダウンロード\bestimage_list.xlsx`（Cell 12）

---

## 特徴量（スコア用の計測）概要

### 1) Retina / Lens 基本量

- **`lens_area`**: lens円形マスク内の画素数
- **`retina_area`**: セグのRetinaマスク画素数（lens内に限定）
- **`retina_ratio`**: retina_area / lens_area × 100

### 2) Disc位置

- **`disc_center_dist_ratio`**: Disc中心からLens中心までの距離 / Lens半径
- **`disc_pos_ok`**: `0.25 <= disc_center_dist_ratio <= 0.75` のとき True
  - Discが中心から25%-75%の同心円内にある = 「使いやすい画像」

### 3) 画質（MBSS）

`compute_mbss_components(cropped, mask01=retina_mask_crop)` により、**網膜マスク内**のみを対象に計算します。

- **`mbss_L_multi`**: マルチスケールLaplacian分散（シャープネス寄り）
- **`mbss_HF_ratio`**: FFT高周波エネルギー比（ディテール寄り）
- **`mbss_Spec_centroid`**: FFTスペクトル重心（周波数分布）
- **`mbss_Grad_p90`**: 勾配強度の90パーセンタイル（エッジ寄り）
- **`mbss_score`**: 上記4つを case_id 内で z-score 正規化し重み付き和

### 4) Disc周辺シャープネス

Discマスクから core/ring を作り、**マスク内**のLaplacian分散を計算します。

- **`disc_core_L_multi`**, **`disc_ring_L_multi`**
- **`disc_core_score`**, **`disc_ring_score`**: case_id 内の z-score

### 5) 色調（脱彩度・白っぽさ対策）

**`S_mean`**（HSVのS=彩度の平均）を **網膜マスク内のみ**で計算します。

- `S_mean` が低いほど、**白っぽい/薄い（脱彩度）**傾向になりやすい

---

## 選定アルゴリズム

### 基本方針

1. **足切りなし** — 品質が低くても候補から除外しない
2. **disc_pos_ok=True を優先** — Disc位置が良い画像を先に選ぶ
3. **retina_area ベース** — 網膜面積が大きい画像を上位に
4. **品質指標でボーナス** — mbss_score, disc_ring_score, S_mean で微調整
5. **補完** — disc_pos_ok=True が足りなければ False から追加

### Step 0: 有効データのみ

以下を満たす行のみを対象とします。

- `lens_detected == True`
- `retina_ratio > 0`

### Step 1: ランキング計算

各指標のランク（小さいほど上位）を計算し、重み付き合計を算出します。

```
rank_sum = 1.5 × retina_area_rank
         + 1.0 × mbss_rank
         + 0.5 × disc_ring_rank
         + 0.5 × s_mean_rank
```

| 指標 | 説明 | 重み |
|------|------|------|
| retina_area_rank | 網膜面積（大きいほど上位） | 1.5 |
| mbss_rank | ピント品質スコア（高いほど上位） | 1.0 |
| disc_ring_rank | Disc周辺のピント（高いほど上位） | 0.5 |
| s_mean_rank | 彩度/コントラスト（高いほど上位） | 0.5 |

### Step 2: Stage 1 — disc_pos_ok=True の画像

1. `disc_pos_ok == True` の画像を抽出
2. `rank_sum` でソート（小さい順 = 上位）
3. 最大30件選出

### Step 3: Stage 2 — disc_pos_ok=False の画像（補完）

1. Stage 1 で30件に満たない場合のみ実行
2. 残りの画像（disc_pos_ok=False または未検出）を対象
3. 同様に `rank_sum` でソート
4. 不足分を補完

---

## ソート順

```python
sort_values(
    by=['rank_sum', 'retina_area'],
    ascending=[True, False]
)
```

1. **rank_sum**（昇順）: 総合スコアが低いほど上位
2. **retina_area**（降順）: 同点なら網膜面積が大きい方を優先

---

## 出力カラム

| カラム | 説明 |
|--------|------|
| rank | 最終順位（1〜30） |
| image_id | 患者番号 |
| image_name | ファイル名 |
| selection_stage | `disc_pos_ok=True` or `disc_pos_ok=False` |
| disc_pos_ok | Disc位置が適切か |
| retina_area | 網膜面積（px） |
| retina_ratio | 網膜占有率（%） |
| mbss_score | ピント品質スコア |
| disc_ring_score | Disc周辺シャープネス |
| S_mean | 彩度平均 |
| rank_sum | 総合ランクスコア |

---

## 運用メモ（なぜこの設計か）

- **足切りなし**: 厳格な足切りで良い画像を逃すリスクを回避
- **disc_pos_ok 優先**: Disc位置が良い画像は「使いやすい」ため優先
- **retina_area ベース**: 網膜面積が大きい = ピントが良いことが多い
- **品質指標はボーナス**: 主軸ではなく微調整として使用
- **Stage 2 補完**: disc_pos_ok=True が少ないケースでも30件確保

---

## 関連ファイル

- 選定の実装（推論〜選定）: `ROP_project/bestimage_validation/validate_images.ipynb`
  - Cell 5: 推論実行（単一ケース）
  - Cell 7-8: バッチ推論
  - Cell 10: Best画像選定（単一ケース）
  - Cell 12: Best画像選定（バッチ）

---

# 要約

1. 対象データ
   - lensと網膜が検出されているもののみを対象とする
   - **足切りなし**（品質が低くても候補に残す）

2. 順位付け
   - **disc_pos_ok=True**（Discが中心から25%-75%の同心円に収まっている）を最優先
   - **retina_area**（網膜面積）が大きいほど上位
   - **品質指標**（mbss_score, disc_ring_score, S_mean）でボーナス加算
   - disc_pos_ok=True が不足する場合は False の画像で補完

3. ランキング計算式
   ```
   rank_sum = 1.5 × retina_area_rank + 1.0 × mbss_rank + 0.5 × disc_ring_rank + 0.5 × s_mean_rank
   ```
   （小さいほど上位）

---

# 開発履歴（Development History）

## 0. 評価指標の定義

### 画像一致率（Image Match Rate）

**定義**: AIが選んだTop5画像のうち、Humanが選んだ画像と一致する枚数の割合

```
画像一致率 = Σ(各動画でAI Top5とHuman選定が一致する枚数) / (動画数 × 5)
```

**計算例**:
- 22動画 × 5画像 = 110画像が対象
- そのうち60枚がHuman選定と一致
- 画像一致率 = 60/110 = 54.5%

### 動画一致率（Video Match Rate）

**定義**: 22動画のうち、**少なくとも1枚以上**AIとHumanの選定が一致する動画の割合

```
動画一致率 = (1枚以上一致した動画数) / 総動画数
```

**計算例**:
- 22動画中、19動画で少なくとも1枚一致
- 動画一致率 = 19/22 = 86.4%

### 両指標の違い

| 指標 | 何を測定するか | 特徴 |
|------|---------------|------|
| **画像一致率** | 選定の「精度」 | 厳しい評価。5枚全て一致して初めて100% |
| **動画一致率** | 選定の「網羅性」 | 緩い評価。1枚でも当たれば成功とカウント |

**例: Video 1703**
- AI Top5とHuman Top5が5枚全て一致
- 画像一致率への寄与: 5/5 = 100%
- 動画一致率への寄与: 1動画 (一致あり)

**例: Video 1632**
- AI Top5とHuman Top5が0枚一致
- 画像一致率への寄与: 0/5 = 0%
- 動画一致率への寄与: 0動画 (一致なし)

---

## 1. 初期アプローチ：ランクベースのスコアリング

### 1.1 最初の指標設計

当初の選定基準は以下の指標を組み合わせたランクベースのスコアリングでした：

| 指標 | 説明 | 重み |
|------|------|------|
| retina_area | 網膜検出面積（大きいほど良い） | 1.5 |
| mbss_score | ピント品質スコア（MBSS系指標のz-score正規化後、重み付き和） | 1.0 |
| disc_ring_score | Disc周辺のピント（z-score正規化後） | 0.5 |
| S_mean | 彩度（HSVのSチャンネル平均） | 0.5 |

### 1.2 MBSS（Modified Blur Score）の構成要素

- **L_multi**: マルチスケールLaplacian分散（重み 0.35）
- **HF_ratio**: FFT高周波エネルギー比（重み 0.25）
- **Spec_centroid**: スペクトル重心（重み 0.20）
- **Grad_p90**: 勾配の90パーセンタイル（重み 0.20）

### 1.3 初期結果

Human選定（YF, HK）との比較で、22動画×5画像=110画像に対して：
- **画像一致率**: 約45-50%
- **動画一致率**（1枚以上一致）: 約80%

---

## 2. Random Forest による最適化試行

### 2.1 目的

Human選定を教師データとして、指標の重みを機械学習で最適化することを試みました。

### 2.2 手法

```python
# 特徴量
features = ['retina_ratio', 'mbss_score', 'mbss_Grad_p90', 'disc_ring_score', 'S_mean']

# ラベル
# Human選定画像 = 1, それ以外 = 0

# モデル
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
```

### 2.3 結果

- **Feature Importance** から `retina_ratio` と `mbss_Grad_p90` が重要と判明
- しかし、大幅な精度向上には至らず（約50%程度で頭打ち）
- **原因分析**: 既存の指標だけではHumanの選定基準を十分に捉えられていない

---

## 3. 問題動画の詳細分析

### 3.1 一致率0%の動画の特定

22動画中、以下の動画でAI-Human一致率が0%でした：

| Video ID | AI Top5 | Human Top5 | 一致数 |
|----------|---------|------------|--------|
| 1632 | Frame 1775-1800 | Frame 795-905 | 0/5 |
| 1732 | - | Frame 410, 415, 505... | 0/5 |

### 3.2 Video 1632 の詳細分析

#### 画像を目視確認した結果：
- **AI選定画像**: 動画後半のフレーム（1775-1800）、retina_ratio高い（86%）
- **Human選定画像**: 動画前半のフレーム（795-905）、retina_ratio低め（79%）

#### 決定的な違い：
- **AI選定画像**: Discが画像の端にあり、辺縁がRetinaマスクからはみ出している
- **Human選定画像**: Discが画像中央寄りにあり、辺縁が完全にRetinaで覆われている

```
AI選定画像の特徴:
  retina_ratio:    86.1%  (高い)
  disc_edge_cov:   0.952  (やや低い)
  disc_center_dist: 0.675 (端寄り)

Human選定画像の特徴:
  retina_ratio:    79.3%  (やや低い)
  disc_edge_cov:   0.981  (高い)
  disc_center_dist: 0.503 (中央寄り)
```

### 3.3 Video 1732 の詳細分析

**原因**: データサンプリングの問題
- Human選定フレーム（410, 415, 505など）がvalidationデータセットに存在しない
- フレームサンプリング間隔（5フレームごと）の関係で欠落
- **結論**: この動画はアルゴリズムの問題ではなく、データの問題

---

## 4. Disc Edge Coverage指標の開発

### 4.1 新指標の設計

Video 1632の分析から、**Discの辺縁がRetinaマスクに覆われているか**が重要と判明。

#### disc_edge_coverage_ratio の計算方法

```python
def compute_disc_edge_coverage(disc_mask, retina_mask):
    # Discマスクの輪郭（辺縁）を抽出
    kernel = np.ones((3, 3), np.uint8)
    disc_eroded = cv2.erode(disc_bin, kernel, iterations=1)
    disc_edge = disc_bin - disc_eroded

    # Retinaマスクを膨張
    retina_dilated = cv2.dilate(retina_bin, kernel, iterations=2)

    # 覆われている辺縁ピクセルをカウント
    covered_edge_pixels = (disc_edge & retina_dilated).sum()
    total_edge_pixels = disc_edge.sum()

    coverage_ratio = covered_edge_pixels / total_edge_pixels
    return coverage_ratio
```

### 4.2 指標の検証

22動画で `disc_edge_coverage_ratio` の分布を確認：
- Human選定画像の平均: 0.95-0.98
- AI選定画像（従来）の平均: 0.85-0.95

---

## 5. 最適化実験

### 5.1 スコアリング方式 vs 足切り方式

#### アプローチA: スコアに組み込む
```python
score = 0.4 * retina_ratio_norm + 0.4 * mbss_Grad_p90_norm + 0.2 * disc_edge_coverage_norm
```

#### アプローチB: 足切りとして使用
```python
# disc_edge_coverage_ratio >= threshold でフィルタ後、スコアリング
```

### 5.2 閾値の最適化

| 閾値 | 画像一致数 | 一致率 | 動画一致率 |
|------|-----------|--------|-----------|
| 0.75 | 58/110 | 52.7% | 19/22 (86.4%) |
| **0.80** | **60/110** | **54.5%** | **19/22 (86.4%)** |
| 0.85 | 58/110 | 52.7% | 18/22 (81.8%) |
| 0.90 | 55/110 | 50.0% | 18/22 (81.8%) |
| 0.95 | 48/110 | 43.6% | 17/22 (77.3%) |

**結論**: 閾値 **0.80** が最適

### 5.3 最終アルゴリズム

1. **足切り**: `disc_edge_coverage_ratio >= 0.80`
2. **スコアリング**: `score = 0.4 × retina_ratio_norm + 0.4 × mbss_Grad_p90_norm + 0.2 × mbss_score_norm`
3. **補完**: 足切りで30件に満たない場合は `retina_ratio` のみでソート

---

## 6. 最終結果

### 6.1 精度改善

| 指標 | 初期（ランクベース） | 最終（Disc指標版） |
|------|---------------------|-------------------|
| 画像一致率 | ~48/110 (43.6%) | **60/110 (54.5%)** |
| 動画一致率 | ~18/22 (81.8%) | **19/22 (86.4%)** |

**改善幅**:
- 画像一致率: +10.9ポイント
- 動画一致率: +4.5ポイント

### 6.2 残存する問題

#### Video 1632（0/5一致）
- **原因**: シーン選好の違い
- AIは後半フレーム（retina_ratio 86%）を選好
- Humanは前半フレーム（retina_ratio 79%、disc位置良好）を選好
- 両者とも `disc_edge_coverage_ratio >= 0.95` を満たす
- **結論**: Disc指標だけでは解決不可能。**時間的なシーン選好**の違いが原因

#### Video 1732（0/5一致）
- **原因**: データサンプリングの問題
- Human選定フレームがデータセットに存在しない
- **結論**: アルゴリズムではなくデータ収集プロセスの問題

---

## 7. 今後の課題

### 7.1 シーン選好の解決策（未実装）

1. **時間的多様性の導入**:
   - Top30を異なる時間帯から均等に選出
   - クラスタリングによるシーン分割

2. **Disc位置のより厳格な評価**:
   - `disc_center_dist_ratio` の重み付け強化
   - 中央寄りの画像を優先

3. **人間の選定パターンの学習**:
   - より多くのHuman選定データの収集
   - Deep Learning による選好学習

### 7.2 データ品質の改善

1. フレームサンプリング間隔の見直し（5フレーム→1フレーム）
2. Human選定との対応付け精度の向上

---

## 8. 関連ファイル

| ファイル | 説明 |
|----------|------|
| `validate_images.ipynb` | 初期版の選定実装（ランクベース） |
| `validate_images_disc.ipynb` | Disc指標版の選定実装 |
| `compare_ai_human_top.ipynb` | AI vs Human 比較・可視化 |
| `bestimage_selection_protocol_development.ipynb` | 最適化過程のコード |

---

## 9. 変更履歴

| 日付 | 変更内容 |
|------|----------|
| 2025-01 | 初期版作成（ランクベーススコアリング） |
| 2025-01 | Random Forest による最適化試行 |
| 2025-01 | 問題動画（1632, 1732）の詳細分析 |
| 2025-01 | disc_edge_coverage_ratio 指標の開発 |
| 2025-01 | 足切り方式への変更、閾値0.80で最適化 |
| 2025-01 | 最終精度: 60/110 (54.5%), 19/22 (86.4%)
