# Best画像選別方法の比較

このドキュメントでは、`validate_images.ipynb`（詳細版）と`validate_images_simple.ipynb`（シンプル版）の2つの選別方法について説明します。

---

## 1. validate_images.ipynb（詳細版）

### 概要
複数の品質指標を組み合わせた重み付きランキング方式でBest画像を選定します。

### 選別基準

#### 1.1 対象画像のフィルタリング
- `lens_detected == True` かつ `retina_ratio > 0` の画像を対象

#### 1.2 Disc位置の判定
- **disc_pos_ok**: Discの中心が画像中心から半径の**25-75%**の範囲内にあるかどうか
  - `disc_center_dist_ratio` が 0.25 ≤ x ≤ 0.75 の場合に `True`

#### 1.3 品質指標

| 指標 | 説明 | 重み |
|------|------|------|
| `retina_area` | 網膜検出面積（大きいほど良い） | **1.5** |
| `mbss_score` | ピント品質スコア（MBSS系指標のz-score正規化後、重み付き和） | **1.0** |
| `disc_ring_score` | Disc周辺のピント（z-score正規化後） | **0.5** |
| `S_mean` | 彩度（HSVのSチャンネル平均、コントラスト指標） | **0.5** |

#### 1.4 MBSS（Modified Blur Score）の構成要素
- `L_multi`: マルチスケールLaplacian分散（重み付き和）
- `HF_ratio`: FFT高周波エネルギー比
- `Spec_centroid`: スペクトル重心
- `Grad_p90`: 勾配の90パーセンタイル

これらの指標をz-score正規化後、以下の重みで合成：
- `L_multi`: 0.35
- `HF_ratio`: 0.25
- `Spec_centroid`: 0.20
- `Grad_p90`: 0.20

#### 1.5 ランキング計算

各指標をランク化（小さいほど上位）し、重み付き和で総合ランクを計算：

```
rank_sum = 1.5 × retina_area_rank 
         + 1.0 × mbss_rank 
         + 0.5 × disc_ring_rank 
         + 0.5 × s_mean_rank
```

**注意**: `rank_sum`が小さいほど上位

#### 1.6 選別ステージ

**Stage 1: disc_pos_ok=True の画像**
- `disc_pos_ok == True` の画像を `rank_sum` でソート（昇順）
- 上位30件を選定

**Stage 2: disc_pos_ok=False の画像（補完）**
- Stage 1で30件に満たない場合、`disc_pos_ok == False` の画像から補完
- 同様に `rank_sum` でソートして不足分を選定

#### 1.7 出力
- **Top30**のBest画像を選定
- Excelファイル: `best_images_{case_id}.xlsx`
- CSVファイル: `validation_results_{case_id}.csv`

---

## 2. validate_images_simple.ipynb（シンプル版）

### 概要
シンプルな選別基準で、計算コストを抑えながらBest画像を選定します。

### 選別基準

#### 2.1 対象画像のフィルタリング
- `lens_detected == True` かつ `retina_ratio > 0` かつ **`disc_detected == True`** の画像を対象
  - **注意**: 詳細版と異なり、disc検出が必須条件

#### 2.2 Disc位置の判定
- **disc_pos_ok_simple**: Discの中心が画像中心から半径の**20-95%**の範囲内にあるかどうか
  - `disc_center_dist_ratio` が 0.20 ≤ x ≤ 0.95 の場合に `True`
  - **注意**: 詳細版（25-75%）より範囲が広い

#### 2.3 ランキング指標
- **`retina_ratio`のみ**を使用（大きい順）
- MBSSやDisc周辺のピント評価は行わない

#### 2.4 選別ステージ

**Stage 1: disc_pos_ok_simple=True の画像**
- `disc_pos_ok_simple == True` の画像を `retina_ratio` でソート（降順）
- 上位30件を選定

**Stage 2: disc_pos_ok_simple=False の画像（補完）**
- Stage 1で30件に満たない場合、`disc_pos_ok_simple == False` の画像から補完
- 同様に `retina_ratio` でソートして不足分を選定

#### 2.5 出力
- **Top30**のBest画像を選定
- Excelファイル: `best_images_simple_{case_id}.xlsx`
- CSVファイル: `validation_results_simple_{case_id}.csv`

---

## 3. 比較表

| 項目 | 詳細版（validate_images.ipynb） | シンプル版（validate_images_simple.ipynb） |
|------|--------------------------------|-------------------------------------------|
| **対象画像** | `lens_detected=True` かつ `retina_ratio>0` | `lens_detected=True` かつ `retina_ratio>0` かつ **`disc_detected=True`** |
| **Disc位置範囲** | 25-75% | 20-95% |
| **ランキング指標** | 複数指標の重み付き和（`rank_sum`） | `retina_ratio`のみ |
| **品質指標** | MBSS、Disc周辺ピント、彩度 | なし |
| **計算コスト** | 高（品質指標計算あり） | 低（品質指標計算なし） |
| **選別精度** | 高（多角的評価） | 中（シンプルな基準） |
| **推論時間** | 長い | 短い |
| **CSVファイル名** | `validation_results_{case_id}.csv` | `validation_results_simple_{case_id}.csv` |
| **Excelファイル名** | `best_images_{case_id}.xlsx` | `best_images_simple_{case_id}.xlsx` |

---

## 4. 使い分けの指針

### 詳細版（validate_images.ipynb）を使用する場合
- **高精度な選別が必要な場合**
- 画像品質（ピント、コントラスト）を重視する場合
- 研究用途で詳細な分析が必要な場合
- 計算時間に余裕がある場合

### シンプル版（validate_images_simple.ipynb）を使用する場合
- **高速処理が必要な場合**
- 大量のケースを一括処理する場合
- シンプルな基準で十分な場合
- 計算リソースが限られている場合
- Disc検出が必須条件で問題ない場合

---

## 5. 共通の処理フロー

両方のノートブックとも以下の処理フローを共有しています：

1. **推論実行**
   - RT-DETRでLens検出
   - YOLO-segでRetina/Disc/Maculaセグメンテーション
   - 各画像の特徴量を計算

2. **CSV保存**
   - 推論結果をCSVファイルに保存（再推論の回避）

3. **Best画像選定**
   - Stage 1: Disc位置が良好な画像を優先選定
   - Stage 2: 不足分を補完

4. **Excel出力**
   - 選定結果をExcelファイルに出力

---

## 6. パラメータ設定

### 詳細版の主要パラメータ
```python
TOP_K_TOTAL = 30              # 最終出力数
WEIGHT_RETINA = 1.5           # retina_area の重み
WEIGHT_MBSS = 1.0             # mbss_score の重み
WEIGHT_DISC_RING = 0.5        # disc_ring_score の重み
WEIGHT_S_MEAN = 0.5           # S_mean の重み
disc_pos_ok 範囲: 0.25 - 0.75  # Disc位置の判定範囲
```

### シンプル版の主要パラメータ
```python
TOP_K_TOTAL = 30              # 最終出力数
DISC_POS_MIN = 0.20           # Disc位置の最小値（20%）
DISC_POS_MAX = 0.95           # Disc位置の最大値（95%）
```

---

## 7. 注意事項

1. **CSVファイルの互換性**
   - 詳細版とシンプル版ではCSVファイル名が異なります
   - 詳細版: `validation_results_{case_id}.csv`
   - シンプル版: `validation_results_simple_{case_id}.csv`

2. **Disc検出の必須性**
   - シンプル版では `disc_detected=True` が必須条件のため、Discが検出されない画像は選定対象外になります
   - 詳細版ではDisc検出は必須ではありませんが、`disc_pos_ok` の判定にはDisc検出が必要です

3. **ランキングの方向**
   - 詳細版: `rank_sum` が**小さいほど上位**（ランクの和）
   - シンプル版: `retina_ratio` が**大きいほど上位**

4. **バッチ処理**
   - 両方のノートブックとも複数ケースの一括処理に対応しています
   - 既存のCSVファイルがある場合は再推論をスキップできます

---

## 8. 参考

- 詳細版の選別アルゴリズムは `validate_images.ipynb` のセル0（markdown）に記載されています
- シンプル版の選別基準は `validate_images_simple.ipynb` のセル0（markdown）に記載されています

