# 画像品質スコア算出・Best10選定方法（現行版）

本ドキュメントでは、ROP（未熟児網膜症）プロジェクトにおける眼底画像の **品質評価スコア**（全体の鮮明度 + Disc周囲の鮮明度）と、そこから **Best10（Top10）を「必ず」抽出するロジック**をまとめます。

現行ロジックの実装は主に以下です。

- `ROP_project/bestimage_validation/validate_images.ipynb`: 1ケース（1つの画像フォルダ）を評価し、CSV保存＋Best10/Best5を抽出
- `ROP_project/bestimage_validation/generate_best_images_list.py`: 複数ケースの `validation_results_*.csv` から全ケース分のBest10一覧を `bestimage_list.xlsx` に集計

---

## 1. 入出力（どこに保存されるか）

### 入力
- `image_dir`: `...\{image_id}\画像` フォルダ（jpg/png等）
- モデル:
  - **検出モデル**: RT-DETR（Lens bbox）
  - **セグメンテーションモデル**: YOLO-seg（Retina/Disc/Macula）

### 出力（1ケース）
`validate_images.ipynb` の出力は以下の2つです。

- **推論＆スコア結果CSV**: `ROP_project/bestimage_validation/validation_results/validation_results_{case_id}.csv`
- **Best画像Excel（任意）**: `ROP_project/bestimage_validation/best_images_{case_id}.xlsx`

### 出力（全ケース集計）
`generate_best_images_list.py` で全ケース分のBest10をまとめて出力します。

- **Best10一覧Excel**: `ROP_project/bestimage_validation/documents/bestimage_list.xlsx`

---

## 2. 推論・前処理フロー（1枚の画像）

`validate_images.ipynb` では、各画像に対して概ね以下の処理を行います。

1. **Lens検出（RT-DETR）**
   - 入力: 元画像
   - 出力: Lens bbox（クラス0想定）
2. **Lensクロップ＋円形マスク**
   - bboxでクロップした画像に対し、レンズ外周をグレー（114）で埋める
3. **Retina/Disc/Macula セグメンテーション（YOLO-seg）**
   - 入力: クロップ画像（アスペクト比を保ったまま固定幅にリサイズ）
   - 出力: Retina/Disc/Macula のマスク
4. **面積比（retina_ratio）の計算**
   - \(retina\_ratio = \frac{retina\_area}{lens\_area} \times 100\)
5. **品質スコアの算出**
   - Retina領域内: MBSS系（4特徴量→z-score→重み付き和）
   - Disc周囲: core/ring の L_multi → z-score

---

## 3. スコア算出ロジック

### A. MBSS Score（Retina領域内の総合鮮明度）

MBSSは、**Retinaとして検出された領域内のみ**を対象に計算します。

#### 使用する4つの特徴量
1. **L_multi（Multi-scale Laplacian Variance）**
2. **HF_ratio（High Frequency Energy Ratio）**
3. **Spec_centroid（Spectral Centroid）**
4. **Grad_p90（Gradient Magnitude 90th Percentile）**

#### 総合スコア
同一ケース内（同一 `case_id`）の画像群で平均・標準偏差を算出し、z-score正規化して重み付き和を取ります。

\[
\text{MBSS} = \sum_{i} w_i \cdot \frac{x_i - \mu_i}{\sigma_i}
\]

**重み（現行）**
- `L_multi`: 0.35
- `HF_ratio`: 0.25
- `Spec_centroid`: 0.20
- `Grad_p90`: 0.20

### B. Disc Core / Disc Ring Score（Disc周囲の鮮明度）

Discマスクから中心 \((cx,cy)\) と代表半径 \(R\) を推定し、以下のROIを定義します。

- **Disc Core**: 半径 \(0.6R\) 以内
- **Disc Ring**: \(0.6R \sim 1.2R\)

各ROI内で **L_multi** を計算し、同一ケース内でz-score化して `disc_core_score` / `disc_ring_score` とします。

---

## 4. Best10（Top10）選定ロジック（「10件担保」）

ここが現行の肝です。目的は **「良いものを優先しつつ、候補が足りなければ条件を緩めてTop10を必ず出す」**ことです。

### 4.1 有効データの定義
まず以下を満たす行のみを対象にします。

- `lens_detected == True`
- `retina_ratio > 0`

### 4.2 候補の作り方（retina_ratio 閾値を段階的に緩和）
`retina_ratio` の閾値を **パーセンタイルで段階的に下げて**候補集合を作ります（例: 90%→80%→…→0%）。

- 各段階で `retina_ratio >= quantile(p)` を満たすものを候補に採用
- 候補数が `top_k(=10)` に達した段階で打ち切り
- 最後（p=0.0）まで行っても足りない場合は「取れた分だけ」で進む

### 4.3 disc_ring_score の足切り（品質の下限）
`disc_ring_score` が使える場合は、**同一ケース内の中央値を下限**として足切りします。

- 例: `disc_ring_score >= median(disc_ring_score)`

※ `validate_images.ipynb` では「`disc_ring_score` が十分に計算できている件数が少ない場合は、この足切りを無効化」します（ケースによってDiscが検出できないことがあるため）。

### 4.4 最終ランキング（rank_sum）
候補集合に対して、以下で順位を付けます。

- `retina_ratio` の **厳しい閾値（高いパーセンタイル）を満たす画像を優先**するため、各画像に「満たす最大パーセンタイル（retina_tier）」を付与し、**retina_tierが高い順**に並べる
- `mbss_score` を降順で順位付け（高いほど良い）
- `disc_core_score` を降順で順位付け（高いほど良い）
- `rank_sum = mbss_rank + disc_core_rank` を最小化する順に並べる
- 同点のタイブレークは `mbss_score` の降順

その上で **先頭10件がBest10** です。

---

## 5. 全ケースのBest10一覧を作る（集計スクリプト）

`generate_best_images_list.py` は、`bestimage_validation/validation_results/validation_results_*.csv` を読み込み、各ケースで上記と同等のロジックでBest10を抽出して、1つのExcelにまとめます。

### 重複防止（重要）
同じ `image_id` のCSVが複数見つかった場合に備えて、現行版では

- **image_id ごとに最新のCSV（更新時刻が新しい方）だけ採用**

するようにしています。

---

## 6. 参照コード（現行）

主に `ROP_project/bestimage_validation/validate_images.ipynb` 内の以下が対応します。

- `process_one_image(...)`: 画像1枚に対する推論（Lens→Seg）と特徴量算出のメイン
- `compute_mbss_components(...)`: MBSS特徴量（4つ）を算出
- `compute_mbss_score(...)`: z-score正規化＋重み付き和
- `compute_disc_sharpness_components(...)`: Disc core/ring の L_multi を算出

集計は `ROP_project/bestimage_validation/generate_best_images_list.py` を参照してください。
