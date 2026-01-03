# Best Image Selection Protocol（ベスト画像選定手順）

対象: `ROP_project/bestimage_validation/validate_images.ipynb` の選定ロジック（Cell 7 / Cell 8）

## 目的

- **網膜が十分に写っている**画像を優先しつつ（`retina_ratio`）
- **画質（ピント/解像感）**も考慮し（MBSS + Disc周辺シャープネス）
- 近年の課題である **「白っぽい（脱彩度・ベール）」**を落とすために、**網膜マスク内の彩度**を用いて足切りします（`S_mean`）

---

## 入力と出力

- **入力**
  - 画像フォルダ: `...\<case_id>\images\`
  - モデル:
    - Lens検出: RT-DETR
    - Retina/Disc/Macula セグメンテーション: YOLO-seg（`retina_masks=True`）

- **出力**
  - per-case CSV: `bestimage_validation/validation_results/validation_results_<case_id>.csv`
  - per-case Best xlsx: `bestimage_validation/best_images_<case_id>.xlsx`（Cell 7）
  - 集計 Best xlsx: `D:\ダウンロード\bestimage_list.xlsx`（Cell 8）

---

## 特徴量（スコア用の計測）概要

### 1) Retina / Lens 基本量

- **`lens_area`**: lens円形マスク内の画素数
- **`retina_area`**: セグのRetinaマスク画素数（lens内に限定）
- **`retina_ratio`**: \(\frac{retina\_area}{lens\_area}\times 100\)

### 2) 画質（MBSS）

`compute_mbss_components(cropped, mask01=retina_mask_crop)` により、**網膜マスク内**のみを対象に計算します。

- **`mbss_L_multi`**: マルチスケールLaplacian分散（シャープネス寄り）
- **`mbss_HF_ratio`**: FFT高周波エネルギー比（ディテール寄り）
- **`mbss_Spec_centroid`**: FFTスペクトル重心（周波数分布）
- **`mbss_Grad_p90`**: 勾配強度の90パーセンタイル（エッジ寄り）
- **`mbss_score`**: 上記4つを case_id 内で z-score 正規化し重み付き和

### 3) Disc周辺シャープネス

Discマスクから core/ring を作り、**マスク内**のLaplacian分散を計算します。

- **`disc_core_L_multi`**, **`disc_ring_L_multi`**
- **`disc_core_score`**, **`disc_ring_score`**: case_id 内の z-score

### 4) 色調（脱彩度・白っぽさ対策）

新規: **`S_mean`**（HSVのS=彩度の平均）を **網膜マスク内のみ**で計算します。

- `S_mean` が低いほど、**白っぽい/薄い（脱彩度）**傾向になりやすい
- 白飛び（飽和）でなくても「ベール（haze）」のような白っぽさを拾う狙い

---

## 候補抽出（足切り）ロジック

### Step 0: 有効データのみ

以下を満たす行のみを対象とします。

- `lens_detected == True`
- `retina_ratio > 0`

### Step 1: `retina_ratio` 閾値を段階的に緩和（候補確保）

候補数が `top_k`（通常10）以上になるまで、retina_ratioの下限を **パーセンタイルで段階的に緩和**します。

- `percentiles = [0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.0]`
- 各 p で `thr = quantile(retina_ratio, p)` を計算し、`retina_ratio >= thr` を満たす画像を候補に
- それでも足りない場合は最後に最大限緩和（p=0.0）

### Step 2: `disc_ring_score`（条件付き足切り）

`disc_ring_score` が **十分に計算できている時のみ**使います。

- `disc_ring_score` の非NaN件数が `need_k`（通常5）以上 → 使用
- 使用する場合は **中央値以上**のみ残す

### Step 3: 色調足切り（`S_mean >= median`）

最初の運用はシンプルに **`S_mean` の中央値以上**を残します（網膜マスク内計算）。

- `S_thr = median(S_mean)` を **valid（有効全体）**から算出
- `S_mean >= S_thr` のみ残す
- **安全策**: 足切り後に候補が `need_k` 未満になる場合は、足切りを **無効化して元に戻す**

---

## “段階緩和でも厳しい閾値を優先”する仕組み（`retina_tier`）

`retina_ratio` を緩和して候補を増やすと、緩和ステップで拾われた低retina_ratio画像が混ざります。
そこで各画像に **「満たせる最大パーセンタイル」**を付与して、厳しい閾値を満たす画像が上位になるようにします。

- `thr_by_p[p] = quantile(valid.retina_ratio, p)`
- 各画像に対し、満たす最大 p を `retina_tier` として付与
- **`retina_tier` が大きいほど上位**

---

## ランキング（順位付け）

### 基本のランク指標

- `mbss_rank`: `mbss_score` の降順ランク（高いほど良い）
- `disc_core_rank`: `disc_core_score` の降順ランク
- `rank_sum = mbss_rank + disc_core_rank`（小さいほど良い）

### ソート順（Cell 7: 単症例のBest Top10/Top5）

1. `retina_tier`（降順）
2. `rank_sum`（昇順）
3. `mbss_score`（降順、タイブレーク）

### ソート順（Cell 8: 全CSV集計のTop10）

1. `retina_tier`（降順）
2. `disc_pos_priority`（降順）: `disc_pos_ok` を 0/1 化した優先度
3. `rank_sum`（昇順）
4. `mbss_score`（降順、タイブレーク）

NOTE: Cell 7 と Cell 8 では `disc_pos_ok` の扱い（優先度）が異なります。運用上は「どちらを正とするか」を決め、将来的に統一するのが望ましいです。

---

## 運用メモ（なぜこの設計か）

- **`retina_ratio`/`retina_tier`**: “網膜がよく写っている”を最優先にするため
- **MBSS/Disc系**: “ピント・解像感・局所シャープネス”を順位に反映するため
- **`S_mean` ゲート**: “白っぽい/脱彩度”はMBSSだけでは拾えないため（白飛びしていなくても起きる）
- **安全策（候補不足なら無効化）**: 症例全体の色調が悪いケースでも候補枯渇を避けるため

---

## 関連ファイル

- 選定の実装（推論〜選定）: `ROP_project/bestimage_validation/validate_images.ipynb`
- 集計スクリプト: `ROP_project/bestimage_validation/generate_best_images_list.py`
  - NOTE: スクリプト側はノートブックと完全一致していない場合があります（運用時はノートブックを正とし、必要なら同期してください）。



# 要約

1. 足切り
・lensと網膜が検出されているもののみを対象とする
・網膜占有率上位10％から順番に評価。数が足りなければ20％、30％と範囲を広げる
・disc ring score (disc周囲の明瞭度)が中央値未満は足切り
・S-mean (コントラスト平均値)が中央値未満は足切り
・ただし数が少なくなりすぎるときは足切りを解除

2. 順位付け
・網膜占有率同等（ex.上位10%以内）の中で、下記アルゴリズムに沿ってスコアリング
・discの位置が中心から25%-75%の同心円に収まっているものを優先とする
・mbss_rank + disc_core_rank（明瞭度の指標2項目の合計のランキング）が低い物から優先的に選ぶ
・mbss_rank + disc_core_rankが同じ場合には、mbss_scoreが高い方を選択する