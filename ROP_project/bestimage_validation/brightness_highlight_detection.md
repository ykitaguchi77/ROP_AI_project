# 網膜画像の暗部・過反射検出方法

網膜画像の品質評価において、暗い部分や明るすぎる反射（ハイライト）が入っていないかを選別するための方法をまとめます。

---

## 0. 暗部・過反射の定義

### 0.1 暗部（Dark Region）の定義

#### 概念的な定義
- **暗部**: 網膜領域内で、輝度が低すぎて詳細な構造（血管、病変など）が識別困難な領域
- **原因**: 露出不足、影、照明の不均一性など

#### 数値的な定義
- **基本閾値**: 輝度値（0-1の正規化値）が **0.15以下** のピクセルを暗部と定義
  - 8bit画像（0-255）では、**約38以下**に相当
  - グレースケール変換後の値で評価

- **検出条件**:
  - 単一ピクセルではなく、**連結した領域**として検出
  - 最小面積: 網膜領域の **1%以上** の領域を有効な暗部として扱う
  - 網膜領域内のみを評価対象とする（背景は除外）

- **深刻度の分類**:
  - **軽度**: 暗部面積比 < 10% かつ 最小輝度 > 0.05
  - **中度**: 暗部面積比 10-25% または 最小輝度 0.02-0.05
  - **重度**: 暗部面積比 ≥ 25% または 最小輝度 < 0.02

#### 評価指標
- `dark_area_ratio`: 暗部領域の面積比（網膜領域に対する割合）
- `dark_count`: 暗部領域の個数（連結成分の数）
- `min_dark_intensity`: 暗部領域内の最小輝度値
- `dark_score`: 暗部の深刻度スコア（0-1、大きいほど悪い）

---

### 0.2 過反射（Highlight/Overexposure）の定義

#### 概念的な定義
- **過反射**: 網膜領域内で、輝度が高すぎて情報が失われている（白飛び）領域
- **原因**: レンズや角膜からの反射光、照明の直接照射、局所的な過露出など

#### 数値的な定義
- **基本閾値**: 輝度値（0-1の正規化値）が **0.9以上** のピクセルを過反射と定義
  - 8bit画像（0-255）では、**約230以上**に相当
  - グレースケール変換後の値で評価

- **検出条件**:
  - 単一ピクセルではなく、**連結した領域**として検出
  - 最小面積: 網膜領域の **0.1%以上** の領域を有効な過反射として扱う
  - 網膜領域内のみを評価対象とする（背景は除外）

- **深刻度の分類**:
  - **軽度**: 過反射面積比 < 5% かつ 過反射領域数 < 3個
  - **中度**: 過反射面積比 5-15% または 過反射領域数 3-10個
  - **重度**: 過反射面積比 ≥ 15% または 過反射領域数 ≥ 10個 または 最大輝度 > 0.98

#### 評価指標
- `highlight_area_ratio`: 過反射領域の面積比（網膜領域に対する割合）
- `highlight_count`: 過反射領域の個数（連結成分の数）
- `max_highlight_intensity`: 過反射領域内の最大輝度値
- `highlight_score`: 過反射の深刻度スコア（0-1、大きいほど悪い）

---

### 0.3 輝度値の正規化について

本ドキュメントでは、輝度値を **0-1の範囲に正規化** して評価します：

```python
# 8bit画像（0-255）を0-1に正規化
gray_normalized = gray_image.astype(np.float32) / 255.0

# 暗部の判定: gray_normalized < 0.15
# 過反射の判定: gray_normalized > 0.9
```

**8bit画像での対応値**:
- 暗部閾値（0.15）→ **約38**（255 × 0.15 = 38.25）
- 過反射閾値（0.9）→ **約230**（255 × 0.9 = 229.5）

---

### 0.4 定義のまとめ表

| 項目 | 暗部（Dark Region） | 過反射（Highlight） |
|------|---------------------|---------------------|
| **基本閾値** | 輝度 < 0.15（約38/255） | 輝度 > 0.9（約230/255） |
| **最小面積** | 網膜領域の1%以上 | 網膜領域の0.1%以上 |
| **軽度** | 面積比 < 10%、最小輝度 > 0.05 | 面積比 < 5%、領域数 < 3個 |
| **中度** | 面積比 10-25%、最小輝度 0.02-0.05 | 面積比 5-15%、領域数 3-10個 |
| **重度** | 面積比 ≥ 25%、最小輝度 < 0.02 | 面積比 ≥ 15%、領域数 ≥ 10個、最大輝度 > 0.98 |
| **主な評価指標** | `dark_area_ratio`, `dark_count`, `min_dark_intensity` | `highlight_area_ratio`, `highlight_count`, `max_highlight_intensity` |

---

### 0.5 数値の根拠と調整の必要性

#### 0.4.1 基本閾値の根拠

**暗部閾値（0.15）の根拠**:
- **一般的な画像処理の経験則**: 8bit画像（0-255）において、輝度値が約38以下（0-1正規化で0.15）の領域は、通常の画像処理では「暗部」として扱われることが多い
- **網膜画像の特性**: 網膜画像は一般的に中程度の輝度（0.2-0.6程度）が多く、0.15以下は明らかに暗すぎる領域と判断される
- **視覚的な評価**: 実際の画像を確認すると、輝度0.15以下の領域では血管や病変の識別が困難になることが多い
- **注意**: この値は**初期提案値**であり、実際のデータに合わせて調整が必要

**過反射閾値（0.9）の根拠**:
- **白飛びの定義**: デジタル画像において、輝度値が最大値（255）に近い領域は情報が失われている（白飛び）とされる
- **一般的な閾値**: 画像処理では、最大値の90%以上（0.9以上）を「過露出」や「ハイライト」として扱うことが多い
- **網膜画像の特性**: 網膜画像では、正常な網膜組織は0.9以上の輝度を持たないため、0.9以上は反射や過露出と判断される
- **注意**: この値は**初期提案値**であり、実際のデータに合わせて調整が必要

#### 0.4.2 面積比の閾値の根拠

**暗部の最小面積（1%）の根拠**:
- **ノイズ除去**: 単一ピクセルや極小領域のノイズを除外するため
- **実用的な影響**: 網膜領域の1%未満の暗部は、診断に影響を与えないと判断
- **一般的な画像処理**: 連結成分解析では、全体の1%未満の領域はノイズとして扱うことが多い

**過反射の最小面積（0.1%）の根拠**:
- **反射の特性**: 過反射は通常、小さな点状や線状の領域として現れる
- **診断への影響**: 網膜領域の0.1%以上の反射は、診断に影響を与える可能性がある
- **暗部より厳しい基準**: 反射は暗部よりも小さな領域でも問題となるため、より低い閾値（0.1%）を設定

**深刻度分類の面積比（10%, 25%, 5%, 15%）の根拠**:
- **経験的な分類**: 一般的な画像品質評価では、問題領域の面積比に基づいて軽度・中度・重度を分類することが多い
- **診断への影響度**: 
  - 軽度（< 10% or < 5%）: 診断に大きな影響はないが、注意が必要
  - 中度（10-25% or 5-15%）: 診断に一定の影響がある可能性
  - 重度（≥ 25% or ≥ 15%）: 診断に重大な影響がある可能性が高い
- **注意**: これらの値は**経験的な分類**であり、臨床的な基準に合わせて調整が必要

#### 0.4.3 輝度の最小値・最大値の根拠

**最小輝度の閾値（0.05, 0.02）の根拠**:
- **極端に暗い領域**: 輝度0.05以下は、ほぼ情報が失われている領域と判断
- **重度の暗部**: 輝度0.02以下は、完全に黒に近い領域で、診断に使用できない
- **一般的な画像処理**: デジタル画像では、輝度0.05以下は「ほぼ黒」として扱われることが多い

**最大輝度の閾値（0.98）の根拠**:
- **白飛びの定義**: 輝度0.98以上は、ほぼ完全に白飛びしている領域と判断
- **情報の損失**: この領域では、網膜の構造情報が完全に失われている
- **一般的な画像処理**: デジタル画像では、最大値の98%以上は「ほぼ白」として扱われることが多い

#### 0.4.4 数値の調整方法

**実際のデータに合わせた調整手順**:

1. **サンプル画像の確認**
   - 実際の網膜画像を複数確認し、暗部・過反射の分布を観察
   - ヒストグラムを確認し、実際の輝度分布を把握

2. **閾値の調整**
   - 暗部閾値: 実際のデータで、暗すぎる領域の輝度分布を確認し、適切な閾値を設定
   - 過反射閾値: 実際のデータで、過反射領域の輝度分布を確認し、適切な閾値を設定

3. **面積比の調整**
   - 実際のデータで、問題となる暗部・過反射の面積比を確認
   - 診断への影響を考慮して、軽度・中度・重度の閾値を調整

4. **検証**
   - 調整後の閾値で、実際の画像を評価
   - 臨床的な基準と照らし合わせて、妥当性を確認

**推奨される調整範囲**:
- 暗部閾値: 0.10 - 0.20（より厳しく: 0.20、より緩く: 0.10）
- 過反射閾値: 0.85 - 0.95（より厳しく: 0.85、より緩く: 0.95）
- 面積比の閾値: 実際のデータの分布に応じて、±5%程度の調整を検討

#### 0.4.5 根拠のまとめ

| 数値 | 根拠 | 調整の必要性 |
|------|------|-------------|
| 暗部閾値 0.15 | 一般的な画像処理の経験則、網膜画像の特性 | **必須**（データに合わせて調整） |
| 過反射閾値 0.9 | 白飛びの定義、一般的な画像処理の慣習 | **必須**（データに合わせて調整） |
| 暗部最小面積 1% | ノイズ除去、実用的な影響の考慮 | **推奨**（必要に応じて調整） |
| 過反射最小面積 0.1% | 反射の特性、診断への影響 | **推奨**（必要に応じて調整） |
| 面積比の閾値（10%, 25%など） | 経験的な分類、診断への影響度 | **推奨**（臨床基準に合わせて調整） |
| 最小輝度（0.05, 0.02） | 情報損失の判断、一般的な画像処理 | **推奨**（必要に応じて調整） |
| 最大輝度（0.98） | 白飛びの定義、情報損失の判断 | **推奨**（必要に応じて調整） |

**重要な注意事項**:
- 本ドキュメントで提示している数値は、**初期提案値**であり、**実際のデータに合わせて調整が必要**です
- 特に、異なる撮影装置や撮影条件では、輝度分布が大きく異なる可能性があります
- 臨床的な基準や、実際の画像品質評価の結果に基づいて、閾値を調整することを強く推奨します

---

## 1. 検出対象の問題

### 1.1 暗部の問題
- **低輝度領域**: 網膜の一部が暗すぎて詳細が見えない
- **露出不足**: 全体的に暗い画像
- **影の影響**: レンズや器具による影

### 1.2 過反射（ハイライト）の問題
- **白飛び**: 高輝度領域で情報が失われている
- **反射光**: レンズや角膜からの反射
- **局所的な過露出**: 特定領域が明るすぎる

---

## 2. 検出方法の提案

### 2.1 輝度分布の統計的分析

#### 方法1: パーセンタイル分析
網膜領域内の輝度分布を分析し、暗部・明部の割合を評価します。

```python
def analyze_brightness_distribution(img_bgr: np.ndarray, retina_mask: np.ndarray) -> dict:
    """
    網膜領域内の輝度分布を分析
    
    Returns:
        {
            'mean_brightness': 平均輝度 (0-1)
            'std_brightness': 輝度の標準偏差
            'p5_brightness': 5パーセンタイル（暗部の指標）
            'p95_brightness': 95パーセンタイル（明部の指標）
            'dark_ratio': 暗部の割合（例: 輝度 < 0.1 のピクセル割合）
            'highlight_ratio': 過反射の割合（例: 輝度 > 0.9 のピクセル割合）
        }
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    
    # 網膜領域内のみを抽出
    mask_bool = retina_mask > 0
    if mask_bool.sum() < 100:
        return None
    
    roi_brightness = gray[mask_bool]
    
    # 統計量
    mean_bright = float(np.mean(roi_brightness))
    std_bright = float(np.std(roi_brightness))
    p5 = float(np.percentile(roi_brightness, 5))
    p95 = float(np.percentile(roi_brightness, 95))
    
    # 暗部・明部の割合
    dark_threshold = 0.1  # 調整可能
    highlight_threshold = 0.9  # 調整可能
    
    dark_ratio = float((roi_brightness < dark_threshold).sum() / roi_brightness.size)
    highlight_ratio = float((roi_brightness > highlight_threshold).sum() / roi_brightness.size)
    
    return {
        'mean_brightness': mean_bright,
        'std_brightness': std_bright,
        'p5_brightness': p5,
        'p95_brightness': p95,
        'dark_ratio': dark_ratio,
        'highlight_ratio': highlight_ratio,
    }
```

#### 評価基準
- **暗部が多い画像**: `dark_ratio > 0.15` または `p5_brightness < 0.05`
- **過反射が多い画像**: `highlight_ratio > 0.10` または `p95_brightness > 0.95`
- **適切な画像**: `0.15 ≤ mean_brightness ≤ 0.70` かつ `dark_ratio < 0.10` かつ `highlight_ratio < 0.05`

---

### 2.2 局所的な過反射検出

#### 方法2: 高輝度領域の検出と評価
局所的に明るすぎる領域（反射）を検出します。

```python
def detect_highlight_regions(img_bgr: np.ndarray, retina_mask: np.ndarray, 
                             highlight_threshold: float = 0.9,
                             min_area_ratio: float = 0.001) -> dict:
    """
    過反射領域を検出
    
    Parameters:
        highlight_threshold: 過反射とみなす輝度閾値（0-1）
        min_area_ratio: 検出対象とする最小面積比（網膜領域に対する割合）
    
    Returns:
        {
            'highlight_area_ratio': 過反射領域の面積比
            'highlight_count': 過反射領域の個数
            'max_highlight_intensity': 最大輝度値
            'highlight_score': 過反射の深刻度スコア（0-1、大きいほど悪い）
        }
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    
    # 網膜領域内のみを対象
    mask_bool = retina_mask > 0
    retina_area = mask_bool.sum()
    
    if retina_area < 100:
        return None
    
    # 過反射領域の検出
    highlight_mask = (gray > highlight_threshold) & mask_bool
    
    # 連結成分解析
    highlight_binary = (highlight_mask.astype(np.uint8) * 255)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(highlight_binary, connectivity=8)
    
    # 最小面積以上の領域のみをカウント
    min_area = int(retina_area * min_area_ratio)
    valid_regions = [i for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    
    highlight_area_ratio = float(highlight_mask.sum() / retina_area)
    highlight_count = len(valid_regions)
    
    # 最大輝度
    if highlight_mask.sum() > 0:
        max_highlight_intensity = float(gray[highlight_mask].max())
    else:
        max_highlight_intensity = 0.0
    
    # 過反射スコア（面積比と最大輝度の組み合わせ）
    highlight_score = min(1.0, highlight_area_ratio * 2.0 + (max_highlight_intensity - highlight_threshold) * 5.0)
    
    return {
        'highlight_area_ratio': highlight_area_ratio,
        'highlight_count': highlight_count,
        'max_highlight_intensity': max_highlight_intensity,
        'highlight_score': highlight_score,
    }
```

#### 評価基準
- **軽度の過反射**: `highlight_area_ratio < 0.05` かつ `highlight_count < 3`
- **中度の過反射**: `0.05 ≤ highlight_area_ratio < 0.15` または `3 ≤ highlight_count < 10`
- **重度の過反射**: `highlight_area_ratio ≥ 0.15` または `highlight_count ≥ 10` または `highlight_score > 0.5`

---

### 2.3 暗部領域の検出

#### 方法3: 低輝度領域の検出と評価
局所的に暗すぎる領域を検出します。

```python
def detect_dark_regions(img_bgr: np.ndarray, retina_mask: np.ndarray,
                       dark_threshold: float = 0.15,
                       min_area_ratio: float = 0.01) -> dict:
    """
    暗部領域を検出
    
    Parameters:
        dark_threshold: 暗部とみなす輝度閾値（0-1）
        min_area_ratio: 検出対象とする最小面積比
    
    Returns:
        {
            'dark_area_ratio': 暗部領域の面積比
            'dark_count': 暗部領域の個数
            'min_dark_intensity': 最小輝度値
            'dark_score': 暗部の深刻度スコア（0-1、大きいほど悪い）
        }
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    
    # 網膜領域内のみを対象
    mask_bool = retina_mask > 0
    retina_area = mask_bool.sum()
    
    if retina_area < 100:
        return None
    
    # 暗部領域の検出
    dark_mask = (gray < dark_threshold) & mask_bool
    
    # 連結成分解析
    dark_binary = (dark_mask.astype(np.uint8) * 255)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dark_binary, connectivity=8)
    
    # 最小面積以上の領域のみをカウント
    min_area = int(retina_area * min_area_ratio)
    valid_regions = [i for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    
    dark_area_ratio = float(dark_mask.sum() / retina_area)
    dark_count = len(valid_regions)
    
    # 最小輝度
    if dark_mask.sum() > 0:
        min_dark_intensity = float(gray[dark_mask].min())
    else:
        min_dark_intensity = 1.0
    
    # 暗部スコア（面積比と最小輝度の組み合わせ）
    dark_score = min(1.0, dark_area_ratio * 2.0 + (dark_threshold - min_dark_intensity) * 3.0)
    
    return {
        'dark_area_ratio': dark_area_ratio,
        'dark_count': dark_count,
        'min_dark_intensity': min_dark_intensity,
        'dark_score': dark_score,
    }
```

#### 評価基準
- **軽度の暗部**: `dark_area_ratio < 0.10` かつ `min_dark_intensity > 0.05`
- **中度の暗部**: `0.10 ≤ dark_area_ratio < 0.25` または `0.02 ≤ min_dark_intensity ≤ 0.05`
- **重度の暗部**: `dark_area_ratio ≥ 0.25` または `min_dark_intensity < 0.02` または `dark_score > 0.5`

---

### 2.4 コントラスト評価

#### 方法4: 局所コントラストの評価
暗部・明部が混在している場合、コントラストが低下している可能性があります。

```python
def evaluate_local_contrast(img_bgr: np.ndarray, retina_mask: np.ndarray,
                           block_size: int = 32) -> dict:
    """
    局所コントラストを評価
    
    Parameters:
        block_size: 局所領域のサイズ
    
    Returns:
        {
            'mean_local_contrast': 平均局所コントラスト
            'min_local_contrast': 最小局所コントラスト
            'low_contrast_ratio': 低コントラスト領域の割合
        }
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    
    mask_bool = retina_mask > 0
    if mask_bool.sum() < 100:
        return None
    
    # 局所標準偏差（コントラストの指標）
    local_std = cv2.GaussianBlur(gray ** 2, (block_size, block_size), block_size // 3)
    local_mean = cv2.GaussianBlur(gray, (block_size, block_size), block_size // 3)
    local_contrast = np.sqrt(np.maximum(0, local_std - local_mean ** 2))
    
    # 網膜領域内のみを評価
    roi_contrast = local_contrast[mask_bool]
    
    mean_contrast = float(np.mean(roi_contrast))
    min_contrast = float(np.min(roi_contrast))
    
    # 低コントラスト領域の割合（閾値: 0.05）
    low_contrast_threshold = 0.05
    low_contrast_ratio = float((roi_contrast < low_contrast_threshold).sum() / roi_contrast.size)
    
    return {
        'mean_local_contrast': mean_contrast,
        'min_local_contrast': min_contrast,
        'low_contrast_ratio': low_contrast_ratio,
    }
```

---

### 2.5 全体的なコントラスト評価（全体的に白い画像の検出）

#### 方法5: 全体的なコントラストの評価
全体的に白く、コントラストが低い画像を検出します。このような画像は、網膜の構造が識別困難です。

```python
def evaluate_global_contrast(img_bgr: np.ndarray, retina_mask: np.ndarray) -> dict:
    """
    全体的なコントラストを評価（全体的に白い画像の検出）
    
    Returns:
        {
            'global_contrast_std': 全体的なコントラスト（標準偏差）
            'brightness_range': 輝度の範囲（最大値 - 最小値）
            'histogram_width': ヒストグラムの幅（95パーセンタイル - 5パーセンタイル）
            'edge_density': エッジ密度（エッジピクセルの割合）
            'low_contrast_score': 低コントラストスコア（0-1、大きいほど悪い）
        }
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    
    # 網膜領域内のみを抽出
    mask_bool = retina_mask > 0
    if mask_bool.sum() < 100:
        return None
    
    roi_brightness = gray[mask_bool]
    
    # 1. 全体的なコントラスト（標準偏差）
    global_contrast_std = float(np.std(roi_brightness))
    
    # 2. 輝度の範囲
    brightness_range = float(np.max(roi_brightness) - np.min(roi_brightness))
    
    # 3. ヒストグラムの幅（95パーセンタイル - 5パーセンタイル）
    p5 = float(np.percentile(roi_brightness, 5))
    p95 = float(np.percentile(roi_brightness, 95))
    histogram_width = p95 - p5
    
    # 4. エッジ密度（Cannyエッジ検出）
    gray_uint8 = (gray * 255).astype(np.uint8)
    edges = cv2.Canny(gray_uint8, 50, 150)
    edge_mask = (edges > 0) & mask_bool
    edge_density = float(edge_mask.sum() / mask_bool.sum())
    
    # 5. 低コントラストスコア（複数の指標を組み合わせ）
    # 全体的に白い画像の特徴:
    # - 標準偏差が低い（< 0.1）
    # - 輝度範囲が狭い（< 0.3）
    # - ヒストグラムの幅が狭い（< 0.4）
    # - エッジが少ない（< 0.05）
    # - 平均輝度が高い（> 0.6）
    
    mean_brightness = float(np.mean(roi_brightness))
    
    # 各指標のスコア（0-1、大きいほど低コントラスト）
    std_score = max(0, (0.1 - global_contrast_std) / 0.1)  # std < 0.1 で問題
    range_score = max(0, (0.3 - brightness_range) / 0.3)  # range < 0.3 で問題
    hist_score = max(0, (0.4 - histogram_width) / 0.4)  # width < 0.4 で問題
    edge_score = max(0, (0.05 - edge_density) / 0.05)  # edge < 0.05 で問題
    brightness_score = max(0, (mean_brightness - 0.6) / 0.4) if mean_brightness > 0.6 else 0  # mean > 0.6 で問題
    
    # 重み付き和（全体的に白い画像の特徴を重視）
    low_contrast_score = (
        0.3 * std_score +
        0.2 * range_score +
        0.2 * hist_score +
        0.2 * edge_score +
        0.1 * brightness_score
    )
    low_contrast_score = min(1.0, low_contrast_score)
    
    return {
        'global_contrast_std': global_contrast_std,
        'brightness_range': brightness_range,
        'histogram_width': histogram_width,
        'edge_density': edge_density,
        'mean_brightness': mean_brightness,
        'low_contrast_score': low_contrast_score,
    }
```

#### 評価基準
- **適切なコントラスト**: 
  - `global_contrast_std ≥ 0.12` かつ
  - `brightness_range ≥ 0.35` かつ
  - `histogram_width ≥ 0.45` かつ
  - `edge_density ≥ 0.06` かつ
  - `low_contrast_score < 0.3`

- **軽度の低コントラスト**: 
  - `0.10 ≤ global_contrast_std < 0.12` または
  - `0.3 ≤ brightness_range < 0.35` または
  - `0.3 ≤ low_contrast_score < 0.5`

- **中度の低コントラスト**: 
  - `0.08 ≤ global_contrast_std < 0.10` または
  - `0.25 ≤ brightness_range < 0.3` または
  - `0.5 ≤ low_contrast_score < 0.7`

- **重度の低コントラスト（全体的に白い）**: 
  - `global_contrast_std < 0.08` または
  - `brightness_range < 0.25` または
  - `histogram_width < 0.3` または
  - `edge_density < 0.03` または
  - `low_contrast_score ≥ 0.7`

#### 全体的に白い画像の特徴
1. **標準偏差が低い**: 輝度のばらつきが小さい（< 0.08-0.10）
2. **輝度範囲が狭い**: 最大値と最小値の差が小さい（< 0.25-0.3）
3. **ヒストグラムが狭い**: 輝度分布が集中している（幅 < 0.3-0.4）
4. **エッジが少ない**: 構造的な変化が少ない（エッジ密度 < 0.03-0.05）
5. **平均輝度が高い**: 全体的に明るい（> 0.6）

#### この方法の一般的性について

**一般的な画像品質評価手法**:
本ドキュメントで提案している方法は、**画像品質評価の分野で広く用いられている標準的な手法**を組み合わせたものです：

1. **標準偏差によるコントラスト評価**: 
   - 画像処理の教科書や研究で一般的に使用される方法
   - 輝度の分散を測定することで、コントラストを定量的に評価

2. **ヒストグラム解析**: 
   - 画像品質評価の基本的な手法
   - 輝度分布の広がり（パーセンタイル差）を評価することで、コントラストを判断

3. **エッジ検出（Canny）**: 
   - 画像品質評価で広く使用される手法
   - エッジの密度や強度を評価することで、画像の詳細度を測定
   - Sobelフィルタなども同様の目的で使用される

4. **複数指標の組み合わせ**: 
   - 単一の指標では誤検出の可能性があるため、複数の指標を組み合わせる手法は一般的
   - 重み付き和や閾値ベースの判定は標準的なアプローチ

**網膜画像の品質評価における使用**:
- 網膜画像（fundus image）の品質評価においても、これらの手法は広く採用されています
- 特に、標準偏差とエッジ検出は、網膜画像の品質評価研究でよく使用される指標です

**他の一般的な手法（参考）**:
- **RMSコントラスト（Root Mean Square Contrast）**: より厳密なコントラスト測定
- **Michelsonコントラスト**: 周期的パターンに対するコントラスト測定
- **Weberコントラスト**: 背景輝度に対するコントラスト測定
- **局所コントラスト（Local Contrast）**: 局所的なコントラストの評価（本ドキュメントの方法4で使用）

**注意**: 
- 本ドキュメントで提案している閾値（0.08, 0.25など）は、一般的な画像処理の経験則に基づく**初期提案値**です
- 実際の網膜画像データに合わせて調整が必要です
- 異なる撮影装置や撮影条件では、適切な閾値が異なる可能性があります

---

### 2.6 統合スコア

#### 方法6: 総合的な品質スコア
上記の指標を組み合わせて、暗部・過反射・低コントラストの総合評価を行います。

```python
def compute_brightness_quality_score(brightness_stats: dict,
                                     highlight_stats: dict,
                                     dark_stats: dict,
                                     local_contrast_stats: dict = None,
                                     global_contrast_stats: dict = None) -> dict:
    """
    輝度品質の総合スコアを計算
    
    Parameters:
        brightness_stats: 輝度分布の統計
        highlight_stats: 過反射の統計
        dark_stats: 暗部の統計
        local_contrast_stats: 局所コントラストの統計
        global_contrast_stats: 全体的なコントラストの統計（全体的に白い画像の検出）
    
    Returns:
        {
            'brightness_quality_score': 総合スコア（0-1、大きいほど良い）
            'is_acceptable': 許容範囲内かどうか（bool）
            'issues': 問題点のリスト
        }
    """
    issues = []
    score_penalty = 0.0
    
    # 暗部の評価
    if dark_stats:
        if dark_stats['dark_area_ratio'] > 0.25:
            issues.append('重度の暗部')
            score_penalty += 0.4
        elif dark_stats['dark_area_ratio'] > 0.10:
            issues.append('中度の暗部')
            score_penalty += 0.2
        
        if dark_stats['min_dark_intensity'] < 0.02:
            issues.append('極端に暗い領域')
            score_penalty += 0.3
    
    # 過反射の評価
    if highlight_stats:
        if highlight_stats['highlight_area_ratio'] > 0.15:
            issues.append('重度の過反射')
            score_penalty += 0.4
        elif highlight_stats['highlight_area_ratio'] > 0.05:
            issues.append('中度の過反射')
            score_penalty += 0.2
        
        if highlight_stats['max_highlight_intensity'] > 0.98:
            issues.append('白飛び領域')
            score_penalty += 0.3
    
    # 輝度分布の評価
    if brightness_stats:
        if brightness_stats['mean_brightness'] < 0.15:
            issues.append('全体的に暗い')
            score_penalty += 0.2
        elif brightness_stats['mean_brightness'] > 0.70:
            issues.append('全体的に明るい')
            score_penalty += 0.1
        
        if brightness_stats['dark_ratio'] > 0.15:
            issues.append('暗部ピクセルが多い')
            score_penalty += 0.15
    
    # 局所コントラストの評価
    if local_contrast_stats:
        if local_contrast_stats['low_contrast_ratio'] > 0.30:
            issues.append('低コントラスト領域が多い')
            score_penalty += 0.2
    
    # 全体的なコントラストの評価（全体的に白い画像の検出）
    if global_contrast_stats:
        low_contrast_score = global_contrast_stats.get('low_contrast_score', 0.0)
        global_contrast_std = global_contrast_stats.get('global_contrast_std', 1.0)
        brightness_range = global_contrast_stats.get('brightness_range', 1.0)
        edge_density = global_contrast_stats.get('edge_density', 1.0)
        
        # 重度の低コントラスト（全体的に白い）
        if low_contrast_score >= 0.7:
            issues.append('全体的に白い（重度の低コントラスト）')
            score_penalty += 0.5  # 重大な問題として高いペナルティ
        elif low_contrast_score >= 0.5:
            issues.append('全体的に白い（中度の低コントラスト）')
            score_penalty += 0.3
        elif low_contrast_score >= 0.3:
            issues.append('全体的に白い（軽度の低コントラスト）')
            score_penalty += 0.15
        
        # 個別指標による判定
        if global_contrast_std < 0.08:
            issues.append('コントラストが極端に低い')
            score_penalty += 0.3
        elif global_contrast_std < 0.10:
            issues.append('コントラストが低い')
            score_penalty += 0.15
        
        if brightness_range < 0.25:
            issues.append('輝度範囲が狭い')
            score_penalty += 0.2
        
        if edge_density < 0.03:
            issues.append('エッジが極端に少ない')
            score_penalty += 0.2
    
    # 総合スコア（1.0から減点方式）
    brightness_quality_score = max(0.0, 1.0 - score_penalty)
    
    # 許容範囲の判定（スコア > 0.6 かつ重大な問題がない）
    is_acceptable = (brightness_quality_score > 0.6) and \
                    ('重度の暗部' not in issues) and \
                    ('重度の過反射' not in issues) and \
                    ('白飛び領域' not in issues) and \
                    ('全体的に白い（重度の低コントラスト）' not in issues) and \
                    ('コントラストが極端に低い' not in issues)
    
    return {
        'brightness_quality_score': brightness_quality_score,
        'is_acceptable': is_acceptable,
        'issues': issues,
    }
```

---

## 3. 除外プロトコル（まとめ）

### 3.1 プロトコル概要

網膜画像の品質評価において、以下の3つの問題を検出・除外します：

1. **暗すぎる画像（暗部が多い）**
2. **明るすぎる画像（過反射が多い）**
3. **コントラストが低い画像（全体的に白い）**

### 3.2 評価フロー

```
画像入力
  ↓
網膜領域の抽出（retina_mask）
  ↓
┌─────────────────────────────────────┐
│ Step 1: 輝度分布の統計的分析        │
│ - 平均輝度、標準偏差                │
│ - パーセンタイル（5%, 95%）         │
│ - 暗部・過反射ピクセルの割合        │
└─────────────────────────────────────┘
  ↓
┌─────────────────────────────────────┐
│ Step 2: 暗部領域の検出              │
│ - 輝度 < 0.15 の連結領域            │
│ - 面積比、個数、最小輝度を評価       │
└─────────────────────────────────────┘
  ↓
┌─────────────────────────────────────┐
│ Step 3: 過反射領域の検出            │
│ - 輝度 > 0.9 の連結領域             │
│ - 面積比、個数、最大輝度を評価       │
└─────────────────────────────────────┘
  ↓
┌─────────────────────────────────────┐
│ Step 4: 全体的なコントラスト評価    │
│ - 標準偏差、輝度範囲                │
│ - ヒストグラム幅、エッジ密度         │
│ - 低コントラストスコアを計算        │
└─────────────────────────────────────┘
  ↓
┌─────────────────────────────────────┐
│ Step 5: 統合スコアの計算            │
│ - 各指標を組み合わせて総合評価       │
│ - 許容範囲の判定（is_acceptable）    │
└─────────────────────────────────────┘
  ↓
除外判定
```

### 3.3 除外基準（デフォルト設定）

#### 3.3.1 暗すぎる画像の除外基準

**除外条件（いずれかを満たす場合）**:
- `dark_area_ratio ≥ 0.25` （暗部面積比が25%以上）
- `min_dark_intensity < 0.02` （最小輝度が0.02未満）
- `mean_brightness < 0.15` （平均輝度が0.15未満）
- `dark_ratio > 0.15` （暗部ピクセルが15%以上）

**警告条件（除外しないが注意が必要）**:
- `0.10 ≤ dark_area_ratio < 0.25` （暗部面積比が10-25%）
- `0.02 ≤ min_dark_intensity < 0.05` （最小輝度が0.02-0.05）

#### 3.3.2 明るすぎる画像の除外基準

**除外条件（いずれかを満たす場合）**:
- `highlight_area_ratio ≥ 0.15` （過反射面積比が15%以上）
- `highlight_count ≥ 10` （過反射領域が10個以上）
- `max_highlight_intensity > 0.98` （最大輝度が0.98以上）
- `highlight_ratio > 0.10` （過反射ピクセルが10%以上）
- `mean_brightness > 0.70` （平均輝度が0.70以上）

**警告条件（除外しないが注意が必要）**:
- `0.05 ≤ highlight_area_ratio < 0.15` （過反射面積比が5-15%）
- `3 ≤ highlight_count < 10` （過反射領域が3-10個）

#### 3.3.3 コントラストが低い画像（全体的に白い）の除外基準

**除外条件（いずれかを満たす場合）**:
- `global_contrast_std < 0.08` （標準偏差が0.08未満）
- `brightness_range < 0.25` （輝度範囲が0.25未満）
- `histogram_width < 0.3` （ヒストグラム幅が0.3未満）
- `edge_density < 0.03` （エッジ密度が0.03未満）
- `low_contrast_score ≥ 0.7` （低コントラストスコアが0.7以上）

**警告条件（除外しないが注意が必要）**:
- `0.08 ≤ global_contrast_std < 0.10` （標準偏差が0.08-0.10）
- `0.25 ≤ brightness_range < 0.30` （輝度範囲が0.25-0.30）
- `0.5 ≤ low_contrast_score < 0.7` （低コントラストスコアが0.5-0.7）

### 3.4 統合判定

#### 3.4.1 許容範囲の判定

```python
is_acceptable = (
    brightness_quality_score > 0.6  # 総合スコアが0.6以上
) and (
    '重度の暗部' not in issues  # 重度の暗部がない
) and (
    '重度の過反射' not in issues  # 重度の過反射がない
) and (
    '白飛び領域' not in issues  # 白飛び領域がない
) and (
    '全体的に白い（重度の低コントラスト）' not in issues  # 重度の低コントラストがない
) and (
    'コントラストが極端に低い' not in issues  # 極端に低いコントラストがない
)
```

#### 3.4.2 除外プロトコルの実装

**基本実装（推奨）**:
```python
# 有効データ抽出
valid = _df[
    (_df['lens_detected'] == True) & 
    (_df['retina_ratio'] > 0) &
    (_df['brightness_acceptable'] == True)  # 統合判定を使用
].copy()
```

**より厳しい除外（オプション）**:
```python
# 基本条件
valid = _df[
    (_df['lens_detected'] == True) & 
    (_df['retina_ratio'] > 0) &
    (_df['brightness_acceptable'] == True)
].copy()

# 追加の厳しい条件
valid = valid[
    # 暗部の除外
    (valid['dark_area_ratio'] < 0.25) &
    (valid['min_dark_intensity'] >= 0.02) &
    # 過反射の除外
    (valid['highlight_area_ratio'] < 0.15) &
    (valid['max_highlight_intensity'] <= 0.98) &
    # 低コントラストの除外
    (valid['low_contrast_score'] < 0.7) &
    (valid['global_contrast_std'] >= 0.08) &
    (valid['brightness_range'] >= 0.25) &
    (valid['edge_density'] >= 0.03)
].copy()
```

### 3.5 除外基準のまとめ表

| 問題 | 除外指標 | 除外閾値 | 警告閾値 |
|------|---------|---------|---------|
| **暗すぎる** | `dark_area_ratio` | ≥ 0.25 | 0.10 - 0.25 |
| | `min_dark_intensity` | < 0.02 | 0.02 - 0.05 |
| | `mean_brightness` | < 0.15 | - |
| | `dark_ratio` | > 0.15 | - |
| **明るすぎる** | `highlight_area_ratio` | ≥ 0.15 | 0.05 - 0.15 |
| | `highlight_count` | ≥ 10 | 3 - 10 |
| | `max_highlight_intensity` | > 0.98 | - |
| | `mean_brightness` | > 0.70 | - |
| **低コントラスト** | `global_contrast_std` | < 0.08 | 0.08 - 0.10 |
| | `brightness_range` | < 0.25 | 0.25 - 0.30 |
| | `histogram_width` | < 0.3 | - |
| | `edge_density` | < 0.03 | - |
| | `low_contrast_score` | ≥ 0.7 | 0.5 - 0.7 |

### 3.6 プロトコルの適用順序

1. **推論実行**: 各画像に対して推論を実行し、網膜マスクを取得
2. **品質評価**: 網膜マスク内で品質指標を計算
3. **統合スコア**: 各指標を組み合わせて総合スコアを計算
4. **除外判定**: `brightness_acceptable == True` の画像のみを選別対象とする
5. **ランキング**: 除外されなかった画像に対して、既存のランキングを適用

### 3.7 パラメータ調整の推奨事項

- **初期設定**: デフォルトの閾値で評価を開始
- **データ確認**: 実際のデータで除外される画像を確認
- **閾値調整**: 除外が多すぎる/少なすぎる場合は、閾値を段階的に調整
- **検証**: 調整後の閾値で、実際の画像を目視確認して妥当性を検証

---

## 4. 実装例（validate_images.ipynbへの統合）

既存の `process_one_image` 関数に追加する例：

```python
def compute_brightness_quality(img_bgr: np.ndarray, retina_mask: np.ndarray) -> dict:
    """網膜画像の輝度品質を評価"""
    # 各指標を計算
    brightness_stats = analyze_brightness_distribution(img_bgr, retina_mask)
    highlight_stats = detect_highlight_regions(img_bgr, retina_mask)
    dark_stats = detect_dark_regions(img_bgr, retina_mask)
    local_contrast_stats = evaluate_local_contrast(img_bgr, retina_mask)
    global_contrast_stats = evaluate_global_contrast(img_bgr, retina_mask)  # 全体的なコントラスト評価を追加
    
    # 統合スコア
    quality_score = compute_brightness_quality_score(
        brightness_stats, highlight_stats, dark_stats, 
        local_contrast_stats, global_contrast_stats  # 全体的なコントラスト評価を追加
    )
    
    # 結果を統合
    result = {
        'mean_brightness': brightness_stats.get('mean_brightness') if brightness_stats else None,
        'dark_ratio': brightness_stats.get('dark_ratio') if brightness_stats else None,
        'highlight_ratio': brightness_stats.get('highlight_ratio') if brightness_stats else None,
        'highlight_area_ratio': highlight_stats.get('highlight_area_ratio') if highlight_stats else None,
        'highlight_count': highlight_stats.get('highlight_count') if highlight_stats else None,
        'dark_area_ratio': dark_stats.get('dark_area_ratio') if dark_stats else None,
        'dark_count': dark_stats.get('dark_count') if dark_stats else None,
        # 全体的なコントラスト指標を追加
        'global_contrast_std': global_contrast_stats.get('global_contrast_std') if global_contrast_stats else None,
        'brightness_range': global_contrast_stats.get('brightness_range') if global_contrast_stats else None,
        'edge_density': global_contrast_stats.get('edge_density') if global_contrast_stats else None,
        'low_contrast_score': global_contrast_stats.get('low_contrast_score') if global_contrast_stats else None,
        'brightness_quality_score': quality_score.get('brightness_quality_score'),
        'brightness_acceptable': quality_score.get('is_acceptable'),
    }
    
    return result
```

`process_one_image` 関数内で呼び出し：

```python
# 既存のMBSS計算の後に追加
if retina_mask_crop is not None:
    brightness_quality = compute_brightness_quality(cropped, retina_mask_crop)
    # 結果に追加
    result.update(brightness_quality)
```

---

## 5. 選別基準への組み込み

### 4.1 フィルタリング条件
選別前に、品質の低い画像を除外：

```python
# 有効データ抽出の際に追加
valid = _df[
    (_df['lens_detected'] == True) & 
    (_df['retina_ratio'] > 0) &
    (_df['brightness_acceptable'] == True)  # 暗部・過反射・低コントラストの評価
].copy()

# より厳しい条件で全体的に白い画像を除外する場合
# valid = valid[
#     (valid['low_contrast_score'] < 0.7) &  # 重度の低コントラストを除外
#     (valid['global_contrast_std'] >= 0.08)  # コントラストが極端に低いものを除外
# ].copy()
```

### 4.2 ランキングへの組み込み
既存の `rank_sum` に輝度品質スコアを追加：

```python
# 輝度品質のランク（大きいほど良い）
valid['brightness_quality_rank'] = valid['brightness_quality_score'].rank(
    ascending=False, method='min', na_option='bottom'
)

# rank_sumに追加（重み: 0.3）
valid['rank_sum'] = (
    1.5 * valid['retina_area_rank'] +
    1.0 * valid['mbss_rank'] +
    0.5 * valid['disc_ring_rank'] +
    0.5 * valid['s_mean_rank'] +
    0.3 * valid['brightness_quality_rank']  # 追加
)
```

---

## 6. パラメータ調整の指針

### 5.1 閾値の調整
実際のデータに応じて以下のパラメータを調整：

- **dark_threshold**: 暗部とみなす輝度（デフォルト: 0.15）
  - より厳しく: 0.20
  - より緩く: 0.10

- **highlight_threshold**: 過反射とみなす輝度（デフォルト: 0.9）
  - より厳しく: 0.85
  - より緩く: 0.95

- **min_area_ratio**: 検出対象の最小面積比
  - 小さな反射を無視: 0.005
  - 小さな反射も検出: 0.001

- **全体的なコントラスト評価の閾値**:
  - `global_contrast_std`: コントラストの標準偏差（デフォルト: 0.08-0.12が問題）
    - より厳しく: 0.10以上を要求
    - より緩く: 0.06以上を許容
  - `brightness_range`: 輝度範囲（デフォルト: 0.25-0.35が問題）
    - より厳しく: 0.30以上を要求
    - より緩く: 0.20以上を許容
  - `edge_density`: エッジ密度（デフォルト: 0.03-0.05が問題）
    - より厳しく: 0.05以上を要求
    - より緩く: 0.02以上を許容
  - Cannyエッジ検出の閾値: `cv2.Canny(gray, 50, 150)` の50, 150を調整可能
    - より敏感に: `cv2.Canny(gray, 30, 100)`
    - より鈍感に: `cv2.Canny(gray, 70, 200)`

### 5.2 許容範囲の調整
`compute_brightness_quality_score` 内の閾値を調整：

- **許容範囲を広げる**: 各 `score_penalty` を小さく
- **許容範囲を狭める**: 各 `score_penalty` を大きく、または閾値を厳しく

---

## 7. 注意事項

1. **網膜領域内での評価**: 必ず網膜マスク内でのみ評価を行う（背景の影響を排除）

2. **ケース間の正規化**: ケースごとに輝度分布が異なる可能性があるため、z-score正規化を検討

3. **計算コスト**: 
   - 局所コントラスト評価は計算コストが高いため、必要に応じてスキップ可能にする
   - 全体的なコントラスト評価（エッジ検出を含む）も計算コストが高いため、必要に応じて最適化を検討

4. **閾値の妥当性**: 実際のデータで検証し、閾値を調整する必要がある

5. **既存指標との重複**: `S_mean`（彩度）と一部重複する可能性があるため、重みの調整を検討

6. **全体的に白い画像の検出**: 
   - 全体的なコントラスト評価は、複数の指標（標準偏差、輝度範囲、エッジ密度など）を組み合わせて判定
   - 単一の指標だけでは誤検出の可能性があるため、複数指標の組み合わせが重要
   - エッジ検出のパラメータ（Cannyの閾値）は、画像の特性に応じて調整が必要な場合がある

7. **低コントラスト画像の除外**: 
   - 全体的に白い画像は、診断に使用できない可能性が高いため、フィルタリング条件に含めることを推奨
   - `low_contrast_score ≥ 0.7` または `global_contrast_std < 0.08` の画像は除外を検討

---

## 7. 参考

- 網膜画像の品質評価に関する研究では、輝度分布、コントラスト、反射検出が重要な指標とされています
- 臨床的な基準に合わせて、閾値や重みを調整することが推奨されます

