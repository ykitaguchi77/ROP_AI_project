"""
画像品質評価モジュール
validate_images.ipynbのアルゴリズムを移植
"""
import cv2
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List
from tqdm import tqdm
import torch
from ultralytics import RTDETR, YOLO

# YOLO入力幅（ノートブック内で固定）
YOLO_INPUT_WIDTH = 640


# ==================== 画像品質特徴量（MBSS） ====================

def to_gray_float(img_bgr_or_gray: np.ndarray) -> np.ndarray:
    """BGR/Gray いずれも float32 [0,1] グレースケールへ"""
    if img_bgr_or_gray.ndim == 3:
        gray = cv2.cvtColor(img_bgr_or_gray, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_bgr_or_gray
    gray = gray.astype(np.float32)
    if gray.max() > 1.0:
        gray /= 255.0
    return gray


def laplacian_multi_var(gray01: np.ndarray, sigmas=(1.0, 2.0, 4.0), weights=(0.5, 0.3, 0.2)) -> float:
    """マルチスケール Laplacian 分散（重み付き和）"""
    vals = []
    for s, w in zip(sigmas, weights):
        ksize = int(6 * s + 1)
        if ksize % 2 == 0:
            ksize += 1
        blur = cv2.GaussianBlur(gray01, (ksize, ksize), s)
        lap = cv2.Laplacian(blur, cv2.CV_32F, ksize=3)
        vals.append(w * float(lap.var()))
    return float(np.sum(vals))


def fft_features(gray01: np.ndarray, high_freq_thresh=0.3) -> tuple:
    """FFT高周波エネルギー比とスペクトル重心（NumPy FFT版）"""
    h, w = gray01.shape

    wy = np.hanning(h).astype(np.float32)
    wx = np.hanning(w).astype(np.float32)
    window = np.outer(wy, wx)
    g = gray01 * window

    F = np.fft.fftshift(np.fft.fft2(g))
    mag2 = (np.abs(F) ** 2).astype(np.float64)

    cy, cx = h // 2, w // 2
    yy, xx = np.indices((h, w))
    ry = (yy - cy) / float(max(cy, 1))
    rx = (xx - cx) / float(max(cx, 1))
    r = np.sqrt(rx ** 2 + ry ** 2)
    r_norm = np.clip(r, 0, 1)

    total = mag2.sum() + 1e-12
    high_mask = r_norm > high_freq_thresh
    hf_ratio = float(mag2[high_mask].sum() / total)
    spec_centroid = float((r_norm * mag2).sum() / total)
    return hf_ratio, spec_centroid


def grad_percentile(gray01: np.ndarray, p=90) -> float:
    """勾配強度のパーセンタイル"""
    gx = cv2.Sobel(gray01, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray01, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return float(np.percentile(mag, p))


def compute_mbss_components(img_bgr: np.ndarray, mask01: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Retina領域（mask）内のみで MBSS コンポーネントを算出

    NOTE:
    - 既存のMBSS系（Laplacian/FFT/Grad）はマスク外を0にして計算
    - 色調用に、網膜マスク内の彩度平均 `S_mean`（HSVのS）も同時に返す
    """
    gray = to_gray_float(img_bgr)

    mask_bool = None
    if mask01 is not None:
        if mask01.shape != gray.shape:
            mask01 = cv2.resize(mask01.astype(np.uint8), (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask_bool = mask01 > 0
        if mask_bool.sum() < 100:
            return {"L_multi": None, "HF_ratio": None, "Spec_centroid": None, "Grad_p90": None, "S_mean": None}
        gray2 = gray.copy()
        gray2[~mask_bool] = 0.0
    else:
        gray2 = gray

    # --- 色調（HSV彩度Sの平均、網膜マスク内） ---
    s_mean = None
    if mask_bool is not None and img_bgr is not None and getattr(img_bgr, "ndim", 0) == 3:
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1].astype(np.float32)
        if s.max() > 1.0:
            s /= 255.0
        roi = s[mask_bool]
        if roi.size > 0:
            s_mean = float(np.mean(roi))

    return {
        "L_multi": laplacian_multi_var(gray2),
        "HF_ratio": fft_features(gray2)[0],
        "Spec_centroid": fft_features(gray2)[1],
        "Grad_p90": grad_percentile(gray2),
        "S_mean": s_mean,
    }


def compute_mbss_score(components: dict, stats: dict, weights=None) -> Optional[float]:
    """z-score正規化後、重み付き和でスコア化"""
    if any(components.get(k) is None for k in ["L_multi", "HF_ratio", "Spec_centroid", "Grad_p90"]):
        return None

    if weights is None:
        weights = {"L_multi": 0.35, "HF_ratio": 0.25, "Spec_centroid": 0.20, "Grad_p90": 0.20}

    score = 0.0
    for k, w in weights.items():
        x = float(components[k])
        m = float(stats[k]["mean"])
        s = float(stats[k]["std"]) + 1e-8
        z = (x - m) / s
        score += w * z
    return float(score)


# ==================== Disc Edge Coverage（辺縁被覆率） ====================

def compute_disc_edge_coverage(disc_mask: np.ndarray, retina_mask: np.ndarray) -> tuple:
    """
    Discの辺縁がRetinaマスクに覆われているかを計算

    Args:
        disc_mask: Discマスク（0/255 or 0/1）
        retina_mask: Retinaマスク（0/255 or 0/1）

    Returns:
        tuple: (disc_edge_covered: bool, disc_edge_coverage_ratio: float)
               - disc_edge_covered: True if coverage >= 95%
               - disc_edge_coverage_ratio: 0-1の被覆率
    """
    if disc_mask is None or retina_mask is None:
        return None, None

    disc_bin = (disc_mask > 0).astype(np.uint8)
    retina_bin = (retina_mask > 0).astype(np.uint8)

    if disc_bin.sum() == 0:
        return None, None

    # Discマスクの輪郭（辺縁）を抽出
    kernel = np.ones((3, 3), np.uint8)
    disc_eroded = cv2.erode(disc_bin, kernel, iterations=1)
    disc_edge = disc_bin - disc_eroded

    total_edge_pixels = disc_edge.sum()
    if total_edge_pixels == 0:
        return None, None

    # Retinaマスクを少し膨張させて境界付近でも検出
    retina_dilated = cv2.dilate(retina_bin, kernel, iterations=2)
    covered_edge_pixels = (disc_edge & retina_dilated).sum()

    coverage_ratio = covered_edge_pixels / total_edge_pixels
    is_covered = coverage_ratio >= 0.95

    return is_covered, float(coverage_ratio)


# ==================== Disc周囲（core/ring）評価 ====================

def estimate_disc_center_radius(disc_mask01: np.ndarray):
    """discマスクから中心(cx,cy)と代表半径Rを推定"""
    m = disc_mask01.astype(np.uint8)
    if m.max() > 1:
        m = (m > 0).astype(np.uint8)

    num_labels, labels = cv2.connectedComponents(m)
    if num_labels > 1:
        areas = [(labels == i).sum() for i in range(1, num_labels)]
        main_label = int(np.argmax(areas) + 1)
        m = (labels == main_label).astype(np.uint8)

    M = cv2.moments(m)
    if M["m00"] == 0:
        return None

    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    area = float(m.sum())
    R = float(np.sqrt(area / np.pi))
    return cx, cy, R


def make_disc_rois(shape_hw, cx, cy, R, inner_ratio=0.6, outer_ratio=1.2):
    """Discのcore/ring領域を作成"""
    h, w = shape_hw
    yy, xx = np.indices((h, w))
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    core = dist < (inner_ratio * R)
    ring = (dist >= (inner_ratio * R)) & (dist < (outer_ratio * R))
    return core.astype(np.uint8), ring.astype(np.uint8)


def laplacian_multi_var_masked(gray01: np.ndarray, mask01: np.ndarray, sigmas=(1.0, 2.0, 4.0), weights=(0.5, 0.3, 0.2)) -> float:
    """マスク内でのマルチスケールLaplacian分散"""
    mask_bool = mask01.astype(bool)
    if mask_bool.sum() < 50:
        return 0.0

    vals = []
    for s, w in zip(sigmas, weights):
        ksize = int(6 * s + 1)
        if ksize % 2 == 0:
            ksize += 1
        blur = cv2.GaussianBlur(gray01, (ksize, ksize), s)
        lap = cv2.Laplacian(blur, cv2.CV_32F, ksize=3)
        roi = lap[mask_bool]
        if roi.size == 0:
            continue
        vals.append(w * float(roi.var()))
    return float(np.sum(vals)) if vals else 0.0


def compute_disc_sharpness_components(img_bgr: np.ndarray, disc_mask01: np.ndarray):
    """disc中心(core)と周辺(ring)の L_multi を返す"""
    gray = to_gray_float(img_bgr)
    est = estimate_disc_center_radius(disc_mask01)
    if est is None:
        return None, None

    cx, cy, R = est
    core_mask, ring_mask = make_disc_rois(gray.shape, cx, cy, R)

    if core_mask.sum() < 50 or ring_mask.sum() < 50:
        return None, None

    L_core = laplacian_multi_var_masked(gray, core_mask)
    L_ring = laplacian_multi_var_masked(gray, ring_mask)
    return L_core, L_ring


# ==================== メイン処理関数 ====================

def process_one_image(image_path: str, detection_model, segmentation_model) -> Optional[dict]:
    """1枚の画像に対して推論 + 特徴量を算出"""
    image = cv2.imread(image_path)
    if image is None:
        return None

    # --- Stage 1: RT-DETRでLens bbox検出（cls=0想定） ---
    det_results = detection_model(image, verbose=False)
    lens_bbox_xyxy = None
    for r in det_results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        for box in r.boxes:
            if int(box.cls) == 0:
                lens_bbox_xyxy = box.xyxy[0].cpu().numpy()
                break
        if lens_bbox_xyxy is not None:
            break

    if lens_bbox_xyxy is None:
        return {
            'image_path': image_path,
            'lens_detected': False,
            'lens_area': 0,
            'retina_area': 0,
            'retina_ratio': 0.0,
            'disc_detected': False,
            'macula_detected': False,
            'mbss_L_multi': None,
            'mbss_HF_ratio': None,
            'mbss_Spec_centroid': None,
            'mbss_Grad_p90': None,
            'S_mean': None,
            'disc_core_L_multi': None,
            'disc_ring_L_multi': None,
            'disc_center_dist_ratio': None,
            'disc_pos_ok': None,
            'disc_edge_covered': None,
            'disc_edge_coverage_ratio': None,
        }

    x1, y1, x2, y2 = [int(c) for c in lens_bbox_xyxy]
    cropped = image[y1:y2, x1:x2]
    if cropped.size == 0:
        return {
            'image_path': image_path,
            'lens_detected': True,
            'lens_area': 0,
            'retina_area': 0,
            'retina_ratio': 0.0,
            'disc_detected': False,
            'macula_detected': False,
            'mbss_L_multi': None,
            'mbss_HF_ratio': None,
            'mbss_Spec_centroid': None,
            'mbss_Grad_p90': None,
            'S_mean': None,
            'disc_core_L_multi': None,
            'disc_ring_L_multi': None,
            'disc_center_dist_ratio': None,
            'disc_pos_ok': None,
            'disc_edge_covered': None,
            'disc_edge_coverage_ratio': None,
        }

    # --- Lens内での円形マスク（レンズ外を灰色にする） ---
    orig_h, orig_w = cropped.shape[:2]
    center_x = orig_w // 2
    center_y = orig_h // 2
    diameter = (orig_w + orig_h) / 2
    radius = int(diameter / 2)

    circle_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
    cv2.circle(circle_mask, (center_x, center_y), radius, 255, -1)

    masked_cropped = cropped.copy()
    masked_cropped[circle_mask == 0] = (114, 114, 114)

    lens_area = int((circle_mask > 0).sum())

    # --- Stage 2: YOLO-seg ---
    aspect_ratio = orig_h / max(orig_w, 1)
    yolo_h = int(YOLO_INPUT_WIDTH * aspect_ratio)
    yolo_input = cv2.resize(masked_cropped, (YOLO_INPUT_WIDTH, yolo_h), interpolation=cv2.INTER_AREA)

    seg_results = segmentation_model(yolo_input, verbose=False, retina_masks=True)

    retina_area = 0
    disc_detected = False
    macula_detected = False

    retina_mask_crop = None
    disc_mask_crop = None

    if seg_results and seg_results[0].masks is not None:
        r0 = seg_results[0]
        masks = r0.masks.data.cpu().numpy()
        classes = r0.boxes.cls.cpu().numpy().astype(int)

        for mask_data, cls_id in zip(masks, classes):
            # mask_data: (H', W') 0..1
            mask_resized = cv2.resize(mask_data, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            mask_bin = (mask_resized > 0.5) & (circle_mask > 0)

            if cls_id == 0:  # Fundus/Retina
                retina_area = int(mask_bin.sum())
                retina_mask_crop = (mask_bin.astype(np.uint8) * 255)
            elif cls_id == 1:  # Disc
                disc_detected = True
                disc_mask_crop = (mask_bin.astype(np.uint8) * 255)
            elif cls_id == 2:  # Macula
                macula_detected = True

    retina_ratio = (retina_area / lens_area * 100.0) if lens_area > 0 else 0.0

    # --- MBSS（Retina領域内） ---
    if retina_mask_crop is not None:
        mb = compute_mbss_components(cropped, mask01=retina_mask_crop)
    else:
        mb = {"L_multi": None, "HF_ratio": None, "Spec_centroid": None, "Grad_p90": None, "S_mean": None}

    # --- Disc周囲（core/ring） ---
    disc_core_L_multi = None
    disc_ring_L_multi = None
    disc_center_dist_ratio = None
    disc_pos_ok = None
    disc_edge_covered = None
    disc_edge_coverage_ratio = None
    if disc_mask_crop is not None:
        disc_core_L_multi, disc_ring_L_multi = compute_disc_sharpness_components(cropped, disc_mask_crop)
        est = estimate_disc_center_radius(disc_mask_crop)
        if est is not None:
            dcx, dcy, _ = est
            dist = ((dcx - center_x) ** 2 + (dcy - center_y) ** 2) ** 0.5
            disc_center_dist_ratio = float(dist / max(radius, 1))
            disc_pos_ok = (0.25 <= disc_center_dist_ratio <= 0.75)
        # Disc Edge Coverage計算
        disc_edge_covered, disc_edge_coverage_ratio = compute_disc_edge_coverage(disc_mask_crop, retina_mask_crop)

    return {
        'image_path': image_path,
        'lens_detected': True,
        'lens_area': lens_area,
        'retina_area': retina_area,
        'retina_ratio': round(float(retina_ratio), 2),
        'disc_detected': bool(disc_detected),
        'macula_detected': bool(macula_detected),
        'mbss_L_multi': mb['L_multi'],
        'mbss_HF_ratio': mb['HF_ratio'],
        'mbss_Spec_centroid': mb['Spec_centroid'],
        'mbss_Grad_p90': mb['Grad_p90'],
        'S_mean': mb.get('S_mean'),
        'disc_core_L_multi': disc_core_L_multi,
        'disc_ring_L_multi': disc_ring_L_multi,
        'disc_center_dist_ratio': disc_center_dist_ratio,
        'disc_pos_ok': disc_pos_ok,
        'disc_edge_covered': disc_edge_covered,
        'disc_edge_coverage_ratio': disc_edge_coverage_ratio,
    }


def load_models(rtdetr_model_path: str, yolo_seg_model_path: str, device: str = "auto"):
    """
    モデルを読み込む（1回だけ呼び出す）

    Args:
        rtdetr_model_path: RT-DETRモデルのパス
        yolo_seg_model_path: YOLO-segモデルのパス
        device: デバイス指定（"auto", "cuda", "cpu"）

    Returns:
        tuple: (detection_model, segmentation_model)
    """
    print("モデルを読み込んでいます...")
    detection_model = RTDETR(rtdetr_model_path)
    segmentation_model = YOLO(yolo_seg_model_path)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        detection_model.to('cuda')
        segmentation_model.to('cuda')
        print("CUDAを使用します")
    else:
        print("CPUを使用します")

    print("モデル読み込み完了")
    return detection_model, segmentation_model


def assess_image_quality(
    image_paths: List[str],
    detection_model,
    segmentation_model,
    image_id: str = None,
) -> pd.DataFrame:
    """
    画像リストに対して品質評価を実行

    Args:
        image_paths: 評価する画像ファイルのパスのリスト
        detection_model: 読み込み済みのRT-DETRモデル
        segmentation_model: 読み込み済みのYOLO-segモデル
        image_id: 画像ID（動画のbasenameなど）

    Returns:
        品質指標を含むDataFrame
    """
    
    # 画像処理
    results = []
    for image_path in tqdm(image_paths, desc="画像を処理中"):
        try:
            r = process_one_image(image_path, detection_model, segmentation_model)
            if r is None:
                continue
            r['image_name'] = str(image_path.split('\\')[-1].split('/')[-1])  # ファイル名のみ
            if image_id is not None:
                r['image_id'] = image_id
            results.append(r)
        except Exception as e:
            print(f"エラー: {image_path}: {e}")
    
    if not results:
        raise RuntimeError("処理結果が0件です。image_dir/モデル/依存関係を確認してください")
    
    # DataFrame化
    df = pd.DataFrame(results)
    
    # -------------------- スコア算出（ID内でz-score） --------------------
    
    # MBSS stats（None除外）
    mb_cols = ['mbss_L_multi', 'mbss_HF_ratio', 'mbss_Spec_centroid', 'mbss_Grad_p90']
    stats = {}
    for c in mb_cols:
        vals = df[c].dropna().astype(float)
        key = c.replace('mbss_', '')
        if len(vals) > 0 and float(vals.std()) > 0:
            stats[key] = {"mean": float(vals.mean()), "std": float(vals.std())}
        elif len(vals) > 0:
            stats[key] = {"mean": float(vals.mean()), "std": 1.0}
    
    # MBSS score
    mbss_scores = []
    for _, row in df.iterrows():
        comps = {
            "L_multi": row.get('mbss_L_multi'),
            "HF_ratio": row.get('mbss_HF_ratio'),
            "Spec_centroid": row.get('mbss_Spec_centroid'),
            "Grad_p90": row.get('mbss_Grad_p90'),
        }
        if set(stats.keys()) == {"L_multi", "HF_ratio", "Spec_centroid", "Grad_p90"}:
            mbss_scores.append(compute_mbss_score(comps, stats=stats))
        else:
            mbss_scores.append(None)
    df['mbss_score'] = mbss_scores
    
    # Disc core/ring score（z-score）
    for col_l, col_s in [('disc_core_L_multi', 'disc_core_score'), ('disc_ring_L_multi', 'disc_ring_score')]:
        vals = df[col_l].dropna().astype(float)
        if len(vals) > 1 and float(vals.std()) > 0:
            m, s = float(vals.mean()), float(vals.std())
        elif len(vals) > 0:
            m, s = float(vals.mean()), 1.0
        else:
            m, s = 0.0, 1.0
        
        scores = []
        for v in df[col_l]:
            if v is None or pd.isna(v):
                scores.append(None)
            else:
                scores.append((float(v) - m) / (s + 1e-8))
        df[col_s] = scores
    
    return df


