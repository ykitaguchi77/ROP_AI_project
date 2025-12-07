"""
画像検証スクリプト
指定されたディレクトリ内の全画像に対して、RT-DETRとYOLO11-segによる推論を実行し、
結果をCSVファイルに保存します。
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from ultralytics import RTDETR, YOLO
import torch
from scipy import fftpack

# クラスIDの定義
CLASS_ID_TO_NAME = {
    0: "Fundus",  # retina
    1: "Disc",
    2: "Macula"
}

def apply_circular_mask(image, lens_bbox_xyxy):
    """
    レンズのbboxに基づいて円形マスクを作成し、マスク外を灰色(114, 114, 114)で塗りつぶす
    
    Args:
        image: 入力画像 (BGR形式)
        lens_bbox_xyxy: レンズのbbox [x1, y1, x2, y2]
    
    Returns:
        マスク適用後の画像
    """
    if image is None or lens_bbox_xyxy is None:
        return image
    
    x1, y1, x2, y2 = [int(c) for c in lens_bbox_xyxy]
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    
    # 円の直径をbboxの平均サイズから計算
    diameter = (bbox_width + bbox_height) / 2
    radius = int(diameter / 2)
    
    # マスクを作成
    processed_image = image.copy()
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (center_x, center_y), radius, 255, -1)
    
    # マスクの外側を灰色(114, 114, 114)で塗りつぶす
    processed_image[mask == 0] = (114, 114, 114)
    
    return processed_image, mask

def calculate_lens_area_from_crop(cropped_mask):
    """
    クロップされた領域でのレンズの面積を計算（円形マスクの面積）
    
    Args:
        cropped_mask: クロップ領域での円形マスク
    
    Returns:
        レンズの面積（ピクセル数）
    """
    if cropped_mask is not None:
        # マスクの面積（白い部分のピクセル数）
        return np.sum(cropped_mask > 0)
    else:
        return 0

# ==================== MBSS (Multi-domain Blind Sharpness Score) 計算関数 ====================

def to_gray_float(img):
    """BGR/RGB/グレースケールいずれも float32 [0,1] に統一"""
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    gray = gray.astype(np.float32)
    if gray.max() > 1.0:
        gray /= 255.0
    return gray

def laplacian_multi_var(gray, sigmas=(1.0, 2.0, 4.0), weights=(0.5, 0.3, 0.2)):
    """マルチスケールLaplacian分散"""
    vars_ = []
    for s, w in zip(sigmas, weights):
        ksize = int(6*s + 1)  # σに応じてカーネルサイズ
        if ksize % 2 == 0:
            ksize += 1
        blur = cv2.GaussianBlur(gray, (ksize, ksize), s)
        lap = cv2.Laplacian(blur, cv2.CV_32F, ksize=3)
        vars_.append(w * lap.var())
    return float(np.sum(vars_))

def fft_features(gray, high_freq_thresh=0.3):
    """FFT高周波エネルギー比とスペクトル重心を計算"""
    h, w = gray.shape
    # ウィンドウ（Hanning）で端の影響を軽減
    wy = np.hanning(h)
    wx = np.hanning(w)
    window = np.outer(wy, wx).astype(np.float32)
    g = gray * window

    # 2D FFT
    F = fftpack.fftshift(fftpack.fft2(g))
    mag2 = np.abs(F)**2

    # 正規化半径
    cy, cx = h // 2, w // 2
    yy, xx = np.indices((h, w))
    ry = (yy - cy) / float(max(cy, 1))
    rx = (xx - cx) / float(max(cx, 1))
    r = np.sqrt(rx**2 + ry**2)
    r_norm = np.clip(r, 0, 1)

    total_energy = mag2.sum() + 1e-8
    high_mask = r_norm > high_freq_thresh
    HF_ratio = float(mag2[high_mask].sum() / total_energy)

    # スペクトル重心
    Spec_centroid = float((r_norm * mag2).sum() / total_energy)

    return HF_ratio, Spec_centroid

def grad_percentile(gray, p=90):
    """勾配の上位分位点を計算"""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    return float(np.percentile(mag, p))

def compute_mbss_components(img, mask=None):
    """
    MBSSの各コンポーネントを計算
    
    Args:
        img: OpenCVで読み込んだBGR画像 or グレースケール
        mask: マスク（オプション、指定された場合はマスク内の領域のみで計算）
    
    Returns:
        各特徴量の辞書
    """
    gray = to_gray_float(img)
    
    # マスクが指定されている場合は、マスク内の領域のみを使用
    if mask is not None:
        # マスクをグレースケール画像と同じサイズにリサイズ（必要に応じて）
        if mask.shape != gray.shape:
            mask_resized = cv2.resize(mask.astype(np.uint8), (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask_bool = mask_resized > 0.5
        else:
            mask_bool = mask > 0.5
        
        # マスク内の領域のみを使用（マスク外は0で埋める）
        gray_masked = gray.copy()
        gray_masked[~mask_bool] = 0.0
    else:
        gray_masked = gray
        mask_bool = np.ones_like(gray, dtype=bool)
    
    # マスク内の領域が十分にある場合のみ計算
    if np.sum(mask_bool) < 100:  # 最小ピクセル数の閾値
        return {
            "L_multi": 0.0,
            "HF_ratio": 0.0,
            "Spec_centroid": 0.0,
            "Grad_p90": 0.0,
        }
    
    L_multi = laplacian_multi_var(gray_masked)
    HF_ratio, Spec_centroid = fft_features(gray_masked)
    Grad_p90 = grad_percentile(gray_masked)
    
    return {
        "L_multi": L_multi,
        "HF_ratio": HF_ratio,
        "Spec_centroid": Spec_centroid,
        "Grad_p90": Grad_p90,
    }

def compute_mbss_score(components, stats=None, weights=None):
    """
    MBSSスコアを計算（z-score正規化後、重み付き和）
    
    Args:
        components: compute_mbss_components の戻り値
        stats: {name: {"mean": m, "std": s}} 形式の辞書
        weights: 各成分の重み（指定がなければデフォルト）
    
    Returns:
        MBSSスコア
    """
    if weights is None:
        weights = {
            "L_multi": 0.35,
            "HF_ratio": 0.25,
            "Spec_centroid": 0.20,
            "Grad_p90": 0.20,
        }
    
    score = 0.0
    for k, w in weights.items():
        x = components[k]
        if stats is not None and k in stats:
            m = stats[k]["mean"]
            s = stats[k]["std"] + 1e-8
            z = (x - m) / s
        else:
            z = x
        score += w * z
    return float(score)

def glare_index(gray, thresh=0.98):
    """
    飽和ハイライト（ギラツキ）の指標を計算
    
    Args:
        gray: float32 [0,1] グレースケール画像
        thresh: 飽和とみなす閾値（デフォルト0.98）
    
    Returns:
        飽和ピクセルの割合（0-1）
    """
    return float((gray > thresh).mean())

# ==================== Disc周囲評価関数 ====================

def estimate_disc_center_radius(disc_mask):
    """
    discマスクから中心と代表半径を推定
    
    Args:
        disc_mask: 0/1 または 0/255 の2値画像 (uint8 or bool)
    
    Returns:
        (cx, cy, R) または None（マスクが無効な場合）
    """
    # 0/255 → 0/1 に正規化
    mask = disc_mask.astype(np.uint8)
    if mask.max() > 1:
        mask = (mask > 0).astype(np.uint8)
    
    # 一番大きい連結成分だけ残す（変なゴミ対策）
    num_labels, labels = cv2.connectedComponents(mask)
    if num_labels > 1:
        # ラベル毎の面積を計算（0は背景）
        areas = [(labels == i).sum() for i in range(1, num_labels)]
        main_label = np.argmax(areas) + 1
        mask = (labels == main_label).astype(np.uint8)
    
    # モーメントから中心
    M = cv2.moments(mask)
    if M["m00"] == 0:
        return None  # マスクが壊れてる場合の保険
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    
    # 面積から「代表半径」を推定（円と仮定）
    area = mask.sum()
    R = np.sqrt(area / np.pi)
    
    return (cx, cy, R)

def make_disc_rois(shape, cx, cy, R, inner_ratio=0.6, outer_ratio=1.2):
    """
    discコアとperipapillary ringのROIを作成
    
    Args:
        shape: (H, W)
        cx, cy, R: estimate_disc_center_radius の出力
        inner_ratio: discの"コア"とみなす割合
        outer_ratio: peripapillary ring の外側の半径比
    
    Returns:
        disc_core_mask, ring_mask (ともに 0/1 np.uint8)
    """
    h, w = shape
    yy, xx = np.indices((h, w))
    # 中心からのユークリッド距離
    dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    
    core = dist < (inner_ratio * R)
    ring = (dist >= (inner_ratio * R)) & (dist < (outer_ratio * R))
    
    return core.astype(np.uint8), ring.astype(np.uint8)

def laplacian_multi_var_masked(gray, mask, sigmas=(1.0, 2.0, 4.0), weights=(0.5, 0.3, 0.2)):
    """
    マスク付きマルチスケールLaplacian分散
    
    Args:
        gray: float32 [0,1]
        mask: 0/1 np.uint8（ROI以外は0）
    """
    mask_bool = mask.astype(bool)
    vars_ = []
    for s, w in zip(sigmas, weights):
        ksize = int(6*s + 1)
        if ksize % 2 == 0:
            ksize += 1
        blur = cv2.GaussianBlur(gray, (ksize, ksize), s)
        lap = cv2.Laplacian(blur, cv2.CV_32F, ksize=3)
        vals = lap[mask_bool]
        if vals.size == 0:
            continue
        vars_.append(w * vals.var())
    if not vars_:
        return 0.0
    return float(np.sum(vars_))

def grad_percentile_masked(gray, mask, p=90):
    """
    マスク付き勾配の上位分位点
    
    Args:
        gray: float32 [0,1]
        mask: 0/1 np.uint8（ROI以外は0）
        p: パーセンタイル
    """
    mask_bool = mask.astype(bool)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    vals = mag[mask_bool]
    if vals.size == 0:
        return 0.0
    return float(np.percentile(vals, p))

def fft_features_roi(gray, mask, high_freq_thresh=0.3):
    """
    ROI内でのFFT高周波エネルギー比とスペクトル重心を計算
    
    Args:
        gray: float32 [0,1], full image
        mask: 0/1 np.uint8, ROI
    Returns:
        HF_ratio, Spec_centroid
    """
    # ROI のバウンディングボックスを取得
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return 0.0, 0.0
    
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    
    patch = gray[y_min:y_max+1, x_min:x_max+1]
    m_patch = mask[y_min:y_max+1, x_min:x_max+1].astype(np.float32)
    
    h, w = patch.shape
    # Hanning 窓 × ROI マスク で外側を弱める
    wy = np.hanning(h)
    wx = np.hanning(w)
    window = np.outer(wy, wx).astype(np.float32)
    win = window * m_patch  # ROI外はほぼ0になる
    g = patch * win
    
    # FFT
    F = fftpack.fftshift(fftpack.fft2(g))
    mag2 = np.abs(F)**2
    
    cy, cx = h // 2, w // 2
    yy, xx = np.indices((h, w))
    ry = (yy - cy) / float(max(cy, 1))
    rx = (xx - cx) / float(max(cx, 1))
    r = np.sqrt(rx**2 + ry**2)
    r_norm = np.clip(r, 0, 1)
    
    total_energy = mag2.sum() + 1e-8
    high_mask = r_norm > high_freq_thresh
    HF_ratio = float(mag2[high_mask].sum() / total_energy)
    Spec_centroid = float((r_norm * mag2).sum() / total_energy)
    
    return HF_ratio, Spec_centroid

def compute_disc_sharpness_components(gray, disc_mask):
    """
    disc周囲（コアとリング）のL_multiのみを計算
    
    Args:
        gray: float32 [0,1] グレースケール画像
        disc_mask: 0/1 np.uint8 discマスク
    
    Returns:
        (core_L_multi, ring_L_multi) または (None, None)
    """
    est = estimate_disc_center_radius(disc_mask)
    if est is None:
        return None, None
    
    cx, cy, R = est
    core_mask, ring_mask = make_disc_rois(gray.shape, cx, cy, R)
    
    # マスク内の領域が十分にあるか確認
    if np.sum(core_mask) < 50 or np.sum(ring_mask) < 50:
        return None, None
    
    # --- core ---
    L_core = laplacian_multi_var_masked(gray, core_mask)
    
    # --- ring ---
    L_ring = laplacian_multi_var_masked(gray, ring_mask)
    
    return L_core, L_ring

# ==================== 既存の関数 ====================

def calculate_mask_area(mask_data, crop_info, circular_mask=None):
    """
    セグメンテーションマスクの面積を計算（クロップ領域での面積、円形マスク内のみ）
    
    Args:
        mask_data: マスクデータ（低解像度、YOLOの出力サイズ）
        crop_info: クロップ情報 {'orig_crop_shape': (width, height), 'resized_crop_shape': (width, height)}
        circular_mask: 円形マスク（オプション、指定された場合はこのマスク内の面積のみを計算）
    
    Returns:
        マスクの面積（ピクセル数、クロップ領域での面積、円形マスク内のみ）
    """
    orig_crop_w, orig_crop_h = crop_info['orig_crop_shape']
    
    # マスクを元のクロップサイズにリサイズ
    mask_resized = cv2.resize(mask_data, (orig_crop_w, orig_crop_h))
    
    # 円形マスクが指定されている場合は、そのマスク内の面積のみを計算
    if circular_mask is not None:
        # 円形マスク内のretinaマスクの面積
        mask_in_circle = (mask_resized > 0.5) & (circular_mask > 0)
        return np.sum(mask_in_circle)
    else:
        return np.sum(mask_resized > 0.5)

def process_image(image_path, detection_model, segmentation_model):
    """
    1枚の画像を処理して、検出結果と面積情報を返す
    
    Args:
        image_path: 画像ファイルのパス
        detection_model: RT-DETRモデル
        segmentation_model: YOLO11-segモデル
    
    Returns:
        dict: 検出結果と面積情報
    """
    # 画像を読み込み
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    original_shape = image.shape[:2]  # (height, width)
    
    # Stage 1: RT-DETRでレンズのbboxを検出
    det_results = detection_model(image, verbose=False)
    lens_bbox_xyxy = None
    
    for r in det_results:
        if r.boxes is not None and len(r.boxes) > 0:
            for box in r.boxes:
                if int(box.cls) == 0:  # レンズのクラスIDは0
                    lens_bbox_xyxy = box.xyxy[0].cpu().numpy()
                    break
            if lens_bbox_xyxy is not None:
                break
    
    if lens_bbox_xyxy is None:
        # レンズが検出されなかった場合
        # MBSSコンポーネントは計算しない（retinaマスクがないため）
        return {
            'image_path': image_path,
            'lens_detected': False,
            'lens_area': 0,
            'retina_area': 0,
            'retina_ratio': 0.0,
            'disc_detected': False,
            'macula_detected': False,
            # MBSSコンポーネント（検出されなかった場合はNone）
            'mbss_L_multi': None,
            'mbss_HF_ratio': None,
            'mbss_Spec_centroid': None,
            'mbss_Grad_p90': None,
            # Disc周囲コンポーネント（検出されなかった場合はNone）
            'disc_core_L_multi': None,
            'disc_ring_L_multi': None,
        }
    
    # レンズ領域をクロップ
    x1, y1, x2, y2 = [int(c) for c in lens_bbox_xyxy]
    cropped_image = image[y1:y2, x1:x2]
    
    if cropped_image.size == 0:
        # MBSSコンポーネントは計算しない（retinaマスクがないため）
        return {
            'image_path': image_path,
            'lens_detected': True,
            'lens_area': 0,
            'retina_area': 0,
            'retina_ratio': 0.0,
            'disc_detected': False,
            'macula_detected': False,
            # MBSSコンポーネント（検出されなかった場合はNone）
            'mbss_L_multi': None,
            'mbss_HF_ratio': None,
            'mbss_Spec_centroid': None,
            'mbss_Grad_p90': None,
            # Disc周囲コンポーネント（検出されなかった場合はNone）
            'disc_core_L_multi': None,
            'disc_ring_L_multi': None,
        }
    
    # クロップ画像に円形マスクを適用（レンズ外を灰色に塗りつぶし）
    # クロップ領域内での円形マスクを作成
    orig_crop_h, orig_crop_w = cropped_image.shape[:2]
    center_x = orig_crop_w // 2
    center_y = orig_crop_h // 2
    diameter = (orig_crop_w + orig_crop_h) / 2
    radius = int(diameter / 2)
    
    cropped_mask = np.zeros((orig_crop_h, orig_crop_w), dtype=np.uint8)
    cv2.circle(cropped_mask, (center_x, center_y), radius, 255, -1)
    
    masked_cropped_image = cropped_image.copy()
    masked_cropped_image[cropped_mask == 0] = (114, 114, 114)
    
    # レンズの面積を計算（クロップ領域での円形マスクの面積）
    lens_area = calculate_lens_area_from_crop(cropped_mask)
    
    # クロップ画像をYOLOの入力サイズにリサイズ（アスペクト比を保持）
    YOLO_INPUT_WIDTH = 640
    aspect_ratio = orig_crop_h / orig_crop_w
    yolo_input_h = int(YOLO_INPUT_WIDTH * aspect_ratio)
    yolo_input_image = cv2.resize(masked_cropped_image, (YOLO_INPUT_WIDTH, yolo_input_h), interpolation=cv2.INTER_AREA)
    
    # Stage 2: YOLO11-segでretina, disc, maculaを検出
    seg_results = segmentation_model(yolo_input_image, verbose=False, retina_masks=True)
    
    # 結果を解析
    retina_area = 0
    disc_detected = False
    macula_detected = False
    
    crop_info = {
        'offset_xy': (x1, y1),
        'orig_crop_shape': (orig_crop_w, orig_crop_h),
        'resized_crop_shape': (YOLO_INPUT_WIDTH, yolo_input_h)
    }
    
    if seg_results and seg_results[0].masks is not None:
        result = seg_results[0]
        masks = result.masks.data.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        
        for i, (mask_data, class_id) in enumerate(zip(masks, classes)):
            class_id = int(class_id)
            
            if class_id == 0:  # Fundus (retina)
                # 円形マスク内のretinaの面積のみを計算
                retina_area = calculate_mask_area(mask_data, crop_info, circular_mask=cropped_mask)
            elif class_id == 1:  # Disc
                disc_detected = True
            elif class_id == 2:  # Macula
                macula_detected = True
    
    # 面積比率を計算
    retina_ratio = (retina_area / lens_area * 100) if lens_area > 0 else 0.0
    
    # MBSSコンポーネントを計算（retinaマスク内のみ、検出された場合のみ）
    # retinaマスクを元画像サイズにマッピング
    retina_mask_full = None
    if seg_results and seg_results[0].masks is not None:
        result = seg_results[0]
        masks = result.masks.data.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        
        # retina (class_id=0) のマスクを探す
        for i, (mask_data, class_id) in enumerate(zip(masks, classes)):
            if int(class_id) == 0:  # Fundus (retina)
                # マスクを元のクロップサイズにリサイズ
                mask_resized = cv2.resize(mask_data, (orig_crop_w, orig_crop_h))
                # 円形マスク内のretinaマスクのみ
                retina_mask_crop = (mask_resized > 0.5) & (cropped_mask > 0)
                
                # 元画像全体のサイズにマッピング
                retina_mask_full = np.zeros((original_shape[0], original_shape[1]), dtype=np.uint8)
                retina_mask_full[y1:y2, x1:x2] = retina_mask_crop.astype(np.uint8) * 255
                break
    
    # retinaマスクが検出された場合のみMBSSを計算
    if retina_mask_full is not None:
        mbss_components = compute_mbss_components(image, mask=retina_mask_full)
    else:
        # retinaマスクが検出されなかった場合はNone
        mbss_components = {
            "L_multi": None,
            "HF_ratio": None,
            "Spec_centroid": None,
            "Grad_p90": None,
        }
    
    # Disc周囲の評価を計算（discが検出された場合のみ、L_multiのみ）
    disc_core_L_multi = None
    disc_ring_L_multi = None
    
    if disc_detected and seg_results and seg_results[0].masks is not None:
        result = seg_results[0]
        masks = result.masks.data.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        
        # disc (class_id=1) のマスクを探す
        disc_mask_full = None
        for i, (mask_data, class_id) in enumerate(zip(masks, classes)):
            if int(class_id) == 1:  # Disc
                # マスクを元のクロップサイズにリサイズ
                mask_resized = cv2.resize(mask_data, (orig_crop_w, orig_crop_h))
                # 円形マスク内のdiscマスクのみ
                disc_mask_crop = (mask_resized > 0.5) & (cropped_mask > 0)
                
                # 元画像全体のサイズにマッピング
                disc_mask_full = np.zeros((original_shape[0], original_shape[1]), dtype=np.uint8)
                disc_mask_full[y1:y2, x1:x2] = disc_mask_crop.astype(np.uint8) * 255
                break
        
        # discマスクが存在する場合、disc周囲の評価を計算（L_multiのみ）
        if disc_mask_full is not None:
            gray_image = to_gray_float(image)
            disc_core_L_multi, disc_ring_L_multi = compute_disc_sharpness_components(gray_image, disc_mask_full)
    
    # 結果を構築
    result_dict = {
        'image_path': image_path,
        'lens_detected': True,
        'lens_area': int(lens_area),
        'retina_area': int(retina_area),
        'retina_ratio': round(retina_ratio, 2),
        'disc_detected': disc_detected,
        'macula_detected': macula_detected,
        # MBSSコンポーネント（後でz-score正規化してスコアを計算）
        'mbss_L_multi': mbss_components['L_multi'],
        'mbss_HF_ratio': mbss_components['HF_ratio'],
        'mbss_Spec_centroid': mbss_components['Spec_centroid'],
        'mbss_Grad_p90': mbss_components['Grad_p90'],
        # Disc周囲のL_multiのみ
        'disc_core_L_multi': disc_core_L_multi,
        'disc_ring_L_multi': disc_ring_L_multi,
    }
    
    return result_dict

def process_directory(image_dir, detection_model_path, segmentation_model_path, output_csv_path):
    """
    指定されたディレクトリ内の全画像を処理してCSVに保存
    
    Args:
        image_dir: 画像ディレクトリのパス
        detection_model_path: RT-DETRモデルのパス
        segmentation_model_path: YOLO11-segモデルのパス
        output_csv_path: 出力CSVファイルのパス
    """
    # モデルを読み込み
    print("モデルを読み込んでいます...")
    detection_model = RTDETR(detection_model_path)
    segmentation_model = YOLO(segmentation_model_path)
    
    if torch.cuda.is_available():
        detection_model.to('cuda')
        segmentation_model.to('cuda')
        print("CUDAを使用します")
    else:
        print("CPUを使用します")
    
    # 画像ファイルのリストを取得
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    image_files = sorted([
        f for f in os.listdir(image_dir) 
        if f.lower().endswith(image_extensions)
    ])
    
    if not image_files:
        print(f"エラー: {image_dir} に画像ファイルが見つかりません")
        return
    
    print(f"{len(image_files)} 枚の画像を処理します...")
    
    # 各画像を処理
    results = []
    for image_file in tqdm(image_files, desc="画像を処理中"):
        image_path = os.path.join(image_dir, image_file)
        try:
            result = process_image(image_path, detection_model, segmentation_model)
            if result:
                # ファイル名のみを保存
                result['image_name'] = image_file
                results.append(result)
        except Exception as e:
            print(f"\nエラー: {image_file} の処理中にエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # CSVに保存（MBSSスコアを計算）
    if results:
        df = pd.DataFrame(results)
        
        # MBSS統計情報を計算（z-score正規化用、Noneを除外）
        mbss_names = ["mbss_L_multi", "mbss_HF_ratio", "mbss_Spec_centroid", "mbss_Grad_p90"]
        stats = {}
        for name in mbss_names:
            if name in df.columns:
                # Noneを除外して計算
                vals = df[name].dropna().astype(float)
                if len(vals) > 0:
                    stats[name.replace('mbss_', '')] = {"mean": float(vals.mean()), "std": float(vals.std())}
        
        # 各画像のMBSSスコアを計算（MBSSコンポーネントが全て存在する場合のみ）
        mbss_scores = []
        for _, row in df.iterrows():
            l_multi = row.get('mbss_L_multi')
            hf_ratio = row.get('mbss_HF_ratio')
            spec_centroid = row.get('mbss_Spec_centroid')
            grad_p90 = row.get('mbss_Grad_p90')
            
            # 全てのコンポーネントがNoneでない場合のみ計算
            if l_multi is not None and hf_ratio is not None and spec_centroid is not None and grad_p90 is not None:
                components = {
                    "L_multi": l_multi,
                    "HF_ratio": hf_ratio,
                    "Spec_centroid": spec_centroid,
                    "Grad_p90": grad_p90,
                }
                score = compute_mbss_score(components, stats=stats)
                mbss_scores.append(score)
            else:
                mbss_scores.append(None)
        
        df['mbss_score'] = mbss_scores
        
        # Disc周囲（コア）統計情報を計算（L_multiのみ）
        disc_core_L_multi_vals = df['disc_core_L_multi'].dropna().astype(float) if 'disc_core_L_multi' in df.columns else pd.Series(dtype=float)
        disc_core_mean = float(disc_core_L_multi_vals.mean()) if len(disc_core_L_multi_vals) > 0 else 0.0
        disc_core_std = float(disc_core_L_multi_vals.std()) if len(disc_core_L_multi_vals) > 0 else 1.0
        
        # Disc周囲（リング）統計情報を計算（L_multiのみ）
        disc_ring_L_multi_vals = df['disc_ring_L_multi'].dropna().astype(float) if 'disc_ring_L_multi' in df.columns else pd.Series(dtype=float)
        disc_ring_mean = float(disc_ring_L_multi_vals.mean()) if len(disc_ring_L_multi_vals) > 0 else 0.0
        disc_ring_std = float(disc_ring_L_multi_vals.std()) if len(disc_ring_L_multi_vals) > 0 else 1.0
        
        # Disc周囲（コア）スコアを計算（L_multiのみ、z-score正規化）
        disc_core_scores = []
        for _, row in df.iterrows():
            l_multi = row.get('disc_core_L_multi')
            if l_multi is not None and not pd.isna(l_multi):
                # z-score正規化
                z_score = (float(l_multi) - disc_core_mean) / (disc_core_std + 1e-8)
                disc_core_scores.append(z_score)
            else:
                disc_core_scores.append(None)
        
        df['disc_core_score'] = disc_core_scores
        
        # Disc周囲（リング）スコアを計算（L_multiのみ、z-score正規化）
        disc_ring_scores = []
        for _, row in df.iterrows():
            l_multi = row.get('disc_ring_L_multi')
            if l_multi is not None and not pd.isna(l_multi):
                # z-score正規化
                z_score = (float(l_multi) - disc_ring_mean) / (disc_ring_std + 1e-8)
                disc_ring_scores.append(z_score)
            else:
                disc_ring_scores.append(None)
        
        df['disc_ring_score'] = disc_ring_scores
        
        # カラムの順序を整理
        columns_order = [
            'image_name',
            'image_path',
            'lens_detected',
            'lens_area',
            'retina_area',
            'retina_ratio',
            'disc_detected',
            'macula_detected',
            'mbss_L_multi',
            'mbss_HF_ratio',
            'mbss_Spec_centroid',
            'mbss_Grad_p90',
            'mbss_score',
            'disc_core_L_multi',
            'disc_core_score',
            'disc_ring_L_multi',
            'disc_ring_score'
        ]
        # 存在するカラムのみを選択
        columns_order = [col for col in columns_order if col in df.columns]
        df = df[columns_order]
        df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        print(f"\n結果をCSVファイルに保存しました: {output_csv_path}")
        print(f"処理した画像数: {len(results)}")
        if 'mbss_score' in df.columns:
            mbss_valid = df['mbss_score'].dropna()
            if len(mbss_valid) > 0:
                print(f"MBSSスコア範囲: {mbss_valid.min():.3f} ~ {mbss_valid.max():.3f}")
                print(f"MBSSスコア平均: {mbss_valid.mean():.3f}")
                print(f"MBSS計算可能な画像数: {len(mbss_valid)} / {len(df)}")
            else:
                print("MBSSスコアが計算可能な画像がありませんでした")
        
        if 'disc_core_score' in df.columns:
            disc_core_valid = df['disc_core_score'].dropna()
            if len(disc_core_valid) > 0:
                print(f"\nDisc Coreスコア範囲: {disc_core_valid.min():.3f} ~ {disc_core_valid.max():.3f}")
                print(f"Disc Coreスコア平均: {disc_core_valid.mean():.3f}")
                print(f"Disc Core計算可能な画像数: {len(disc_core_valid)} / {len(df)}")
            else:
                print("\nDisc Coreスコアが計算可能な画像がありませんでした")
        
        if 'disc_ring_score' in df.columns:
            disc_ring_valid = df['disc_ring_score'].dropna()
            if len(disc_ring_valid) > 0:
                print(f"\nDisc Ringスコア範囲: {disc_ring_valid.min():.3f} ~ {disc_ring_valid.max():.3f}")
                print(f"Disc Ringスコア平均: {disc_ring_valid.mean():.3f}")
                print(f"Disc Ring計算可能な画像数: {len(disc_ring_valid)} / {len(df)}")
            else:
                print("\nDisc Ringスコアが計算可能な画像がありませんでした")
    else:
        print("エラー: 処理結果がありません")

def process_all_image_ids(base_dir, detection_model_path, segmentation_model_path):
    """
    bestimage_validation内の全imageIDを処理
    
    Args:
        base_dir: bestimage_validationディレクトリのパス
        detection_model_path: RT-DETRモデルのパス
        segmentation_model_path: YOLO11-segモデルのパス
    """
    # bestimage_validation内の全imageIDを取得
    image_ids = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            # 「画像」フォルダが存在するか確認
            image_folder = os.path.join(item_path, "画像")
            if os.path.exists(image_folder):
                image_ids.append(item)
    
    image_ids = sorted(image_ids)
    print(f"処理対象のimageID: {image_ids}")
    print(f"合計 {len(image_ids)} 件のimageIDを処理します")
    
    # モデルを読み込み（全imageID処理時は1回だけ読み込む）
    print("\nモデルを読み込んでいます...")
    detection_model = RTDETR(detection_model_path)
    segmentation_model = YOLO(segmentation_model_path)
    
    if torch.cuda.is_available():
        detection_model.to('cuda')
        segmentation_model.to('cuda')
        print("CUDAを使用します")
    else:
        print("CPUを使用します")
    
    # 全imageIDを処理
    all_results = []
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    
    for image_id in tqdm(image_ids, desc="imageIDを処理中"):
        image_dir = os.path.join(base_dir, image_id, "画像")
        
        if not os.path.exists(image_dir):
            print(f"警告: {image_dir} が存在しません。スキップします。")
            continue
        
        # 画像ファイルのリストを取得
        image_files = sorted([
            f for f in os.listdir(image_dir) 
            if f.lower().endswith(image_extensions)
        ])
        
        if not image_files:
            print(f"警告: {image_dir} に画像ファイルが見つかりません。スキップします。")
            continue
        
        print(f"\n{image_id}: {len(image_files)} 枚の画像を処理中...")
        
        # 各画像を処理
        for image_file in tqdm(image_files, desc=f"{image_id}の画像を処理中", leave=False):
            image_path = os.path.join(image_dir, image_file)
            try:
                result = process_image(image_path, detection_model, segmentation_model)
                if result:
                    result['image_name'] = image_file
                    result['image_id'] = image_id
                    all_results.append(result)
            except Exception as e:
                print(f"\nエラー: {image_file} の処理中にエラーが発生しました: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # 結果をCSVに保存（MBSSスコアを計算）
    if all_results:
        df = pd.DataFrame(all_results)
        
        # MBSS統計情報を計算（z-score正規化用、Noneを除外）
        mbss_names = ["mbss_L_multi", "mbss_HF_ratio", "mbss_Spec_centroid", "mbss_Grad_p90"]
        stats = {}
        for name in mbss_names:
            if name in df.columns:
                # Noneを除外して計算
                vals = df[name].dropna().astype(float)
                if len(vals) > 0:
                    stats[name.replace('mbss_', '')] = {"mean": float(vals.mean()), "std": float(vals.std())}
        
        # 各画像のMBSSスコアを計算（MBSSコンポーネントが全て存在する場合のみ）
        mbss_scores = []
        for _, row in df.iterrows():
            l_multi = row.get('mbss_L_multi')
            hf_ratio = row.get('mbss_HF_ratio')
            spec_centroid = row.get('mbss_Spec_centroid')
            grad_p90 = row.get('mbss_Grad_p90')
            
            # 全てのコンポーネントがNoneでない場合のみ計算
            if l_multi is not None and hf_ratio is not None and spec_centroid is not None and grad_p90 is not None:
                components = {
                    "L_multi": l_multi,
                    "HF_ratio": hf_ratio,
                    "Spec_centroid": spec_centroid,
                    "Grad_p90": grad_p90,
                }
                score = compute_mbss_score(components, stats=stats)
                mbss_scores.append(score)
            else:
                mbss_scores.append(None)
        
        df['mbss_score'] = mbss_scores
        
        # Disc周囲（コア）統計情報を計算（L_multiのみ）
        disc_core_L_multi_vals = df['disc_core_L_multi'].dropna().astype(float) if 'disc_core_L_multi' in df.columns else pd.Series(dtype=float)
        disc_core_mean = float(disc_core_L_multi_vals.mean()) if len(disc_core_L_multi_vals) > 0 else 0.0
        disc_core_std = float(disc_core_L_multi_vals.std()) if len(disc_core_L_multi_vals) > 0 else 1.0
        
        # Disc周囲（リング）統計情報を計算（L_multiのみ）
        disc_ring_L_multi_vals = df['disc_ring_L_multi'].dropna().astype(float) if 'disc_ring_L_multi' in df.columns else pd.Series(dtype=float)
        disc_ring_mean = float(disc_ring_L_multi_vals.mean()) if len(disc_ring_L_multi_vals) > 0 else 0.0
        disc_ring_std = float(disc_ring_L_multi_vals.std()) if len(disc_ring_L_multi_vals) > 0 else 1.0
        
        # Disc周囲（コア）スコアを計算（L_multiのみ、z-score正規化）
        disc_core_scores = []
        for _, row in df.iterrows():
            l_multi = row.get('disc_core_L_multi')
            if l_multi is not None and not pd.isna(l_multi):
                # z-score正規化
                z_score = (float(l_multi) - disc_core_mean) / (disc_core_std + 1e-8)
                disc_core_scores.append(z_score)
            else:
                disc_core_scores.append(None)
        
        df['disc_core_score'] = disc_core_scores
        
        # Disc周囲（リング）スコアを計算（L_multiのみ、z-score正規化）
        disc_ring_scores = []
        for _, row in df.iterrows():
            l_multi = row.get('disc_ring_L_multi')
            if l_multi is not None and not pd.isna(l_multi):
                # z-score正規化
                z_score = (float(l_multi) - disc_ring_mean) / (disc_ring_std + 1e-8)
                disc_ring_scores.append(z_score)
            else:
                disc_ring_scores.append(None)
        
        df['disc_ring_score'] = disc_ring_scores
        
        # カラムの順序を整理
        columns_order = [
            'image_id',
            'image_name',
            'image_path',
            'lens_detected',
            'lens_area',
            'retina_area',
            'retina_ratio',
            'disc_detected',
            'macula_detected',
            'mbss_L_multi',
            'mbss_HF_ratio',
            'mbss_Spec_centroid',
            'mbss_Grad_p90',
            'mbss_score',
            'disc_core_L_multi',
            'disc_core_score',
            'disc_ring_L_multi',
            'disc_ring_score'
        ]
        # 存在するカラムのみを選択
        columns_order = [col for col in columns_order if col in df.columns]
        df = df[columns_order]
        
        # 出力CSVファイルのパス
        output_csv_path = os.path.join(base_dir, "validation_results_all.csv")
        df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        print(f"\n結果をCSVファイルに保存しました: {output_csv_path}")
        print(f"処理した画像数: {len(all_results)}")
        
        # 統計情報を表示
        print("\n=== 統計情報 ===")
        print(f"レンズ検出率: {df['lens_detected'].sum() / len(df) * 100:.2f}%")
        print(f"Disc検出率: {df['disc_detected'].sum() / len(df) * 100:.2f}%")
        print(f"Macula検出率: {df['macula_detected'].sum() / len(df) * 100:.2f}%")
        if df[df['lens_detected']]['retina_ratio'].count() > 0:
            print(f"平均Retina面積比率: {df[df['lens_detected']]['retina_ratio'].mean():.2f}%")
        if 'mbss_score' in df.columns:
            mbss_valid = df['mbss_score'].dropna()
            if len(mbss_valid) > 0:
                print(f"\n=== MBSS統計 ===")
                print(f"MBSSスコア範囲: {mbss_valid.min():.3f} ~ {mbss_valid.max():.3f}")
                print(f"MBSSスコア平均: {mbss_valid.mean():.3f}")
                print(f"MBSSスコア標準偏差: {mbss_valid.std():.3f}")
                print(f"MBSS計算可能な画像数: {len(mbss_valid)} / {len(df)}")
                # 下位20%のしきい値を表示
                threshold_20 = mbss_valid.quantile(0.2)
                print(f"下位20%しきい値: {threshold_20:.3f} (これ以下をピンボケと判定可能)")
            else:
                print("\n=== MBSS統計 ===")
                print("MBSSスコアが計算可能な画像がありませんでした")
        
        if 'disc_core_score' in df.columns:
            disc_core_valid = df['disc_core_score'].dropna()
            if len(disc_core_valid) > 0:
                print(f"\n=== Disc Core統計 ===")
                print(f"Disc Coreスコア範囲: {disc_core_valid.min():.3f} ~ {disc_core_valid.max():.3f}")
                print(f"Disc Coreスコア平均: {disc_core_valid.mean():.3f}")
                print(f"Disc Coreスコア標準偏差: {disc_core_valid.std():.3f}")
                print(f"Disc Core計算可能な画像数: {len(disc_core_valid)} / {len(df)}")
                threshold_20 = disc_core_valid.quantile(0.2)
                print(f"下位20%しきい値: {threshold_20:.3f}")
            else:
                print("\n=== Disc Core統計 ===")
                print("Disc Coreスコアが計算可能な画像がありませんでした")
        
        if 'disc_ring_score' in df.columns:
            disc_ring_valid = df['disc_ring_score'].dropna()
            if len(disc_ring_valid) > 0:
                print(f"\n=== Disc Ring統計 ===")
                print(f"Disc Ringスコア範囲: {disc_ring_valid.min():.3f} ~ {disc_ring_valid.max():.3f}")
                print(f"Disc Ringスコア平均: {disc_ring_valid.mean():.3f}")
                print(f"Disc Ringスコア標準偏差: {disc_ring_valid.std():.3f}")
                print(f"Disc Ring計算可能な画像数: {len(disc_ring_valid)} / {len(df)}")
                threshold_20 = disc_ring_valid.quantile(0.2)
                print(f"下位20%しきい値: {threshold_20:.3f}")
            else:
                print("\n=== Disc Ring統計 ===")
                print("Disc Ringスコアが計算可能な画像がありませんでした")
    else:
        print("エラー: 処理結果がありません")

if __name__ == "__main__":
    import sys
    
    # パス設定
    base_dir = r"C:\Users\ykita\ROP_AI_project\ROP_project\bestimage_validation"
    detection_model_path = r"C:\Users\ykita\ROP_AI_project\ROP_project\models\rtdetr-l-1697_1703.pt"
    segmentation_model_path = r"C:\Users\ykita\ROP_AI_project\ROP_project\models\yolo11n-seg_19movies.pt"
    
    # コマンドライン引数でモードを選択
    # 引数なしまたは "all" の場合: 全imageIDを処理
    # 引数にimageIDを指定した場合: そのimageIDのみを処理
    if len(sys.argv) > 1 and sys.argv[1].lower() != "all":
        # モード1: 単一のimageIDを処理
        image_id = sys.argv[1]
        image_dir = os.path.join(base_dir, image_id, "画像")
        output_csv_path = os.path.join(base_dir, f"validation_results_{image_id}.csv")
        
        print(f"モード1: 単一のimageID ({image_id}) を処理します")
        process_directory(image_dir, detection_model_path, segmentation_model_path, output_csv_path)
    else:
        # モード2: 全imageIDを処理
        print("モード2: bestimage_validation内の全imageIDを処理します")
        process_all_image_ids(base_dir, detection_model_path, segmentation_model_path)

