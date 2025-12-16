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
        return {
            'image_path': image_path,
            'lens_detected': False,
            'lens_area': 0,
            'retina_area': 0,
            'retina_ratio': 0.0,
            'disc_detected': False,
            'macula_detected': False
        }
    
    # レンズ領域をクロップ
    x1, y1, x2, y2 = [int(c) for c in lens_bbox_xyxy]
    cropped_image = image[y1:y2, x1:x2]
    
    if cropped_image.size == 0:
        return {
            'image_path': image_path,
            'lens_detected': True,
            'lens_area': 0,
            'retina_area': 0,
            'retina_ratio': 0.0,
            'disc_detected': False,
            'macula_detected': False
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
    
    return {
        'image_path': image_path,
        'lens_detected': True,
        'lens_area': int(lens_area),
        'retina_area': int(retina_area),
        'retina_ratio': round(retina_ratio, 2),
        'disc_detected': disc_detected,
        'macula_detected': macula_detected
    }

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
    
    # CSVに保存
    if results:
        df = pd.DataFrame(results)
        # カラムの順序を整理
        columns_order = [
            'image_name',
            'image_path',
            'lens_detected',
            'lens_area',
            'retina_area',
            'retina_ratio',
            'disc_detected',
            'macula_detected'
        ]
        df = df[columns_order]
        df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        print(f"\n結果をCSVファイルに保存しました: {output_csv_path}")
        print(f"処理した画像数: {len(results)}")
    else:
        print("エラー: 処理結果がありません")

if __name__ == "__main__":
    # パス設定
    image_dir = r"C:\Users\ykita\ROP_AI_project\ROP_project\bestimage_validation\1227\画像"
    detection_model_path = r"C:\Users\ykita\ROP_AI_project\ROP_project\models\rtdetr-l-1697_1703.pt"
    segmentation_model_path = r"C:\Users\ykita\ROP_AI_project\ROP_project\models\yolo11n-seg_19movies.pt"
    
    # 出力CSVファイルのパス
    output_dir = r"C:\Users\ykita\ROP_AI_project\ROP_project\bestimage_validation"
    case_id = os.path.basename(os.path.dirname(image_dir))  # "1227"
    output_csv_path = os.path.join(output_dir, f"validation_results_{case_id}.csv")
    
    # 処理実行
    process_directory(image_dir, detection_model_path, segmentation_model_path, output_csv_path)
