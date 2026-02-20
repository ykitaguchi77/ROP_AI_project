"""一時スクリプト: prepare_data.ipnb で処理済みの512x512データを
Human_Annotation/ にコピー（マスクはRGBA形式に変換）。
実行後に削除すること。
"""
import cv2
import numpy as np
import shutil
from pathlib import Path

# Source: prepare_data.ipynb の出力
DATA_DIR = Path(r'C:\Users\ykita\ROP_AI_project\ROP_project\vascular_segmentation\data')
SRC_IMG_DIR = DATA_DIR / 'images'
SRC_MASK_DIR = DATA_DIR / 'masks'

# Destination
DST_DIR = Path(r'E:\Kaisho_vascular_annotation\Human_Annotation')
DST_IMG_DIR = DST_DIR / 'images'
DST_MASK_DIR = DST_DIR / 'masks'

DST_IMG_DIR.mkdir(parents=True, exist_ok=True)
DST_MASK_DIR.mkdir(parents=True, exist_ok=True)


def binary_to_rgba(binary_mask: np.ndarray) -> np.ndarray:
    """Binary mask (0/255) → RGBA PNG (Red=vessel, White=background)."""
    h, w = binary_mask.shape[:2]
    rgba = np.full((h, w, 4), 255, dtype=np.uint8)
    vessel = binary_mask > 0
    rgba[vessel, 1] = 0  # G=0
    rgba[vessel, 2] = 0  # B=0
    return rgba


# Process
src_images = sorted(SRC_IMG_DIR.glob('*.png'))
print(f'Source images: {len(src_images)}')

for img_path in src_images:
    fname = img_path.name

    # Copy image (already 512x512, circular-masked)
    shutil.copy2(str(img_path), str(DST_IMG_DIR / fname))

    # Convert mask to RGBA and save
    mask_gray = cv2.imread(str(SRC_MASK_DIR / fname), cv2.IMREAD_GRAYSCALE)
    mask_rgba = binary_to_rgba(mask_gray)
    mask_bgra = cv2.cvtColor(mask_rgba, cv2.COLOR_RGBA2BGRA)
    cv2.imwrite(str(DST_MASK_DIR / fname), mask_bgra)

    print(f'  {fname}: done')

# Verify
n_dst_img = len(list(DST_IMG_DIR.glob('*.png')))
n_dst_mask = len(list(DST_MASK_DIR.glob('*.png')))
print(f'\nResult: images={n_dst_img}, masks={n_dst_mask}')

# Verify format
sample = sorted(DST_MASK_DIR.glob('*.png'))[0]
m = cv2.imread(str(sample), cv2.IMREAD_UNCHANGED)
print(f'Mask format check: shape={m.shape}, dtype={m.dtype}')
assert m.shape == (512, 512, 4), f'Expected (512,512,4), got {m.shape}'
print('All done!')
