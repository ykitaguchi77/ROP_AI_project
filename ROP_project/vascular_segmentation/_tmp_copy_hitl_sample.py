"""Copy 100 random image+mask pairs from Before_HITL to 改正先生送付用, excluding Human_Annotation basenames."""
import random, shutil
from pathlib import Path

human_dir = Path(r'E:\Kaisho_vascular_annotation\Human_Annotation\images')
before_img = Path(r'E:\Kaisho_vascular_annotation\Before_HITL\images')
before_mask = Path(r'E:\Kaisho_vascular_annotation\Before_HITL\masks')
dst_img = Path(r'E:\Kaisho_vascular_annotation\改正先生送付用\images')
dst_mask = Path(r'E:\Kaisho_vascular_annotation\改正先生送付用\masks')

dst_img.mkdir(parents=True, exist_ok=True)
dst_mask.mkdir(parents=True, exist_ok=True)

# Basenames to exclude
exclude = set(p.name for p in human_dir.glob('*'))
print(f'Excluding {len(exclude)} Human_Annotation basenames')

# Candidates (exist in both images and masks, not in exclude)
candidates = sorted([
    p.name for p in before_img.glob('*.png')
    if p.name not in exclude and (before_mask / p.name).exists()
])
print(f'Candidates after exclusion: {len(candidates)}')

# Random sample
random.seed(42)
selected = random.sample(candidates, 100)
print(f'Selected: {len(selected)}')

# Copy
for name in selected:
    shutil.copy2(before_img / name, dst_img / name)
    shutil.copy2(before_mask / name, dst_mask / name)

# Verify
n_img = len(list(dst_img.glob('*.png')))
n_mask = len(list(dst_mask.glob('*.png')))
print(f'Copied: {n_img} images, {n_mask} masks')
print(f'Output: {dst_img.parent}')
print(f'Sample: {selected[:5]}')
