"""
RETFound utilities for feature extraction.

Self-contained module extracted from RETFound_MAE (rmaphoh/RETFound_MAE):
- VisionTransformer with global average pooling support
- RETFound_mae factory (ViT-Large: patch16, embed_dim=1024, depth=24, num_heads=16)
- Positional embedding interpolation for resolution changes
- load_retfound(): one-call loader returning a frozen model ready for inference

Reference:
  Zhou et al., "A foundation model for generalizable disease detection from retinal images"
  Nature 2023. https://doi.org/10.1038/s41586-023-06555-x
"""

from functools import partial
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import timm.models.vision_transformer


# ============================================================
# Vision Transformer with global average pooling
# (from models_vit.py)
# ============================================================

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """Vision Transformer with support for global average pooling."""

    def __init__(self, global_pool=False, **kwargs):
        super().__init__(**kwargs)
        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)
            del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1, keepdim=True)  # global pool without cls token
            outcome = self.fc_norm(x).squeeze(1)  # (B, embed_dim)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome


def RETFound_mae(**kwargs):
    """Create RETFound ViT-Large model (MAE pre-trained on 1.6M colour fundus photos)."""
    model = VisionTransformer(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


# ============================================================
# Positional embedding interpolation
# (from util/pos_embed.py)
# ============================================================

def interpolate_pos_embed(model, checkpoint_model):
    """Interpolate position embeddings for different image resolutions."""
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        new_size = int(num_patches ** 0.5)
        if orig_size != new_size:
            print(f'Position interpolate from {orig_size}x{orig_size} to {new_size}x{new_size}')
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False,
            )
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed


# ============================================================
# One-call loader
# ============================================================

def load_retfound(
    weights_path: Union[str, Path],
    device: torch.device = torch.device('cuda'),
    img_size: int = 224,
) -> nn.Module:
    """
    Load RETFound ViT-Large with pre-trained weights, frozen for feature extraction.

    Args:
        weights_path: Path to RETFound_cfp_weights.pth
        device: torch device
        img_size: Input image size (default 224, matching RETFound training)

    Returns:
        Frozen VisionTransformer model producing 1024-dim feature vectors.
    """
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(
            f'RETFound weights not found at {weights_path}.\n'
            'Download from: https://drive.google.com/file/d/1l62zbWUFTlp214SvK6eMwPQZAzcwoeBE/view\n'
            f'Save to: {weights_path}'
        )

    # Create model with global pooling → 1024-dim output
    model = RETFound_mae(global_pool=True, img_size=img_size)

    # Load checkpoint
    checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
    checkpoint_model = checkpoint.get('model', checkpoint)

    # Interpolate positional embeddings if image size differs
    interpolate_pos_embed(model, checkpoint_model)

    # Filter out head/decoder keys not present in the encoder
    state_dict = model.state_dict()
    filtered = {
        k: v for k, v in checkpoint_model.items()
        if k in state_dict and v.shape == state_dict[k].shape
    }
    # Load with strict=False to skip missing head weights
    msg = model.load_state_dict(filtered, strict=False)
    print(f'RETFound loaded: {len(filtered)}/{len(state_dict)} params matched')
    if msg.missing_keys:
        # fc_norm is expected to be missing from MAE checkpoint
        expected_missing = {'fc_norm.weight', 'fc_norm.bias', 'head.weight', 'head.bias'}
        unexpected = set(msg.missing_keys) - expected_missing
        if unexpected:
            print(f'  Unexpected missing keys: {unexpected}')

    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    return model
