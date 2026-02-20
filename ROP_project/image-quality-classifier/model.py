"""EfficientNet-B0 wrapper for quality classification."""

import timm
import torch.nn as nn


class QualityClassifier(nn.Module):
    def __init__(self, model_name: str = "efficientnet_b0", num_classes: int = 4, dropout: float = 0.5):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        num_features = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

    def freeze_backbone(self, unfreeze_last_n_blocks: int = 2):
        """Freeze backbone except last N blocks and head."""
        # Freeze all backbone parameters first
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze last N blocks of EfficientNet
        # EfficientNet-B0 has blocks[0..6] (7 blocks total)
        blocks = list(self.backbone.blocks)
        for block in blocks[-unfreeze_last_n_blocks:]:
            for param in block.parameters():
                param.requires_grad = True

        # Always unfreeze batch norm in conv_head and bn2
        if hasattr(self.backbone, 'conv_head'):
            for param in self.backbone.conv_head.parameters():
                param.requires_grad = True
        if hasattr(self.backbone, 'bn2'):
            for param in self.backbone.bn2.parameters():
                param.requires_grad = True

        # Head is always trainable
        for param in self.head.parameters():
            param.requires_grad = True

        # Count trainable params
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print("  Backbone frozen: {}/{} params trainable ({:.1f}%)".format(
            trainable, total, 100.0 * trainable / total))


def create_model(model_name: str = "efficientnet_b0", num_classes: int = 4, dropout: float = 0.5) -> QualityClassifier:
    return QualityClassifier(model_name=model_name, num_classes=num_classes, dropout=dropout)
