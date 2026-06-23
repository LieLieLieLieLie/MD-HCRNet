"""
Baseline 1: ResNet-50 + Direct 3D Joint Regression.

Architecture:
  ResNet-50 (ImageNet pretrained) → global average pool → MLP → 21×3 joints

No MANO prior. No depth input. No iterative refinement.
Used as a classical backbone baseline to demonstrate the value of:
  (a) ViT backbone over CNN
  (b) MANO structural prior
  (c) Depth guidance
"""
import torch
import torch.nn as nn
import torchvision.models as tvm


class ResNetRegressor(nn.Module):
    """
    ResNet-50 backbone with direct 3D joint regression head.

    Args:
        num_joints:    Number of output joints (default 21 for FreiHAND)
        freeze_backbone: Freeze ResNet conv layers (default False for baselines)
        pretrained:    Use ImageNet pretrained weights
    """

    def __init__(self, num_joints: int = 21,
                 freeze_backbone: bool = False,
                 pretrained: bool = True):
        super().__init__()
        weights = tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = tvm.resnet50(weights=weights)
        # Remove FC layer; keep up to avgpool
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # → (B,2048,1,1)
        feat_dim = 2048

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        # Regression head: 2048 → 512 → 21×3
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_joints * 3),
        )
        # Zero-init output layer (same stability trick as Stage 1)
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

        self.num_joints = num_joints

    def forward(self, rgb: torch.Tensor,
                focal: torch.Tensor = None,
                cx:    torch.Tensor = None,
                cy:    torch.Tensor = None,
                depth: torch.Tensor = None) -> dict:
        """
        Args:
            rgb:   (B, 3, H, W)
            focal, cx, cy, depth: accepted but ignored (API compatibility)

        Returns dict:
            joints3d:    (B, 21, 3)  — root-relative 3D joints in metres
            joints3d_s1: (B, 21, 3)  — alias for intermediate supervision compat.
            joints2d:    (B, 21, 2)  — dummy zeros (no projection head)
            joints2d_s1: (B, 21, 2)  — dummy zeros
        """
        B = rgb.shape[0]
        feat = self.backbone(rgb).flatten(1)        # (B, 2048)
        out  = self.head(feat).view(B, self.num_joints, 3)   # (B, 21, 3)

        dummy_2d = torch.zeros(B, self.num_joints, 2, device=rgb.device)
        return dict(
            joints3d    = out,
            joints3d_s1 = out,
            joints2d    = dummy_2d,
            joints2d_s1 = dummy_2d,
        )
