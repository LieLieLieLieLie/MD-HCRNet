"""
Baseline 2: ViT-B/16 + Direct 3D Joint Regression.

Architecture:
  ViT-B/16 (ImageNet-21k pretrained via timm) → [CLS] token → MLP → 21×3 joints

No MANO prior. No depth input. No iterative refinement.
Shares the same ViT backbone as MD-HCRNet Stage 1 but without:
  - Dual-branch depth encoding
  - MANO structural constraints
  - Stage 2 / Stage 3 refinement

Used as the "ablation" baseline that proves the contribution of:
  (a) MANO prior and structural constraints
  (b) Depth branch and metric depth guidance
  (c) Multi-stage refinement pipeline
"""
import torch
import torch.nn as nn

try:
    import timm
    _HAS_TIMM = True
except ImportError:
    _HAS_TIMM = False


class ViTRegressor(nn.Module):
    """
    ViT-B/16 backbone (timm) with direct 3D joint regression.

    Args:
        num_joints:      Output joint count (21 for FreiHAND)
        freeze_backbone: Freeze ViT weights
        embed_dim:       ViT embedding dim (768 for ViT-B)
    """

    def __init__(self, num_joints: int = 21,
                 freeze_backbone: bool = False,
                 embed_dim: int = 768):
        super().__init__()
        self.num_joints = num_joints

        if _HAS_TIMM:
            self.vit = timm.create_model(
                "vit_base_patch16_224",
                pretrained=True,
                num_classes=0,       # remove classification head → returns CLS token
                img_size=(256, 192), # match training image size
            )
            feat_dim = embed_dim
        else:
            # Fallback: simple CNN encoder
            import torchvision.models as tvm
            weights = tvm.ResNet18_Weights.IMAGENET1K_V1
            bb = tvm.resnet18(weights=weights)
            self.vit = nn.Sequential(*list(bb.children())[:-1])
            feat_dim = 512

        self._has_timm = _HAS_TIMM

        if freeze_backbone:
            for p in self.vit.parameters():
                p.requires_grad_(False)

        # Regression head: feat_dim → 512 → 21×3
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_joints * 3),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, rgb: torch.Tensor,
                focal: torch.Tensor = None,
                cx:    torch.Tensor = None,
                cy:    torch.Tensor = None,
                depth: torch.Tensor = None) -> dict:
        """
        Args:
            rgb:   (B, 3, H, W)
            focal, cx, cy, depth: accepted but ignored (API compatibility)

        Returns dict with joints3d, joints3d_s1, joints2d, joints2d_s1.
        """
        B = rgb.shape[0]
        if self._has_timm:
            feat = self.vit(rgb)                   # (B, embed_dim) — CLS token
        else:
            feat = self.vit(rgb).flatten(1)        # (B, 512)

        out = self.head(feat).view(B, self.num_joints, 3)
        dummy_2d = torch.zeros(B, self.num_joints, 2, device=rgb.device)
        return dict(
            joints3d    = out,
            joints3d_s1 = out,
            joints2d    = dummy_2d,
            joints2d_s1 = dummy_2d,
        )
