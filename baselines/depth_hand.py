"""
DepthFusionHand — Early-fusion RGB+Depth regression (SCI 1区 journal level).

Inspired by: 2022-2023 journal works on depth-guided hand pose estimation
(e.g., works in Pattern Recognition, IEEE TIP that use simple depth fusion).

Core idea:
  1. Concatenate RGB (3ch) + Depth (1ch) → 4-channel input
  2. ResNet-50 with modified first conv (4ch instead of 3ch)
  3. Global avg pool → MLP → 21×3 joint positions

Key distinction from MD-HCRNet:
  This shows NAIVE depth fusion (early concatenation, single branch, no MANO)
  vs our METHOD's METRIC DEPTH GUIDANCE (dual-branch ViT, knowledge-guided
  sampling, iterative MANO refinement). Simple depth ≠ our structured approach.

Model name for tables: "DepthFusion"
"""
import torch
import torch.nn as nn
import torchvision.models as tvm


def _resnet50_4ch(pretrained: bool = True) -> nn.Module:
    """
    ResNet-50 with first conv modified to accept 4-channel input (RGB + Depth).
    RGB channels keep ImageNet pretrained weights; depth channel is initialised
    to the average of the 3 RGB channels (conservative RGB-init strategy).
    """
    weights  = tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    backbone = tvm.resnet50(weights=weights)

    old_conv   = backbone.conv1                    # 3-ch → 64-ch
    new_conv   = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # Copy pretrained RGB weights; depth channel ← mean of RGB weights
    with torch.no_grad():
        new_conv.weight[:, :3, :, :] = old_conv.weight.data
        new_conv.weight[:, 3:4, :, :] = old_conv.weight.data.mean(dim=1, keepdim=True)

    backbone.conv1 = new_conv
    return backbone


class DepthFusionHand(nn.Module):
    """
    Early-fusion RGB+Depth hand pose estimator.

    Input:  RGB (B,3,H,W) + optional Depth (B,1,H,W)
    Output: joints3d (B, 21, 3) in camera space (metres)

    If depth is None (no cache), uses zero depth channel — degrades to RGB-only.

    Model name: "DepthFusion"
    """

    MODEL_NAME = "DepthFusion"

    def __init__(self, num_joints: int = 21, pretrained: bool = True):
        super().__init__()
        self.num_joints = num_joints

        # ── ResNet-50 with 4-channel input ────────────────────────────────────
        backbone = _resnet50_4ch(pretrained)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # → (B,2048,1,1)

        # ── Regression head ───────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_joints * 3),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, rgb: torch.Tensor,
                focal: torch.Tensor = None,
                cx: torch.Tensor = None,
                cy: torch.Tensor = None,
                depth: torch.Tensor = None) -> dict:
        B   = rgb.shape[0]
        dev = rgb.device
        J   = self.num_joints

        # ── Build 4-channel input ─────────────────────────────────────────────
        if depth is None:
            # No depth available → zero channel (pure RGB mode)
            d = torch.zeros(B, 1, rgb.shape[2], rgb.shape[3], device=dev)
        else:
            # Depth cache may have shape (B,1,H,W) already
            d = depth.float()
            if d.ndim == 3:
                d = d.unsqueeze(1)
            # Normalise depth to [0,1] for stable training
            d_min = d.view(B, -1).min(1)[0].view(B, 1, 1, 1)
            d_max = d.view(B, -1).max(1)[0].view(B, 1, 1, 1).clamp(min=d_min + 1e-6)
            d = (d - d_min) / (d_max - d_min)

        inp = torch.cat([rgb, d], dim=1)         # (B, 4, H, W)

        # ── Backbone + head ───────────────────────────────────────────────────
        feat     = self.backbone(inp).flatten(1) # (B, 2048)
        joints3d = self.head(feat).view(B, J, 3) # (B, J, 3)

        dummy_2d = torch.zeros(B, J, 2, device=dev)
        return dict(
            joints3d    = joints3d,
            joints3d_s1 = joints3d,
            joints2d    = dummy_2d,
            joints2d_s1 = dummy_2d,
        )
