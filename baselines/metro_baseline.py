"""
HandOccNet-style Hand Mesh Recovery Baseline.

Based on: Park et al., "HandOccNet: Occlusion-Robust 3D Hand Mesh Estimation
Network", CVPR 2022.

Core architecture (simplified):
  1. ResNet-50 backbone → multi-scale feature maps
  2. Spatial channel attention (SE-style) for occlusion-robust features
  3. Iterative MANO parameter regression (3 iterations, HMR-style feedback)
  4. MANO forward pass → vertices(778×3) + joints(21×3)

Key differences from full HandOccNet:
  - Single-scale global feature (no FPN) for training efficiency
  - SE channel attention instead of full transformer occlusion module
  - Same MANO output format as MD-HCRNet for fair comparison

Model name for tables: "HandOccNet"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.mano_layer import MANOLayer


class _SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        w = x.mean(dim=[2, 3])          # (B, C) global avg pool
        w = self.fc(w).view(B, C, 1, 1)
        return x * w


class _IterativeRegressor(nn.Module):
    """
    Iterative MANO parameter regressor (HMR-style).

    Each iteration receives [global_feat | beta | theta | trans] and predicts
    residual updates, refining the estimate step by step.
    """

    def __init__(self, feat_dim: int, num_iter: int = 3):
        super().__init__()
        self.num_iter = num_iter
        in_dim = feat_dim + 10 + 48 + 3   # feat + beta + theta + trans

        self.regressor = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 10 + 48 + 3),   # delta_beta + delta_theta + delta_trans
        )
        nn.init.zeros_(self.regressor[-1].weight)
        nn.init.zeros_(self.regressor[-1].bias)

    def forward(self, feat: torch.Tensor):
        """
        Args:
            feat: (B, feat_dim) global image feature

        Returns:
            beta:  (B, 10)
            theta: (B, 48)
            trans: (B,  3)
        """
        B   = feat.shape[0]
        dev = feat.device

        beta  = torch.zeros(B, 10, device=dev, dtype=feat.dtype)
        theta = torch.zeros(B, 48, device=dev, dtype=feat.dtype)
        trans = torch.zeros(B,  3, device=dev, dtype=feat.dtype)

        for _ in range(self.num_iter):
            inp   = torch.cat([feat, beta, theta, trans], dim=-1)
            delta = self.regressor(inp)
            beta  = beta  + delta[:, :10]
            theta = theta + delta[:, 10:58]
            trans = trans + delta[:, 58:]

        return beta, theta, trans


class HandOccNetBaseline(nn.Module):
    """
    HandOccNet-style baseline for 3D hand mesh recovery.

    Input:  RGB image (B, 3, H, W)
    Output: joints3d (B, 21, 3), vertices (B, 778, 3), beta (B, 10), theta (B, 48)

    Model name: "HandOccNet"
    """

    MODEL_NAME = "HandOccNet"

    def __init__(self,
                 num_iter: int = 3,
                 mano_root: str = "data",
                 is_rhand: bool = True,
                 use_pca: bool = False,
                 flat_hand_mean: bool = False,
                 **kwargs):
        super().__init__()

        # ── ResNet-50 backbone (same family as I2L for fair comparison) ────────
        resnet = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
        # Remove final avgpool + fc; keep up to layer4
        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
        )
        feat_channels = 2048   # ResNet-50 layer4 output channels

        # ── SE channel attention for occlusion-robust feature weighting ────────
        self.se_attn = _SEBlock(feat_channels, reduction=16)

        # ── Global feature projection ──────────────────────────────────────────
        self.feat_proj = nn.Sequential(
            nn.Linear(feat_channels, 512),
            nn.ReLU(inplace=True),
        )
        feat_dim = 512

        # ── Iterative MANO parameter regressor ────────────────────────────────
        self.regressor = _IterativeRegressor(feat_dim, num_iter=num_iter)

        # ── MANO layer ─────────────────────────────────────────────────────────
        self.mano = MANOLayer(
            mano_root=mano_root,
            is_rhand=is_rhand,
            use_pca=use_pca,
            flat_hand_mean=flat_hand_mean,
        )

    def forward(self,
                rgb: torch.Tensor,
                focal: torch.Tensor = None,
                cx: torch.Tensor = None,
                cy: torch.Tensor = None,
                depth: torch.Tensor = None) -> dict:
        B   = rgb.shape[0]
        dev = rgb.device

        # ── Feature extraction ─────────────────────────────────────────────────
        feat_map = self.backbone(rgb)              # (B, 2048, H', W')
        feat_map = self.se_attn(feat_map)          # channel-attended
        feat_vec = feat_map.mean(dim=[2, 3])       # (B, 2048) global avg pool
        feat_vec = self.feat_proj(feat_vec)        # (B, 512)

        # ── Iterative MANO regression ──────────────────────────────────────────
        beta, theta, trans = self.regressor(feat_vec)

        # ── MANO forward ───────────────────────────────────────────────────────
        vertices, joints3d = self.mano(beta, theta)
        joints3d = joints3d + trans.unsqueeze(1)   # (B, 21, 3)
        vertices = vertices + trans.unsqueeze(1)   # (B, 778, 3)

        # ── 2D projection ──────────────────────────────────────────────────────
        if focal is not None:
            eps  = 1e-6
            z    = joints3d[:, :, 2:3].clamp(min=eps)
            f_   = focal.view(B, 1, 1)
            cx_  = cx.view(B, 1, 1)
            cy_  = cy.view(B, 1, 1)
            u    = joints3d[:, :, 0:1] / z * f_ + cx_
            v    = joints3d[:, :, 1:2] / z * f_ + cy_
            joints2d = torch.cat([u, v, torch.ones_like(u)], dim=-1)
        else:
            joints2d = torch.zeros(B, 21, 3, device=dev)

        return dict(
            joints3d    = joints3d,
            joints3d_s1 = joints3d,
            joints2d    = joints2d,
            joints2d_s1 = joints2d,
            vertices    = vertices,
            beta        = beta,
            theta       = theta,
        )
