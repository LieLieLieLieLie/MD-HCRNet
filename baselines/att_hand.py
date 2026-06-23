"""
LightAttHand — Lightweight spatial attention (SCI 1区 journal level).

Inspired by: 2022-2023 journal works on efficient hand pose estimation using
joint-specific attention maps (e.g., "Efficient Hand Pose Estimation via
Spatial Attention and Lightweight Backbone", various IEEE TIP/PR submissions).

Core idea:
  1. MobileNetV3-Large backbone (efficient, suitable for journal "lightweight" angle)
  2. For each joint: predict a spatial attention map over the feature map
  3. Attention-weighted pooling → joint-specific feature vector
  4. Per-joint MLP → 3D position

Key distinction from direct regression:
  Joint-specific attention allows the model to focus on relevant image regions
  for each joint (e.g., fingertip detection focuses on finger-tip regions).
  More interpretable than global pooling.

Model name for tables: "LightAttHand"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


class _JointAttention(nn.Module):
    """
    Predict joint-specific spatial attention maps from feature maps.

    For each of J joints, predict an (H', W') attention map; pool to get
    a J×C joint feature matrix.
    """

    def __init__(self, in_channels: int, num_joints: int):
        super().__init__()
        # Channel reduction + attention map per joint
        hidden = max(in_channels // 4, 64)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, num_joints, 1),   # (B, J, H', W')
        )

    def forward(self, feat: torch.Tensor, node_feat: torch.Tensor = None):
        """
        Args:
            feat:      (B, C, H', W') feature map
            node_feat: unused (for API compatibility)
        Returns:
            joint_feats: (B, J, C) attention-pooled joint features
            attn_maps:   (B, J, H', W') attention weights (for visualisation)
        """
        B, C, H, W = feat.shape
        attn_logits = self.conv(feat)                       # (B, J, H, W)
        attn_weights = attn_logits.view(B, -1, H * W).softmax(-1).view(B, -1, H, W)

        # Weighted average pooling per joint
        feat_flat   = feat.view(B, C, H * W)               # (B, C, H*W)
        attn_flat   = attn_weights.view(B, -1, H * W)      # (B, J, H*W)
        joint_feats = torch.bmm(attn_flat, feat_flat.permute(0, 2, 1))  # (B, J, C)

        return joint_feats, attn_weights


class LightAttHand(nn.Module):
    """
    Lightweight attention-based 3D hand joint estimator.

    Input:  RGB image (B, 3, 256, 192)
    Output: joints3d (B, 21, 3) in camera space (metres)

    Model name: "LightAttHand"
    """

    MODEL_NAME = "LightAttHand"

    def __init__(self, num_joints: int = 21):
        super().__init__()
        self.num_joints = num_joints

        # ── MobileNetV3-Large backbone (efficient) ────────────────────────────
        weights  = tvm.MobileNet_V3_Large_Weights.IMAGENET1K_V2
        backbone = tvm.mobilenet_v3_large(weights=weights)
        self.backbone = backbone.features   # → (B, 960, H/32, W/32)
        feat_dim      = 960

        # ── Joint-specific spatial attention ──────────────────────────────────
        self.attention = _JointAttention(feat_dim, num_joints)

        # ── Per-joint MLP: pooled feature → 3D position ───────────────────────
        hidden = 256
        self.mlp = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 3),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

        # ── Global depth prior (prevents all joints collapsing to z=0) ────────
        self.depth_bias = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feat_dim, num_joints),
        )
        nn.init.zeros_(self.depth_bias[-1].weight)
        nn.init.constant_(self.depth_bias[-1].bias, 0.5)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, rgb: torch.Tensor,
                focal: torch.Tensor = None,
                cx: torch.Tensor = None,
                cy: torch.Tensor = None,
                depth: torch.Tensor = None) -> dict:
        B   = rgb.shape[0]
        dev = rgb.device
        J   = self.num_joints

        # ── Backbone ──────────────────────────────────────────────────────────
        feat = self.backbone(rgb)                  # (B, 960, H', W')

        # ── Joint attention → per-joint pooled features ───────────────────────
        joint_feats, _ = self.attention(feat)      # (B, J, 960)

        # ── Per-joint 3D position ─────────────────────────────────────────────
        joints3d = self.mlp(joint_feats)           # (B, J, 3)

        # Add global depth bias to z channel
        z_bias = self.depth_bias(feat).unsqueeze(-1)   # (B, J, 1)
        joints3d = joints3d + torch.cat(
            [torch.zeros(B, J, 2, device=dev), z_bias], dim=-1
        )

        dummy_2d = torch.zeros(B, J, 2, device=dev)
        return dict(
            joints3d    = joints3d,
            joints3d_s1 = joints3d,
            joints2d    = dummy_2d,
            joints2d_s1 = dummy_2d,
        )
