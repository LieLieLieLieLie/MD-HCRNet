"""
I2L-style Baseline (top-conference level).

Based on: Moon et al., "I2L-MeshNet: Image-to-Lixel Prediction Network for
Accurate 3D Human Pose and Mesh Estimation from a Single RGB Image",
ECCV 2020.

Core idea (simplified implementation):
  1. ResNet-50 backbone → multi-scale feature maps
  2. Per-joint 2D heatmap head → soft-argmax → pixel-space (x, y) per joint
  3. Per-joint depth head (global feature → MLP) → metric depth per joint
  4. Back-project (x, y, z) using camera intrinsics → 3D joints in camera space

Differences from full I2L-MeshNet:
  - We predict 21 hand joints only (no mesh vertices)
  - Depth is predicted as direct regression (not a 1D heatmap over D bins)
  - No mesh regression branch

Model name for tables: "I2L-Baseline"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


class _PositionMapHead(nn.Module):
    """Per-joint 2D soft-argmax over feature-map spatial dimensions."""

    def __init__(self, in_channels: int, num_joints: int, feat_h: int, feat_w: int):
        super().__init__()
        self.num_joints = num_joints
        self.feat_h     = feat_h
        self.feat_w     = feat_w
        self.conv = nn.Conv2d(in_channels, num_joints, kernel_size=1)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, C, H, W) feature map
        Returns:
            xy: (B, J, 2) in normalised [0,1] image coordinates
        """
        B = x.shape[0]
        J = self.num_joints
        hm = self.conv(x)                              # (B, J, H, W)
        # Soft-argmax per joint
        hm_flat = hm.view(B, J, -1).softmax(dim=-1)   # (B, J, H*W)
        hm_2d   = hm_flat.view(B, J, self.feat_h, self.feat_w)

        # x coordinate (normalised 0→1 over width)
        device   = x.device
        x_lin    = torch.linspace(0, 1, self.feat_w, device=device)
        x_weight = hm_2d.sum(dim=2)                   # (B, J, W)
        x_coord  = (x_weight * x_lin).sum(dim=-1)     # (B, J)

        # y coordinate (normalised 0→1 over height)
        y_lin    = torch.linspace(0, 1, self.feat_h, device=device)
        y_weight = hm_2d.sum(dim=3)                   # (B, J, H)
        y_coord  = (y_weight * y_lin).sum(dim=-1)     # (B, J)

        return torch.stack([x_coord, y_coord], dim=-1)  # (B, J, 2) in [0,1]


class I2LBaseline(nn.Module):
    """
    I2L-style 3D hand joint estimation.

    Input:  RGB image (B, 3, 256, 192)
    Output: joints3d (B, 21, 3) in camera space (metres)

    Model name: "I2L-Baseline"
    """

    MODEL_NAME = "I2L"

    def __init__(self, num_joints: int = 21,
                 img_h: int = 256, img_w: int = 192):
        super().__init__()
        self.num_joints = num_joints
        self.img_h      = img_h
        self.img_w      = img_w

        # ── ResNet-50 backbone (ImageNet pretrained) ──────────────────────────
        weights  = tvm.ResNet50_Weights.IMAGENET1K_V2
        backbone = tvm.resnet50(weights=weights)
        self.stem   = nn.Sequential(backbone.conv1, backbone.bn1,
                                    backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1   # 256 ch, H/4, W/4
        self.layer2 = backbone.layer2   # 512 ch, H/8, W/8
        self.layer3 = backbone.layer3   # 1024 ch, H/16, W/16
        self.layer4 = backbone.layer4   # 2048 ch, H/32, W/32

        # Feature-map size at layer4 for input (256, 192):
        # H/32 = 8, W/32 = 6
        feat_h, feat_w = img_h // 32, img_w // 32

        # Channel reduction before heatmap head
        self.reduce = nn.Sequential(
            nn.Conv2d(2048, 512, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # ── 2D position-map head ──────────────────────────────────────────────
        self.pos_head = _PositionMapHead(512, num_joints, feat_h, feat_w)

        # ── Depth head (per-joint metric depth from global avg pool) ─────────
        self.depth_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_joints),
        )
        # Init depth to predict ~0.5m (typical hand distance)
        nn.init.zeros_(self.depth_head[-1].weight)
        nn.init.constant_(self.depth_head[-1].bias, 0.5)

        # Store image grid for back-projection
        self.register_buffer(
            "_x_grid",
            torch.linspace(0, img_w - 1, img_w).view(1, 1, img_w).float()
        )
        self.register_buffer(
            "_y_grid",
            torch.linspace(0, img_h - 1, img_h).view(1, img_h, 1).float()
        )

    # ── Back-projection ───────────────────────────────────────────────────────

    def _backproject(self, xy_norm: torch.Tensor, depth: torch.Tensor,
                     focal: torch.Tensor, cx: torch.Tensor, cy: torch.Tensor):
        """
        Convert normalised image coords + depth → 3D camera-space coords.

        Args:
            xy_norm: (B, J, 2) x,y in [0,1]
            depth:   (B, J)   metric depth in metres
            focal:   (B,)
            cx, cy:  (B,)
        Returns:
            joints3d: (B, J, 3)
        """
        x_px = xy_norm[..., 0] * self.img_w   # (B, J)
        y_px = xy_norm[..., 1] * self.img_h

        f  = focal.view(-1, 1)
        cx = cx.view(-1, 1)
        cy = cy.view(-1, 1)
        Z  = depth.clamp(min=0.05)

        X = (x_px - cx) / f * Z
        Y = (y_px - cy) / f * Z
        return torch.stack([X, Y, Z], dim=-1)  # (B, J, 3)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, rgb: torch.Tensor,
                focal: torch.Tensor = None,
                cx: torch.Tensor = None,
                cy: torch.Tensor = None,
                depth: torch.Tensor = None) -> dict:
        B   = rgb.shape[0]
        dev = rgb.device

        # Backbone
        x = self.stem(rgb)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)          # (B, 2048, H/32, W/32)

        # Global feature for depth
        gf       = x.mean(dim=[2, 3])   # (B, 2048)
        feat_red = self.reduce(x)        # (B, 512, H/32, W/32)

        # 2D position (normalised)
        xy_norm = self.pos_head(feat_red)   # (B, J, 2) in [0,1]

        # Per-joint metric depth
        z = self.depth_head(gf)             # (B, J)

        # Back-project → 3D camera space
        if focal is None:
            # Fallback: assume 500px focal, principal at image centre
            focal = torch.full((B,), 500.0, device=dev)
            cx    = torch.full((B,), self.img_w / 2, device=dev)
            cy    = torch.full((B,), self.img_h / 2, device=dev)

        joints3d = self._backproject(xy_norm, z, focal, cx, cy)  # (B, J, 3)

        dummy_2d = torch.zeros(B, self.num_joints, 2, device=dev)
        return dict(
            joints3d    = joints3d,
            joints3d_s1 = joints3d,
            joints2d    = dummy_2d,
            joints2d_s1 = dummy_2d,
        )
