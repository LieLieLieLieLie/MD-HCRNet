"""
Stage 1 — Initialization (Geometry-aware Encoding).

• Dual-branch ViT encodes RGB and Depth
• Special tokens (β, θ, t) are fused across branches
• MANO decoder produces Stage-1 3D joints
• Focal projection produces 2D joints used as anchors for Stage 2
"""
import torch
import torch.nn as nn

from .vit_encoder import DualBranchViT
from .mano_layer  import MANOLayer
from utils.geometry import focal_projection


class TokenFusion(nn.Module):
    """Fuse two (B, C) tokens from the RGB and Depth branches."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(torch.cat([a, b], dim=-1)))


class ParameterHead(nn.Module):
    """MLP that maps a single token to MANO parameter vector."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim // 2),
            nn.GELU(),
            nn.Linear(in_dim // 2, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Stage1(nn.Module):
    """
    Args:
        cfg: model config section (embed_dim, num_heads, num_layers,
             patch_size, img_height, img_width, freeze_backbone)
        mano_cfg: mano config section
    """

    def __init__(self, cfg, mano_cfg):
        super().__init__()
        D = cfg.embed_dim
        H, W = cfg.img_height, cfg.img_width

        # ── Dual-branch ViT ────────────────────────────────────────────────
        self.backbone = DualBranchViT(
            img_h=H, img_w=W,
            patch_size=cfg.patch_size,
            embed_dim=D,
            depth=cfg.num_layers,
            num_heads=cfg.num_heads,
            pretrained=True,
        )
        if getattr(cfg, "freeze_backbone", True):
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        # ── Fusion of matching special tokens ──────────────────────────────
        self.beta_fuse  = TokenFusion(D)
        self.theta_fuse = TokenFusion(D)
        self.trans_fuse = TokenFusion(D)

        # ── MANO parameter heads ───────────────────────────────────────────
        self.beta_head  = ParameterHead(D, 10)
        self.theta_head = ParameterHead(D, 48)
        self.trans_head = ParameterHead(D, 3)
        # 输出层初始化为零权重，避免随机初始化产生极端关节位置导致 NaN
        nn.init.zeros_(self.beta_head.net[-1].weight)
        nn.init.zeros_(self.beta_head.net[-1].bias)
        nn.init.zeros_(self.theta_head.net[-1].weight)
        nn.init.zeros_(self.theta_head.net[-1].bias)
        nn.init.zeros_(self.trans_head.net[-1].weight)
        # trans 偏置初始化为 (0, 0, 0.5)：手在相机前方约 50cm，确保 Z > 0
        with torch.no_grad():
            self.trans_head.net[-1].bias.copy_(
                torch.tensor([0.0, 0.0, 0.5]))

        # ── MANO decoder (trainable) ───────────────────────────────────────
        self.mano = MANOLayer(
            mano_root=mano_cfg.model_path,
            is_rhand=mano_cfg.is_rhand,
            use_pca=mano_cfg.use_pca,
            flat_hand_mean=mano_cfg.flat_hand_mean,
        )

        # image size stored for projection
        self.img_h = H
        self.img_w = W

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor,
                focal: torch.Tensor, cx: torch.Tensor, cy: torch.Tensor):
        """
        Args:
            rgb:   (B, 3, H, W)
            depth: (B, 1, H, W)
            focal: (B,)  focal length in pixels
            cx:    (B,)  principal-point x
            cy:    (B,)  principal-point y

        Returns dict:
            rgb_tokens:   (B, N+3, C)
            depth_tokens: (B, N+3, C)
            beta_s1:      (B, 10)
            theta_s1:     (B, 48)
            trans_s1:     (B, 3)
            vertices_s1:  (B, 778, 3)
            joints3d_s1:  (B, 21, 3)
            joints2d_s1:  (B, 21, 2)
        """
        # ── Dual ViT forward ────────────────────────────────────────────────
        rgb_tok, depth_tok = self.backbone(rgb, depth)
        # Both: B, N+3, C  (indices 0=β, 1=θ, 2=t, 3:=patches)

        # ── Extract and fuse special tokens ─────────────────────────────────
        rgb_beta,   rgb_theta,   rgb_trans   = rgb_tok[:, 0], rgb_tok[:, 1], rgb_tok[:, 2]
        dep_beta,   dep_theta,   dep_trans   = depth_tok[:, 0], depth_tok[:, 1], depth_tok[:, 2]

        fused_beta  = self.beta_fuse(rgb_beta,  dep_beta)
        fused_theta = self.theta_fuse(rgb_theta, dep_theta)
        fused_trans = self.trans_fuse(rgb_trans, dep_trans)

        # ── Predict MANO parameters ──────────────────────────────────────────
        beta_s1  = self.beta_head(fused_beta)
        theta_s1 = self.theta_head(fused_theta)
        trans_s1 = self.trans_head(fused_trans)

        # ── MANO forward ─────────────────────────────────────────────────────
        vertices_s1, joints3d_local = self.mano(beta_s1, theta_s1)
        # translate to camera frame
        joints3d_s1 = joints3d_local + trans_s1.unsqueeze(1)

        # ── Focal projection → 2D joints ─────────────────────────────────────
        joints2d_s1 = focal_projection(joints3d_s1, focal, cx, cy)

        return dict(
            rgb_tokens   = rgb_tok,
            depth_tokens = depth_tok,
            beta_s1      = beta_s1,
            theta_s1     = theta_s1,
            trans_s1     = trans_s1,
            vertices_s1  = vertices_s1,
            joints3d_s1  = joints3d_s1,
            joints2d_s1  = joints2d_s1,
        )
