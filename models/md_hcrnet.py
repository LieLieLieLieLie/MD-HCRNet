"""
MD-HCRNet — Metric Depth-Guided Hand-Centric Reasoning Network
for Monocular 3D Hand Mesh Reconstruction.

Full three-stage forward pass:
  Stage 1  — Geometry-aware Encoding (dual-branch ViT + MANO initialisation)
  Stage 2  — Unified Hand-centric Reasoning (knowledge sampling + Mamba + cross-attn)
  Stage 3  — Metric-aware Reconstruction (residual refinement)
"""
import torch
import torch.nn as nn

from .depth_estimator import MetricDepthEstimator
from .stage1          import Stage1
from .stage2_module   import Stage2
from .stage3          import Stage3
from utils.geometry import focal_projection


class JointQueryRefineBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.query_norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True, dropout=0.0
        )
        self.query_norm2 = nn.LayerNorm(embed_dim)
        self.memory_norm = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True, dropout=0.0
        )
        self.query_norm3 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        q = self.query_norm1(queries)
        queries = queries + self.self_attn(q, q, q, need_weights=False)[0]
        queries = queries + self.cross_attn(
            self.query_norm2(queries),
            self.memory_norm(memory),
            self.memory_norm(memory),
            need_weights=False,
        )[0]
        return queries + self.mlp(self.query_norm3(queries))


class JointQueryRefiner(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_layers: int = 2,
        zero_init_output: bool = True,
    ):
        super().__init__()
        self.token_fuse = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.joint_queries = nn.Parameter(torch.zeros(1, 21, embed_dim))
        self.blocks = nn.ModuleList([
            JointQueryRefineBlock(embed_dim, num_heads)
            for _ in range(num_layers)
        ])
        self.out = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 3),
        )
        nn.init.trunc_normal_(self.joint_queries, std=0.02)
        if zero_init_output:
            nn.init.zeros_(self.out[-1].weight)
        else:
            nn.init.trunc_normal_(self.out[-1].weight, std=0.001)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, rgb_tokens: torch.Tensor, depth_tokens: torch.Tensor) -> torch.Tensor:
        memory = self.token_fuse(torch.cat([rgb_tokens, depth_tokens], dim=-1))
        queries = self.joint_queries.expand(rgb_tokens.shape[0], -1, -1)
        for block in self.blocks:
            queries = block(queries, memory)
        delta = self.out(queries)
        return delta - delta[:, :1, :]


class UHRJointResidualDecoder(nn.Module):
    """Predict a small root-relative joint residual from UHR fused tokens."""

    def __init__(self, embed_dim: int, zero_init_output: bool = True):
        super().__init__()
        self.out = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 3),
        )
        if zero_init_output:
            nn.init.zeros_(self.out[-1].weight)
        else:
            nn.init.trunc_normal_(self.out[-1].weight, std=0.001)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, uhr_tokens: torch.Tensor) -> torch.Tensor:
        # UHR sequence is [beta, theta, trans, joint_1, ..., joint_21].
        joint_tokens = uhr_tokens[:, 3:, :]
        residual = self.out(joint_tokens)
        return residual - residual[:, :1, :]


class MDHCRNet(nn.Module):
    MODEL_NAME = "MD-HCR"   # display name in tables / figures

    def __init__(self, cfg):
        """
        Args:
            cfg: the full config object (with sub-namespaces .model, .mano, .data)
        """
        super().__init__()
        mcfg = cfg.model
        dcfg = cfg.data

        # ── Depth estimator (frozen pretrained) ────────────────────────────
        self.depth_estimator = MetricDepthEstimator(mcfg.depth_estimator)
        for p in self.depth_estimator.parameters():
            p.requires_grad_(False)

        # ── Stage 1 ─────────────────────────────────────────────────────────
        self.stage1 = Stage1(mcfg, cfg.mano)

        # ── Stage 2 ─────────────────────────────────────────────────────────
        self.stage2 = Stage2(mcfg, mcfg.embed_dim)

        # ── Stage 3 shares MANO from Stage 1 ──────────────────────────────
        self.stage3 = Stage3(self.stage1.mano)

        self.global_joint_refine = nn.Sequential(
            nn.LayerNorm(mcfg.embed_dim * 4),
            nn.Linear(mcfg.embed_dim * 4, mcfg.embed_dim),
            nn.GELU(),
            nn.Linear(mcfg.embed_dim, 21 * 3),
        )
        nn.init.zeros_(self.global_joint_refine[-1].weight)
        nn.init.zeros_(self.global_joint_refine[-1].bias)
        self.joint_query_refine = JointQueryRefiner(
            mcfg.embed_dim,
            mcfg.num_heads,
            getattr(mcfg, "joint_query_layers", 2),
        )
        self.use_direct_joint_decoder = getattr(mcfg, "use_direct_joint_decoder", False)
        if self.use_direct_joint_decoder:
            self.direct_joint_decoder = UHRJointResidualDecoder(
                mcfg.embed_dim,
                getattr(mcfg, "direct_joint_zero_init", True),
            )

    # ─────────────────────────────────────────────────────────────────────────

    def forward(self,
                rgb: torch.Tensor,
                focal: torch.Tensor,
                cx: torch.Tensor,
                cy: torch.Tensor,
                depth: torch.Tensor = None):
        """
        Args:
            rgb:   (B, 3, H, W)  float32 in [0, 1]
            focal: (B,)           focal length in pixels
            cx:    (B,)           principal point x
            cy:    (B,)           principal point y
            depth: (B, 1, H, W)  optional; estimated if None

        Returns dict with keys:
            Stage-1: joints3d_s1, joints2d_s1, vertices_s1, beta_s1, theta_s1, trans_s1
            Stage-3: joints3d,    joints2d,    vertices,    beta,    theta,    trans
        """
        # ── Depth estimation ───────────────────────────────────────────────
        if depth is None:
            with torch.no_grad():
                depth = self.depth_estimator(rgb)

        # ── Stage 1 ─────────────────────────────────────────────────────────
        s1 = self.stage1(rgb, depth, focal, cx, cy)

        # ── Stage 2 ─────────────────────────────────────────────────────────
        delta_beta, delta_theta, delta_trans, delta_joints, uhr_tokens = self.stage2(
            s1["rgb_tokens"],
            s1["depth_tokens"],
            s1["joints2d_s1"],
            return_features=True,
        )

        # ── Stage 3 ─────────────────────────────────────────────────────────
        s3 = self.stage3(
            s1["beta_s1"],   s1["theta_s1"],  s1["trans_s1"],
            delta_beta,       delta_theta,      delta_trans,
            delta_joints,
            focal, cx, cy,
        )

        rgb_global = torch.cat([
            s1["rgb_tokens"][:, :3].mean(dim=1),
            s1["rgb_tokens"][:, 3:].mean(dim=1),
        ], dim=-1)
        depth_global = torch.cat([
            s1["depth_tokens"][:, :3].mean(dim=1),
            s1["depth_tokens"][:, 3:].mean(dim=1),
        ], dim=-1)
        coarse_global_delta = self.global_joint_refine(
            torch.cat([rgb_global, depth_global], dim=-1)
        ).view(rgb.shape[0], 21, 3)
        coarse_global_delta = coarse_global_delta - coarse_global_delta[:, :1, :]
        query_global_delta = self.joint_query_refine(
            s1["rgb_tokens"],
            s1["depth_tokens"],
        )
        global_delta = coarse_global_delta + query_global_delta

        s3["joints3d"] = s3["joints3d"] + global_delta
        s3["joints3d_uhr"] = s3["joints3d"]
        s3["joints3d_coarse_global"] = s3["joints3d_mano"] + coarse_global_delta
        s3["joints3d_query_global"] = s3["joints3d_mano"] + query_global_delta
        s3["joints3d_global_residual"] = global_delta
        s3["joints3d_query_residual"] = query_global_delta
        if self.use_direct_joint_decoder:
            direct_residual = self.direct_joint_decoder(uhr_tokens)
            s3["joints3d_mano_refined"] = s3["joints3d_uhr"]
            s3["joints3d_direct_residual"] = direct_residual
            s3["joints3d_direct"] = s3["joints3d_uhr"] + direct_residual
            s3["joints3d"] = s3["joints3d_direct"]
        s3["joints2d"] = focal_projection(s3["joints3d"], focal, cx, cy)

        return {
            # Stage-1 outputs (for intermediate supervision)
            "joints3d_s1": s1["joints3d_s1"],
            "joints2d_s1": s1["joints2d_s1"],
            "vertices_s1": s1["vertices_s1"],
            "beta_s1":     s1["beta_s1"],
            "theta_s1":    s1["theta_s1"],
            "trans_s1":    s1["trans_s1"],
            # Stage-3 final outputs
            **s3,
        }
