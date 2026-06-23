"""
Stage 2 — Unified Hand-centric Reasoning (Core Module).

Combines:
  (1) Knowledge-Guided Token Sampling
  (2) Structure-Aware Encoding (Mamba / BiGRU)
  (3) Deformable Cross-Attention + Residual Heads
"""
import torch
import torch.nn as nn

from .stage2.knowledge_sampling      import KnowledgeGuidedSampling
from .stage2.structure_encoding      import StructureAwareEncoding
from .stage2.deformable_cross_attention import DeformableCrossAttentionModule


class Stage2(nn.Module):
    def __init__(self, cfg, embed_dim: int):
        super().__init__()
        H, W = cfg.img_height, cfg.img_width
        P    = cfg.patch_size
        use_mamba = getattr(cfg, "use_mamba", False)
        num_heads = cfg.num_heads

        # ── (1) Knowledge-guided sampling — one per branch ─────────────────
        self.rgb_kg_sample   = KnowledgeGuidedSampling(embed_dim, H, W, P)
        self.depth_kg_sample = KnowledgeGuidedSampling(embed_dim, H, W, P)

        # ── (2) Structure-aware encoding — one per branch ──────────────────
        self.rgb_struct   = StructureAwareEncoding(embed_dim, use_mamba)
        self.depth_struct = StructureAwareEncoding(embed_dim, use_mamba)

        # ── (3) Deformable cross-attention ─────────────────────────────────
        self.cross_attn = DeformableCrossAttentionModule(embed_dim, num_heads)

    def forward(self,
                rgb_tokens:   torch.Tensor,
                depth_tokens: torch.Tensor,
                joints2d_s1:  torch.Tensor,
                return_features: bool = False):
        """
        Args:
            rgb_tokens:   (B, N+3, C) — full output of Stage-1 RGB branch
            depth_tokens: (B, N+3, C) — full output of Stage-1 Depth branch
            joints2d_s1:  (B, 21, 2)  — pixel-space 2D joints from Stage 1

        Returns:
            delta_beta:  (B, 10)
            delta_theta: (B, 48)
            delta_trans: (B, 3)
            delta_joints: (B, 21, 3)
            uhr_tokens:  (B, 24, C), only when return_features=True
        """
        # ── Split special tokens from patch tokens ─────────────────────────
        rgb_special   = rgb_tokens[:,   :3, :]    # B,3,C
        depth_special = depth_tokens[:, :3, :]    # B,3,C
        rgb_patches   = rgb_tokens[:,   3:, :]    # B,N,C
        depth_patches = depth_tokens[:, 3:, :]    # B,N,C

        # ── (1) Knowledge-guided sampling ──────────────────────────────────
        rgb_kp   = self.rgb_kg_sample(rgb_patches,   joints2d_s1)   # B,21,C
        depth_kp = self.depth_kg_sample(depth_patches, joints2d_s1) # B,21,C

        # ── (2) Structure-aware encoding ───────────────────────────────────
        rgb_enc   = self.rgb_struct(rgb_kp)     # B,21,C
        depth_enc = self.depth_struct(depth_kp) # B,21,C

        # ── (3) Deformable cross-attention → residuals ─────────────────────
        out = self.cross_attn(
            rgb_special, rgb_enc, depth_special, depth_enc,
            return_features=return_features)

        if return_features:
            delta_beta, delta_theta, delta_trans, delta_joints, uhr_tokens = out
            return delta_beta, delta_theta, delta_trans, delta_joints, uhr_tokens

        delta_beta, delta_theta, delta_trans, delta_joints = out

        return delta_beta, delta_theta, delta_trans, delta_joints
