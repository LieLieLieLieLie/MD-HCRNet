"""
Knowledge-Guided Token Sampling (Stage 2, Step 1).

Uses Stage-1 2D joints as anchor points.  For each anchor, four learnable
offset vectors are predicted from the anchor's local feature, giving 5 sample
locations per joint.  Bilinear interpolation on the spatial feature map
aggregates them into one token per joint.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class KnowledgeGuidedSampling(nn.Module):
    """
    Args:
        embed_dim: feature channel dimension C
        img_h, img_w: input image spatial size
        patch_size:   patch size used by ViT
        num_offsets:  number of offset directions (default 4)
    """

    def __init__(self, embed_dim: int, img_h: int, img_w: int,
                 patch_size: int, num_offsets: int = 4):
        super().__init__()
        self.img_h = img_h
        self.img_w = img_w
        self.Ph = img_h // patch_size
        self.Pw = img_w // patch_size
        self.num_offsets = num_offsets

        # Offset predictor: for each of 21 anchors predict (num_offsets × 2) offsets
        # in normalised [-1,1] coordinate space
        self.offset_fc = nn.Linear(embed_dim, num_offsets * 2)

        # Attention weights over (1 anchor + num_offsets) samples
        self.weight_fc = nn.Linear(embed_dim, 1 + num_offsets)

        nn.init.zeros_(self.offset_fc.weight)
        nn.init.zeros_(self.offset_fc.bias)

    def _sample(self, feat_map: torch.Tensor,
                 grid: torch.Tensor) -> torch.Tensor:
        """
        Bilinear sample from feat_map at grid positions.

        Args:
            feat_map: (B, C, Ph, Pw)
            grid:     (B, Q, 2)  — x,y in [-1,1]
        Returns:
            sampled: (B, Q, C)
        """
        B, C, Ph, Pw = feat_map.shape
        Q = grid.shape[1]
        g = grid.view(B, Q, 1, 2)                                       # B,Q,1,2
        out = F.grid_sample(feat_map, g, mode="bilinear",
                             padding_mode="border", align_corners=True)  # B,C,Q,1
        return out.squeeze(-1).permute(0, 2, 1)                          # B,Q,C

    def forward(self, patch_tokens: torch.Tensor,
                joints_2d: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_tokens: (B, N, C)   — patch tokens from ViT (no special tokens)
            joints_2d:    (B, 21, 2)  — 2-D joints in pixel coords (x, y)

        Returns:
            kp_tokens: (B, 21, C)
        """
        B, N, C = patch_tokens.shape
        J = joints_2d.shape[1]  # 21
        device = patch_tokens.device

        # ── reshape to spatial map ──────────────────────────────────────────
        feat_map = patch_tokens.reshape(B, self.Ph, self.Pw, C).permute(0, 3, 1, 2)
        # feat_map: B, C, Ph, Pw

        # ── normalise anchor coords to [-1, 1] ─────────────────────────────
        # joints_2d in pixel space; feature map spans the same spatial extent
        # We normalise using the full image size (Ph/Pw cells)
        anchors = joints_2d.float()
        anchors_norm = anchors.clone()
        # x/y are pixel coordinates in the resized input image, not patch-grid indices.
        anchors_norm[..., 0] = (anchors_norm[..., 0] / max(float(self.img_w - 1), 1.0)) * 2.0 - 1.0
        anchors_norm[..., 1] = (anchors_norm[..., 1] / max(float(self.img_h - 1), 1.0)) * 2.0 - 1.0
        anchors_norm = anchors_norm.clamp(-1.0, 1.0)               # B,21,2

        # ── sample at anchor positions ──────────────────────────────────────
        anchor_feats = self._sample(feat_map, anchors_norm)         # B,21,C

        # ── predict offsets from anchor features ───────────────────────────
        offsets = self.offset_fc(anchor_feats)                      # B,21,4*2
        offsets = offsets.reshape(B, J, self.num_offsets, 2).tanh() * 0.1
        # scale: offsets are small perturbations in normalised space

        # ── expand anchors and add offsets ─────────────────────────────────
        anchors_exp = anchors_norm.unsqueeze(2).expand(-1, -1, self.num_offsets, -1)
        sample_pts  = (anchors_exp + offsets).clamp(-1.0, 1.0)     # B,21,4,2

        # ── sample at offset positions ──────────────────────────────────────
        sample_pts_flat = sample_pts.reshape(B, J * self.num_offsets, 2)
        offset_feats    = self._sample(feat_map, sample_pts_flat)   # B,21*4,C
        offset_feats    = offset_feats.reshape(B, J, self.num_offsets, C)

        # ── weighted aggregation over (anchor + 4 offsets) ──────────────────
        all_feats = torch.cat([anchor_feats.unsqueeze(2), offset_feats], dim=2)
        # all_feats: B,21,5,C
        weights = self.weight_fc(anchor_feats).softmax(dim=-1)      # B,21,5
        kp_tokens = (all_feats * weights.unsqueeze(-1)).sum(dim=2)  # B,21,C

        return kp_tokens
