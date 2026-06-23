"""
Deformable Cross-Attention (Stage 2, Step 3).

Two cross-attention streams:
  RGB  → Depth: RGB keypoint tokens (Q) × Depth keypoint tokens (K,V)
  Depth → RGB:  Depth keypoint tokens (Q) × RGB keypoint tokens (K,V)

Each Q sequence also includes the three special tokens [β, θ, t] from Stage 1.
After cross-attending, the two refined sequences are fused element-wise and
three residual heads extract Δβ, Δθ, Δt from the first 3 tokens.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (embed_dim // num_heads) ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(drop)
        self.norm_q  = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)

    def forward(self, query: torch.Tensor, key_val: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query:   (B, Nq, C)
            key_val: (B, Nk, C)
        Returns:
            (B, Nq, C) with residual
        """
        B, Nq, C = query.shape
        Nk = key_val.shape[1]
        h  = self.num_heads
        d  = C // h

        q = self.q_proj(self.norm_q(query)).reshape(B, Nq, h, d).transpose(1, 2)
        k = self.k_proj(self.norm_kv(key_val)).reshape(B, Nk, h, d).transpose(1, 2)
        v = self.v_proj(self.norm_kv(key_val)).reshape(B, Nk, h, d).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))

        out = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
        return query + self.out_proj(out)


class DeformableCrossAttentionModule(nn.Module):
    """
    Full Stage-2 Step-3 module.

    Inputs (per branch):
        special_tokens: (B, 3, C)   — [β, θ, t] from Stage 1
        kp_tokens:      (B, 21, C)  — structure-encoded keypoint tokens

    The query sequence is cat([special, kp]) = (B, 24, C).
    K,V come from the other branch's kp_tokens (B, 21, C).
    """

    def __init__(self, embed_dim: int, num_heads: int, drop: float = 0.0):
        super().__init__()
        # RGB queries depth
        self.rgb2depth = CrossAttention(embed_dim, num_heads, drop)
        # Depth queries RGB
        self.depth2rgb = CrossAttention(embed_dim, num_heads, drop)

        # Fusion: combine two (B,24,C) refined sequences → (B,24,C)
        self.fusion_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
        )
        self.norm = nn.LayerNorm(embed_dim)

        # Residual heads – extract from positions 0 (beta), 1 (theta), 2 (t)
        self.beta_head  = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 10))
        self.theta_head = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 48))
        self.trans_head = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 3))
        # Joint residual head – one residual vector for each sampled hand joint.
        # This keeps MANO mesh prediction intact while allowing the final joint
        # estimate to use Stage-2 hand-centric evidence directly.
        self.joint_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 3),
        )
        for head in (self.beta_head, self.theta_head, self.trans_head, self.joint_head):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def forward(self,
                rgb_special:   torch.Tensor,
                rgb_kp:        torch.Tensor,
                depth_special: torch.Tensor,
                depth_kp:      torch.Tensor,
                return_features: bool = False):
        """
        Args:
            rgb_special:   (B, 3, C)
            rgb_kp:        (B, 21, C)
            depth_special: (B, 3, C)
            depth_kp:      (B, 21, C)

        Returns:
            delta_beta:  (B, 10)
            delta_theta: (B, 48)
            delta_trans: (B, 3)
            delta_joints: (B, 21, 3)
            fused:        (B, 24, C), only when return_features=True
        """
        # Build query sequences
        rgb_Q   = torch.cat([rgb_special,   rgb_kp],   dim=1)   # B,24,C
        depth_Q = torch.cat([depth_special, depth_kp], dim=1)   # B,24,C

        # Cross-attend
        rgb_refined   = self.rgb2depth(rgb_Q,   depth_kp)   # B,24,C  (RGB queries Depth)
        depth_refined = self.depth2rgb(depth_Q, rgb_kp)     # B,24,C  (Depth queries RGB)

        # Fuse
        fused = self.fusion_proj(
            torch.cat([rgb_refined, depth_refined], dim=-1))    # B,24,C
        fused = self.norm(fused)

        # Residual predictions from the three special-token positions
        delta_beta  = self.beta_head(fused[:, 0])    # B,10
        delta_theta = self.theta_head(fused[:, 1])   # B,48
        delta_trans = self.trans_head(fused[:, 2])   # B,3
        delta_joints = self.joint_head(fused[:, 3:])  # B,21,3

        if return_features:
            return delta_beta, delta_theta, delta_trans, delta_joints, fused
        return delta_beta, delta_theta, delta_trans, delta_joints
