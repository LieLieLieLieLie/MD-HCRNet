"""
Structure-Aware Encoding (Stage 2, Step 2).

Processes the 21 sampled keypoint tokens in two complementary orderings:
  • Linear scan    — sequential index order 0 → 20
  • Kinematic scan — cross-finger depth-level order (wrist → tips):
        [0,
         1, 4, 7, 10, 13,   ← Level 1: MCP/proximal (Index,Middle,Pinky,Ring,Thumb_CMC)
         2, 5, 8, 11, 14,   ← Level 2: PIP/middle
         3, 6, 9, 12, 15,   ← Level 3: DIP/distal
         16,17,18,19,20]    ← Level 4: TIPs (Index,Middle,Pinky,Ring,Thumb)

Each ordering is processed by either Mamba (if mamba-ssm is installed) or a
bidirectional GRU (fallback).  The two outputs are fused via a linear layer.
"""
import torch
import torch.nn as nn


# ── Joint convention (MANO-21 output from MANOLayer) ──────────────────────────
#   0:  Wrist
#   1-3:  Index  (MCP, PIP, DIP)    4-6:  Middle (MCP, PIP, DIP)
#   7-9:  Pinky  (MCP, PIP, DIP)    10-12:Ring   (MCP, PIP, DIP)
#   13-15:Thumb  (CMC, MCP, IP)
#   16: Index TIP   17: Middle TIP   18: Pinky TIP   19: Ring TIP   20: Thumb TIP

# ── Kinematic scan: wrist → level-by-level across all 5 fingers ───────────────
#   Level 0: Wrist
#   Level 1: MCP/proximal — Index[1], Middle[4], Pinky[7], Ring[10], Thumb_CMC[13]
#   Level 2: PIP/middle   — Index[2], Middle[5], Pinky[8], Ring[11], Thumb_MCP[14]
#   Level 3: DIP/distal   — Index[3], Middle[6], Pinky[9], Ring[12], Thumb_IP [15]
#   Level 4: TIPs         — Index[16],Middle[17],Pinky[18],Ring[19], Thumb[20]
KINEMATIC_ORDER = [0,
                   1, 4, 7, 10, 13,
                   2, 5, 8, 11, 14,
                   3, 6, 9, 12, 15,
                   16, 17, 18, 19, 20]
LINEAR_ORDER    = list(range(21))


def _try_build_mamba(embed_dim: int):
    try:
        from mamba_ssm import Mamba
        return Mamba(d_model=embed_dim, d_state=16, d_conv=4, expand=2)
    except ImportError:
        return None


class _SeqModel(nn.Module):
    """Either Mamba or BiGRU, exposing the same (B,N,C) → (B,N,C) interface."""

    def __init__(self, embed_dim: int, use_mamba: bool):
        super().__init__()
        mamba = _try_build_mamba(embed_dim) if use_mamba else None
        if mamba is not None:
            self.net  = mamba
            self.proj = nn.Identity()
            self._is_mamba = True
        else:
            self.net  = nn.GRU(embed_dim, embed_dim, batch_first=True,
                               bidirectional=True)
            self.proj = nn.Linear(embed_dim * 2, embed_dim)
            self._is_mamba = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._is_mamba:
            return self.net(x)
        out, _ = self.net(x)
        return self.proj(out)


class StructureAwareEncoding(nn.Module):
    def __init__(self, embed_dim: int, use_mamba: bool = False):
        super().__init__()
        self.register_buffer("linear_idx",
                             torch.tensor(LINEAR_ORDER, dtype=torch.long))
        self.register_buffer("kinematic_idx",
                             torch.tensor(KINEMATIC_ORDER, dtype=torch.long))
        self.register_buffer("kinematic_inv",
                             torch.argsort(torch.tensor(KINEMATIC_ORDER, dtype=torch.long)))

        self.linear_seq   = _SeqModel(embed_dim, use_mamba)
        self.kinematic_seq = _SeqModel(embed_dim, use_mamba)

        self.fusion = nn.Linear(embed_dim * 2, embed_dim)
        self.norm   = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 21, C)  — 21 joint tokens in MANO-21 order
        Returns:
            (B, 21, C)
        """
        J = x.shape[1]  # 21
        # ── linear scan ────────────────────────────────────────────────────
        out_lin = self.linear_seq(x)                                 # B,J,C

        # ── kinematic scan (reorder → process → restore) ───────────────────
        x_kin = x[:, self.kinematic_idx, :]                         # B,J,C
        out_kin = self.kinematic_seq(x_kin)
        out_kin = out_kin[:, self.kinematic_inv, :]                  # back to 0-20 order

        # ── fuse and residual ───────────────────────────────────────────────
        fused = self.fusion(torch.cat([out_lin, out_kin], dim=-1))  # B,21,C
        return self.norm(x + fused)
