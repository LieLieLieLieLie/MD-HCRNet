"""
Thin wrapper around smplx.MANO.

Place MANO_RIGHT.pkl inside data/mano/ (register at https://mano.is.tue.mpg.de/).

smplx MANO outputs 16 kinematic-chain joints:
  0:Wrist
  1-3: Index  (MCP, PIP, DIP)
  4-6: Middle (MCP, PIP, DIP)
  7-9: Pinky  (MCP, PIP, DIP)
  10-12: Ring (MCP, PIP, DIP)
  13-15: Thumb (CMC, MCP, IP)

We append 5 fingertip positions extracted from specific mesh vertices to obtain
the standard 21-joint hand skeleton used by FreiHAND / RHD / all SOTA methods:
  16: Index TIP  (vertex 317)
  17: Middle TIP (vertex 445)
  18: Pinky TIP  (vertex 673)
  19: Ring TIP   (vertex 556)
  20: Thumb TIP  (vertex 745)
"""
import torch
import torch.nn as nn
import numpy as np
import warnings


# chumpy is an old MANO dependency and still imports removed numpy aliases.
# Patch them before smplx/chumpy is imported so MANO does not silently fall back
# to zero vertices/joints on modern numpy.
for _alias, _value in {
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "object": object,
    "str": str,
    "unicode": str,
}.items():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        missing = not hasattr(np, _alias)
    if missing:
        setattr(np, _alias, _value)


# Standard MANO right-hand fingertip vertex indices (same as original MANO paper)
# Order corresponds to MANO21 joint indices 16-20:
#   Index TIP, Middle TIP, Pinky TIP, Ring TIP, Thumb TIP
FINGERTIP_VERTICES = [317, 445, 673, 556, 745]


class MANOLayer(nn.Module):
    """
    Wraps smplx MANO: (beta B×10, theta B×48) → (vertices B×778×3, joints B×21×3)

    theta layout (48-dim):
        [0:3]  global_orient  (wrist rotation, axis-angle)
        [3:48] hand_pose      (15 joints × 3, axis-angle)

    Output joint order (21 joints):
        0:Wrist
        1-3:   Index  (MCP, PIP, DIP)
        4-6:   Middle (MCP, PIP, DIP)
        7-9:   Pinky  (MCP, PIP, DIP)
        10-12: Ring   (MCP, PIP, DIP)
        13-15: Thumb  (CMC, MCP, IP)
        16:    Index  TIP
        17:    Middle TIP
        18:    Pinky  TIP
        19:    Ring   TIP
        20:    Thumb  TIP
    """

    NUM_JOINTS = 21

    def __init__(self, mano_root: str, is_rhand: bool = True,
                 use_pca: bool = False, flat_hand_mean: bool = False):
        super().__init__()
        # Register fingertip vertex indices as a buffer so they move with the module
        self.register_buffer(
            "tip_idx",
            torch.tensor(FINGERTIP_VERTICES, dtype=torch.long)
        )
        try:
            import smplx
            self.mano = smplx.create(
                model_path=mano_root,
                model_type="mano",
                is_rhand=is_rhand,
                use_pca=use_pca,
                num_pca_comps=45,
                flat_hand_mean=flat_hand_mean,
                batch_size=1,
            )
            self._has_smplx = True
            print(f"[MANOLayer] Loaded MANO ({'right' if is_rhand else 'left'} hand), "
                  f"21 joints (16 kinematic + 5 fingertips from mesh vertices).")
        except Exception as e:
            print(f"[MANOLayer] smplx not available or model not found: {e}")
            print("[MANOLayer] Using identity fallback — outputs will be zeros.")
            self._has_smplx = False

    def forward(self, beta: torch.Tensor, theta: torch.Tensor):
        """
        Args:
            beta:  (B, 10)
            theta: (B, 48)  global_orient[:3] + hand_pose[3:]

        Returns:
            vertices: (B, 778, 3)
            joints:   (B, 21,  3)  — 16 kinematic + 5 fingertips
        """
        B = beta.shape[0]
        if not self._has_smplx:
            return (torch.zeros(B, 778, 3, device=beta.device),
                    torch.zeros(B, 21,  3, device=beta.device))

        # smplx 内部使用 float32；强制转换并 clamp 到物理合理范围
        with torch.cuda.amp.autocast(enabled=False):
            b = beta.float().clamp(-5.0, 5.0)
            t = theta.float().clamp(-3.14159, 3.14159)
            output = self.mano(
                betas=b,
                global_orient=t[:, :3],
                hand_pose=t[:, 3:],
                return_verts=True,
            )
        verts       = output.vertices.to(beta.dtype)           # (B, 778, 3)
        joints16    = output.joints[:, :16, :].to(beta.dtype)  # (B,  16, 3)

        # Append 5 fingertip positions from mesh vertices → (B, 21, 3)
        fingertips  = verts[:, self.tip_idx, :]                # (B,   5, 3)
        joints21    = torch.cat([joints16, fingertips], dim=1) # (B,  21, 3)

        return verts, joints21
