"""
Stage 3 — Metric-aware Reconstruction Head.

Adds Stage-2 residuals to Stage-1 parameters and runs MANO again to obtain
the final refined mesh and 3D joints.
"""
import torch
import torch.nn as nn

from utils.geometry import focal_projection


class Stage3(nn.Module):
    """
    Shared MANO decoder is passed in from the parent model to avoid
    duplicating weights.  Only the residual combination happens here.
    """

    def __init__(self, mano_layer: nn.Module):
        super().__init__()
        self.mano = mano_layer

    def forward(self,
                beta_s1:    torch.Tensor,
                theta_s1:   torch.Tensor,
                trans_s1:   torch.Tensor,
                delta_beta:  torch.Tensor,
                delta_theta: torch.Tensor,
                delta_trans: torch.Tensor,
                delta_joints: torch.Tensor,
                focal:       torch.Tensor,
                cx:          torch.Tensor,
                cy:          torch.Tensor):
        """
        Returns dict:
            beta:      (B, 10)
            theta:     (B, 48)
            trans:     (B, 3)
            vertices:  (B, 778, 3)
            joints3d:  (B, 21, 3)
            joints2d:  (B, 21, 2)
        """
        beta  = beta_s1  + delta_beta
        theta = theta_s1 + delta_theta
        trans = trans_s1 + delta_trans

        vertices, joints3d_local = self.mano(beta, theta)
        joints3d_mano = joints3d_local + trans.unsqueeze(1)
        joints3d = joints3d_mano + delta_joints
        joints2d = focal_projection(joints3d, focal, cx, cy)

        return dict(
            beta     = beta,
            theta    = theta,
            trans    = trans,
            vertices = vertices,
            joints3d_mano = joints3d_mano,
            joints3d_residual = delta_joints,
            joints3d = joints3d,
            joints2d = joints2d,
        )
