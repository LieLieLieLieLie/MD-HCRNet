import torch
import torch.nn.functional as F
import numpy as np


def focal_projection(joints_3d: torch.Tensor, focal: torch.Tensor,
                     cx: torch.Tensor, cy: torch.Tensor) -> torch.Tensor:
    """
    Project 3D joints to 2D using pinhole camera model.

    Args:
        joints_3d: (B, J, 3) joints in camera coordinates (already translated)
        focal:     (B,) or scalar focal length in pixels
        cx, cy:    (B,) or scalar principal point

    Returns:
        joints_2d: (B, J, 2)
    """
    # 强制 float32：BF16 精度不足以稳定做透视除法
    orig_dtype = joints_3d.dtype
    joints_3d = joints_3d.float()
    focal = focal.float()
    cx    = cx.float()
    cy    = cy.float()

    X = joints_3d[..., 0]
    Y = joints_3d[..., 1]
    Z = joints_3d[..., 2].clamp(min=0.1)   # 最小深度 10 cm，防止除以 0

    if focal.dim() == 1:
        focal = focal.view(-1, 1)
        cx = cx.view(-1, 1)
        cy = cy.view(-1, 1)

    u = focal * X / Z + cx
    v = focal * Y / Z + cy
    return torch.stack([u, v], dim=-1).to(orig_dtype)


def compute_similarity_transform(S1: np.ndarray, S2: np.ndarray) -> np.ndarray:
    """
    Align S1 to S2 via rigid+scale transform (Procrustes).
    Both are (J, 3). Returns aligned S1.
    """
    mu1 = S1.mean(axis=0)
    mu2 = S2.mean(axis=0)
    S1c = S1 - mu1
    S2c = S2 - mu2

    var1 = (S1c ** 2).sum() / len(S1c)
    K = S2c.T @ S1c / len(S1c)
    U, sigma, Vt = np.linalg.svd(K)

    # Reflection fix
    det = np.linalg.det(U @ Vt)
    D = np.diag([1, 1, np.sign(det)])

    R = U @ D @ Vt
    scale = (sigma * D.diagonal()).sum() / var1
    t = mu2 - scale * R @ mu1

    S1_aligned = scale * (S1 @ R.T) + t
    return S1_aligned


def batch_rodrigues(theta: torch.Tensor) -> torch.Tensor:
    """
    Convert axis-angle (B, 3) → rotation matrices (B, 3, 3).
    """
    angle = theta.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    axis = theta / angle

    cos = angle.cos().unsqueeze(-1)   # B,1,1
    sin = angle.sin().unsqueeze(-1)
    cos = cos.squeeze(-1)
    sin = sin.squeeze(-1)

    K = torch.zeros(theta.shape[0], 3, 3, device=theta.device, dtype=theta.dtype)
    K[:, 0, 1] = -axis[:, 2]
    K[:, 0, 2] =  axis[:, 1]
    K[:, 1, 0] =  axis[:, 2]
    K[:, 1, 2] = -axis[:, 0]
    K[:, 2, 0] = -axis[:, 1]
    K[:, 2, 1] =  axis[:, 0]

    I = torch.eye(3, device=theta.device, dtype=theta.dtype).unsqueeze(0)
    angle = angle.squeeze(-1)
    R = cos.view(-1, 1, 1) * I + \
        (1 - cos.view(-1, 1, 1)) * (axis.unsqueeze(-1) @ axis.unsqueeze(-2)) + \
        sin.view(-1, 1, 1) * K
    return R


def normalize_joints_2d(joints_2d: torch.Tensor, img_h: int, img_w: int) -> torch.Tensor:
    """Normalize pixel-space 2D joints to [-1, 1] for grid_sample."""
    norm = joints_2d.clone().float()
    norm[..., 0] = 2.0 * norm[..., 0] / img_w - 1.0
    norm[..., 1] = 2.0 * norm[..., 1] / img_h - 1.0
    return norm
