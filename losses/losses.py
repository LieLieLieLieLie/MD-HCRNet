import torch
import torch.nn as nn
import torch.nn.functional as F


class MDHCRNetLoss(nn.Module):
    """
    joints3d loss: 单位 metres，量级 ~0.01–0.5
    joints2d loss: GT 是像素坐标，量级 ~0–256
    → 用图像尺寸归一化 joints2d，使两者量级一致，避免 2D 项淹没 3D 梯度。
    归一化后 joints2d 范围 [0,1]，与 joints3d 的 metres 量级相当。
    """

    def __init__(self, cfg):
        super().__init__()
        lc = cfg.loss
        self.lambda_joints3d       = lc.lambda_joints3d
        self.lambda_joints2d       = lc.lambda_joints2d
        self.lambda_vertices       = lc.lambda_vertices
        self.lambda_beta           = lc.lambda_beta
        self.lambda_theta          = lc.lambda_theta
        self.lambda_joint_residual = getattr(lc, "lambda_joint_residual", 0.01)
        self.lambda_aux_refine     = getattr(lc, "lambda_aux_refine", 0.0)
        self.lambda_uhr_joints3d   = getattr(lc, "lambda_uhr_joints3d", 0.5)
        self.lambda_depth_joints3d = lc.lambda_depth_joints3d
        self.lambda_depth_joints2d = lc.lambda_depth_joints2d
        # 图像尺寸，用于归一化像素坐标 → [0,1]
        self.img_w = float(cfg.data.img_width)
        self.img_h = float(cfg.data.img_height)

    def _norm2d(self, uv: torch.Tensor) -> torch.Tensor:
        """像素坐标 (u,v) → [0,1] 归一化：u/W, v/H"""
        scale = uv.new_tensor([self.img_w, self.img_h])  # (2,)
        return uv / scale

    @staticmethod
    def _root_rel(j: torch.Tensor) -> torch.Tensor:
        """相机坐标系 → root-relative（减去 joint-0 手腕），与 MPJPE 评估指标对齐"""
        return j - j[:, :1, :]

    def forward(self, pred: dict, gt: dict) -> dict:
        losses = {}

        # ── Stage 1 supervision ─────────────────────────────────────────────
        if "joints3d_s1" in pred and "joints3d" in gt:
            # root-relative 3D loss：与 MPJPE 评估指标一致
            losses["s1_joints3d"] = (
                F.l1_loss(self._root_rel(pred["joints3d_s1"]),
                          self._root_rel(gt["joints3d"]))
                * self.lambda_depth_joints3d
            )
        if "joints2d_s1" in pred and "joints2d" in gt:
            gt2d = self._norm2d(gt["joints2d"][..., :2])
            losses["s1_joints2d"] = (
                F.l1_loss(self._norm2d(pred["joints2d_s1"]), gt2d)
                * self.lambda_depth_joints2d
            )

        # ── Stage 3 (final) supervision ─────────────────────────────────────
        if "joints3d" in pred and "joints3d" in gt:
            # root-relative 3D loss：直接优化 MPJPE 所衡量的量
            losses["joints3d"] = (
                F.l1_loss(self._root_rel(pred["joints3d"]),
                          self._root_rel(gt["joints3d"]))
                * self.lambda_joints3d
            )
        if "joints2d" in pred and "joints2d" in gt:
            gt2d = self._norm2d(gt["joints2d"][..., :2])
            losses["joints2d"] = (
                F.l1_loss(self._norm2d(pred["joints2d"]), gt2d)
                * self.lambda_joints2d
            )
        if "vertices" in pred and "vertices" in gt:
            losses["vertices"] = (
                F.l1_loss(pred["vertices"], gt["vertices"])
                * self.lambda_vertices
            )
        if self.lambda_aux_refine > 0 and "joints3d" in gt:
            aux_terms = []
            for key in ("joints3d_coarse_global", "joints3d_query_global"):
                if key in pred:
                    aux_terms.append(
                        F.l1_loss(self._root_rel(pred[key]),
                                  self._root_rel(gt["joints3d"]))
                    )
            if aux_terms:
                losses["aux_refine_joints3d"] = (
                    sum(aux_terms) / len(aux_terms) * self.lambda_aux_refine
                )
        if self.lambda_uhr_joints3d > 0 and "joints3d_uhr" in pred and "joints3d" in gt:
            losses["uhr_joints3d"] = (
                F.l1_loss(self._root_rel(pred["joints3d_uhr"]),
                          self._root_rel(gt["joints3d"]))
                * self.lambda_uhr_joints3d
            )

        # ── MANO parameter supervision / regularisation ─────────────────────
        # FreiHAND provides MANO beta/theta. Use it when available; otherwise
        # fall back to a mild zero prior for datasets without MANO labels.
        if "beta" in pred:
            if "beta" in gt:
                losses["mano_beta"] = F.smooth_l1_loss(pred["beta"], gt["beta"]) * self.lambda_beta
                if "beta_s1" in pred:
                    losses["s1_mano_beta"] = F.smooth_l1_loss(pred["beta_s1"], gt["beta"]) * self.lambda_beta
            else:
                losses["reg_beta"] = (pred["beta"] ** 2).mean() * self.lambda_beta
        if "theta" in pred:
            if "theta" in gt:
                losses["mano_theta"] = F.smooth_l1_loss(pred["theta"], gt["theta"]) * self.lambda_theta
                if "theta_s1" in pred:
                    losses["s1_mano_theta"] = F.smooth_l1_loss(pred["theta_s1"], gt["theta"]) * self.lambda_theta
            else:
                losses["reg_theta"] = (pred["theta"] ** 2).mean() * self.lambda_theta
        residual_regs = []
        for key in ("joints3d_residual", "joints3d_global_residual", "joints3d_direct_residual"):
            if key in pred:
                residual_regs.append(pred[key].abs().mean())
        if residual_regs:
            losses["reg_joint_residual"] = (
                sum(residual_regs) / len(residual_regs) * self.lambda_joint_residual
            )

        losses["total"] = sum(losses.values())
        return losses
