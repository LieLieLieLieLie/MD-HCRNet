import numpy as np
from .geometry import compute_similarity_transform


def MPJPE(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Mean Per-Joint Position Error (mm).
    pred, gt: (N, J, 3)
    """
    return np.sqrt(((pred - gt) ** 2).sum(axis=-1)).mean()


def PA_MPJPE(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Procrustes-aligned MPJPE (mm).
    pred, gt: (N, J, 3)
    """
    errors = []
    for p, g in zip(pred, gt):
        p_aligned = compute_similarity_transform(p, g)
        errors.append(np.sqrt(((p_aligned - g) ** 2).sum(axis=-1)).mean())
    return float(np.mean(errors))


def F_score(pred_verts: np.ndarray, gt_verts: np.ndarray,
            threshold: float = 5.0) -> float:
    """
    F-score at a given distance threshold (mm).
    pred_verts, gt_verts: (N, V, 3)
    """
    from scipy.spatial import cKDTree

    scores = []
    for pv, gv in zip(pred_verts, gt_verts):
        tree_pred = cKDTree(pv)
        tree_gt   = cKDTree(gv)

        dist_p2g, _ = tree_gt.query(pv)
        dist_g2p, _ = tree_pred.query(gv)

        precision = (dist_p2g < threshold).mean()
        recall    = (dist_g2p < threshold).mean()

        if precision + recall < 1e-8:
            scores.append(0.0)
        else:
            scores.append(2 * precision * recall / (precision + recall))
    return float(np.mean(scores))
