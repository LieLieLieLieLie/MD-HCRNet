from .i2l_baseline   import I2LBaseline
from .metro_baseline import HandOccNetBaseline
from .graph_hand     import GraphHand
from .att_hand       import LightAttHand
from .depth_hand     import DepthFusionHand

# Legacy simple baselines (kept for reference)
from .resnet_reg import ResNetRegressor
from .vit_reg    import ViTRegressor

# Registry: model_key → (class, description)
BASELINE_REGISTRY = {
    "i2l":         (I2LBaseline,        "I2L-Baseline   (ECCV 2020 style, position maps)"),
    "handoccnet":  (HandOccNetBaseline, "HandOccNet     (CVPR 2022 style, SE-attn + iterative MANO)"),
    "graphhand":   (GraphHand,          "GraphHand      (GCN on skeleton, journal 2021-22)"),
    "lightatt":    (LightAttHand,       "LightAttHand   (spatial attention, journal 2022-23)"),
    "depthfusion": (DepthFusionHand,    "DepthFusion    (early RGB+D fusion, journal 2022-23)"),
}

def build_baseline(key: str, cfg=None) -> "nn.Module":
    """
    Instantiate a baseline model by key.

    Args:
        key: one of 'i2l', 'handoccnet', 'graphhand', 'lightatt', 'depthfusion'
        cfg: optional config namespace (used for mano params etc.)
    Returns:
        nn.Module instance
    """
    import torch.nn as nn
    key = key.lower()
    if key not in BASELINE_REGISTRY:
        raise ValueError(f"Unknown baseline: {key!r}. "
                         f"Available: {list(BASELINE_REGISTRY.keys())}")
    cls, desc = BASELINE_REGISTRY[key]
    print(f"  Building baseline: {desc}")

    kwargs = {}
    if key == "handoccnet" and cfg is not None:
        if hasattr(cfg, "mano"):
            kwargs["mano_root"]      = getattr(cfg.mano, "model_path", "data")
            kwargs["is_rhand"]       = getattr(cfg.mano, "is_rhand", True)
            kwargs["use_pca"]        = getattr(cfg.mano, "use_pca", False)
            kwargs["flat_hand_mean"] = getattr(cfg.mano, "flat_hand_mean", False)

    return cls(**kwargs)


__all__ = [
    "I2LBaseline", "HandOccNetBaseline", "GraphHand", "LightAttHand",
    "DepthFusionHand", "ResNetRegressor", "ViTRegressor",
    "BASELINE_REGISTRY", "build_baseline",
]
