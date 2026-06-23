"""
Unified evaluation script 鈥?evaluate MD-HCRNet + all baselines together.

瀵规墍鏈夋ā鍨嬪湪鐩稿悓楠岃瘉闆嗕笂璇勪及锛岀敓鎴愶細
  鈶?outputs/freihand/tables/main_comparison.xlsx    鈥?MPJPE, PA-MPJPE, F@5mm, F@15mm
  鈶?outputs/freihand/tables/per_joint_error.xlsx    鈥?鍚勫叧鑺?MPJPE
  鈶?outputs/freihand/tables/pck_data.xlsx           鈥?PCK@threshold 鏁版嵁
  鈶?outputs/freihand/tables/efficiency.xlsx         鈥?Params, FLOPs, FPS
  鈶?outputs/freihand/figures/*.pdf                  鈥?瀵瑰簲姣忓紶琛ㄧ殑鍙鍖栧浘

Usage锛堝湪 code/ 鐩綍涓嬶級:
    python eval_all.py --config config/freihand.yaml

    # 鍙瘎浼伴儴鍒嗘ā鍨嬶紙榛樿璇勪及鎵€鏈夋湁 best.pth 鐨勬ā鍨嬶級
    python eval_all.py --config config/freihand.yaml --models md_hcrnet i2l handoccnet

姣忔杩愯锛?  - 浠?model_key 涓轰富閿紝杩藉姞/瑕嗙洊鍚勮〃鏍肩殑琛?  - 鍙湁 models/ 涓嬪瓨鍦ㄥ搴?best.pth 鐨勬ā鍨嬫墠浼氳璇勪及
"""
import argparse
import os
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

import yaml
from types import SimpleNamespace

from data             import build_dataloaders
from utils            import MPJPE, PA_MPJPE, F_score
from utils.table_logger import TableLogger
from baselines        import build_baseline, BASELINE_REGISTRY
from models           import MDHCRNet


# 鈹€鈹€ Colours 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

_PALETTE = [
    "#FFAA53", "#50CC55", "#9933FF", "#3399FF",
    "#6666FF", "#FF6666", "#8C8C8C",
]

_MODEL_COLOR = {}  # filled at runtime
_DATASET_TITLE = "FreiHAND"

_MODEL_COLORS = {
    "DepthFusion":   "#FFAA53",
    "GraphHand":     "#50CC55",
    "HandOccNet":    "#9933FF",
    "I2L":           "#3399FF",
    "LightAttHand":  "#6666FF",
    "MD-HCR":        "#FF6666",
}

_HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "error_heatmap", ["#FFFFFF", "#FF4F4F"], N=256
)

# MANO-21 joint names (display)
JOINT_NAMES = [
    "Wrist",
    "Idx_MCP", "Idx_PIP", "Idx_DIP",
    "Mid_MCP", "Mid_PIP", "Mid_DIP",
    "Pky_MCP", "Pky_PIP", "Pky_DIP",
    "Rng_MCP", "Rng_PIP", "Rng_DIP",
    "Thb_CMC", "Thb_MCP", "Thb_IP",
    "Idx_TIP", "Mid_TIP", "Pky_TIP", "Rng_TIP", "Thb_TIP",
]


# 鈹€鈹€ Config 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

def _coerce(v):
    if not isinstance(v, str): return v
    try:    return int(v)
    except: pass
    try:    return float(v)
    except: return v

def _dict_to_ns(d):
    ns = SimpleNamespace()
    for k, v in d.items():
        if isinstance(v, dict):   setattr(ns, k, _dict_to_ns(v))
        elif isinstance(v, list): setattr(ns, k, [_coerce(i) for i in v])
        else:                     setattr(ns, k, _coerce(v))
    return ns

def load_config(path: str):
    with open(path, encoding="utf-8") as f:
        return _dict_to_ns(yaml.safe_load(f))


# 鈹€鈹€ Metric helpers 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

def root_rel(j):
    return j - j[:, :1, :]

def pck_curve(pred_r, gt_r, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0, 80, 300)
    errs = np.linalg.norm(pred_r - gt_r, axis=-1).mean(axis=1)  # (N,)
    pck  = [(errs <= t).mean() * 100 for t in thresholds]
    auc  = float(np.trapz(pck, thresholds) / thresholds[-1])
    return thresholds, np.array(pck), auc


def bootstrap_ci(per_sample_errs, n_boot=2000, alpha=0.05):
    """95% bootstrap confidence interval for the mean."""
    n = len(per_sample_errs)
    boot = np.array([per_sample_errs[np.random.randint(0, n, n)].mean()
                     for _ in range(n_boot)])
    lo = np.percentile(boot, 100 * alpha / 2)
    hi = np.percentile(boot, 100 * (1 - alpha / 2))
    return lo, hi


# 鈹€鈹€ Load model 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

def load_model(model_key: str, cfg, device: torch.device, models_dir: str):
    # MD-HCRNet 鐨?train.py 鏃х増鏈繚瀛樹负 best.pth锛屾柊鐗堜繚瀛樹负 md_hcrnet_best.pth
    ckpt_path = os.path.join(models_dir, f"{model_key}_best.pth")
    if not os.path.exists(ckpt_path) and model_key == "md_hcrnet":
        ckpt_path = os.path.join(models_dir, "best.pth")
    if not os.path.exists(ckpt_path):
        print(f"  [skip] {model_key}: no checkpoint at {ckpt_path}")
        return None, None

    if model_key == "md_hcrnet":
        model = MDHCRNet(cfg).to(device)
    else:
        model = build_baseline(model_key, cfg).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    model_name = getattr(model, "MODEL_NAME", model_key)
    epoch      = ckpt.get("epoch", "?")
    print(f"  Loaded: {model_name:<22s}  (epoch {epoch})  鈫?{ckpt_path}")
    return model, model_name


# 鈹€鈹€ Inference 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    pred_j3d, gt_j3d = [], []
    pred_verts, gt_verts = [], []

    for batch in tqdm(loader, desc="   Inference", leave=False):
        rgb   = batch["image"].to(device)
        focal = batch["focal"].to(device)
        cx    = batch["cx"].to(device)
        cy    = batch["cy"].to(device)
        depth = batch["depth"].to(device) if "depth" in batch else None

        out = model(rgb, focal, cx, cy, depth)

        pred_j3d.append(out["joints3d"].float().cpu().numpy())
        gt_j3d.append(batch["joints3d"].numpy())

        if "vertices" in out and out["vertices"] is not None \
                and "vertices" in batch:
            pred_verts.append(out["vertices"].float().cpu().numpy())
            gt_verts.append(batch["vertices"].numpy())

    pred = np.concatenate(pred_j3d, 0) * 1000  # 鈫?mm
    gt   = np.concatenate(gt_j3d,   0) * 1000

    verts_pred = (np.concatenate(pred_verts, 0) * 1000) if pred_verts else None
    verts_gt   = (np.concatenate(gt_verts,   0) * 1000) if gt_verts  else None

    pred_r = pred - pred[:, :1, :]
    gt_r   = gt   - gt[:, :1, :]
    per_sample_mpjpe = np.linalg.norm(pred_r - gt_r, axis=-1).mean(axis=1)  # (N,)

    return pred, gt, verts_pred, verts_gt, per_sample_mpjpe


# 鈹€鈹€ Efficiency measurement 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

def measure_efficiency(model, device, img_h=256, img_w=192, n_runs=100):
    """Return (n_params_M, flops_G, fps).

    Handles both MDHCRNet (needs focal/cx/cy) and baselines (rgb-only).
    """
    import inspect
    dummy_rgb   = torch.randn(1, 3, img_h, img_w, device=device)
    dummy_focal = torch.tensor([512.0], device=device)
    dummy_cx    = torch.tensor([float(img_w) / 2], device=device)
    dummy_cy    = torch.tensor([float(img_h) / 2], device=device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6

    # Detect whether model.forward expects focal/cx/cy
    sig          = inspect.signature(model.forward)
    needs_camera = "focal" in sig.parameters

    def _call():
        if needs_camera:
            return model(dummy_rgb, dummy_focal, dummy_cx, dummy_cy)
        else:
            return model(dummy_rgb)

    # FLOPs via thop (optional)
    flops_g = None
    try:
        from thop import profile as thop_profile
        if needs_camera:
            _inputs = (dummy_rgb, dummy_focal, dummy_cx, dummy_cy)
        else:
            _inputs = (dummy_rgb,)
        flops, _ = thop_profile(model, inputs=_inputs, verbose=False)
        flops_g  = round(flops / 1e9, 2)
    except Exception:
        pass

    # FPS
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            _call()
    if device.type == "cuda":
        torch.cuda.synchronize()
    fps = round(n_runs / (time.time() - t0), 1)

    return round(n_params, 2), flops_g, fps


# 鈹€鈹€ Plotting style 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

def _apply_style():
    plt.rcParams.update({
        "font.family":        "serif",
        "font.serif":         ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size":          16,
        "axes.titlesize":     18,
        "axes.labelsize":     17,
        "xtick.labelsize":    15,
        "ytick.labelsize":    15,
        "legend.fontsize":    15,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          True,
        "grid.alpha":         0.25,
        "grid.linestyle":     "--",
        "figure.dpi":         180,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.04,
    })


def _model_color(name):
    lower = str(name).lower()
    if "md_hcr" in lower or "md-hcr" in lower:
        return "#FF6666"
    if name in _MODEL_COLORS:
        return _MODEL_COLORS[name]
    for key, color in _MODEL_COLORS.items():
        if key.lower() in lower:
            return color
    return _MODEL_COLOR.get(name, "#8C8C8C")


def _sort_last_ours(names):
    others = [n for n in names if "MD-HCR" not in n]
    ours = [n for n in names if "MD-HCR" in n]
    return others + ours


def _sort_df_last_ours(df, metric_col=None):
    if df.empty or "Model" not in df.columns:
        return df
    others = df[~df["Model"].astype(str).str.contains("MD-HCR", case=False, na=False)]
    ours = df[df["Model"].astype(str).str.contains("MD-HCR", case=False, na=False)]
    if metric_col and metric_col in others.columns:
        others = others.sort_values(metric_col, ascending=True)
    return __import__("pandas").concat([others, ours], ignore_index=True)


def _ensure_colors(names):
    """Assign palette colors to any model names not yet in _MODEL_COLOR."""
    for i, name in enumerate(names):
        _MODEL_COLOR[name] = _model_color(name)

def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure 鈫?{path}")


# 鈹€鈹€ Figure generators (one figure per table) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

def fig_main_comparison(df, fig_dir, per_sample_dict=None):
    """Paper-style grouped bar chart: MPJPE + PA-MPJPE."""
    _apply_style()
    df = _sort_df_last_ours(df, "MPJPE (mm)")
    models    = df["Model"].tolist()
    mpjpes    = df["MPJPE (mm)"].tolist()
    pa_mpjpes = df["PA-MPJPE (mm)"].tolist()

    n = len(models)
    x = np.arange(n)
    width = 0.36
    fig, ax = plt.subplots(figsize=(max(9, n * 1.6), 5.8))
    colors = [_model_color(m) for m in models]

    for i, (model, mpjpe, pa) in enumerate(zip(models, mpjpes, pa_mpjpes)):
        is_ours = "MD-HCR" in model
        ax.text(i - width/2, mpjpe + 0.3, f"{mpjpe:.1f}",
                ha="center", va="bottom", fontsize=12,
                fontweight="bold" if is_ours else "normal")
        ax.text(i + width/2, pa + 0.3, f"{pa:.1f}",
                ha="center", va="bottom", fontsize=12)

    ax.bar(x - width/2, mpjpes, width, color=colors, alpha=0.90,
           label="MPJPE", zorder=3)
    ax.bar(x + width/2, pa_mpjpes, width, color=colors, alpha=0.50,
           label="PA-MPJPE", zorder=3, hatch="//")

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=18, ha="right")
    ax.set_ylabel("Error (mm)", labelpad=2)
    ax.legend(framealpha=0.9, loc="upper left")
    ax.set_ylim(0, max(max(mpjpes), max(pa_mpjpes)) * 1.16)
    fig.tight_layout()

    _save(fig, os.path.join(fig_dir, "main_comparison.pdf"))


def fig_per_joint(df_per_joint, fig_dir, model_names):
    """Paper-style per-finger MPJPE grouped bar chart."""
    _apply_style()
    model_names = _sort_last_ours([n for n in model_names
                                   if not df_per_joint[df_per_joint["Model"] == n].empty])

    FINGER_GROUPS = [
        ("Thumb",  [13, 14, 15, 20]),
        ("Index",  [1, 2, 3, 16]),
        ("Middle", [4, 5, 6, 17]),
        ("Ring",   [10, 11, 12, 19]),
        ("Pinky",  [7, 8, 9, 18]),
    ]

    n_groups  = len(FINGER_GROUPS)
    n_models  = len(model_names)
    total_w   = 0.75
    bar_w     = total_w / max(n_models, 1)
    offsets   = np.linspace(-total_w/2 + bar_w/2, total_w/2 - bar_w/2, n_models)

    fig, ax = plt.subplots(figsize=(max(10, n_groups * 1.8), 5.5))
    x = np.arange(n_groups)

    for mi, (name, offset) in enumerate(zip(model_names, offsets)):
        row = df_per_joint[df_per_joint["Model"] == name]
        if row.empty:
            continue
        vals = []
        for _, jidxs in FINGER_GROUPS:
            vals.append(float(np.mean([float(row[JOINT_NAMES[j]].values[0])
                                       for j in jidxs if JOINT_NAMES[j] in row.columns])))
        is_ours = "MD-HCR" in name
        color = _model_color(name)
        bars = ax.bar(x + offset, vals, bar_w, color=color, alpha=0.88,
                      label=name, zorder=3,
                      linewidth=1.5 if is_ours else 0.5,
                      edgecolor="#333333" if is_ours else color)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=7.5,
                    fontweight="bold" if is_ours else "normal")

    ax.set_xticks(x)
    ax.set_xticklabels([fg[0] for fg in FINGER_GROUPS], fontsize=15)
    ax.set_ylabel("MPJPE (mm)", labelpad=4)
    ax.legend(ncol=n_models, loc="upper right", fontsize=11,
              framealpha=0.9, handlelength=1.4)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    fig.tight_layout()
    _save(fig, os.path.join(fig_dir, "per_joint_error.pdf"))


def fig_pck(df_pck, fig_dir, model_names):
    """Draw PCK curves for all evaluated models."""
    _apply_style()
    model_names = _sort_last_ours([n for n in model_names
                                   if not df_pck[df_pck["Model"] == n].empty])
    fig, ax = plt.subplots(figsize=(8, 5.8))

    thrs = [c for c in df_pck.columns if c.startswith("thr_")]
    thr_vals = np.array([float(c[4:]) for c in thrs])
    # Paper figure uses the full 0-80 mm PCK range.
    mask = thr_vals <= 80
    thr_plot = thr_vals[mask]

    for name in model_names:
        row = df_pck[df_pck["Model"] == name]
        if row.empty:
            continue
        pck_vals = np.array([row[c].values[0] for c in thrs])[mask]
        auc      = row["AUC@80mm"].values[0] if "AUC@80mm" in row.columns else 0
        color    = _model_color(name)
        is_ours  = "MD-HCR" in name
        lw       = 3.0 if is_ours else 1.8

        ax.plot(thr_plot, pck_vals, color=color, lw=lw,
                label=f"{name}  (AUC={auc:.1f})", zorder=4 if is_ours else 3)
        if is_ours:
            ax.fill_between(thr_plot, pck_vals, alpha=0.08, color=color, zorder=2)

    # Reference lines
    for ref_mm in [5, 15, 30]:
        ax.axvline(ref_mm, color="#777777", lw=0.9, ls=":", zorder=1)

    ax.set_xlabel("MPJPE threshold (mm)")
    ax.set_ylabel("PCK (%)", labelpad=2)
    ax.set_xlim(0, 80)
    ax.set_ylim(0, 101)
    ax.legend(framealpha=0.9, loc="lower right")
    fig.tight_layout()
    _save(fig, os.path.join(fig_dir, "pck_curve.pdf"))


def fig_training_curves(tables_dir, fig_dir, highlight_names=None):
    """Draw normalized convergence curves from training logs."""
    import pandas as pd
    path = os.path.join(tables_dir, "training_curves.xlsx")
    if not os.path.exists(path): return
    _apply_style()
    df = pd.read_excel(path, engine="openpyxl")

    # Plot every model recorded in xlsx
    all_tc_models = df["Model"].unique().tolist()
    if not all_tc_models:
        return

    # Ensure all models have a color assigned
    extra_palette = ["#FF6666", "#FFAA53", "#50CC55", "#00CCCC",
                     "#3399FF", "#AA44AA", "#FF9999", "#888888"]
    ci = 0
    for name in all_tc_models:
        if name not in _MODEL_COLOR:
            _MODEL_COLOR[name] = extra_palette[ci % len(extra_palette)]
            ci += 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    for name in all_tc_models:
        sub = df[df["Model"] == name].sort_values("Epoch")
        if sub.empty: continue
        c       = _MODEL_COLOR.get(name, "#AAAAAA")
        is_ours = "MD-HCR" in name
        lw      = 2.5 if is_ours else 1.5
        zo      = 4   if is_ours else 3
        ax1.plot(sub["Epoch"], sub["TrainLoss"], color=c, lw=lw, label=name, zorder=zo)
        ax2.plot(sub["Epoch"], sub["ValLoss"],   color=c, lw=lw, label=name, zorder=zo)

    for ax, title in [(ax1, "Training Loss Convergence"),
                      (ax2, "Validation Loss Convergence")]:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Normalized Loss (epoch-1 = 1.0)")
        ax.set_title(title)
        ax.set_ylim(0, 1.05)

    handles, labels = ax1.get_legend_handles_labels()
    n_cols = min(len(all_tc_models), 6)
    fig.legend(handles, labels, loc="lower center", ncol=n_cols,
               fontsize=12, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"Normalized Convergence Curves - {_DATASET_TITLE}", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    _save(fig, os.path.join(fig_dir, "training_curves.pdf"))


def fig_efficiency(df_eff, fig_dir, model_names):
    """Draw efficiency versus accuracy bubble chart."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(9, 6))

    pts = []  # (params, mpjpe, fps, name, color)
    for name in model_names:
        row = df_eff[df_eff["Model"] == name]
        if row.empty:
            continue
        try:
            params = float(row["Params (M)"].values[0])
            mpjpe  = float(row["MPJPE (mm)"].values[0])
        except (ValueError, TypeError):
            continue
        fps = row["FPS"].values[0] if "FPS" in row.columns else 10
        try:
            fps = float(fps)
        except (ValueError, TypeError):
            fps = 10.0
        color = _model_color(name)
        pts.append((params, mpjpe, fps, name, color))

    if not pts:
        plt.close(fig)
        return

    # Scale bubble size: area proportional to FPS, normalised so median 鈮?300 px虏
    fps_vals = np.array([p[2] for p in pts])
    fps_med  = np.median(fps_vals) if fps_vals.size else 1.0
    scale    = 300.0 / max(fps_med, 1.0)

    for params, mpjpe, fps, name, color in pts:
        is_ours = "MD-HCR" in name
        size    = max(fps * scale, 30)
        ax.scatter(params, mpjpe, s=size, color=color, alpha=0.85, zorder=5,
                   edgecolors="black" if is_ours else "white",
                   linewidths=1.5 if is_ours else 0.6)
        ax.annotate(f"{name}\n({fps:.0f} FPS)", (params, mpjpe),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=10, color=color,
                    fontweight="bold" if is_ours else "normal")

    # Pareto frontier (lower MPJPE and fewer params = better)
    sorted_pts = sorted(pts, key=lambda p: p[0])  # sort by params
    pareto = []
    best_mpjpe = float("inf")
    for p in sorted_pts:
        if p[1] < best_mpjpe:
            pareto.append(p)
            best_mpjpe = p[1]
    if len(pareto) >= 2:
        px = [p[0] for p in pareto]
        py = [p[1] for p in pareto]
        ax.plot(px, py, "k--", lw=1.0, alpha=0.4, label="Pareto frontier", zorder=2)
        ax.legend(framealpha=0.9, loc="upper right")

    ax.set_xlabel("Parameters (M)")
    ax.set_ylabel("MPJPE (mm)")
    _save(fig, os.path.join(fig_dir, "efficiency.pdf"))


def fig_pck_bar(df_pck, fig_dir, model_names):
    """Draw fixed-threshold PCK bar chart."""
    _apply_style()
    model_names = _sort_last_ours([n for n in model_names
                                   if not df_pck[df_pck["Model"] == n].empty])
    _ensure_colors(model_names)

    thresholds  = [5, 15, 30, 50]
    col_names   = [f"thr_{t}" for t in thresholds]
    n_models    = len(model_names)
    n_thr       = len(thresholds)
    group_w     = 0.75
    bar_w       = group_w / max(n_models, 1)
    x           = np.arange(n_thr)

    fig, ax = plt.subplots(figsize=(11, 5.8))
    bar_handles = []

    for mi, name in enumerate(model_names):
        row = df_pck[df_pck["Model"] == name]
        if row.empty:
            continue
        vals    = [float(row[c].values[0]) if c in row.columns else 0 for c in col_names]
        color   = _model_color(name)
        is_ours = "MD-HCR" in name
        offset  = (mi - (n_models - 1) / 2.0) * bar_w

        bars = ax.bar(x + offset, vals, bar_w * 0.92,
                      color=color, alpha=0.85 if not is_ours else 1.0,
                      edgecolor="black" if is_ours else "none",
                      linewidth=1.4 if is_ours else 0,
                      label=name, zorder=3)
        bar_handles.append(bars)
        if is_ours:
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{v:.0f}", ha="center", va="bottom",
                        fontsize=15, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"PCK@{t}mm" for t in thresholds], fontsize=14)
    ax.set_ylabel("PCK (%)")
    ax.set_ylim(0, 108)
    ax.set_title(f"PCK at Fixed Thresholds on {_DATASET_TITLE} Validation Set")

    ax.legend([b[0] for b in bar_handles], model_names, framealpha=0.9,
              ncol=n_models, loc="upper center", bbox_to_anchor=(0.5, -0.08),
              handlelength=1.6, columnspacing=1.0)
    fig.subplots_adjust(bottom=0.20)
    _save(fig, os.path.join(fig_dir, "pck_bar.pdf"))


def fig_mpjpe_distribution(per_sample_dict, fig_dir, model_names):
    """Draw per-sample MPJPE violin and box distributions."""
    _apply_style()
    model_names = _sort_last_ours(model_names)
    _ensure_colors(model_names)

    names_present = [n for n in model_names if n in per_sample_dict]
    if not names_present:
        return

    data   = [per_sample_dict[n] for n in names_present]
    colors = [_model_color(n) for n in names_present]

    fig, ax = plt.subplots(figsize=(max(9, len(names_present) * 1.4), 5.8))
    positions = np.arange(1, len(names_present) + 1)

    # Violin
    vp = ax.violinplot(data, positions=positions,
                       showmedians=False, showextrema=False, widths=0.7)
    for body, color in zip(vp["bodies"], colors):
        body.set_facecolor(color)
        body.set_alpha(0.45)

    # Box
    bp = ax.boxplot(data, positions=positions,
                    widths=0.25, patch_artist=True,
                    medianprops=dict(color="black", lw=2),
                    flierprops=dict(marker=".", markersize=2, alpha=0.3),
                    zorder=4)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    # Highlight ours
    for i, name in enumerate(names_present):
        if "MD-HCR" in name:
            ax.axvspan(positions[i] - 0.45, positions[i] + 0.45,
                       color=colors[i], alpha=0.05, zorder=1)

    ax.set_xticks(positions)
    ax.set_xticklabels(names_present, rotation=14, ha="right")
    ax.set_ylabel("Per-sample MPJPE (mm)", labelpad=2)
    fig.tight_layout()
    _save(fig, os.path.join(fig_dir, "mpjpe_distribution.pdf"))


def fig_joint_heatmap(df_per_joint, fig_dir, model_names):
    """Draw compact per-joint MPJPE heatmaps on a hand skeleton."""
    _apply_style()
    model_names = _sort_last_ours([
        n for n in model_names
        if not df_per_joint[df_per_joint["Model"] == n].empty
    ])
    _ensure_colors(model_names)

    # Canonical 2D positions for 21 MANO joints (normalized coords [0,1])
    # x: 0=thumb-side, 1=pinky-side; y: 0=wrist, 1=fingertips
    JOINT_POS = np.array([
        [0.50, 0.05],                                      # 0  Wrist
        [0.35, 0.25], [0.33, 0.43], [0.32, 0.58],         # 1-3  Index  MCP/PIP/DIP
        [0.47, 0.28], [0.46, 0.47], [0.46, 0.63],         # 4-6  Middle MCP/PIP/DIP
        [0.71, 0.22], [0.72, 0.38], [0.73, 0.51],         # 7-9  Pinky  MCP/PIP/DIP
        [0.59, 0.26], [0.60, 0.45], [0.60, 0.61],         # 10-12 Ring  MCP/PIP/DIP
        [0.25, 0.18], [0.14, 0.28], [0.07, 0.40],         # 13-15 Thumb CMC/MCP/IP
        [0.31, 0.73], [0.46, 0.80], [0.74, 0.63],         # 16-18 TIPs: Idx/Mid/Pky
        [0.61, 0.76], [0.02, 0.52],                        # 19-20 TIPs: Rng/Thb
    ])
    BONES = [
        (0, 1),  (1, 2),  (2, 3),  (3, 16),   # Index
        (0, 4),  (4, 5),  (5, 6),  (6, 17),   # Middle
        (0, 7),  (7, 8),  (8, 9),  (9, 18),   # Pinky
        (0, 10), (10,11), (11,12), (12,19),   # Ring
        (0, 13), (13,14), (14,15), (15,20),   # Thumb
        (1, 4),  (4, 10), (10, 7),            # Palm cross
    ]

    names_present = model_names
    if not names_present:
        return

    n_models = len(names_present)
    fig, axes = plt.subplots(1, n_models,
                              figsize=(2.50 * n_models + 0.85, 3.45))
    axes = np.array(axes).reshape(-1)
    fig.subplots_adjust(hspace=0.02, wspace=-0.26,
                        left=0.015, right=0.905, top=0.88, bottom=0.08)

    # Collect per-joint errors and find shared color range
    model_errors = {}
    all_errs = []
    for name in names_present:
        row  = df_per_joint[df_per_joint["Model"] == name]
        errs = []
        for jn in JOINT_NAMES:
            v = float(row[jn].values[0]) if jn in row.columns else np.nan
            errs.append(v)
        model_errors[name] = np.array(errs)
        all_errs.extend([v for v in errs if not np.isnan(v) and v > 0])  # skip wrist=0

    vmin = 0.0
    vmax = max(all_errs) if all_errs else 50.0
    cmap = _HEATMAP_CMAP
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    for i, name in enumerate(names_present):
        ax   = axes[i]
        errs = model_errors[name]

        ax.set_xlim(-0.08, 1.08)
        ax.set_ylim(-0.08, 1.02)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.text(0.5, 0.85, name, transform=ax.transData,
                ha="center", va="bottom",
                fontweight="bold" if "MD-HCR" in name else "normal",
                fontsize=13, clip_on=False)

        # Bones
        for (a, b) in BONES:
            ax.plot([JOINT_POS[a, 0], JOINT_POS[b, 0]],
                    [JOINT_POS[a, 1], JOINT_POS[b, 1]],
                    color="#C9C9C9", lw=1.8, zorder=1)

        # Joints (colored circles)
        for j, (x, y) in enumerate(JOINT_POS):
            v      = errs[j]
            color  = cmap(norm(v)) if (not np.isnan(v) and j != 0) else "#EEEEEE"
            radius = 0.042
            circle = plt.Circle((x, y), radius, facecolor=color, zorder=3,
                                 linewidth=1.25 if "MD-HCR" in name else 0.75,
                                 edgecolor="#555555" if "MD-HCR" in name else "#999999")
            ax.add_patch(circle)
            # Error label (skip wrist which is always 0 after root-rel)
            if not np.isnan(v) and j != 0:
                ax.text(x, y - 0.072, f"{v:.1f}",
                        ha="center", va="top", fontsize=7.5,
                        color="#222222", zorder=4)

    # Hide unused subplots
    for j in range(n_models, len(axes)):
        axes[j].set_visible(False)

    # Shared horizontal colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cax = fig.add_axes([0.888, 0.24, 0.014, 0.48])
    cbar = fig.colorbar(sm, cax=cax, orientation="vertical")
    cbar.set_label("MPJPE (mm)  鈫? lower is better", fontsize=12)

    fig.suptitle(f"Per-Joint Error Heatmap on {_DATASET_TITLE} Validation Set",
                 fontsize=16, y=1.01)
    plt.tight_layout()
    if fig._suptitle is not None:
        fig._suptitle.set_text("")
    cbar.set_label("MPJPE (mm)", fontsize=10)
    cbar.ax.yaxis.labelpad = 3
    cbar.ax.tick_params(labelsize=9, pad=1)
    fig.subplots_adjust(hspace=0.02, wspace=-0.26,
                        left=0.015, right=0.905, top=0.88, bottom=0.08)
    _save(fig, os.path.join(fig_dir, "joint_heatmap.pdf"))


# 鈹€鈹€ Main evaluation loop 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config",  default="config/freihand.yaml")
    parser.add_argument("--models",  nargs="+", default=None,
                        help="Subset of model keys to evaluate (default: all with checkpoints)")
    parser.add_argument("--skip_efficiency", action="store_true",
                        help="Skip FLOPs/FPS measurement (faster)")
    args   = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg    = load_config(args.config)
    global _DATASET_TITLE
    _DATASET_TITLE = str(getattr(cfg.data, "dataset", "dataset")).upper()

    base_dir   = cfg.train.output_dir
    models_dir = os.path.join(base_dir, "models")
    tables_dir = os.path.join(base_dir, "tables")
    fig_dir    = os.path.join(base_dir, "figures")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(fig_dir,    exist_ok=True)

    # 鈹€鈹€ Discover available models 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    all_keys = ["md_hcrnet"] + list(BASELINE_REGISTRY.keys())
    if args.models:
        all_keys = [k for k in all_keys if k in args.models]

    # 鈹€鈹€ Assign colours 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    for k in all_keys:
        _MODEL_COLOR[k] = _model_color(k)

    # 鈹€鈹€ Table loggers 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    main_log       = TableLogger(os.path.join(tables_dir, "main_comparison.xlsx"))
    per_joint_log  = TableLogger(os.path.join(tables_dir, "per_joint_error.xlsx"))
    pck_log        = TableLogger(os.path.join(tables_dir, "pck_data.xlsx"))
    eff_log        = TableLogger(os.path.join(tables_dir, "efficiency.xlsx"))
    per_sample_log = TableLogger(os.path.join(tables_dir, "per_sample_mpjpe.xlsx"))

    print(f"\n{'='*65}")
    print(f"  Unified Evaluation - {_DATASET_TITLE} Validation Set")
    print(f"  Device : {device}")
    print(f"  Models : {all_keys}")
    print(f"{'='*65}\n")

    _, val_loader = build_dataloaders(cfg)

    evaluated_names = []
    per_sample_dict = {}   # model_name 鈫?per_sample_mpjpe array

    for key in all_keys:
        print(f"\n鈹€鈹€ Evaluating: {key} {'鈹€'*40}")
        model, model_name = load_model(key, cfg, device, models_dir)
        if model is None:
            continue

        _MODEL_COLOR[model_name] = _MODEL_COLOR.pop(key, "#AAAAAA")
        evaluated_names.append(model_name)

        # 鈹€鈹€ Inference 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        pred, gt, verts_p, verts_g, per_sample_mpjpe = run_inference(model, val_loader, device)
        per_sample_dict[model_name] = per_sample_mpjpe

        pred_r = root_rel(pred)
        gt_r   = root_rel(gt)

        # 鈹€鈹€ Main metrics 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        mpjpe    = float(np.linalg.norm(pred_r - gt_r, axis=-1).mean())
        pa_mpjpe = float(PA_MPJPE(pred_r, gt_r))

        f5 = f15 = None
        if verts_p is not None:
            vr_p = root_rel(verts_p)
            vr_g = root_rel(verts_g)
            f5   = float(F_score(vr_p, vr_g,  5.0))
            f15  = float(F_score(vr_p, vr_g, 15.0))

        print(f"  MPJPE:      {mpjpe:.2f} mm")
        print(f"  PA-MPJPE:   {pa_mpjpe:.2f} mm")
        if f5 is not None:
            print(f"  F@5mm:      {f5:.4f}")
            print(f"  F@15mm:     {f15:.4f}")

        ci_lo, ci_hi = bootstrap_ci(per_sample_mpjpe)
        main_row = {
            "MPJPE (mm)":    round(mpjpe,    2),
            "PA-MPJPE (mm)": round(pa_mpjpe, 2),
            "F@5mm":         round(f5,  4) if f5  is not None else "-",
            "F@15mm":        round(f15, 4) if f15 is not None else "-",
            "CI_lo":         round(float(ci_lo), 3),
            "CI_hi":         round(float(ci_hi), 3),
        }
        main_log.update(model_name, main_row)

        # 鈹€鈹€ Per-joint MPJPE 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        pj = np.linalg.norm(pred_r - gt_r, axis=-1).mean(axis=0)  # (21,)
        pj_row = {jn: round(float(pj[i]), 2)
                  for i, jn in enumerate(JOINT_NAMES) if i < len(pj)}
        per_joint_log.update(model_name, pj_row)

        # 鈹€鈹€ PCK data 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        thrs = np.linspace(0, 80, 81)
        _, pck_vals, auc = pck_curve(pred_r, gt_r, thrs)
        pck_row = {f"thr_{int(t)}": round(float(v), 2)
                   for t, v in zip(thrs, pck_vals)}
        pck_row["AUC@80mm"] = round(auc, 3)
        pck_log.update(model_name, pck_row)

        # 鈹€鈹€ Efficiency 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        if not args.skip_efficiency:
            params, flops, fps = measure_efficiency(model, device)
            eff_row = {
                "Params (M)": params,
                "FLOPs (G)":  flops if flops else "N/A",
                "FPS":        fps,
                "MPJPE (mm)": round(mpjpe, 2),
            }
            eff_log.update(model_name, eff_row)

        # 鈹€鈹€ Per-sample MPJPE distribution锛堢敤浜?violin plot锛夆攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        ps_row = {f"s{i}": round(float(v), 3)
                  for i, v in enumerate(per_sample_mpjpe)}
        per_sample_log.update(model_name, ps_row)

    # 鈹€鈹€ Generate all figures 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    import pandas as pd
    print(f"\n{'='*65}")
    print("  Generating figures from tables...")

    df_main = main_log.read()
    if not df_main.empty:
        fig_main_comparison(df_main, fig_dir, per_sample_dict)

    df_pj = per_joint_log.read()
    if not df_pj.empty:
        fig_per_joint(df_pj, fig_dir, evaluated_names)

    df_pck = pck_log.read()
    if not df_pck.empty:
        fig_pck(df_pck, fig_dir, evaluated_names)

    fig_training_curves(tables_dir, fig_dir)

    if not args.skip_efficiency:
        df_eff = eff_log.read()
        if not df_eff.empty:
            fig_efficiency(df_eff, fig_dir, evaluated_names)

    # 鈹€鈹€ New experiment figures 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    # 1. PCK bar chart at fixed thresholds
    if not df_pck.empty:
        fig_pck_bar(df_pck, fig_dir, evaluated_names)

    # 2. MPJPE distribution violin plot
    if per_sample_dict:
        fig_mpjpe_distribution(per_sample_dict, fig_dir, evaluated_names)

    # 3. Joint error heatmap on canonical hand skeleton
    if not df_pj.empty:
        fig_joint_heatmap(df_pj, fig_dir, evaluated_names)

    # 鈹€鈹€ Print summary 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    print(f"\n{'='*65}")
    print(f"  {'Model':<22s} {'MPJPE':>9} {'PA-MPJPE':>12} {'F@5mm':>8} {'F@15mm':>8}")
    print(f"  {'-'*22} {'-'*9} {'-'*12} {'-'*8} {'-'*8}")
    df_main = main_log.read()
    for _, row in df_main.iterrows():
        m    = str(row["Model"])
        mp   = f"{row['MPJPE (mm)']:.2f}"    if "MPJPE (mm)"    in row else "-"
        pa   = f"{row['PA-MPJPE (mm)']:.2f}" if "PA-MPJPE (mm)" in row else "-"
        f5s  = f"{row['F@5mm']}"             if "F@5mm"         in row else "-"
        f15s = f"{row['F@15mm']}"            if "F@15mm"        in row else "-"
        mark = "*" if "MD-HCR" in m else " "
        print(f"  {mark} {m:<20s} {mp:>9} {pa:>12} {f5s:>8} {f15s:>8}")
    print(f"{'='*65}")
    print(f"\n  Tables 鈫?{tables_dir}")
    print(f"  Figures 鈫?{fig_dir}\n")

    # 鈹€鈹€ Qualitative visualization (integrated from qualitative_viz.py) 鈹€鈹€鈹€鈹€鈹€
    print(f"{'='*65}")
    print("  Generating qualitative figures ...")
    _run_qualitative(cfg, fig_dir, models_dir, device, n_samples=5)
    print(f"{'='*65}\n")


def _run_qualitative(cfg, fig_dir, models_dir, device, n_samples=5):
    """Integrate qualitative_viz.py into the single eval command.

    Loads all available checkpoints, collects n_samples from the val set,
    and calls gen_comparison_figure / gen_ours_figure from qualitative_viz.py.
    """
    try:
        from qualitative_viz import (gen_comparison_figure, gen_ours_figure,
                                     infer_single)
    except ImportError as e:
        print(f"  [qualitative] Cannot import qualitative_viz: {e}")
        return

    # Batch-1 val loader for sample collection
    import copy
    cfg_q = copy.deepcopy(cfg)
    cfg_q.data.batch_size = 1
    _, val_loader_q = build_dataloaders(cfg_q)

    total = len(val_loader_q)
    step  = max(1, total // n_samples)
    sel   = list(range(0, min(total, step * n_samples), step))[:n_samples]

    samples = []
    for i, batch in enumerate(val_loader_q):
        if i in sel:
            samples.append(batch)
        if len(samples) == len(sel):
            break

    print(f"  Collected {len(samples)} samples for qualitative figures.")

    # Load all models (same order as eval loop)
    all_keys = ["md_hcrnet"] + list(BASELINE_REGISTRY.keys())
    qmodels      = {}
    qmodel_names = {}
    for key in all_keys:
        m, name = load_model(key, cfg, device, models_dir)
        if m is not None:
            qmodels[key]      = m
            qmodel_names[key] = name

    if "md_hcrnet" not in qmodels:
        print("  [qualitative] MD-HCR checkpoint not found 鈥?skipping.")
        return

    # Run single-sample inference for each model on selected samples
    predictions = {}
    for key, model in qmodels.items():
        preds = []
        for batch in samples:
            out = infer_single(model, batch, device)
            preds.append(out)
        predictions[key] = preds

    try:
        gen_comparison_figure(samples, predictions, qmodels, qmodel_names,
                              fig_dir, n_samples)
    except Exception as e:
        print(f"  [qualitative] gen_comparison_figure failed: {e}")

    try:
        gen_ours_figure(samples, predictions, fig_dir, n_samples)
    except Exception as e:
        print(f"  [qualitative] gen_ours_figure failed: {e}")


if __name__ == "__main__":
    main()
