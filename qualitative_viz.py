"""
Qualitative visualization for MD-HCRNet paper.

Generates:
  figures/qualitative_comparison.pdf  — Multi-model side-by-side on real hand images
  figures/qualitative_ours.pdf        — MD-HCRNet detailed pipeline visualization

Usage:
    python qualitative_viz.py --config config/freihand.yaml [--n_samples 5]
"""
import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import yaml
from types import SimpleNamespace

from data      import build_dataloaders
from baselines import build_baseline, BASELINE_REGISTRY
from models    import MDHCRNet

QUAL_HEADER_FONTSIZE = 30
QUAL_HEADER_PAD = 4
QUAL_WSPACE = -0.6
QUAL_HSPACE = 0.04
QUAL_LEGEND_FONTSIZE = 26
QUAL_COMPARISON_MARGINS = dict(left=0.01, right=0.99, top=0.955, bottom=0.055)
QUAL_OURS_MARGINS = dict(left=0.01, right=0.99, top=0.955, bottom=0.02)


# ── MANO-21 skeleton ──────────────────────────────────────────────────────────

FINGER_COLORS = {
    "Index":  "#FF8C42",
    "Middle": "#4CAF50",
    "Pinky":  "#2196F3",
    "Ring":   "#00BCD4",
    "Thumb":  "#F44336",
    "Wrist":  "#9E9E9E",
}

# 20 bones: (parent, child)
BONES_21 = [
    (0, 1),  (0, 4),  (0, 7),  (0, 10), (0, 13),   # wrist to finger roots
    (1, 2),  (2, 3),  (3, 16),                        # Index chain
    (4, 5),  (5, 6),  (6, 17),                        # Middle chain
    (7, 8),  (8, 9),  (9, 18),                        # Pinky chain
    (10, 11),(11, 12),(12, 19),                        # Ring chain
    (13, 14),(14, 15),(15, 20),                        # Thumb chain
]

# Color for each bone (same order as BONES_21)
BONE_COLORS_21 = (
    [FINGER_COLORS["Index"],  FINGER_COLORS["Middle"],
     FINGER_COLORS["Pinky"],  FINGER_COLORS["Ring"], FINGER_COLORS["Thumb"]] +
    [FINGER_COLORS["Index"]]  * 3 +
    [FINGER_COLORS["Middle"]] * 3 +
    [FINGER_COLORS["Pinky"]]  * 3 +
    [FINGER_COLORS["Ring"]]   * 3 +
    [FINGER_COLORS["Thumb"]]  * 3
)

# Joint colors (21 joints)
JOINT_COLORS_21 = (
    [FINGER_COLORS["Wrist"]] +
    [FINGER_COLORS["Index"]]  * 3 +
    [FINGER_COLORS["Middle"]] * 3 +
    [FINGER_COLORS["Pinky"]]  * 3 +
    [FINGER_COLORS["Ring"]]   * 3 +
    [FINGER_COLORS["Thumb"]]  * 3 +
    [FINGER_COLORS["Index"],  FINGER_COLORS["Middle"],
     FINGER_COLORS["Pinky"],  FINGER_COLORS["Ring"], FINGER_COLORS["Thumb"]]
)

# Green palette for GT
_GT_GREEN = "#2E7D32"
BONE_COLORS_GT  = [_GT_GREEN] * len(BONES_21)
JOINT_COLORS_GT = [_GT_GREEN] * 21


# ── Config helpers ────────────────────────────────────────────────────────────

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


# ── Plotting style ────────────────────────────────────────────────────────────

def _apply_style():
    plt.rcParams.update({
        "font.family":        "serif",
        "font.serif":         ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size":          12,
        "axes.titlesize":     13,
        "axes.labelsize":     11,
        "xtick.labelsize":    10,
        "ytick.labelsize":    10,
        "legend.fontsize":    10,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          False,
        "figure.dpi":         200,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.05,
    })


# ── Image helpers ─────────────────────────────────────────────────────────────

def denorm_img(img_tensor):
    """Denormalize ImageNet-normalized tensor → HWC uint8."""
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = img_tensor.float().cpu().permute(1, 2, 0).numpy()
    img  = (img * std + mean).clip(0, 1)
    return (img * 255).astype(np.uint8)


# ── Skeleton drawing ──────────────────────────────────────────────────────────

def draw_skeleton_2d(ax, joints2d, bone_colors, joint_colors, lw=2.0, ms=20):
    """Draw 2D hand skeleton on matplotlib axes."""
    for (p, q), c in zip(BONES_21, bone_colors):
        ax.plot([joints2d[p, 0], joints2d[q, 0]],
                [joints2d[p, 1], joints2d[q, 1]],
                color=c, lw=lw, solid_capstyle="round", zorder=3)
    ax.scatter(joints2d[:, 0], joints2d[:, 1],
               c=joint_colors, s=ms, zorder=5,
               linewidths=0.5, edgecolors="white")


def draw_skeleton_3d(ax, joints3d, bone_colors, joint_colors, lw=2.0, ms=15):
    """Draw 3D hand skeleton on 3D matplotlib axes."""
    for (p, q), c in zip(BONES_21, bone_colors):
        ax.plot([joints3d[p, 0], joints3d[q, 0]],
                [joints3d[p, 1], joints3d[q, 1]],
                [joints3d[p, 2], joints3d[q, 2]],
                color=c, lw=lw, zorder=3)
    ax.scatter(joints3d[:, 0], joints3d[:, 1], joints3d[:, 2],
               c=joint_colors, s=ms, zorder=5, depthshade=False)
    ax.set_axis_off()
    ax.view_init(elev=15, azim=-70)


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(model_key: str, cfg, device: torch.device, models_dir: str):
    """Load model from checkpoint. Returns (model, display_name) or (None, None)."""
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
    print(f"  Loaded: {model_name:<22s}  (epoch {epoch})  <- {ckpt_path}")
    return model, model_name


# ── Single-sample inference ───────────────────────────────────────────────────

@torch.no_grad()
def infer_single(model, batch, device):
    """Run model on a single-sample batch. Returns dict with joints2d/joints3d."""
    rgb   = batch["image"].to(device)
    focal = batch["focal"].to(device)
    cx    = batch["cx"].to(device)
    cy    = batch["cy"].to(device)
    depth = batch["depth"].to(device) if "depth" in batch else None

    out = model(rgb, focal, cx, cy, depth)

    j2d = out["joints2d"].float().cpu().numpy()[0]   # (21, 2)
    j3d = out["joints3d"].float().cpu().numpy()[0]   # (21, 3) in metres
    j3d_r = j3d - j3d[:1]                            # root-relative
    return {"joints2d": j2d, "joints3d_r": j3d_r * 1000}  # mm


def sample_mpjpe_mm(pred, batch):
    """Root-relative MPJPE in mm for one qualitative candidate."""
    gt = batch["joints3d"][0].float().cpu().numpy() * 1000.0
    gt = gt - gt[:1]
    return float(np.linalg.norm(pred["joints3d_r"] - gt, axis=1).mean())


def collect_spread_samples(val_loader, n_samples):
    total = len(val_loader)
    step = max(1, total // n_samples)
    selected_indices = list(range(0, min(total, step * n_samples), step))[:n_samples]
    samples = []
    for i, batch in enumerate(val_loader):
        if i not in selected_indices:
            continue
        samples.append(batch)
        if len(samples) == len(selected_indices):
            break
    print(f"  Selecting {len(selected_indices)} spread samples from {total} total "
          f"(indices: {selected_indices})")
    return samples, selected_indices


def collect_best_samples(val_loader, md_model, device, n_samples):
    scored = []
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            pred = infer_single(md_model, batch, device)
            score = sample_mpjpe_mm(pred, batch)
            scored.append((score, i, batch))

    scored.sort(key=lambda x: x[0])
    min_gap = max(1, len(val_loader) // max(n_samples * 12, 1))
    selected = []
    for score, idx, batch in scored:
        if all(abs(idx - chosen_idx) >= min_gap for _, chosen_idx, _ in selected):
            selected.append((score, idx, batch))
        if len(selected) == n_samples:
            break

    if len(selected) < n_samples:
        chosen = {idx for _, idx, _ in selected}
        for item in scored:
            if item[1] not in chosen:
                selected.append(item)
                chosen.add(item[1])
            if len(selected) == n_samples:
                break

    selected.sort(key=lambda x: x[1])
    scores = [score for score, _, _ in selected]
    indices = [idx for _, idx, _ in selected]
    samples = [batch for _, _, batch in selected]
    print(f"  Selecting {len(samples)} best-MPJPE samples from {len(val_loader)} total "
          f"(indices: {indices}, MPJPE: {[round(s, 2) for s in scores]})")
    return samples, indices


# ── Figure 1: multi-model comparison ─────────────────────────────────────────

def gen_comparison_figure(samples, predictions, models, model_names,
                          fig_dir, n_samples):
    """
    Layout: rows=n_samples, cols=7
      Col 0: Input
      Col 1: Ours (2D)
      Col 2: Ours (3D)
      Col 3: I2L (2D)
      Col 4: HandOccNet (2D)
      Col 5: GraphHand (2D)
      Col 6: LightAttHand (2D)
      Col 7: DepthFusion (2D)
      Col 8: GT (2D)
    """
    _apply_style()

    # Column specification: (key_or_tag, label)
    # key_or_tag: model key for model columns, "input" for raw image, "gt" for GT
    COL_SPECS = [
        ("input",     "Input"),
        ("md_hcrnet", "Ours (2D)"),
        ("md_hcrnet", "Ours (3D)"),
        ("i2l",       "I2L"),
        ("handoccnet", "HandOccNet"),
        ("graphhand", "GraphHand"),
        ("lightatt",  "LightAttHand"),
        ("depthfusion", "DepthFusion"),
        ("gt",        "GT"),
    ]
    n_cols = len(COL_SPECS)
    n_rows = len(samples)

    fig_w = n_cols * 3.0
    fig_h = n_rows * 3.5

    # Build axes: use 3D projection for col 2
    fig = plt.figure(figsize=(fig_w, fig_h))
    axes = []
    for r in range(n_rows):
        row_axes = []
        for c in range(n_cols):
            idx = r * n_cols + c + 1
            if c == 2:  # 3D column
                ax = fig.add_subplot(n_rows, n_cols, idx, projection="3d")
            else:
                ax = fig.add_subplot(n_rows, n_cols, idx)
            row_axes.append(ax)
        axes.append(row_axes)

    # Column headers on row 0
    for c, (_, label) in enumerate(COL_SPECS):
        axes[0][c].set_title(label, fontsize=QUAL_HEADER_FONTSIZE,
                             fontweight="bold", pad=QUAL_HEADER_PAD)

    for r, batch in enumerate(samples):
        img_np = denorm_img(batch["image"][0])  # HWC uint8
        H, W   = img_np.shape[:2]

        # GT 2D joints (drop visibility channel)
        gt_j2d = batch["joints2d"][0, :, :2].numpy()   # (21, 2)
        gt_j3d = batch["joints3d"][0].numpy() * 1000    # (21, 3) mm, may not be root-relative
        gt_j3d_r = gt_j3d - gt_j3d[:1]

        for c, (key, label) in enumerate(COL_SPECS):
            ax = axes[r][c]

            if key == "input":
                ax.imshow(img_np)
                ax.axis("off")

            elif key == "gt":
                ax.imshow(img_np)
                draw_skeleton_2d(ax, gt_j2d, BONE_COLORS_GT, JOINT_COLORS_GT,
                                 lw=1.8, ms=16)
                ax.axis("off")
                ax.set_xlim(0, W); ax.set_ylim(H, 0)

            elif c == 2 and key == "md_hcrnet":
                # 3D column — MD-HCRNet prediction
                if "md_hcrnet" in predictions and r < len(predictions["md_hcrnet"]):
                    j3d_r = predictions["md_hcrnet"][r]["joints3d_r"]
                    draw_skeleton_3d(ax, j3d_r, BONE_COLORS_21, JOINT_COLORS_21,
                                     lw=2.0, ms=15)
                else:
                    ax.set_axis_off()
                    ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                            transform=ax.transAxes, fontsize=11)

            else:
                # 2D overlay for a model
                ax.imshow(img_np)
                if key in predictions and r < len(predictions[key]):
                    j2d = predictions[key][r]["joints2d"]
                    draw_skeleton_2d(ax, j2d, BONE_COLORS_21, JOINT_COLORS_21,
                                     lw=1.8, ms=16)
                ax.axis("off")
                ax.set_xlim(0, W); ax.set_ylim(H, 0)

    # Finger color legend at bottom
    legend_handles = [
        mpatches.Patch(facecolor=FINGER_COLORS[f], label=f)
        for f in ["Index", "Middle", "Pinky", "Ring", "Thumb"]
    ]
    legend_handles.append(mpatches.Patch(facecolor=_GT_GREEN, label="GT"))
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=len(legend_handles), framealpha=0.9,
               fontsize=QUAL_LEGEND_FONTSIZE, bbox_to_anchor=(0.5, 0.0))

    fig.subplots_adjust(wspace=QUAL_WSPACE, hspace=QUAL_HSPACE,
                        **QUAL_COMPARISON_MARGINS)
    out_path = os.path.join(fig_dir, "qualitative_comparison.pdf")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure → {out_path}")


# ── Figure 2: MD-HCRNet detailed results ─────────────────────────────────────

def gen_ours_figure(samples, predictions, fig_dir, n_samples):
    """
    Layout: rows=n_samples, cols=5
      Col 0: Input RGB
      Col 1: Metric Depth (plasma) or N/A
      Col 2: Pred 2D (image + colored skeleton)
      Col 3: Pred 3D (3D axes)
      Col 4: GT 3D (green, 3D axes)
    """
    _apply_style()

    COL_LABELS = ["Input RGB", "Metric Depth", "Pred. 2D", "Pred. 3D", "GT 3D"]
    n_cols = len(COL_LABELS)
    n_rows = len(samples)

    fig_w = 17.5
    fig_h = n_rows * 3.5

    fig = plt.figure(figsize=(fig_w, fig_h))
    axes = []
    for r in range(n_rows):
        row_axes = []
        for c in range(n_cols):
            idx = r * n_cols + c + 1
            if c in (3, 4):  # 3D columns
                ax = fig.add_subplot(n_rows, n_cols, idx, projection="3d")
            else:
                ax = fig.add_subplot(n_rows, n_cols, idx)
            row_axes.append(ax)
        axes.append(row_axes)

    # Column headers
    for c, label in enumerate(COL_LABELS):
        axes[0][c].set_title(label, fontsize=QUAL_HEADER_FONTSIZE,
                             fontweight="bold", pad=QUAL_HEADER_PAD)

    for r, batch in enumerate(samples):
        img_np = denorm_img(batch["image"][0])
        H, W   = img_np.shape[:2]

        # GT 3D (root-relative, mm)
        gt_j3d = batch["joints3d"][0].numpy() * 1000
        gt_j3d_r = gt_j3d - gt_j3d[:1]

        # Predicted joints
        has_pred = "md_hcrnet" in predictions and r < len(predictions["md_hcrnet"])
        pred_j2d = predictions["md_hcrnet"][r]["joints2d"]  if has_pred else None
        pred_j3d_r = predictions["md_hcrnet"][r]["joints3d_r"] if has_pred else None

        # Col 0: Input RGB
        axes[r][0].imshow(img_np)
        axes[r][0].axis("off")

        # Col 1: Depth map
        if "depth" in batch and batch["depth"] is not None:
            depth_np = batch["depth"][0].float().cpu().numpy()
            # Handle multi-channel depth (take first channel)
            if depth_np.ndim == 3:
                depth_np = depth_np[0]
            axes[r][1].imshow(depth_np, cmap="plasma")
            axes[r][1].axis("off")
        else:
            axes[r][1].set_facecolor("#DDDDDD")
            axes[r][1].text(0.5, 0.5, "N/A", ha="center", va="center",
                            transform=axes[r][1].transAxes,
                            fontsize=12, color="#555555")
            axes[r][1].axis("off")

        # Col 2: Pred 2D overlay
        axes[r][2].imshow(img_np)
        if pred_j2d is not None:
            draw_skeleton_2d(axes[r][2], pred_j2d,
                             BONE_COLORS_21, JOINT_COLORS_21, lw=2.0, ms=20)
        axes[r][2].axis("off")
        axes[r][2].set_xlim(0, W); axes[r][2].set_ylim(H, 0)

        # Col 3: Pred 3D
        if pred_j3d_r is not None:
            draw_skeleton_3d(axes[r][3], pred_j3d_r,
                             BONE_COLORS_21, JOINT_COLORS_21, lw=2.0, ms=15)
        else:
            axes[r][3].set_axis_off()
            axes[r][3].text(0.5, 0.5, "N/A", ha="center", va="center",
                            transform=axes[r][3].transAxes, fontsize=11)

        # Col 4: GT 3D
        draw_skeleton_3d(axes[r][4], gt_j3d_r,
                         BONE_COLORS_GT, JOINT_COLORS_GT, lw=2.0, ms=15)

    fig.subplots_adjust(wspace=QUAL_WSPACE, hspace=QUAL_HSPACE,
                        **QUAL_OURS_MARGINS)
    out_path = os.path.join(fig_dir, "qualitative_ours.pdf")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config",   default="config/freihand.yaml")
    parser.add_argument("--n_samples", type=int, default=5)
    parser.add_argument("--sample_strategy", choices=["spread", "best"], default="spread",
                        help="spread: evenly spaced samples; best: lowest MD-HCR MPJPE samples")
    args   = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg    = load_config(args.config)

    # Output dirs
    fig_dir    = os.path.join(cfg.train.output_dir, "figures")
    models_dir = os.path.join(cfg.train.output_dir, "models")
    os.makedirs(fig_dir, exist_ok=True)

    # Load val loader (batch_size=1 for cleaner per-sample handling)
    cfg.data.batch_size = 1
    _, val_loader = build_dataloaders(cfg)

    # Load all models
    all_model_keys = ["md_hcrnet"] + list(BASELINE_REGISTRY.keys())
    models      = {}   # key -> model
    model_names = {}   # key -> display name
    for key in all_model_keys:
        m, name = load_model(key, cfg, device, models_dir)
        if m is not None:
            models[key]      = m
            model_names[key] = name

    if "md_hcrnet" not in models:
        print("ERROR: MD-HCRNet checkpoint not found. Cannot generate qualitative figures.")
        return

    if args.sample_strategy == "best":
        samples, selected_indices = collect_best_samples(
            val_loader, models["md_hcrnet"], device, args.n_samples)
    else:
        samples, selected_indices = collect_spread_samples(val_loader, args.n_samples)

    print(f"  Collected {len(samples)} sample batches.")

    # Run inference for all loaded models on the selected samples
    predictions = {}   # model_key -> list of output dicts
    for key, model in models.items():
        print(f"  Running inference: {model_names[key]}")
        preds = []
        with torch.no_grad():
            for batch in samples:
                out = infer_single(model, batch, device)
                preds.append(out)
        predictions[key] = preds

    # Generate both figures
    print("\n  Generating qualitative_comparison.pdf ...")
    gen_comparison_figure(samples, predictions, models, model_names,
                          fig_dir, args.n_samples)

    print("\n  Generating qualitative_ours.pdf ...")
    gen_ours_figure(samples, predictions, fig_dir, args.n_samples)

    print(f"\n  Done. Figures saved to: {fig_dir}")


if __name__ == "__main__":
    main()
