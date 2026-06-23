"""
utils/visualization.py — MD-HCRNet 结果可视化

颜色规范（参考 FedPOT plot_style.py 风格）:
  MD-HCRNet (ours)   → #FF6666
  Stage-1 (中间结果)  → #FFAA53
  其他对比方法 (按序) → #50CC55, #00DDDD, #3399FF, #6666FF, #9933FF
  热力图(纯正值)      → 白 → #007FFF
  热力图(含负值)      → #FF4F4F → 白 → #007FFF
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401


# ── 颜色 ─────────────────────────────────────────────────────────────────────

OURS_COLOR   = "#FF6666"   # MD-HCRNet
STAGE1_COLOR = "#FFAA53"   # Stage-1 中间输出

OTHER_COLORS = [
    "#50CC55",  # 0
    "#00DDDD",  # 1
    "#3399FF",  # 2
    "#6666FF",  # 3
    "#9933FF",  # 4
]

_METHODS_IN_ORDER = [
    "Baseline",
    "HandOccNet",
    "MeshGraphormer",
    "HDR",
]


def method_color(name: str) -> str:
    if any(k in name for k in ("MD-HCRNet", "Ours", "ours")):
        return OURS_COLOR
    if "Stage1" in name or "Stage-1" in name:
        return STAGE1_COLOR
    try:
        idx = _METHODS_IN_ORDER.index(name)
        return OTHER_COLORS[idx % len(OTHER_COLORS)]
    except ValueError:
        return OTHER_COLORS[abs(hash(name)) % len(OTHER_COLORS)]


# 关节颜色（左/右手骨架）
JOINT_COLOR  = "#FF6666"
BONE_COLOR   = "#3399FF"

# ── 字号 ─────────────────────────────────────────────────────────────────────

FS_TICK   = 14
FS_LABEL  = 15
FS_TITLE  = 16
FS_LEGEND = 13
FS_ANNOT  = 12

# ── Colormap ──────────────────────────────────────────────────────────────────

def make_seq_cmap():
    """纯正值热力图: 白 → #007FFF"""
    return mcolors.LinearSegmentedColormap.from_list(
        "mdhcr_seq", ["#FFFFFF", "#007FFF"], N=256)


def make_div_cmap():
    """含正负值热力图: #FF4F4F → 白 → #007FFF"""
    return mcolors.LinearSegmentedColormap.from_list(
        "mdhcr_div", ["#FF4F4F", "#FFFFFF", "#007FFF"], N=256)


# ── 全局 rcParams ─────────────────────────────────────────────────────────────

def apply_style():
    plt.rcParams.update({
        "font.family":        "serif",
        "font.serif":         ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size":          FS_TICK,
        "axes.titlesize":     FS_TITLE,
        "axes.labelsize":     FS_LABEL,
        "xtick.labelsize":    FS_TICK,
        "ytick.labelsize":    FS_TICK,
        "legend.fontsize":    FS_LEGEND,
        "axes.grid":          True,
        "grid.alpha":         0.3,
        "grid.linestyle":     "--",
        "grid.color":         "#CCCCCC",
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "lines.linewidth":    2.5,
        "lines.markersize":   8,
        "pdf.fonttype":       42,
        "ps.fonttype":        42,
        "figure.dpi":         150,
        "savefig.dpi":        300,
    })


# ── 保存工具 ──────────────────────────────────────────────────────────────────

def save_pdf(fig, path: str, tight: bool = True):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    kw = dict(format="pdf", bbox_inches="tight") if tight else dict(format="pdf")
    fig.savefig(path, **kw)
    plt.close(fig)


def save_png(fig, path: str, dpi: int = 300):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_xlsx(df_or_rows, path: str, sheet_name: str = "Sheet1",
              header=None, index: bool = False):
    try:
        import pandas as pd
        import openpyxl
    except ImportError:
        print(f"  [Vis] openpyxl/pandas 未安装，跳过 XLSX: {path}")
        return

    import pandas as pd
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df = pd.DataFrame(df_or_rows) if not isinstance(df_or_rows, pd.DataFrame) \
         else df_or_rows
    if header is not None:
        df.columns = header

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=index)
        ws = writer.sheets[sheet_name]
        for col in ws.columns:
            max_len = max(
                (len(str(cell.value or "")) for cell in col), default=8)
            ws.column_dimensions[col[0].column_letter].width = max_len + 4


# ── 训练曲线 ──────────────────────────────────────────────────────────────────

def plot_training_curves(train_losses: list, val_losses: list,
                         save_path: str = None):
    """
    绘制 train / val loss 曲线并保存。

    Args:
        train_losses: 每个 epoch 的训练 loss 列表
        val_losses:   每个 epoch 的验证 loss 列表
        save_path:    保存路径（pdf/png），None 时只显示
    """
    apply_style()
    epochs = list(range(1, len(train_losses) + 1))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, color=OURS_COLOR,   label="Train Loss", linewidth=2.5)
    ax.plot(epochs, val_losses,   color=STAGE1_COLOR, label="Val Loss",   linewidth=2.5,
            linestyle="--")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("MD-HCRNet — Training Curves")
    ax.legend()

    if save_path:
        if save_path.endswith(".pdf"):
            save_pdf(fig, save_path)
        else:
            save_png(fig, save_path)
    else:
        plt.show()
    return fig


# ── 手部骨架结构 ──────────────────────────────────────────────────────────────

# 21关节骨架连接（wrist=0, Thumb=1-4, Index=5-8, Middle=9-12, Ring=13-16, Pinky=17-20）
HAND_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
]

FINGER_COLORS = {
    "thumb":  "#FF6666",
    "index":  "#FFAA53",
    "middle": "#50CC55",
    "ring":   "#3399FF",
    "pinky":  "#9933FF",
    "wrist":  "#AAAAAA",
}

_BONE_COLORS = (
    [FINGER_COLORS["thumb"]]  * 4 +
    [FINGER_COLORS["index"]]  * 4 +
    [FINGER_COLORS["middle"]] * 4 +
    [FINGER_COLORS["ring"]]   * 4 +
    [FINGER_COLORS["pinky"]]  * 4
)


def visualize_joints_2d(image: np.ndarray,
                        joints_pred: np.ndarray,
                        joints_gt:   np.ndarray = None,
                        save_path:   str = None):
    """
    在图像上叠加 2D 关键点和骨架。

    Args:
        image:       (H, W, 3) uint8
        joints_pred: (21, 2) 预测 2D 关节
        joints_gt:   (21, 2) 真实 2D 关节（可选）
        save_path:   保存路径
    """
    apply_style()
    ncols = 2 if joints_gt is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))
    if ncols == 1:
        axes = [axes]

    for ax, joints, title in zip(
            axes,
            [joints_pred] + ([joints_gt] if joints_gt is not None else []),
            ["Predicted", "Ground Truth"]):
        ax.imshow(image)
        for i, (a, b) in enumerate(HAND_BONES):
            c = _BONE_COLORS[i]
            ax.plot([joints[a, 0], joints[b, 0]],
                    [joints[a, 1], joints[b, 1]],
                    color=c, linewidth=2.0)
        ax.scatter(joints[:, 0], joints[:, 1],
                   c=JOINT_COLOR, s=30, zorder=5, edgecolors="white", linewidths=0.5)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        if save_path.endswith(".pdf"):
            save_pdf(fig, save_path)
        else:
            save_png(fig, save_path)
    else:
        plt.show()
    return fig


def visualize_joints_3d(joints_pred: np.ndarray,
                        joints_gt:   np.ndarray = None,
                        save_path:   str = None):
    """
    绘制 3D 手部关节骨架（根节点对齐）。

    Args:
        joints_pred: (21, 3)
        joints_gt:   (21, 3) 可选
        save_path:   保存路径
    """
    apply_style()
    ncols = 2 if joints_gt is not None else 1
    fig = plt.figure(figsize=(5 * ncols, 5))

    def _draw(ax, joints, title, color_joints=JOINT_COLOR):
        # 根节点对齐
        joints = joints - joints[0:1]
        for i, (a, b) in enumerate(HAND_BONES):
            c = _BONE_COLORS[i]
            ax.plot([joints[a, 0], joints[b, 0]],
                    [joints[a, 1], joints[b, 1]],
                    [joints[a, 2], joints[b, 2]],
                    color=c, linewidth=2.0)
        ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2],
                   c=color_joints, s=30, zorder=5)
        ax.set_title(title)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.grid(True, alpha=0.3)

    items = [(joints_pred, "Predicted", JOINT_COLOR)]
    if joints_gt is not None:
        items.append((joints_gt, "Ground Truth", STAGE1_COLOR))

    for i, (j, t, c) in enumerate(items):
        ax = fig.add_subplot(1, ncols, i + 1, projection="3d")
        _draw(ax, j, t, c)

    plt.tight_layout()
    if save_path:
        if save_path.endswith(".pdf"):
            save_pdf(fig, save_path)
        else:
            save_png(fig, save_path)
    else:
        plt.show()
    return fig


# ── 指标柱状图 ────────────────────────────────────────────────────────────────

def plot_metrics_bar(metrics: dict, title: str = "Evaluation Metrics",
                     save_path: str = None):
    """
    Args:
        metrics: {"MPJPE": 12.3, "PA-MPJPE": 8.1, "F@5mm": 0.72, ...}
    """
    apply_style()
    names  = list(metrics.keys())
    values = list(metrics.values())
    colors = [OURS_COLOR, STAGE1_COLOR] + OTHER_COLORS * 5

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.5), 5))
    bars = ax.bar(names, values, color=colors[:len(names)], width=0.5, zorder=3)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01 * max(values),
                f"{val:.2f}", ha="center", va="bottom", fontsize=FS_ANNOT)

    ax.set_title(title)
    ax.set_ylabel("Value")

    plt.tight_layout()
    if save_path:
        if save_path.endswith(".pdf"):
            save_pdf(fig, save_path)
        else:
            save_png(fig, save_path)
    else:
        plt.show()
    return fig
