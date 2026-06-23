"""
RHD (Rendered Hand Dataset v2) loader.

Directory layout under data_root/RHD_published_v2/:
    training/
        color/   00000.png …  (41258 images)
        depth/   00000.png …
        mask/    00000.png …
        anno_training.pickle   # dict {idx: {'xyz':(42,3), 'uv_vis':(42,3), 'K':(3,3)}}
    evaluation/
        color/   00000.png …  (2728 images)
        depth/   00000.png …
        mask/    00000.png …
        anno_evaluation.pickle

Download: https://lmb.informatik.uni-freiburg.de/resources/datasets/RenderedHandposeDataset.en.html

Annotation fields per sample:
    xyz    : (42, 3)  3D coords in camera space, metres  (21 left + 21 right)
    uv_vis : (42, 3)  (u, v, visibility)
    K      : (3, 3)   camera intrinsics

Joint order per hand (indices 0-20 left, 21-41 right):
    0  Wrist
    1-4   Thumb  (CMC, MCP, IP, TIP)
    5-8   Index  (MCP, PIP, DIP, TIP)
    9-12  Middle (MCP, PIP, DIP, TIP)
    13-16 Ring   (MCP, PIP, DIP, TIP)
    17-20 Pinky  (MCP, PIP, DIP, TIP)

This matches the FreiHAND joint order, so FREIHAND_TO_MANO21 applies directly.
We use only the right hand (offset +21).
3-D values are in metres; ×1000 → mm is done in the metric layer.
"""
import os
import pickle
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

_RH_OFFSET = 21   # right-hand joints start at index 21

# FreiHAND → MANO-21 reordering
FREIHAND_TO_MANO21 = [0,
                       5,  6,  7,       # Index  MCP, PIP, DIP
                       9,  10, 11,      # Middle MCP, PIP, DIP
                       17, 18, 19,      # Pinky  MCP, PIP, DIP
                       13, 14, 15,      # Ring   MCP, PIP, DIP
                       1,  2,  3,       # Thumb  CMC, MCP, IP
                       8,  12, 20, 16, 4]  # TIPs: Index, Middle, Pinky, Ring, Thumb


class RHDDataset(Dataset):
    """
    Returns sample dicts compatible with transforms.py.

    split : "training"   → 41 258 samples  (train)
            "evaluation" → 2 728 samples   (test, public GT)
    """

    def __init__(self, root: str, split: str = "training",
                 transform=None, depth_cache_dir: str = None):
        super().__init__()
        self.root            = root
        self.split           = split
        self.transform       = transform
        self.depth_cache_dir = depth_cache_dir

        split_dir  = os.path.join(root, split)
        anno_name  = "anno_training.pickle" if split == "training" \
                     else "anno_evaluation.pickle"
        anno_path  = os.path.join(split_dir, anno_name)

        if not os.path.exists(anno_path):
            raise FileNotFoundError(f"RHD annotation not found: {anno_path}")

        with open(anno_path, "rb") as f:
            self._anno = pickle.load(f)   # dict {int: {'xyz', 'uv_vis', 'K'}}

        self.color_dir = os.path.join(split_dir, "color")
        self.n         = len(self._anno)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        img_path = os.path.join(self.color_dir, f"{idx:05d}.png")
        image    = np.array(Image.open(img_path).convert("RGB"))

        ann = self._anno[idx]
        xyz    = ann["xyz"].astype(np.float32)     # (42,3) metres
        uv_vis = ann["uv_vis"].astype(np.float32)  # (42,3)
        K      = ann["K"].astype(np.float32)       # (3,3)

        # ── Right-hand joints ──────────────────────────────────────────────
        rh_xyz = xyz[_RH_OFFSET:_RH_OFFSET + 21]          # (21,3)
        rh_uv  = uv_vis[_RH_OFFSET:_RH_OFFSET + 21, :2]   # (21,2)

        # Reorder to MANO-21 convention
        xyz21 = rh_xyz[FREIHAND_TO_MANO21]   # (21,3)
        uv21  = rh_uv[FREIHAND_TO_MANO21]    # (21,2)

        # ── Camera intrinsics ──────────────────────────────────────────────
        focal = float((K[0, 0] + K[1, 1]) / 2.0)
        cx    = float(K[0, 2])
        cy    = float(K[1, 2])

        joints2d = np.hstack([uv21, np.ones((21, 1), np.float32)])  # (21,3)

        sample = dict(
            image    = image,
            focal    = focal,
            cx       = cx,
            cy       = cy,
            joints2d = joints2d,
            joints3d = xyz21,        # metres
        )

        if self.depth_cache_dir is not None:
            depth_path = os.path.join(self.depth_cache_dir, f"{idx:05d}.npy")
            if os.path.exists(depth_path):
                sample["depth"] = np.load(depth_path)

        if self.transform is not None:
            sample = self.transform(sample)
        return sample
