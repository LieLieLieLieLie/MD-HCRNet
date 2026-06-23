"""
HO3D v3 dataset loader.

Expected layout under data_root/HO3D/:
    train.txt
    train/<SEQ>/rgb/<FID>.jpg
    train/<SEQ>/depth/<FID>.png
    train/<SEQ>/meta/<FID>.pkl
    evaluation/...

The public evaluation split only exposes limited hand GT locally, so this
loader uses the annotated train split and creates deterministic train/val
partitions from train.txt.
"""
import os
import pickle
import random

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class HO3DDataset(Dataset):
    def __init__(self, root: str, split: str = "train", transform=None,
                 depth_cache_dir: str = None, val_ratio: float = 0.1,
                 split_seed: int = 2026):
        super().__init__()
        if split not in ("train", "val", "evaluation"):
            raise ValueError(f"Unknown HO3D split: {split}")

        self.root = root
        self.split = split
        self.transform = transform
        self.depth_cache_dir = depth_cache_dir

        if split in ("train", "val"):
            list_path = os.path.join(root, "train.txt")
            base_split = "train"
        else:
            list_path = os.path.join(root, "evaluation.txt")
            base_split = "evaluation"

        if not os.path.exists(list_path):
            raise FileNotFoundError(f"HO3D split list not found: {list_path}")

        with open(list_path, "r", encoding="utf-8") as f:
            entries = [line.strip() for line in f if line.strip()]

        if split in ("train", "val"):
            rng = random.Random(int(split_seed))
            indices = list(range(len(entries)))
            rng.shuffle(indices)
            n_val = max(1, int(round(len(indices) * float(val_ratio))))
            val_ids = set(indices[:n_val])
            if split == "train":
                entries = [e for i, e in enumerate(entries) if i not in val_ids]
            else:
                entries = [e for i, e in enumerate(entries) if i in val_ids]

        self.base_split = base_split
        self.entries = []
        for entry in entries:
            seq, fid = entry.split("/")
            rgb_path = os.path.join(root, base_split, seq, "rgb", f"{fid}.jpg")
            if not os.path.exists(rgb_path):
                rgb_path = os.path.join(root, base_split, seq, "rgb", f"{fid}.png")
            meta_path = os.path.join(root, base_split, seq, "meta", f"{fid}.pkl")
            self.entries.append((seq, fid, rgb_path, meta_path))

    def __len__(self):
        return len(self.entries)

    def _cache_path(self, seq: str, fid: str):
        if self.depth_cache_dir is None:
            return None
        return os.path.join(self.depth_cache_dir, self.base_split,
                            f"{seq}_{fid}.npy")

    def __getitem__(self, idx):
        seq, fid, rgb_path, meta_path = self.entries[idx]
        image = np.array(Image.open(rgb_path).convert("RGB"))

        with open(meta_path, "rb") as f:
            ann = pickle.load(f)

        K = np.array(ann["camMat"], dtype=np.float32)
        focal = float((K[0, 0] + K[1, 1]) * 0.5)
        cx = float(K[0, 2])
        cy = float(K[1, 2])

        sample = dict(image=image, focal=focal, cx=cx, cy=cy)

        joints3d_gl = np.asarray(ann.get("handJoints3D", np.zeros((21, 3))),
                                 dtype=np.float32)
        if joints3d_gl.shape == (21, 3):
            coord_change = np.array([1.0, -1.0, -1.0], dtype=np.float32)
            joints3d = joints3d_gl * coord_change
            xyz_h = (K @ joints3d.T).T
            joints2d_xy = xyz_h[:, :2] / np.maximum(xyz_h[:, 2:3], 1e-6)
            joints2d = np.hstack(
                [joints2d_xy, np.ones((21, 1), dtype=np.float32)])
            sample["joints3d"] = joints3d
            sample["joints2d"] = joints2d.astype(np.float32)
        else:
            sample["joints3d"] = np.zeros((21, 3), dtype=np.float32)
            sample["joints2d"] = np.zeros((21, 3), dtype=np.float32)

        if "handBeta" in ann:
            sample["beta"] = np.asarray(ann["handBeta"], dtype=np.float32)[:10]

        depth_path = self._cache_path(seq, fid)
        if depth_path is not None and os.path.exists(depth_path):
            sample["depth"] = np.load(depth_path)

        if self.transform is not None:
            sample = self.transform(sample)
        return sample
