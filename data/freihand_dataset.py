"""
FreiHAND dataset loader.

Directory layout expected under data_root/freihand/:
    training/
        rgb/           00000000.jpg … 00032559.jpg
        mask/          00000000.jpg …
    evaluation/
        rgb/           00000000.jpg … 00003959.jpg
    training_mano.json     — list of MANO param dicts
    training_K.json        — list of 3×3 camera intrinsics
    training_xyz.json      — list of 21-joint 3D coords (in mm)
    evaluation_K.json
    evaluation_xyz.json    (ground-truth for online eval only)

Download: https://lmb.informatik.uni-freiburg.de/resources/datasets/FreihandDataset.en.html
"""
import os, json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

# FreiHAND GT has 21 joints (wrist + 5×4 fingers including tips).
# We rearrange them from FreiHAND order → MANO-21 order so that GT aligns with
# the model's output convention (smplx 16 kinematic joints + 5 fingertips):
#
#   MANO21 order:
#     0:Wrist
#     1-3:   Index  (MCP,PIP,DIP)   4-6:  Middle (MCP,PIP,DIP)
#     7-9:   Pinky  (MCP,PIP,DIP)   10-12:Ring   (MCP,PIP,DIP)
#     13-15: Thumb  (CMC,MCP,IP)
#     16:Index TIP  17:Middle TIP  18:Pinky TIP  19:Ring TIP  20:Thumb TIP
#
#   FreiHAND order:
#     0:Wrist
#     1-4:  Thumb (CMC,MCP,IP,TIP)   5-8:  Index  (MCP,PIP,DIP,TIP)
#     9-12: Middle(MCP,PIP,DIP,TIP)  13-16:Ring   (MCP,PIP,DIP,TIP)
#     17-20:Pinky (MCP,PIP,DIP,TIP)
FREIHAND_TO_MANO21 = [0,                       # Wrist
                       5, 6, 7,                 # Index  MCP,PIP,DIP
                       9, 10, 11,               # Middle MCP,PIP,DIP
                       17, 18, 19,              # Pinky  MCP,PIP,DIP
                       13, 14, 15,              # Ring   MCP,PIP,DIP
                       1, 2, 3,                 # Thumb  CMC,MCP,IP
                       8, 12, 20, 16, 4]        # TIPs: Index,Middle,Pinky,Ring,Thumb


class FreiHANDDataset(Dataset):
    """
    Returns a sample dict compatible with the transforms in data/transforms.py.

    Note: FreiHAND provides ground-truth 3D joints in metre units.
    We convert to metres here (they are already in metres in the JSON).
    """

    # How many distinct scene images there are (before 4× augmentation)
    NUM_TRAIN_SCENES = 32560

    def __init__(self, root: str, split: str = "train", transform=None,
                 use_aug: bool = True, depth_cache_dir: str = None):
        """
        Args:
            root:      path to the freihand/ directory
            split:     "train" or "val" or "eval"
            transform: composed transform callable
            use_aug:   if True, use all 4 augmented variants; otherwise only
                       the first set (no colour aug)
        """
        super().__init__()
        self.root            = root
        self.split           = split
        self.transform       = transform
        self.depth_cache_dir = depth_cache_dir

        if split in ("train", "val"):
            rgb_dir = os.path.join(root, "training", "rgb")
            with open(os.path.join(root, "training_K.json"))    as f:
                all_K    = json.load(f)
            with open(os.path.join(root, "training_xyz.json"))  as f:
                all_xyz  = json.load(f)
            with open(os.path.join(root, "training_mano.json")) as f:
                all_mano = json.load(f)

            n_aug  = 4 if use_aug else 1
            n_base = self.NUM_TRAIN_SCENES
            indices = list(range(n_base * n_aug))

            # 90/10 train-val split (by scene, not by augmented index)
            scene_ids  = list(range(n_base))
            val_scenes = set(scene_ids[int(0.9 * n_base):])

            if split == "train":
                self.indices = [i for i in indices
                                if (i % n_base) not in val_scenes]
            else:
                self.indices = [i for i in indices
                                if (i % n_base) in val_scenes]

            self.rgb_dir  = rgb_dir
            self.all_K    = all_K
            self.all_xyz  = all_xyz
            self.all_mano = all_mano
        else:
            # eval split — no GT labels
            self.rgb_dir  = os.path.join(root, "evaluation", "rgb")
            with open(os.path.join(root, "evaluation_K.json")) as f:
                self.all_K = json.load(f)
            self.all_xyz  = None
            self.all_mano = None
            n_eval = len(self.all_K)
            self.indices = list(range(n_eval))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]

        # ── RGB image ──────────────────────────────────────────────────────
        img_path = os.path.join(self.rgb_dir, f"{real_idx:08d}.jpg")
        image = np.array(Image.open(img_path).convert("RGB"))

        # ── Camera intrinsics ──────────────────────────────────────────────
        K = np.array(self.all_K[real_idx], dtype=np.float32)
        focal = float((K[0, 0] + K[1, 1]) / 2.0)
        cx    = float(K[0, 2])
        cy    = float(K[1, 2])

        sample = dict(image=image, focal=focal, cx=cx, cy=cy)

        # ── Ground-truth joints3d (metres) ─────────────────────────────────
        if self.all_xyz is not None:
            xyz21 = np.array(self.all_xyz[real_idx], dtype=np.float32)  # (21,3)
            xyz   = xyz21[FREIHAND_TO_MANO21]                            # (21,3) MANO order
            # project to 2D using K
            xyz_h = (K @ xyz.T).T
            joints2d_xy = xyz_h[:, :2] / xyz_h[:, 2:3]
            joints2d    = np.hstack([joints2d_xy, np.ones((21, 1), np.float32)])
            sample["joints2d"] = joints2d
            sample["joints3d"] = xyz
        else:
            sample["joints2d"] = np.zeros((21, 3), np.float32)
            sample["joints3d"] = np.zeros((21, 3), np.float32)

        # ── MANO parameters ────────────────────────────────────────────────
        # training_mano.json 格式: 每条是 [[pose(48), betas(10), trans(3)]]
        # 即外层 list 长度为 1，内层 flat list 长度为 61
        if self.all_mano is not None:
            mano = self.all_mano[real_idx]
            if isinstance(mano, dict):
                # dict 格式（兼容旧版）
                beta = np.array(mano["betas"], dtype=np.float32)[:10]
                pose = np.array(mano["pose"],  dtype=np.float32)[:48]
            else:
                # list 格式：[[pose(48), betas(10), trans(3)]]
                params = np.array(mano[0] if len(mano) == 1 else mano,
                                  dtype=np.float32)
                pose = params[:48]      # (48,)
                beta = params[48:58]    # (10,)
            sample["beta"]  = beta
            sample["theta"] = pose
        else:
            sample["beta"]  = np.zeros(10, dtype=np.float32)
            sample["theta"] = np.zeros(48, dtype=np.float32)

        # ── 预计算深度图（如果有缓存）──────────────────────────────────────
        cache_dir = getattr(self, "depth_cache_dir", None)
        if cache_dir is not None:
            depth_path = os.path.join(cache_dir, f"{real_idx:08d}.npy")
            if os.path.exists(depth_path):
                sample["depth"] = np.load(depth_path)  # (1, H, W) float16

        if self.transform is not None:
            sample = self.transform(sample)
        return sample
