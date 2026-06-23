"""
Data augmentation for hand pose estimation.

All transforms operate on a unified sample dict:
    image:     (H, W, 3)  uint8 numpy
    joints2d:  (21, 3)    float32 [x, y, visibility]
    focal:     float
    cx, cy:    float
"""
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


class RandomHorizontalFlip:
    """Flip image and mirror 2D joints (right hand → left hand convention)."""

    # Pair indices to swap when flipping a right-hand skeleton
    FLIP_PAIRS = []  # single-hand: no pairs needed, just mirror x

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        if np.random.rand() < self.prob:
            img = sample["image"]
            W = img.shape[1]
            sample["image"] = img[:, ::-1].copy()
            j2d = sample["joints2d"].copy()
            j2d[:, 0] = W - 1 - j2d[:, 0]
            sample["joints2d"] = j2d
            sample["cx"] = W - sample["cx"]
        return sample


class RandomRotation:
    def __init__(self, max_deg=30.0):
        self.max_deg = max_deg

    def __call__(self, sample):
        angle = np.random.uniform(-self.max_deg, self.max_deg)
        img = sample["image"]
        H, W = img.shape[:2]
        cx, cy = sample["cx"], sample["cy"]

        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        sample["image"] = cv2.warpAffine(img, M, (W, H),
                                          flags=cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_REFLECT)
        j2d = sample["joints2d"].copy()
        pts = np.hstack([j2d[:, :2], np.ones((len(j2d), 1))])
        j2d[:, :2] = (M @ pts.T).T
        sample["joints2d"] = j2d
        return sample


class RandomScale:
    def __init__(self, scale_range=(0.85, 1.15)):
        self.lo, self.hi = scale_range

    def __call__(self, sample):
        scale = np.random.uniform(self.lo, self.hi)
        img = sample["image"]
        H, W = img.shape[:2]
        nH, nW = int(H * scale), int(W * scale)
        img = cv2.resize(img, (nW, nH), interpolation=cv2.INTER_LINEAR)

        # Pad or crop back to original size
        out = np.zeros((H, W, 3), dtype=img.dtype)
        oh = min(nH, H)
        ow = min(nW, W)
        out[:oh, :ow] = img[:oh, :ow]
        sample["image"] = out

        sample["joints2d"][:, :2] *= scale
        sample["focal"] *= scale
        sample["cx"]    *= scale
        sample["cy"]    *= scale
        return sample


class ColorJitter:
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1):
        from torchvision import transforms
        self.jitter = transforms.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, sample):
        img = Image.fromarray(sample["image"])
        sample["image"] = np.array(self.jitter(img))
        return sample


class Resize:
    def __init__(self, height: int, width: int):
        self.H = height
        self.W = width

    def __call__(self, sample):
        img = sample["image"]
        orig_H, orig_W = img.shape[:2]
        sample["image"] = cv2.resize(img, (self.W, self.H),
                                     interpolation=cv2.INTER_LINEAR)
        scale_x = self.W / orig_W
        scale_y = self.H / orig_H
        j2d = sample["joints2d"].copy()
        j2d[:, 0] *= scale_x
        j2d[:, 1] *= scale_y
        sample["joints2d"] = j2d
        sample["focal"] = sample.get("focal", 500.0) * ((scale_x + scale_y) * 0.5)
        sample["cx"] = sample.get("cx", orig_W / 2) * scale_x
        sample["cy"] = sample.get("cy", orig_H / 2) * scale_y
        return sample


class ToTensor:
    """Normalise image to [0,1] and convert all arrays to torch tensors."""

    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    def __call__(self, sample):
        img = sample["image"].astype(np.float32) / 255.0
        img = (img - self.MEAN) / self.STD
        sample["image"]    = torch.from_numpy(img).permute(2, 0, 1).float()
        sample["joints2d"] = torch.from_numpy(sample["joints2d"]).float()
        for k in ("focal", "cx", "cy"):
            sample[k] = torch.tensor(sample[k], dtype=torch.float32)
        if "joints3d" in sample:
            sample["joints3d"] = torch.from_numpy(sample["joints3d"]).float()
        if "vertices" in sample:
            sample["vertices"] = torch.from_numpy(sample["vertices"]).float()
        if "beta" in sample:
            sample["beta"]  = torch.from_numpy(sample["beta"]).float()
        if "theta" in sample:
            sample["theta"] = torch.from_numpy(sample["theta"]).float()
        if "depth" in sample:
            d = sample["depth"]
            if not isinstance(d, torch.Tensor):
                d = torch.from_numpy(d.astype("float32"))
            if d.ndim == 2:
                d = d.unsqueeze(0)      # (H,W) → (1,H,W)
            sample["depth"] = d.float()
        return sample


def build_train_transforms(cfg) -> Compose:
    dcfg = cfg.data
    tfms = [Resize(dcfg.img_height, dcfg.img_width)]
    if not getattr(dcfg, "use_aug", True):
        tfms.append(ToTensor())
        return Compose(tfms)
    if getattr(dcfg, "random_flip",  False):
        tfms.append(RandomHorizontalFlip(0.5))
    if getattr(dcfg, "color_jitter", False):
        tfms.append(ColorJitter())
    tfms.append(RandomRotation(getattr(dcfg, "rotation_deg", 30)))
    tfms.append(RandomScale(getattr(dcfg, "crop_scale", [0.85, 1.15])))
    tfms.append(ToTensor())
    return Compose(tfms)


def build_val_transforms(cfg) -> Compose:
    dcfg = cfg.data
    return Compose([Resize(dcfg.img_height, dcfg.img_width), ToTensor()])
