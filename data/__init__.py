import os
import random
from torch.utils.data import DataLoader, Subset

from .freihand_dataset import FreiHANDDataset
from .rhd_dataset      import RHDDataset
from .ho3d_dataset     import HO3DDataset
from .transforms       import build_train_transforms, build_val_transforms


def _subset_dataset(dataset, max_samples, seed):
    """Deterministically keep a fixed subset so every method sees the same data."""
    if max_samples is None or int(max_samples) <= 0:
        return dataset
    max_samples = min(int(max_samples), len(dataset))
    rng = random.Random(int(seed))
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    indices = sorted(indices[:max_samples])
    return Subset(dataset, indices)


def build_dataloaders(cfg):
    """Return (train_loader, val_loader) for the dataset specified in cfg."""
    dcfg = cfg.data
    root = dcfg.root

    use_aug = getattr(dcfg, "use_aug", True)
    depth_cache_dir = getattr(dcfg, "depth_cache_dir", None)
    subset_seed = getattr(dcfg, "subset_seed", 2026)

    if dcfg.dataset == "freihand":
        train_ds = FreiHANDDataset(
            os.path.join(root, "freihand"), split="train",
            transform=build_train_transforms(cfg),
            use_aug=use_aug,
            depth_cache_dir=depth_cache_dir)
        val_ds = FreiHANDDataset(
            os.path.join(root, "freihand"), split="val",
            transform=build_val_transforms(cfg),
            use_aug=False,
            depth_cache_dir=depth_cache_dir)
    elif dcfg.dataset == "rhd":
        train_ds = RHDDataset(
            os.path.join(root, "RHD_published_v2"), split="training",
            transform=build_train_transforms(cfg),
            depth_cache_dir=depth_cache_dir)
        val_ds = RHDDataset(
            os.path.join(root, "RHD_published_v2"), split="evaluation",
            transform=build_val_transforms(cfg),
            depth_cache_dir=depth_cache_dir)
    elif dcfg.dataset == "ho3d":
        ho3d_root = os.path.join(root, "HO3D")
        val_ratio = getattr(dcfg, "val_ratio", 0.1)
        train_ds = HO3DDataset(
            ho3d_root, split="train",
            transform=build_train_transforms(cfg),
            depth_cache_dir=depth_cache_dir,
            val_ratio=val_ratio,
            split_seed=subset_seed)
        val_ds = HO3DDataset(
            ho3d_root, split="val",
            transform=build_val_transforms(cfg),
            depth_cache_dir=depth_cache_dir,
            val_ratio=val_ratio,
            split_seed=subset_seed)
    else:
        raise ValueError(f"Unknown dataset: {dcfg.dataset}")

    train_ds = _subset_dataset(train_ds, getattr(dcfg, "train_subset", 0), subset_seed)
    val_ds = _subset_dataset(val_ds, getattr(dcfg, "val_subset", 0), subset_seed + 1)

    pin_memory = bool(getattr(dcfg, "pin_memory", True))
    persistent_workers = bool(getattr(dcfg, "persistent_workers", False)) and int(dcfg.num_workers) > 0

    train_loader = DataLoader(
        train_ds, batch_size=dcfg.batch_size, shuffle=True,
        num_workers=dcfg.num_workers, pin_memory=pin_memory,
        drop_last=True, persistent_workers=persistent_workers)
    val_loader = DataLoader(
        val_ds, batch_size=dcfg.batch_size, shuffle=False,
        num_workers=dcfg.num_workers, pin_memory=pin_memory,
        persistent_workers=persistent_workers)

    return train_loader, val_loader
