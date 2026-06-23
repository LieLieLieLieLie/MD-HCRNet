"""
一次性预计算深度图并缓存到磁盘。

使用方法（在 code/ 目录下运行）:
    python precompute_depths.py --dataset freihand
    python precompute_depths.py --dataset rhd
    python precompute_depths.py --dataset all

完成后，深度图保存在:
    data/freihand/depth_cache/{idx:08d}.npy          shape=(1,256,192) float16
    data/RHD_published_v2/depth_cache/{idx:05d}.npy  shape=(1,256,192) float16

训练时配置文件中 depth_cache_dir 指向该目录，
模型会直接加载缓存而跳过 Depth-Anything-V2 在线推理。
预计耗时: FreiHAND ~15-30 分钟，RHD ~20-40 分钟。
"""
import argparse
import os
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


# ── 深度模型 ───────────────────────────────────────────────────────────────────
HF_MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
IMG_H, IMG_W = 256, 192
BATCH_SIZE   = 16          # 预计算时可以用小批量，稳定 VRAM

# ImageNet 反归一化（ToTensor 做了归一化，这里无需，直接读原始图像）

def load_depth_model(device):
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    print(f"  Loading {HF_MODEL_ID} ...")
    processor = AutoImageProcessor.from_pretrained(HF_MODEL_ID)
    model     = AutoModelForDepthEstimation.from_pretrained(HF_MODEL_ID)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return processor, model


@torch.no_grad()
def compute_depth_batch(pil_images, processor, model, device):
    """
    Args:
        pil_images: list of PIL.Image (raw RGB)
    Returns:
        depths: (N, 1, IMG_H, IMG_W) float16 numpy
    """
    inputs = processor(images=pil_images, return_tensors="pt")
    inputs = {k: v.to(device=device, dtype=torch.float32) for k, v in inputs.items()}
    outputs = model(**inputs)
    depth = outputs.predicted_depth.float()          # (N, H_out, W_out)
    depth = F.interpolate(
        depth.unsqueeze(1), size=(IMG_H, IMG_W),
        mode="bilinear", align_corners=False
    )                                                 # (N, 1, H, W)
    return depth.cpu().to(torch.float16).numpy()


# ── FreiHAND ───────────────────────────────────────────────────────────────────
def precompute_freihand(device):
    root       = Path("data/freihand")
    cache_dir  = root / "depth_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir    = root / "training" / "rgb"

    # FreiHAND: 32560 base images (0 ~ 32559)
    n_base = 32560
    # Collect indices that still need processing
    todo = [i for i in range(n_base)
            if not (cache_dir / f"{i:08d}.npy").exists()]

    if not todo:
        print("[FreiHAND] 所有深度图已缓存，跳过。")
        return

    print(f"[FreiHAND] 待计算: {len(todo)} / {n_base} 张图像")
    processor, model = load_depth_model(device)
    t0 = time.time()

    for batch_start in tqdm(range(0, len(todo), BATCH_SIZE),
                            desc="FreiHAND depth"):
        batch_ids = todo[batch_start: batch_start + BATCH_SIZE]
        pil_imgs  = []
        for idx in batch_ids:
            img_path = rgb_dir / f"{idx:08d}.jpg"
            pil_imgs.append(Image.open(img_path).convert("RGB").resize(
                (IMG_W, IMG_H), Image.BILINEAR))

        depths = compute_depth_batch(pil_imgs, processor, model, device)

        for i, idx in enumerate(batch_ids):
            np.save(cache_dir / f"{idx:08d}.npy", depths[i])

    elapsed = time.time() - t0
    print(f"[FreiHAND] 完成。耗时 {elapsed/60:.1f} 分钟。缓存于 {cache_dir}")


# ── RHD ───────────────────────────────────────────────────────────────────────
def precompute_rhd(device):
    root      = Path("data/RHD_published_v2")
    cache_dir = root / "depth_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # RHD: training (41258) then evaluation (2728), indexed 0…43985
    all_imgs = []
    for split in ("training", "evaluation"):
        color_dir = root / split / "color"
        if not color_dir.is_dir():
            continue
        all_imgs.extend(sorted(color_dir.glob("*.png")))

    todo_ids = [i for i, _ in enumerate(all_imgs)
                if not (cache_dir / f"{i:05d}.npy").exists()]

    if not todo_ids:
        print("[RHD] 所有深度图已缓存，跳过。")
        return

    print(f"[RHD] 待计算: {len(todo_ids)} / {len(all_imgs)} 张图像")
    processor, model = load_depth_model(device)
    t0 = time.time()

    for batch_start in tqdm(range(0, len(todo_ids), BATCH_SIZE),
                            desc="RHD depth"):
        batch_ids = todo_ids[batch_start: batch_start + BATCH_SIZE]
        pil_imgs  = [Image.open(all_imgs[i]).convert("RGB").resize(
                         (IMG_W, IMG_H), Image.BILINEAR)
                     for i in batch_ids]

        depths = compute_depth_batch(pil_imgs, processor, model, device)

        for i, idx in enumerate(batch_ids):
            np.save(cache_dir / f"{idx:05d}.npy", depths[i])

    elapsed = time.time() - t0
    print(f"[RHD] 完成。耗时 {elapsed/60:.1f} 分钟。缓存于 {cache_dir}")


# ── 入口 ──────────────────────────────────────────────────────────────────────
def _decode_ho3d_depth(depth_path: Path):
    """Decode HO3D RGB-packed depth PNG to metres."""
    bgr = cv2.imread(str(depth_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(depth_path)
    depth_scale = 0.00012498664727900177
    depth = bgr[:, :, 2].astype(np.float32) + \
            bgr[:, :, 1].astype(np.float32) * 256.0
    depth = depth * depth_scale
    depth = cv2.resize(depth, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
    return depth[None].astype(np.float16)


def _read_ho3d_entries(root: Path, split_name: str):
    list_path = root / f"{split_name}.txt"
    if not list_path.exists():
        return []
    entries = []
    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            seq, fid = line.split("/")
            entries.append((split_name, seq, fid,
                            root / split_name / seq / "depth" / f"{fid}.png"))
    return entries


def precompute_ho3d(device):
    root = Path("data/HO3D")
    cache_dir = root / "depth_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    all_entries = _read_ho3d_entries(root, "train") + \
                  _read_ho3d_entries(root, "evaluation")
    todo = []
    for split_name, seq, fid, depth_path in all_entries:
        out_path = cache_dir / split_name / f"{seq}_{fid}.npy"
        if not out_path.exists():
            todo.append((split_name, seq, fid, depth_path, out_path))

    if not todo:
        print("[HO3D] All depth maps are cached, skipping.")
        return

    print(f"[HO3D] Need to cache {len(todo)} / {len(all_entries)} depth maps.")
    t0 = time.time()
    for split_name, seq, fid, depth_path, out_path in tqdm(todo, desc="HO3D depth"):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, _decode_ho3d_depth(depth_path))

    elapsed = time.time() - t0
    print(f"[HO3D] Done in {elapsed/60:.1f} min. Cache: {cache_dir}")


def main():
    parser = argparse.ArgumentParser(description="预计算深度图缓存")
    parser.add_argument("--dataset", choices=["freihand", "rhd", "ho3d", "all"],
                        default="freihand")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    if args.dataset in ("freihand", "all"):
        precompute_freihand(device)
    if args.dataset in ("rhd", "all"):
        precompute_rhd(device)
    if args.dataset in ("ho3d", "all"):
        precompute_ho3d(device)


if __name__ == "__main__":
    main()
