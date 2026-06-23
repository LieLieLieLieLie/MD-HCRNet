"""
Baseline training script — identical experimental conditions as MD-HCRNet.

所有 baseline 与 MD-HCRNet 使用完全相同的训练设置：
  • 数据集 / 分割 / 增广
  • Epochs / batch size / optimizer / lr schedule / AMP
  • 评估指标

训练结果记录：
  • 每个 epoch 的 loss → outputs/freihand/tables/training_curves.xlsx  (追加/覆盖行)
  • 训练超参数快照   → outputs/freihand/tables/training_config.xlsx   (追加/覆盖行)
  • 最优模型        → outputs/freihand/models/{key}_best.pth
  • 最新模型        → outputs/freihand/models/{key}_latest.pth        (每5轮覆盖)

Usage:
    python train_baseline.py --model i2l         --config config/freihand.yaml
    python train_baseline.py --model handoccnet  --config config/freihand.yaml
    python train_baseline.py --model graphhand   --config config/freihand.yaml
    python train_baseline.py --model lightatt    --config config/freihand.yaml
    python train_baseline.py --model depthfusion --config config/freihand.yaml

After training, run eval_all.py to compare all models.
"""
import argparse
import math
import os
import time
import yaml
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp   import autocast
from torch.cuda.amp import GradScaler
from tqdm        import tqdm

from data             import build_dataloaders
from utils            import MPJPE, PA_MPJPE
from utils.table_logger import TableLogger, TrainingCurveLogger
from baselines        import build_baseline, BASELINE_REGISTRY


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


# ── Loss (joints3d L1 — works for all baselines) ─────────────────────────────

class BaselineLoss(nn.Module):
    def forward(self, pred: dict, gt: dict) -> dict:
        losses = {}
        if "joints3d" in pred and "joints3d" in gt:
            losses["joints3d"] = F.l1_loss(pred["joints3d"], gt["joints3d"])
        losses["total"] = sum(losses.values()) if losses else torch.tensor(0.0)
        return losses


# ── Utilities ─────────────────────────────────────────────────────────────────

def _fmt(s):
    s = int(s); h, r = divmod(s, 3600); m, sc = divmod(r, 60)
    return f"{h}h {m:02d}m {sc:02d}s" if h else f"{m}m {sc:02d}s"

def _sep(c="─", w=72): print(c * w)
def _now(): return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ── Build optimizer and scheduler (same as MD-HCRNet) ────────────────────────

def build_optimizer(model: nn.Module, cfg):
    return torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )

def build_scheduler(optimizer, cfg):
    def lr_lambda(ep):
        if ep < cfg.train.warmup_epochs:
            return ep / max(1, cfg.train.warmup_epochs)
        p = (ep - cfg.train.warmup_epochs) / max(1, cfg.train.epochs - cfg.train.warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * p))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── One epoch ─────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion,
                    device, amp_dtype, use_amp, scaler):
    model.train()
    total_loss, times = 0.0, []

    bar = tqdm(loader, desc="  Train", leave=False, dynamic_ncols=True)
    for batch in bar:
        t0    = time.time()
        rgb   = batch["image"].to(device)
        depth = batch["depth"].to(device) if "depth" in batch else None
        focal = batch["focal"].to(device)
        cx    = batch["cx"].to(device)
        cy    = batch["cy"].to(device)
        gt    = {"joints3d": batch["joints3d"].to(device)}

        optimizer.zero_grad()
        with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            pred   = model(rgb, focal, cx, cy, depth)
            losses = criterion(pred, gt)

        if use_amp:
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
        else:
            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        ms = (time.time() - t0) * 1000
        times.append(ms)
        total_loss += losses["total"].item()
        bar.set_postfix(loss=f"{losses['total'].item():.4f}",
                        ms=f"{sum(times[-50:])/len(times[-50:]):.0f}ms")

    avg_ms = sum(times) / len(times)
    return total_loss / len(loader), avg_ms


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total = 0.0
    for batch in tqdm(loader, desc="  Val  ", leave=False, dynamic_ncols=True):
        rgb   = batch["image"].to(device)
        focal = batch["focal"].to(device)
        cx    = batch["cx"].to(device)
        cy    = batch["cy"].to(device)
        depth = batch["depth"].to(device) if "depth" in batch else None
        gt    = {"joints3d": batch["joints3d"].to(device)}
        pred  = model(rgb, focal, cx, cy, depth)
        total += criterion(pred, gt)["total"].item()
    return total / len(loader)


# ── Main training loop ────────────────────────────────────────────────────────

def run_training(model_key: str, cfg, device: torch.device, resume: str = ""):
    # ── PyTorch 2.0.x workaround: flash attention硬编码is_sm80检查在Ada Lovelace
    #    (SM89)上会失败，禁掉flash SDP回退到math backend，对精度无影响 ──────────
    if device.type == "cuda":
        try:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        except AttributeError:
            pass  # PyTorch版本不支持时忽略

    # ── Output directories ────────────────────────────────────────────────────
    base_dir   = cfg.train.output_dir
    models_dir = os.path.join(base_dir, "models")
    tables_dir = os.path.join(base_dir, "tables")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    # ── Table loggers ──────────────────────────────────────────────────────────
    curve_log  = TrainingCurveLogger(os.path.join(tables_dir, "training_curves.xlsx"))
    config_log = TableLogger(os.path.join(tables_dir, "training_config.xlsx"))

    # ── Build model ───────────────────────────────────────────────────────────
    model     = build_baseline(model_key, cfg).to(device)
    model_name = getattr(model, "MODEL_NAME", model_key)
    criterion  = BaselineLoss().to(device)
    if not resume:
        curve_log.clear_model(model_name)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader = build_dataloaders(cfg)

    # ── Optimizer / scheduler ─────────────────────────────────────────────────
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    # ── AMP ───────────────────────────────────────────────────────────────────
    use_amp    = getattr(cfg.train, "amp", False) and device.type == "cuda"
    dtype_str  = getattr(cfg.train, "amp_dtype", "bfloat16")
    amp_dtype  = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16
    scaler     = GradScaler(enabled=(use_amp and amp_dtype == torch.float16))

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

    _sep("═")
    print(f"  Baseline Training  |  {_now()}")
    print(f"  Model     : {model_name}")
    print(f"  Params    : {n_params:.1f}M trainable")
    print(f"  Device    : {device}")
    print(f"  Epochs    : {cfg.train.epochs}  |  Batch: {cfg.data.batch_size}")
    print(f"  Subset    : train={getattr(cfg.data, 'train_subset', 0)}"
          f"  val={getattr(cfg.data, 'val_subset', 0)}"
          f"  seed={getattr(cfg.data, 'subset_seed', '-')}")
    print(f"  LR        : {cfg.train.lr}  |  WD: {cfg.train.weight_decay}")
    print(f"  AMP       : {use_amp} ({dtype_str})")
    print(f"  Models →  : {models_dir}")
    print(f"  Tables →  : {tables_dir}")
    _sep("═")

    # Log config
    config_log.update(model_name, {
        "Params (M)":   round(n_params, 2),
        "Epochs":       cfg.train.epochs,
        "BatchSize":    cfg.data.batch_size,
        "LR":           cfg.train.lr,
        "WeightDecay":  cfg.train.weight_decay,
        "AMP":          f"{use_amp}/{dtype_str}",
        "Dataset":      cfg.data.dataset,
        "TrainDate":    _now(),
    })

    best_val    = float("inf")
    start_epoch = 0
    best_path   = os.path.join(models_dir, f"{model_key}_best.pth")
    latest_path = os.path.join(models_dir, f"{model_key}_latest.pth")
    if resume:
        ckpt = torch.load(resume, map_location=device)
        if ckpt.get("model_key") not in (None, model_key):
            raise ValueError(f"Checkpoint model_key={ckpt.get('model_key')} does not match {model_key}")
        model.load_state_dict(ckpt["model"])
        best_val = float(ckpt.get("best_val", best_val))
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        print(f"  Resume    : {resume}  (start epoch {start_epoch + 1})")
    else:
        for stale_path in (best_path, latest_path):
            if os.path.exists(stale_path):
                os.remove(stale_path)
    save_freq   = getattr(cfg.train, "save_freq", 5)
    t0          = time.time()
    _loss_scale_train: float = 0.0   # epoch-1 基准，用于归一化
    _loss_scale_val:   float = 0.0

    for epoch in range(start_epoch, cfg.train.epochs):
        t_ep = time.time()

        train_loss, avg_ms = train_one_epoch(
            model, train_loader, optimizer, criterion,
            device, amp_dtype, use_amp, scaler)
        val_loss   = validate(model, val_loader, criterion, device)
        scheduler.step()

        if getattr(cfg.train, "empty_cache_each_epoch", False) and device.type == "cuda":
            torch.cuda.empty_cache()

        ep_sec = time.time() - t_ep
        lr     = optimizer.param_groups[0]["lr"]
        is_best = val_loss < best_val

        # ── 归一化后记录训练曲线（epoch 0 建立基准）───────────────────────
        if epoch == 0 or _loss_scale_train == 0.0 or _loss_scale_val == 0.0:
            _loss_scale_train = train_loss if train_loss > 0 else 1.0
            _loss_scale_val   = val_loss   if val_loss   > 0 else 1.0
        curve_log.log(model_name, epoch,
                      train_loss / _loss_scale_train,
                      val_loss   / _loss_scale_val,
                      lr, avg_ms)

        if is_best:
            best_val = val_loss

        # ── Save checkpoints ───────────────────────────────────────────────
        ckpt = {"epoch": epoch, "model": model.state_dict(),
                "best_val": best_val, "model_key": model_key}

        if save_freq > 0 and (epoch + 1) % save_freq == 0:
            torch.save(ckpt, latest_path)

        if is_best:
            torch.save(ckpt, best_path)

        # ── Print epoch summary ────────────────────────────────────────────
        remaining = ep_sec * (cfg.train.epochs - epoch - 1)
        _sep()
        print(f"  Epoch [{epoch+1:03d}/{cfg.train.epochs}]  {_now()}")
        print(f"  Train: {train_loss:.6f}  Val: {val_loss:.6f}  "
              f"{'★ Best' if is_best else f'best={best_val:.6f}'}")
        print(f"  LR: {lr:.2e}  |  {avg_ms:.0f}ms/batch  |  "
              f"epoch: {_fmt(ep_sec)}  ETA: {_fmt(remaining)}")
        _sep()

    _sep("═")
    print(f"  Done!  {model_name}  |  Best Val: {best_val:.6f}")
    print(f"  Best   → {best_path}")
    print(f"  Latest → {latest_path}")
    _sep("═")
    return best_path


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model", required=True,
        choices=list(BASELINE_REGISTRY.keys()),
        help="Baseline model key. Choices: " + ", ".join(BASELINE_REGISTRY.keys()))
    parser.add_argument("--config", default="config/freihand.yaml")
    parser.add_argument("--resume", default="", help="Resume from a baseline checkpoint")
    args   = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg    = load_config(args.config)

    run_training(args.model, cfg, device, resume=args.resume)


if __name__ == "__main__":
    main()
