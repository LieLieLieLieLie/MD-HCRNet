"""
Training script for MD-HCRNet.

Usage:
    python train.py --config config/default.yaml
    python train.py --config config/default.yaml --resume outputs/latest.pth
"""
import argparse
import math
import os
import time
import yaml
from datetime import datetime, timedelta
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from tqdm import tqdm

from models import MDHCRNet
from losses import MDHCRNetLoss
from data   import build_dataloaders
from utils.visualization import apply_style, plot_training_curves, save_pdf, save_xlsx
from utils.table_logger  import TableLogger, TrainingCurveLogger
from eval import run_eval


def _make_output_dirs(base: str):
    """创建标准三目录：figures/ tables/ models/"""
    dirs = {}
    for name in ("figures", "tables", "models"):
        p = os.path.join(base, name)
        os.makedirs(p, exist_ok=True)
        dirs[name] = p
    return dirs


# ── 格式化时间 ────────────────────────────────────────────────────────────────

def _fmt(seconds: float) -> str:
    """把秒数格式化成 Xh Ym Zs 或 Ym Zs。"""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s   = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ── Config helpers ────────────────────────────────────────────────────────────

def _coerce(v):
    """将 YAML 解析出的字符串数值（如 '1e-4'）转换为 float/int。"""
    if not isinstance(v, str):
        return v
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


def _dict_to_ns(d):
    ns = SimpleNamespace()
    for k, v in d.items():
        if isinstance(v, dict):
            setattr(ns, k, _dict_to_ns(v))
        elif isinstance(v, list):
            setattr(ns, k, [_coerce(i) for i in v])
        else:
            setattr(ns, k, _coerce(v))
    return ns


def load_config(path: str):
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return _dict_to_ns(raw)


# ── Optimiser / scheduler ─────────────────────────────────────────────────────

def build_optimizer(model: nn.Module, cfg):
    tcfg = cfg.train
    backbone_lr = getattr(tcfg, "backbone_lr", tcfg.lr * 0.1)
    refine_lr = getattr(tcfg, "refine_lr", tcfg.lr)

    backbone_params, refine_params, other_params = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(p)
        elif any(k in name for k in (
            "global_joint_refine",
            "joint_query_refine",
            "direct_joint_decoder",
            "stage2",
        )):
            refine_params.append(p)
        else:
            other_params.append(p)

    param_groups = []
    if other_params:
        param_groups.append({"params": other_params, "lr": tcfg.lr, "name": "base"})
    if refine_params:
        param_groups.append({"params": refine_params, "lr": refine_lr, "name": "refine"})
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_lr, "name": "backbone"})

    return torch.optim.AdamW(
        param_groups,
        weight_decay=tcfg.weight_decay,
    )


def build_scheduler(optimizer, cfg):
    tcfg = cfg.train

    def lr_lambda(epoch):
        if epoch < tcfg.warmup_epochs:
            return epoch / max(1, tcfg.warmup_epochs)
        progress = (epoch - tcfg.warmup_epochs) / \
                   max(1, tcfg.epochs - tcfg.warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Train / validate one epoch ────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion,
                    device, epoch, cfg, scaler, amp_dtype):
    model.train()
    tcfg     = cfg.train
    use_amp  = getattr(tcfg, "amp", False) and device.type == "cuda"

    total_loss  = 0.0
    loss_detail = {}
    batch_times = []

    bar = tqdm(loader, desc=f"  Train E{epoch:03d}", leave=False,
               dynamic_ncols=True)

    for step, batch in enumerate(bar):
        t_batch = time.time()

        rgb   = batch["image"].to(device)
        focal = batch["focal"].to(device)
        cx    = batch["cx"].to(device)
        cy    = batch["cy"].to(device)
        depth = batch["depth"].to(device) if "depth" in batch else None
        gt    = {k: batch[k].to(device)
                 for k in ("joints3d", "joints2d", "vertices", "beta", "theta")
                 if k in batch}

        optimizer.zero_grad()

        with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            pred   = model(rgb, focal, cx, cy, depth=depth)
            losses = criterion(pred, gt)

        if use_amp:
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # ── 计时 ─────────────────────────────────────────────────────────
        batch_ms = (time.time() - t_batch) * 1000
        batch_times.append(batch_ms)
        avg_ms = sum(batch_times[-50:]) / len(batch_times[-50:])  # 近50批次均值

        total_loss += losses["total"].item()
        for k, v in losses.items():
            loss_detail[k] = loss_detail.get(k, 0.0) + v.item()

        # ── tqdm 后缀：实时 loss + 每批耗时 ──────────────────────────────
        bar.set_postfix(
            loss=f"{losses['total'].item():.4f}",
            ms=f"{avg_ms:.0f}ms/batch",
        )

    n = len(loader)
    avg_loss    = total_loss / n
    avg_details = {k: v / n for k, v in loss_detail.items()}
    avg_batch_ms = sum(batch_times) / len(batch_times)

    return avg_loss, avg_details, avg_batch_ms


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss  = 0.0
    loss_detail = {}
    total_mpjpe = 0.0
    total_count = 0

    bar = tqdm(loader, desc="  Val  ", leave=False, dynamic_ncols=True)
    for batch in bar:
        rgb   = batch["image"].to(device)
        focal = batch["focal"].to(device)
        cx    = batch["cx"].to(device)
        cy    = batch["cy"].to(device)
        depth = batch["depth"].to(device) if "depth" in batch else None
        gt    = {k: batch[k].to(device)
                 for k in ("joints3d", "joints2d", "vertices", "beta", "theta")
                 if k in batch}

        pred   = model(rgb, focal, cx, cy, depth=depth)
        losses = criterion(pred, gt)
        pred_r = pred["joints3d"] - pred["joints3d"][:, :1, :]
        gt_r = gt["joints3d"] - gt["joints3d"][:, :1, :]
        mpjpe = torch.linalg.norm(pred_r - gt_r, dim=-1).mean()
        total_mpjpe += mpjpe.item() * rgb.shape[0]
        total_count += rgb.shape[0]

        total_loss += losses["total"].item()
        for k, v in losses.items():
            loss_detail[k] = loss_detail.get(k, 0.0) + v.item()

        bar.set_postfix(loss=f"{losses['total'].item():.4f}")

    n = len(loader)
    return total_loss / n, {k: v / n for k, v in loss_detail.items()}, (total_mpjpe / max(total_count, 1)) * 1000.0


# ── 打印分隔线 ────────────────────────────────────────────────────────────────

def _sep(char="─", width=72):
    print(char * width)


def _print_epoch_summary(epoch, total_epochs, train_loss, val_loss,
                         lr, epoch_sec, elapsed_total,
                         avg_batch_ms, best_val, is_best):
    remaining = epoch_sec * (total_epochs - epoch - 1)
    eta_str   = _fmt(remaining)
    _sep()
    print(
        f"  Epoch [{epoch+1:03d}/{total_epochs}]  "
        f"{_now()}\n"
        f"  Train Loss : {train_loss:.6f}    "
        f"Val Loss : {val_loss:.6f}  "
        f"{'★ Best' if is_best else ''}\n"
        f"  Best Val   : {best_val:.6f}    "
        f"LR : {lr:.2e}\n"
        f"  Epoch Time : {_fmt(epoch_sec)}    "
        f"Elapsed : {_fmt(elapsed_total)}    "
        f"ETA : {eta_str}\n"
        f"  Batch Avg  : {avg_batch_ms:.1f} ms/batch"
    )
    _sep()


# ── 单次训练（可复用）────────────────────────────────────────────────────────

def run_training(config_path: str, device: torch.device,
                 resume: str = "", finetune: str = "") -> str:
    """
    完整训练一个数据集，返回 best.pth 路径。

    Args:
        config_path: yaml 配置文件路径
        device:      torch device
        resume:      断点续训 checkpoint（完整恢复 optimizer/scheduler）
        finetune:    跨数据集微调起点（只加载模型权重，重置 optimizer）
    """
    cfg = load_config(config_path)

    MODEL_NAME = "MD-HCR"

    if device.type == "cuda":
        try:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        except AttributeError:
            pass

    apply_style()
    os.makedirs(cfg.train.output_dir, exist_ok=True)
    dirs = _make_output_dirs(cfg.train.output_dir)

    # ── Shared table loggers (same files used by baselines) ────────────────
    curve_log  = TrainingCurveLogger(
        os.path.join(dirs["tables"], "training_curves.xlsx"))
    config_log = TableLogger(
        os.path.join(dirs["tables"], "training_config.xlsx"))
    if not resume:
        curve_log.clear_model(MODEL_NAME)

    best_path = os.path.join(dirs["models"], "md_hcrnet_best.pth")
    latest_path = os.path.join(dirs["models"], "latest.pth")
    if not resume and not finetune:
        for stale_path in (best_path, latest_path):
            if os.path.exists(stale_path):
                os.remove(stale_path)

    _sep("═")
    print(f"  MD-HCRNet Training  |  {_now()}")
    print(f"  Device   : {device}")
    print(f"  Config   : {config_path}")
    print(f"  Dataset  : {cfg.data.dataset}")
    print(f"  Epochs   : {cfg.train.epochs}")
    print(f"  Batch    : {cfg.data.batch_size}")
    print(f"  Subset   : train={getattr(cfg.data, 'train_subset', 0)}"
          f"  val={getattr(cfg.data, 'val_subset', 0)}"
          f"  seed={getattr(cfg.data, 'subset_seed', '-')}")
    print(f"  Outputs  : {cfg.train.output_dir}")
    _sep("═")

    t0_build = time.time()
    model    = MDHCRNet(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total  = sum(p.numel() for p in model.parameters())
    print(f"  Model built in {_fmt(time.time()-t0_build)}"
          f"  |  可训练参数: {n_params/1e6:.1f}M / 总参数: {n_total/1e6:.1f}M")

    t0_data = time.time()
    train_loader, val_loader = build_dataloaders(cfg)
    print(f"  Dataset loaded in {_fmt(time.time()-t0_data)}"
          f"  |  Train batches: {len(train_loader)}"
          f"  |  Val batches: {len(val_loader)}")

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    criterion = MDHCRNetLoss(cfg).to(device)

    use_amp    = getattr(cfg.train, "amp", False) and device.type == "cuda"
    _dtype_str = getattr(cfg.train, "amp_dtype", "bfloat16")
    amp_dtype  = torch.bfloat16 if _dtype_str == "bfloat16" else torch.float16
    scaler     = GradScaler(enabled=(use_amp and amp_dtype == torch.float16))
    if use_amp:
        print(f"  AMP 已开启: {_dtype_str}  |  GradScaler: "
              f"{'on' if amp_dtype==torch.float16 else 'off (BF16不需要)'}")

    # ── 记录训练配置到共享表格 ──────────────────────────────────────────────
    config_log.update(MODEL_NAME, {
        "Params (M)":   round(n_params / 1e6, 2),
        "Epochs":       cfg.train.epochs,
        "BatchSize":    cfg.data.batch_size,
        "LR":           cfg.train.lr,
        "WeightDecay":  cfg.train.weight_decay,
        "AMP":          f"{use_amp}/{_dtype_str}",
        "Dataset":      cfg.data.dataset,
        "TrainDate":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })

    start_epoch   = 0
    best_val_loss = float("inf")
    best_val_mpjpe = float("inf")
    train_loss_hist, val_loss_hist = [], []
    # 归一化基准：epoch 1 的 loss，记录后所有值 ÷ 该基准使曲线从 1.0 出发
    # 不同模型损失量纲不同（MD-HCRNet 含像素坐标 2D 项，baseline 仅含 3D 米制项）
    # 归一化后可在同一图上对比相对收敛速度，论文中需声明此处理
    _loss_scale_train: float = 0.0
    _loss_scale_val:   float = 0.0

    # ── 断点续训（完整恢复）────────────────────────────────────────────────
    if resume:
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch       = ckpt["epoch"] + 1
        best_val_loss     = ckpt.get("best_val_loss", float("inf"))
        best_val_mpjpe    = ckpt.get("best_val_mpjpe", best_val_loss)
        train_loss_hist   = ckpt.get("train_loss_hist", [])
        val_loss_hist     = ckpt.get("val_loss_hist",   [])
        print(f"  ↩  断点续训 epoch {ckpt['epoch']}  |  best_val={best_val_loss:.6f}")

    # ── 跨数据集微调（只加载模型权重）────────────────────────────────────────
    elif finetune:
        ckpt = torch.load(finetune, map_location=device)
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        print(f"  🔧 微调初始化: {finetune}  (optimizer 重置，从 epoch 0 开始)")
        if missing:
            print(f"  Partial load missing keys: {len(missing)}")
        if unexpected:
            print(f"  Partial load unexpected keys: {len(unexpected)}")

    t_train_start = time.time()

    for epoch in range(start_epoch, cfg.train.epochs):
        t_epoch = time.time()

        train_loss, train_det, avg_ms = train_one_epoch(
            model, train_loader, optimizer, criterion,
            device, epoch, cfg, scaler, amp_dtype)
        val_loss, val_det, val_mpjpe = validate(model, val_loader, criterion, device)
        scheduler.step()

        if getattr(cfg.train, "empty_cache_each_epoch", False) and device.type == "cuda":
            torch.cuda.empty_cache()

        epoch_sec     = time.time() - t_epoch
        elapsed_total = time.time() - t_train_start
        lr            = optimizer.param_groups[0]["lr"]
        is_best       = val_mpjpe < best_val_mpjpe

        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)

        # ── 归一化 loss 后记录（epoch 0 建立基准，后续相对值）────────────────
        if epoch == start_epoch:
            _loss_scale_train = train_loss if train_loss > 0 else 1.0
            _loss_scale_val   = val_loss   if val_loss   > 0 else 1.0
        train_norm = train_loss / _loss_scale_train
        val_norm   = val_loss   / _loss_scale_val
        curve_log.log(MODEL_NAME, epoch, train_norm, val_norm, lr, avg_ms)

        _print_epoch_summary(epoch, cfg.train.epochs,
                             train_loss, val_loss,
                             lr, epoch_sec, elapsed_total,
                             avg_ms, best_val_loss, is_best)

        det_str = "  Loss detail → " + \
                  "  ".join(f"{k}: {v:.4f}" for k, v in train_det.items()
                             if k != "total")
        print(det_str)

        if is_best:
            best_val_loss = val_loss
            best_val_mpjpe = val_mpjpe
        print(f"  Val MPJPE  : {val_mpjpe:.3f} mm  |  Best MPJPE: {best_val_mpjpe:.3f} mm")

        ckpt = dict(
            epoch           = epoch,
            model           = model.state_dict(),
            optimizer       = optimizer.state_dict(),
            scheduler       = scheduler.state_dict(),
            best_val_loss   = best_val_loss,
            best_val_mpjpe  = best_val_mpjpe,
            train_loss_hist = train_loss_hist,
            val_loss_hist   = val_loss_hist,
        )

        # ── 保存最新 checkpoint（覆盖）──────────────────────────────────────
        save_freq = getattr(cfg.train, "save_freq", 0)
        if save_freq > 0 and (epoch + 1) % save_freq == 0:
            torch.save(ckpt, latest_path)
            print(f"  💾 Latest checkpoint → {latest_path}  (epoch {epoch+1})")

        if is_best:
            torch.save(ckpt, best_path)
            print(f"  ★  Best model saved → {best_path}")

    total_time = time.time() - t_train_start
    _sep("═")
    print(f"  训练完成！  {_now()}")
    print(f"  数据集    : {cfg.data.dataset}")
    print(f"  总耗时    : {_fmt(total_time)}")
    print(f"  Best Val  : {best_val_loss:.6f}")
    print(f"  Best MPJPE: {best_val_mpjpe:.3f} mm")
    print(f"  Best model: {best_path}")
    _sep("═")

    return best_path


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
MD-HCRNet 训练脚本

单数据集训练:
  python train.py --config config/freihand.yaml

断点续训:
  python train.py --config config/freihand.yaml --resume outputs/freihand/latest.pth

串行训练（FreiHAND → RHD 自动衔接）:
  python train.py --seq config/freihand.yaml config/rhd.yaml
        """)
    parser.add_argument("--config",  default="config/freihand.yaml",
                        help="单数据集训练时的配置文件")
    parser.add_argument("--resume",   default="",
                        help="断点续训 checkpoint 路径（完整恢复 optimizer/scheduler）")
    parser.add_argument("--finetune", default="",
                        help="微调起点 checkpoint 路径（只加载模型权重，optimizer 重置）")
    parser.add_argument("--eval",    action="store_true",
                        help="训练结束后自动评估 best.pth")
    parser.add_argument("--seq",     nargs="+", metavar="CONFIG",
                        help="串行训练：依次传入多个 yaml，后一个自动用前一个的 best.pth 初始化")
    args   = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 串行训练模式 ──────────────────────────────────────────────────────────
    if args.seq:
        t_total = time.time()
        _sep("═")
        print(f"  🔗 串行训练模式  |  共 {len(args.seq)} 个数据集  |  {_now()}")
        for i, cfg_path in enumerate(args.seq):
            print(f"     [{i+1}/{len(args.seq)}] {cfg_path}")
        _sep("═")

        prev_best    = ""
        all_results  = {}   # 汇总每个数据集的评估指标

        for i, cfg_path in enumerate(args.seq):
            dataset_name = load_config(cfg_path).data.dataset

            # ── 训练 ──────────────────────────────────────────────────────
            _sep("═")
            print(f"\n  ▶ 阶段 {i+1}/{len(args.seq)} — 训练  [{dataset_name}]  {cfg_path}")
            _sep("═")
            prev_best = run_training(
                cfg_path, device,
                finetune=prev_best if i > 0 else "")

            # ── 评估 ──────────────────────────────────────────────────────
            _sep("─")
            print(f"  ▶ 阶段 {i+1}/{len(args.seq)} — 评估  [{dataset_name}]")
            _sep("─")
            results = run_eval(cfg_path, prev_best, device)
            all_results[dataset_name] = results

        # ── 汇总打印 ──────────────────────────────────────────────────────
        _sep("═")
        print(f"  🏁 全部完成！总耗时：{_fmt(time.time() - t_total)}  |  {_now()}")
        print(f"\n  {'数据集':<12s}  {'MPJPE':>10s}  {'PA-MPJPE':>12s}")
        print(f"  {'─'*12}  {'─'*10}  {'─'*12}")
        for ds, res in all_results.items():
            mpjpe    = res.get("MPJPE (mm)",    float("nan"))
            pa_mpjpe = res.get("PA-MPJPE (mm)", float("nan"))
            print(f"  {ds:<12s}  {mpjpe:>10.2f}  {pa_mpjpe:>12.2f}")
        _sep("═")

    # ── 单数据集训练模式 ──────────────────────────────────────────────────────
    else:
        best_path = run_training(args.config, device,
                                  resume=args.resume, finetune=args.finetune)
        if args.eval:
            _sep("─")
            print(f"  ▶ 训练完成，开始评估...")
            _sep("─")
            run_eval(args.config, best_path, device)


if __name__ == "__main__":
    main()
