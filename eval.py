"""
Evaluation script for MD-HCRNet.

Computes MPJPE, PA-MPJPE, and F-score on the validation / test split.

Usage:
    python eval.py --config config/default.yaml --checkpoint outputs/best.pth
"""
import argparse
import os
import yaml
from types import SimpleNamespace

import numpy as np
import torch
from tqdm import tqdm

from models import MDHCRNet
from data   import build_dataloaders
from utils  import MPJPE, PA_MPJPE, F_score


def _coerce(v):
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
        return _dict_to_ns(yaml.safe_load(f))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    all_pred_joints3d  = []
    all_gt_joints3d    = []
    all_pred_verts     = []
    all_gt_verts       = []

    for batch in tqdm(loader, desc="Eval"):
        rgb   = batch["image"].to(device)
        focal = batch["focal"].to(device)
        cx    = batch["cx"].to(device)
        cy    = batch["cy"].to(device)

        pred = model(rgb, focal, cx, cy)

        pred_j3d = pred["joints3d"].cpu().numpy()    # B,21,3
        gt_j3d   = batch["joints3d"].numpy()

        all_pred_joints3d.append(pred_j3d)
        all_gt_joints3d.append(gt_j3d)

        if "vertices" in pred and "vertices" in batch:
            all_pred_verts.append(pred["vertices"].cpu().numpy())
            all_gt_verts.append(batch["vertices"].numpy())

    pred_j3d = np.concatenate(all_pred_joints3d, axis=0) * 1000.0   # → mm
    gt_j3d   = np.concatenate(all_gt_joints3d,   axis=0) * 1000.0

    # Root-relative (wrist-aligned)
    pred_rel = pred_j3d - pred_j3d[:, :1, :]
    gt_rel   = gt_j3d   - gt_j3d[:,   :1, :]

    mpjpe    = MPJPE(pred_rel,    gt_rel)
    pa_mpjpe = PA_MPJPE(pred_rel, gt_rel)

    results = {"MPJPE (mm)": mpjpe, "PA-MPJPE (mm)": pa_mpjpe}

    if all_pred_verts:
        pred_v = np.concatenate(all_pred_verts, axis=0) * 1000.0
        gt_v   = np.concatenate(all_gt_verts,   axis=0) * 1000.0
        for thr in (5.0, 15.0):
            results[f"F@{thr:.0f}mm"] = F_score(pred_v, gt_v, thr)

    return results


def run_eval(config_path: str, checkpoint_path: str,
             device: torch.device) -> dict:
    """
    可被外部调用的评估入口，返回 metrics dict。
    供 train.py 串行流程调用。
    """
    from datetime import datetime
    cfg   = load_config(config_path)
    model = MDHCRNet(cfg).to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    epoch = ckpt.get("epoch", "?")

    print(f"\n  📐 评估  |  数据集: {cfg.data.dataset}"
          f"  |  checkpoint epoch {epoch}"
          f"  |  {datetime.now().strftime('%H:%M:%S')}")

    _, val_loader = build_dataloaders(cfg)
    results = evaluate(model, val_loader, device)

    print("\n  ── 评估结果 " + "─" * 50)
    for k, v in results.items():
        print(f"     {k:<20s}: {v:.4f}")
    print("  " + "─" * 62)

    # 保存到 tables/ 子目录
    tables_dir = os.path.join(cfg.train.output_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)
    xlsx_path = os.path.join(tables_dir, "eval_results.xlsx")
    try:
        import pandas as pd, openpyxl
        pd.DataFrame([results]).to_excel(xlsx_path, index=False)
        print(f"  📊 评估结果已保存 → {xlsx_path}")
    except ImportError:
        pass

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="config/freihand.yaml")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MDHCRNet(cfg).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    _, val_loader = build_dataloaders(cfg)

    results = evaluate(model, val_loader, device)

    print("\n── Evaluation Results ──────────────────────────────")
    for k, v in results.items():
        print(f"  {k:<20s}: {v:.2f}")
    print("────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
