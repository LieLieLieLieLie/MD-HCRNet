# MD-HCRNet: Metric Depth-Guided Hand-Centric Reasoning Network for Monocular 3D Hand Mesh Reconstruction

> **Paper:** *Metric Depth-Guided Hand-Centric Reasoning Network for Monocular 3D Hand Mesh Reconstruction*  
> Yangzhi Lin, Yongshu Li, Xingguo Chen, Dengze Wang, Zilong Yin  
> Under review at *Pattern Recognition* (Elsevier)

---

## Overview

MD-HCRNet is a three-stage cascade for recovering a metric-scale 3D hand mesh from a single unconstrained RGB image. The central novelty is a **modality-preserving dual-branch encoding** strategy that encodes RGB and estimated metric depth in independent ViT pathways, deferring cross-modal fusion to a set of 21 anatomically grounded joint-specific tokens.

**Stage 1 — Geometry-Aware Encoding**  
A frozen Depth-Anything-V2 estimator produces per-pixel metric depth. A Dual-Branch ViT processes RGB and depth independently, with three task-specific special tokens (τ_β, τ_θ, τ_t) per branch for MANO shape, pose, and translation. The branches are fused at the token level to produce initial MANO parameters and 2D/3D joint estimates.

**Stage 2 — Unified Hand-Centric Reasoning (UHR)**  
Three sequential modules refine the initial estimates:
- **KGTS** (Knowledge-Guided Token Sampling): extracts 21 joint-specific tokens from ViT feature maps via differentiable bilinear sampling anchored at Stage-1 2D projections with learnable deformable offsets.
- **SAE** (Structure-Aware Dual-Scan Encoding): two independent bidirectional GRUs process joint tokens under complementary orderings—linear MANO-index order (within-finger) and kinematic depth-level order (cross-finger co-level).
- **DCA** (Deformable Cross-Attention): bidirectional cross-modal fusion of RGB and depth joint tokens.

**Stage 3 — Hierarchical Joint Refinement**  
MANO decoding with three complementary correction pathways (coarse global MLP, query-attention branch, UHR-token direct decoder) under dedicated intermediate supervision.

---

## Results

### FreiHAND

| Method | MPJPE (mm) ↓ | PA-MPJPE (mm) ↓ |
|--------|-------------|----------------|
| I2L-MeshNet | 25.9 | 10.0 |
| METRO | 21.8 | 8.1 |
| Mesh Graphormer | 20.2 | 7.6 |
| HandOccNet | 19.8 | 7.4 |
| DepthFusion | 18.8 | 8.2 |
| **MD-HCRNet (ours)** | **17.32** | **9.50** |

### HO3D v3

| Method | MPJPE (mm) ↓ | PA-MPJPE (mm) ↓ |
|--------|-------------|----------------|
| HandOccNet | 15.1 | 9.5 |
| DepthFusion | 14.9 | 9.8 |
| **MD-HCRNet (ours)** | **11.78** | **7.72** |

---

## Installation

**Requirements:** Python 3.9+, PyTorch 2.0+, CUDA 11.8+

```bash
git clone https://github.com/LieLieLieLieLie/MD-HCRNet.git
cd MD-HCRNet
pip install -r requirements.txt
```

---

## Data Preparation

### 1. MANO Model

Register and download MANO from [https://mano.is.tue.mpg.de/](https://mano.is.tue.mpg.de/).  
Place the model files as:

```
data/mano/
    MANO_RIGHT.pkl
    MANO_LEFT.pkl
```

### 2. FreiHAND Dataset

Download from [https://lmb.informatik.uni-freiburg.de/projects/freihand/](https://lmb.informatik.uni-freiburg.de/projects/freihand/) and place under `data/freihand/`:

```
data/freihand/
    training/
        rgb/
        mask/
        ...
    evaluation/
    freihand_train_xyz.json
    freihand_train_K.json
    ...
```

### 3. HO3D v3 Dataset

Download from [https://github.com/shreyashampali/ho3d](https://github.com/shreyashampali/ho3d) and place under `data/HO3D/`:

```
data/HO3D/
    train/
    evaluation/
```

### 4. Precompute Depth Maps (Optional but Recommended)

Pre-caching depth maps avoids running Depth-Anything-V2 online during training and significantly accelerates data loading.

```bash
python precompute_depths.py --dataset freihand   # ~15-30 min
python precompute_depths.py --dataset ho3d       # ~20-40 min
```

Cached depth maps are saved to `data/freihand/depth_cache/` and `data/HO3D/depth_cache/` as float16 `.npy` files.

---

## Training

```bash
# FreiHAND
python train.py --config config/freihand_tuned.yaml

# HO3D v3
python train.py --config config/ho3d_tuned.yaml

# Resume from checkpoint
python train.py --config config/freihand_tuned.yaml \
    --resume outputs/freihand_uhr_tuned/models/checkpoint_ep10.pth
```

Training outputs (checkpoints, loss curves, metric tables) are saved to `outputs/freihand_uhr_tuned/` or `outputs/ho3d_uhr_tuned/`.

Key hyperparameters (see `config/freihand_tuned.yaml`):

| Parameter | FreiHAND | HO3D |
|-----------|----------|------|
| Epochs | 15 | 10 |
| Learning rate | 5e-5 | 4e-5 |
| Batch size | 8 | 8 |
| AMP dtype | bfloat16 | bfloat16 |

---

## Evaluation

```bash
# FreiHAND
python eval.py --config config/freihand_tuned.yaml \
    --checkpoint outputs/freihand_uhr_tuned/models/best.pth

# HO3D v3
python eval.py --config config/ho3d_tuned.yaml \
    --checkpoint outputs/ho3d_uhr_tuned/models/best.pth
```

Metrics reported: MPJPE, PA-MPJPE, and F-score @ 5/15 mm.

---

## Qualitative Visualisation

```bash
python qualitative_viz.py --config config/freihand_tuned.yaml \
    --checkpoint outputs/freihand_uhr_tuned/models/best.pth \
    --num_samples 16
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{lin2026mdhcrnet,
  author  = {Lin, Yangzhi and Li, Yongshu and Chen, Xingguo
             and Wang, Dengze and Yin, Zilong},
  title   = {Metric Depth-Guided Hand-Centric Reasoning Network
             for Monocular {3D} Hand Mesh Reconstruction},
  journal = {Pattern Recognition},
  year    = {2026},
  note    = {Under review}
}
```

---

## Acknowledgements

We thank the creators of [FreiHAND](https://lmb.informatik.uni-freiburg.de/projects/freihand/), [HO3D](https://github.com/shreyashampali/ho3d), [MANO](https://mano.is.tue.mpg.de/), and [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) for making their resources publicly available.
