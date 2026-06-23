"""
Metric depth estimator wrapper.

Defaults to Depth Anything V2 (metric, indoor) via HuggingFace transformers.
Falls back to a lightweight U-Net if the model cannot be loaded.

Key design decisions:
  - Uses AutoImageProcessor + AutoModelForDepthEstimation directly (NOT pipeline)
    so the whole batch is processed in one GPU forward pass instead of N serial calls.
  - Input rgb_batch is ImageNet-normalised; we denormalise back to [0,1] before
    feeding to the depth model which expects raw RGB.
  - Entire depth forward is wrapped in autocast(enabled=False) so bilinear
    interpolation and the ViT ops inside depth model stay in float32.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# ImageNet stats used by ToTensor in transforms.py
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ──────────────────────────────────────────────────────────────────────────────
# Lightweight fallback depth network
# ──────────────────────────────────────────────────────────────────────────────

class _ConvBnRelu(nn.Sequential):
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__(
            nn.Conv2d(in_c, out_c, k, s, p, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )


class _SimpleUNetDepth(nn.Module):
    """Tiny U-Net that predicts a single-channel metric-ish depth map."""

    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(_ConvBnRelu(3, 32), _ConvBnRelu(32, 32))
        self.enc2 = nn.Sequential(nn.MaxPool2d(2), _ConvBnRelu(32, 64), _ConvBnRelu(64, 64))
        self.enc3 = nn.Sequential(nn.MaxPool2d(2), _ConvBnRelu(64, 128), _ConvBnRelu(128, 128))
        self.bot  = nn.Sequential(nn.MaxPool2d(2), _ConvBnRelu(128, 256), _ConvBnRelu(256, 128))

        self.up3  = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec3 = _ConvBnRelu(128 + 128, 64)
        self.up2  = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = _ConvBnRelu(64 + 64, 32)
        self.up1  = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = _ConvBnRelu(32 + 32, 32)

        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b  = self.bot(e3)

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        depth = F.softplus(self.head(d1))
        return depth


# ──────────────────────────────────────────────────────────────────────────────
# Depth Anything V2 — batched, float32, denormalised input
# ──────────────────────────────────────────────────────────────────────────────

class _DepthAnythingV2(nn.Module):
    """
    Wraps Depth-Anything-V2 using AutoImageProcessor + AutoModelForDepthEstimation.

    Processes the WHOLE batch in ONE forward pass (not N serial pipeline calls).
    Images are denormalised from ImageNet stats to raw [0, 1] before processing.
    """

    HF_MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"

    def __init__(self):
        super().__init__()
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        self.processor = AutoImageProcessor.from_pretrained(self.HF_MODEL_ID)
        self.depth_model = AutoModelForDepthEstimation.from_pretrained(self.HF_MODEL_ID)
        self.depth_model.eval()
        for p in self.depth_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, rgb_batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb_batch: (B, 3, H, W) ImageNet-normalised float tensor
        Returns:
            depth:     (B, 1, H, W) metric depth in metres, float32
        """
        B, _, H, W = rgb_batch.shape
        device = rgb_batch.device

        # ── 1. Denormalise: ImageNet-norm → raw [0, 1] ────────────────────────
        mean = _MEAN.to(device=device, dtype=torch.float32)
        std  = _STD .to(device=device, dtype=torch.float32)
        rgb_raw = (rgb_batch.float() * std + mean).clamp(0.0, 1.0)  # (B,3,H,W)

        # ── 2. Convert to list of PIL images for processor ────────────────────
        from PIL import Image as PILImage
        import numpy as np
        pil_list = []
        rgb_np = (rgb_raw.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        for i in range(B):
            pil_list.append(PILImage.fromarray(rgb_np[i]))

        # ── 3. Batch-process through depth model (single GPU forward pass) ─────
        with torch.cuda.amp.autocast(enabled=False):
            inputs = self.processor(images=pil_list, return_tensors="pt")
            inputs = {k: v.to(device=device, dtype=torch.float32)
                      for k, v in inputs.items()}
            outputs = self.depth_model(**inputs)
            # predicted_depth: (B, H_out, W_out)
            depth_out = outputs.predicted_depth.float()

        # ── 4. Resize to input resolution ─────────────────────────────────────
        depth = F.interpolate(
            depth_out.unsqueeze(1),          # (B, 1, H_out, W_out)
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )                                    # (B, 1, H, W)

        return depth.to(device=device, dtype=torch.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

class MetricDepthEstimator(nn.Module):
    """
    Wraps either Depth-Anything-V2 (batched, frozen) or a lightweight U-Net.
    """

    def __init__(self, estimator_type: str = "depth_anything_v2"):
        super().__init__()
        if estimator_type == "depth_anything_v2":
            try:
                self.net = _DepthAnythingV2()
                print("[DepthEstimator] Using Depth-Anything-V2 (batched, float32).")
            except Exception as e:
                print(f"[DepthEstimator] Could not load Depth-Anything-V2 ({e}).")
                print("[DepthEstimator] Falling back to simple U-Net.")
                self.net = _SimpleUNetDepth()
        else:
            self.net = _SimpleUNetDepth()
            print("[DepthEstimator] Using simple U-Net depth estimator.")

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.net(rgb)
