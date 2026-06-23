"""
Dual-branch ViT encoder for Stage 1.

Each branch (RGB and Depth) shares the same transformer architecture but has
independent patch-embedding weights and independent special tokens (β, θ, t).
RGB branch is initialised from a pretrained ViT-B/16; the depth branch copies
the same weights then replaces the input projection for 1-channel input.
"""
import math
import torch
import torch.nn as nn
from einops import rearrange


# ──────────────────────────────────────────────────────────────────────────────
# Building blocks
# ──────────────────────────────────────────────────────────────────────────────

class PatchEmbed(nn.Module):
    def __init__(self, img_h, img_w, patch_size, in_chans, embed_dim):
        super().__init__()
        self.Ph = img_h // patch_size
        self.Pw = img_w // patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, patch_size, patch_size)

    def forward(self, x):
        x = self.proj(x)                         # B,C,Ph,Pw
        return rearrange(x, "b c h w -> b (h w) c")


class MLP(nn.Sequential):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0):
        hidden = int(dim * mlp_ratio)
        super().__init__(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
            nn.Dropout(drop),
        )


class Attention(nn.Module):
    def __init__(self, dim, num_heads, drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv  = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(drop)
        self.proj_drop = nn.Dropout(drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = Attention(dim, num_heads, drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = MLP(dim, mlp_ratio, drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ──────────────────────────────────────────────────────────────────────────────
# Single-branch ViT (shared block weights between RGB and Depth if desired)
# ──────────────────────────────────────────────────────────────────────────────

class BranchViT(nn.Module):
    """
    One branch of the dual-branch ViT.

    Sequence layout fed to the transformer:
        [beta_token | theta_token | trans_token | patch_1 ... patch_N]
    """

    def __init__(self, img_h: int, img_w: int, patch_size: int,
                 in_chans: int, embed_dim: int, depth: int, num_heads: int,
                 mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        self.patch_embed = PatchEmbed(img_h, img_w, patch_size, in_chans, embed_dim)
        num_patches = (img_h // patch_size) * (img_w // patch_size)

        # Three special tokens
        self.beta_token  = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.theta_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.trans_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional embeddings (3 special + N patch positions)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 3, embed_dim))

        self.blocks = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, mlp_ratio, drop)
              for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.beta_token, std=0.02)
        nn.init.trunc_normal_(self.theta_token, std=0.02)
        nn.init.trunc_normal_(self.trans_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def load_pretrained_vit(self, timm_model_name: str = "vit_base_patch16_224"):
        """Load RGB weights from a timm pretrained ViT (best-effort)."""
        try:
            import timm
            src = timm.create_model(timm_model_name, pretrained=True)
            src_sd = src.state_dict()
            tgt_sd = self.state_dict()

            # Map patch_embed.proj
            if "patch_embed.proj.weight" in src_sd:
                pw = src_sd["patch_embed.proj.weight"]  # (C_out, C_in, P, P)
                if pw.shape[1] == self.patch_embed.proj.weight.shape[1]:
                    tgt_sd["patch_embed.proj.weight"] = pw
                    tgt_sd["patch_embed.proj.bias"]   = src_sd["patch_embed.proj.bias"]
                else:
                    # depth branch: 1-channel — sum over RGB channels
                    tgt_sd["patch_embed.proj.weight"] = pw.mean(dim=1, keepdim=True)

            # Map transformer blocks
            for i in range(len(self.blocks)):
                prefix_src = f"blocks.{i}."
                prefix_tgt = f"blocks.{i}."
                for k, v in src_sd.items():
                    if k.startswith(prefix_src):
                        new_k = prefix_tgt + k[len(prefix_src):]
                        if new_k in tgt_sd and tgt_sd[new_k].shape == v.shape:
                            tgt_sd[new_k] = v

            # pos_embed: interpolate if sizes differ
            if "pos_embed" in src_sd:
                src_pe = src_sd["pos_embed"]   # (1, 197, 768) for ViT-B/16 on 224²
                tgt_pe = tgt_sd["pos_embed"]   # (1, N+3, 768) for our size
                # skip CLS from src, interpolate patch positions
                src_patch_pe = src_pe[:, 1:]   # (1, 196, 768)
                tgt_num = tgt_pe.shape[1] - 3
                src_num = src_patch_pe.shape[1]
                if src_num != tgt_num:
                    src_hw = int(src_num ** 0.5)
                    src_pe_2d = src_patch_pe.reshape(1, src_hw, src_hw, -1).permute(0, 3, 1, 2)
                    # target grid
                    Ph = self.patch_embed.Ph
                    Pw = self.patch_embed.Pw
                    tgt_pe_2d = torch.nn.functional.interpolate(
                        src_pe_2d, size=(Ph, Pw), mode="bilinear", align_corners=False)
                    tgt_patch_pe = tgt_pe_2d.permute(0, 2, 3, 1).reshape(1, tgt_num, -1)
                else:
                    tgt_patch_pe = src_patch_pe
                # keep special-token positions as their random init
                tgt_sd["pos_embed"][:, 3:] = tgt_patch_pe

            self.load_state_dict(tgt_sd, strict=False)
            print(f"[BranchViT] Loaded pretrained weights from '{timm_model_name}'.")
        except Exception as e:
            print(f"[BranchViT] Could not load pretrained weights: {e}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) image or depth map
        Returns:
            tokens: (B, N+3, embed_dim)  — first 3 are special tokens
        """
        B = x.shape[0]
        patches = self.patch_embed(x)   # B, N, C

        beta_tok  = self.beta_token.expand(B, -1, -1)
        theta_tok = self.theta_token.expand(B, -1, -1)
        trans_tok = self.trans_token.expand(B, -1, -1)

        seq = torch.cat([beta_tok, theta_tok, trans_tok, patches], dim=1)  # B, N+3, C
        seq = seq + self.pos_embed
        seq = self.blocks(seq)
        return self.norm(seq)


# ──────────────────────────────────────────────────────────────────────────────
# Dual-branch wrapper
# ──────────────────────────────────────────────────────────────────────────────

class DualBranchViT(nn.Module):
    def __init__(self, img_h, img_w, patch_size, embed_dim, depth, num_heads,
                 mlp_ratio=4.0, drop=0.0, pretrained: bool = True):
        super().__init__()
        common = dict(img_h=img_h, img_w=img_w, patch_size=patch_size,
                      embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                      mlp_ratio=mlp_ratio, drop=drop)
        self.rgb_branch   = BranchViT(in_chans=3, **common)
        self.depth_branch = BranchViT(in_chans=1, **common)

        if pretrained:
            self.rgb_branch.load_pretrained_vit("vit_base_patch16_224.augreg_in21k_ft_in1k")
            self.depth_branch.load_pretrained_vit("vit_base_patch16_224.augreg_in21k_ft_in1k")

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor):
        """
        Returns:
            rgb_tokens:   (B, N+3, C)
            depth_tokens: (B, N+3, C)
        """
        return self.rgb_branch(rgb), self.depth_branch(depth)
