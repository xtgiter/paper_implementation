"""Vision Transformer (ViT) — An Image is Worth 16x16 Words.

Reference: Dosovitskiy et al., ICLR 2021.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    """Split image into patches and linearly project them.

    Uses a strided Conv2d, which is equivalent to patch splitting + linear projection.
    """

    def __init__(self, img_size: int, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, D, H/P, W/P) -> (B, D, N) -> (B, N, D)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Attention(nn.Module):
    """Multi-head self-attention."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                        # each: (B, heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale           # (B, heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, D)         # (B, N, D)
        x = self.proj(x)
        return x


class Block(nn.Module):
    """Transformer encoder block: pre-norm MSA + MLP with residual connections."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    """Vision Transformer.

    Args:
        img_size:  input image spatial size (assumed square).
        patch_size:  spatial size of each patch (assumed square).
        in_chans:    number of input channels (3 for RGB).
        num_classes: number of output classes.
        embed_dim:   hidden dimension D.
        depth:       number of Transformer blocks.
        num_heads:   number of attention heads per block.
        mlp_ratio:   hidden-dim multiplier for the MLP.
        dropout:     dropout rate applied everywhere.
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_chans: int = 3,
        num_classes: int = 10,
        embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        # Truncated normal(0.02) following the paper and timm.
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)

        x = self.patch_embed(x)                               # (B, N, D)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)  # (B, 1, D)
        x = torch.cat((cls_tokens, x), dim=1)                 # (B, N+1, D)
        x = x + self.pos_embed
        x = self.drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        x = self.head(x[:, 0])                                  # class token output
        return x


# ---------------------------------------------------------------------------
# Predefined model configurations
# ---------------------------------------------------------------------------

def vit_tiny(patch_size: int = 4, num_classes: int = 10, dropout: float = 0.0) -> ViT:
    """ViT-Tiny: a lightweight variant suitable for CIFAR-scale experiments."""
    return ViT(
        img_size=32, patch_size=patch_size, in_chans=3, num_classes=num_classes,
        embed_dim=192, depth=12, num_heads=3, mlp_ratio=4.0, dropout=dropout,
    )


def vit_small(patch_size: int = 4, num_classes: int = 10, dropout: float = 0.0) -> ViT:
    """ViT-Small: roughly between Tiny and Base."""
    return ViT(
        img_size=32, patch_size=patch_size, in_chans=3, num_classes=num_classes,
        embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.0, dropout=dropout,
    )


def vit_base(patch_size: int = 4, num_classes: int = 10, dropout: float = 0.0) -> ViT:
    """ViT-Base: the standard configuration from the paper."""
    return ViT(
        img_size=32, patch_size=patch_size, in_chans=3, num_classes=num_classes,
        embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, dropout=dropout,
    )


if __name__ == "__main__":
    model = vit_tiny()
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print(f"Input:  {tuple(x.shape)}")
    print(f"Output: {tuple(y.shape)}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params:,}")
