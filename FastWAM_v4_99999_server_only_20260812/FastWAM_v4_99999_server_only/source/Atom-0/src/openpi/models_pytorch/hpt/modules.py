"""Core HPT modules: cross-attn stems, transformer trunk, dual heads."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def trunc_normal_(tensor: torch.Tensor, std: float = 0.02) -> torch.Tensor:
    return nn.init.trunc_normal_(tensor, std=std)


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


class CrossAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0):
        super().__init__()
        inner = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_kv = nn.Linear(dim, inner * 2, bias=False)
        self.to_out = nn.Linear(inner, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        h = self.heads
        q = self.to_q(x).reshape(b, n, h, -1).transpose(1, 2)
        kv = self.to_kv(context).reshape(b, context.shape[1], h, 2, -1)
        k, v = kv.unbind(dim=-2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.dropout(attn.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, widths: list[int] | None = None):
        super().__init__()
        widths = widths or [128]
        layers: list[nn.Module] = [nn.Linear(input_dim, widths[0]), nn.SiLU()]
        for i in range(len(widths) - 1):
            layers.extend([nn.Linear(widths[i], widths[i + 1]), nn.LayerNorm(widths[i + 1]), nn.SiLU()])
        layers.append(nn.Linear(widths[-1], output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Stem(nn.Module):
    """Project modality features → fixed learnable tokens via cross-attention."""

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        num_tokens: int,
        *,
        widths: list[int] | None = None,
        crossattn_heads: int = 8,
        crossattn_dim_head: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj = MLP(input_dim, embed_dim, widths=widths or [128])
        self.tokens = nn.Parameter(torch.randn(1, num_tokens, embed_dim) * 0.02)
        self.cross_attn = CrossAttention(embed_dim, heads=crossattn_heads, dim_head=crossattn_dim_head, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D_in] or [B, D_in]
        if x.ndim == 2:
            x = x.unsqueeze(1)
        feat = self.proj(x)
        feat = feat.reshape(feat.shape[0], -1, feat.shape[-1])
        tokens = self.tokens.expand(feat.shape[0], -1, -1)
        return self.cross_attn(tokens, feat)


class AttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: int = 4, drop_path: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        hidden = dim * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop_path(h)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerTrunk(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_blocks: int,
        num_heads: int,
        mlp_ratio: int = 4,
        drop_path: float = 0.1,
    ):
        super().__init__()
        dpr = torch.linspace(0, drop_path, num_blocks).tolist()
        self.blocks = nn.ModuleList(
            [AttentionBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio, drop_path=dpr[i]) for i in range(num_blocks)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            tokens = blk(tokens)
        return self.norm(tokens)


def sinusoidal_time_embedding(time: torch.Tensor, dim: int) -> torch.Tensor:
    """time: [B] in [0, 1] → [B, dim]."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=time.device, dtype=torch.float32) / half)
    args = time.float()[:, None] * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class FlowMatchingActionHead(nn.Module):
    """pi05-style flow matching head conditioned on trunk action-query features."""

    def __init__(
        self,
        action_dim: int,
        action_horizon: int,
        cond_dim: int,
        hidden_dim: int = 512,
        time_dim: int = 128,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.action_in = nn.Linear(action_dim, hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.time_dim = time_dim

    def forward(self, x_t: torch.Tensor, time: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_t: [B, H, A] noisy actions
            time: [B]
            cond: [B, N, D] or [B, D] trunk features
        """
        if cond.ndim == 3:
            cond = cond.mean(dim=1)
        t_emb = self.time_mlp(sinusoidal_time_embedding(time, self.time_dim))
        c = self.cond_proj(cond)
        h = self.action_in(x_t) + t_emb[:, None, :] + c[:, None, :]
        return self.net(h)


class WorldHead(nn.Module):
    """Predict future DINO features from trunk future-query tokens."""

    def __init__(self, cond_dim: int, dino_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dino_dim),
        )

    def forward(self, future_features: torch.Tensor) -> torch.Tensor:
        # [B, N, D] → [B, dino_dim]
        if future_features.ndim == 3:
            future_features = future_features.mean(dim=1)
        return self.net(future_features)
