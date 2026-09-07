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

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        *,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, n, _ = x.shape
        h = self.heads
        q = self.to_q(x).reshape(b, n, h, -1).transpose(1, 2)
        kv = self.to_kv(context).reshape(b, context.shape[1], h, 2, -1)
        k, v = kv.unbind(dim=-2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if context_mask is not None:
            # context_mask: [B, N_ctx], True = valid token.
            pad = ~context_mask.to(device=attn.device, dtype=torch.bool)
            attn = attn.masked_fill(pad.unsqueeze(1).unsqueeze(2), float("-inf"))
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


def get_sinusoid_encoding_table(
    n_position: int,
    dim: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Official HPT sinusoid table: [1, n_position, dim] for sequence positions 0..n-1."""
    position = torch.arange(n_position, device=device, dtype=torch.float32).unsqueeze(1)
    idx = torch.arange(dim, device=device, dtype=torch.float32)
    div = torch.pow(10000.0, (2 * (idx / 2).floor()) / dim)
    table = position * (1.0 / div)
    table = table.clone()
    table[:, 0::2] = table[:, 0::2].sin()
    table[:, 1::2] = table[:, 1::2].cos()
    if dtype is not None:
        table = table.to(dtype=dtype)
    return table.unsqueeze(0)


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
    """Original HPT action head: mean-pool trunk cond + per-timestep MLP (no DiT).

    Matches the earliest Atom-0 HPT implementation used by ckpt ``99999``.
    """

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
        self.time_dim = time_dim
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

    def forward(self, x_t: torch.Tensor, time: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_t: [B, H, A] noisy actions
            time: [B]
            cond: [B, N, D] or [B, D] trunk features
        Returns:
            v_t: [B, H, A] predicted flow velocity
        """
        if cond.ndim == 3:
            cond = cond.mean(dim=1)
        t_emb = self.time_mlp(sinusoidal_time_embedding(time, self.time_dim))
        c = self.cond_proj(cond)
        h = self.action_in(x_t) + t_emb[:, None, :] + c[:, None, :]
        return self.net(h)


class FlowMatchingActionDiTHead(nn.Module):
    """Action-DiT: flow-matching head over the action chunk.

    Tokens are the H noisy actions. Each block does self-attn on the horizon, then
    cross-attn to trunk **action-query** features (not the full obs sequence).
    """

    def __init__(
        self,
        action_dim: int,
        action_horizon: int,
        cond_dim: int,
        *,
        hidden_dim: int = 256,
        num_blocks: int = 4,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        time_dim: int = 128,
        dropout: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim

        self.action_in = nn.Linear(action_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, action_horizon, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

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

        dpr = torch.linspace(0, drop_path, num_blocks).tolist()
        self.blocks = nn.ModuleList(
            [
                _ActionDiTBlock(
                    hidden_dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    drop_path=dpr[i],
                )
                for i in range(num_blocks)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.final_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim))
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)
        self.action_out = nn.Linear(hidden_dim, action_dim)
        nn.init.zeros_(self.action_out.weight)
        nn.init.zeros_(self.action_out.bias)

    def forward(self, x_t: torch.Tensor, time: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if x_t.ndim != 3:
            raise ValueError(f"x_t must be [B,H,A], got {tuple(x_t.shape)}")
        b, h, _ = x_t.shape
        if h != self.action_horizon:
            raise ValueError(f"action horizon mismatch: got {h}, expected {self.action_horizon}")
        if cond.ndim == 2:
            cond = cond.unsqueeze(1)
        elif cond.ndim != 3:
            raise ValueError(f"cond must be [B,N,D] or [B,D], got {tuple(cond.shape)}")

        t_emb = self.time_mlp(sinusoidal_time_embedding(time, self.time_dim))
        context = self.cond_proj(cond)
        tokens = self.action_in(x_t) + self.pos_embed[:, :h]
        for blk in self.blocks:
            tokens = blk(tokens, context, t_emb)
        shift, scale = self.final_adaLN(t_emb).chunk(2, dim=-1)
        tokens = self.final_norm(tokens) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return self.action_out(tokens)


class _ActionDiTBlock(nn.Module):
    """Self-attn over H, then cross-attn to trunk tokens; time via adaLN-zero."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = CrossAttention(
            dim, heads=num_heads, dim_head=max(dim // num_heads, 1), dropout=dropout
        )
        self.norm3 = nn.LayerNorm(dim)
        hidden = dim * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        # shift/scale/gate for self-attn, cross-attn, mlp
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 9 * dim))
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    @staticmethod
    def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(self, x: torch.Tensor, context: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        (
            shift_sa,
            scale_sa,
            gate_sa,
            shift_ca,
            scale_ca,
            gate_ca,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(time_emb).chunk(9, dim=-1)

        h = self._modulate(self.norm1(x), shift_sa, scale_sa)
        h, _ = self.self_attn(h, h, h, need_weights=False)
        x = x + self.drop_path(gate_sa.unsqueeze(1) * h)

        h = self._modulate(self.norm2(x), shift_ca, scale_ca)
        x = x + self.drop_path(gate_ca.unsqueeze(1) * self.cross_attn(h, context))

        h = self._modulate(self.norm3(x), shift_mlp, scale_mlp)
        x = x + self.drop_path(gate_mlp.unsqueeze(1) * self.mlp(h))
        return x


class CrossTransformerBlock(nn.Module):
    """EgoWAM-style block: pre-norm self-attn → cross-attn (Q=action, KV=cond) → FFN."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = CrossAttention(
            dim, heads=num_heads, dim_head=max(dim // num_heads, 1), dropout=dropout
        )
        self.norm3 = nn.LayerNorm(dim)
        hidden = dim * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        *,
        action_key_padding_mask: torch.Tensor | None = None,
        condition_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.self_attn(
            h,
            h,
            h,
            key_padding_mask=action_key_padding_mask,
            need_weights=False,
        )
        x = x + self.drop_path(h)

        h = self.norm2(x)
        x = x + self.drop_path(self.cross_attn(h, cond, context_mask=condition_mask))

        h = self.norm3(x)
        x = x + self.drop_path(self.mlp(h))
        return x


class CrossTransformerActionHead(nn.Module):
    """CFM action decoder: noisy action tokens + timestep add → CrossTransformer → velocity.

    Conditioning: cross-attention with Q=action hidden states, K/V=projected trunk tokens.
    Timestep is added per token (not AdaLN). Replaces Action-DiT for architecture ablations.
    """

    def __init__(
        self,
        action_dim: int,
        action_horizon: int,
        cond_dim: int,
        *,
        hidden_dim: int = 128,
        num_layers: int = 6,
        num_heads: int = 4,
        mlp_ratio: int = 4,
        time_dim: int = 128,
        dropout: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim

        self.action_in = nn.Linear(action_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, action_horizon, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

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

        dpr = torch.linspace(0, drop_path, num_layers).tolist()
        self.blocks = nn.ModuleList(
            [
                CrossTransformerBlock(
                    hidden_dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    drop_path=dpr[i],
                )
                for i in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.action_out = nn.Linear(hidden_dim, action_dim)

    def forward(
        self,
        x_t: torch.Tensor,
        time: torch.Tensor,
        cond: torch.Tensor,
        *,
        action_mask: torch.Tensor | None = None,
        condition_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x_t.ndim != 3:
            raise ValueError(f"x_t must be [B,H,A], got {tuple(x_t.shape)}")
        b, h, _ = x_t.shape
        if h != self.action_horizon:
            raise ValueError(f"action horizon mismatch: got {h}, expected {self.action_horizon}")
        if cond.ndim == 2:
            cond = cond.unsqueeze(1)
        elif cond.ndim != 3:
            raise ValueError(f"cond must be [B,N,D] or [B,D], got {tuple(cond.shape)}")

        t_emb = self.time_mlp(sinusoidal_time_embedding(time, self.time_dim))
        context = self.cond_proj(cond)
        tokens = self.action_in(x_t) + t_emb[:, None, :] + self.pos_embed[:, :h]

        action_kpm = None
        if action_mask is not None:
            # Optional per-timestep mask [B,T] or [B,T,A] (any invalid action dim → mask step).
            if action_mask.ndim == 3:
                action_kpm = ~action_mask.any(dim=-1)
            elif action_mask.ndim == 2 and action_mask.shape[1] == h:
                action_kpm = ~action_mask.to(dtype=torch.bool)

        for blk in self.blocks:
            tokens = blk(
                tokens,
                context,
                action_key_padding_mask=action_kpm,
                condition_mask=condition_mask,
            )
        return self.action_out(self.final_norm(tokens))


class AdaLNSelfAttentionBlock(nn.Module):
    """Pre-norm self-attn + FFN, AdaLN-zero from a global condition (no cross-attn)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = dim * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    @staticmethod
    def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift_sa, scale_sa, gate_sa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(cond).chunk(6, dim=-1)
        h = self._modulate(self.norm1(x), shift_sa, scale_sa)
        h, _ = self.self_attn(h, h, h, need_weights=False)
        x = x + self.drop_path(gate_sa.unsqueeze(1) * h)
        h = self._modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + self.drop_path(gate_mlp.unsqueeze(1) * self.mlp(h))
        return x


class WorldDiTHead(nn.Module):
    """Flow-matching / v-prediction world head over future DINO patch tokens.

    Conditioning is a global vector (mean-pooled trunk obs) fused with the flow
    timestep and injected via AdaLN. No cross-attention to the trunk sequence.
    """

    def __init__(
        self,
        *,
        dino_dim: int = 768,
        cond_dim: int = 256,
        num_patches: int = 256,
        dit_dim: int = 384,
        dit_blocks: int = 6,
        dit_heads: int = 6,
        dit_mlp_ratio: int = 4,
        wide_dim: int = 2048,
        wide_blocks: int = 2,
        wide_heads: int = 16,
        wide_mlp_ratio: int = 4,
        time_dim: int = 128,
        dropout: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        if dit_dim % dit_heads != 0:
            raise ValueError(f"dit_dim={dit_dim} must be divisible by dit_heads={dit_heads}")
        if wide_dim % wide_heads != 0:
            raise ValueError(f"wide_dim={wide_dim} must be divisible by wide_heads={wide_heads}")
        self.dino_dim = dino_dim
        self.num_patches = num_patches
        self.dit_dim = dit_dim
        self.wide_dim = wide_dim
        self.time_dim = time_dim

        self.input_proj = nn.Linear(dino_dim, dit_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, dit_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, dit_dim),
            nn.SiLU(),
            nn.Linear(dit_dim, dit_dim),
        )
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, dit_dim),
            nn.SiLU(),
            nn.Linear(dit_dim, dit_dim),
        )

        dpr = torch.linspace(0, drop_path, dit_blocks).tolist()
        self.dit_blocks = nn.ModuleList(
            [
                AdaLNSelfAttentionBlock(
                    dit_dim,
                    dit_heads,
                    mlp_ratio=dit_mlp_ratio,
                    dropout=dropout,
                    drop_path=dpr[i],
                )
                for i in range(dit_blocks)
            ]
        )
        self.dit_to_wide = nn.Linear(dit_dim, wide_dim)
        self.wide_cond_proj = nn.Sequential(nn.SiLU(), nn.Linear(dit_dim, wide_dim))

        wide_dpr = torch.linspace(0, drop_path, wide_blocks).tolist()
        self.wide_blocks = nn.ModuleList(
            [
                AdaLNSelfAttentionBlock(
                    wide_dim,
                    wide_heads,
                    mlp_ratio=wide_mlp_ratio,
                    dropout=dropout,
                    drop_path=wide_dpr[i],
                )
                for i in range(wide_blocks)
            ]
        )
        self.final_norm = nn.LayerNorm(wide_dim)
        self.final_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(wide_dim, 2 * wide_dim))
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)
        self.out_proj = nn.Linear(wide_dim, dino_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def _fused_condition(self, time: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if cond.ndim == 3:
            cond = cond.mean(dim=1)
        t_emb = self.time_mlp(sinusoidal_time_embedding(time, self.time_dim))
        c_emb = self.cond_proj(cond)
        fused = t_emb + c_emb
        return fused, self.wide_cond_proj(fused)

    def forward(self, x_t: torch.Tensor, time: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_t: [B, N, dino_dim] noisy future DINO patches
            time: [B] flow time in (0, 1)
            cond: [B, cond_dim] or [B, L, cond_dim] (mean-pooled if 3D)
        Returns:
            v_hat: [B, N, dino_dim] predicted velocity
        """
        if x_t.ndim != 3:
            raise ValueError(f"x_t must be [B,N,D], got {tuple(x_t.shape)}")
        n = x_t.shape[1]
        if n != self.num_patches:
            raise ValueError(f"num patches mismatch: got {n}, expected {self.num_patches}")
        fused, fused_wide = self._fused_condition(time, cond)
        tokens = self.input_proj(x_t) + self.pos_embed[:, :n]
        for blk in self.dit_blocks:
            tokens = blk(tokens, fused)
        tokens = self.dit_to_wide(tokens)
        for blk in self.wide_blocks:
            tokens = blk(tokens, fused_wide)
        shift, scale = self.final_adaLN(fused_wide).chunk(2, dim=-1)
        tokens = self.final_norm(tokens) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return self.out_proj(tokens)
