"""Official HPT-style action heads (Diffusion Policy + TransformerDecoder path B).

Diffusion: mean-pooled trunk embedding conditions a DDIM denoiser over ``[H, A]``.
TransformerDecoder: full obs trunk token sequence as cross-attn context; ``H`` learnable
queries regress the action chunk directly (no flow matching).
"""

from __future__ import annotations

import math
from typing import Union

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDIMScheduler

from openpi.models_pytorch.hpt.modules import CrossAttention


# ---------------------------------------------------------------------------
# Conditional UNet 1D (adapted from liruiw/HPT, MIT License)
# ---------------------------------------------------------------------------


class _SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class _Conv1dBlock(nn.Module):
    def __init__(self, inp_channels: int, out_channels: int, kernel_size: int, n_groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _Downsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _Upsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _ConditionalResidualBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        *,
        kernel_size: int = 3,
        n_groups: int = 8,
        cond_predict_scale: bool = True,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                _Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
                _Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
            ]
        )
        cond_channels = out_channels * 2 if cond_predict_scale else out_channels
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            nn.Unflatten(-1, (-1, 1)),
        )
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        if self.cond_predict_scale:
            embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
            scale, bias = embed[:, 0, ...], embed[:, 1, ...]
            out = scale * out + bias
        else:
            out = out + embed
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class ConditionalUnet1D(nn.Module):
    """1D UNet for action trajectories, FiLM-conditioned on diffusion step + trunk embed."""

    def __init__(
        self,
        input_dim: int,
        *,
        global_cond_dim: int | None = None,
        diffusion_step_embed_dim: int = 32,
        down_dims: tuple[int, ...] = (64, 128),
        kernel_size: int = 3,
        n_groups: int = 8,
        cond_predict_scale: bool = True,
    ):
        super().__init__()
        all_dims = [input_dim, *down_dims]
        start_dim = down_dims[0]
        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            _SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + (global_cond_dim or 0)

        in_out = list(zip(all_dims[:-1], all_dims[1:], strict=True))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                _ConditionalResidualBlock1D(
                    mid_dim, mid_dim, cond_dim, kernel_size=kernel_size, n_groups=n_groups, cond_predict_scale=cond_predict_scale
                )
                for _ in range(2)
            ]
        )
        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(
                nn.ModuleList(
                    [
                        _ConditionalResidualBlock1D(
                            dim_in, dim_out, cond_dim, kernel_size=kernel_size, n_groups=n_groups, cond_predict_scale=cond_predict_scale
                        ),
                        _ConditionalResidualBlock1D(
                            dim_out, dim_out, cond_dim, kernel_size=kernel_size, n_groups=n_groups, cond_predict_scale=cond_predict_scale
                        ),
                        _Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )
        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(
                nn.ModuleList(
                    [
                        _ConditionalResidualBlock1D(
                            dim_out * 2,
                            dim_in,
                            cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            cond_predict_scale=cond_predict_scale,
                        ),
                        _ConditionalResidualBlock1D(
                            dim_in, dim_in, cond_dim, kernel_size=kernel_size, n_groups=n_groups, cond_predict_scale=cond_predict_scale
                        ),
                        _Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )
        final_conv = nn.Sequential(
            _Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )
        self.diffusion_step_encoder = diffusion_step_encoder
        self.down_modules = down_modules
        self.up_modules = up_modules
        self.final_conv = final_conv

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        *,
        global_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        sample = einops.rearrange(sample, "b h t -> b t h")
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif timesteps.ndim == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])
        global_feature = self.diffusion_step_encoder(timesteps)
        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], dim=-1)

        x = sample
        h: list[torch.Tensor] = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)
        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)
        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)
        x = self.final_conv(x)
        return einops.rearrange(x, "b t h -> b h t")


class DiffusionActionHead(nn.Module):
    """Official HPT diffusion head: mean-pooled trunk cond + DDIM over action chunk."""

    def __init__(
        self,
        *,
        action_dim: int,
        action_horizon: int,
        cond_dim: int,
        train_timesteps: int = 100,
        num_inference_steps: int | None = None,
        down_dims: tuple[int, ...] = (64, 128),
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.num_inference_steps = num_inference_steps or train_timesteps
        self.unet = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=cond_dim,
            down_dims=down_dims,
        )
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=train_timesteps,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type="epsilon",
        )

    def _mask_trajectory(self, traj: torch.Tensor, action_mask: torch.Tensor | None) -> torch.Tensor:
        if action_mask is None:
            return traj
        if action_mask.ndim == 2:
            mask = action_mask[:, None, :].expand_as(traj)
        else:
            mask = action_mask.float()
        return traj * mask

    @torch.no_grad()
    def sample(self, global_cond: torch.Tensor, *, num_steps: int | None = None) -> torch.Tensor:
        b = global_cond.shape[0]
        device = global_cond.device
        trajectory = torch.randn(
            b,
            self.action_horizon,
            self.action_dim,
            device=device,
            dtype=global_cond.dtype,
        )
        steps = num_steps or self.num_inference_steps
        self.noise_scheduler.set_timesteps(steps, device=device)
        for t in self.noise_scheduler.timesteps:
            pred = self.unet(trajectory, t, global_cond=global_cond)
            trajectory = self.noise_scheduler.step(pred, t, trajectory).prev_sample
        return trajectory

    def compute_loss(
        self,
        global_cond: torch.Tensor,
        actions: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        trajectory = actions
        if action_mask is not None:
            trajectory = self._mask_trajectory(trajectory, action_mask)
        noise = torch.randn_like(trajectory)
        bsz = trajectory.shape[0]
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=trajectory.device,
        ).long()
        noisy = self.noise_scheduler.add_noise(trajectory, noise, timesteps)
        pred = self.unet(noisy, timesteps, global_cond=global_cond)
        sq = (pred - noise) ** 2
        if action_mask is not None:
            if action_mask.ndim == 2:
                mask_h = action_mask[:, None, :].expand_as(sq)
            else:
                mask_h = action_mask.float()
            sq = sq * mask_h
            denom = mask_h.sum(dim=(1, 2)).clamp_min(1.0)
            return sq.sum(dim=(1, 2)) / denom
        return sq.mean(dim=(1, 2))


class TransformerDecoderActionHead(nn.Module):
    """Official HPT TransformerDecoder path B: cross-attn queries over full trunk context."""

    def __init__(
        self,
        *,
        action_dim: int,
        action_horizon: int,
        cond_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        huber_delta: float = 0.1,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.huber_delta = huber_delta
        self.query_tokens = nn.Parameter(torch.randn(1, action_horizon, hidden_dim) * 0.02)
        self.context_proj = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        dim_head = hidden_dim // num_heads
        self.cross_attn = CrossAttention(hidden_dim, heads=num_heads, dim_head=dim_head, dropout=dropout)
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.action_out = nn.Linear(hidden_dim, action_dim)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """context: ``[B, L, cond_dim]`` trunk obs tokens → ``[B, H, A]`` actions."""
        if context.ndim != 3:
            raise ValueError(f"context must be [B,L,D], got {tuple(context.shape)}")
        ctx = self.context_proj(context)
        queries = self.query_tokens.expand(context.shape[0], -1, -1)
        hidden = self.cross_attn(queries, ctx)
        return self.action_out(self.out_norm(hidden))

    def compute_loss(
        self,
        context: torch.Tensor,
        actions: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pred = self.forward(context)
        if action_mask is not None:
            if action_mask.ndim == 2:
                mask_h = action_mask[:, None, :].expand_as(pred)
            else:
                mask_h = action_mask.float()
            err = F.smooth_l1_loss(pred, actions, reduction="none", beta=self.huber_delta) * mask_h
            denom = mask_h.sum(dim=(1, 2)).clamp_min(1.0)
            return err.sum(dim=(1, 2)) / denom
        return F.smooth_l1_loss(pred, actions, reduction="none", beta=self.huber_delta).mean(dim=(1, 2))
