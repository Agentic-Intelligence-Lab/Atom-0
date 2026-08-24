"""Smoke tests for the DINO flow-matching WorldDiTHead."""

from __future__ import annotations

import torch

from openpi.models.hpt_config import HPTConfig
from openpi.models_pytorch.hpt.model import HPTModel
from openpi.models_pytorch.hpt.modules import WorldDiTHead


def test_world_dit_head_direct_shapes():
    b, n, dino = 2, 16, 768
    head = WorldDiTHead(
        dino_dim=dino,
        cond_dim=256,
        num_patches=n,
        dit_dim=32,
        dit_blocks=1,
        dit_heads=4,
        wide_dim=64,
        wide_blocks=1,
        wide_heads=4,
    )
    x_t = torch.randn(b, n, dino)
    time = torch.rand(b)
    cond = torch.randn(b, 256)
    out = head(x_t, time, cond)
    assert out.shape == (b, n, dino)
    loss = ((out - torch.randn_like(out)) ** 2).mean()
    loss.backward()
    assert head.input_proj.weight.grad is not None
    assert head.dit_blocks[0].adaLN_modulation[-1].weight.grad is not None
    assert head.wide_blocks[0].adaLN_modulation[-1].weight.grad is not None


def test_sample_actions_skips_world_head():
    cfg = HPTConfig(
        load_encoders=False,
        observation_horizon=1,
        random_horizon_masking=False,
        action_horizon=8,
        num_blocks=2,
        action_head_type="transformer_decoder",
        action_head_dim=32,
        action_head_heads=4,
        head_mode="action_world",
        world_dit_dim=32,
        world_dit_blocks=1,
        world_dit_heads=4,
        world_wide_dim=64,
        world_wide_blocks=1,
        world_wide_heads=4,
        world_num_patches=256,
    )
    model = HPTModel(cfg, device="cpu")
    model.eval()
    sample = {
        "base_0_rgb": torch.zeros(1, 1, 32, 32, 3),
        "state": torch.zeros(1, cfg.proprio_dim),
        "prompts": [""],
        "is_ego": torch.zeros(1, dtype=torch.bool),
    }
    out = model.sample_actions(sample)
    assert out.shape == (1, cfg.action_horizon, cfg.action_dim)
    assert torch.isfinite(out).all()
    assert model.world_head is not None
