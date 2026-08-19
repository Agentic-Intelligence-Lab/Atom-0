"""Smoke tests for CrossTransformerActionHead (CFM velocity head)."""

from __future__ import annotations

import torch

from openpi.models.hpt_config import HPTConfig
from openpi.models_pytorch.hpt.model import HPTModel, count_action_head_params
from openpi.models_pytorch.hpt.modules import CrossTransformerActionHead, FlowMatchingActionDiTHead


def _tiny_cfg(action_head_type: str) -> HPTConfig:
    return HPTConfig(
        load_encoders=False,
        observation_horizon=1,
        random_horizon_masking=False,
        action_horizon=8,
        action_dim=80,
        num_blocks=2,
        num_action_tokens=16,
        num_future_tokens=4,
        head_mode="action_only",
        action_head_type=action_head_type,  # type: ignore[arg-type]
        action_head_dim=64,
        action_head_blocks=2,
        action_head_heads=4,
        num_inference_steps=2,
    )


def test_cross_transformer_forward_backward_shapes():
    cfg = _tiny_cfg("cross_transformer")
    model = HPTModel(cfg, device="cpu")
    model.train()

    b, t, h, w = 2, 1, 32, 32
    sample = {
        "base_0_rgb": torch.zeros(b, t, h, w, 3),
        "state": torch.zeros(b, cfg.proprio_dim),
        "state_history": torch.zeros(b, 1, cfg.proprio_dim),
        "action": torch.randn(b, cfg.action_horizon, cfg.action_dim),
        "action_mask": torch.ones(b, cfg.action_dim, dtype=torch.bool),
        "prompts": ["a", "b"],
        "is_ego": torch.zeros(b, dtype=torch.bool),
    }
    loss, _ = model.training_loss(sample)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.action_head.action_in.weight.grad is not None
    assert model.action_head.cond_proj[0].weight.grad is not None
    assert model.trunk.blocks[0].attn.in_proj_weight.grad is not None


def test_cross_transformer_head_direct():
    cfg = _tiny_cfg("cross_transformer")
    head = CrossTransformerActionHead(
        action_dim=cfg.action_dim,
        action_horizon=cfg.action_horizon,
        cond_dim=cfg.embed_dim,
        hidden_dim=cfg.action_head_dim,
        num_layers=cfg.action_head_blocks,
        num_heads=cfg.action_head_heads,
    )
    b = 2
    x_t = torch.randn(b, cfg.action_horizon, cfg.action_dim)
    t = torch.rand(b)
    cond = torch.randn(b, cfg.num_action_tokens, cfg.embed_dim)
    cond_mask = torch.ones(b, cfg.num_action_tokens, dtype=torch.bool)
    out = head(x_t, t, cond, condition_mask=cond_mask)
    assert out.shape == x_t.shape


def test_dit_vs_cross_transformer_param_count():
    cfg_dit = _tiny_cfg("dit")
    cfg_xt = _tiny_cfg("cross_transformer")
    dit = FlowMatchingActionDiTHead(
        action_dim=cfg_dit.action_dim,
        action_horizon=cfg_dit.action_horizon,
        cond_dim=cfg_dit.embed_dim,
        hidden_dim=cfg_dit.action_head_dim,
        num_blocks=cfg_dit.action_head_blocks,
        num_heads=cfg_dit.action_head_heads,
    )
    xt = CrossTransformerActionHead(
        action_dim=cfg_xt.action_dim,
        action_horizon=cfg_xt.action_horizon,
        cond_dim=cfg_xt.embed_dim,
        hidden_dim=cfg_xt.action_head_dim,
        num_layers=cfg_xt.action_head_blocks,
        num_heads=cfg_xt.action_head_heads,
    )
    assert count_action_head_params(xt) > 0
    assert count_action_head_params(dit) > 0


def test_sample_actions_cross_transformer():
    cfg = _tiny_cfg("cross_transformer")
    model = HPTModel(cfg, device="cpu")
    model.eval()
    b = 1
    sample = {
        "base_0_rgb": torch.zeros(b, 1, 32, 32, 3),
        "state": torch.zeros(b, cfg.proprio_dim),
        "state_history": torch.zeros(b, 1, cfg.proprio_dim),
        "prompts": [""],
        "is_ego": torch.zeros(b, dtype=torch.bool),
    }
    out = model.sample_actions(sample, num_steps=2)
    assert out.shape == (b, cfg.action_horizon, cfg.action_dim)
