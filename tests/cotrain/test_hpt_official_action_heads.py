"""Smoke tests for official HPT-style diffusion and transformer_decoder heads."""

from __future__ import annotations

import torch

from openpi.models.hpt_config import HPTConfig
from openpi.models_pytorch.hpt.model import HPTModel
from openpi.models_pytorch.hpt.official_action_heads import DiffusionActionHead, TransformerDecoderActionHead


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
        action_head_heads=4,
        num_inference_steps=4,
        diffusion_train_timesteps=20,
        diffusion_down_dims=(32, 64),
    )


def _sample(cfg: HPTConfig, batch: int = 2) -> dict:
    b, t, h, w = batch, 1, 32, 32
    return {
        "base_0_rgb": torch.zeros(b, t, h, w, 3),
        "state": torch.zeros(b, cfg.proprio_dim),
        "state_history": torch.zeros(b, 1, cfg.proprio_dim),
        "action": torch.randn(b, cfg.action_horizon, cfg.action_dim),
        "action_mask": torch.ones(b, cfg.action_dim, dtype=torch.bool),
        "prompts": ["a"] * b,
        "is_ego": torch.zeros(b, dtype=torch.bool),
    }


def test_diffusion_head_train_and_sample():
    cfg = _tiny_cfg("diffusion")
    model = HPTModel(cfg, device="cpu")
    model.train()
    loss, stats = model.training_loss(_sample(cfg))
    assert torch.isfinite(loss)
    loss.backward()
    assert model.action_head.unet.final_conv[1].weight.grad is not None

    model.eval()
    out = model.sample_actions(_sample(cfg, batch=1), num_steps=4)
    assert out.shape == (1, cfg.action_horizon, cfg.action_dim)
    assert torch.isfinite(out).all()
    assert stats["loss_action"] >= 0


def test_transformer_decoder_head_train_and_sample():
    cfg = _tiny_cfg("transformer_decoder")
    model = HPTModel(cfg, device="cpu")
    model.train()
    loss, _ = model.training_loss(_sample(cfg))
    assert torch.isfinite(loss)
    loss.backward()
    assert model.action_head.query_tokens.grad is not None

    model.eval()
    out = model.sample_actions(_sample(cfg, batch=1))
    assert out.shape == (1, cfg.action_horizon, cfg.action_dim)


def test_official_heads_use_obs_only_trunk():
    cfg = _tiny_cfg("transformer_decoder")
    model = HPTModel(cfg, device="cpu")
    b = 2
    obs = torch.randn(b, 72, cfg.embed_dim)
    out = model.forward_trunk_obs(obs)
    assert out.shape == (b, 72, cfg.embed_dim)


def test_diffusion_head_direct():
    cfg = _tiny_cfg("diffusion")
    head = DiffusionActionHead(
        action_dim=cfg.action_dim,
        action_horizon=cfg.action_horizon,
        cond_dim=cfg.embed_dim,
        train_timesteps=20,
        num_inference_steps=4,
        down_dims=(32, 64),
    )
    b = 2
    cond = torch.randn(b, cfg.embed_dim)
    actions = torch.randn(b, cfg.action_horizon, cfg.action_dim)
    per = head.compute_loss(cond, actions)
    assert per.shape == (b,)
    pred = head.sample(cond, num_steps=4)
    assert pred.shape == actions.shape


def test_transformer_decoder_head_direct():
    cfg = _tiny_cfg("transformer_decoder")
    head = TransformerDecoderActionHead(
        action_dim=cfg.action_dim,
        action_horizon=cfg.action_horizon,
        cond_dim=cfg.embed_dim,
        hidden_dim=64,
        num_heads=4,
    )
    b = 2
    ctx = torch.randn(b, 72, cfg.embed_dim)
    out = head(ctx)
    assert out.shape == (b, cfg.action_horizon, cfg.action_dim)
