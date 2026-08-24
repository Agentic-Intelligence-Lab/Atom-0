"""HPT current-frame obs + optional future frame for world head."""

from __future__ import annotations

import torch

from openpi.cotrain.rlds_dataset import build_video_frame_indices
from openpi.models.hpt_config import HPTConfig
from openpi.models_pytorch.hpt.model import HPTModel


def _tiny_world_kwargs() -> dict:
    return {
        "world_dit_dim": 32,
        "world_dit_blocks": 1,
        "world_dit_heads": 4,
        "world_wide_dim": 64,
        "world_wide_blocks": 1,
        "world_wide_heads": 4,
        "world_num_patches": 256,
    }


def test_video_frame_indices_hpt_current_and_future():
    idx = build_video_frame_indices(
        video_num_frames=2,
        action_video_freq_ratio=50,
        action_chunk_size=50,
        observation_horizon=1,
    )
    assert idx == [0, 50]


def test_video_frame_indices_fastwam_unchanged():
    idx = build_video_frame_indices(
        video_num_frames=9,
        action_video_freq_ratio=4,
        action_chunk_size=32,
        observation_horizon=1,
    )
    assert idx is not None
    assert idx[0] == 0
    assert all(i >= 0 for i in idx)


def test_hpt_config_image_time_length():
    cfg = HPTConfig(load_encoders=False, observation_horizon=1, video_num_frames=2)
    assert cfg.history_length == 1
    assert cfg.image_time_length == 2
    assert cfg.obs_token_count == 72


def test_encode_obs_current_frame_token_layout():
    cfg = HPTConfig(
        load_encoders=False,
        observation_horizon=1,
        random_horizon_masking=False,
        action_horizon=8,
        num_inference_steps=2,
        num_blocks=2,
        head_mode="action_only",
        action_head_type="transformer_decoder",
        action_head_dim=32,
        action_head_heads=4,
    )
    model = HPTModel(cfg, device="cpu")
    model.eval()
    b, t, h, w = 2, 1, 32, 32
    obs, aux = model.encode_obs(
        base_img=torch.zeros(b, t, h, w, 3),
        left_wrist=torch.zeros(b, t, h, w, 3),
        right_wrist=torch.zeros(b, t, h, w, 3),
        state=torch.zeros(b, t, cfg.proprio_dim),
        prompts=["pick"] * b,
        is_ego=torch.tensor([False, True]),
    )
    assert obs.shape == (b, cfg.obs_token_count, cfg.embed_dim)
    trunk = model.forward_trunk_obs(obs)
    assert trunk.shape == (b, cfg.obs_token_count, cfg.embed_dim)
    assert not hasattr(model, "robot_stem")
    assert not hasattr(model, "action_tokens")
    assert not hasattr(model, "future_tokens")
    assert aux["is_ego"].tolist() == [False, True]
    # Human wrist tokens are zeroed.
    wrist = obs[:, cfg.ego_tokens : cfg.ego_tokens + cfg.wrist_tokens]
    assert torch.allclose(wrist[1], torch.zeros_like(wrist[1]))


def test_training_loss_splits_current_and_future_frames():
    cfg = HPTConfig(
        load_encoders=False,
        observation_horizon=1,
        random_horizon_masking=False,
        video_num_frames=2,
        action_horizon=8,
        num_inference_steps=2,
        num_blocks=2,
        action_head_type="transformer_decoder",
        action_head_dim=32,
        action_head_heads=4,
        **_tiny_world_kwargs(),
    )
    model = HPTModel(cfg, device="cpu")
    model.train()
    b, t, h, w = 2, 2, 32, 32
    sample = {
        "base_0_rgb": torch.zeros(b, t, h, w, 3),
        "left_wrist_0_rgb": torch.zeros(b, t, h, w, 3),
        "right_wrist_0_rgb": torch.zeros(b, t, h, w, 3),
        "state": torch.zeros(b, cfg.proprio_dim),
        "state_history": torch.zeros(b, 1, cfg.proprio_dim),
        "action": torch.zeros(b, cfg.action_horizon, cfg.action_dim),
        "action_mask": torch.ones(b, cfg.action_dim, dtype=torch.bool),
        "prompts": ["pick", "place"],
        "is_ego": torch.tensor([False, True]),
    }
    loss, stats = model.training_loss(sample)
    assert torch.isfinite(loss)
    assert "loss_action" in stats
    assert "loss_world" in stats
    loss.backward()
    assert model.world_head.input_proj.weight.grad is not None
    assert model.action_head.query_tokens.grad is not None
