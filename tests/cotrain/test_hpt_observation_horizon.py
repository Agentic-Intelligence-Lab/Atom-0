"""HPT observation_horizon=4: RLDS offsets + stem compression keeps trunk length."""

from __future__ import annotations

import torch

from openpi.cotrain.rlds_dataset import build_video_frame_indices
from openpi.models.hpt_config import HPTConfig
from openpi.models_pytorch.hpt.model import HPTModel


def test_video_frame_indices_hpt_horizon4():
    idx = build_video_frame_indices(
        video_num_frames=2,
        action_video_freq_ratio=50,
        action_chunk_size=50,
        observation_horizon=4,
    )
    assert idx == [-3, -2, -1, 0, 50]


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
    cfg = HPTConfig(load_encoders=False, observation_horizon=4, video_num_frames=2)
    assert cfg.history_length == 4
    assert cfg.image_time_length == 5


def test_encode_obs_horizon4_keeps_stem_token_count():
    cfg = HPTConfig(
        load_encoders=False,
        observation_horizon=4,
        random_horizon_masking=False,
        action_horizon=8,
        num_inference_steps=2,
        num_blocks=2,
        num_action_tokens=64,
        num_future_tokens=16,
        action_head_type="mlp",
        action_head_dim=32,
    )
    model = HPTModel(cfg, device="cpu")
    model.eval()
    b, t, h, w = 2, 4, 32, 32
    obs, aux = model.encode_obs(
        base_img=torch.zeros(b, t, h, w, 3),
        left_wrist=torch.zeros(b, t, h, w, 3),
        right_wrist=torch.zeros(b, t, h, w, 3),
        state=torch.zeros(b, t, cfg.proprio_dim),
        prompts=["pick"] * b,
        is_ego=torch.tensor([False, True]),
    )
    n_obs = cfg.ego_tokens + cfg.robot_tokens + cfg.wrist_tokens + cfg.state_tokens + cfg.language_tokens
    assert obs.shape == (b, n_obs, cfg.embed_dim)
    tokens, action_f, future_f = model.forward_trunk(obs)
    assert tokens.shape[1] == n_obs + cfg.num_action_tokens + cfg.num_future_tokens
    assert action_f.shape == (b, cfg.num_action_tokens, cfg.embed_dim)
    assert future_f.shape == (b, cfg.num_future_tokens, cfg.embed_dim)
    assert aux["is_ego"].tolist() == [False, True]


def test_training_loss_splits_history_and_future_frames():
    cfg = HPTConfig(
        load_encoders=False,
        observation_horizon=4,
        random_horizon_masking=False,
        video_num_frames=2,
        action_horizon=8,
        num_inference_steps=2,
        num_blocks=2,
        action_head_type="mlp",
        action_head_dim=32,
    )
    model = HPTModel(cfg, device="cpu")
    model.train()
    b, t, h, w = 2, 5, 32, 32
    sample = {
        "base_0_rgb": torch.zeros(b, t, h, w, 3),
        "left_wrist_0_rgb": torch.zeros(b, t, h, w, 3),
        "right_wrist_0_rgb": torch.zeros(b, t, h, w, 3),
        "state": torch.zeros(b, cfg.proprio_dim),
        "state_history": torch.zeros(b, 4, cfg.proprio_dim),
        "action": torch.zeros(b, cfg.action_horizon, cfg.action_dim),
        "action_mask": torch.ones(b, cfg.action_dim, dtype=torch.bool),
        "prompts": ["pick", "place"],
        "is_ego": torch.tensor([False, True]),
    }
    loss, stats = model.training_loss(sample)
    assert torch.isfinite(loss)
    assert "loss_action" in stats

