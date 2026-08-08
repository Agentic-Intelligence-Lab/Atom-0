"""Unit tests for MoT video↔action attention masks (no Wan weight load)."""

from __future__ import annotations

import torch

from openpi.models_pytorch.fastwam.wan22.fastwam import FastWAM


class _FakeVideoExpert:
    def build_video_to_video_mask(self, video_seq_len, video_tokens_per_frame, device):
        return torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)


def _make_fastwam(
    *,
    video_to_action: bool = True,
    action_to_video: str = "first_frame",
    video_to_action_mode: str = "group_diagonal",
) -> FastWAM:
    fw = FastWAM.__new__(FastWAM)
    fw.video_expert = _FakeVideoExpert()
    fw.mot_video_attends_to_action = video_to_action
    fw.mot_action_attends_to_video = action_to_video
    fw.mot_video_to_action_mode = video_to_action_mode
    return fw


def test_legacy_mask_no_video_to_action() -> None:
    fw = _make_fastwam(video_to_action=False)
    # 3 frames × 4 tokens; 16 action tokens
    mask = fw._build_mot_attention_mask(12, 16, 4, torch.device("cpu"))
    assert mask.shape == (28, 28)
    # action -> first frame only
    assert mask[12:, :4].all()
    assert not mask[12:, 4:12].any()
    # video ↛ action
    assert not mask[:12, 12:].any()


def test_video_to_action_group_diagonal_skips_first_frame() -> None:
    fw = _make_fastwam(video_to_action=True, video_to_action_mode="group_diagonal")
    # 3 frames × 4 tokens → 2 temporal groups; 16 actions → 8 per group
    mask = fw._build_mot_attention_mask(12, 16, 4, torch.device("cpu"))
    v2a = mask[:12, 12:]
    assert not v2a[:4].any()  # first frame does not attend to action
    # frame1 (rows 4:8) attends only to first action group (cols 0:8)
    assert v2a[4:8, :8].all()
    assert not v2a[4:8, 8:].any()
    # frame2 (rows 8:12) attends only to second action group (cols 8:16)
    assert not v2a[8:12, :8].any()
    assert v2a[8:12, 8:].all()


def test_action_to_full_video() -> None:
    fw = _make_fastwam(video_to_action=False, action_to_video="full")
    mask = fw._build_mot_attention_mask(12, 16, 4, torch.device("cpu"))
    assert mask[12:, :12].all()


def test_config_defaults_enable_video_to_action() -> None:
    from openpi.models.fastwam_config import FastWAMConfig

    cfg = FastWAMConfig()
    assert cfg.mot_video_attends_to_action is True
    assert cfg.mot_action_attends_to_video == "first_frame"
    assert cfg.mot_video_to_action_mode == "group_diagonal"
