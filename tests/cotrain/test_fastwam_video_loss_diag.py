"""Unit tests for FastWAM video-loss diagnostics / fixed-sigma helpers."""

from __future__ import annotations

import torch

from openpi.cotrain import config
from openpi.models_pytorch.fastwam.wan22.schedulers.scheduler_continuous import (
    WanContinuousFlowMatchScheduler,
)


def test_timestep_from_sigma_half() -> None:
    sched = WanContinuousFlowMatchScheduler(num_train_timesteps=1000, shift=5.0)
    t = sched.timestep_from_sigma(0.5, batch_size=4, device=torch.device("cpu"), dtype=torch.float32)
    assert t.shape == (4,)
    assert torch.allclose(t, torch.full((4,), 500.0))
    w = sched.training_weight(t)
    # Mid-timestep weight is the peak (~2.08 after normalization).
    assert float(w.mean()) > 1.5


def test_wam_cross_piper_overfit_config() -> None:
    cfg = config.get_config("wam-cross-piper-overfit")
    assert {d.uid for d in cfg.data.datasets} == {"piper2", "piper30"}
    assert cfg.overfit_fixed_batch is True
    assert cfg.overfit_fixed_noise is True
    assert cfg.fixed_video_sigma == 0.5
    assert cfg.fixed_action_sigma == 0.5
    assert cfg.batch_size == 8
    assert cfg.num_train_steps == 500


def test_video_raw_vs_weighted_at_fixed_sigma() -> None:
    """At σ=0.5, weighted loss should be raw * w(t) for uniform per-sample losses."""
    sched = WanContinuousFlowMatchScheduler(num_train_timesteps=1000, shift=5.0)
    t = sched.timestep_from_sigma(0.5, batch_size=8, device=torch.device("cpu"), dtype=torch.float32)
    w = sched.training_weight(t).float()
    raw = torch.full((8,), 0.2, dtype=torch.float32)
    weighted = (raw * w).mean()
    assert torch.isclose(weighted, raw.mean() * w.mean())
    assert float(w.mean()) != 1.0  # confirms weighting is active at σ=0.5
