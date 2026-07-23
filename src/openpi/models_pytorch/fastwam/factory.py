"""Factory helpers for constructing the basic FastWAM MoT model (no Hydra)."""

from __future__ import annotations

from typing import Any

import torch

from openpi.models_pytorch.fastwam.wan22.fastwam import FastWAM


def create_fastwam(
    *,
    model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
    tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
    video_dit_config: dict[str, Any],
    action_dit_config: dict[str, Any] | None = None,
    tokenizer_max_len: int = 128,
    load_text_encoder: bool = True,
    proprio_dim: int | None = None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    video_scheduler: dict[str, Any] | None = None,
    action_scheduler: dict[str, Any] | None = None,
    loss: dict[str, Any] | None = None,
    mot_checkpoint_mixed_attn: bool = True,
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> FastWAM:
    """Create the basic (uncond) FastWAM model used for Atom-0 integration."""
    if not isinstance(video_dit_config, dict):
        raise ValueError(f"`video_dit_config` must be a dict, got {type(video_dit_config)}")
    action_dit_config = {} if action_dit_config is None else dict(action_dit_config)
    video_scheduler = {} if video_scheduler is None else dict(video_scheduler)
    if action_scheduler is None:
        action_scheduler = {"train_shift": 5.0, "infer_shift": 5.0, "num_train_timesteps": 1000}
    action_scheduler = dict(action_scheduler)
    required = {"train_shift", "infer_shift", "num_train_timesteps"}
    missing = required - set(action_scheduler)
    if missing:
        raise ValueError(f"`action_scheduler` missing keys: {sorted(missing)}")
    loss = {} if loss is None else dict(loss)

    return FastWAM.from_wan22_pretrained(
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        load_text_encoder=bool(load_text_encoder),
        proprio_dim=(None if proprio_dim is None else int(proprio_dim)),
        redirect_common_files=bool(redirect_common_files),
        video_dit_config=video_dit_config,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
    )
