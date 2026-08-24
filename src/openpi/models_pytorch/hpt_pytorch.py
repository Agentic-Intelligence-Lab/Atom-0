"""Atom-0 Observation ↔ HPTModel adapter (mirrors FastWAMPytorch pattern)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn

import openpi.models.model as _model
from openpi.models.hpt_config import HPTConfig
from openpi.models_pytorch.hpt.model import HPTModel

logger = logging.getLogger("openpi")


def _as_torch(x, *, device, dtype=None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        out = x.to(device=device)
    else:
        out = torch.as_tensor(np.asarray(x), device=device)
    if dtype is not None:
        out = out.to(dtype=dtype)
    return out


class HPTPytorch(nn.Module):
    """Thin wrapper: Observation/Actions → HPT training_loss / sample_actions."""

    def __init__(self, config: HPTConfig, *, device: str = "cuda"):
        super().__init__()
        self.config = config
        self.device = torch.device(device)
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.hpt = HPTModel(config, device=device)
        self.to(self.device)

    def freeze_encoders(self) -> None:
        self.hpt.freeze_encoders()

    def freeze_trunk(self) -> None:
        self.hpt.freeze_trunk()

    def trainable_parameters(self) -> list[nn.Parameter]:
        return self.hpt.trainable_parameters()

    def observation_to_sample(
        self,
        observation: _model.Observation,
        actions: torch.Tensor | None = None,
        *,
        prompts: list[str] | None = None,
    ) -> dict[str, Any]:
        device = self.device
        dtype = torch.float32

        if prompts is None:
            prompts = getattr(observation, "_fastwam_prompts", None)
        if prompts is None:
            prompts = [""] * int(_as_torch(observation.state, device=device).shape[0])

        sample: dict[str, Any] = {"prompts": [str(p) for p in prompts]}

        for key in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"):
            if key not in observation.images:
                continue
            img = _as_torch(observation.images[key], device=device, dtype=dtype)
            # [B,H,W,3] or [B,T,H,W,3]
            sample[key] = img
            if observation.image_masks and key in observation.image_masks:
                sample[f"{key}_mask"] = _as_torch(observation.image_masks[key], device=device, dtype=torch.bool)

        sample["state"] = _as_torch(observation.state, device=device, dtype=dtype)
        if observation.state_history is not None:
            sample["state_history"] = _as_torch(observation.state_history, device=device, dtype=dtype)
            if sample["state_history"].shape[-1] < self.config.proprio_dim:
                pad = torch.zeros(
                    *sample["state_history"].shape[:-1],
                    self.config.proprio_dim - sample["state_history"].shape[-1],
                    device=device,
                    dtype=dtype,
                )
                sample["state_history"] = torch.cat([sample["state_history"], pad], dim=-1)
            elif sample["state_history"].shape[-1] > self.config.proprio_dim:
                sample["state_history"] = sample["state_history"][..., : self.config.proprio_dim]
        if sample["state"].shape[-1] < self.config.proprio_dim:
            pad = torch.zeros(
                *sample["state"].shape[:-1],
                self.config.proprio_dim - sample["state"].shape[-1],
                device=device,
                dtype=dtype,
            )
            sample["state"] = torch.cat([sample["state"], pad], dim=-1)
        elif sample["state"].shape[-1] > self.config.proprio_dim:
            sample["state"] = sample["state"][..., : self.config.proprio_dim]

        if actions is not None:
            action = _as_torch(actions, device=device, dtype=dtype)
            if action.shape[-1] < self.action_dim:
                pad = torch.zeros(*action.shape[:-1], self.action_dim - action.shape[-1], device=device, dtype=dtype)
                action = torch.cat([action, pad], dim=-1)
            elif action.shape[-1] > self.action_dim:
                action = action[..., : self.action_dim]
            sample["action"] = action

        if observation.action_mask is not None:
            sample["action_mask"] = _as_torch(observation.action_mask, device=device, dtype=torch.bool)

        is_ego = getattr(observation, "_fastwam_is_ego", None)
        if is_ego is not None:
            sample["is_ego"] = _as_torch(is_ego, device=device, dtype=torch.bool).reshape(-1)

        return sample

    def compute_loss(
        self,
        observation: _model.Observation,
        actions: torch.Tensor,
        *,
        prompts: list[str] | None = None,
        train: bool = True,
    ) -> dict[str, torch.Tensor]:
        was_training = self.training
        self.train(train)
        sample = self.observation_to_sample(observation, actions, prompts=prompts)
        loss, stats = self.hpt.training_loss(sample)
        self.train(was_training)
        out = {"loss": loss}
        for k, v in stats.items():
            if k == "loss":
                continue  # stats["loss"] is detached; keep grad-enabled loss above.
            out[k] = torch.as_tensor(v, device=self.device) if not isinstance(v, torch.Tensor) else v
        return out

    @torch.no_grad()
    def sample_actions(
        self,
        device,
        observation: _model.Observation,
        *,
        prompts: list[str] | None = None,
        num_steps: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del device, kwargs
        sample = self.observation_to_sample(observation, actions=None, prompts=prompts)
        return self.hpt.sample_actions(sample, num_steps=num_steps)
