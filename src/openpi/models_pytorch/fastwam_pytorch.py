"""PyTorch FastWAM adapter that consumes Atom-0 Observation / Actions batches."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

import openpi.models.fastwam_config as fastwam_config
import openpi.models.model as _model
from openpi.models_pytorch.fastwam.factory import create_fastwam


def _as_torch(x, *, device: torch.device, dtype: torch.dtype | None = None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        out = x.to(device=device)
    else:
        out = torch.as_tensor(np.asarray(x), device=device)
    if dtype is not None:
        out = out.to(dtype=dtype)
    return out


def _images_to_video(
    images: dict[str, torch.Tensor],
    camera_keys: tuple[str, ...],
    concat_mode: str,
) -> torch.Tensor:
    """Convert Atom-0 image dict (B,T,H,W,C) in [-1,1] to FastWAM video (B,3,T,H,W')."""
    frames = []
    for key in camera_keys:
        if key not in images:
            raise KeyError(f"Missing camera key '{key}' in observation.images; have {list(images)}")
        img = images[key]
        if img.ndim == 4:
            # (B,H,W,C) → single-frame video
            img = img.unsqueeze(1)
        if img.ndim != 5:
            raise ValueError(f"Expected image '{key}' as (B,T,H,W,C) or (B,H,W,C), got {tuple(img.shape)}")
        # (B,T,H,W,C) → (B,3,T,H,W)
        frames.append(img.permute(0, 4, 1, 2, 3).contiguous())

    if concat_mode == "none":
        return frames[0]
    if concat_mode == "horizontal":
        return torch.cat(frames, dim=-1)
    if concat_mode == "vertical":
        return torch.cat(frames, dim=-2)
    raise ValueError(f"Unknown concat_multi_camera mode: {concat_mode}")


class FastWAMPytorch(nn.Module):
    """Atom-0 facing wrapper around the FastWAM MoT world-action model."""

    def __init__(self, config: fastwam_config.FastWAMConfig, *, device: str = "cuda"):
        super().__init__()
        self.config = config
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.max_token_len = config.max_token_len

        dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float32
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"

        self.fastwam = create_fastwam(
            model_id=config.model_id,
            tokenizer_model_id=config.tokenizer_model_id,
            video_dit_config=config.resolved_video_dit_config(),
            action_dit_config=config.resolved_action_dit_config(),
            tokenizer_max_len=config.max_token_len,
            load_text_encoder=config.load_text_encoder,
            proprio_dim=config.proprio_dim,
            action_dit_pretrained_path=config.action_dit_pretrained_path,
            skip_dit_load_from_pretrain=config.skip_dit_load_from_pretrain,
            skip_vae_load_from_pretrain=config.skip_vae_load_from_pretrain,
            video_scheduler=dict(config.video_scheduler),
            action_scheduler=dict(config.action_scheduler),
            loss=dict(config.loss),
            mot_checkpoint_mixed_attn=config.mot_checkpoint_mixed_attn,
            redirect_common_files=config.redirect_common_files,
            model_dtype=dtype,
            device=device,
        )
        # Expose MoT as `.dit` for freeze / optimizer patterns matching upstream FastWAM.
        self.dit = self.fastwam.dit

    @property
    def device(self) -> torch.device:
        return self.fastwam.device

    def freeze_encoders(self) -> None:
        """Freeze VAE / text encoder; train MoT (+ optional proprio encoder) only."""
        for p in self.fastwam.vae.parameters():
            p.requires_grad = False
        if self.fastwam.text_encoder is not None:
            for p in self.fastwam.text_encoder.parameters():
                p.requires_grad = False

    def observation_to_sample(
        self,
        observation: _model.Observation,
        actions: torch.Tensor | None = None,
        *,
        prompts: list[str] | None = None,
    ) -> dict[str, Any]:
        """Map Atom-0 Observation (+ Actions) to a FastWAM training/inference sample dict."""
        device = self.device
        dtype = self.fastwam.torch_dtype

        if prompts is None:
            prompts = getattr(observation, "_fastwam_prompts", None)

        images = {k: _as_torch(v, device=device, dtype=torch.float32) for k, v in observation.images.items()}
        video = _images_to_video(images, self.config.camera_keys, self.config.concat_multi_camera)
        video = video.to(dtype=dtype)

        state = _as_torch(observation.state, device=device, dtype=dtype)
        # proprio: [B, T, D] — use current state broadcast over action horizon when no history.
        if state.ndim == 2:
            # Trim / pad state to proprio_dim.
            if state.shape[-1] < self.config.proprio_dim:
                pad = torch.zeros(state.shape[0], self.config.proprio_dim - state.shape[-1], device=device, dtype=dtype)
                state = torch.cat([state, pad], dim=-1)
            elif state.shape[-1] > self.config.proprio_dim:
                state = state[..., : self.config.proprio_dim]
            proprio = state.unsqueeze(1).expand(-1, self.action_horizon, -1)
        else:
            proprio = state[..., : self.config.proprio_dim]

        sample: dict[str, Any] = {
            "video": video,
            "proprio": proprio,
        }

        if actions is not None:
            action = _as_torch(actions, device=device, dtype=dtype)
            # Trim / pad to configured action_dim.
            if action.shape[-1] < self.action_dim:
                pad = torch.zeros(
                    *action.shape[:-1], self.action_dim - action.shape[-1], device=device, dtype=dtype
                )
                action = torch.cat([action, pad], dim=-1)
            elif action.shape[-1] > self.action_dim:
                action = action[..., : self.action_dim]
            sample["action"] = action

        # Language conditioning: prefer precomputed T5 context on Observation; else encode prompts.
        if observation.context is not None and observation.context_mask is not None:
            sample["context"] = _as_torch(observation.context, device=device, dtype=dtype)
            sample["context_mask"] = _as_torch(observation.context_mask, device=device, dtype=torch.bool)
        elif prompts is not None and self.fastwam.text_encoder is not None:
            context, context_mask = self.fastwam.encode_prompt(prompts)
            sample["context"] = context.to(dtype=dtype)
            sample["context_mask"] = context_mask
        elif self.fastwam.text_encoder is not None:
            bsz = video.shape[0]
            context, context_mask = self.fastwam.encode_prompt([""] * bsz)
            sample["context"] = context.to(dtype=dtype)
            sample["context_mask"] = context_mask
        else:
            # Debug / offline path without T5: zero context (cross-attn sees empty conditioning).
            bsz = video.shape[0]
            sample["context"] = torch.zeros(
                bsz, self.max_token_len, self.fastwam.text_dim, device=device, dtype=dtype
            )
            sample["context_mask"] = torch.ones(bsz, self.max_token_len, device=device, dtype=torch.bool)

        # Optional pad masks from temporal image masks.
        if observation.image_masks:
            first_key = self.config.camera_keys[0]
            mask = observation.image_masks.get(first_key)
            if mask is not None:
                mask_t = _as_torch(mask, device=device, dtype=torch.bool)
                if mask_t.ndim == 2:
                    sample["image_is_pad"] = ~mask_t

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
        """Return dict with total / ego|robot × video|action losses (scalars)."""
        was_training = self.training
        self.train(train)
        sample = self.observation_to_sample(observation, actions, prompts=prompts)
        loss_total, loss_dict = self.fastwam.training_loss(sample)
        self.train(was_training)
        out = {
            "loss": loss_total,
            "loss_ego_video": torch.as_tensor(loss_dict["loss_ego_video"], device=self.device),
            "loss_ego_action": torch.as_tensor(loss_dict["loss_ego_action"], device=self.device),
            "loss_robot_video": torch.as_tensor(loss_dict["loss_robot_video"], device=self.device),
            "loss_robot_action": torch.as_tensor(loss_dict["loss_robot_action"], device=self.device),
            "loss_video": torch.as_tensor(loss_dict["loss_video"], device=self.device),
            "loss_action": torch.as_tensor(loss_dict["loss_action"], device=self.device),
        }
        return out

    @torch.no_grad()
    def sample_actions(
        self,
        device,
        observation: _model.Observation,
        *,
        prompts: list[str] | None = None,
        num_inference_steps: int | None = None,
        seed: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Action-only Fast inference (cached first-frame video KV).

        Signature matches Atom-0 PyTorch Policy: ``sample_actions(device, observation, ...)``.
        Returns (B, H, D).
        """
        self.eval()
        if prompts is None:
            prompts = getattr(observation, "_fastwam_prompts", None)

        if isinstance(device, str):
            torch_device = torch.device(device)
        elif isinstance(device, torch.device):
            torch_device = device
        else:
            torch_device = self.device
        if str(torch_device) != str(self.device):
            self.to(torch_device)

        dtype = self.fastwam.torch_dtype
        steps = num_inference_steps or self.config.num_inference_steps

        images = {k: _as_torch(v, device=self.device, dtype=torch.float32) for k, v in observation.images.items()}
        current = {}
        for k, img in images.items():
            if img.ndim == 5:
                current[k] = img[:, 0]
            else:
                current[k] = img
        video = _images_to_video(
            {k: v.unsqueeze(1) for k, v in current.items()},
            self.config.camera_keys,
            self.config.concat_multi_camera,
        )
        input_image = video[:, :, 0]

        state = _as_torch(observation.state, device=self.device, dtype=dtype)
        if state.ndim == 2:
            proprio = state[..., : self.config.proprio_dim]
        else:
            proprio = state[:, -1, : self.config.proprio_dim]

        context = context_mask = None
        prompt_arg: str | None = None
        if observation.context is not None and observation.context_mask is not None:
            context = _as_torch(observation.context, device=self.device, dtype=dtype)
            context_mask = _as_torch(observation.context_mask, device=self.device, dtype=torch.bool)
        elif prompts is not None:
            if len(prompts) == 1:
                prompt_arg = prompts[0]
            else:
                context, context_mask = self.fastwam.encode_prompt(prompts)
                context = context.to(dtype=dtype)
        else:
            prompt_arg = ""

        actions_out = []
        bsz = input_image.shape[0]
        for i in range(bsz):
            ctx_i = context[i : i + 1] if context is not None else None
            ctxm_i = context_mask[i : i + 1] if context_mask is not None else None
            out = self.fastwam.infer_action(
                prompt=prompt_arg if context is None else None,
                input_image=input_image[i],
                action_horizon=self.action_horizon,
                proprio=proprio[i],
                context=ctx_i,
                context_mask=ctxm_i,
                num_inference_steps=steps,
                seed=None if seed is None else seed + i,
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k in ("sigma_shift", "tiled", "rand_device", "text_cfg_scale", "negative_prompt")
                },
            )
            actions_out.append(out["action"])
        return torch.stack(actions_out, dim=0).to(device=self.device, dtype=torch.float32)
