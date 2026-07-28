"""Atom-0 / openpi config for the FastWAM (World-Action Model) algorithm."""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
from typing_extensions import override

import openpi.models.model as _model
import openpi.shared.array_typing as at


def default_video_dit_config(*, action_dim: int) -> dict[str, Any]:
    return {
        "has_image_input": False,
        "patch_size": [1, 2, 2],
        "in_dim": 48,
        "hidden_dim": 3072,
        "ffn_dim": 14336,
        "freq_dim": 256,
        "text_dim": 4096,
        "out_dim": 48,
        "num_heads": 24,
        "attn_head_dim": 128,
        "num_layers": 30,
        "eps": 1.0e-06,
        "seperated_timestep": True,
        "require_clip_embedding": False,
        "require_vae_embedding": False,
        "fuse_vae_embedding_in_latents": True,
        "use_gradient_checkpointing": True,
        "video_attention_mask_mode": "first_frame_causal",
        "action_conditioned": False,
        "action_dim": action_dim,
        "action_group_causal_mask_mode": "group_diagonal",
    }


def default_action_dit_config(*, action_dim: int) -> dict[str, Any]:
    return {
        "action_dim": action_dim,
        "hidden_dim": 1024,
        "ffn_dim": 4096,
        "num_heads": 24,
        "attn_head_dim": 128,
        "num_layers": 30,
        "text_dim": 4096,
        "freq_dim": 256,
        "eps": 1.0e-06,
        "use_gradient_checkpointing": True,
    }


@dataclasses.dataclass(frozen=True)
class FastWAMConfig(_model.BaseModelConfig):
    """Configuration for the FastWAM MoT (Video DiT ‖ Action DiT) algorithm.

    Primary training path: Atom-0 **cotrain RLDS**
    (``openpi.cotrain.config`` + ``scripts/train_fastwam.py``).
    The implementation is PyTorch (``models_pytorch.fastwam``).
    """

    # Match LIBERO FastWAM defaults: 32 actions ↔ 9 video frames (ratio=4).
    action_dim: int = 7
    action_horizon: int = 32
    max_token_len: int = 128  # Wan / UMT5 context length (not PaliGemma)

    # FastWAM-specific.
    proprio_dim: int = 8
    video_num_frames: int = 9
    action_video_freq_ratio: int = 4
    image_resolution: tuple[int, int] = (224, 224)
    concat_multi_camera: str = "horizontal"  # "horizontal" | "none"
    camera_keys: tuple[str, ...] = ("base_0_rgb", "left_wrist_0_rgb")

    model_id: str = "Wan-AI/Wan2.2-TI2V-5B"
    tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B"
    load_text_encoder: bool = True
    redirect_common_files: bool = False  # use Wan2.2 .pth on PFS; DiffSynth mirror unavailable
    mot_checkpoint_mixed_attn: bool = True
    skip_dit_load_from_pretrain: bool = False
    skip_vae_load_from_pretrain: bool = False
    action_dit_pretrained_path: str | None = None

    # Optional override dicts; None → defaults above.
    video_dit_config: dict[str, Any] | None = None
    action_dit_config: dict[str, Any] | None = None
    video_scheduler: dict[str, Any] = dataclasses.field(
        default_factory=lambda: {"train_shift": 5.0, "infer_shift": 5.0, "num_train_timesteps": 1000}
    )
    action_scheduler: dict[str, Any] = dataclasses.field(
        default_factory=lambda: {"train_shift": 5.0, "infer_shift": 5.0, "num_train_timesteps": 1000}
    )
    # Four-way domain × head loss weights. Ego samples are those whose dataset_id
    # starts with ``egoverse`` (see DispatchNormalize ``is_ego`` tag).
    # Legacy keys ``lambda_video`` / ``lambda_action`` still work as a shared fallback
    # for both ego and robot when the four-way keys are absent.
    loss: dict[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "lambda_ego_video": 1.0,
            "lambda_ego_action": 1.0,
            "lambda_robot_video": 1.0,
            "lambda_robot_action": 1.0,
        }
    )

    dtype: str = "bfloat16"
    device: str = "cuda"
    num_inference_steps: int = 20

    # Used by Atom-0 data_loader to sample a future video window (not past MEM history).
    # video_num_frames frames spanning action_horizon steps.
    @property
    def history_length(self) -> int:
        # Intentionally 1: FastWAM uses future video deltas, not MEM past history.
        return 1

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.FASTWAM

    def resolved_video_dit_config(self) -> dict[str, Any]:
        cfg = default_video_dit_config(action_dim=self.action_dim)
        if self.video_dit_config:
            cfg.update(self.video_dit_config)
        cfg["action_dim"] = self.action_dim
        cfg["use_gradient_checkpointing"] = self.mot_checkpoint_mixed_attn
        return cfg

    def resolved_action_dit_config(self) -> dict[str, Any]:
        cfg = default_action_dit_config(action_dim=self.action_dim)
        if self.action_dit_config:
            cfg.update(self.action_dit_config)
        cfg["action_dim"] = self.action_dim
        cfg["use_gradient_checkpointing"] = self.mot_checkpoint_mixed_attn
        return cfg

    @override
    def create(self, rng: at.KeyArrayLike) -> "_model.BaseModel":
        # FastWAM is PyTorch-only; JAX create() is unsupported.
        raise NotImplementedError(
            "FastWAM is a PyTorch model. Use FastWAMConfig.create_pytorch() "
            "or scripts/train_fastwam.py instead of the JAX trainer."
        )

    def create_pytorch(self, *, device: str | None = None):
        """Instantiate the PyTorch FastWAM adapter used by Atom-0 training/inference."""
        from openpi.models_pytorch.fastwam_pytorch import FastWAMPytorch

        return FastWAMPytorch(self, device=device or self.device)

    @override
    def load_pytorch(self, train_config, weight_path: str):
        import safetensors.torch

        model = self.create_pytorch(device=getattr(train_config.model, "device", self.device))
        safetensors.torch.load_model(model, weight_path)
        return model

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        t = self.video_num_frames
        h, w = self.image_resolution
        image_spec = jax.ShapeDtypeStruct([batch_size, t, h, w, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size, t], jnp.bool_)
        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.proprio_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.bool_),
                context=jax.ShapeDtypeStruct([batch_size, self.max_token_len, 4096], jnp.float32),
                context_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.bool_),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)
        return observation_spec, action_spec
