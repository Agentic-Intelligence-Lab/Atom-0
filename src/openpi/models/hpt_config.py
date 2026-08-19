"""Atom-0 config for the Heterogeneous Pre-trained Transformer (HPT) algorithm."""

from __future__ import annotations

import dataclasses
from typing import Any, Literal

import jax
import jax.numpy as jnp
from typing_extensions import override

import openpi.models.model as _model
import openpi.shared.array_typing as at


@dataclasses.dataclass(frozen=True)
class HPTConfig(_model.BaseModelConfig):
    """HPT co-training config (PyTorch), HPT-base style trunk + dual heads.

    Data path reuses Atom-0 cotrain RLDS (same as pi05 / FastWAM): unified 80D
    actions, three camera slots, ``is_ego`` domain tags.
    """

    action_dim: int = 80
    action_horizon: int = 50
    # Prompt length for T5 tokenization (not PaliGemma).
    max_token_len: int = 32
    proprio_dim: int = 80

    # Trunk / stem width (DINOv2/T5 stay 768; stems project into this dim).
    embed_dim: int = 256
    num_blocks: int = 16
    num_heads: int = 8
    mlp_ratio: int = 4
    drop_path: float = 0.1

    # Stem token budgets (after cross-attn pooling).
    ego_tokens: int = 16
    robot_tokens: int = 16
    wrist_tokens: int = 16
    state_tokens: int = 16
    language_tokens: int = 8

    # Learnable query tokens into the shared trunk.
    num_action_tokens: int = 64
    num_future_tokens: int = 16

    # ``action_world``: action head + world head (future query tokens).
    # ``action_only``: no world head / future tokens / world loss.
    head_mode: Literal["action_world", "action_only"] = "action_world"

    # Action head: ``dit`` = Action-DiT (AdaLN + horizon self-attn + cross-attn).
    # ``cross_transformer`` = EgoWAM-style pre-norm self/cross-attn blocks + timestep add.
    # ``mlp`` = mean-pool + per-step MLP (legacy FM).
    # ``diffusion`` = official HPT Diffusion Policy (mean-pool cond + DDIM).
    # ``transformer_decoder`` = official HPT path B (full trunk context + cross-attn queries).
    action_head_type: Literal["mlp", "dit", "cross_transformer", "diffusion", "transformer_decoder"] = "dit"
    # Action-DiT width / depth (6 blocks / 128 hidden / 4 heads).
    action_head_dim: int = 128
    action_head_blocks: int = 6
    action_head_heads: int = 4
    action_head_mlp_ratio: int = 4
    action_head_dropout: float = 0.0
    action_head_drop_path: float = 0.0

    # Official diffusion head (HPT Diffusion Policy).
    diffusion_train_timesteps: int = 100
    diffusion_down_dims: tuple[int, ...] = (64, 128)
    # TransformerDecoder path B (Huber / smooth L1).
    transformer_decoder_huber_delta: float = 0.1

    # Frozen encoders.
    image_encoder: str = "facebook/dinov2-base"
    language_encoder: str = "t5-base"
    dino_dim: int = 768
    t5_dim: int = 768
    freeze_encoders: bool = True
    load_encoders: bool = True

    # Image / future-world window: last frame = action-horizon end (world head).
    # Observation history is ``observation_horizon`` frames ending at t=0; RLDS
    # concatenates those with the future frame(s) from ``video_num_frames``.
    image_resolution: tuple[int, int] = (224, 224)
    video_num_frames: int = 2
    action_video_freq_ratio: int = 50  # with horizon=50 → future frame at t=50
    # Official HPT observation_horizon: T past+current frames compressed in each stem.
    observation_horizon: int = 4
    # Official stem_spec.random_horizon_masking: train with a random 1..T suffix.
    random_horizon_masking: bool = True

    # Training regime.
    train_mode: Literal["pretrain", "finetune"] = "pretrain"
    # When True, ego samples do not update robot/wrist stems and vice versa.
    domain_stem_grad_gate: bool = True

    loss: dict[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "lambda_ego_world": 1.0,
            "lambda_ego_action": 0.5,
            "lambda_robot_world": 0.5,
            "lambda_robot_action": 1.0,
            # Smoothness on implied actions â=noise-v vs GT consecutive diffs (norm space).
            "lambda_action_smooth": 0.1,
        }
    )

    dtype: str = "bfloat16"
    device: str = "cuda"
    num_inference_steps: int = 50

    # Optional local / HF trunk init (``trunk.pth`` + ``config.yaml`` layout from liruiw/HPT).
    pretrained_trunk_path: str | None = None

    @property
    def history_length(self) -> int:
        """Obs history length (serves HistoryBufferTransform at inference)."""
        return int(self.observation_horizon)

    @property
    def image_time_length(self) -> int:
        """RLDS image time dim: obs history (+ future frames only when world head is on)."""
        if self.head_mode == "action_only":
            return int(self.observation_horizon)
        extra = max(int(self.video_num_frames) - 1, 0)
        return int(self.observation_horizon) + extra

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.HPT

    @override
    def create(self, rng: at.KeyArrayLike) -> "_model.BaseModel":
        raise NotImplementedError(
            "HPT is a PyTorch model. Use HPTConfig.create_pytorch() or scripts/train_hpt.py."
        )

    def create_pytorch(self, *, device: str | None = None):
        from openpi.models_pytorch.hpt_pytorch import HPTPytorch

        return HPTPytorch(self, device=device or self.device)

    @override
    def load_pytorch(self, train_config, weight_path: str):
        import safetensors.torch

        from openpi.models_pytorch.hpt.model import load_hpt_weights

        model = self.create_pytorch(device=getattr(train_config.model, "device", self.device))
        load_hpt_weights(model, weight_path)
        return model

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        t = self.image_time_length
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
                state_history=jax.ShapeDtypeStruct(
                    [batch_size, self.observation_horizon, self.proprio_dim], jnp.float32
                ),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)
        return observation_spec, action_spec
