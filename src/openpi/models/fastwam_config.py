"""Atom-0 / openpi config for the FastWAM (World-Action Model) algorithm."""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
from typing_extensions import override

import openpi.models.model as _model
import openpi.shared.array_typing as at


def robot_wrist_slot_hw(image_resolution: tuple[int, int]) -> dict[str, tuple[int, int]]:
    """Per-camera (H, W) for ``robot_wrist`` layout that stacks to ``image_resolution``.

    Layout (H×W composed)::

        head (2H/3 × W)
        ─────────────────
        left (H/3 × W/2) | right (H/3 × W/2)

    Used by RLDS decode-time resize (before batch, to cut CPU RAM) and by
    ``fastwam_pytorch._compose_robot_wrist_video``.
    """
    h, w = image_resolution
    if h % 3 != 0 or w % 2 != 0:
        raise ValueError(
            f"robot_wrist image_resolution must have H%3==0 and W%2==0, got {image_resolution}"
        )
    return {
        "base_0_rgb": (h * 2 // 3, w),
        "left_wrist_0_rgb": (h // 3, w // 2),
        "right_wrist_0_rgb": (h // 3, w // 2),
    }


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
    # "horizontal" | "vertical" | "none" | "robot_wrist" (head over left|right wrists -> image_resolution)
    concat_multi_camera: str = "horizontal"
    camera_keys: tuple[str, ...] = ("base_0_rgb", "left_wrist_0_rgb")

    model_id: str = "Wan-AI/Wan2.2-TI2V-5B"
    tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B"
    load_text_encoder: bool = True
    redirect_common_files: bool = False  # use Wan2.2 .pth on PFS; DiffSynth mirror unavailable
    mot_checkpoint_mixed_attn: bool = True
    # MoT cross-modal mask (see ``FastWAM._build_mot_attention_mask``):
    # - action always attends to video (first frame by default; set "full" for Joint-style).
    # - video→action was off upstream; enable so the world model can condition on actions.
    mot_video_attends_to_action: bool = True
    mot_action_attends_to_video: str = "first_frame"  # "first_frame" | "full"
    # When video attends to action: temporally aligned groups (skip clean first frame),
    # or dense ``full``. Falls back to full if seq lens are not divisible.
    mot_video_to_action_mode: str = "group_diagonal"  # "group_diagonal" | "causal" | "full"
    skip_dit_load_from_pretrain: bool = False
    skip_vae_load_from_pretrain: bool = False
    # Match upstream FastWAM: load Video-DiT→ActionDiT linear-interp backbone;
    # ``action_encoder`` / ``head`` stay randomly initialized.
    # Generate with: ``scripts/preprocess_action_dit_backbone.py``.
    action_dit_pretrained_path: str | None = (
        "checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"
    )

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
