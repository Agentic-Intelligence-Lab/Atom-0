import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    pytorch_compile_mode: str | None = "max-autotune"

    # KI (Knowledge Insulation) settings. KI is a training-only mechanism; inference path is unchanged.
    ki_enabled: bool = False
    ki_alpha: float = 1.0       # weight of the FAST auxiliary CE loss
    ki_insulate: bool = True    # enable stop_gradient; can be disabled independently for ablation
    ki_fast_max_len: int = 256  # length of FAST token sequence stored in ki_fast_tokens

    # MEM (Multi-Scale Embodied Memory) short-term observation memory.
    # history_length=1 preserves the original single-observation pi0/pi0.5 behavior.
    history_length: int = 1
    history_stride_seconds: float = 1.0
    temporal_attention_every_n_layers: int = 4
    mem_include_state_history: bool = True

    # MEM long-term language memory.
    long_memory_enabled: bool = False
    long_memory_loss_weight: float = 0.2
    memory_summary_max_len: int = 96
    memory_generation_max_new_tokens: int = 64
    memory_update_interval_steps: int = 30

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.long_memory_enabled and self.max_token_len < 384:
            object.__setattr__(self, "max_token_len", 384)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.history_length < 1:
            raise ValueError("history_length must be >= 1")
        if self.history_stride_seconds <= 0:
            raise ValueError("history_stride_seconds must be > 0")
        if self.temporal_attention_every_n_layers < 1:
            raise ValueError("temporal_attention_every_n_layers must be >= 1")
        if self.long_memory_loss_weight < 0:
            raise ValueError("long_memory_loss_weight must be >= 0")
        if self.memory_summary_max_len < 2:
            raise ValueError("memory_summary_max_len must be >= 2")
        if self.memory_generation_max_new_tokens < 1:
            raise ValueError("memory_generation_max_new_tokens must be >= 1")
        if self.memory_update_interval_steps < 1:
            raise ValueError("memory_update_interval_steps must be >= 1")
        if self.pytorch_compile_mode is not None:
            assert self.pytorch_compile_mode in [
                "default",
                "reduce-overhead",
                "max-autotune",
                "max-autotune-no-cudagraphs",
            ]

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        if self.history_length == 1:
            image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
            image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)
            state_history_spec = None
        else:
            image_spec = jax.ShapeDtypeStruct(
                [batch_size, self.history_length, *_model.IMAGE_RESOLUTION, 3], jnp.float32
            )
            image_mask_spec = jax.ShapeDtypeStruct([batch_size, self.history_length], jnp.bool_)
            state_history_spec = jax.ShapeDtypeStruct([batch_size, self.history_length, self.action_dim], jnp.float32)

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
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                state_history=state_history_spec,
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                ki_fast_tokens=jax.ShapeDtypeStruct([batch_size, self.ki_fast_max_len], jnp.int32) if self.ki_enabled else None,
                ki_fast_mask=jax.ShapeDtypeStruct([batch_size, self.ki_fast_max_len], bool) if self.ki_enabled else None,
                token_loss_mask=jax.ShapeDtypeStruct([batch_size, self.ki_fast_max_len], bool) if self.ki_enabled else None,
                memory_summary_tokens=jax.ShapeDtypeStruct([batch_size, self.memory_summary_max_len], jnp.int32)
                if self.long_memory_enabled
                else None,
                memory_summary_mask=jax.ShapeDtypeStruct([batch_size, self.memory_summary_max_len], bool)
                if self.long_memory_enabled
                else None,
                memory_summary_ar_mask=jax.ShapeDtypeStruct([batch_size, self.memory_summary_max_len], bool)
                if self.long_memory_enabled
                else None,
                memory_summary_loss_mask=jax.ShapeDtypeStruct([batch_size, self.memory_summary_max_len], bool)
                if self.long_memory_enabled
                else None,
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)
