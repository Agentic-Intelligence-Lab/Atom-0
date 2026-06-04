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
    from openpi.models.pi0_high_level import Pi0HL


@dataclasses.dataclass(frozen=True)
class Pi0HLConfig(_model.BaseModelConfig):
    """High-level policy π_HL(l_{t+1}, m_{t+1} | o_t, m_t, g).

    Faithful to the MEM paper (Torne et al., π0.6-MEM): the high-level policy jointly
    generates the next subtask string l_{t+1} AND the updated language memory m_{t+1}
    via text next-token cross-entropy. It has NO action expert and produces NO actions.
    The low-level action policy (Pi0) consumes only the subtask l_{t+1}, never the memory.

    See reproduction_plan/π0.7 第五步：High-Level Policy复现方案.md and the MEM migration plan.
    """

    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    # An action-expert config is still constructed so we can reuse the exact dual-stream
    # Gemma calling convention from pi0.py (`self.PaliGemma.llm([tokens, None], ...)`). The
    # second stream is never fed tokens, so the action expert contributes no params to the
    # forward graph. TODO(phase-follow-up): slim to a single-config Gemma to drop dead params.
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # action_dim / action_horizon are required by BaseModel and only used to size the optional
    # proprioceptive-history projection; π_HL emits no actions.
    action_dim: int = 32
    action_horizon: int = 50
    # Prefix budget for "Task: {g}\nMemory: {m_t}".
    max_token_len: int = 384

    pytorch_compile_mode: str | None = None

    # Short-term observation memory (shared with the low-level video encoder design).
    # The factorization conditions π_HL on the current observation o_t, so default to a
    # single frame; set >1 to give the high-level policy a short visual history as well.
    history_length: int = 1
    history_stride_seconds: float = 1.0
    temporal_attention_every_n_layers: int = 4
    mem_include_state_history: bool = True

    # Joint subtask + memory target. memory_summary_max_len bounds the teacher-forced target
    # sequence "Next subtask: {l}\nNew memory: {m_next}".
    memory_summary_max_len: int = 128
    memory_generation_max_new_tokens: int = 96
    max_subtask_len: int = 32
    max_history_subtasks: int = 4

    def __post_init__(self):
        if self.history_length < 1:
            raise ValueError("history_length must be >= 1")
        if self.history_stride_seconds <= 0:
            raise ValueError("history_stride_seconds must be > 0")
        if self.temporal_attention_every_n_layers < 1:
            raise ValueError("temporal_attention_every_n_layers must be >= 1")
        if self.memory_summary_max_len < 2:
            raise ValueError("memory_summary_max_len must be >= 2")
        if self.memory_generation_max_new_tokens < 1:
            raise ValueError("memory_generation_max_new_tokens must be >= 1")

    @property
    @override
    def model_type(self) -> _model.ModelType:
        # π_HL reuses the pi05 PaliGemma stack; it is a text policy, but PI05 is the closest tag.
        return _model.ModelType.PI05

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0HL":
        from openpi.models.pi0_high_level import Pi0HL

        return Pi0HL(self, rngs=nnx.Rngs(rng))

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
                # Prefix: "Task: {g}\nMemory: {m_t}".
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                # Teacher-forced joint target: "Next subtask: {l}\nNew memory: {m_next}".
                memory_summary_tokens=jax.ShapeDtypeStruct([batch_size, self.memory_summary_max_len], jnp.int32),
                memory_summary_mask=jax.ShapeDtypeStruct([batch_size, self.memory_summary_max_len], bool),
                memory_summary_ar_mask=jax.ShapeDtypeStruct([batch_size, self.memory_summary_max_len], bool),
                memory_summary_loss_mask=jax.ShapeDtypeStruct([batch_size, self.memory_summary_max_len], bool),
            )
        # action_spec is unused by π_HL but kept for interface symmetry with other configs.
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)
        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Freeze filter: when a LoRA variant is used, train only LoRA params."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        if "lora" in self.paligemma_variant:
            filters.append(gemma_params_filter)
            has_lora = True
        if has_lora:
            filters.append(nnx.Not(nnx_utils.PathRegex(".*lora.*")))
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)
