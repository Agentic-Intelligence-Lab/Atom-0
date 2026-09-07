"""High-level policy π_HL for MEM long-term memory.

π_HL(l_{t+1}, m_{t+1} | o_t, m_t, g): jointly generates the next subtask string and the
updated language memory via text next-token cross-entropy. This is the faithful home of
MEM's long-term memory (see the MEM paper, Torne et al., π0.6-MEM, Section III-A/B). The
action-generating low-level policy (`Pi0`) consumes only the subtask, never the memory.

The memory CE and greedy generation logic here are moved (and extended from memory-only to
subtask+memory) out of `pi0.py`'s `compute_memory_summary_loss` / `generate_memory_summary_tokens`.
"""

import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_high_level_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Same prefix-LM attention construction as pi0.py (see big_vision)."""
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


class Pi0HL(_model.BaseModel):
    def __init__(self, config: pi0_high_level_config.Pi0HLConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # Dual-config Gemma so the `[tokens, None]` calling convention from pi0.py transfers
        # verbatim; the second (action-expert) stream is never fed tokens. No adaRMS / KI:
        # π_HL is a pure text policy.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=False,
                ki_insulate=False,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=config.history_length == 1,
                dtype_mm=config.dtype,
                history_length=config.history_length,
                temporal_attention_every_n_layers=config.temporal_attention_every_n_layers,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        if config.history_length > 1 and config.mem_include_state_history:
            self.state_memory_proj = nnx.Linear(config.action_dim, paligemma_config.width, rngs=rngs)

        self.history_length = config.history_length
        self.mem_include_state_history = config.mem_include_state_history
        self.memory_summary_max_len = config.memory_summary_max_len
        self.memory_generation_max_new_tokens = config.memory_generation_max_new_tokens

        # Set by model.train() / model.eval().
        self.deterministic = True

    def _embed_state_history(self, obs: _model.Observation):
        """Project proprioceptive state history into one prefix token per timestep."""
        if self.history_length == 1 or not self.mem_include_state_history:
            return None, None
        if obs.state_history is None:
            return None, None
        state_tokens = self.state_memory_proj(obs.state_history)
        state_mask = jnp.ones(state_tokens.shape[:2], dtype=jnp.bool_)
        return state_tokens, state_mask

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        """Context for generation: images (+optional state history) + "Task: {g}\nMemory: {m_t}"."""
        input_mask = []
        ar_mask = []
        tokens = []
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
            tokens.append(image_tokens)
            image_mask = obs.image_masks[name]
            if image_mask.ndim == 2:
                # The video encoder compresses history into the current-timestep tokens.
                image_mask = image_mask[:, -1]
            input_mask.append(einops.repeat(image_mask, "b -> b s", s=image_tokens.shape[1]))
            ar_mask += [False] * image_tokens.shape[1]

        state_memory_tokens, state_memory_mask = self._embed_state_history(obs)
        if state_memory_tokens is not None:
            tokens.append(state_memory_tokens)
            input_mask.append(state_memory_mask)
            ar_mask += [False] * state_memory_tokens.shape[1]

        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            ar_mask += [False] * tokenized_inputs.shape[1]

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, " b"]:
        """Teacher-forced next-token CE over the joint (subtask + new memory) target.

        `actions` is ignored: π_HL emits no actions.
        """
        if observation.memory_summary_tokens is None or observation.memory_summary_loss_mask is None:
            raise ValueError("Pi0HL.compute_loss requires memory_summary_* target fields.")
        observation = _model.preprocess_observation(rng, observation, train=train)

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        target_tokens_emb = self.PaliGemma.llm(observation.memory_summary_tokens, method="embed")

        prefix_ar_mask = jnp.broadcast_to(prefix_ar_mask, prefix_mask.shape)
        target_ar_mask = observation.memory_summary_ar_mask
        if target_ar_mask.ndim == 1:
            target_ar_mask = jnp.broadcast_to(target_ar_mask, observation.memory_summary_mask.shape)

        tokens = jnp.concatenate([prefix_tokens, target_tokens_emb], axis=1)
        input_mask = jnp.concatenate([prefix_mask, observation.memory_summary_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, target_ar_mask], axis=1)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1

        (out, _), _ = self.PaliGemma.llm([tokens, None], mask=attn_mask, positions=positions)
        logits = self.PaliGemma.llm(out[:, :-1], method="decode_logits").astype(jnp.float32)

        target_ids = jnp.concatenate(
            [
                jnp.zeros((observation.memory_summary_tokens.shape[0], prefix_tokens.shape[1]), dtype=jnp.int32),
                observation.memory_summary_tokens,
            ],
            axis=1,
        )
        target_loss_mask = jnp.concatenate(
            [
                jnp.zeros(prefix_mask.shape, dtype=jnp.bool_),
                observation.memory_summary_loss_mask,
            ],
            axis=1,
        )
        shifted_targets = target_ids[:, 1:]
        shifted_loss_mask = target_loss_mask[:, 1:]
        logp = jax.nn.log_softmax(logits, axis=-1)
        token_logp = jnp.take_along_axis(logp, shifted_targets[..., None], axis=-1)[..., 0]
        return -jnp.sum(token_logp * shifted_loss_mask, axis=-1) / jnp.clip(jnp.sum(shifted_loss_mask, axis=-1), 1)

    @override
    def sample_actions(self, rng: at.KeyArrayLike, observation: _model.Observation, **kwargs) -> _model.Actions:
        raise NotImplementedError("Pi0HL is a high-level text policy; use generate() for (subtask, memory).")

    def generate(
        self,
        observation: _model.Observation,
        *,
        max_new_tokens: int | None = None,
    ) -> at.Int[at.Array, "b m"]:
        """Greedy-decode the joint "Next subtask: ... New memory: ..." token sequence.

        Callers (HLInferenceServer) decode and split on "New memory:" to recover (l, m).
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        max_new_tokens = max_new_tokens or self.memory_generation_max_new_tokens
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1

        (prefix_out, _), kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
        )
        last_logit = self.PaliGemma.llm(prefix_out[:, -1:], method="decode_logits").astype(jnp.float32)
        batch_size = prefix_tokens.shape[0]
        output_tokens = jnp.zeros((batch_size, max_new_tokens), dtype=jnp.int32)
        eos_token = jnp.asarray(1, dtype=jnp.int32)
        done = jnp.zeros((batch_size,), dtype=jnp.bool_)

        # Non-jitted so the KV cache grows one token at a time.
        for step_idx in range(max_new_tokens):
            token = jnp.argmax(last_logit, axis=-1).astype(jnp.int32)
            token = jnp.where(done[:, None], eos_token, token)
            output_tokens = output_tokens.at[:, step_idx].set(token[:, 0])
            done = jnp.logical_or(done, token[:, 0] == eos_token)
            if step_idx == max_new_tokens - 1:
                break
            token_embedding = self.PaliGemma.llm(token, method="embed")
            query_position = jnp.sum(prefix_mask, axis=-1)[:, None] + step_idx
            cache_len = prefix_tokens.shape[1] + step_idx + 1
            mask = jnp.ones((batch_size, 1, cache_len), dtype=jnp.bool_)
            (out, _), kv_cache = self.PaliGemma.llm(
                [token_embedding, None],
                mask=mask,
                positions=query_position,
                kv_cache=kv_cache,
            )
            last_logit = self.PaliGemma.llm(out[:, -1:], method="decode_logits").astype(jnp.float32)
        return output_tokens
