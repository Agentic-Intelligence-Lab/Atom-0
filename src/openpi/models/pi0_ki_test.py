"""KI-V1: Gradient path unit tests for Knowledge Insulation.

Five assertions that must all pass before any KI training run:
  V1.1  flow_loss  -> VLM grad == 0      (stop_gradient blocks action loss from polluting VLM)
  V1.2  ki_loss   -> VLM grad != 0      (FAST auxiliary loss reaches VLM)
  V1.3  flow_loss  -> action grad != 0  (flow loss still updates action expert)
  V1.4  ki_loss   -> action grad == 0   (auxiliary loss does not pollute action expert)
  V1.5  ki_off     -> forward identical  (backward compatibility regression guard)
"""

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import numpy as np
import pytest

from openpi.models.pi0_config import Pi0Config
from openpi.models.pi0 import Pi0
from openpi.models import model as _model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model(ki_enabled: bool, ki_insulate: bool) -> Pi0:
    cfg = Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        ki_enabled=ki_enabled,
        ki_insulate=ki_insulate,
        ki_alpha=1.0,
    )
    return Pi0(cfg, nnx.Rngs(jax.random.key(0)))


def _toy_obs(model: Pi0, batch_size: int = 2) -> tuple[_model.Observation, _model.Actions]:
    """Build minimal fake observation + actions compatible with the model config."""
    cfg = model.config if hasattr(model, "config") else None
    # Derive shapes from model attributes
    action_dim = model.action_out_proj.out_features
    action_horizon = model.action_horizon
    max_token_len = model.max_token_len
    ki_fast_max_len = model.ki_fast_max_len

    B = batch_size
    H, W = 224, 224
    img = jnp.zeros((B, H, W, 3))
    img_mask = jnp.ones((B,), dtype=jnp.bool_)

    with _model.at.disable_typechecking():
        obs = _model.Observation(
            images={k: img for k in _model.IMAGE_KEYS},
            image_masks={k: img_mask for k in _model.IMAGE_KEYS},
            state=jnp.zeros((B, action_dim)),
            tokenized_prompt=jnp.zeros((B, max_token_len), dtype=jnp.int32),
            tokenized_prompt_mask=jnp.ones((B, max_token_len), dtype=jnp.bool_),
            # KI fields: always provide them; pi0.py checks self.ki_enabled before using.
            ki_fast_tokens=jnp.zeros((B, ki_fast_max_len), dtype=jnp.int32),
            ki_fast_mask=jnp.ones((B, ki_fast_max_len), dtype=jnp.bool_),
            token_ar_mask=jnp.zeros((B, ki_fast_max_len), dtype=jnp.int32),
            token_loss_mask=jnp.ones((B, ki_fast_max_len), dtype=jnp.bool_),
        )
    actions = jnp.zeros((B, action_horizon, action_dim))
    return obs, actions


def _key_contains(k, substr: str) -> bool:
    """Check if any component of an NNX flat_state path key contains substr.

    NNX flat_state keys are tuples like ('PaliGemma', 'llm', ..., 'q_einsum_1', 'w'),
    so we must check each component, not the tuple itself.
    """
    if isinstance(k, str):
        return substr in k
    return any(substr in part for part in k)


def _param_grad_norm(model: Pi0, obs, actions, loss_key: str | None) -> float:
    """Gradient norm for VLM params (path has 'llm', no component has '_1')."""
    graphdef, params = nnx.split(model)

    def scalar_loss(p):
        m = nnx.merge(graphdef, p)
        out = m.compute_loss(jax.random.key(1), obs, actions, train=True)
        if isinstance(out, dict):
            return jnp.mean(out[loss_key])
        return jnp.mean(out)

    grads = jax.grad(scalar_loss)(params)
    flat_grads = grads.flat_state()
    selected = [
        v.value for k, v in flat_grads.items()
        if _key_contains(k, "llm") and not _key_contains(k, "_1")
    ]
    if not selected:
        return 0.0
    return float(sum(jnp.sum(jnp.square(g)) for g in selected) ** 0.5)


def _action_grad_norm(model: Pi0, obs, actions, loss_key: str | None) -> float:
    """Gradient norm for action expert params (any path component contains '_1')."""
    graphdef, params = nnx.split(model)

    def scalar_loss(p):
        m = nnx.merge(graphdef, p)
        out = m.compute_loss(jax.random.key(1), obs, actions, train=True)
        if isinstance(out, dict):
            return jnp.mean(out[loss_key])
        return jnp.mean(out)

    grads = jax.grad(scalar_loss)(params)
    flat_grads = grads.flat_state()
    selected = [
        v.value for k, v in flat_grads.items()
        if _key_contains(k, "_1")
    ]
    if not selected:
        return 0.0
    return float(sum(jnp.sum(jnp.square(g)) for g in selected) ** 0.5)


# ---------------------------------------------------------------------------
# KI-V1 Tests
# ---------------------------------------------------------------------------

class TestKIGradientPaths:
    """KI-V1: verify that stop_gradient creates the correct gradient topology."""

    def test_v1_1_flow_loss_zero_vlm_grad(self):
        """V1.1: flow_loss should produce zero VLM gradient when ki_insulate=True."""
        model = _make_model(ki_enabled=True, ki_insulate=True)
        obs, actions = _toy_obs(model)
        norm = _param_grad_norm(model, obs, actions, "flow")
        assert norm < 1e-5, (
            f"VLM gradient from flow_loss should be ~0 with ki_insulate=True, got {norm:.2e}"
        )

    def test_v1_2_ki_loss_nonzero_vlm_grad(self):
        """V1.2: ki_fast loss should produce non-zero VLM gradient."""
        model = _make_model(ki_enabled=True, ki_insulate=True)
        obs, actions = _toy_obs(model)
        norm = _param_grad_norm(model, obs, actions, "ki_fast")
        assert norm > 1e-7, (
            f"VLM gradient from ki_fast loss should be non-zero, got {norm:.2e}"
        )

    def test_v1_3_flow_loss_nonzero_action_grad(self):
        """V1.3: flow_loss should still update action expert parameters."""
        model = _make_model(ki_enabled=True, ki_insulate=True)
        obs, actions = _toy_obs(model)
        norm = _action_grad_norm(model, obs, actions, "flow")
        assert norm > 1e-7, (
            f"Action expert gradient from flow_loss should be non-zero, got {norm:.2e}"
        )

    def test_v1_4_ki_loss_zero_action_grad(self):
        """V1.4: ki_fast loss should NOT update action expert parameters."""
        model = _make_model(ki_enabled=True, ki_insulate=True)
        obs, actions = _toy_obs(model)
        norm = _action_grad_norm(model, obs, actions, "ki_fast")
        assert norm < 1e-5, (
            f"Action expert gradient from ki_fast loss should be ~0, got {norm:.2e}"
        )

    def test_v1_5_ki_forward_equals_baseline(self):
        """V1.5: stop_gradient is identity in forward pass.

        KI model (ki_enabled=True, ki_insulate=True) should produce the same flow loss
        as the baseline model (ki_enabled=False) when initialized with the same RNG.
        This is the true regression guard: our KI code must not change forward numerics.
        """
        model_ki   = _make_model(ki_enabled=True,  ki_insulate=True)
        model_base = _make_model(ki_enabled=False, ki_insulate=False)
        # Both use jax.random.key(0) → same initial params.

        obs, actions = _toy_obs(model_ki)

        out_ki   = model_ki.compute_loss(jax.random.key(42), obs, actions)    # dict
        out_base = model_base.compute_loss(jax.random.key(42), obs, actions)  # array

        assert isinstance(out_ki, dict), "KI model must return dict"
        max_diff = jnp.max(jnp.abs(out_ki["flow"] - out_base))
        assert max_diff < 1e-4, (
            f"KI forward (flow) differs from baseline by {max_diff:.2e}. "
            "stop_gradient must be identity in forward pass."
        )

    def test_v1_insulate_false_no_grad_blocking(self):
        """Extra: with ki_enabled=True but ki_insulate=False, VLM receives flow_loss gradient."""
        model = _make_model(ki_enabled=True, ki_insulate=False)
        obs, actions = _toy_obs(model)
        norm = _param_grad_norm(model, obs, actions, "flow")
        # Without stop_gradient, VLM grad from flow_loss is non-zero.
        assert norm > 1e-7, (
            f"With ki_insulate=False, VLM should receive flow_loss gradient, got {norm:.2e}"
        )

    def test_forward_equiv_ki_insulate_toggle(self):
        """Extra: ki_insulate only affects backward; forward output must be identical."""
        m_off = _make_model(ki_enabled=True, ki_insulate=False)
        m_on  = _make_model(ki_enabled=True, ki_insulate=True)

        # Sync params: copy m_off params into m_on.
        _, params_off = nnx.split(m_off)
        graphdef_on, _ = nnx.split(m_on)
        m_on_synced = nnx.merge(graphdef_on, params_off)

        obs, actions = _toy_obs(m_off)
        out_off = m_off.compute_loss(jax.random.key(7), obs, actions)
        out_on  = m_on_synced.compute_loss(jax.random.key(7), obs, actions)

        # Both return dicts in ki_enabled=True mode.
        assert isinstance(out_off, dict) and isinstance(out_on, dict)
        for key in ("flow", "ki_fast"):
            assert jnp.allclose(out_off[key], out_on[key], atol=1e-5), (
                f"Forward {key} loss differs between ki_insulate=False and True. "
                f"Max diff: {jnp.max(jnp.abs(out_off[key] - out_on[key])):.2e}"
            )

    def test_flow_loss_does_not_depend_on_ki_fast_tokens(self):
        """Changing teacher-forced FAST tokens must not change flow loss.

        This catches target leakage from the VLM-side KI action tokens into the
        continuous action expert. The KI tokens should affect only `ki_fast`.
        """
        model = _make_model(ki_enabled=True, ki_insulate=True)
        obs, actions = _toy_obs(model)
        alt_tokens = (obs.ki_fast_tokens + 17) % 100
        obs_alt = obs.replace(ki_fast_tokens=alt_tokens)

        out = model.compute_loss(jax.random.key(9), obs, actions)
        out_alt = model.compute_loss(jax.random.key(9), obs_alt, actions)

        assert isinstance(out, dict) and isinstance(out_alt, dict)
        flow_diff = jnp.max(jnp.abs(out["flow"] - out_alt["flow"]))
        assert flow_diff < 1e-5, f"flow_loss depends on KI FAST tokens; max diff={flow_diff:.2e}"
