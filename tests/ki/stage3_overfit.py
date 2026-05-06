"""Stage 3 Test: Small-data overfit (KI-V2 regression + overfit sanity).

目标：
  - KI-V2：alpha=0, ki_insulate=False 时，loss 曲线与 baseline 完全一致（回归保护）。
  - Overfit：KI 模型在单 batch 上训练 100 步，flow_loss 和 ki_fast_loss 都应下降。

运行方式：
    cd pi07_reproduction
    python tests/ki/stage3_overfit.py

预期时间：< 5 分钟（dummy 模型，CPU）
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax

from openpi.models.pi0_config import Pi0Config
from openpi.models.pi0 import Pi0
from openpi.models import model as _model


# ──────────────────────────────────────────────

def make_model(ki_enabled: bool, ki_insulate: bool, ki_alpha: float = 1.0) -> Pi0:
    cfg = Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        ki_enabled=ki_enabled,
        ki_insulate=ki_insulate,
        ki_alpha=ki_alpha,
    )
    return Pi0(cfg, nnx.Rngs(jax.random.key(0)))


def make_toy_batch(model: Pi0, B: int = 2):
    action_dim      = model.action_out_proj.out_features
    action_horizon  = model.action_horizon
    max_token_len   = model.max_token_len
    ki_fast_max_len = model.ki_fast_max_len

    img      = jax.random.normal(jax.random.key(99), (B, 224, 224, 3)) * 0.1
    img_mask = jnp.ones((B,), dtype=jnp.bool_)

    with _model.at.disable_typechecking():
        obs = _model.Observation(
            images={k: img for k in _model.IMAGE_KEYS},
            image_masks={k: img_mask for k in _model.IMAGE_KEYS},
            state=jax.random.normal(jax.random.key(1), (B, action_dim)) * 0.1,
            tokenized_prompt=jnp.zeros((B, max_token_len), dtype=jnp.int32),
            tokenized_prompt_mask=jnp.ones((B, max_token_len), dtype=jnp.bool_),
            ki_fast_tokens=jnp.zeros((B, ki_fast_max_len), dtype=jnp.int32),
            ki_fast_mask=jnp.ones((B, ki_fast_max_len), dtype=jnp.bool_),
            token_ar_mask=jnp.zeros((B, ki_fast_max_len), dtype=jnp.int32),
            token_loss_mask=jnp.ones((B, ki_fast_max_len), dtype=jnp.bool_),
        )
    actions = jax.random.normal(jax.random.key(2), (B, action_horizon, action_dim)) * 0.1
    return obs, actions


def train_loop(model: Pi0, obs, actions, steps: int = 100, lr: float = 1e-3):
    """Simple AdamW training loop; returns (loss_history, flow_history, ki_history)."""
    optimizer = optax.adam(lr)
    graphdef, params = nnx.split(model)
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state, rng):
        def loss_fn(p):
            m = nnx.merge(graphdef, p)
            out = m.compute_loss(rng, obs, actions, train=True)
            if isinstance(out, dict):
                ki_alpha = getattr(m, "ki_alpha", 1.0)
                total = jnp.mean(out["flow"]) + ki_alpha * jnp.mean(out["ki_fast"])
                return total, (jnp.mean(out["flow"]), jnp.mean(out["ki_fast"]))
            return jnp.mean(out), (jnp.mean(out), jnp.zeros(()))

        (loss, (flow, ki)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_opt = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt, loss, flow, ki

    loss_hist, flow_hist, ki_hist = [], [], []
    for i in range(steps):
        rng = jax.random.fold_in(jax.random.key(42), i)
        params, opt_state, loss, flow, ki = step(params, opt_state, rng)
        loss_hist.append(float(loss))
        flow_hist.append(float(flow))
        ki_hist.append(float(ki))
        if (i + 1) % 20 == 0:
            print(f"    step {i+1:3d}: total={loss:.4f}  flow={flow:.4f}  ki={ki:.4f}")

    return loss_hist, flow_hist, ki_hist


# ──────────────────────────────────────────────

def test_v2_alpha0_regression():
    """KI-V2: alpha=0, ki_insulate=False -> loss identical to baseline (no regression)."""
    print("  [V2] alpha=0, ki_insulate=False: loss must match baseline ...")
    m_ki   = make_model(ki_enabled=True,  ki_insulate=False, ki_alpha=0.0)
    m_base = make_model(ki_enabled=False, ki_insulate=False)
    obs, acts = make_toy_batch(m_ki)

    rng = jax.random.key(42)
    out_ki   = m_ki.compute_loss(rng, obs, acts)
    out_base = m_base.compute_loss(rng, obs, acts)

    # ki mode returns dict; extract flow component.
    loss_ki   = jnp.mean(out_ki["flow"]) if isinstance(out_ki, dict) else jnp.mean(out_ki)
    loss_base = jnp.mean(out_base)
    diff = float(jnp.abs(loss_ki - loss_base))
    assert diff < 1e-4, f"FAIL: regression detected, diff={diff:.2e}"
    print(f"        PASS  diff={diff:.2e}")


def test_overfit_ki():
    """KI overfit: after 100 gradient steps, both flow_loss and ki_fast_loss decrease."""
    print("  [Overfit] Training KI model for 100 steps ...")
    m = make_model(ki_enabled=True, ki_insulate=True)
    obs, acts = make_toy_batch(m)

    loss_hist, flow_hist, ki_hist = train_loop(m, obs, acts, steps=100)

    # Check that losses at end are lower than at start.
    window = 10
    flow_start = sum(flow_hist[:window]) / window
    flow_end   = sum(flow_hist[-window:]) / window
    ki_start   = sum(ki_hist[:window]) / window
    ki_end     = sum(ki_hist[-window:]) / window

    print(f"        flow_loss:  {flow_start:.4f} -> {flow_end:.4f}")
    print(f"        ki_fast:    {ki_start:.4f} -> {ki_end:.4f}")

    assert flow_end < flow_start, f"FAIL: flow_loss did not decrease ({flow_start:.4f} -> {flow_end:.4f})"
    assert ki_end < ki_start,     f"FAIL: ki_fast loss did not decrease ({ki_start:.4f} -> {ki_end:.4f})"
    print("        PASS")


def test_overfit_baseline():
    """Baseline overfit: original pi05 model also converges (sanity)."""
    print("  [Overfit baseline] Training baseline model for 100 steps ...")
    m = make_model(ki_enabled=False, ki_insulate=False)
    obs, acts = make_toy_batch(m)

    loss_hist, flow_hist, _ = train_loop(m, obs, acts, steps=100)

    window = 10
    start = sum(flow_hist[:window]) / window
    end   = sum(flow_hist[-window:]) / window
    print(f"        flow_loss:  {start:.4f} -> {end:.4f}")
    assert end < start, f"FAIL: baseline loss did not decrease ({start:.4f} -> {end:.4f})"
    print("        PASS")


# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("\n========== Stage 3: Overfit + KI-V2 Regression Tests ==========\n")
    tests = [test_v2_alpha0_regression, test_overfit_ki, test_overfit_baseline]
    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"        FAIL: {e}")
            failed += 1
        except Exception as e:
            import traceback
            print(f"        ERROR: {type(e).__name__}: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n========== Results: {passed} passed, {failed} failed ==========\n")
    if failed:
        sys.exit(1)
