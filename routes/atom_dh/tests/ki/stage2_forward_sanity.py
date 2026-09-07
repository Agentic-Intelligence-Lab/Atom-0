"""Stage 2 Test: Forward pass sanity checks.

目标：
  - KI 模型能正常完成前向传播，不报错。
  - compute_loss 在 ki_enabled=True 时返回字典 {"flow": ..., "ki_fast": ...}。
  - compute_loss 在 ki_enabled=False 时返回标量（与原始 openpi 一致）。
  - sample_actions 路径（推理）在 KI 模型上完全不变（KI 推理 no-op）。
  - ki_insulate 切换不影响 compute_loss 返回值的形状/类型。

运行方式：
    cd Atom-0
    python tests/ki/stage2_forward_sanity.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

import jax
import jax.numpy as jnp
import flax.nnx as nnx

from openpi.models.pi0_config import Pi0Config
from openpi.models.pi0 import Pi0
from openpi.models import model as _model


def make_model(ki_enabled: bool, ki_insulate: bool = True) -> Pi0:
    cfg = Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        ki_enabled=ki_enabled,
        ki_insulate=ki_insulate,
        ki_alpha=1.0,
    )
    return Pi0(cfg, nnx.Rngs(jax.random.key(0)))


def make_toy_batch(model: Pi0, B: int = 2):
    action_dim      = model.action_out_proj.out_features
    action_horizon  = model.action_horizon
    max_token_len   = model.max_token_len
    ki_fast_max_len = model.ki_fast_max_len

    img      = jnp.zeros((B, 224, 224, 3))
    img_mask = jnp.ones((B,), dtype=jnp.bool_)

    with _model.at.disable_typechecking():
        obs = _model.Observation(
            images={k: img for k in _model.IMAGE_KEYS},
            image_masks={k: img_mask for k in _model.IMAGE_KEYS},
            state=jnp.zeros((B, action_dim)),
            tokenized_prompt=jnp.zeros((B, max_token_len), dtype=jnp.int32),
            tokenized_prompt_mask=jnp.ones((B, max_token_len), dtype=jnp.bool_),
            ki_fast_tokens=jnp.zeros((B, ki_fast_max_len), dtype=jnp.int32),
            ki_fast_mask=jnp.ones((B, ki_fast_max_len), dtype=jnp.bool_),
            token_ar_mask=jnp.zeros((B, ki_fast_max_len), dtype=jnp.int32),
            token_loss_mask=jnp.ones((B, ki_fast_max_len), dtype=jnp.bool_),
        )
    actions = jnp.zeros((B, action_horizon, action_dim))
    return obs, actions


# ──────────────────────────────────────────────

def test_ki_compute_loss_returns_dict():
    print("  [S2.1] ki_enabled=True -> compute_loss returns dict ...")
    m = make_model(ki_enabled=True)
    obs, acts = make_toy_batch(m)
    out = m.compute_loss(jax.random.key(0), obs, acts)
    assert isinstance(out, dict), f"Expected dict, got {type(out)}"
    assert "flow" in out and "ki_fast" in out, f"Missing keys: {list(out.keys())}"
    # flow_loss shape: (B, action_horizon) — same as original openpi compute_loss return
    assert out["flow"].ndim == 2 and out["flow"].shape[0] == 2, f"Wrong flow shape: {out['flow'].shape}"
    # ki_fast_loss shape: (B,) — averaged over tokens per sample
    assert out["ki_fast"].shape == (2,), f"Wrong ki_fast shape: {out['ki_fast'].shape}"
    print(f"        PASS  flow={float(jnp.mean(out['flow'])):.4f}  ki_fast={float(jnp.mean(out['ki_fast'])):.4f}")


def test_no_ki_compute_loss_returns_array():
    print("  [S2.2] ki_enabled=False -> compute_loss returns scalar array ...")
    m = make_model(ki_enabled=False)
    obs, acts = make_toy_batch(m)
    out = m.compute_loss(jax.random.key(0), obs, acts)
    assert not isinstance(out, dict), f"Expected array, got dict"
    assert hasattr(out, "shape"), f"Expected jax array"
    print(f"        PASS  loss shape={out.shape}  mean={float(jnp.mean(out)):.4f}")


def test_sample_actions_unchanged():
    """推理路径（sample_actions）在 KI 模型上正常工作，KI is no-op during inference。"""
    print("  [S2.3] sample_actions unchanged for KI model ...")
    m_ki   = make_model(ki_enabled=True)
    m_base = make_model(ki_enabled=False)
    obs, _ = make_toy_batch(m_ki, B=1)

    acts_ki   = m_ki.sample_actions(jax.random.key(5), obs, num_steps=2)
    acts_base = m_base.sample_actions(jax.random.key(5), obs, num_steps=2)

    assert acts_ki.shape == acts_base.shape, f"Shape mismatch: {acts_ki.shape} vs {acts_base.shape}"
    # Values won't match because models have different params, but shapes must match.
    print(f"        PASS  actions shape={acts_ki.shape}")


def test_ki_insulate_toggle_same_output_type():
    print("  [S2.4] ki_insulate toggle does not change output type ...")
    m_off = make_model(ki_enabled=True, ki_insulate=False)
    m_on  = make_model(ki_enabled=True, ki_insulate=True)
    obs, acts = make_toy_batch(m_off)

    out_off = m_off.compute_loss(jax.random.key(0), obs, acts)
    out_on  = m_on.compute_loss(jax.random.key(0), obs, acts)

    assert type(out_off) == type(out_on), "Output type differs between ki_insulate=False/True"
    assert set(out_off.keys()) == set(out_on.keys()), "Output keys differ"
    print("        PASS")


def test_losses_finite():
    print("  [S2.5] All loss values are finite ...")
    m = make_model(ki_enabled=True)
    obs, acts = make_toy_batch(m)
    out = m.compute_loss(jax.random.key(0), obs, acts)
    for k, v in out.items():
        assert jnp.all(jnp.isfinite(v)), f"{k} loss contains non-finite values: {v}"
    print("        PASS")


# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("\n========== Stage 2: Forward Pass Sanity Tests ==========\n")
    tests = [
        test_ki_compute_loss_returns_dict,
        test_no_ki_compute_loss_returns_array,
        test_sample_actions_unchanged,
        test_ki_insulate_toggle_same_output_type,
        test_losses_finite,
    ]
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
