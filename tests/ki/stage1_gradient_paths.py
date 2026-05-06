"""Stage 1 Test: Gradient path verification (KI-V1).

目标：验证 stop_gradient 机制在 Attention 层的梯度路径是否正确。
不需要任何数据集或 GPU，只用 dummy 模型（几秒内完成）。

运行方式：
    cd pi07_reproduction
    python tests/ki/stage1_gradient_paths.py

通过条件：全部 PASS，无 AssertionError。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import numpy as np

from openpi.models.pi0_config import Pi0Config
from openpi.models.pi0 import Pi0
from openpi.models import model as _model


# ──────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────

def make_model(ki_enabled: bool, ki_insulate: bool) -> Pi0:
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
    action_dim    = model.action_out_proj.out_features
    action_horizon = model.action_horizon
    max_token_len  = model.max_token_len
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


def _key_contains(k, substr: str) -> bool:
    """Check if any component of an NNX flat_state path key contains the substring.

    NNX flat_state keys are tuples of strings, e.g.:
      ('PaliGemma', 'llm', 'variables', 'params', 'layers', 'attn', 'q_einsum_1', 'w')
    so we must check each component individually, not the tuple itself.
    """
    if isinstance(k, str):
        return substr in k
    return any(substr in part for part in k)


def vlm_grad_norm(model: Pi0, obs, actions, loss_key: str) -> float:
    """VLM backbone（path 含 'llm'，任一分量不含 '_1'）对指定 loss 的梯度 L2 范数。"""
    graphdef, params = nnx.split(model)

    def f(p):
        m = nnx.merge(graphdef, p)
        out = m.compute_loss(jax.random.key(1), obs, actions, train=True)
        return jnp.mean(out[loss_key]) if isinstance(out, dict) else jnp.mean(out)

    grads = jax.grad(f)(params)
    leaves = [
        v.value for k, v in grads.flat_state().items()
        if _key_contains(k, "llm") and not _key_contains(k, "_1")
    ]
    return float(sum(jnp.sum(jnp.square(g)) for g in leaves) ** 0.5) if leaves else 0.0


def action_grad_norm(model: Pi0, obs, actions, loss_key: str) -> float:
    """Action expert（path 任一分量含 '_1'）对指定 loss 的梯度 L2 范数。"""
    graphdef, params = nnx.split(model)

    def f(p):
        m = nnx.merge(graphdef, p)
        out = m.compute_loss(jax.random.key(1), obs, actions, train=True)
        return jnp.mean(out[loss_key]) if isinstance(out, dict) else jnp.mean(out)

    grads = jax.grad(f)(params)
    leaves = [
        v.value for k, v in grads.flat_state().items()
        if _key_contains(k, "_1")
    ]
    return float(sum(jnp.sum(jnp.square(g)) for g in leaves) ** 0.5) if leaves else 0.0


# ──────────────────────────────────────────────
# 测试函数
# ──────────────────────────────────────────────

def test_v1_1():
    """V1.1: flow_loss 在 ki_insulate=True 时 VLM 梯度为 0。"""
    print("  [V1.1] flow_loss -> VLM grad == 0 ...")
    m = make_model(ki_enabled=True, ki_insulate=True)
    obs, acts = make_toy_batch(m)
    norm = vlm_grad_norm(m, obs, acts, "flow")
    assert norm < 1e-5, f"FAIL: VLM grad from flow_loss = {norm:.2e}, expected ~0"
    print(f"        PASS (norm={norm:.2e})")


def test_v1_2():
    """V1.2: ki_fast loss 有非零 VLM 梯度。"""
    print("  [V1.2] ki_fast loss -> VLM grad != 0 ...")
    m = make_model(ki_enabled=True, ki_insulate=True)
    obs, acts = make_toy_batch(m)
    norm = vlm_grad_norm(m, obs, acts, "ki_fast")
    assert norm > 1e-7, f"FAIL: VLM grad from ki_fast = {norm:.2e}, expected > 0"
    print(f"        PASS (norm={norm:.2e})")


def test_v1_3():
    """V1.3: flow_loss 正常更新 action expert。"""
    print("  [V1.3] flow_loss -> action grad != 0 ...")
    m = make_model(ki_enabled=True, ki_insulate=True)
    obs, acts = make_toy_batch(m)
    norm = action_grad_norm(m, obs, acts, "flow")
    assert norm > 1e-7, f"FAIL: action grad from flow_loss = {norm:.2e}, expected > 0"
    print(f"        PASS (norm={norm:.2e})")


def test_v1_4():
    """V1.4: ki_fast loss 不污染 action expert。"""
    print("  [V1.4] ki_fast loss -> action grad == 0 ...")
    m = make_model(ki_enabled=True, ki_insulate=True)
    obs, acts = make_toy_batch(m)
    norm = action_grad_norm(m, obs, acts, "ki_fast")
    assert norm < 1e-5, f"FAIL: action grad from ki_fast = {norm:.2e}, expected ~0"
    print(f"        PASS (norm={norm:.2e})")


def test_v1_5():
    """V1.5: stop_gradient 不影响前向传播——KI 开启时 flow loss 与基线数值完全一致。

    比较对象：
      m_ki   = ki_enabled=True,  ki_insulate=True   （KI 全开）
      m_base = ki_enabled=False, ki_insulate=False   （原始基线）

    两者用相同 RNG 初始化（参数相同），验证前向结果一致。
    这才是真正的回归保护：证明 KI 代码路径不破坏前向数值。
    """
    print("  [V1.5] KI forward (flow component) == baseline forward ...")
    m_ki   = make_model(ki_enabled=True,  ki_insulate=True)
    m_base = make_model(ki_enabled=False, ki_insulate=False)
    # 两者都用 jax.random.key(0) 初始化，参数相同

    obs, acts = make_toy_batch(m_ki)

    out_ki   = m_ki.compute_loss(jax.random.key(42), obs, acts)    # dict
    out_base = m_base.compute_loss(jax.random.key(42), obs, acts)  # array

    # KI 模型返回字典，取 flow 分量与基线比较
    assert isinstance(out_ki, dict), "KI model should return dict"
    flow_ki   = out_ki["flow"]
    flow_base = out_base  # scalar array

    max_diff = float(jnp.max(jnp.abs(flow_ki - flow_base)))
    assert max_diff < 1e-4, (
        f"FAIL: KI forward (flow) differs from baseline, max_diff={max_diff:.2e}. "
        "stop_gradient should be identity in forward pass."
    )
    print(f"        PASS (max_diff={max_diff:.2e})")


def test_forward_equiv():
    """Extra: ki_insulate 只影响反向，前向必须数值等价。"""
    print("  [Extra] ki_insulate toggle: forward identical ...")
    m_off = make_model(ki_enabled=True, ki_insulate=False)
    m_on  = make_model(ki_enabled=True, ki_insulate=True)
    obs, acts = make_toy_batch(m_off)
    out_off = m_off.compute_loss(jax.random.key(7), obs, acts)
    out_on  = m_on.compute_loss(jax.random.key(7), obs, acts)
    for key in ("flow", "ki_fast"):
        diff = float(jnp.max(jnp.abs(out_off[key] - out_on[key])))
        assert diff < 1e-5, f"FAIL: {key} differs between insulate=off/on: {diff:.2e}"
    print("        PASS")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("\n========== Stage 1: Gradient Path Tests (KI-V1) ==========\n")
    tests = [test_v1_1, test_v1_2, test_v1_3, test_v1_4, test_v1_5, test_forward_equiv]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"        FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"        ERROR: {type(e).__name__}: {e}")
            failed += 1

    print(f"\n========== Results: {passed} passed, {failed} failed ==========\n")
    if failed:
        sys.exit(1)
