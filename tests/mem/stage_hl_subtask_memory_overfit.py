"""Phase 1: Pi0HL forward + single-batch overfit.

目标：
  - 验证新的 high-level policy Pi0HL 能 forward（联合 subtask+memory 目标的 teacher-forced CE）。
  - 验证它能被 optimizer 更新并在单 batch 上把 CE 拟合下降，证明 memory 生成已搬到 π_HL。
  - 不依赖外部数据集 / checkpoint / 网络（用 dummy Gemma/SigLIP + 直接构造 token 数组）。

运行方式：
    cd pi07_reproduction
    python tests/mem/stage_hl_subtask_memory_overfit.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax

from openpi.models import model as _model
from openpi.models.pi0_high_level import Pi0HL
from openpi.models.pi0_high_level_config import Pi0HLConfig


def make_model() -> tuple[Pi0HLConfig, Pi0HL]:
    config = Pi0HLConfig(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        history_length=1,
        memory_summary_max_len=32,
    )
    return config, config.create(jax.random.key(0))


def make_single_batch(config: Pi0HLConfig, batch_size: int = 2):
    image = jax.random.normal(jax.random.key(10), (batch_size, 224, 224, 3)) * 0.1
    image_mask = jnp.ones((batch_size,), dtype=jnp.bool_)

    # Prefix "Task: ...\nMemory: ..." — content irrelevant for the overfit smoke; use small ids.
    prompt = jnp.arange(1, config.max_token_len + 1, dtype=jnp.int32)[None].repeat(batch_size, axis=0) % 100
    prompt_mask = jnp.ones((batch_size, config.max_token_len), dtype=jnp.bool_)

    m = config.memory_summary_max_len
    # Two distinct fixed targets so overfit must actually use the prefix to separate them.
    target = jnp.stack(
        [
            (jnp.arange(m, dtype=jnp.int32) * 3 + 5) % 200,
            (jnp.arange(m, dtype=jnp.int32) * 7 + 11) % 200,
        ]
    )[:batch_size]
    target_mask = jnp.ones((batch_size, m), dtype=jnp.bool_)
    # First 12 tokens are the "real" target; rest is padding (no loss).
    loss_mask = (jnp.arange(m) < 12)[None].repeat(batch_size, axis=0)
    ar_mask = jnp.ones((batch_size, m), dtype=jnp.bool_)

    with _model.at.disable_typechecking():
        obs = _model.Observation(
            images={key: image for key in _model.IMAGE_KEYS},
            image_masks={key: image_mask for key in _model.IMAGE_KEYS},
            state=jnp.zeros((batch_size, config.action_dim), dtype=jnp.float32),
            tokenized_prompt=prompt,
            tokenized_prompt_mask=prompt_mask,
            memory_summary_tokens=target,
            memory_summary_mask=target_mask,
            memory_summary_ar_mask=ar_mask,
            memory_summary_loss_mask=loss_mask,
        )
    actions = jnp.zeros((batch_size, config.action_horizon, config.action_dim), dtype=jnp.float32)
    return obs, actions


def train_single_batch(model: Pi0HL, obs, actions, steps: int = 80, lr: float = 1e-3) -> list[float]:
    optimizer = optax.adam(lr)
    graphdef, params = nnx.split(model)
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state):
        def loss_fn(p):
            m = nnx.merge(graphdef, p)
            return jnp.mean(m.compute_loss(jax.random.key(42), obs, actions, train=True))

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    losses = []
    for i in range(steps):
        params, opt_state, loss = step(params, opt_state)
        losses.append(float(loss))
        if (i + 1) % 20 == 0:
            print(f"    step {i + 1:3d}: loss={float(loss):.6f}")
    return losses


def main():
    config, model = make_model()
    obs, actions = make_single_batch(config)

    # Forward smoke.
    loss0 = model.compute_loss(jax.random.key(1), obs, actions, train=True)
    assert loss0.shape == (obs.memory_summary_tokens.shape[0],), loss0.shape
    print(f"forward OK: per-example CE shape={loss0.shape}, mean={float(jnp.mean(loss0)):.4f}")

    # Generation smoke.
    gen = model.generate(obs, max_new_tokens=8)
    assert gen.shape == (obs.memory_summary_tokens.shape[0], 8), gen.shape
    print(f"generate OK: token shape={gen.shape}")

    # Overfit.
    losses = train_single_batch(model, obs, actions, steps=80)
    assert losses[-1] < losses[0] * 0.5, f"CE did not drop enough: {losses[0]:.4f} -> {losses[-1]:.4f}"
    print(f"PASS: Pi0HL overfit CE {losses[0]:.4f} -> {losses[-1]:.4f}")


if __name__ == "__main__":
    main()
