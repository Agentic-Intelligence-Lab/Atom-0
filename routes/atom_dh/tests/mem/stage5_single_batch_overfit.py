"""MEM Stage 5: single-batch overfit sanity.

目标：
  - 在同一个 toy batch 上训练 MEM 短期记忆模型，确认 loss 能下降。
  - 这个测试覆盖“代码不仅能 forward/backward，还能被 optimizer 更新并拟合同一批数据”。

运行方式：
    cd Atom-0
    python tests/mem/stage5_single_batch_overfit.py

说明：
  - 使用 dummy Gemma/SigLIP，history_length=6。
  - 不依赖外部数据集或 checkpoint。
  - 这是慢一点的 smoke/overfit 脚本，不放进默认 pytest。
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax

from openpi.models import model as _model
from openpi.models.pi0 import Pi0
from openpi.models.pi0_config import Pi0Config


def make_model() -> tuple[Pi0Config, Pi0]:
    config = Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        discrete_state_input=False,
        history_length=6,
        history_stride_seconds=1.0,
        ki_enabled=False,
    )
    return config, config.create(jax.random.key(0))


def make_single_batch(config: Pi0Config, batch_size: int = 1):
    image = jax.random.normal(jax.random.key(10), (batch_size, config.history_length, 224, 224, 3)) * 0.1
    image_mask = jnp.ones((batch_size, config.history_length), dtype=jnp.bool_)
    state_history = jax.random.normal(
        jax.random.key(11), (batch_size, config.history_length, config.action_dim)
    ) * 0.1
    state = state_history[:, -1]

    with _model.at.disable_typechecking():
        obs = _model.Observation(
            images={key: image for key in _model.IMAGE_KEYS},
            image_masks={key: image_mask for key in _model.IMAGE_KEYS},
            state=state,
            state_history=state_history,
            tokenized_prompt=jnp.zeros((batch_size, config.max_token_len), dtype=jnp.int32),
            tokenized_prompt_mask=jnp.ones((batch_size, config.max_token_len), dtype=jnp.bool_),
        )

    actions = jax.random.normal(jax.random.key(12), (batch_size, config.action_horizon, config.action_dim)) * 0.1
    return obs, actions


def train_single_batch(model: Pi0, obs, actions, steps: int = 80, lr: float = 1e-3) -> list[float]:
    optimizer = optax.adam(lr)
    graphdef, params = nnx.split(model)
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state):
        def loss_fn(p):
            m = nnx.merge(graphdef, p)
            loss = m.compute_loss(jax.random.key(42), obs, actions, train=True)
            return jnp.mean(loss["flow"] if isinstance(loss, dict) else loss)

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
    print("\n========== MEM Stage 5: Single-Batch Overfit ==========\n")
    config, model = make_model()
    obs, actions = make_single_batch(config)
    losses = train_single_batch(model, obs, actions)

    window = 10
    start = sum(losses[:window]) / window
    end = sum(losses[-window:]) / window
    print(f"\n    loss: {start:.6f} -> {end:.6f}")

    assert jnp.isfinite(jnp.array(losses)).all(), "loss contains NaN/Inf"
    assert end < start * 0.8, f"loss did not clearly decrease enough: {start:.6f} -> {end:.6f}"
    print("\nPASS: MEM single-batch overfit\n")


if __name__ == "__main__":
    main()
