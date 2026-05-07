"""MEM Stage 1: short-term visual/state memory forward sanity."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

import jax
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.models.pi0_config import Pi0Config


def main():
    config = Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        discrete_state_input=False,
        history_length=6,
    )
    model = config.create(jax.random.key(0))
    batch_size = 2
    with _model.at.disable_typechecking():
        obs = _model.Observation(
            images={key: jnp.zeros((batch_size, 6, 224, 224, 3), dtype=jnp.float32) for key in _model.IMAGE_KEYS},
            image_masks={key: jnp.ones((batch_size, 6), dtype=jnp.bool_) for key in _model.IMAGE_KEYS},
            state=jnp.zeros((batch_size, config.action_dim), dtype=jnp.float32),
            state_history=jnp.zeros((batch_size, 6, config.action_dim), dtype=jnp.float32),
            tokenized_prompt=jnp.zeros((batch_size, config.max_token_len), dtype=jnp.int32),
            tokenized_prompt_mask=jnp.ones((batch_size, config.max_token_len), dtype=jnp.bool_),
        )
    actions = jnp.zeros((batch_size, config.action_horizon, config.action_dim), dtype=jnp.float32)
    loss = model.compute_loss(jax.random.key(1), obs, actions)
    assert loss.shape == (batch_size, config.action_horizon), loss.shape
    assert jnp.all(jnp.isfinite(loss))
    print("PASS: MEM short-term forward")


if __name__ == "__main__":
    main()
