"""MEM Stage 3: debug config/data-loader train-step smoke.

This validates that the debug_pi05_mem config can create a fake MEM batch and compute a loss.
It intentionally avoids downloading external checkpoints or datasets.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

import jax
import jax.numpy as jnp

from openpi.training import config as _config
from openpi.training import data_loader


def main():
    config = _config.get_config("debug_pi05_mem")
    loader = data_loader.create_data_loader(config, shuffle=False, num_batches=1, skip_norm_stats=True)
    obs, actions = next(iter(loader))
    model = config.model.create(jax.random.key(0))
    loss = model.compute_loss(jax.random.key(1), obs, actions)
    if isinstance(loss, dict):
        assert jnp.all(jnp.isfinite(loss["flow"]))
        assert jnp.all(jnp.isfinite(loss["ki_fast"]))
    else:
        assert jnp.all(jnp.isfinite(loss))
    print("PASS: MEM debug train-step smoke")


if __name__ == "__main__":
    main()
