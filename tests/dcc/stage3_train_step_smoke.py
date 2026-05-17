"""DCC Stage 3: debug config/data-loader train-step smoke."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

import jax
import jax.numpy as jnp

from openpi.training import config as _config
from openpi.training import data_loader


def main():
    config = _config.get_config("debug_pi05_dcc")
    loader = data_loader.create_data_loader(config, shuffle=False, num_batches=1, skip_norm_stats=True)
    obs, actions = next(iter(loader))
    model = config.model.create(jax.random.key(0))

    prefix_tokens, prefix_mask, _ = model.embed_prefix(obs)
    obs_without_subgoal = obs.replace(subgoal_image_masks=None, subgoal_images=None)
    prefix_tokens_without_subgoal, prefix_mask_without_subgoal, _ = model.embed_prefix(obs_without_subgoal)
    assert prefix_tokens.shape[1] == prefix_tokens_without_subgoal.shape[1]
    assert prefix_mask.shape == prefix_mask_without_subgoal.shape

    loss = model.compute_loss(jax.random.key(1), obs, actions)
    assert set(loss) == {"flow", "ki_fast"}
    assert jnp.all(jnp.isfinite(loss["flow"]))
    assert jnp.all(jnp.isfinite(loss["ki_fast"]))
    print("PASS: DCC debug train-step smoke")


if __name__ == "__main__":
    main()
