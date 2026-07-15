import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import pi0
from openpi.models import pi0_config


def test_action_mask_broadcasts_across_horizon() -> None:
    mask = jnp.asarray([[True, False, True], [False, True, False]])
    broadcast = pi0._broadcast_action_mask(mask, (2, 4, 3))
    assert broadcast.shape == (2, 4, 3)
    np.testing.assert_array_equal(broadcast[:, 0], mask)
    np.testing.assert_array_equal(broadcast[:, -1], mask)


def test_missing_action_mask_keeps_all_dimensions_valid() -> None:
    broadcast = pi0._broadcast_action_mask(None, (2, 4, 3))
    np.testing.assert_array_equal(broadcast, np.ones((2, 4, 3), dtype=bool))


def test_action_mask_rejects_wrong_width() -> None:
    with pytest.raises(ValueError, match="action_mask shape"):
        pi0._broadcast_action_mask(jnp.ones((2, 2), dtype=bool), (2, 4, 3))


def test_flow_loss_ignores_masked_action_values() -> None:
    config = pi0_config.Pi0Config(
        paligemma_variant="dummy", action_expert_variant="dummy", action_dim=5, action_horizon=2, max_token_len=8
    )
    model = config.create(jax.random.key(0))
    observation, actions = config.fake_obs(batch_size=1), config.fake_act(batch_size=1)
    observation = observation.replace(action_mask=jnp.asarray([[True, False, True, False, False]]))
    changed = actions.at[..., 1].set(1000).at[..., 3:].set(-1000)

    expected = model.compute_loss(jax.random.key(1), observation, actions)
    actual = model.compute_loss(jax.random.key(1), observation, changed)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0)


def test_sampling_keeps_masked_dimensions_zero() -> None:
    config = pi0_config.Pi0Config(
        paligemma_variant="dummy", action_expert_variant="dummy", action_dim=5, action_horizon=2, max_token_len=8
    )
    model = config.create(jax.random.key(0))
    observation = config.fake_obs(batch_size=1).replace(action_mask=jnp.asarray([[True, False, True, False, False]]))

    actions = model.sample_actions(jax.random.key(1), observation, num_steps=2)
    np.testing.assert_array_equal(actions[..., [1, 3, 4]], 0)
