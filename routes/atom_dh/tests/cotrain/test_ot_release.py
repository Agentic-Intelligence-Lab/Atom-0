"""CPU checks of the transferred trajectory alignment and masking contracts."""
import jax
import jax.numpy as jnp
import numpy as np

from openpi.cotrain import action_space
from openpi.cotrain import ot_loss


def test_registered_action_masks_match_native_mappings():
    for spec in action_space.UNIFIED_ACTION_SPECS.values():
        assert len(spec.action_mask) == 80
        assert sum(spec.action_mask) == len(spec.action_mapping)
        assert set(spec.absolute_to_delta_slots) <= set(spec.action_target_slots)
        assert set(spec.absolute_to_delta_slots) <= set(spec.state_target_slots)


def test_soft_dtw_prefers_matching_trajectory():
    trajectory = jnp.arange(12, dtype=jnp.float32).reshape(4, 3) / 10
    matched = ot_loss.soft_dtw_distance(trajectory, trajectory)
    displaced = ot_loss.soft_dtw_distance(trajectory, trajectory + 10)
    assert np.isfinite(matched)
    assert matched < displaced


def test_sinkhorn_ignores_padded_support_and_has_finite_gradients():
    human = jnp.array([[0.0, 0.0], [1.0, 1.0], [100.0, 100.0]])
    robot = jnp.array([[0.1, 0.0], [1.0, 1.1], [-100.0, -100.0]])
    valid = jnp.array([True, True, False])
    scale = jnp.ones((3, 3))

    def loss(values):
        return ot_loss.sinkhorn_ot_loss(values, robot, scale, valid_h=valid, valid_r=valid)

    padded = loss(human)
    compact = ot_loss.sinkhorn_ot_loss(human[:2], robot[:2], scale[:2, :2])
    np.testing.assert_allclose(padded, compact, rtol=1e-5, atol=1e-6)
    gradient = jax.grad(loss)(human)
    assert np.isfinite(gradient).all()
    np.testing.assert_array_equal(gradient[-1], np.zeros(2))
