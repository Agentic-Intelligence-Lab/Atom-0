from __future__ import annotations

import numpy as np
import pytest

from openpi.cotrain import rlds_dataset


def test_resample_precomputed_action_chunk_uses_full_source_window() -> None:
    tf = pytest.importorskip("tensorflow")
    actions = np.arange(100, dtype=np.float32).reshape(1, 100, 1)
    output = rlds_dataset.resample_precomputed_action_chunk(
        {"actions": tf.constant(actions)},
        action_chunk_size=5,
    )
    np.testing.assert_array_equal(
        output["actions"].numpy().reshape(-1),
        np.array([0, 25, 50, 74, 99], dtype=np.float32),
    )
    assert output["actions"].shape == (1, 5, 1)


def test_resample_precomputed_action_chunk_rejects_rank_two_actions() -> None:
    tf = pytest.importorskip("tensorflow")
    # TensorFlow raises ValueError when the static rank is known, InvalidArgumentError
    # when the same assertion is evaluated from a traced/dynamic shape.
    with pytest.raises((ValueError, tf.errors.InvalidArgumentError), match="precomputed_action_chunk"):
        rlds_dataset.resample_precomputed_action_chunk(
            {"actions": tf.zeros([10, 12], tf.float32)},
            action_chunk_size=5,
        )


def test_resample_precomputed_action_chunk_rejects_nonpositive_model_horizon() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        rlds_dataset.resample_precomputed_action_chunk(
            {"actions": np.zeros([1, 100, 12], dtype=np.float32)},
            action_chunk_size=0,
        )
