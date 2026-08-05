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


def test_aligned_restructure_accepts_per_step_constant_eef_frame() -> None:
    tf = pytest.importorskip("tensorflow")
    steps = 2
    trajectory = {
        "actions": tf.zeros([steps, 100, 7], tf.float32),
        "state": tf.zeros([steps, 7], tf.float32),
        "image_base": tf.zeros([steps, 2, 2, 3], tf.uint8),
        "image_left_wrist": tf.zeros([steps, 2, 2, 3], tf.uint8),
        "image_right_wrist": tf.zeros([steps, 2, 2, 3], tf.uint8),
        "image_mask_base": tf.ones([steps], tf.bool),
        "image_mask_left_wrist": tf.zeros([steps], tf.bool),
        "image_mask_right_wrist": tf.ones([steps], tf.bool),
        "prompt": tf.constant(["task", "task"]),
        "eef_frame": tf.constant(
            ["fixed_head_color_optical_camera", "fixed_head_color_optical_camera"]
        ),
    }

    output = rlds_dataset._aligned_parallel_gripper_restructure(  # noqa: SLF001
        trajectory, "aligned_hangzhou_human_right"
    )

    assert output["prompt_prefix"].shape == (steps,)
    assert output["prompt_prefix"].numpy().tolist() == [
        b"Action Mode: eef. EEF Frame: fixed_head_color_optical_camera. ",
        b"Action Mode: eef. EEF Frame: fixed_head_color_optical_camera. ",
    ]


def test_aligned_restructure_rejects_inconsistent_eef_frame() -> None:
    tf = pytest.importorskip("tensorflow")
    with pytest.raises(tf.errors.InvalidArgumentError, match="eef_frame must be constant"):
        rlds_dataset._episode_scalar_string(  # noqa: SLF001
            tf.constant(["camera_a", "camera_b"]), field_name="eef_frame"
        )
