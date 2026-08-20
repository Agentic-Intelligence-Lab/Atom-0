from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from scripts import convert_self_collected_aligned_to_rlds as conversion


def _pose_trajectory(timestamps: np.ndarray) -> np.ndarray:
    poses = np.repeat(np.eye(4, dtype=np.float64)[None], len(timestamps), axis=0)
    poses[:, 0, 3] = timestamps
    poses[:, :3, :3] = Rotation.from_euler("z", 0.2 * timestamps).as_matrix()
    return poses


def _episode(poses: np.ndarray, timestamps: np.ndarray) -> conversion._EpisodeArrays:
    closure = timestamps.astype(np.float32) / 2.0
    state = np.concatenate((conversion._pose_vectors(poses), closure[:, None]), axis=-1)
    return conversion._EpisodeArrays(
        timestamps=timestamps,
        state=state,
        poses=(poses,),
        closures=(closure,),
        valid=np.ones(len(timestamps), dtype=bool),
    )


def test_relative_action_chunk_spans_future_one_second_without_zero_target() -> None:
    timestamps = np.linspace(0.0, 2.0, 101)
    chunk = _episode(_pose_trajectory(timestamps), timestamps).action_chunk(0)

    assert chunk.shape == (50, 7)
    np.testing.assert_allclose(chunk[0, :3], [0.02, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(chunk[-1, :3], [1.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(chunk[[0, -1], 3:6], [[0.0, 0.0, 0.004], [0.0, 0.0, 0.2]], atol=1e-6)
    np.testing.assert_allclose(chunk[[0, -1], 6], [0.01, 0.5], atol=1e-6)


def test_relative_actions_are_invariant_to_station_extrinsic() -> None:
    timestamps = np.linspace(0.0, 2.0, 101)
    poses = _pose_trajectory(timestamps)
    station_transform = np.eye(4)
    station_transform[:3, :3] = Rotation.from_euler("xyz", [0.3, -0.2, 0.4]).as_matrix()
    station_transform[:3, 3] = [1.2, -0.7, 0.5]
    transformed = np.einsum("ij,tjk->tik", station_transform, poses)

    expected = _episode(poses, timestamps).action_chunk(25)
    actual = _episode(transformed, timestamps).action_chunk(25)
    np.testing.assert_allclose(actual, expected, atol=2e-6)


def test_current_observation_anchors_delayed_robot_action_timeline() -> None:
    observation_times = np.linspace(0.0, 2.0, 101)
    observation_poses = _pose_trajectory(observation_times)
    action_times = observation_times + 0.04
    action_poses = _pose_trajectory(action_times)
    observation_closure = observation_times.astype(np.float32) / 2.0
    action_closure = action_times.astype(np.float32) / 2.0
    state = np.concatenate(
        (conversion._pose_vectors(observation_poses), observation_closure[:, None]), axis=-1
    )
    episode = conversion._EpisodeArrays(
        timestamps=observation_times,
        state=state,
        poses=(observation_poses,),
        closures=(observation_closure,),
        valid=np.ones(len(observation_times), dtype=bool),
        action_timestamps=action_times,
        action_poses=(action_poses,),
        action_closures=(action_closure,),
    )

    chunk = episode.action_chunk(0)
    assert chunk.shape == (50, 7)
    np.testing.assert_allclose(chunk[0, :3], [0.02, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(chunk[-1, :3], [1.0, 0.0, 0.0], atol=1e-6)


def test_piper_native_tcp_is_moved_to_canonical_gripper_center() -> None:
    raw = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0, 0.0, 0.0, np.pi / 2],
        ]
    )
    canonical = conversion._right_multiply_poses(
        conversion._piper_pose_matrices(raw), conversion.PIPER_T_NATIVE_TO_CANONICAL
    )

    np.testing.assert_allclose(canonical[0, :3, 3], [0.07503, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(canonical[1, :3, 3], [1.0, 2.07503, 3.0], atol=1e-9)
    np.testing.assert_allclose(canonical[0, :3, :3], conversion.PIPER_T_ALIGN[:3, :3], atol=1e-9)
