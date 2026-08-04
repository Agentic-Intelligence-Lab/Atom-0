#!/usr/bin/env python3
"""Convert audited Hangzhou/Shenzhen visual labels into aligned Stage-2 RLDS.

The action frame is the fixed head/front color optical camera. Human chunks
cover one physical second; Piper robot chunks cover four seconds to compensate
for the slower embodiment. Both are sampled to 100 points over the full window
and are later uniformly resampled to pi0's 50-point horizon.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
import dataclasses
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import tensorflow_datasets as tfds


CONFIG_DIMS = {
    "aligned_hangzhou_human_right": 7,
    "aligned_hangzhou_robot_right": 7,
    "aligned_shenzhen_human_bimanual": 14,
}
IMAGE_SHAPE = (480, 640, 3)
SOURCE_HORIZON = 100


def _decode_jpeg(payload: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode JPEG frame")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _nearest_index(timestamps: np.ndarray, timestamp: float) -> int:
    position = int(np.searchsorted(timestamps, timestamp))
    if position <= 0:
        return 0
    if position >= len(timestamps):
        return len(timestamps) - 1
    return position if abs(timestamps[position] - timestamp) < abs(timestamps[position - 1] - timestamp) else position - 1


def _runs(indices: np.ndarray) -> list[np.ndarray]:
    if len(indices) == 0:
        return []
    boundaries = np.flatnonzero(np.diff(indices) != 1) + 1
    return [part for part in np.split(indices, boundaries) if len(part)]


def _pose_vectors(poses: np.ndarray) -> np.ndarray:
    euler_ypr = Rotation.from_matrix(poses[:, :3, :3]).as_euler("ZYX")
    euler_ypr = np.unwrap(euler_ypr, axis=0)
    return np.concatenate((poses[:, :3, 3], euler_ypr), axis=-1).astype(np.float32)


def _quality_valid(poses: np.ndarray, valid: np.ndarray) -> np.ndarray:
    result = valid.copy()
    if len(poses) < 2:
        return result
    translation_step = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=-1)
    rotation_step = Rotation.from_matrix(
        np.einsum("tji,tjk->tik", poses[:-1, :3, :3], poses[1:, :3, :3])
    ).magnitude()
    bad = np.flatnonzero((translation_step > 0.05) | (rotation_step > np.deg2rad(30.0)))
    result[bad] = False
    result[np.minimum(bad + 1, len(result) - 1)] = False
    return result


@dataclasses.dataclass
class _EpisodeArrays:
    timestamps: np.ndarray
    state: np.ndarray
    poses: tuple[np.ndarray, ...]
    valid: np.ndarray
    horizon_seconds: float

    def candidate_runs(self) -> list[np.ndarray]:
        candidates = []
        for run in _runs(np.flatnonzero(self.valid)):
            if len(run) < 2:
                continue
            last_time = self.timestamps[run[-1]]
            keep = run[self.timestamps[run] + self.horizon_seconds <= last_time + 1e-9]
            if len(keep):
                candidates.extend(keep.tolist())
        return _runs(np.asarray(candidates, dtype=np.int64))

    def action_chunk(self, index: int) -> np.ndarray:
        end_time = self.timestamps[index] + self.horizon_seconds
        end = int(np.searchsorted(self.timestamps, end_time, side="left"))
        segment = np.arange(index, min(end + 1, len(self.timestamps)))
        if len(segment) < 2 or not np.all(self.valid[segment]):
            raise ValueError(f"Action horizon crosses an invalid region at frame {index}")
        target_times = np.linspace(self.timestamps[index], end_time, SOURCE_HORIZON)
        values = self.state[segment]
        return np.stack(
            [np.interp(target_times, self.timestamps[segment], values[:, dim]) for dim in range(values.shape[-1])],
            axis=-1,
        ).astype(np.float32)


def _load_hangzhou_arrays(source: Path, domain: str) -> _EpisodeArrays:
    labels_path = source / ("labels_calibrated.npz" if domain == "human" else "labels.npz")
    with np.load(labels_path) as labels:
        timestamps = np.asarray(labels["timestamps"], dtype=np.float64)
        poses = np.asarray(labels["pose_cam_smooth"], dtype=np.float64)
        valid = np.asarray(labels["valid_filled"], dtype=bool)
        closure_key = "closure_calibrated" if domain == "human" else "closure_smooth"
        closure = np.asarray(labels[closure_key], dtype=np.float32)
    valid &= np.isfinite(closure)
    valid = _quality_valid(poses, valid)
    state = np.concatenate((_pose_vectors(poses), closure[:, None]), axis=-1)
    return _EpisodeArrays(timestamps, state, (poses,), valid, 1.0 if domain == "human" else 4.0)


def _load_shenzhen_arrays(source: Path) -> _EpisodeArrays:
    with np.load(source / "left_hand" / "labels.npz") as left:
        timestamps = np.asarray(left["timestamps"], dtype=np.float64)
        left_pose = np.asarray(left["pose_cam_smooth"], dtype=np.float64)
        left_valid = np.asarray(left["valid_filled"], dtype=bool)
        # This value is retained in native 14D for provenance, but the action
        # mapping masks it until a left-hand gauge calibration is recorded.
        left_closure = np.asarray(left["closure_smooth"], dtype=np.float32)
    with np.load(source / "right_hand" / "labels_calibrated.npz") as right:
        right_timestamps = np.asarray(right["timestamps"], dtype=np.float64)
        right_pose = np.asarray(right["pose_cam_smooth"], dtype=np.float64)
        right_valid = np.asarray(right["valid_filled"], dtype=bool)
        right_closure = np.asarray(right["closure_calibrated"], dtype=np.float32)
    if len(timestamps) != len(right_timestamps) or not np.allclose(timestamps, right_timestamps, atol=1e-4):
        raise ValueError(f"Left/right label timelines do not match: {source}")
    left_valid = _quality_valid(left_pose, left_valid & np.isfinite(left_closure))
    right_valid = _quality_valid(right_pose, right_valid & np.isfinite(right_closure))
    valid = left_valid & right_valid
    state = np.concatenate(
        (
            _pose_vectors(left_pose),
            left_closure[:, None],
            _pose_vectors(right_pose),
            right_closure[:, None],
        ),
        axis=-1,
    )
    return _EpisodeArrays(timestamps, state, (left_pose, right_pose), valid, 1.0)


def _read_mcap_images(source: Path, domain: str) -> dict[str, tuple[np.ndarray, list[bytes]]]:
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore

    metadata = json.loads((source / "episode_metadata.json").read_text(encoding="utf-8"))
    bag = Path(metadata["source_bag"])
    topics = {
        "base": "/piper/camera_front/color/image_raw/compressed",
        "right_wrist": (
            "/piper/camera_left/color/image_raw/compressed"
            if domain == "human"
            else "/piper/camera_right/color/image_raw/compressed"
        ),
    }
    streams: dict[str, list[tuple[float, bytes]]] = {slot: [] for slot in topics}
    with AnyReader([bag], default_typestore=get_typestore(Stores.ROS2_JAZZY)) as reader:
        by_topic = {connection.topic: connection for connection in reader.connections}
        missing = set(topics.values()) - by_topic.keys()
        if missing:
            raise ValueError(f"{bag} is missing image topics {sorted(missing)}")
        wanted = [by_topic[topic] for topic in topics.values()]
        topic_to_slot = {topic: slot for slot, topic in topics.items()}
        for connection, bag_timestamp, raw in reader.messages(connections=wanted):
            message = reader.deserialize(raw, connection.msgtype)
            streams[topic_to_slot[connection.topic]].append((bag_timestamp * 1e-9, bytes(message.data)))
    return {
        slot: (np.asarray([item[0] for item in rows], dtype=np.float64), [item[1] for item in rows])
        for slot, rows in streams.items()
    }


def _video_frames(path: Path, indices: list[int]) -> dict[int, np.ndarray]:
    wanted = set(indices)
    result = {}
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video {path}")
    frame_index = 0
    while wanted:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index in wanted:
            result[frame_index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            wanted.remove(frame_index)
        frame_index += 1
    capture.release()
    if wanted:
        raise ValueError(f"Missing video frames {sorted(wanted)[:10]} in {path}")
    return result


class _AlignedConfig(tfds.core.BuilderConfig):
    def __init__(self, *, name: str, manifest: Path, rows: list[dict]):
        super().__init__(name=name, version="1.0.0", description=f"Atom aligned {name}")
        self.manifest = manifest
        self.rows = rows


class AtomAlignedRlds(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")

    def _info(self) -> tfds.core.DatasetInfo:
        action_dim = CONFIG_DIMS[self.builder_config.name]
        step = tfds.features.FeaturesDict(
            {
                "state": tfds.features.Tensor(shape=(action_dim,), dtype=np.float32),
                "action": tfds.features.Tensor(shape=(action_dim,), dtype=np.float32),
                "actions": tfds.features.Tensor(shape=(SOURCE_HORIZON, action_dim), dtype=np.float32),
                "image_base": tfds.features.Image(shape=IMAGE_SHAPE, dtype=np.uint8, encoding_format="jpeg"),
                "image_left_wrist": tfds.features.Image(shape=IMAGE_SHAPE, dtype=np.uint8, encoding_format="jpeg"),
                "image_right_wrist": tfds.features.Image(shape=IMAGE_SHAPE, dtype=np.uint8, encoding_format="jpeg"),
                "image_mask_base": np.bool_,
                "image_mask_left_wrist": np.bool_,
                "image_mask_right_wrist": np.bool_,
                "prompt": tfds.features.Text(),
                "eef_frame": tfds.features.Text(),
                "is_first": np.bool_,
                "is_last": np.bool_,
                "is_terminal": np.bool_,
                "discount": np.float32,
                "reward": np.float32,
            }
        )
        return tfds.core.DatasetInfo(
            builder=self,
            description="Self-collected human/robot aligned demonstrations with visual RGB-D 6-DoF labels.",
            features=tfds.features.FeaturesDict(
                {
                    "episode_metadata": tfds.features.FeaturesDict(
                        {
                            "source_episode": tfds.features.Text(),
                            "source_episode_id": tfds.features.Text(),
                            "source_start_index": np.int64,
                            "source_end_index": np.int64,
                            "split_policy": tfds.features.Text(),
                            "eef_frame": tfds.features.Text(),
                        }
                    ),
                    "steps": tfds.features.Dataset(step),
                }
            ),
        )

    def _split_generators(self, dl_manager):
        del dl_manager
        split_rows = {
            name: [row for row in self.builder_config.rows if row["split"] == name]
            for name in ("train", "seen_test", "unseen_test")
        }
        if any(not rows for rows in split_rows.values()):
            raise ValueError({name: len(rows) for name, rows in split_rows.items()})
        return {name: self._generate_examples(rows) for name, rows in split_rows.items()}

    def _generate_examples(self, rows: list[dict]) -> Iterator[tuple[str, dict[str, Any]]]:
        blank = np.zeros(IMAGE_SHAPE, dtype=np.uint8)
        config_name = self.builder_config.name
        for row in rows:
            source = Path(row["source"])
            if config_name.startswith("aligned_hangzhou"):
                domain = "human" if "human" in config_name else "robot"
                arrays = _load_hangzhou_arrays(source, domain)
                streams = _read_mcap_images(source, domain)

                def images_for(index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, bool, bool]:
                    base_idx = _nearest_index(streams["base"][0], arrays.timestamps[index])
                    wrist_idx = _nearest_index(streams["right_wrist"][0], arrays.timestamps[index])
                    base_ok = abs(streams["base"][0][base_idx] - arrays.timestamps[index]) <= 0.050
                    wrist_ok = abs(streams["right_wrist"][0][wrist_idx] - arrays.timestamps[index]) <= 0.050
                    return (
                        _decode_jpeg(streams["base"][1][base_idx]) if base_ok else blank,
                        blank,
                        _decode_jpeg(streams["right_wrist"][1][wrist_idx]) if wrist_ok else blank,
                        base_ok,
                        False,
                        wrist_ok,
                    )
            else:
                arrays = _load_shenzhen_arrays(source)
                episode_metadata = json.loads((source / "episode_metadata.json").read_text(encoding="utf-8"))
                original = Path(episode_metadata["source_episode"])
                all_indices = [int(index) for run in arrays.candidate_runs() for index in run]
                video_frames = {
                    "base": _video_frames(original / "head" / "rgb.mp4", all_indices),
                    "left": _video_frames(original / "left_hand" / "rgb.mp4", all_indices),
                    "right": _video_frames(original / "right_hand" / "rgb.mp4", all_indices),
                }

                def images_for(index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, bool, bool]:
                    return video_frames["base"][index], video_frames["left"][index], video_frames["right"][index], True, True, True

            for run_index, run in enumerate(arrays.candidate_runs()):
                for chunk_index, start in enumerate(range(0, len(run), 256)):
                    chunk = run[start : start + 256]
                    episode_key = f"{row['episode_id'].replace('/', '__')}__run_{run_index:03d}__chunk_{chunk_index:03d}"
                    yield episode_key, {
                        "episode_metadata": {
                            "source_episode": str(source),
                            "source_episode_id": row["episode_id"],
                            "source_start_index": np.int64(chunk[0]),
                            "source_end_index": np.int64(chunk[-1] + 1),
                            "split_policy": "task-disjoint unseen; deterministic trajectory-disjoint seen",
                            "eef_frame": "fixed_head_color_optical_camera",
                        },
                        "steps": self._steps(row, arrays, chunk, images_for),
                    }

    @staticmethod
    def _steps(row: dict, arrays: _EpisodeArrays, indices: np.ndarray, images_for) -> Iterator[dict[str, Any]]:
        for offset, index in enumerate(indices):
            base, left, right, base_mask, left_mask, right_mask = images_for(int(index))
            actions = arrays.action_chunk(int(index))
            yield {
                "state": arrays.state[index],
                "action": actions[0],
                "actions": actions,
                "image_base": base,
                "image_left_wrist": left,
                "image_right_wrist": right,
                "image_mask_base": base_mask,
                "image_mask_left_wrist": left_mask,
                "image_mask_right_wrist": right_mask,
                "prompt": row["prompt"],
                "eef_frame": "fixed_head_color_optical_camera",
                "is_first": offset == 0,
                "is_last": offset == len(indices) - 1,
                "is_terminal": False,
                "discount": np.float32(1.0),
                "reward": np.float32(0.0),
            }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", choices=tuple(CONFIG_DIMS), required=True)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--output-data-dir", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    rows = [row for row in manifest["episodes"] if row["dataset_id"] == args.config_name and row["split"] != "quality_excluded"]
    if not rows:
        raise ValueError(f"No eligible rows for {args.config_name}")
    config = _AlignedConfig(name=args.config_name, manifest=args.split_manifest.resolve(), rows=rows)
    builder = AtomAlignedRlds(data_dir=str(args.output_data_dir.resolve()), config=config)
    target = Path(builder.data_dir)
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing builder {target}")
    print(f"config={args.config_name} rows={len(rows)} target={target}")
    builder.download_and_prepare()
    print(f"Prepared self-collected RLDS builder: {builder.data_dir}")


if __name__ == "__main__":
    main()
