#!/usr/bin/env python3
"""Convert a public EgoMimic HDF5 file into Atom's RLDS/TFDS input format.

EgoMimic publishes robomimic-style files, not RLDS builders. This converter
preserves its official 100-point future trajectories and keeps the human and
robot domains in separate TFDS configs so they receive independent norm stats.

Examples:

    python scripts/convert_egomimic_hdf5_to_rlds.py \
      --source-hdf5 /data/junhe/datasets/EgoMimic/groceries_human.hdf5 \
      --output-data-dir /data/junhe/RLDS/EgoMimic_smoke \
      --max-train-episodes 2 --max-validation-episodes 1

    python scripts/convert_egomimic_hdf5_to_rlds.py \
      --source-hdf5 /data/junhe/datasets/EgoMimic/groceries_robot.hdf5 \
      --output-data-dir /data/junhe/RLDS/EgoMimic_smoke \
      --max-train-episodes 2 --max-validation-episodes 1

The Hugging Face dataset page currently does not declare a dataset license.
Confirm research-use permission before using these files beyond internal
evaluation; the MIT license in the EgoMimic code repository is not assumed to
license the separately hosted dataset.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
import dataclasses
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import tensorflow_datasets as tfds

_TASK_PROMPTS = {
    "bowlplace": "Pick up the toy and place it in the bowl.",
    "groceries": "Pick up the grocery item and place it in the shopping bag.",
    "smallclothfold": "Fold the small cloth.",
}
_SINGLE_ARM_TASKS = frozenset({"bowlplace"})
_BIMANUAL_TASKS = frozenset({"groceries", "smallclothfold"})


def _parse_source_name(path: Path) -> tuple[str, str]:
    stem = path.stem
    for task in _TASK_PROMPTS:
        for domain in ("human", "robot"):
            if stem == f"{task}_{domain}":
                return task, domain
    expected = ", ".join(f"{task}_{domain}" for task in _TASK_PROMPTS for domain in ("human", "robot"))
    raise ValueError(f"Unsupported EgoMimic filename {path.name!r}; expected one of: {expected}")


def _decode_demo_ids(values) -> list[str]:
    result = []
    for value in values:
        decoded = value.decode("utf-8") if isinstance(value, bytes) else value
        result.append(str(decoded))
    return result


def _demo_sort_key(name: str) -> tuple[int, str]:
    try:
        return int(name.rsplit("_", 1)[1]), name
    except (IndexError, ValueError):
        return 2**31 - 1, name


@dataclasses.dataclass(frozen=True)
class _Schema:
    state_dim: int
    action_dim: int
    xyz_dim: int
    joint_dim: int
    horizon: int
    image_shapes: dict[str, tuple[int, int, int]]
    image_keys: dict[str, str | None]


def _inspect_schema(source: Path, task: str, domain: str) -> _Schema:
    with h5py.File(source, "r") as h5:
        demo_names = sorted(h5["data"].keys(), key=_demo_sort_key)
        if not demo_names:
            raise ValueError(f"No demos found under data/ in {source}")
        demo = h5[f"data/{demo_names[0]}"]
        obs = demo["obs"]

        xyz_shape = demo["actions_xyz_act"].shape
        if len(xyz_shape) != 3:
            raise ValueError(f"actions_xyz_act must be [T,H,D], got {xyz_shape}")
        horizon = int(xyz_shape[1])
        xyz_dim = int(xyz_shape[2])
        expected_xyz_dim = 3 if task in _SINGLE_ARM_TASKS else 6
        if xyz_dim != expected_xyz_dim:
            raise ValueError(f"{task} expects XYZ width {expected_xyz_dim}, got {xyz_dim}")
        if int(obs["ee_pose"].shape[-1]) != xyz_dim:
            raise ValueError("obs/ee_pose and actions_xyz_act have different XYZ widths")

        joint_dim = 0
        if domain == "robot":
            joint_shape = demo["actions_joints_act"].shape
            if len(joint_shape) != 3 or int(joint_shape[1]) != horizon:
                raise ValueError(
                    f"actions_joints_act must be [T,{horizon},D], got {joint_shape}"
                )
            joint_dim = int(joint_shape[2])
            expected_joint_dim = 7 if task in _SINGLE_ARM_TASKS else 14
            if joint_dim != expected_joint_dim:
                raise ValueError(f"{task} robot expects joint/gripper width {expected_joint_dim}, got {joint_dim}")
            if int(obs["joint_positions"].shape[-1]) != joint_dim:
                raise ValueError("obs/joint_positions and actions_joints_act have different widths")

        if "front_img_1_line" not in obs:
            raise ValueError(f"{source} is missing required obs/front_img_1_line")
        base_shape = tuple(int(v) for v in obs["front_img_1_line"].shape[1:])
        if len(base_shape) != 3 or base_shape[-1] != 3:
            raise ValueError(f"front_img_1_line must be RGB, got {base_shape}")

        image_keys: dict[str, str | None] = {
            "base": "front_img_1_line",
            "left_wrist": "left_wrist_img" if "left_wrist_img" in obs else None,
            "right_wrist": "right_wrist_img" if "right_wrist_img" in obs else None,
        }
        image_shapes = {"base": base_shape}
        for slot in ("left_wrist", "right_wrist"):
            key = image_keys[slot]
            shape = base_shape if key is None else tuple(int(v) for v in obs[key].shape[1:])
            if len(shape) != 3 or shape[-1] != 3:
                raise ValueError(f"{key} must be RGB, got {shape}")
            image_shapes[slot] = shape

    return _Schema(
        state_dim=joint_dim + xyz_dim,
        action_dim=joint_dim + xyz_dim,
        xyz_dim=xyz_dim,
        joint_dim=joint_dim,
        horizon=horizon,
        image_shapes=image_shapes,
        image_keys=image_keys,
    )


class _EgoMimicConfig(tfds.core.BuilderConfig):
    def __init__(
        self,
        *,
        name: str,
        source_path: Path,
        task: str,
        domain: str,
        prompt: str,
        max_train_episodes: int | None,
        max_validation_episodes: int | None,
    ):
        super().__init__(name=name, version="1.0.0", description=f"EgoMimic {task} {domain}")
        self.source_path = source_path
        self.task = task
        self.domain = domain
        self.prompt = prompt
        self.max_train_episodes = max_train_episodes
        self.max_validation_episodes = max_validation_episodes


class EgoMimicRlds(tfds.core.GeneratorBasedBuilder):
    """One task/domain EgoMimic file as an episode-level RLDS builder."""

    VERSION = tfds.core.Version("1.0.0")

    def _info(self) -> tfds.core.DatasetInfo:
        self._schema = _inspect_schema(
            self.builder_config.source_path,
            self.builder_config.task,
            self.builder_config.domain,
        )
        image_features = {
            slot: tfds.features.Image(
                shape=self._schema.image_shapes[slot],
                dtype=np.uint8,
                encoding_format="jpeg",
            )
            for slot in ("base", "left_wrist", "right_wrist")
        }
        step_features = tfds.features.FeaturesDict(
            {
                "state": tfds.features.Tensor(shape=(self._schema.state_dim,), dtype=np.float32),
                "action": tfds.features.Tensor(shape=(self._schema.action_dim,), dtype=np.float32),
                "actions": tfds.features.Tensor(
                    shape=(self._schema.horizon, self._schema.action_dim),
                    dtype=np.float32,
                ),
                "image_base": image_features["base"],
                "image_left_wrist": image_features["left_wrist"],
                "image_right_wrist": image_features["right_wrist"],
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
            description=(
                "EgoMimic human/robot demonstrations converted from the public "
                "robomimic-style HDF5 release."
            ),
            features=tfds.features.FeaturesDict(
                {
                    "episode_metadata": tfds.features.FeaturesDict(
                        {
                            "source_file": tfds.features.Text(),
                            "source_demo_id": tfds.features.Text(),
                            "task": tfds.features.Text(),
                            "domain": tfds.features.Text(),
                            "eef_frame": tfds.features.Text(),
                        }
                    ),
                    "steps": tfds.features.Dataset(step_features),
                }
            ),
            homepage="https://egomimic.github.io/",
        )

    def _split_generators(self, dl_manager):
        del dl_manager
        source = self.builder_config.source_path
        with h5py.File(source, "r") as h5:
            all_demos = set(h5["data"].keys())
            if "mask/train" not in h5 or "mask/valid" not in h5:
                raise ValueError(f"{source} must contain official mask/train and mask/valid")
            train = _decode_demo_ids(h5["mask/train"][:])
            valid = _decode_demo_ids(h5["mask/valid"][:])
        if not train or not valid:
            raise ValueError(f"{source} has an empty official train or valid mask")
        unknown = (set(train) | set(valid)) - all_demos
        if unknown:
            raise ValueError(f"Official masks reference missing demos: {sorted(unknown)[:10]}")
        overlap = set(train) & set(valid)
        if overlap:
            # The published groceries files contain one long demo referenced by
            # both official masks. Preserve the publisher's split contract for
            # compatibility, but make the leakage impossible to overlook.
            print(
                "WARNING: official train/valid masks overlap for "
                f"{len(overlap)} demo(s): {sorted(overlap)[:10]}. "
                "Validation is an in-trajectory smoke metric, not a held-out result."
            )

        train = sorted(train, key=_demo_sort_key)
        valid = sorted(valid, key=_demo_sort_key)
        if self.builder_config.max_train_episodes is not None:
            train = train[: self.builder_config.max_train_episodes]
        if self.builder_config.max_validation_episodes is not None:
            valid = valid[: self.builder_config.max_validation_episodes]
        splits = {"train": self._generate_examples(train)}
        if set(train) != set(valid):
            splits["seen_test"] = self._generate_examples(valid)
            # EgoMimic has no semantic unseen split. This alias only preserves
            # Atom's two-label evaluator contract and must not be reported as
            # unseen-task performance.
            splits["unseen_test"] = self._generate_examples(valid)
        return splits

    def _generate_examples(self, demo_names: list[str]) -> Iterator[tuple[str, dict[str, Any]]]:
        source = self.builder_config.source_path
        with h5py.File(source, "r") as h5:
            for demo_name in demo_names:
                demo = h5[f"data/{demo_name}"]
                yield demo_name, {
                    "episode_metadata": {
                        "source_file": source.name,
                        "source_demo_id": demo_name,
                        "task": self.builder_config.task,
                        "domain": self.builder_config.domain,
                        "eef_frame": "current_egocentric_camera",
                    },
                    "steps": self._generate_steps(demo),
                }

    def _generate_steps(self, demo) -> Iterator[dict[str, Any]]:
        obs = demo["obs"]
        xyz_actions = demo["actions_xyz_act"]
        joint_actions = demo.get("actions_joints_act")
        xyz_state = obs["ee_pose"]
        joint_state = obs.get("joint_positions")
        length = int(xyz_state.shape[0])
        arrays = [xyz_actions]
        if joint_actions is not None:
            arrays.append(joint_actions)
        if joint_state is not None:
            arrays.append(joint_state)
        if any(int(array.shape[0]) != length for array in arrays):
            raise ValueError(f"Inconsistent trajectory lengths in {demo.name}")

        blank_images = {
            slot: np.zeros(self._schema.image_shapes[slot], dtype=np.uint8)
            for slot in ("left_wrist", "right_wrist")
        }
        for index in range(length):
            xyz_chunk = np.asarray(xyz_actions[index], dtype=np.float32)
            xyz_now = np.asarray(xyz_state[index], dtype=np.float32)
            if joint_actions is None:
                action_chunk = xyz_chunk
                state = xyz_now
            else:
                action_chunk = np.concatenate(
                    [np.asarray(joint_actions[index], dtype=np.float32), xyz_chunk],
                    axis=-1,
                )
                state = np.concatenate(
                    [np.asarray(joint_state[index], dtype=np.float32), xyz_now],
                    axis=-1,
                )

            images = {}
            masks = {}
            for slot in ("base", "left_wrist", "right_wrist"):
                source_key = self._schema.image_keys[slot]
                if source_key is None:
                    images[slot] = blank_images[slot]
                    masks[slot] = False
                else:
                    images[slot] = np.asarray(obs[source_key][index], dtype=np.uint8)
                    masks[slot] = True

            yield {
                "state": state,
                "action": action_chunk[0],
                "actions": action_chunk,
                "image_base": images["base"],
                "image_left_wrist": images["left_wrist"],
                "image_right_wrist": images["right_wrist"],
                "image_mask_base": masks["base"],
                "image_mask_left_wrist": masks["left_wrist"],
                "image_mask_right_wrist": masks["right_wrist"],
                "prompt": self.builder_config.prompt,
                "eef_frame": "current_egocentric_camera",
                "is_first": index == 0,
                "is_last": index == length - 1,
                "is_terminal": index == length - 1,
                "discount": np.float32(1.0),
                "reward": np.float32(1.0 if index == length - 1 else 0.0),
            }


def _positive_or_none(value: str) -> int | None:
    value = int(value)
    if value == 0:
        return None
    if value < 0:
        raise argparse.ArgumentTypeError("must be non-negative; use 0 for all episodes")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-hdf5", type=Path, required=True)
    parser.add_argument("--output-data-dir", type=Path, required=True)
    parser.add_argument("--max-train-episodes", type=_positive_or_none, default=None)
    parser.add_argument("--max-validation-episodes", type=_positive_or_none, default=None)
    args = parser.parse_args()

    source = args.source_hdf5.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    task, domain = _parse_source_name(source)
    config_name = f"{task}_{domain}"
    config = _EgoMimicConfig(
        name=config_name,
        source_path=source,
        task=task,
        domain=domain,
        prompt=_TASK_PROMPTS[task],
        max_train_episodes=args.max_train_episodes,
        max_validation_episodes=args.max_validation_episodes,
    )
    builder = EgoMimicRlds(
        data_dir=str(args.output_data_dir.expanduser().resolve()),
        config=config,
    )
    target = Path(builder.data_dir)
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite existing TFDS builder {target}. "
            "Use a new --output-data-dir, or move the old builder aside after inspection."
        )
    print(f"source={source}")
    print(f"task={task} domain={domain}")
    print(f"schema={_inspect_schema(source, task, domain)}")
    print(f"target={target}")
    builder.download_and_prepare()
    print(f"Prepared EgoMimic RLDS builder: {builder.data_dir}")


if __name__ == "__main__":
    main()
