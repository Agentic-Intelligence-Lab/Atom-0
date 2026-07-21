#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import dataclasses
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from typing import Any


CONFIG_NAME_DEFAULT = "cotrain_all_2ep"
DATASET_ID = "piper30"
PIPER_ACTION_DIM = 14
UNIFIED_ACTION_DIM = 80
LEFT_JOINT_DIMS = tuple(range(0, 6))
LEFT_GRIPPER_DIM = 6
RIGHT_JOINT_DIMS = tuple(range(7, 13))
RIGHT_GRIPPER_DIM = 13
JOINT_DELTA_DIMS = LEFT_JOINT_DIMS + RIGHT_JOINT_DIMS
GRIPPER_DIMS = (LEFT_GRIPPER_DIM, RIGHT_GRIPPER_DIM)
UNIFIED_LEFT_JOINT_DIMS = tuple(range(0, 6))
UNIFIED_LEFT_GRIPPER_DIM = 16
UNIFIED_RIGHT_JOINT_DIMS = tuple(range(29, 35))
UNIFIED_RIGHT_GRIPPER_DIM = 45
UNIFIED_PIPER_DIMS = (
    UNIFIED_LEFT_JOINT_DIMS
    + (UNIFIED_LEFT_GRIPPER_DIM,)
    + UNIFIED_RIGHT_JOINT_DIMS
    + (UNIFIED_RIGHT_GRIPPER_DIM,)
)
UNIFIED_JOINT_DELTA_DIMS = UNIFIED_LEFT_JOINT_DIMS + UNIFIED_RIGHT_JOINT_DIMS


SOURCE_EVIDENCE = {
    "config_piper30": "src/openpi/cotrain/config.py:194-210",
    "config_alias": "src/openpi/cotrain/config.py:256-264",
    "standardized_inputs": "src/openpi/cotrain/transforms.py:48-85",
    "dispatch_delta": "src/openpi/cotrain/transforms.py:101-119",
    "dispatch_normalize": "src/openpi/cotrain/transforms.py:122-147",
    "raw_dataset_mapping": "src/openpi/cotrain/rlds_dataset.py:229-257",
    "action_chunking": "src/openpi/cotrain/rlds_dataset.py:413-425,461-466",
    "delta_relative_to_anchor": "src/openpi/transforms.py:567-583",
    "mask_semantics": "src/openpi/transforms.py:852-871",
    "policy_restore_and_transforms": "src/openpi/policies/policy_config.py:17-99",
    "policy_infer": "src/openpi/policies/policy.py:70-104",
    "sampler_random_noise": "src/openpi/models/pi0.py:379-442",
    "native_validation_eval": "src/openpi/cotrain/eval.py:37-40,75-92,97-108",
    "action_state_note": "docs/cotrain_技术文档.md:97-103",
}




def piper14_from_action(values, name: str = "action"):
    import numpy as np

    arr = np.asarray(values, dtype=np.float32)
    if arr.shape[-1] == PIPER_ACTION_DIM:
        return arr.astype(np.float32, copy=False)
    if arr.shape[-1] == UNIFIED_ACTION_DIM:
        return arr[..., np.asarray(UNIFIED_PIPER_DIMS)].astype(np.float32, copy=True)
    raise ValueError(f"{name} last dim must be 14 or 80, got shape {arr.shape}")


def piper_state_to_unified(state):
    import numpy as np

    piper = np.asarray(state, dtype=np.float32)
    if piper.shape != (PIPER_ACTION_DIM,):
        raise ValueError(f"Expected Piper state shape (14,), got {piper.shape}")
    unified = np.zeros((UNIFIED_ACTION_DIM,), dtype=np.float32)
    unified[np.asarray(UNIFIED_PIPER_DIMS)] = piper
    return unified


def piper_actions_to_unified(actions):
    import numpy as np

    piper = np.asarray(actions, dtype=np.float32)
    if piper.shape[-1] != PIPER_ACTION_DIM:
        raise ValueError(f"Expected Piper actions last dim 14, got {piper.shape}")
    unified = np.zeros((*piper.shape[:-1], UNIFIED_ACTION_DIM), dtype=np.float32)
    unified[..., np.asarray(UNIFIED_PIPER_DIMS)] = piper
    return unified


def state_for_policy(state, policy_state_dim: int):
    import numpy as np

    piper = piper14_from_action(state, "state")
    if piper.shape != (PIPER_ACTION_DIM,):
        raise ValueError(f"Expected one Piper state vector, got {piper.shape}")
    if policy_state_dim == UNIFIED_ACTION_DIM:
        return piper_state_to_unified(piper)
    if policy_state_dim == PIPER_ACTION_DIM:
        return piper.astype(np.float32, copy=True)
    raise ValueError(f"Unsupported policy state dim {policy_state_dim}; expected 14 or 80")


def env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else None


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value else default


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    openpi_root = env_path("OPENPI_ROOT")
    checkpoint_dir = env_path("CHECKPOINT_DIR")
    norm_stats_path = env_path("NORM_STATS_PATH")
    dataset_dir = env_path("DATASET_DIR")

    parser = argparse.ArgumentParser(
        description="Offline Piper JAX/Orbax validation trajectory evaluator for prepared TFDS data."
    )
    parser.add_argument("--openpi-root", type=Path, default=openpi_root, required=openpi_root is None)
    parser.add_argument("--checkpoint-dir", type=Path, default=checkpoint_dir, required=checkpoint_dir is None)
    parser.add_argument("--norm-stats-path", type=Path, default=norm_stats_path, required=norm_stats_path is None)
    parser.add_argument("--dataset-dir", type=Path, default=dataset_dir, required=dataset_dir is None)
    parser.add_argument("--config-name", default=os.environ.get("CONFIG_NAME", CONFIG_NAME_DEFAULT))
    parser.add_argument("--split", choices=("seen_test", "unseen_test"), default=os.environ.get("SPLIT", "seen_test"))
    parser.add_argument("--episodes", type=int, default=env_int("EPISODES", 1))
    parser.add_argument("--anchors-per-episode", type=int, default=env_int("ANCHORS_PER_EPISODE", 1))
    parser.add_argument("--stride", type=int, default=env_int("STRIDE", 1))
    parser.add_argument("--actions-per-inference", type=int, default=env_int("ACTIONS_PER_INFERENCE", 1))
    parser.add_argument("--seed", type=int, default=env_int("SEED", 0))
    parser.add_argument("--num-samples", type=int, default=env_int("NUM_SAMPLES", 1))
    parser.add_argument("--device", choices=("cpu", "cuda"), default=os.environ.get("DEVICE", "cuda"))
    parser.add_argument("--output-json", type=Path, default=env_path("OUTPUT_JSON"))
    parser.add_argument("--output-csv", type=Path, default=env_path("OUTPUT_CSV"))
    parser.add_argument("--task-filter", default=os.environ.get("TASK_FILTER"))
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument(
        "--native-val-loss-samples",
        type=int,
        default=env_int("NATIVE_VAL_LOSS_SAMPLES", 1),
        help="Number of flow-loss noise samples per evaluated anchor; 0 disables this diagnostic.",
    )
    return parser.parse_args(argv)


def configure_device(device: str) -> None:
    if device == "cpu":
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    else:
        os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def resolve_required_paths(args: argparse.Namespace) -> None:
    args.openpi_root = args.openpi_root.expanduser().resolve()
    args.checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    args.norm_stats_path = args.norm_stats_path.expanduser().resolve()
    args.dataset_dir = args.dataset_dir.expanduser().resolve()
    if args.output_json is not None:
        args.output_json = args.output_json.expanduser().resolve()
    if args.output_csv is not None:
        args.output_csv = args.output_csv.expanduser().resolve()


def add_openpi_path(openpi_root: Path) -> None:
    package_dir = openpi_root / "src/openpi"
    if not package_dir.is_dir():
        raise FileNotFoundError(f"Missing OpenPI package directory: {package_dir}")
    src_dir = str((openpi_root / "src").resolve())
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


def require_positive_args(args: argparse.Namespace) -> None:
    for name in ("episodes", "anchors_per_episode", "stride", "actions_per_inference", "num_samples"):
        value = getattr(args, name)
        if value < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 1")
    if args.native_val_loss_samples < 0:
        raise ValueError("--native-val-loss-samples must be >= 0")


def import_status() -> dict[str, bool]:
    return {
        "jax": importlib.util.find_spec("jax") is not None,
        "flax": importlib.util.find_spec("flax") is not None,
        "orbax.checkpoint": importlib.util.find_spec("orbax.checkpoint") is not None,
        "tensorflow": importlib.util.find_spec("tensorflow") is not None,
        "tensorflow_datasets": importlib.util.find_spec("tensorflow_datasets") is not None,
    }


def require_modules(names: tuple[str, ...]) -> None:
    missing = [name for name in names if importlib.util.find_spec(name) is None]
    if missing:
        raise ModuleNotFoundError("Missing required Python modules: " + ", ".join(missing))


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")


def require_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"Missing {label}: {path}")


def require_static_files(args: argparse.Namespace) -> dict[str, bool]:
    layout = {
        "openpi_package": (args.openpi_root / "src/openpi").is_dir(),
        "checkpoint_params": (args.checkpoint_dir / "params").is_dir(),
        "checkpoint_params_manifest": (args.checkpoint_dir / "params/manifest.ocdbt").is_file(),
        "dataset_info": (args.dataset_dir / "dataset_info.json").is_file(),
        "features": (args.dataset_dir / "features.json").is_file(),
        "norm_stats": args.norm_stats_path.is_file(),
    }
    missing = [name for name, ok in layout.items() if not ok]
    if missing:
        raise FileNotFoundError(f"Missing required offline files: {missing}")
    return layout


def tfds_metadata(dataset_dir: Path) -> dict[str, Any]:
    require_modules(("tensorflow", "tensorflow_datasets"))
    import tensorflow as tf
    import tensorflow_datasets as tfds

    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass

    builder = tfds.builder_from_directory(builder_dir=str(dataset_dir))
    splits = {}
    for split_name, split_info in builder.info.splits.items():
        splits[split_name] = int(split_info.num_examples)
    return {
        "dataset_dir": str(dataset_dir),
        "name": builder.info.name,
        "version": str(builder.info.version),
        "splits": splits,
        "feature_schema": str(builder.info.features),
    }


def infer_assets_base_dir(norm_stats_path: Path, config_name: str) -> Path:
    # Expected H800 layout: <assets_base>/<config_name>/piper30/norm_stats.json.
    if (
        norm_stats_path.name == "norm_stats.json"
        and norm_stats_path.parent.name == DATASET_ID
        and norm_stats_path.parent.parent.name == config_name
    ):
        return norm_stats_path.parent.parent.parent.resolve()
    return norm_stats_path.parent.parent.parent.resolve()


def load_config_and_norm_stats(args: argparse.Namespace):
    add_openpi_path(args.openpi_root)
    from openpi.cotrain import config as cotrain_config
    from openpi.shared import normalize

    train_config = cotrain_config.get_config(args.config_name)
    train_config = dataclasses.replace(
        train_config,
        assets_base_dir=str(infer_assets_base_dir(args.norm_stats_path, args.config_name)),
    )
    norm_stats = normalize.deserialize_json(args.norm_stats_path.read_text())
    return train_config, norm_stats


def config_summary(train_config) -> dict[str, Any]:
    datasets = []
    for ds in train_config.data.datasets:
        datasets.append(
            {
                "uid": ds.uid,
                "name": ds.name,
                "version": ds.version,
                "restructure_name": ds.restructure_name,
                "action_dim": ds.action_dim,
                "delta_action_mask_dims": ds.delta_action_mask_dims,
                "train_split": ds.train_split,
                "val_splits": dict(ds.val_splits),
            }
        )
    return {
        "name": train_config.name,
        "model_action_dim": train_config.model.action_dim,
        "model_action_horizon": train_config.model.action_horizon,
        "model_type": str(train_config.model.model_type),
        "assets_base_dir": str(train_config.assets_base_dir),
        "assets_dirs": str(train_config.assets_dirs),
        "datasets": datasets,
    }


def validate_static(args: argparse.Namespace) -> dict[str, Any]:
    require_positive_args(args)
    layout = require_static_files(args)
    train_config, norm_stats = load_config_and_norm_stats(args)
    stats_summary = {
        key: {
            "mean_shape": list(value.mean.shape),
            "std_shape": list(value.std.shape),
            "has_quantiles": value.q01 is not None and value.q99 is not None,
        }
        for key, value in norm_stats.items()
    }
    return {
        "paths": resolved_path_summary(args),
        "file_layout": layout,
        "dependency_import_status": import_status(),
        "tfds_metadata": tfds_metadata(args.dataset_dir),
        "config": config_summary(train_config),
        "norm_stats": stats_summary,
        "source_evidence": SOURCE_EVIDENCE,
    }


def create_policy(args: argparse.Namespace, train_config, norm_stats):
    from openpi.policies import policy_config

    return policy_config.create_trained_policy(
        train_config,
        args.checkpoint_dir,
        default_prompt="",
        norm_stats=norm_stats,
    )


def decode_text(value: Any) -> str:
    item = value
    try:
        import numpy as np

        arr = np.asarray(value)
        item = arr.reshape(-1)[0] if arr.ndim > 0 else arr.item()
    except Exception:
        pass
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def maybe_decode_image(value: Any):
    import numpy as np

    arr = np.asarray(value)
    if arr.dtype.kind in {"S", "O"}:
        import tensorflow as tf

        raw = arr.item() if arr.ndim == 0 else arr.reshape(-1)[0]
        if isinstance(raw, str):
            raw = raw.encode("utf-8")
        return tf.io.decode_image(raw, expand_animations=False, dtype=tf.uint8).numpy()
    return arr


def get_nested(tree: dict[str, Any], *keys: str) -> Any:
    cur: Any = tree
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            raise KeyError("Missing dataset key: " + "/".join(keys))
        cur = cur[key]
    return cur


def step_from_columnar(steps: dict[str, Any], index: int) -> dict[str, Any]:
    out = {}
    for key, value in steps.items():
        if isinstance(value, dict):
            out[key] = step_from_columnar(value, index)
        else:
            out[key] = value[index]
    return out


def iter_episode_steps(raw_episode: dict[str, Any]):
    steps = raw_episode.get("steps", raw_episode)
    if isinstance(steps, dict):
        first_value = next(iter(steps.values()))
        while isinstance(first_value, dict):
            first_value = next(iter(first_value.values()))
        for i in range(len(first_value)):
            yield step_from_columnar(steps, i)
        return

    if hasattr(steps, "as_numpy_iterator"):
        import tensorflow_datasets as tfds

        for step in tfds.as_numpy(steps):
            yield step
        return

    for step in steps:
        yield step


def read_validation_episodes(args: argparse.Namespace):
    require_modules(("tensorflow", "tensorflow_datasets"))
    import tensorflow as tf
    import tensorflow_datasets as tfds

    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass

    builder = tfds.builder_from_directory(builder_dir=str(args.dataset_dir))
    dataset = builder.as_dataset(split=args.split, shuffle_files=False)
    yielded = 0
    for raw_episode in tfds.as_numpy(dataset):
        steps = list(iter_episode_steps(raw_episode))
        if not steps:
            continue
        task = decode_text(get_nested(steps[0], "task"))
        if args.task_filter and args.task_filter not in task:
            continue
        yielded += 1
        yield normalize_episode(steps, yielded - 1, task)
        if yielded >= args.episodes:
            break


def normalize_episode(steps: list[dict[str, Any]], episode_index: int, task: str) -> dict[str, Any]:
    import numpy as np

    states, actions, prompts = [], [], []
    images = {"cam_high": [], "cam_left_wrist": [], "cam_right_wrist": []}
    for step in steps:
        states.append(np.asarray(get_nested(step, "observation", "state"), dtype=np.float32))
        actions.append(np.asarray(get_nested(step, "action"), dtype=np.float32))
        prompts.append(decode_text(step.get("task", task)))
        image_dict = get_nested(step, "observation", "images")
        for key in images:
            images[key].append(maybe_decode_image(image_dict[key]))
    return {
        "episode_index": episode_index,
        "task": task,
        "states": np.stack(states).astype(np.float32),
        "actions": np.stack(actions).astype(np.float32),
        "prompts": prompts,
        "images": images,
    }


def anchor_indices(num_steps: int, stride: int, limit: int) -> list[int]:
    return list(range(0, num_steps, stride))[:limit]


def target_action_chunk(actions, anchor_index: int, horizon: int):
    import numpy as np

    indices = np.minimum(np.arange(anchor_index, anchor_index + horizon), len(actions) - 1)
    return piper14_from_action(actions[indices], "target actions")


def standardized_policy_observation(
    episode: dict[str, Any],
    anchor_index: int,
    policy_state_dim: int,
) -> dict[str, Any]:
    import numpy as np

    return {
        "state": state_for_policy(episode["states"][anchor_index], policy_state_dim),
        "image": {
            "base_0_rgb": episode["images"]["cam_high"][anchor_index],
            "left_wrist_0_rgb": episode["images"]["cam_left_wrist"][anchor_index],
            "right_wrist_0_rgb": episode["images"]["cam_right_wrist"][anchor_index],
        },
        "image_mask": {
            "base_0_rgb": np.asarray(True),
            "left_wrist_0_rgb": np.asarray(True),
            "right_wrist_0_rgb": np.asarray(True),
        },
        "prompt": episode["prompts"][anchor_index],
    }


def standardized_training_sample(
    episode: dict[str, Any],
    anchor_index: int,
    horizon: int,
    policy_state_dim: int,
) -> dict[str, Any]:
    import numpy as np

    sample = standardized_policy_observation(episode, anchor_index, policy_state_dim)
    native_actions = target_action_chunk(episode["actions"], anchor_index, horizon)
    if policy_state_dim == UNIFIED_ACTION_DIM:
        sample["actions"] = piper_actions_to_unified(native_actions)
        action_mask = np.zeros((UNIFIED_ACTION_DIM,), dtype=bool)
        action_mask[np.asarray(UNIFIED_PIPER_DIMS)] = True
        sample["action_mask"] = action_mask
    else:
        sample["actions"] = native_actions
    sample["dataset_id"] = DATASET_ID
    return sample


def native_to_absolute_actions(native_actions, current_state, actions_per_inference: int):
    import numpy as np

    actions = np.asarray(native_actions, dtype=np.float32)
    state = np.asarray(current_state, dtype=np.float32)
    if state.shape != (PIPER_ACTION_DIM,):
        raise ValueError(f"Expected state shape (14,), got {state.shape}")
    if actions.ndim != 2:
        raise ValueError(f"Expected native action chunk shape (H, D), got {actions.shape}")
    if actions_per_inference > actions.shape[0]:
        raise ValueError(
            f"--actions-per-inference ({actions_per_inference}) exceeds model horizon ({actions.shape[0]})"
        )
    if actions.shape[-1] >= UNIFIED_ACTION_DIM:
        unified_state = piper_state_to_unified(state)
        absolute_unified = actions[:actions_per_inference, :UNIFIED_ACTION_DIM].astype(np.float32, copy=True)
        absolute_unified[:, np.asarray(UNIFIED_JOINT_DELTA_DIMS)] += unified_state[np.asarray(UNIFIED_JOINT_DELTA_DIMS)]
        return piper14_from_action(absolute_unified, "absolute_unified")
    if actions.shape[-1] >= PIPER_ACTION_DIM:
        absolute = actions[:actions_per_inference, :PIPER_ACTION_DIM].astype(np.float32, copy=True)
        absolute[:, np.asarray(JOINT_DELTA_DIMS)] += state[np.asarray(JOINT_DELTA_DIMS)]
        return absolute
    raise ValueError(f"Expected native action chunk last dim >=14 or >=80, got {actions.shape}")


def metric_dict(pred_absolute, target_absolute, current_state) -> dict[str, Any]:
    import numpy as np

    pred = piper14_from_action(pred_absolute, "pred")
    target = piper14_from_action(target_absolute, "target")
    state = np.asarray(current_state, dtype=np.float32)
    err = pred - target
    abs_err = np.abs(err)
    pred_joint_move = pred[:, JOINT_DELTA_DIMS] - state[list(JOINT_DELTA_DIMS)]
    target_joint_move = target[:, JOINT_DELTA_DIMS] - state[list(JOINT_DELTA_DIMS)]
    valid_direction = np.abs(target_joint_move) > 1e-8
    direction_match = (
        float(np.mean(np.sign(pred_joint_move[valid_direction]) == np.sign(target_joint_move[valid_direction])))
        if np.any(valid_direction)
        else float("nan")
    )
    return {
        "inference_mae": float(np.mean(abs_err)),
        "inference_mse": float(np.mean(err**2)),
        "left_joints_mae": float(np.mean(abs_err[:, LEFT_JOINT_DIMS])),
        "left_gripper_mae": float(np.mean(abs_err[:, LEFT_GRIPPER_DIM])),
        "right_joints_mae": float(np.mean(abs_err[:, RIGHT_JOINT_DIMS])),
        "right_gripper_mae": float(np.mean(abs_err[:, RIGHT_GRIPPER_DIM])),
        "per_dimension_mae": np.mean(abs_err, axis=0).astype(float).tolist(),
        "max_absolute_error": float(np.max(abs_err)),
        "direction_match": direction_match,
        "finite": {
            "state": bool(np.isfinite(state).all()),
            "prediction": bool(np.isfinite(pred).all()),
            "target": bool(np.isfinite(target).all()),
            "error": bool(np.isfinite(err).all()),
        },
    }


def summarize_metric_dicts(items: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np

    if not items:
        return {}
    scalar_keys = [
        "inference_mae",
        "inference_mse",
        "left_joints_mae",
        "left_gripper_mae",
        "right_joints_mae",
        "right_gripper_mae",
        "max_absolute_error",
        "direction_match",
    ]
    out: dict[str, Any] = {"count": len(items)}
    for key in scalar_keys:
        out[key] = float(np.nanmean(np.asarray([x[key] for x in items], dtype=np.float64)))
    out["per_dimension_mae"] = np.nanmean(
        np.asarray([x["per_dimension_mae"] for x in items], dtype=np.float64), axis=0
    ).tolist()
    out["finite"] = {
        "state": bool(all(x["finite"]["state"] for x in items)),
        "prediction": bool(all(x["finite"]["prediction"] for x in items)),
        "target": bool(all(x["finite"]["target"] for x in items)),
        "error": bool(all(x["finite"]["error"] for x in items)),
    }
    return out


def sample_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np

    metrics = [s["metrics"] for s in samples]
    best_index = int(np.nanargmin([m["inference_mae"] for m in metrics]))
    mean_metrics = summarize_metric_dicts(metrics)
    median_metrics = {
        key: float(np.nanmedian([m[key] for m in metrics]))
        for key in (
            "inference_mae",
            "inference_mse",
            "left_joints_mae",
            "left_gripper_mae",
            "right_joints_mae",
            "right_gripper_mae",
            "max_absolute_error",
            "direction_match",
        )
    }
    median_metrics["per_dimension_mae"] = np.nanmedian(
        np.asarray([m["per_dimension_mae"] for m in metrics], dtype=np.float64), axis=0
    ).tolist()
    median_metrics["finite"] = mean_metrics["finite"]
    return {
        "anchor": samples[0]["metrics"],
        "mean": mean_metrics,
        "median": median_metrics,
        "best_of_n": {
            "sample_index": best_index,
            "diagnostic_only": True,
            **metrics[best_index],
        },
    }


def sample_noise(seed: int, anchor_ordinal: int, sample_index: int, action_horizon: int, action_dim: int):
    import numpy as np

    rng = np.random.default_rng(seed + anchor_ordinal * 1009 + sample_index * 9176)
    return rng.standard_normal((action_horizon, action_dim), dtype=np.float32)


def infer_one_sample(policy, obs: dict[str, Any], noise):
    return policy.infer(obs, noise=noise)


def compute_native_validation_loss(
    policy, train_config, episode: dict[str, Any], anchor_index: int, seed: int, num_samples: int
) -> dict[str, Any]:
    if num_samples <= 0:
        return {"available": False, "reason": "disabled"}
    try:
        import jax
        import jax.numpy as jnp

        from openpi import transforms as openpi_transforms
        from openpi.cotrain import eval as cotrain_eval
        from openpi.models import model as openpi_model
    except Exception as exc:
        return {"available": False, "reason": f"import_failed: {type(exc).__name__}: {exc}"}

    try:
        data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
        sample = standardized_training_sample(
            episode,
            anchor_index,
            train_config.model.action_horizon,
            train_config.model.action_dim,
        )
        transform = openpi_transforms.compose(
            [*data_config.data_transforms.inputs, *data_config.model_transforms.inputs]
        )
        transformed = transform(sample)
        batch = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], transformed)
        observation = openpi_model.Observation.from_dict(batch)
        actions = batch["actions"]
        rng = jax.random.key(seed)
        fixed = float(jax.device_get(cotrain_eval._flow_loss(policy._model, rng, observation, actions)))
        values = []
        for i in range(num_samples):
            sub_rng = jax.random.fold_in(rng, i)
            values.append(float(jax.device_get(cotrain_eval._flow_loss(policy._model, sub_rng, observation, actions))))
        return {
            "available": True,
            "definition": "openpi.cotrain.eval._flow_loss",
            "fixed": fixed,
            "multi_sample_mean": float(sum(values) / len(values)),
            "num_samples": num_samples,
        }
    except Exception as exc:
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}


def resolved_path_summary(args: argparse.Namespace) -> dict[str, str | None]:
    return {
        "openpi_root": str(args.openpi_root),
        "checkpoint_dir": str(args.checkpoint_dir),
        "norm_stats_path": str(args.norm_stats_path),
        "dataset_dir": str(args.dataset_dir),
        "output_json": str(args.output_json) if args.output_json else None,
        "output_csv": str(args.output_csv) if args.output_csv else None,
    }


def evaluate(args: argparse.Namespace, train_config, norm_stats) -> dict[str, Any]:
    import numpy as np

    metadata = tfds_metadata(args.dataset_dir)
    if args.split not in metadata["splits"]:
        raise ValueError(f"Requested split {args.split!r} not present in dataset: {sorted(metadata['splits'])}")

    policy = create_policy(args, train_config, norm_stats)
    if args.actions_per_inference > train_config.model.action_horizon:
        raise ValueError(
            f"--actions-per-inference ({args.actions_per_inference}) exceeds model horizon "
            f"({train_config.model.action_horizon})"
        )

    records: list[dict[str, Any]] = []
    anchor_metrics: list[dict[str, Any]] = []
    episode_metrics: dict[str, list[dict[str, Any]]] = {}
    task_metrics: dict[str, list[dict[str, Any]]] = {}
    anchor_ordinal = 0

    for episode in read_validation_episodes(args):
        episode_key = str(episode["episode_index"])
        for anchor_index in anchor_indices(len(episode["states"]), args.stride, args.anchors_per_episode):
            state = piper14_from_action(episode["states"][anchor_index], "state")
            obs = standardized_policy_observation(episode, anchor_index, train_config.model.action_dim)
            target = target_action_chunk(episode["actions"], anchor_index, args.actions_per_inference)
            samples = []
            for sample_index in range(args.num_samples):
                noise = sample_noise(
                    args.seed,
                    anchor_ordinal,
                    sample_index,
                    train_config.model.action_horizon,
                    train_config.model.action_dim,
                )
                result = infer_one_sample(policy, obs, noise)
                native = piper14_from_action(
                    np.asarray(result["actions"], dtype=np.float32)[: args.actions_per_inference],
                    "native prediction",
                )
                pred_absolute = native_to_absolute_actions(result["actions"], state, args.actions_per_inference)
                samples.append(
                    {
                        "sample_index": sample_index,
                        "native_prediction": native,
                        "absolute_prediction": pred_absolute,
                        "absolute_error": np.abs(pred_absolute - target),
                        "metrics": metric_dict(pred_absolute, target, state),
                    }
                )
            summary = sample_summary(samples)
            native_loss = compute_native_validation_loss(
                policy, train_config, episode, anchor_index, args.seed + anchor_ordinal, args.native_val_loss_samples
            )
            record = {
                "task": episode["task"],
                "episode": episode["episode_index"],
                "anchor": anchor_index,
                "frame": anchor_index,
                "state": state,
                "target_absolute_action": target,
                "samples": samples,
                "sample_summary": summary,
            }
            if native_loss.get("available"):
                record["native_validation_loss"] = native_loss
            else:
                record["native_validation_loss_unavailable"] = native_loss
                record["training_loss_note"] = (
                    "Inference MAE/MSE are open-loop inference metrics and are not the training validation loss."
                )
            records.append(record)
            anchor_metrics.append(summary["anchor"])
            episode_metrics.setdefault(episode_key, []).append(summary["anchor"])
            task_metrics.setdefault(episode["task"], []).append(summary["anchor"])
            anchor_ordinal += 1

    if not records:
        raise RuntimeError("No validation anchors evaluated. Check split, episode count, and task filter.")

    return {
        "run": {
            "paths": resolved_path_summary(args),
            "config_name": args.config_name,
            "split": args.split,
            "episodes": args.episodes,
            "anchors_per_episode": args.anchors_per_episode,
            "stride": args.stride,
            "actions_per_inference": args.actions_per_inference,
            "seed": args.seed,
            "num_samples": args.num_samples,
            "best_of_n_note": "best_of_n is diagnostic only and does not represent single deployment performance.",
        },
        "source_evidence": SOURCE_EVIDENCE,
        "tfds_metadata": metadata,
        "config": config_summary(train_config),
        "records": records,
        "summaries": {
            "split": summarize_metric_dicts(anchor_metrics),
            "episodes": {key: summarize_metric_dicts(value) for key, value in episode_metrics.items()},
            "tasks": {key: summarize_metric_dicts(value) for key, value in task_metrics.items()},
        },
    }


def to_jsonable(value: Any) -> Any:
    import numpy as np

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(data), indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task",
        "episode",
        "anchor",
        "frame",
        "sample_index",
        "inference_mae",
        "inference_mse",
        "left_joints_mae",
        "left_gripper_mae",
        "right_joints_mae",
        "right_gripper_mae",
        "max_absolute_error",
        "direction_match",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            for sample in record["samples"]:
                metrics = sample["metrics"]
                writer.writerow(
                    {
                        "task": record["task"],
                        "episode": record["episode"],
                        "anchor": record["anchor"],
                        "frame": record["frame"],
                        "sample_index": sample["sample_index"],
                        **{key: metrics[key] for key in fieldnames[5:]},
                    }
                )


def print_json(data: dict[str, Any]) -> None:
    print(json.dumps(to_jsonable(data), indent=2, ensure_ascii=False))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_device(args.device)
    try:
        resolve_required_paths(args)
        require_positive_args(args)
        if args.metadata_only:
            require_file(args.dataset_dir / "dataset_info.json", "dataset_info.json")
            require_file(args.dataset_dir / "features.json", "features.json")
            print_json(
                {
                    "metadata_only": True,
                    "paths": resolved_path_summary(args),
                    "dependency_import_status": import_status(),
                    "tfds_metadata": tfds_metadata(args.dataset_dir),
                }
            )
            return 0

        static = validate_static(args)
        if args.validate_only:
            print_json({"validate_only": True, **static})
            return 0

        train_config, norm_stats = load_config_and_norm_stats(args)
        result = evaluate(args, train_config, norm_stats)
        if args.output_json:
            write_json(args.output_json, result)
        if args.output_csv:
            write_csv(args.output_csv, result["records"])
        print_json(result)
        return 0
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
