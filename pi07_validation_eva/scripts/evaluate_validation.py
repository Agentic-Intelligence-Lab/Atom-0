#!/usr/bin/env python3
"""Offline validation trajectory evaluator for Piper OpenPI JAX checkpoints."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import html
import importlib.metadata as importlib_metadata
import json
import logging
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable

import numpy as np

FORBIDDEN_IMPORT_PREFIXES = (
    "PiperRobot",
    "PiperMotorsBus",
    "piper_sdk",
    "RealSense",
)
IMAGE_SLOTS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
RAW_IMAGE_KEYS = {
    "base_0_rgb": "cam_high",
    "left_wrist_0_rgb": "cam_left_wrist",
    "right_wrist_0_rgb": "cam_right_wrist",
}
PIPER_DATASET_ID = "piper30"
PROMPT_PREFIX = "Action Mode: joint. "
DEFAULT_TARGET_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OPENPI_ROOT = Path("/mnt/workspace/xule/pi07_reproduction")
DEFAULT_PYTHON = "/mnt/data/xule/pi07_reproduction/.venv/bin/python"
DEFAULT_NORM_STATS = DEFAULT_OPENPI_ROOT / "assets/cotrain_real_only/piper30"
DEFAULT_DATASET_DIR = Path(
    "/mnt/data/RLDS/realworld_piper_task_split/"
    "piper_s14_a14_fps30_c4_ee_pose_cam_front_cam_high_cam_left_wrist_cam_right_wrist/"
    "realworld_piper_infidata/1.1.0"
)


def compute_uniform_anchors(trajectory_length: int, horizon: int, limit: int) -> list[int]:
    """Uniform deterministic anchors over valid inclusive starts, with no tail padding."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if limit <= 0:
        return []
    valid_count = int(trajectory_length) - int(horizon) + 1
    if valid_count <= 0:
        return []
    count = min(int(limit), valid_count)
    raw = np.linspace(0, valid_count - 1, num=count)
    anchors: list[int] = []
    seen: set[int] = set()
    for value in raw:
        idx = int(np.rint(value))
        idx = max(0, min(valid_count - 1, idx))
        if idx not in seen:
            anchors.append(idx)
            seen.add(idx)
    return anchors


def build_anchor_records(
    episodes: Iterable[dict[str, Any]], *, split: str, horizon: int, anchors_per_episode: int
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for ep in episodes:
        traj_len = int(ep["trajectory_length"])
        anchors = compute_uniform_anchors(traj_len, horizon, anchors_per_episode)
        for anchor in anchors:
            records.append(
                {
                    "split": split,
                    "episode_index": int(ep["episode_index"]),
                    "task": str(ep.get("task", "")),
                    "trajectory_length": traj_len,
                    "anchor_index": int(anchor),
                    "horizon": int(horizon),
                    "target_start": int(anchor),
                    "target_end": int(anchor + horizon),
                }
            )
    return records


def normal_swapped_mae(pred: Any, target: Any) -> dict[str, float]:
    pred_arr = np.asarray(pred, dtype=np.float64)[..., :14]
    target_arr = np.asarray(target, dtype=np.float64)[..., :14]
    normal = float(np.mean(np.abs(pred_arr - target_arr)))
    swapped_target = np.concatenate([target_arr[..., 7:14], target_arr[..., 0:7]], axis=-1)
    swapped = float(np.mean(np.abs(pred_arr - swapped_target)))
    return {"normal_mae": normal, "swapped_mae": swapped}


def decode_text(value: Any) -> str:
    arr = np.asarray(value)
    item = arr.item() if arr.shape == () else arr.reshape(-1)[0]
    if isinstance(item, bytes):
        return item.decode("utf-8", errors="replace")
    return str(item)


def json_default(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return repr(value)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=json_default) + "\n")


def run_cmd(cmd: list[str], cwd: Path | None = None) -> dict[str, Any]:
    try:
        proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, text=True, capture_output=True, timeout=120)
        return {"cmd": cmd, "returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}
    except Exception as exc:
        return {"cmd": cmd, "error": repr(exc)}


def add_openpi_to_path(openpi_root: Path) -> None:
    src = openpi_root / "src"
    for p in (str(src), str(openpi_root)):
        if p not in sys.path:
            sys.path.insert(0, p)


def disable_tf_gpu():
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    return tf


def check_forbidden_modules() -> list[str]:
    loaded = []
    for name in sys.modules:
        if any(name == prefix or name.startswith(prefix + ".") for prefix in FORBIDDEN_IMPORT_PREFIXES):
            loaded.append(name)
    return sorted(loaded)


def package_versions() -> dict[str, str]:
    packages = [
        "jax",
        "jaxlib",
        "flax",
        "orbax-checkpoint",
        "tensorflow",
        "tensorflow-datasets",
        "numpy",
        "ml-dtypes",
        "pandas",
        "matplotlib",
    ]
    out = {}
    for pkg in packages:
        try:
            out[pkg] = importlib_metadata.version(pkg)
        except Exception as exc:
            out[pkg] = f"unavailable: {exc!r}"
    return out


def jax_device_manifest() -> dict[str, Any]:
    try:
        import jax
        import jaxlib.version as jaxlib_version

        devices = [str(d) for d in jax.devices()]
        gpu_error = None
        try:
            gpu_devices = [str(d) for d in jax.devices("gpu")]
        except Exception as exc:
            gpu_devices = []
            gpu_error = repr(exc)
        return {
            "default_backend": jax.default_backend(),
            "devices": devices,
            "gpu_devices": gpu_devices,
            "gpu_error": gpu_error,
            "jaxlib_cuda_version": getattr(jaxlib_version, "__cuda_version__", None),
            "jaxlib_cudnn_version": getattr(jaxlib_version, "__cudnn_version__", None),
        }
    except Exception as exc:
        return {"error": repr(exc)}


def parse_wandb_config(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return data
    text = path.read_text(encoding="utf-8", errors="replace")
    data["raw_head"] = text[:8000]
    try:
        import yaml

        parsed = yaml.safe_load(text)
        data["parsed"] = parsed
    except Exception as exc:
        data["parse_error"] = repr(exc)
    return data


def find_wandb_run(openpi_root: Path, exp_name: str, config_name: str) -> dict[str, Any]:
    candidates = []
    wandb_root = openpi_root / "wandb"
    if not wandb_root.exists():
        return {"wandb_root": str(wandb_root), "candidates": []}
    for config_path in wandb_root.glob("run-*/files/config.yaml"):
        try:
            text = config_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if exp_name in text and config_name in text:
            run_dir = config_path.parents[1]
            candidates.append(
                {
                    "run_dir": str(run_dir),
                    "config_yaml": str(config_path),
                    "output_log": str(run_dir / "files" / "output.log"),
                    "metadata_json": str(run_dir / "files" / "wandb-metadata.json"),
                    "debug_log": str(run_dir / "logs" / "debug.log"),
                }
            )
    candidates.sort(key=lambda x: x["run_dir"])
    return {"wandb_root": str(wandb_root), "candidates": candidates, "selected": candidates[-1] if candidates else None}


def checkpoint_manifest(checkpoint_root: Path, steps: Iterable[int]) -> dict[str, Any]:
    out: dict[str, Any] = {"checkpoint_root": str(checkpoint_root), "steps": {}}
    required_top = ["_CHECKPOINT_METADATA", "params", "train_state"]
    required_params = ["_METADATA", "_sharding", "manifest.ocdbt"]
    for step in steps:
        step_dir = checkpoint_root / str(step)
        params_dir = step_dir / "params"
        entry = {
            "path": str(step_dir),
            "exists": step_dir.is_dir(),
            "top_level": sorted([p.name for p in step_dir.iterdir()]) if step_dir.is_dir() else [],
            "required_top_present": {name: (step_dir / name).exists() for name in required_top},
            "params_required_present": {name: (params_dir / name).exists() for name in required_params},
            "params_file_count": sum(1 for p in params_dir.rglob("*") if p.is_file()) if params_dir.exists() else 0,
        }
        entry["complete_orbax_params"] = bool(
            entry["exists"]
            and entry["required_top_present"].get("params")
            and all(entry["params_required_present"].values())
            and entry["params_file_count"] > 0
        )
        out["steps"][str(step)] = entry
    return out


def load_tfds_metadata(dataset_dir: Path) -> dict[str, Any]:
    info_path = dataset_dir / "dataset_info.json"
    features_path = dataset_dir / "features.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    features = json.loads(features_path.read_text(encoding="utf-8")) if features_path.exists() else {}
    split_rows = {}
    for split in info.get("splits", []):
        name = split.get("name")
        shard_lengths = [int(x) for x in split.get("shardLengths", [])]
        split_rows[name] = {
            "num_examples": sum(shard_lengths),
            "num_shards": len(shard_lengths),
            "num_bytes": int(split.get("numBytes", 0)),
        }
    shard_counts = {
        split: len(list(dataset_dir.glob(f"*-{split}.tfrecord-*"))) for split in ("train", "seen_test", "unseen_test")
    }
    return {
        "dataset_dir": str(dataset_dir),
        "dataset_info_exists": info_path.exists(),
        "features_exists": features_path.exists(),
        "name": info.get("name"),
        "version": info.get("version"),
        "splits": split_rows,
        "tfrecord_shard_counts": shard_counts,
        "features_head": features,
    }


def env_manifest(openpi_root: Path) -> dict[str, Any]:
    return {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "packages": package_versions(),
        "jax": jax_device_manifest(),
        "nvidia_smi": run_cmd(["bash", "-lc", "command -v nvidia-smi && nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader"]),
        "cuda_device_files": {p: Path(p).exists() for p in ["/dev/nvidia0", "/proc/driver/nvidia/version"]},
        "pip_check": run_cmd([sys.executable, "-m", "pip", "check"]),
        "git_commit_current": run_cmd(["git", "rev-parse", "HEAD"], cwd=openpi_root),
        "git_status_short": run_cmd(["git", "status", "--short"], cwd=openpi_root),
        "git_diff_stat": run_cmd(["git", "diff", "--stat"], cwd=openpi_root),
        "git_diff_config_head": run_cmd(["git", "diff", "--", "src/openpi/cotrain/config.py"], cwd=openpi_root),
    }


def load_train_config(openpi_root: Path, config_name: str, assets_base_dir: Path | None = None):
    add_openpi_to_path(openpi_root)
    old_cwd = Path.cwd()
    try:
        os.chdir(openpi_root)
        from openpi.cotrain import config as cotrain_config

        cfg = cotrain_config.get_config(config_name)
    finally:
        os.chdir(old_cwd)
    if assets_base_dir is not None:
        cfg = dataclasses.replace(cfg, assets_base_dir=str(assets_base_dir))
    return cfg


def assets_base_from_norm_stats(norm_stats_path: Path) -> Path:
    # /.../assets/<config_name>/<dataset_id> -> /.../assets
    return norm_stats_path.parent.parent


def norm_stats_summary(norm_stats_path: Path) -> dict[str, Any]:
    files = sorted(str(p.relative_to(norm_stats_path)) for p in norm_stats_path.rglob("*") if p.is_file()) if norm_stats_path.exists() else []
    summary = {"path": str(norm_stats_path), "exists": norm_stats_path.exists(), "files": files}
    try:
        add_openpi_to_path(DEFAULT_OPENPI_ROOT)
        import openpi.shared.normalize as normalize

        stats = normalize.load(str(norm_stats_path))
        summary["keys"] = sorted(stats.keys())
        summary["shapes"] = {
            key: {name: list(np.asarray(getattr(value, name)).shape) if getattr(value, name) is not None else None for name in ("mean", "std", "q01", "q99")}
            for key, value in stats.items()
        }
    except Exception as exc:
        summary["load_error"] = repr(exc)
    return summary


def extract_wandb_execution(parsed_config: dict[str, Any]) -> dict[str, Any]:
    value = parsed_config.get("_wandb", {}).get("value", {}) if parsed_config else {}
    executions = value.get("e", {})
    if isinstance(executions, dict) and executions:
        first_key = sorted(executions)[0]
        info = dict(executions[first_key] or {})
        info["writer_key"] = first_key
    else:
        info = {}
    return {
        "args": info.get("args"),
        "executable": info.get("executable"),
        "program": info.get("program"),
        "root": info.get("root"),
        "python": info.get("python") or value.get("python_version"),
        "git_commit": (info.get("git") or {}).get("commit"),
        "git_remote": (info.get("git") or {}).get("remote"),
        "cuda_version": info.get("cudaVersion"),
        "gpu": info.get("gpu"),
        "gpu_count": info.get("gpu_count"),
        "host": info.get("host"),
        "started_at": info.get("startedAt"),
        "writer_key": info.get("writer_key"),
    }

def collect_training_metadata(args: argparse.Namespace) -> dict[str, Any]:
    train_config = load_train_config(args.openpi_root, args.config_name, assets_base_from_norm_stats(args.norm_stats_path))
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    ds = data_config.datasets[0]
    exp_name = args.checkpoint_dir.parent.name
    wandb = find_wandb_run(args.openpi_root, exp_name, args.config_name)
    selected = wandb.get("selected") or {}
    wandb_config = parse_wandb_config(Path(selected.get("config_yaml", ""))) if selected else {}
    wandb_metadata_path = Path(selected.get("metadata_json", "")) if selected else Path("")
    wandb_metadata = {}
    if selected and wandb_metadata_path.exists():
        try:
            wandb_metadata = json.loads(wandb_metadata_path.read_text(encoding="utf-8"))
        except Exception as exc:
            wandb_metadata = {"error": repr(exc)}

    wandb_execution = extract_wandb_execution(wandb_config.get("parsed", {}))

    metadata = {
        "source_policy": "confirmed from H800 filesystem, OpenPI config imports, wandb run files, checkpoint metadata, and TFDS metadata",
        "wandb": {"match": wandb, "config": wandb_config, "metadata": wandb_metadata, "execution": wandb_execution},
        "training_command": wandb_execution.get("args"),
        "training_executable": wandb_execution.get("executable"),
        "training_program": wandb_execution.get("program"),
        "training_root": wandb_execution.get("root"),
        "wandb_git_commit": wandb_execution.get("git_commit"),
        "wandb_id": selected.get("run_dir", "").split("-")[-1] if selected else None,
        "config_name": args.config_name,
        "config_source_file": str(args.openpi_root / "src/openpi/cotrain/config.py"),
        "dataset": {
            "dataset_id": ds.uid,
            "name": ds.name,
            "version": ds.version,
            "builder_dir": ds.builder_dir,
            "rlds_data_dir": data_config.rlds_data_dir,
            "train_split": ds.train_split,
            "val_splits": ds.val_splits,
            "restructure_name": ds.restructure_name,
            "image_keys": RAW_IMAGE_KEYS,
        },
        "norm_stats_path": str(args.norm_stats_path),
        "norm_stats": norm_stats_summary(args.norm_stats_path),
        "model": {
            "action_dim_config": int(train_config.model.action_dim),
            "dataset_action_dim": int(ds.action_dim),
            "action_horizon": int(train_config.model.action_horizon),
            "model_type": str(train_config.model.model_type),
            "pi05": bool(getattr(train_config.model, "pi05", False)),
            "delta_action_mask_dims": ds.delta_action_mask_dims,
            "delta_action_mask": list(make_delta_mask(ds.delta_action_mask_dims)),
            "action_semantics": "dataset action is absolute joint targets; training transform converts left/right 6 joint dims to delta and keeps grippers absolute; evaluator unnormalizes and applies native AbsoluteActions for open-loop absolute targets",
            "left_right_order": "dims 0:6 left joints, dim 6 left gripper, dims 7:13 right joints, dim 13 right gripper, confirmed by config mask (6,-1,6,-1) and Piper 14D schema comments",
        },
        "checkpoint_format": checkpoint_manifest(args.checkpoint_dir.parent, [20000, 25000, 30000]),
        "tfds_metadata": load_tfds_metadata(args.dataset_dir),
        "environment": env_manifest(args.openpi_root),
        "current_openpi_root": str(args.openpi_root),
        "checkpoint_dir": str(args.checkpoint_dir),
    }
    return metadata


def make_delta_mask(delta_dims: tuple[int, ...] | list[int] | None) -> tuple[bool, ...]:
    if delta_dims is None:
        return ()
    mask: list[bool] = []
    for dim in delta_dims:
        if dim > 0:
            mask.extend([True] * int(dim))
        else:
            mask.extend([False] * int(-dim))
    return tuple(mask)


def iter_episode_summaries(dataset_dir: Path, split: str, limit: int | None = None) -> list[dict[str, Any]]:
    tf = disable_tf_gpu()
    del tf
    import tensorflow_datasets as tfds

    builder = tfds.builder_from_directory(str(dataset_dir))
    ds = builder.as_dataset(split=split, shuffle_files=False)
    rows: list[dict[str, Any]] = []
    for i, ex in enumerate(ds):
        meta = ex["episode_metadata"]
        rows.append(
            {
                "split": split,
                "split_ordinal": i,
                "episode_index": int(meta["episode_index"].numpy()),
                "trajectory_length": int(meta["num_frames"].numpy()),
                "task": decode_text(meta["task"].numpy()),
            }
        )
        if limit is not None and len(rows) >= limit:
            break
    return rows


def ensure_anchor_manifest(path: Path, dataset_dir: Path, split: str, horizon: int, anchors_per_episode: int, max_episodes: int | None) -> list[dict[str, Any]]:
    if path.exists():
        return read_jsonl(path)
    episodes = iter_episode_summaries(dataset_dir, split, max_episodes)
    records = build_anchor_records(episodes, split=split, horizon=horizon, anchors_per_episode=anchors_per_episode)
    write_jsonl(path, records)
    return records


def load_episode_by_index(dataset_dir: Path, split: str, wanted_episode_index: int) -> dict[str, Any] | None:
    tf = disable_tf_gpu()
    del tf
    import tensorflow_datasets as tfds

    builder = tfds.builder_from_directory(str(dataset_dir))
    for ex in builder.as_dataset(split=split, shuffle_files=False):
        ep_idx = int(ex["episode_metadata"]["episode_index"].numpy())
        if ep_idx == wanted_episode_index:
            return extract_episode(ex)
    return None


def extract_episode(ex: Any) -> dict[str, Any]:
    meta = ex["episode_metadata"]
    steps = list(ex["steps"].as_numpy_iterator())
    actions = np.stack([s["action"] for s in steps], axis=0).astype(np.float32)
    states = np.stack([s["observation"]["state"] for s in steps], axis=0).astype(np.float32)
    images = {slot: np.stack([s["observation"]["images"][raw] for s in steps], axis=0) for slot, raw in RAW_IMAGE_KEYS.items()}
    tasks = [decode_text(s["task"]) for s in steps]
    return {
        "episode_index": int(meta["episode_index"].numpy()),
        "trajectory_length": len(steps),
        "task": decode_text(meta["task"].numpy()) if "task" in meta else tasks[0],
        "actions": actions,
        "states": states,
        "images": images,
        "tasks": tasks,
    }


def raw_sample_from_episode(episode: dict[str, Any], anchor: int, horizon: int, include_actions: bool) -> dict[str, Any]:
    sample = {
        "state": episode["states"][anchor].copy(),
        "image": {slot: episode["images"][slot][anchor].copy() for slot in IMAGE_SLOTS},
        "image_mask": {slot: np.asarray(True, dtype=np.bool_) for slot in IMAGE_SLOTS},
        "prompt": episode["tasks"][anchor] if episode.get("tasks") else episode.get("task", ""),
        "prompt_prefix": PROMPT_PREFIX,
        "dataset_id": PIPER_DATASET_ID,
    }
    if include_actions:
        sample["actions"] = episode["actions"][anchor : anchor + horizon].copy()
    return sample


def transform_for_model(sample: dict[str, Any], train_config: Any, norm_stats: dict[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
    from openpi import transforms as openpi_transforms
    from openpi.models import model as model_lib

    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    data = sample
    for transform in data_config.data_transforms.inputs:
        data = transform(data)
    # Built-in global Normalize is intentionally skipped for cotrain; DispatchNormalize did it.
    for transform in data_config.model_transforms.inputs:
        data = transform(data)
    actions = data.pop("actions") if "actions" in data else None
    data = {k: v for k, v in data.items() if k not in ("dataset_id", "prompt", "prompt_prefix")}
    batch = _batch_tree(data)
    obs = model_lib.Observation.from_dict(batch)
    batched_actions = None if actions is None else np.asarray(actions, dtype=np.float32)[None, ...]
    return obs, batched_actions, data


def _batch_tree(tree: Any) -> Any:
    import jax
    import jax.numpy as jnp

    return jax.tree.map(lambda x: jnp.asarray(x)[None, ...], tree)


def load_norm_stats(openpi_root: Path, norm_stats_path: Path) -> dict[str, Any]:
    add_openpi_to_path(openpi_root)
    import openpi.shared.normalize as normalize

    return normalize.load(str(norm_stats_path))


def create_policy_and_postprocess(args: argparse.Namespace, norm_stats: dict[str, Any]):
    from openpi import transforms as openpi_transforms
    from openpi.cotrain import config as cotrain_config
    from openpi.policies import policy_config

    train_config = dataclasses.replace(cotrain_config.get_config(args.config_name), assets_base_dir=str(assets_base_from_norm_stats(args.norm_stats_path)))
    policy = policy_config.create_trained_policy(
        train_config,
        args.checkpoint_dir,
        sample_kwargs={"num_steps": 10},
        norm_stats={},
    )
    delta_mask = make_delta_mask(train_config.data.datasets[0].delta_action_mask_dims)
    unnormalize = openpi_transforms.Unnormalize(norm_stats, use_quantiles=True)
    absolute = openpi_transforms.AbsoluteActions(delta_mask)

    def infer_absolute(raw_obs: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        out = policy.infer(raw_obs)
        native = unnormalize({"state": out["state"].copy(), "actions": out["actions"].copy()})
        native = absolute(native)
        return np.asarray(native["actions"])[..., :14], out

    return policy, infer_absolute


def load_model_for_flow(args: argparse.Namespace):
    import jax.numpy as jnp
    from openpi.cotrain import config as cotrain_config
    from openpi.models import model as model_lib

    train_config = dataclasses.replace(cotrain_config.get_config(args.config_name), assets_base_dir=str(assets_base_from_norm_stats(args.norm_stats_path)))
    model = train_config.model.load(model_lib.restore_params(args.checkpoint_dir / "params", dtype=jnp.bfloat16))
    model.eval()
    return train_config, model


def evaluate_flow_loss(args: argparse.Namespace, records: list[dict[str, Any]], norm_stats: dict[str, Any]) -> list[dict[str, Any]]:
    import jax
    import jax.numpy as jnp

    train_config, model = load_model_for_flow(args)
    rows: list[dict[str, Any]] = []
    base_rng = jax.random.key(args.seed)
    episodes_cache: dict[int, dict[str, Any]] = {}
    start = time.monotonic()
    for rec_i, rec in enumerate(records):
        ep_idx = int(rec["episode_index"])
        if ep_idx not in episodes_cache:
            ep = load_episode_by_index(args.dataset_dir, args.split, ep_idx)
            if ep is None:
                continue
            episodes_cache[ep_idx] = ep
        ep = episodes_cache[ep_idx]
        sample = raw_sample_from_episode(ep, int(rec["anchor_index"]), int(rec["horizon"]), include_actions=True)
        obs, actions, _ = transform_for_model(sample, train_config, norm_stats)
        vals = []
        for k in range(args.flow_loss_samples):
            rng = jax.random.fold_in(base_rng, rec_i * 1000 + k)
            out = model.compute_loss(rng, obs, jnp.asarray(actions), train=False)
            flow = out["flow"] if isinstance(out, dict) else out
            vals.append(float(jax.device_get(jnp.mean(flow))))
        rows.append({**rec, "native_flow_matching_validation_loss": float(np.mean(vals)), "flow_loss_samples": len(vals)})
    logging.info("Flow loss evaluated %d anchors in %.2fs", len(rows), time.monotonic() - start)
    return rows


def action_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    pred = np.asarray(pred, dtype=np.float64)[..., :14]
    target = np.asarray(target, dtype=np.float64)[..., :14]
    err = pred - target
    finite = np.isfinite(pred) & np.isfinite(target)
    abs_err = np.abs(err)
    per_dim = np.nanmean(abs_err, axis=tuple(range(abs_err.ndim - 1)))
    diff_pred = np.diff(pred, axis=0)
    diff_target = np.diff(target, axis=0)
    direction_match = float(np.mean(np.sign(diff_pred) == np.sign(diff_target))) if diff_pred.size else float("nan")
    out = {
        "open_loop_action_mae": float(np.nanmean(abs_err)),
        "open_loop_action_rmse": float(np.sqrt(np.nanmean(err**2))),
        "left_joint_mae": float(np.nanmean(abs_err[..., 0:6])),
        "left_gripper_mae": float(np.nanmean(abs_err[..., 6])),
        "right_joint_mae": float(np.nanmean(abs_err[..., 7:13])),
        "right_gripper_mae": float(np.nanmean(abs_err[..., 13])),
        "per_dimension_mae": per_dim.tolist(),
        "max_absolute_error": float(np.nanmax(abs_err)),
        "movement_direction_match": direction_match,
        "nan_inf_rate": float(1.0 - np.mean(finite)),
    }
    out.update(normal_swapped_mae(pred, target))
    return out


def evaluate_open_loop(args: argparse.Namespace, records: list[dict[str, Any]], norm_stats: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    add_openpi_to_path(args.openpi_root)
    policy, infer_absolute = create_policy_and_postprocess(args, norm_stats)
    del policy
    rows: list[dict[str, Any]] = []
    episodes_cache: dict[int, dict[str, Any]] = {}
    timings = []
    start = time.monotonic()
    for rec in records:
        ep_idx = int(rec["episode_index"])
        if ep_idx not in episodes_cache:
            ep = load_episode_by_index(args.dataset_dir, args.split, ep_idx)
            if ep is None:
                continue
            episodes_cache[ep_idx] = ep
        ep = episodes_cache[ep_idx]
        anchor = int(rec["anchor_index"])
        sample = raw_sample_from_episode(ep, anchor, args.actions_per_inference, include_actions=False)
        t0 = time.monotonic()
        pred, raw_out = infer_absolute(sample)
        timings.append(time.monotonic() - t0)
        pred8 = pred[: args.actions_per_inference, :14]
        target = ep["actions"][anchor : anchor + args.actions_per_inference, :14]
        metrics = action_metrics(pred8, target)
        rows.append({**rec, **metrics, "policy_infer_ms": float(raw_out.get("policy_timing", {}).get("infer_ms", math.nan))})
    total = time.monotonic() - start
    perf = {
        "anchors": len(rows),
        "total_seconds": total,
        "anchors_per_second": len(rows) / total if total > 0 else 0.0,
        "median_wall_infer_seconds": float(np.median(timings)) if timings else None,
    }
    return rows, perf


def bootstrap_ci(values: list[float], seed: int, reps: int = 1000) -> dict[str, float | None]:
    vals = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if vals.size == 0:
        return {"mean": None, "ci95_low": None, "ci95_high": None, "n_episodes": 0}
    rng = np.random.default_rng(seed)
    boot = [float(np.mean(rng.choice(vals, size=vals.size, replace=True))) for _ in range(reps)]
    return {
        "mean": float(np.mean(vals)),
        "ci95_low": float(np.percentile(boot, 2.5)),
        "ci95_high": float(np.percentile(boot, 97.5)),
        "n_episodes": int(vals.size),
    }


def save_tables_and_summaries(
    args: argparse.Namespace,
    output_dir: Path,
    open_rows: list[dict[str, Any]],
    flow_rows: list[dict[str, Any]],
    perf: dict[str, Any],
    anchor_request: dict[str, Any],
) -> dict[str, Any]:
    import pandas as pd

    output_dir.mkdir(parents=True, exist_ok=True)
    open_df = pd.DataFrame(open_rows)
    flow_df = pd.DataFrame(flow_rows)
    if not flow_df.empty and not open_df.empty:
        anchor_df = open_df.merge(
            flow_df[["episode_index", "anchor_index", "native_flow_matching_validation_loss"]],
            on=["episode_index", "anchor_index"],
            how="left",
        )
    else:
        anchor_df = open_df if not open_df.empty else flow_df
    anchor_df.to_csv(output_dir / "anchor_metrics.csv", index=False)

    metric_cols = [
        "open_loop_action_mae",
        "open_loop_action_rmse",
        "left_joint_mae",
        "right_joint_mae",
        "left_gripper_mae",
        "right_gripper_mae",
        "max_absolute_error",
        "movement_direction_match",
        "nan_inf_rate",
        "normal_mae",
        "swapped_mae",
        "native_flow_matching_validation_loss",
    ]
    present = [c for c in metric_cols if c in anchor_df.columns]
    if not anchor_df.empty and present:
        episode_df = anchor_df.groupby(["episode_index", "task"], as_index=False)[present].mean(numeric_only=True)
        episode_counts = anchor_df.groupby(["episode_index", "task"], as_index=False).size().rename(columns={"size": "actual_anchor_count"})
        episode_df = episode_df.merge(episode_counts, on=["episode_index", "task"], how="left")
        task_df = episode_df.groupby("task", as_index=False)[present].mean(numeric_only=True)
        task_counts = anchor_df.groupby("task", as_index=False).agg(episode_count=("episode_index", "nunique"), anchor_count=("anchor_index", "count"))
        task_df = task_df.merge(task_counts, on="task", how="left")
    else:
        episode_df = pd.DataFrame()
        task_df = pd.DataFrame()
    episode_df.to_csv(output_dir / "episode_metrics.csv", index=False)
    task_df.to_csv(output_dir / "task_metrics.csv", index=False)

    normal_swapped = anchor_df[[c for c in ["episode_index", "task", "anchor_index", "normal_mae", "swapped_mae"] if c in anchor_df.columns]] if not anchor_df.empty else pd.DataFrame()
    normal_swapped.to_csv(output_dir / "normal_vs_swapped.csv", index=False)

    per_dim = []
    if "per_dimension_mae" in anchor_df.columns:
        arrs = [np.asarray(v if isinstance(v, list) else json.loads(v), dtype=float) for v in anchor_df["per_dimension_mae"]]
        if arrs:
            means = np.mean(np.stack(arrs), axis=0)
            per_dim = [{"dimension": i, "mae": float(v)} for i, v in enumerate(means)]
    pd.DataFrame(per_dim).to_csv(output_dir / "per_dimension_mae.csv", index=False)

    open_summary = {
        "metrics": {c: float(anchor_df[c].mean()) for c in present if c.startswith("open_loop") or c.endswith("mae") or c in ("max_absolute_error", "movement_direction_match", "nan_inf_rate", "normal_mae", "swapped_mae")}
        if not anchor_df.empty
        else {},
        "episode_bootstrap_ci": {
            c: bootstrap_ci(episode_df[c].tolist(), args.seed) for c in ["open_loop_action_mae", "open_loop_action_rmse"] if c in episode_df.columns
        },
        "performance": perf,
        "anchor_request": anchor_request,
    }
    flow_summary = {
        "native_flow_matching_validation_loss": float(flow_df["native_flow_matching_validation_loss"].mean()) if "native_flow_matching_validation_loss" in flow_df else None,
        "episode_bootstrap_ci": {"native_flow_matching_validation_loss": bootstrap_ci(episode_df["native_flow_matching_validation_loss"].tolist(), args.seed)} if "native_flow_matching_validation_loss" in episode_df else {},
        "strict_reproduction": bool(flow_rows),
        "strict_reproduction_notes": "Uses OpenPI model.compute_loss with the same cotrain data/model transforms, per-dataset DispatchNormalize, delta mask, and full action_horizon anchors. Not a MAE substitute.",
    }
    write_json(output_dir / "open_loop_summary.json", open_summary)
    write_json(output_dir / "flow_loss_summary.json", flow_summary)

    paper = []
    if open_summary["metrics"]:
        paper.append({"metric": "open_loop_action_mae", "value": open_summary["metrics"].get("open_loop_action_mae")})
        paper.append({"metric": "open_loop_action_rmse", "value": open_summary["metrics"].get("open_loop_action_rmse")})
    paper.append({"metric": "native_flow_matching_validation_loss", "value": flow_summary["native_flow_matching_validation_loss"]})
    pd.DataFrame(paper).to_csv(output_dir / "paper_table.csv", index=False)
    make_figures(output_dir, anchor_df, task_df, pd.DataFrame(per_dim), normal_swapped)
    return {"open_loop_summary": open_summary, "flow_loss_summary": flow_summary}


def make_figures(output_dir: Path, anchor_df: Any, task_df: Any, per_dim_df: Any, normal_swapped: Any) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    def empty_plot(name: str, title: str):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "无可用数据", ha="center", va="center")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(fig_dir / name)
        plt.close(fig)

    if not task_df.empty and "native_flow_matching_validation_loss" in task_df:
        task_df.plot.bar(x="task", y="native_flow_matching_validation_loss", legend=False, figsize=(10, 4))
        plt.ylabel("loss")
        plt.tight_layout()
        plt.savefig(fig_dir / "validation_loss_by_task.png")
        plt.close()
    else:
        empty_plot("validation_loss_by_task.png", "Native validation loss by task")

    if not task_df.empty and "open_loop_action_mae" in task_df:
        task_df.plot.bar(x="task", y="open_loop_action_mae", legend=False, figsize=(10, 4))
        plt.ylabel("MAE")
        plt.tight_layout()
        plt.savefig(fig_dir / "action_mae_by_task.png")
        plt.close()
    else:
        empty_plot("action_mae_by_task.png", "Open-loop MAE by task")

    if not per_dim_df.empty:
        per_dim_df.plot.bar(x="dimension", y="mae", legend=False, figsize=(8, 4))
        plt.ylabel("MAE")
        plt.tight_layout()
        plt.savefig(fig_dir / "per_dimension_mae.png")
        plt.close()
    else:
        empty_plot("per_dimension_mae.png", "Per-dimension MAE")

    if not anchor_df.empty and "open_loop_action_mae" in anchor_df:
        anchor_df["open_loop_action_mae"].hist(bins=40, figsize=(8, 4))
        plt.xlabel("MAE")
        plt.ylabel("anchors")
        plt.tight_layout()
        plt.savefig(fig_dir / "error_distribution.png")
        plt.close()
    else:
        empty_plot("error_distribution.png", "Error distribution")

    if not normal_swapped.empty:
        normal_swapped[["normal_mae", "swapped_mae"]].mean().plot.bar(figsize=(6, 4))
        plt.ylabel("MAE")
        plt.tight_layout()
        plt.savefig(fig_dir / "normal_vs_swapped.png")
        plt.close()
    else:
        empty_plot("normal_vs_swapped.png", "Normal vs swapped")



def manifest_report_summary(args: argparse.Namespace) -> dict[str, Any]:
    split_tag = args.split.replace("/", "_")
    files = {
        "open_loop": args.anchor_manifest
        or DEFAULT_TARGET_ROOT / "manifests" / f"{split_tag}_open_loop_h{args.actions_per_inference}_seed{args.seed}.jsonl",
        "flow_loss": DEFAULT_TARGET_ROOT
        / "manifests"
        / f"{split_tag}_flow_loss_MODEL_HORIZON_seed{args.seed}.jsonl",
    }
    out: dict[str, Any] = {}
    for name, path in files.items():
        rows = read_jsonl(path) if path.exists() else []
        out[name] = {
            "path": str(path),
            "exists": path.exists(),
            "actual_anchors": len(rows),
            "episodes": len({r.get("episode_index") for r in rows}),
            "requested_anchors_per_episode": args.anchors_per_episode,
            "short_trajectory_reductions": max(0, args.anchors_per_episode * len({r.get("episode_index") for r in rows}) - len(rows)),
        }
    return out

def generate_report(output_dir: Path, args: argparse.Namespace, metadata: dict[str, Any], summaries: dict[str, Any] | None, validation_status: dict[str, Any]) -> None:
    open_summary = (summaries or {}).get("open_loop_summary", {})
    flow_summary = (summaries or {}).get("flow_loss_summary", {})
    task_csv = output_dir / "task_metrics.csv"
    task_text = task_csv.read_text(encoding="utf-8")[:6000] if task_csv.exists() else "未生成"
    cmd = (
        f"CHECKPOINT_STEP={args.checkpoint_dir.name} bash "
        f"{DEFAULT_TARGET_ROOT}/scripts/run_validation.sh"
    )
    lines = [
        "# Piper JAX validation trajectory 离线评测报告",
        "",
        "## 1. checkpoint信息",
        f"- checkpoint: `{args.checkpoint_dir}`",
        f"- checkpoint完整性: `{metadata.get('checkpoint_format', {}).get('steps', {}).get(args.checkpoint_dir.name, {})}`",
        "",
        "## 2. 数据集和split",
        f"- dataset: `{args.dataset_dir}`",
        f"- split: `{args.split}`",
        f"- seen_test episodes: `{metadata.get('tfds_metadata', {}).get('splits', {}).get('seen_test', {}).get('num_examples')}`",
        f"- unseen_test episodes: `{metadata.get('tfds_metadata', {}).get('splits', {}).get('unseen_test', {}).get('num_examples')}`",
        "",
        "## 3. 评测协议",
        f"- open-loop horizon: `{args.actions_per_inference}`",
        f"- flow-loss horizon: `{metadata.get('model', {}).get('action_horizon')}`",
        f"- anchors_per_episode: `{args.anchors_per_episode}`",
        "- anchor使用有效区间 linspace 均匀采样，不做尾部padding。",
        "",
        "## 4. 实际anchor数量",
        json.dumps(open_summary.get("anchor_request") or manifest_report_summary(args), ensure_ascii=False, indent=2),
        "",
        "## 5. native validation loss",
        json.dumps(flow_summary, ensure_ascii=False, indent=2, default=json_default),
        "",
        "## 6. open-loop MAE/RMSE",
        json.dumps(open_summary.get("metrics", {}), ensure_ascii=False, indent=2, default=json_default),
        "",
        "## 7. 每任务结果",
        "```csv",
        task_text,
        "```",
        "",
        "## 8. 左右臂诊断",
        f"- normal/swapped 汇总见 `{output_dir / 'normal_vs_swapped.csv'}`。",
        "",
        "## 9. 95% CI",
        json.dumps(open_summary.get("episode_bootstrap_ci", {}), ensure_ascii=False, indent=2, default=json_default),
        "",
        "## 10. 异常样本",
        "- 详见 `anchor_metrics.csv` 中 `nan_inf_rate`、`max_absolute_error` 排序。",
        "",
        "## 11. 运行速度和显存",
        json.dumps(open_summary.get("performance", {}), ensure_ascii=False, indent=2, default=json_default),
        json.dumps(metadata.get("environment", {}).get("jax", {}), ensure_ascii=False, indent=2, default=json_default),
        "",
        "## 12. 结论与限制",
        json.dumps(validation_status, ensure_ascii=False, indent=2, default=json_default),
        "",
        "## 13. 完整复现命令",
        f"```bash\n{cmd}\n```",
    ]
    md = "\n".join(lines) + "\n"
    (output_dir / "report.md").write_text(md, encoding="utf-8")
    html_text = "<html><head><meta charset=utf-8><title>Piper validation report</title></head><body><pre>" + html.escape(md) + "</pre></body></html>"
    (output_dir / "report.html").write_text(html_text, encoding="utf-8")


def validate_runtime(args: argparse.Namespace, metadata: dict[str, Any]) -> dict[str, Any]:
    errors = []
    warnings = []
    ckpt = metadata.get('checkpoint_format', {}).get('steps', {}).get(args.checkpoint_dir.name, {})
    if not ckpt.get("complete_orbax_params"):
        errors.append(f"checkpoint params incomplete: {args.checkpoint_dir}")
    tfds_meta = metadata.get("tfds_metadata", {})
    for split in ("seen_test", "unseen_test"):
        if split not in tfds_meta.get("splits", {}):
            errors.append(f"missing split {split} in dataset_info.json")
        if tfds_meta.get("tfrecord_shard_counts", {}).get(split, 0) <= 0:
            errors.append(f"missing TFRecord shards for {split}")
    if not args.norm_stats_path.exists():
        errors.append(f"norm stats path missing: {args.norm_stats_path}")
    jax_info = metadata.get("environment", {}).get("jax", {})
    if args.device == "cuda" and not jax_info.get("gpu_devices"):
        errors.append("requested DEVICE=cuda but JAX reports no GPU devices in this SSH environment")
    forbidden = check_forbidden_modules()
    if forbidden:
        errors.append(f"forbidden modules loaded: {forbidden}")
    return {"ok": not errors, "errors": errors, "warnings": warnings}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--openpi-root", type=Path, default=DEFAULT_OPENPI_ROOT)
    p.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Checkpoint trained with the task-disjoint Piper30 builder; historical Piper checkpoints are invalid.",
    )
    p.add_argument("--norm-stats-path", type=Path, default=DEFAULT_NORM_STATS)
    p.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    p.add_argument("--config-name", default="cotrain_real_only")
    p.add_argument("--split", default="seen_test")
    p.add_argument("--episodes", type=int, default=0, help="0 means all episodes")
    p.add_argument("--anchors-per-episode", type=int, default=20)
    p.add_argument("--actions-per-inference", type=int, default=8)
    p.add_argument("--flow-loss-samples", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--anchor-manifest", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_TARGET_ROOT / "results" / "step_020000")
    p.add_argument("--metadata-only", action="store_true")
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--skip-flow-loss", action="store_true")
    p.add_argument("--skip-open-loop", action="store_true")
    p.add_argument("--generate-report", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(args.output_dir / "evaluation.log", encoding="utf-8")],
    )
    add_openpi_to_path(args.openpi_root)
    metadata = collect_training_metadata(args)
    write_json(DEFAULT_TARGET_ROOT / "config" / "training_metadata_0629.json", metadata)
    write_json(args.output_dir / "environment_manifest.json", metadata["environment"])
    write_json(args.output_dir / "checkpoint_manifest.json", metadata["checkpoint_format"])
    run_config = vars(args).copy()
    write_json(args.output_dir / "run_config.json", run_config)
    validation_status = validate_runtime(args, metadata)
    write_json(args.output_dir / "validation_status.json", validation_status)

    if args.metadata_only:
        generate_report(args.output_dir, args, metadata, None, validation_status)
        return 0
    if args.validate_only:
        generate_report(args.output_dir, args, metadata, None, validation_status)
        return 0 if validation_status["ok"] else 2
    if not validation_status["ok"]:
        generate_report(args.output_dir, args, metadata, None, validation_status)
        logging.error("Validation failed before model evaluation: %s", validation_status["errors"])
        return 2

    train_config = load_train_config(args.openpi_root, args.config_name, assets_base_from_norm_stats(args.norm_stats_path))
    model_horizon = int(train_config.model.action_horizon)
    max_eps = args.episodes if args.episodes and args.episodes > 0 else None
    split_tag = args.split.replace("/", "_")
    open_manifest = (
        args.anchor_manifest
        or DEFAULT_TARGET_ROOT
        / "manifests"
        / f"{split_tag}_open_loop_h{args.actions_per_inference}_seed{args.seed}.jsonl"
    )
    flow_manifest = (
        DEFAULT_TARGET_ROOT / "manifests" / f"{split_tag}_flow_loss_MODEL_HORIZON_seed{args.seed}.jsonl"
    )
    open_records = ensure_anchor_manifest(open_manifest, args.dataset_dir, args.split, args.actions_per_inference, args.anchors_per_episode, max_eps)
    flow_records = ensure_anchor_manifest(flow_manifest, args.dataset_dir, args.split, model_horizon, args.anchors_per_episode, max_eps)
    if max_eps is not None:
        allowed_eps = {r["episode_index"] for r in open_records[: max_eps * args.anchors_per_episode]}
        open_records = [r for r in open_records if r["episode_index"] in allowed_eps]
        flow_records = [r for r in flow_records if r["episode_index"] in allowed_eps]
    anchor_request = {
        "requested_episodes": args.episodes or "all",
        "requested_anchors_per_episode": args.anchors_per_episode,
        "open_loop_manifest": str(open_manifest),
        "flow_manifest": str(flow_manifest),
        "open_loop_actual_anchors": len(open_records),
        "flow_actual_anchors": len(flow_records),
    }
    norm_stats = load_norm_stats(args.openpi_root, args.norm_stats_path)
    flow_rows: list[dict[str, Any]] = []
    if not args.skip_flow_loss:
        flow_rows = evaluate_flow_loss(args, flow_records, norm_stats)
    open_rows: list[dict[str, Any]] = []
    perf: dict[str, Any] = {}
    if not args.skip_open_loop:
        open_rows, perf = evaluate_open_loop(args, open_records, norm_stats)
    summaries = save_tables_and_summaries(args, args.output_dir, open_rows, flow_rows, perf, anchor_request)
    if args.generate_report:
        generate_report(args.output_dir, args, metadata, summaries, validation_status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
