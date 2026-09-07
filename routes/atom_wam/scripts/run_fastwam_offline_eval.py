#!/usr/bin/env python3
"""Offline FastWAM evaluation on Piper2/Piper30 (Unified80), pi05-style protocol.

Runs the universal ``fastwam-unified80`` backend on:
  piper2 / piper30  ×  seen_test / unseen_test
and writes one JSON with six metric blocks (including piper2+piper30 combined).

Each block reports:
  mae, rmse, direction, flow_loss

Example::

  cd /path/to/Atom-0/routes/atom_wam && source scripts/atom0_env.sh
  export DIFFSYNTH_MODEL_BASE_PATH=\"$(pwd)/checkpoints/fastwam\"
  export RLDS_DATA_DIR=/path/to/rlds
  PYTHONPATH=src .venv/bin/python scripts/run_fastwam_offline_eval.py \\
    --checkpoint checkpoints/wam-cross-fix/fw-wam-cross-b800/99999 \\
    --config-name wam-cross-fix \\
    --output-json tmp/fastwam_offline/fw-wam-cross-b800-99999/metrics.json
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import tyro

_UNIVERSAL_ROOT = Path(__file__).resolve().parents[2] / "universal_offline_eval"
if _UNIVERSAL_ROOT.is_dir() and str(_UNIVERSAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_UNIVERSAL_ROOT))

_FASTWAM_SPEC_KEYS = (
    "action_dim",
    "proprio_dim",
    "action_horizon",
    "max_token_len",
    "video_num_frames",
    "action_video_freq_ratio",
    "image_resolution",
    "concat_multi_camera",
    "camera_keys",
    "load_text_encoder",
    "redirect_common_files",
    "skip_dit_load_from_pretrain",
    "skip_vae_load_from_pretrain",
    "loss",
)


def init_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


@dataclasses.dataclass
class Args:
    checkpoint: Path
    """Step dir, exp dir, or ``model.safetensors``."""

    output_json: Path = Path("tmp/fastwam_offline/metrics.json")
    config_name: str = "wam-cross-fix"
    assets_base_dir: Path | None = None
    rlds_data_dir: Path | None = None
    universal_profiles_dir: Path | None = None
    model_spec: Path | None = None
    """Optional architecture JSON; default derives from ``--config-name``."""

    tokenizer_dir: Path | None = None
    """Local umt5-xxl tokenizer; default ``checkpoints/fastwam/.../umt5-xxl``."""

    device: str = "cuda:0"
    trajectories_per_task: int = 3
    max_tasks: int = 0
    action_anchors_per_episode: int = 20
    flow_anchors_per_episode: int = 10
    action_horizon: int = 8
    num_inference_steps: int = 20
    flow_noise_samples: int = 1
    seed: int = 42
    smoke: bool = False
    keep_detail: bool = False
    """If True, keep per-split debug dirs; default writes only --output-json."""


def _configure_env(args: Args, repo_root: Path) -> None:
    os.environ.setdefault(
        "DIFFSYNTH_MODEL_BASE_PATH",
        str(repo_root / "checkpoints" / "fastwam"),
    )
    if args.rlds_data_dir is not None:
        os.environ["RLDS_DATA_DIR"] = str(args.rlds_data_dir)
    elif "RLDS_DATA_DIR" not in os.environ:
        os.environ["RLDS_DATA_DIR"] = "/path/to/rlds"


def _resolve_checkpoint(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file():
        return path
    direct = path / "model.safetensors"
    if direct.is_file():
        return direct
    steps: list[tuple[int, Path]] = []
    if path.is_dir():
        for child in path.iterdir():
            if child.is_dir() and child.name.isdigit():
                cand = child / "model.safetensors"
                if cand.is_file():
                    steps.append((int(child.name), cand))
    if steps:
        steps.sort()
        return steps[-1][1]
    raise FileNotFoundError(f"No model.safetensors under {path}")


def _default_tokenizer_dir(repo_root: Path) -> Path:
    return (
        repo_root
        / "checkpoints/fastwam/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl"
    )


def _model_spec_from_config(config_name: str) -> dict[str, Any]:
    import openpi.cotrain.config as cotrain_config
    from openpi.models.fastwam_config import FastWAMConfig

    config = cotrain_config.get_config(config_name)
    if not isinstance(config.model, FastWAMConfig):
        raise TypeError(f"{config_name} is not a FastWAM config")
    spec: dict[str, Any] = {}
    for key in _FASTWAM_SPEC_KEYS:
        value = getattr(config.model, key)
        if isinstance(value, tuple):
            value = list(value)
        spec[key] = value
    spec["load_text_encoder"] = False
    spec["skip_dit_load_from_pretrain"] = True
    spec["skip_vae_load_from_pretrain"] = True
    return spec


def _resolve_model_spec(args: Args, repo_root: Path) -> Path:
    if args.model_spec is not None:
        path = args.model_spec.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Model spec not found: {path}")
        return path
    spec = _model_spec_from_config(args.config_name)
    path = Path(tempfile.mkdtemp(prefix="fastwam-offline-spec-")) / "model_spec.json"
    path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    return path


def _builder_and_assets(dataset: str, config_name: str, assets_base_dir: Path | None) -> tuple[Path, Path]:
    import openpi.cotrain.config as cotrain_config

    config = cotrain_config.get_config(config_name)
    if assets_base_dir is not None:
        assets_dir = assets_base_dir.expanduser().resolve()
    else:
        assets_dir = Path(config.assets_dirs)
    if dataset == "piper2":
        return Path(cotrain_config._PIPER2_BUILDER_DIR), assets_dir
    if dataset == "piper30":
        return Path(cotrain_config._PIPER30_BUILDER_DIR), assets_dir
    raise ValueError(f"Unsupported dataset {dataset!r}")


def _public_metrics(block: dict[str, Any]) -> dict[str, float | None]:
    return {
        "mae": block.get("mae"),
        "rmse": block.get("rmse"),
        "direction": block.get("direction"),
        "flow_loss": block.get("flow_loss"),
    }


def _compact_metrics(run_summary: dict[str, Any]) -> dict[str, float | int | None]:
    action = (run_summary.get("summaries") or {}).get("action") or {}
    flow = (run_summary.get("summaries") or {}).get("flow") or {}
    action_metrics = action.get("metrics") or {}
    flow_metrics = (flow.get("metrics") or {}).get("flow_loss") or {}
    direction = action_metrics.get("movement_direction_match_moving")
    if direction is None:
        direction = action_metrics.get("movement_direction_match_legacy")
    flow_loss = flow_metrics.get("mean") if isinstance(flow_metrics, dict) else flow_metrics
    out: dict[str, float | int | None] = {
        "mae": action_metrics.get("action_mae"),
        "rmse": action_metrics.get("action_rmse"),
        "direction": direction,
        "flow_loss": flow_loss,
    }
    for suffix in ("moving", "legacy"):
        matches_key = f"movement_direction_matches_{suffix}"
        comparisons_key = f"movement_direction_comparisons_{suffix}"
        if matches_key in action_metrics and comparisons_key in action_metrics:
            out[f"direction_matches_{suffix}"] = int(action_metrics[matches_key])
            out[f"direction_comparisons_{suffix}"] = int(action_metrics[comparisons_key])
            break
    return out


def _aggregate_blocks(blocks: list[dict[str, Any]]) -> dict[str, float | None]:
    valid = [block for block in blocks if block.get("mae") is not None]
    if not valid:
        return {"mae": None, "rmse": None, "direction": None, "flow_loss": None}

    mae = float(np.mean([block["mae"] for block in valid]))
    rmse = float(math.sqrt(np.mean([block["rmse"] ** 2 for block in valid if block["rmse"] is not None])))

    matches = 0
    comparisons = 0
    for block in valid:
        for suffix in ("moving", "legacy"):
            m_key = f"direction_matches_{suffix}"
            c_key = f"direction_comparisons_{suffix}"
            if m_key in block and c_key in block:
                matches += int(block[m_key])
                comparisons += int(block[c_key])
                break
    direction = matches / comparisons if comparisons else None

    flow_vals = [block["flow_loss"] for block in valid if block.get("flow_loss") is not None]
    flow_loss = float(np.mean(flow_vals)) if flow_vals else None
    return {
        "mae": mae,
        "rmse": rmse,
        "direction": direction,
        "flow_loss": flow_loss,
    }


def _aggregate_from_runs(runs: list[dict[str, Any]]) -> dict[str, float | None]:
    return _aggregate_blocks([_compact_metrics(run) for run in runs])


def _run_single(
    *,
    args: Args,
    repo_root: Path,
    checkpoint: Path,
    dataset: str,
    split: str,
    profile_path: Path,
    assets_dir: Path,
    dataset_dir: Path,
    model_spec_path: Path,
    tokenizer_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    from universal_eval.fastwam_unified80 import run as run_fastwam_eval

    norm_dir = assets_dir / dataset
    argv = [
        "--repo-root",
        str(repo_root),
        "--checkpoint",
        str(checkpoint),
        "--model-spec",
        str(model_spec_path),
        "--tokenizer-dir",
        str(tokenizer_dir),
        "--norm-dir",
        str(norm_dir),
        "--dataset-dir",
        str(dataset_dir),
        "--data-profile",
        str(profile_path),
        "--split",
        split,
        "--output-dir",
        str(output_dir),
        "--device",
        args.device,
        "--trajectories-per-task",
        str(args.trajectories_per_task),
        "--max-tasks",
        str(args.max_tasks),
        "--action-anchors-per-episode",
        str(args.action_anchors_per_episode),
        "--flow-anchors-per-episode",
        str(args.flow_anchors_per_episode),
        "--action-horizon",
        str(args.action_horizon),
        "--num-inference-steps",
        str(args.num_inference_steps),
        "--flow-noise-samples",
        str(args.flow_noise_samples),
        "--seed",
        str(args.seed),
        "--mode",
        "both",
    ]
    if args.smoke:
        argv.append("--smoke")
    return run_fastwam_eval(argv)


def main(args: Args) -> None:
    init_logging()
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    _configure_env(args, repo_root)

    checkpoint = _resolve_checkpoint(args.checkpoint)
    profiles_dir = args.universal_profiles_dir or (_UNIVERSAL_ROOT / "profiles")
    tokenizer_dir = (args.tokenizer_dir or _default_tokenizer_dir(repo_root)).expanduser().resolve()
    if not tokenizer_dir.is_dir():
        raise FileNotFoundError(f"Tokenizer directory not found: {tokenizer_dir}")

    model_spec_path = _resolve_model_spec(args, repo_root)
    spec_is_temp = args.model_spec is None
    if args.model_spec is not None:
        model_spec_dict = json.loads(model_spec_path.read_text(encoding="utf-8"))
    else:
        model_spec_dict = _model_spec_from_config(args.config_name)

    from universal_eval.checkpoint import inspect_checkpoint

    ckpt_info = inspect_checkpoint(checkpoint)
    logging.info("Checkpoint=%s backend=%s", checkpoint, ckpt_info.get("backend"))

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    detail_root = args.output_json.expanduser().resolve().parent if args.keep_detail else Path(
        tempfile.mkdtemp(prefix="fastwam-offline-eval-")
    )
    if args.keep_detail:
        detail_root.mkdir(parents=True, exist_ok=True)

    eval_specs = (
        ("piper2", "seen_test", "piper2_seen"),
        ("piper2", "unseen_test", "piper2_unseen"),
        ("piper30", "seen_test", "piper30_seen"),
        ("piper30", "unseen_test", "piper30_unseen"),
    )

    runs: dict[str, dict[str, Any]] = {}
    t0 = time.perf_counter()
    try:
        for dataset, split, key in eval_specs:
            builder_dir, assets_dir = _builder_and_assets(
                dataset, args.config_name, args.assets_base_dir
            )
            profile_path = profiles_dir / f"{dataset}.json"
            if not profile_path.is_file():
                raise FileNotFoundError(f"Missing data profile: {profile_path}")
            out_dir = detail_root / f"{dataset}-{split}"
            logging.info("Evaluating %s...", key)
            runs[key] = _run_single(
                args=args,
                repo_root=repo_root,
                checkpoint=checkpoint,
                dataset=dataset,
                split=split,
                profile_path=profile_path,
                assets_dir=assets_dir,
                dataset_dir=builder_dir,
                model_spec_path=model_spec_path,
                tokenizer_dir=tokenizer_dir,
                output_dir=out_dir,
            )
    finally:
        if spec_is_temp:
            shutil.rmtree(model_spec_path.parent, ignore_errors=True)

    combined_seen = _aggregate_from_runs([runs["piper2_seen"], runs["piper30_seen"]])
    combined_unseen = _aggregate_from_runs([runs["piper2_unseen"], runs["piper30_unseen"]])

    if not args.keep_detail and str(detail_root).startswith(tempfile.gettempdir()):
        shutil.rmtree(detail_root, ignore_errors=True)

    results = {
        "checkpoint": str(checkpoint),
        "config_name": args.config_name,
        "model_spec": model_spec_dict,
        "piper2_seen": _public_metrics(_compact_metrics(runs["piper2_seen"])),
        "piper2_unseen": _public_metrics(_compact_metrics(runs["piper2_unseen"])),
        "piper30_seen": _public_metrics(_compact_metrics(runs["piper30_seen"])),
        "piper30_unseen": _public_metrics(_compact_metrics(runs["piper30_unseen"])),
        "piper_merged_seen": _public_metrics(combined_seen),
        "piper_merged_unseen": _public_metrics(combined_unseen),
    }

    args.output_json.write_text(json.dumps(results, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    logging.info("Wrote %s (%.1fs)", args.output_json, time.perf_counter() - t0)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main(tyro.cli(Args))
