#!/usr/bin/env python3
"""Drive piper RLDS episodes through the *real* serve_fastwam_piper.py policy.

Does not reimplement compose/standardize/infer — only loads an episode and calls
``PiperFastWAMPolicy.infer`` from the deploy script.

Example::

  cd /data/zjyang/Atom-0 && source scripts/atom0_env.sh
  export RLDS_DATA_DIR=/mnt/bos/bo23lu
  export DIFFSYNTH_MODEL_BASE_PATH=\"$(pwd)/checkpoints/fastwam\"
  PYTHONPATH=src .venv/bin/python scripts/run_fastwam_server_episode.py \\
    --episode-index 29 --split train --split-ordinal 2020 \\
    --output-dir tmp/fastwam_server/piper30-ep29-v4-99999-direct
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = Path(__file__).resolve().parent
_DEFAULT_DEPLOY = (
    _REPO
    / "FastWAM_v4_99999_server_only_20260812"
    / "FastWAM_v4_99999_server_only"
    / "deploy"
)
_DEFAULT_OPENPI = _DEFAULT_DEPLOY.parent / "source" / "Atom-0"
_DEFAULT_CKPT = _REPO / "checkpoints" / "wam-cross-piper" / "fw-wam-cross-piper-v4" / "99999"

if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from run_fastwam_joint_sample import _save_native_trajectories, init_logging  # noqa: E402

PROMPT_PREFIX = "Action Mode: joint. "
RAW_TO_SERVE = {
    "head": "cam_high",
    "left_wrist": "cam_left_wrist",
    "right_wrist": "cam_right_wrist",
}


def _decode_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "numpy"):
        value = value.numpy()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _ensure_server_assets(openpi_root: Path) -> None:
    """Serve package often ships without assets/DiffSynth; link from main Atom-0."""
    assets = openpi_root / "assets"
    if not assets.exists():
        assets.symlink_to(_REPO / "assets")
    ckpt_fw = openpi_root / "checkpoints" / "fastwam"
    ckpt_fw.parent.mkdir(parents=True, exist_ok=True)
    if not ckpt_fw.exists():
        ckpt_fw.symlink_to(_REPO / "checkpoints" / "fastwam")


def _import_serve(deploy_dir: Path):
    deploy_dir = deploy_dir.resolve()
    if str(deploy_dir) not in sys.path:
        sys.path.insert(0, str(deploy_dir))
    import serve_fastwam_piper as serve  # noqa: WPS001

    return serve


def _builder_dir_piper30() -> Path:
    # Use server openpi cotrain paths after add_openpi_paths.
    import openpi.cotrain.config as cotrain_config

    return Path(cotrain_config._PIPER30_BUILDER_DIR)


def _load_episode(builder_dir: Path, split: str, ordinal: int) -> dict[str, Any]:
    import tensorflow_datasets as tfds

    builder = tfds.builder_from_directory(str(builder_dir))
    ds = builder.as_dataset(split=f"{split}[{ordinal}:{ordinal + 1}]", shuffle_files=False)
    ep = next(iter(ds))
    meta = ep["episode_metadata"]
    states, actions, prompts = [], [], []
    images = {k: [] for k in RAW_TO_SERVE}
    for step in ep["steps"].as_numpy_iterator():
        states.append(np.asarray(step["observation"]["state"], dtype=np.float32))
        actions.append(np.asarray(step["action"], dtype=np.float32))
        prompts.append(_decode_text(step["task"]))
        obs_imgs = step["observation"]["images"]
        for serve_name, raw_key in RAW_TO_SERVE.items():
            images[serve_name].append(np.asarray(obs_imgs[raw_key]))
    return {
        "episode_index": int(meta["episode_index"].numpy()),
        "num_frames": len(actions),
        "task": _decode_text(meta["task"]),
        "state": np.stack(states, axis=0),
        "action": np.stack(actions, axis=0),
        "images": {k: np.stack(v, axis=0) for k, v in images.items()},
        "prompt": prompts[0] if prompts else _decode_text(meta["task"]),
    }


def _make_anchors(num_frames: int, chunk: int, stride: int) -> list[int]:
    last_start = max(0, num_frames - chunk)
    anchors = list(range(0, last_start + 1, stride))
    if not anchors or anchors[-1] != last_start:
        anchors.append(last_start)
    return anchors


def _client_obs(episode: dict[str, Any], t: int, prompt: str) -> dict[str, Any]:
    """Same observation keys a Piper robot client sends to the websocket server."""
    return {
        "active_state": np.asarray(episode["state"][t], dtype=np.float32),
        "images": {
            "head": episode["images"]["head"][t],
            "left_wrist": episode["images"]["left_wrist"][t],
            "right_wrist": episode["images"]["right_wrist"][t],
        },
        "prompt": prompt,
    }


def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--deploy-dir", type=Path, default=_DEFAULT_DEPLOY)
    p.add_argument("--openpi-root", type=Path, default=_DEFAULT_OPENPI)
    p.add_argument("--checkpoint-dir", type=Path, default=_DEFAULT_CKPT)
    p.add_argument("--norm-stats-path", type=Path, default=None)
    p.add_argument("--config-name", default="wam-cross-piper")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--actions-per-inference", type=int, default=32)
    p.add_argument("--num-inference-steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--dataset", default="piper30")
    p.add_argument("--split", default="train")
    p.add_argument("--episode-index", type=int, default=29)
    p.add_argument("--split-ordinal", type=int, default=2020)
    p.add_argument("--prompt", default=None)
    p.add_argument("--output-dir", type=Path, default=Path("tmp/fastwam_server/piper30-ep29-direct"))
    p.add_argument("--rlds-data-dir", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    init_logging()
    args = parse_cli()
    if args.dataset != "piper30":
        raise SystemExit("This driver currently wires piper30 only")

    if args.rlds_data_dir is not None:
        os.environ["RLDS_DATA_DIR"] = str(args.rlds_data_dir)
    elif "RLDS_DATA_DIR" not in os.environ:
        os.environ["RLDS_DATA_DIR"] = "/mnt/bos/bo23lu"

    openpi_root = args.openpi_root.resolve()
    _ensure_server_assets(openpi_root)
    os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(_REPO / "checkpoints" / "fastwam"))
    os.environ.setdefault("HF_HOME", str(Path(os.environ["DIFFSYNTH_MODEL_BASE_PATH"]) / "hf_cache"))
    os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")
    os.environ.setdefault("FASTWAM_SKIP_PRETRAIN_INIT", "true")

    serve = _import_serve(args.deploy_dir)
    serve.add_openpi_paths(openpi_root)

    norm_stats = args.norm_stats_path or (
        openpi_root / "assets" / "cotrain_real_robot_ego_fix" / "piper30" / "norm_stats.json"
    )
    serve_args = argparse.Namespace(
        openpi_root=openpi_root,
        checkpoint_dir=args.checkpoint_dir.resolve(),
        norm_stats_path=norm_stats.resolve(),
        config_name=args.config_name,
        prompt=args.prompt or "",
        host="127.0.0.1",
        port=8011,
        device=args.device,
        actions_per_inference=args.actions_per_inference,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        rtc_enabled=False,
        rtc_delay_steps=3,
        rtc_execution_horizon=args.actions_per_inference,
        rtc_soft_mask_decay=0.6,
        rtc_guidance_scale=1.0,
        validate_only=False,
    )
    # Match serve CLI guard.
    if serve_args.rtc_enabled:
        raise SystemExit("RTC is not supported by the B200 wam-cross-piper v4 runtime.")

    logging.info("Creating policy via serve_fastwam_piper.create_policy ...")
    logging.info("  deploy=%s", args.deploy_dir.resolve())
    logging.info("  openpi_root=%s", openpi_root)
    logging.info("  checkpoint=%s", serve_args.checkpoint_dir)
    policy = serve.create_policy(serve_args)

    builder_dir = _builder_dir_piper30()
    logging.info("Loading piper30 %s[%s:%s] episode_index=%s", args.split, args.split_ordinal, args.split_ordinal + 1, args.episode_index)
    episode = _load_episode(builder_dir, args.split, args.split_ordinal)
    if int(episode["episode_index"]) != int(args.episode_index):
        raise RuntimeError(
            f"Ordinal {args.split_ordinal} is episode_index={episode['episode_index']}, "
            f"expected {args.episode_index}"
        )

    prompt = args.prompt if args.prompt is not None else f"{PROMPT_PREFIX}{episode['prompt']}"
    t_len = int(episode["num_frames"])
    chunk = int(args.actions_per_inference)
    stride = int(args.stride) if args.stride is not None else chunk
    anchors = _make_anchors(t_len, chunk, stride)
    logging.info("Sync open-loop via PiperFastWAMPolicy.infer: T=%s chunk=%s windows=%s", t_len, chunk, len(anchors))

    pred = np.full((t_len, serve.PIPER_DIM), np.nan, dtype=np.float64)
    covered = np.zeros(t_len, dtype=bool)
    t0 = time.perf_counter()
    for win_i, anchor in enumerate(anchors):
        policy._policy._sample_kwargs["seed"] = args.seed + win_i
        obs = _client_obs(episode, anchor, prompt)
        logging.info("Window %d/%d anchor=%d", win_i + 1, len(anchors), anchor)
        result = policy.infer(obs)  # exact serve path
        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != serve.PIPER_DIM:
            raise RuntimeError(f"Expected [H,14] from serve policy, got {actions.shape}")
        next_boundary = anchors[win_i + 1] if win_i + 1 < len(anchors) else t_len
        n = min(chunk, actions.shape[0], t_len - anchor)
        for i in range(n):
            idx = anchor + i
            if idx >= next_boundary:
                break
            pred[idx] = actions[i]
            covered[idx] = True

    if not np.all(covered):
        missing = np.flatnonzero(~covered)
        raise RuntimeError(f"Uncovered steps: {missing[:20]}... (n={len(missing)})")

    out_dir = args.output_dir / f"episode_{episode['episode_index']:05d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    traj = _save_native_trajectories(
        out_dir,
        dataset_id="piper30",
        gt_native=episode["action"].astype(np.float64),
        pred_native=pred,
    )
    meta = {
        "infer_module": str((args.deploy_dir / "serve_fastwam_piper.py").resolve()),
        "infer_entry": "serve_fastwam_piper.create_policy -> PiperFastWAMPolicy.infer",
        "rtc_enabled": False,
        "dataset": "piper30",
        "split": args.split,
        "episode_index": episode["episode_index"],
        "split_ordinal": args.split_ordinal,
        "num_frames": t_len,
        "task": episode["task"],
        "prompt": prompt,
        "checkpoint": str(serve_args.checkpoint_dir),
        "openpi_root": str(openpi_root),
        "actions_per_inference": chunk,
        "stride": stride,
        "anchors": anchors,
        "num_windows": len(anchors),
        "num_inference_steps": args.num_inference_steps,
        "trajectory_mae": traj["mae"],
        "overlay": f"action_native_{traj['native_dim']}d_overlay.png",
        "elapsed_sec": time.perf_counter() - t0,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "output_dir": str(out_dir),
                "overlay": str(out_dir / meta["overlay"]),
                "trajectory_mae": traj["mae"],
                "infer_module": meta["infer_module"],
                "checkpoint": meta["checkpoint"],
                "elapsed_sec": meta["elapsed_sec"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    logging.info("Done via serve_fastwam_piper. MAE=%.4f overlay=%s", traj["mae"], out_dir / meta["overlay"])


if __name__ == "__main__":
    main()
