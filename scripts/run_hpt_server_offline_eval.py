#!/usr/bin/env python3
"""Offline piper30 metrics through the *real* HPT server policy path.

Uses ``serve_hpt_piper.create_policy -> PiperHPTPolicy.infer`` (Policy.infer +
HistoryBufferTransform).  At each anchor we ``reset()`` and replay the last
``observation_horizon`` frames so history matches training / ``run_hpt_episode``.

Example::

  cd /data/zjyang/Atom-0 && source scripts/atom0_env.sh
  export HF_HOME=/data/zjyang/cache/huggingface HF_HUB_OFFLINE=1
  PYTHONPATH=src .venv/bin/python scripts/run_hpt_server_offline_eval.py \\
    --checkpoint checkpoints/hpt_cotrain_real_only/hpt-robot-tdec-bs256/99999 \\
    --output-json tmp/hpt_offline_eval/hpt-robot-tdec-bs256-99999-server/metrics.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import tyro

_REPO = Path(__file__).resolve().parents[1]
_DEFAULT_DEPLOY = _REPO.parent / "HPT_server" / "deploy"
_UNIVERSAL_ROOT = _REPO.parent / "universal_offline_eval"
if _UNIVERSAL_ROOT.is_dir() and str(_UNIVERSAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_UNIVERSAL_ROOT))


def init_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


@dataclasses.dataclass
class Args:
    checkpoint: Path
    output_json: Path = Path("tmp/hpt_offline_eval/hpt-server/metrics.json")
    deploy_dir: Path = _DEFAULT_DEPLOY
    openpi_root: Path | None = None
    atom0_root: Path | None = None
    """Main Atom-0 repo on cloud/NFS (checkpoints + assets). Default: $ATOM0_ROOT or repo root."""
    config_name: str = "hpt_cotrain_real_only"
    dataset: str = "piper30"
    device: str = "cuda:0"
    action_horizon: int = 8
    """Metric horizon (matches universal offline eval; model still predicts 50)."""
    trajectories_per_task: int = 3
    max_tasks: int = 0
    action_anchors_per_episode: int = 20
    seed: int = 42
    smoke: bool = False
    rlds_data_dir: Path | None = None


SERVE_IMAGE_KEYS = {
    "base_0_rgb": "head",
    "left_wrist_0_rgb": "left_wrist",
    "right_wrist_0_rgb": "right_wrist",
}


def _configure_env(args: Args) -> None:
    os.environ.setdefault("HF_HOME", "/data/zjyang/cache/huggingface")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(os.environ["HF_HOME"], "hub"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if args.rlds_data_dir is not None:
        os.environ["RLDS_DATA_DIR"] = str(args.rlds_data_dir)
    elif "RLDS_DATA_DIR" not in os.environ:
        os.environ["RLDS_DATA_DIR"] = "/mnt/bos/bo23lu"


def _import_serve(deploy_dir: Path):
    deploy_dir = deploy_dir.resolve()
    if str(deploy_dir) not in sys.path:
        sys.path.insert(0, str(deploy_dir))
    import serve_hpt_piper as serve  # noqa: WPS433

    return serve


def _builder_dir(dataset: str) -> Path:
    import openpi.cotrain.config as cotrain_config

    if dataset == "piper30":
        return Path(cotrain_config._PIPER30_BUILDER_DIR)
    raise ValueError(f"Only piper30 supported, got {dataset!r}")


def _client_obs(episode: dict[str, Any], t: int, prompt: str) -> dict[str, Any]:
    return {
        "active_state": np.asarray(episode["state"][t], dtype=np.float32),
        "images": {
            SERVE_IMAGE_KEYS[slot]: np.asarray(episode["images"][slot][t], dtype=np.uint8)
            for slot in SERVE_IMAGE_KEYS
        },
        "prompt": prompt,
    }


def _warm_and_infer(policy: Any, episode: dict[str, Any], anchor: int, prompt: str, *, obs_h: int) -> np.ndarray:
    """Reset history and replay frames [anchor-T+1 .. anchor] before returning actions."""
    policy.reset()
    for dt in range(obs_h - 1, -1, -1):
        idx = max(0, anchor - dt)
        obs = _client_obs(episode, idx, prompt)
        result = policy.infer(obs)
    return np.asarray(result["full_actions"], dtype=np.float64)


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    if not rows:
        return {"mae": None, "rmse": None, "direction": None}
    mae = float(np.mean([row["action_mae"] for row in rows]))
    rmse = float(math.sqrt(np.mean([row["action_rmse"] ** 2 for row in rows])))
    matches = comparisons = 0
    for row in rows:
        for suffix in ("moving", "legacy"):
            m_key = f"movement_direction_matches_{suffix}"
            c_key = f"movement_direction_comparisons_{suffix}"
            if m_key in row and c_key in row:
                matches += int(row[m_key])
                comparisons += int(row[c_key])
                break
    direction = matches / comparisons if comparisons else None
    return {"mae": mae, "rmse": rmse, "direction": direction}


def _eval_split(
    *,
    args: Args,
    serve: Any,
    policy: Any,
    split: str,
    profile: Any,
    builder_dir: Path,
) -> dict[str, float | None]:
    from run_fastwam_joint_episode import _load_full_episode
    from universal_eval.common import action_metrics, build_manifest, select_episodes

    trajectories = args.trajectories_per_task
    max_tasks = args.max_tasks
    anchors = args.action_anchors_per_episode
    if args.smoke:
        trajectories = max_tasks = anchors = 1

    episodes = select_episodes(
        builder_dir,
        split,
        trajectories_per_task=trajectories,
        max_tasks=max_tasks,
    )
    if not episodes:
        raise RuntimeError(f"No episodes selected for split={split!r}")

    manifest = build_manifest(
        episodes,
        horizon=args.action_horizon,
        anchors_per_episode=anchors,
        precomputed_chunk=False,
    )
    by_ordinal: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest:
        by_ordinal[int(row["split_ordinal"])].append(row)

    obs_h = int(getattr(policy._policy._model.config, "observation_horizon", 1))
    metric_rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()

    for ordinal in sorted(by_ordinal):
        episode = _load_full_episode(builder_dir, split, ordinal)
        prompt = str(episode["prompt"])
        gt_actions = episode["action"].astype(np.float64)
        for row in by_ordinal[ordinal]:
            anchor = int(row["anchor_index"])
            horizon = int(row["horizon"])
            target = gt_actions[anchor : anchor + horizon]
            if target.shape[0] < horizon:
                continue
            pred = _warm_and_infer(policy, episode, anchor, prompt, obs_h=obs_h)[:horizon]
            metrics = action_metrics(
                pred,
                target,
                direction_deadband=profile.direction_deadband,
                direction_groups=profile.direction_groups,
            )
            metric_rows.append(
                {
                    "episode_index": int(row["episode_index"]),
                    "anchor_index": anchor,
                    **metrics,
                }
            )

    logging.info(
        "Split %s: %d anchors on %d episodes (%.1fs)",
        split,
        len(metric_rows),
        len(by_ordinal),
        time.perf_counter() - t0,
    )
    return _aggregate_rows(metric_rows)


def main(args: Args) -> None:
    init_logging()
    _configure_env(args)

    serve = _import_serve(args.deploy_dir)
    openpi_root = (args.openpi_root or (args.deploy_dir.parent / "source" / "Atom-0")).resolve()
    atom0_root = (
        args.atom0_root.expanduser().resolve()
        if args.atom0_root is not None
        else serve.resolve_atom0_root(None)
    )
    serve.add_openpi_paths(openpi_root)
    serve.configure_hf_offline(openpi_root)

    from universal_eval.common import DataProfile

    profile_path = _UNIVERSAL_ROOT / "profiles" / f"{args.dataset}.json"
    profile = dataclasses.replace(
        DataProfile.load(profile_path),
        prompt_prefix="",
    )

    norm_stats = atom0_root / "assets" / "cotrain_real_only" / args.dataset / "norm_stats.json"
    serve_args = argparse.Namespace(
        openpi_root=openpi_root,
        atom0_root=atom0_root,
        checkpoint_dir=args.checkpoint.expanduser().resolve(),
        norm_stats_path=norm_stats.resolve(),
        config_name=args.config_name,
        dataset_id=args.dataset,
        prompt="",
        host="127.0.0.1",
        port=8012,
        device=args.device,
        actions_per_inference=50,
        num_inference_steps=50,
        seed=args.seed,
    )
    logging.info("Creating HPT server policy from %s", args.deploy_dir)
    policy = serve.create_hpt_policy(serve_args)

    builder_dir = _builder_dir(args.dataset)
    ckpt = serve._resolve_checkpoint(serve_args.checkpoint_dir)
    head_mode = serve._infer_head_mode(ckpt)
    action_head_type = serve._infer_action_head_type(ckpt)

    results: dict[str, Any] = {
        "backend": "hpt_server_policy",
        "checkpoint": str(ckpt),
        "head_mode": head_mode,
        "action_head_type": action_head_type,
        "action_horizon_metric": args.action_horizon,
        "model_action_horizon": int(policy._policy._model.action_horizon),
        "observation_horizon": int(getattr(policy._policy._model.config, "observation_horizon", 1)),
        "prompt_prefix": "",
        "note": "History reset+warm per anchor; flow_loss N/A for transformer_decoder server path.",
    }

    t0 = time.perf_counter()
    for split, key in (("seen_test", "piper30_seen"), ("unseen_test", "piper30_unseen")):
        logging.info("Evaluating %s (%s)...", key, split)
        results[key] = _eval_split(
            args=args,
            serve=serve,
            policy=policy,
            split=split,
            profile=profile,
            builder_dir=builder_dir,
        )

    results["elapsed_sec"] = time.perf_counter() - t0
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    logging.info("Wrote %s", args.output_json)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main(tyro.cli(Args))
