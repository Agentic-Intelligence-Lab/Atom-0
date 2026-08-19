#!/usr/bin/env python3
"""D2(e) vs training step for hpt-robot-ao checkpoints on a fixed episode."""

from __future__ import annotations

import dataclasses
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import tyro

_REPO = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from run_fastwam_joint_episode import (  # noqa: E402
    _load_full_episode,
    _make_anchors,
    _normalize_state,
    _select_episode_ordinal,
)
from run_fastwam_joint_sample import (  # noqa: E402
    _build_output_transforms,
    _postprocess_actions,
    init_logging,
)
from run_hpt_episode import (  # noqa: E402
    _builder_dir_for_dataset,
    _configure_hf_offline,
    _configure_rlds_root,
    _infer_head_mode,
    _infer_window,
)


def d2_error(e: np.ndarray) -> float:
    if e.shape[0] < 3:
        return float("nan")
    d2 = e[2:] - 2.0 * e[1:-1] + e[:-2]
    return float(np.linalg.norm(d2, axis=-1).mean())


def trajectory_metrics(pred: np.ndarray, gt: np.ndarray, *, horizon: int, stride: int) -> dict[str, float]:
    e = pred.astype(np.float64) - gt.astype(np.float64)
    mae = float(np.abs(e).mean())
    d2_e = d2_error(e)
    d2_gt = d2_error(np.zeros_like(gt))
    d2_pred = d2_error(pred.astype(np.float64))
    window_d2: list[float] = []
    for a in range(0, pred.shape[0] - horizon + 1, stride):
        if a + horizon > pred.shape[0]:
            break
        window_d2.append(d2_error(pred[a : a + horizon] - gt[a : a + horizon]))
    return {
        "mae": mae,
        "D2_error": d2_e,
        "D2_pred_traj": d2_pred,
        "D2_gt_traj": d2_gt,
        "D2_error_over_gt_traj": d2_e / max(d2_gt, 1e-12),
        "D2_pred_over_gt_traj": d2_pred / max(d2_gt, 1e-12),
        "D2_error_per_window_mean": float(np.mean(window_d2)) if window_d2 else float("nan"),
    }


@dataclasses.dataclass
class Args:
    ckpt_dir: Path = Path("checkpoints/hpt_cotrain_real_only/hpt-robot-ao")
    steps: tuple[int, ...] = tuple(range(10_000, 100_000, 10_000)) + (99_999,)
    output_json: Path = Path("tmp/hpt_denoise_sweep/hpt-robot-ao-d2-by-step.json")
    config_name: str = "hpt_cotrain_real_only"
    dataset: str = "piper2"
    split: str = "seen_test"
    episode_index: int = 117
    seed: int = 42
    num_inference_steps: int = 50
    device: str = "cuda:0"
    rlds_data_dir: Path | None = None


def main(args: Args) -> None:
    init_logging()
    _configure_rlds_root(type("A", (), {"rlds_data_dir": args.rlds_data_dir})())
    _configure_hf_offline()

    import dataclasses as dc
    import os
    import safetensors.torch
    import torch

    import openpi.cotrain.config as cotrain_config
    from openpi.cotrain import action_space
    from openpi.cotrain.config import load_per_dataset_norm_stats
    from openpi.models.hpt_config import HPTConfig

    os.environ.setdefault("RLDS_DATA_DIR", "/mnt/bos/bo23lu")

    config = cotrain_config.get_config(args.config_name)
    if not isinstance(config.model, HPTConfig):
        raise TypeError(f"{args.config_name} is not HPT")
    keep = tuple(ds for ds in config.data.datasets if ds.uid == args.dataset)
    config = dc.replace(config, data=dc.replace(config.data, datasets=keep))

    ckpt_dir = args.ckpt_dir.expanduser().resolve()
    first_ckpt = ckpt_dir / str(args.steps[0]) / "model.safetensors"
    head_mode = _infer_head_mode(first_ckpt)
    config = dc.replace(config, model=dc.replace(config.model, head_mode=head_mode))

    horizon = int(config.model.action_horizon)
    stride = horizon
    obs_h = int(getattr(config.model, "observation_horizon", 1))
    builder_dir = _builder_dir_for_dataset(args.dataset)
    ordinal, _ = _select_episode_ordinal(
        builder_dir,
        args.split,
        episode_index=args.episode_index,
        prefer_short=False,
        min_frames=horizon,
    )
    episode = _load_full_episode(builder_dir, args.split, ordinal)
    t_len = episode["num_frames"]
    anchors = _make_anchors(t_len, horizon, stride)
    prompt = episode["task"]
    gt_native = episode["action"].astype(np.float64)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logging.info("Creating HPT once on %s (head_mode=%s)", device, head_mode)
    model = config.model.create_pytorch(device=str(device))
    if config.model.freeze_encoders:
        model.freeze_encoders()
    model.to(device)
    model.eval()

    data_config = config.data.create(config.assets_dirs, config.model)
    output_transforms = _build_output_transforms(config, data_config)
    per_dataset_stats = load_per_dataset_norm_stats(config.assets_dirs, data_config.datasets)
    norm_stats = per_dataset_stats[args.dataset]
    spec = action_space.UNIFIED_ACTION_SPECS[args.dataset]
    action_mask = np.asarray(spec.action_mask, dtype=np.bool_)
    native_dim = int(keep[0].action_dim) if keep[0].action_dim > 0 else 14

    results: dict[str, Any] = {
        "ckpt_dir": str(ckpt_dir),
        "dataset": args.dataset,
        "split": args.split,
        "episode_index": episode["episode_index"],
        "seed": args.seed,
        "num_inference_steps": args.num_inference_steps,
        "head_mode": head_mode,
        "steps": {},
    }

    for step in args.steps:
        ckpt_path = ckpt_dir / str(step) / "model.safetensors"
        if not ckpt_path.is_file():
            logging.warning("Skip missing checkpoint step=%d (%s)", step, ckpt_path)
            continue
        logging.info("=== step=%d ===", step)
        t0 = time.perf_counter()
        safetensors.torch.load_model(model, str(ckpt_path), strict=False, device=str(device))
        model.eval()

        pred_action_native = np.zeros((t_len, native_dim), dtype=np.float64)
        covered = np.zeros(t_len, dtype=bool)
        for win_i, anchor in enumerate(anchors):
            window_seed = args.seed + win_i
            hist = []
            for dt in range(obs_h - 1, -1, -1):
                idx = max(0, anchor - dt)
                hist.append(
                    _normalize_state(
                        episode["state"][idx],
                        dataset_id=args.dataset,
                        norm_stats=norm_stats,
                        use_quantiles=data_config.use_quantile_norm,
                    )
                )
            norm_hist = np.stack(hist, axis=0)
            norm_state = hist[-1]
            pred_action_norm = _infer_window(
                model=model,
                model_cfg=config.model,
                device=device,
                episode=episode,
                anchor=anchor,
                prompt=prompt,
                normalized_state=norm_state,
                normalized_state_history=norm_hist,
                action_mask=action_mask,
                num_inference_steps=args.num_inference_steps,
                seed=window_seed,
            )
            pred_native = _postprocess_actions(
                pred_action_norm,
                norm_state,
                dataset_id=args.dataset,
                output_transforms=output_transforms,
            )
            next_boundary = anchors[win_i + 1] if win_i + 1 < len(anchors) else t_len
            for i in range(horizon):
                idx = anchor + i
                if idx >= next_boundary or idx >= t_len:
                    break
                pred_action_native[idx] = pred_native[i]
                covered[idx] = True
        if not np.all(covered):
            raise RuntimeError(f"Uncovered steps at ckpt step={step}")

        metrics = trajectory_metrics(pred_action_native, gt_native, horizon=horizon, stride=stride)
        metrics["elapsed_sec"] = time.perf_counter() - t0
        metrics["checkpoint"] = str(ckpt_path)
        results["steps"][str(step)] = metrics
        logging.info(
            "step=%d mae=%.4f D2=%.4f D2/gt=%.1fx window_D2=%.4f (%.1fs)",
            step,
            metrics["mae"],
            metrics["D2_error"],
            metrics["D2_error_over_gt_traj"],
            metrics["D2_error_per_window_mean"],
            metrics["elapsed_sec"],
        )

    out_path = args.output_json.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    logging.info("Wrote %s", out_path)


if __name__ == "__main__":
    main(tyro.cli(Args))
