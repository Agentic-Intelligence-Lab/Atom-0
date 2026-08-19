#!/usr/bin/env python3
"""Sweep HPT num_inference_steps on one fixed episode; measure trajectory jitter."""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
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
    _save_native_trajectories,
    init_logging,
)
from run_hpt_episode import (  # noqa: E402
    _builder_dir_for_dataset,
    _configure_hf_offline,
    _configure_rlds_root,
    _infer_head_mode,
    _infer_window,
    _resolve_checkpoint,
)


@dataclasses.dataclass
class Args:
    checkpoint: Path = Path("checkpoints/hpt_cotrain_real_only/hpt-robot-ao/99999")
    output_dir: Path = Path("tmp/hpt_denoise_sweep/hpt-robot-ao-99999")
    config_name: str = "hpt_cotrain_real_only"
    dataset: str = "piper2"
    split: str = "seen_test"
    episode_index: int = 117
    seed: int = 42
    device: str = "cuda:0"
    inference_steps: tuple[int, ...] = (10, 20, 50, 100, 200)
    rlds_data_dir: Path | None = None


def _jitter_metrics(pred: np.ndarray, gt: np.ndarray, anchors: list[int]) -> dict[str, float]:
    """High-frequency proxies on stitched native trajectory [T, D]."""
    pred = pred.astype(np.float64)
    gt = gt.astype(np.float64)
    d_pred = np.abs(np.diff(pred, axis=0))
    d_gt = np.abs(np.diff(gt, axis=0))
    consec_pred = float(d_pred.mean())
    consec_gt = float(d_gt.mean())
    jitter_ratio = consec_pred / max(consec_gt, 1e-12)

    if pred.shape[0] >= 3:
        d2_pred = np.abs(pred[2:] - 2.0 * pred[1:-1] + pred[:-2])
        d2_gt = np.abs(gt[2:] - 2.0 * gt[1:-1] + gt[:-2])
        second_order_pred = float(d2_pred.mean())
        second_order_gt = float(d2_gt.mean())
        second_order_ratio = second_order_pred / max(second_order_gt, 1e-12)
    else:
        second_order_pred = second_order_gt = second_order_ratio = float("nan")

    seam_idx = [a for a in anchors[1:] if 0 < a < pred.shape[0]]
    if seam_idx:
        seam_jumps = [float(np.abs(pred[i] - pred[i - 1]).mean()) for i in seam_idx]
        median_seam = float(np.median(seam_jumps))
    else:
        median_seam = float("nan")

    within_mask = np.ones(pred.shape[0] - 1, dtype=bool)
    for a in seam_idx:
        if 1 <= a < pred.shape[0]:
            within_mask[a - 1] = False
    if within_mask.any():
        median_within = float(np.median(d_pred[within_mask].mean(axis=1)))
    else:
        median_within = float(np.median(d_pred.mean(axis=1)))

    seam_over_within = median_seam / max(median_within, 1e-12) if np.isfinite(median_seam) else float("nan")
    mae = float(np.abs(pred - gt).mean())
    return {
        "trajectory_mae": mae,
        "consec_abs_diff_pred": consec_pred,
        "consec_abs_diff_gt": consec_gt,
        "jitter_ratio_pred_over_gt": jitter_ratio,
        "second_order_pred": second_order_pred,
        "second_order_gt": second_order_gt,
        "second_order_ratio_pred_over_gt": second_order_ratio,
        "median_within_chunk_consec_abs": median_within,
        "median_seam_abs_jump": median_seam,
        "seam_over_within_ratio": seam_over_within,
    }


def main(args: Args) -> None:
    init_logging()
    _configure_rlds_root(
        type("A", (), {"rlds_data_dir": args.rlds_data_dir})()
    )
    _configure_hf_offline()

    import dataclasses as dc
    import safetensors.torch

    import openpi.cotrain.config as cotrain_config
    from openpi.cotrain import action_space
    from openpi.cotrain.config import load_per_dataset_norm_stats
    from openpi.models.hpt_config import HPTConfig

    config = cotrain_config.get_config(args.config_name)
    if not isinstance(config.model, HPTConfig):
        raise TypeError(f"{args.config_name} is not HPT")
    keep = tuple(ds for ds in config.data.datasets if ds.uid == args.dataset)
    config = dc.replace(config, data=dc.replace(config.data, datasets=keep))

    ckpt_path = _resolve_checkpoint(args.checkpoint)
    head_mode = _infer_head_mode(ckpt_path)
    config = dc.replace(config, model=dc.replace(config.model, head_mode=head_mode))

    horizon = int(config.model.action_horizon)
    stride = horizon
    builder_dir = _builder_dir_for_dataset(args.dataset)
    ordinal, meta_row = _select_episode_ordinal(
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

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logging.info("Loading HPT on %s from %s", device, ckpt_path)
    model = config.model.create_pytorch(device=str(device))
    if config.model.freeze_encoders:
        model.freeze_encoders()
    model.to(device)
    safetensors.torch.load_model(model, str(ckpt_path), strict=False, device=str(device))
    model.eval()

    data_config = config.data.create(config.assets_dirs, config.model)
    output_transforms = _build_output_transforms(config, data_config)
    per_dataset_stats = load_per_dataset_norm_stats(config.assets_dirs, data_config.datasets)
    norm_stats = per_dataset_stats[args.dataset]
    spec = action_space.UNIFIED_ACTION_SPECS[args.dataset]
    action_mask = np.asarray(spec.action_mask, dtype=np.bool_)
    native_dim = int(keep[0].action_dim) if keep[0].action_dim > 0 else 14
    obs_h = int(getattr(config.model, "observation_horizon", 1))

    out_root = args.output_dir.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {
        "checkpoint": str(ckpt_path),
        "dataset": args.dataset,
        "split": args.split,
        "episode_index": episode["episode_index"],
        "seed": args.seed,
        "action_horizon": horizon,
        "stride": stride,
        "num_frames": t_len,
        "task": episode["task"],
        "head_mode": head_mode,
        "note": "Fixed obs/prompt/seed; only num_inference_steps varies.",
        "runs": {},
    }

    gt_native = episode["action"].astype(np.float64)

    for num_steps in args.inference_steps:
        logging.info("=== num_inference_steps=%d ===", num_steps)
        t0 = time.perf_counter()
        pred_action_native = np.zeros((t_len, native_dim), dtype=np.float64)
        covered = np.zeros(t_len, dtype=bool)

        for win_i, anchor in enumerate(anchors):
            seed = args.seed + win_i
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
                normalized_state=hist[-1],
                normalized_state_history=norm_hist,
                action_mask=action_mask,
                num_inference_steps=num_steps,
                seed=seed,
            )
            pred_native = _postprocess_actions(
                pred_action_norm,
                hist[-1],
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
            raise RuntimeError(f"Uncovered steps for steps={num_steps}")

        run_dir = out_root / f"steps_{num_steps:03d}" / f"episode_{episode['episode_index']:05d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        traj_info = _save_native_trajectories(
            run_dir,
            dataset_id=args.dataset,
            gt_native=gt_native,
            pred_native=pred_action_native,
        )
        metrics = _jitter_metrics(pred_action_native, gt_native, anchors)
        metrics["num_inference_steps"] = num_steps
        metrics["elapsed_sec"] = time.perf_counter() - t0
        metrics["overlay"] = str(run_dir / f"action_native_{native_dim}d_overlay.png")
        results["runs"][str(num_steps)] = metrics
        logging.info(
            "steps=%d mae=%.4f jitter_ratio=%.2f second_order_ratio=%.2f seam/within=%.3f (%.1fs)",
            num_steps,
            metrics["trajectory_mae"],
            metrics["jitter_ratio_pred_over_gt"],
            metrics["second_order_ratio_pred_over_gt"],
            metrics["seam_over_within_ratio"],
            metrics["elapsed_sec"],
        )

    summary_path = out_root / "sweep_summary.json"
    summary_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    logging.info("Wrote %s", summary_path)


if __name__ == "__main__":
    main(tyro.cli(Args))
