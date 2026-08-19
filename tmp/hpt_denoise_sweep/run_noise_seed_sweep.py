#!/usr/bin/env python3
"""Sweep HPT flow noise seed (epsilon) on one fixed episode; measure trajectory jitter."""

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
_SWEEP = Path(__file__).resolve().parent
for p in (_SCRIPTS, _SWEEP):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

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
from run_inference_steps_sweep import _jitter_metrics  # noqa: E402


@dataclasses.dataclass
class Args:
    checkpoint: Path = Path("checkpoints/hpt_cotrain_real_only/hpt-robot-ao/99999")
    output_dir: Path = Path("tmp/hpt_denoise_sweep/hpt-robot-ao-99999-seed-sweep")
    config_name: str = "hpt_cotrain_real_only"
    dataset: str = "piper2"
    split: str = "seen_test"
    episode_index: int = 117
    num_inference_steps: int = 50
    """Fixed Euler steps (same as training default)."""
    seeds: tuple[int, ...] = tuple(range(20))
    """20 base seeds for flow noise epsilon (window i uses seed + i)."""
    device: str = "cuda:0"
    save_overlays: bool = True
    rlds_data_dir: Path | None = None


def _aggregate(metrics_list: list[dict[str, float]]) -> dict[str, float]:
    keys = [
        "trajectory_mae",
        "jitter_ratio_pred_over_gt",
        "second_order_ratio_pred_over_gt",
        "seam_over_within_ratio",
        "consec_abs_diff_pred",
        "second_order_pred",
    ]
    out: dict[str, float] = {}
    for k in keys:
        vals = [m[k] for m in metrics_list if k in m and np.isfinite(m[k])]
        if vals:
            out[f"{k}_mean"] = float(np.mean(vals))
            out[f"{k}_std"] = float(np.std(vals))
            out[f"{k}_min"] = float(np.min(vals))
            out[f"{k}_max"] = float(np.max(vals))
    return out


def main(args: Args) -> None:
    init_logging()
    _configure_rlds_root(type("A", (), {"rlds_data_dir": args.rlds_data_dir})())
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

    device = __import__("torch").device(args.device if __import__("torch").cuda.is_available() else "cpu")
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

    gt_native = episode["action"].astype(np.float64)
    results: dict[str, Any] = {
        "checkpoint": str(ckpt_path),
        "dataset": args.dataset,
        "split": args.split,
        "episode_index": episode["episode_index"],
        "num_inference_steps": args.num_inference_steps,
        "seeds": list(args.seeds),
        "action_horizon": horizon,
        "stride": stride,
        "num_frames": t_len,
        "task": episode["task"],
        "head_mode": head_mode,
        "note": "Fixed obs/prompt/inference_steps; only flow-noise base seed varies.",
        "runs": {},
    }

    all_preds: list[np.ndarray] = []
    all_metrics: list[dict[str, float]] = []

    for base_seed in args.seeds:
        logging.info("=== seed=%d ===", base_seed)
        t0 = time.perf_counter()
        pred_action_native = np.zeros((t_len, native_dim), dtype=np.float64)
        covered = np.zeros(t_len, dtype=bool)

        for win_i, anchor in enumerate(anchors):
            window_seed = base_seed + win_i
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
            raise RuntimeError(f"Uncovered steps for seed={base_seed}")

        metrics = _jitter_metrics(pred_action_native, gt_native, anchors)
        metrics["seed"] = base_seed
        metrics["elapsed_sec"] = time.perf_counter() - t0
        all_preds.append(pred_action_native.copy())
        all_metrics.append(metrics)

        overlay_path = None
        if args.save_overlays:
            run_dir = out_root / f"seed_{base_seed:04d}" / f"episode_{episode['episode_index']:05d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            _save_native_trajectories(
                run_dir,
                dataset_id=args.dataset,
                gt_native=gt_native,
                pred_native=pred_action_native,
            )
            overlay_path = str(run_dir / f"action_native_{native_dim}d_overlay.png")
            metrics["overlay"] = overlay_path

        results["runs"][str(base_seed)] = metrics
        logging.info(
            "seed=%d mae=%.4f jitter_ratio=%.2f second_order_ratio=%.2f (%.1fs)",
            base_seed,
            metrics["trajectory_mae"],
            metrics["jitter_ratio_pred_over_gt"],
            metrics["second_order_ratio_pred_over_gt"],
            metrics["elapsed_sec"],
        )

    # Cross-seed variability of full trajectories.
    stacked = np.stack(all_preds, axis=0)  # [S, T, D]
    ref = stacked[0]
    mae_to_ref = [float(np.abs(p - ref).mean()) for p in all_preds[1:]]
    results["cross_seed"] = {
        "num_seeds": len(args.seeds),
        "pred_std_over_time_dims": float(stacked.std(axis=0).mean()),
        "pairwise_mae_to_seed0_mean": float(np.mean(mae_to_ref)) if mae_to_ref else 0.0,
        "pairwise_mae_to_seed0_std": float(np.std(mae_to_ref)) if mae_to_ref else 0.0,
        "aggregate_metrics": _aggregate(all_metrics),
    }

    summary_path = out_root / "seed_sweep_summary.json"
    summary_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    agg = results["cross_seed"]["aggregate_metrics"]
    logging.info(
        "Done %d seeds. jitter_ratio mean=%.2f±%.2f mae mean=%.4f±%.4f -> %s",
        len(args.seeds),
        agg.get("jitter_ratio_pred_over_gt_mean", float("nan")),
        agg.get("jitter_ratio_pred_over_gt_std", float("nan")),
        agg.get("trajectory_mae_mean", float("nan")),
        agg.get("trajectory_mae_std", float("nan")),
        summary_path,
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
