#!/usr/bin/env python3
"""Stratified HPT validation flow loss L_v(tau) by timestep bucket."""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

import openpi.cotrain.config as cotrain_config
from openpi.cotrain import data_loader as cotrain_data_loader
from openpi.models.hpt_config import HPTConfig
from openpi.models_pytorch.hpt.model import _sample_beta

BUCKETS: list[tuple[float, float]] = [
    (0.0, 0.2),
    (0.2, 0.4),
    (0.4, 0.6),
    (0.6, 0.8),
    (0.8, 1.0),
]
BUCKET_LABELS = [f"{lo:g}-{hi:g}" for lo, hi in BUCKETS]
BUCKET_MIDPOINTS = [0.1, 0.3, 0.5, 0.7, 0.9]


def init_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")


def _bucket_index(t: float) -> int:
    if t >= 1.0:
        return len(BUCKETS) - 1
    for i, (lo, hi) in enumerate(BUCKETS):
        if lo <= t < hi or (hi == 1.0 and lo <= t <= hi):
            return i
    return len(BUCKETS) - 1


def beta_bucket_mass(alpha: float = 1.5, beta: float = 1.0) -> dict[str, float]:
    """P(t in bucket) for t ~ Beta(alpha,beta)."""
    from scipy import stats

    dist = stats.beta(alpha, beta)
    out: dict[str, float] = {}
    for (lo, hi), label in zip(BUCKETS, BUCKET_LABELS):
        out[label] = float(dist.cdf(hi) - dist.cdf(lo))
    return out


@torch.no_grad()
def _flow_action_loss_per_sample(
    hpt,
    sample: dict[str, Any],
    *,
    time: torch.Tensor,
    noise: torch.Tensor,
) -> torch.Tensor:
    """Per-sample FM velocity MSE (training loss_action, no smooth term)."""
    device = sample["base_0_rgb"].device
    obs_h = int(hpt.config.observation_horizon)
    base = sample["base_0_rgb"]
    base_obs = base[:, :obs_h] if base.ndim == 5 else base

    def _cam_obs(key: str) -> torch.Tensor | None:
        x = sample.get(key)
        if x is None:
            return None
        if x.ndim == 5:
            return x[:, :obs_h] if x.shape[1] > obs_h else x
        return x

    prompts = sample.get("prompts") or [""] * base_obs.shape[0]
    is_ego = sample.get("is_ego")
    if is_ego is None:
        is_ego = torch.zeros(base_obs.shape[0], device=device, dtype=torch.bool)
    else:
        is_ego = torch.as_tensor(is_ego, device=device, dtype=torch.bool).reshape(-1)

    state_hist = sample.get("state_history")
    if state_hist is None:
        state_hist = sample["state"]

    image_masks = {}
    for k in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"):
        m = sample.get(f"{k}_mask")
        if m is not None:
            image_masks[k] = m[:, :obs_h] if m.ndim == 2 and m.shape[1] > obs_h else m

    obs_tokens, _ = hpt.encode_obs(
        base_img=base_obs,
        left_wrist=_cam_obs("left_wrist_0_rgb"),
        right_wrist=_cam_obs("right_wrist_0_rgb"),
        state=state_hist,
        prompts=prompts,
        is_ego=is_ego,
        image_masks=image_masks or None,
    )
    _, action_features, _ = hpt.forward_trunk(obs_tokens)

    actions = sample["action"].float()
    action_mask = sample.get("action_mask")
    if action_mask is not None:
        action_mask_h = action_mask.to(device=device, dtype=torch.float32)
        if action_mask_h.ndim == 2:
            action_mask_h = action_mask_h[:, None, :].expand_as(actions)
        actions = actions * action_mask_h
    else:
        action_mask_h = torch.ones_like(actions)

    t = time[:, None, None]
    x_t = t * noise + (1.0 - t) * actions
    u_t = noise - actions
    x_t = x_t * action_mask_h
    u_t = u_t * action_mask_h

    v_t = hpt.action_head(x_t, time, action_features)
    action_sq = ((v_t - u_t) ** 2) * action_mask_h
    denom = action_mask_h.sum(dim=(1, 2)).clamp_min(1.0)
    return action_sq.sum(dim=(1, 2)) / denom


def _empty_bucket_stats() -> dict[str, dict[str, float]]:
    return {label: {"count": 0, "loss_sum": 0.0} for label in BUCKET_LABELS}


def _finalize_buckets(raw: dict[str, dict[str, float]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    total_count = sum(int(v["count"]) for v in raw.values())
    weighted_sum = 0.0
    for label in BUCKET_LABELS:
        count = int(raw[label]["count"])
        loss_sum = float(raw[label]["loss_sum"])
        mean = loss_sum / count if count > 0 else float("nan")
        out[label] = {
            "count": count,
            "fraction_of_samples": (count / total_count) if total_count > 0 else float("nan"),
            "loss_mean": mean,
        }
        if count > 0 and math.isfinite(mean):
            weighted_sum += mean * count
    out["_overall"] = {
        "count": total_count,
        "loss_mean": weighted_sum / total_count if total_count > 0 else float("nan"),
    }
    return out


@dataclasses.dataclass
class Args:
    checkpoint: Path = Path("checkpoints/hpt_cotrain_real_only/hpt-robot-ao/99999")
    output_json: Path = Path("tmp/hpt_denoise_sweep/hpt-robot-ao-99999/flow_loss_by_tau.json")
    config_name: str = "hpt_cotrain_real_only"
    dataset: str = "piper2"
    val_label: str = "seen"
    num_val_batches: int = 100
    val_batch_size: int = 32
    device: str = "cuda:0"
    seed: int = 42
    beta_alpha: float = 1.5
    beta_beta: float = 1.0


def main(args: Args) -> None:
    init_logging()
    import dataclasses as dc
    import os
    import safetensors.torch

    os.environ.setdefault("RLDS_DATA_DIR", "/mnt/bos/bo23lu")
    os.environ.setdefault("HF_HOME", "/data/zjyang/cache/huggingface")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    config = cotrain_config.get_config(args.config_name)
    if not isinstance(config.model, HPTConfig):
        raise TypeError(f"{args.config_name} is not HPT")

    keep = tuple(ds for ds in config.data.datasets if ds.uid == args.dataset)
    if not keep:
        raise ValueError(f"dataset {args.dataset!r} not in config")
    config = dc.replace(
        config,
        data=dc.replace(config.data, datasets=keep),
        val_batch_size=args.val_batch_size,
        num_val_batches=args.num_val_batches,
    )

    ckpt = args.checkpoint.expanduser().resolve()
    if ckpt.is_dir():
        ckpt = ckpt / "model.safetensors"

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = config.model.create_pytorch(device=str(device))
    if config.model.freeze_encoders:
        model.freeze_encoders()
    model.to(device)
    safetensors.torch.load_model(model, str(ckpt), strict=False, device=str(device))
    model.eval()
    hpt = model.hpt

    val_loaders = cotrain_data_loader.build_val_loaders(
        config, framework="pytorch", single_process=True
    )
    if args.val_label not in val_loaders or args.dataset not in val_loaders[args.val_label]:
        raise KeyError(f"val loader missing for label={args.val_label} dataset={args.dataset}")
    loader = val_loaders[args.val_label][args.dataset]

    beta_raw = _empty_bucket_stats()
    beta_losses_all: list[float] = []
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    for batch_i, (obs, acts) in enumerate(loader):
        if batch_i >= args.num_val_batches:
            break
        prompts = getattr(obs, "_fastwam_prompts", None)
        sample = model.observation_to_sample(obs, acts, prompts=prompts)
        b = sample["action"].shape[0]
        batch_seed = args.seed + batch_i * 10007
        g = torch.Generator(device=device)
        g.manual_seed(batch_seed)
        noise = torch.randn(sample["action"].shape, generator=g, device=device, dtype=torch.float32)
        time = _sample_beta(args.beta_alpha, args.beta_beta, b, device) * 0.999 + 0.001
        per = _flow_action_loss_per_sample(hpt, sample, time=time, noise=noise)
        per_cpu = per.detach().float().cpu().numpy()
        time_cpu = time.detach().float().cpu().numpy()
        for t_val, loss_val in zip(time_cpu, per_cpu):
            label = BUCKET_LABELS[_bucket_index(float(t_val))]
            beta_raw[label]["count"] += 1
            beta_raw[label]["loss_sum"] += float(loss_val)
            beta_losses_all.append(float(loss_val))

    beta_stats = _finalize_buckets(beta_raw)

    fixed_raw = _empty_bucket_stats()
    for bucket_i, (mid, label) in enumerate(zip(BUCKET_MIDPOINTS, BUCKET_LABELS)):
        torch.manual_seed(args.seed + bucket_i * 99991)
        for batch_i, (obs, acts) in enumerate(loader):
            if batch_i >= args.num_val_batches:
                break
            prompts = getattr(obs, "_fastwam_prompts", None)
            sample = model.observation_to_sample(obs, acts, prompts=prompts)
            b = sample["action"].shape[0]
            batch_seed = args.seed + batch_i * 10007
            g = torch.Generator(device=device)
            g.manual_seed(batch_seed)
            noise = torch.randn(sample["action"].shape, generator=g, device=device, dtype=torch.float32)
            time = torch.full((b,), mid, device=device, dtype=torch.float32)
            per = _flow_action_loss_per_sample(hpt, sample, time=time, noise=noise)
            for loss_val in per.detach().float().cpu().numpy().tolist():
                fixed_raw[label]["count"] += 1
                fixed_raw[label]["loss_sum"] += float(loss_val)
    fixed_stats = _finalize_buckets(fixed_raw)

    try:
        beta_mass = beta_bucket_mass(args.beta_alpha, args.beta_beta)
    except Exception:
        beta_mass = None

    low_label, high_label = BUCKET_LABELS[0], BUCKET_LABELS[-1]
    result: dict[str, Any] = {
        "checkpoint": str(ckpt),
        "config_name": args.config_name,
        "dataset": args.dataset,
        "val_label": args.val_label,
        "num_val_batches": args.num_val_batches,
        "val_batch_size": args.val_batch_size,
        "total_samples_beta": len(beta_losses_all),
        "time_convention": "t=0 clean action, t=1 pure noise; x_t = t*noise + (1-t)*action",
        "beta_sampling": {
            "alpha": args.beta_alpha,
            "beta": args.beta_beta,
            "theoretical_bucket_mass": beta_mass,
            "by_bucket": {k: v for k, v in beta_stats.items() if not k.startswith("_")},
            "overall_loss_mean": beta_stats["_overall"]["loss_mean"],
        },
        "fixed_tau_midpoints": {
            "midpoints": dict(zip(BUCKET_LABELS, BUCKET_MIDPOINTS)),
            "by_bucket": {k: v for k, v in fixed_stats.items() if not k.startswith("_")},
            "overall_loss_mean": fixed_stats["_overall"]["loss_mean"],
        },
        "ratios": {},
    }

    for mode_key, stats in [("beta_sampled", beta_stats), ("fixed_midpoint", fixed_stats)]:
        lo = stats[low_label]["loss_mean"]
        hi = stats[high_label]["loss_mean"]
        result["ratios"][mode_key] = {
            f"L_{high_label}_over_L_{low_label}": (hi / lo) if lo and math.isfinite(lo) and math.isfinite(hi) else float("nan"),
            f"L_{low_label}": lo,
            f"L_{high_label}": hi,
        }

    out_path = args.output_json.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    logging.info("Wrote %s", out_path)
    logging.info(
        "Beta-sampled: L[%s]=%.4f L[%s]=%.4f ratio=%.2f overall=%.4f",
        low_label,
        result["ratios"]["beta_sampled"][f"L_{low_label}"],
        high_label,
        result["ratios"]["beta_sampled"][f"L_{high_label}"],
        result["ratios"]["beta_sampled"][f"L_{high_label}_over_L_{low_label}"],
        beta_stats["_overall"]["loss_mean"],
    )
    logging.info(
        "Fixed-mid:    L[%s]=%.4f L[%s]=%.4f ratio=%.2f overall=%.4f",
        low_label,
        result["ratios"]["fixed_midpoint"][f"L_{low_label}"],
        high_label,
        result["ratios"]["fixed_midpoint"][f"L_{high_label}"],
        result["ratios"]["fixed_midpoint"][f"L_{high_label}_over_L_{low_label}"],
        fixed_stats["_overall"]["loss_mean"],
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
