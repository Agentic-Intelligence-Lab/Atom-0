#!/usr/bin/env python3
"""Full-episode HPT action inference (slide windows, stitch, overlay).

Mirrors ``run_fastwam_joint_episode.py`` but **action-only** (no video):

  gt_action_native_14d.{npy,png,json}
  pred_action_native_14d.{npy,png,json}
  action_native_14d_overlay.png
  meta.json / run_summary.json

Example (piper2 seen_test, pretrain ckpt 99999)::

  cd /data/zjyang/Atom-0 && source scripts/atom0_env.sh
  export HF_HOME=/data/zjyang/cache/huggingface
  export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  PYTHONPATH=src .venv/bin/python scripts/run_hpt_episode.py \\
    --checkpoint checkpoints/hpt_cotrain_real_only/hpt-real-bs224-1e5-100k/99999 \\
    --dataset piper2 --split seen_test --prefer-short \\
    --output-dir tmp/hpt_episode/piper2-99999
"""

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

_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from run_fastwam_joint_episode import (  # noqa: E402
    CAMERA_SLOTS,
    RAW_CAM_KEYS,
    _decode_text,
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

# Matching piper RLDS / FastWAM joint training prefix (HPT T5 path usually sees task
# text only; keep optional so you can A/B).
DEFAULT_PROMPT_PREFIX = "Action Mode: joint. "


@dataclasses.dataclass
class Args:
    checkpoint: Path
    """Step dir, exp dir, or ``model.safetensors``."""

    config_name: str = "hpt_cotrain_real_only"
    dataset: str = "piper2"
    """``piper2`` or ``piper30`` (must be in config data mixture)."""

    split: str = "seen_test"
    """TFDS split: train / seen_test / unseen_test."""

    episode_index: int | None = None
    prefer_short: bool = False
    stride: int | None = None
    """Action stitch stride. Default = action_horizon (non-overlapping)."""

    output_dir: Path = Path("tmp/hpt_episode")
    device: str = "cuda:0"
    num_inference_steps: int | None = None
    """Euler steps for flow sampling; default = model.num_inference_steps."""
    seed: int = 42
    rlds_data_dir: Path | None = None
    assets_base_dir: Path | None = None

    prompt_prefix: str | None = None
    """If set, prepended to episode task. Default: no prefix (matches HPT train collate)."""
    use_joint_prompt_prefix: bool = False
    """If True, use ``Action Mode: joint. `` (same as FastWAM episode script)."""

    embed_dim: int | None = None
    """Override ``HPTConfig.embed_dim`` (needed for older 128-d ckpts)."""
    action_head_type: str | None = None
    """Override head type; default infers from checkpoint keys."""
    action_head_dim: int | None = None
    """Override action-head hidden size (e.g. DiT-128 used 256)."""

    head_mode: str | None = None
    """``action_world`` | ``action_only``. Default: infer from checkpoint (world head present?)."""


def _configure_rlds_root(args: Args) -> None:
    if args.rlds_data_dir is not None:
        os.environ["RLDS_DATA_DIR"] = str(args.rlds_data_dir)
    elif "RLDS_DATA_DIR" not in os.environ:
        os.environ["RLDS_DATA_DIR"] = "/mnt/bos/bo23lu"


def _configure_hf_offline() -> None:
    """Prefer PFS HF cache so DINOv2/T5 init works offline before ckpt overwrite."""
    os.environ.setdefault("HF_HOME", "/data/zjyang/cache/huggingface")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(os.environ["HF_HOME"], "hub"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _builder_dir_for_dataset(dataset: str) -> Path:
    import openpi.cotrain.config as cotrain_config

    if dataset == "piper2":
        return Path(cotrain_config._PIPER2_BUILDER_DIR)
    if dataset == "piper30":
        return Path(cotrain_config._PIPER30_BUILDER_DIR)
    raise ValueError(f"Supported datasets: piper2, piper30; got {dataset!r}")


def _infer_head_mode(ckpt_path: Path) -> str:
    import safetensors.torch

    keys = safetensors.torch.load_file(str(ckpt_path)).keys()
    if any("world_head" in k or "future_tokens" in k for k in keys):
        return "action_world"
    return "action_only"


def _infer_action_head_type(ckpt_path: Path) -> str:
    import safetensors.torch

    keys = [k for k in safetensors.torch.load_file(str(ckpt_path)).keys() if k.startswith("hpt.action_head.")]
    if any("adaLN_modulation" in k or "final_adaLN" in k for k in keys):
        return "dit"
    if any("unet." in k for k in keys):
        return "diffusion"
    if any("query_tokens" in k for k in keys):
        return "transformer_decoder"
    if any("blocks." in k for k in keys):
        return "cross_transformer"
    return "mlp"


def _resolve_checkpoint(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file():
        return path
    direct = path / "model.safetensors"
    if direct.is_file():
        return direct
    # Exp dir: pick highest numeric step with model.safetensors.
    steps: list[tuple[int, Path]] = []
    if path.is_dir():
        for child in path.iterdir():
            if not child.is_dir() or not child.name.isdigit():
                continue
            cand = child / "model.safetensors"
            if cand.is_file():
                steps.append((int(child.name), cand))
    if steps:
        steps.sort(key=lambda x: x[0])
        return steps[-1][1]
    raise FileNotFoundError(f"No model.safetensors under {path}")


def _resize_hwc_uint8(img: np.ndarray, height: int, width: int) -> np.ndarray:
    from openpi_client import image_tools

    return image_tools.resize_with_pad(img, height, width)


def _uint8_to_model_float(img: np.ndarray) -> np.ndarray:
    """HWC uint8 -> HWC float32 in [-1, 1] (same convention as openpi Observation)."""
    return img.astype(np.float32) / 255.0 * 2.0 - 1.0


def _infer_window(
    *,
    model,
    model_cfg,
    device: torch.device,
    episode: dict[str, Any],
    anchor: int,
    prompt: str,
    normalized_state: np.ndarray,
    normalized_state_history: np.ndarray,
    action_mask: np.ndarray,
    num_inference_steps: int,
    seed: int,
) -> np.ndarray:
    """Run HPT ``sample_actions`` at ``anchor``. Returns pred action in norm space [H, 80]."""
    h, w = model_cfg.image_resolution
    obs_h = int(getattr(model_cfg, "observation_horizon", 1))
    sample: dict[str, Any] = {
        "prompts": [prompt],
        "is_ego": torch.zeros(1, device=device, dtype=torch.bool),
        "state": torch.as_tensor(
            normalized_state[: model_cfg.proprio_dim],
            device=device,
            dtype=torch.float32,
        ).unsqueeze(0),
        "state_history": torch.as_tensor(
            normalized_state_history[..., : model_cfg.proprio_dim],
            device=device,
            dtype=torch.float32,
        ).unsqueeze(0),
        "action_mask": torch.as_tensor(action_mask, device=device, dtype=torch.bool).unsqueeze(0),
    }
    for slot in CAMERA_SLOTS:
        if slot not in episode["images"]:
            continue
        frames = []
        for dt in range(obs_h - 1, -1, -1):
            idx = max(0, anchor - dt)
            frames.append(_uint8_to_model_float(_resize_hwc_uint8(episode["images"][slot][idx], h, w)))
        arr = np.stack(frames, axis=0)[None, ...]  # [1, T, H, W, 3]
        sample[slot] = torch.from_numpy(arr).to(device=device, dtype=torch.float32)
        sample[f"{slot}_mask"] = torch.ones(1, obs_h, device=device, dtype=torch.bool)

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    with torch.no_grad():
        pred = model.hpt.sample_actions(sample, num_steps=num_inference_steps)
    pred_np = pred[0].detach().float().cpu().numpy()
    # Zero padded unified dims (piper uses 14/80).
    pred_np = pred_np * action_mask.astype(np.float64)[None, :]
    return pred_np.astype(np.float32)


def main(args: Args) -> None:
    init_logging()
    _configure_rlds_root(args)
    _configure_hf_offline()

    import openpi.cotrain.config as cotrain_config
    from openpi.cotrain import action_space
    from openpi.cotrain.config import load_per_dataset_norm_stats
    from openpi.models.hpt_config import HPTConfig
    import safetensors.torch

    if args.dataset not in ("piper2", "piper30"):
        raise ValueError(f"Unsupported dataset={args.dataset!r}; use piper2 or piper30")

    config = cotrain_config.get_config(args.config_name)
    if not isinstance(config.model, HPTConfig):
        raise TypeError(f"{args.config_name} is not an HPT config (got {type(config.model)})")

    keep = tuple(ds for ds in config.data.datasets if ds.uid == args.dataset)
    if not keep:
        raise ValueError(f"dataset={args.dataset!r} not in {args.config_name} data mixture")

    replace_kw: dict[str, Any] = {
        "exp_name": "hpt_episode",
        "wandb_enabled": False,
        "data": dataclasses.replace(config.data, datasets=keep),
    }
    if args.assets_base_dir is not None:
        replace_kw["assets_base_dir"] = str(args.assets_base_dir)
    config = dataclasses.replace(config, **replace_kw)

    ckpt_path = _resolve_checkpoint(args.checkpoint)

    head_mode = args.head_mode or _infer_head_mode(ckpt_path)
    if head_mode not in ("action_only", "action_world"):
        raise ValueError(f"head_mode must be action_only or action_world, got {head_mode!r}")
    action_head_type = args.action_head_type or _infer_action_head_type(ckpt_path)
    model_overrides: dict[str, Any] = {
        "head_mode": head_mode,
        "action_head_type": action_head_type,
    }
    if args.embed_dim is not None:
        model_overrides["embed_dim"] = int(args.embed_dim)
    if args.action_head_dim is not None:
        model_overrides["action_head_dim"] = int(args.action_head_dim)
    if model_overrides:
        config = dataclasses.replace(
            config, model=dataclasses.replace(config.model, **model_overrides)
        )
        logging.info("Model overrides: %s", model_overrides)

    horizon = int(config.model.action_horizon)
    stride = int(args.stride) if args.stride is not None else horizon
    num_steps = (
        int(args.num_inference_steps)
        if args.num_inference_steps is not None
        else int(config.model.num_inference_steps)
    )

    builder_dir = _builder_dir_for_dataset(args.dataset)
    logging.info("Selecting episode from %s split=%s", builder_dir, args.split)
    ordinal, meta_row = _select_episode_ordinal(
        builder_dir,
        args.split,
        episode_index=args.episode_index,
        prefer_short=args.prefer_short,
        min_frames=horizon,
    )
    logging.info(
        "Selected episode_index=%s ordinal=%s T=%s task=%r",
        meta_row["episode_index"],
        ordinal,
        meta_row["num_frames"],
        meta_row["task"],
    )

    logging.info("Loading full episode frames...")
    episode = _load_full_episode(builder_dir, args.split, ordinal)
    t_len = episode["num_frames"]
    anchors = _make_anchors(t_len, horizon, stride)
    logging.info("Anchors (%d, stride=%d): %s ...", len(anchors), stride, anchors[:8])

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        logging.warning("CUDA unavailable; HPT inference will be slow on CPU.")

    logging.info("Creating HPT model on %s (head_mode=%s) ...", device, head_mode)
    model = config.model.create_pytorch(device=str(device))
    if config.model.freeze_encoders:
        model.freeze_encoders()
    model.to(device)
    logging.info("Loading weights from %s", ckpt_path)
    # Encoders may already match HF cache; allow missing processor-only keys if any.
    missing, unexpected = safetensors.torch.load_model(
        model, str(ckpt_path), strict=False, device=str(device)
    )
    if missing:
        logging.warning("Missing keys (%d): %s", len(missing), list(missing)[:8])
    if unexpected:
        logging.warning("Unexpected keys (%d): %s", len(unexpected), list(unexpected)[:8])
    model.eval()

    data_config = config.data.create(config.assets_dirs, config.model)
    output_transforms = _build_output_transforms(config, data_config)
    per_dataset_stats = load_per_dataset_norm_stats(config.assets_dirs, data_config.datasets)
    norm_stats = per_dataset_stats[args.dataset]
    spec = action_space.UNIFIED_ACTION_SPECS[args.dataset]
    action_mask = np.asarray(spec.action_mask, dtype=np.bool_)
    native_dim = int(keep[0].action_dim) if keep[0].action_dim > 0 else 14

    if args.use_joint_prompt_prefix:
        prefix = DEFAULT_PROMPT_PREFIX
    elif args.prompt_prefix is not None:
        prefix = args.prompt_prefix
    else:
        prefix = ""
    prompt = f"{prefix}{episode['prompt']}"

    pred_action_native = np.full((t_len, native_dim), np.nan, dtype=np.float64)
    covered_action = np.zeros(t_len, dtype=bool)

    t0 = time.perf_counter()
    for win_i, anchor in enumerate(anchors):
        seed = args.seed + win_i
        obs_h = int(getattr(config.model, "observation_horizon", 1))
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
        logging.info(
            "Window %d/%d anchor=%d seed=%d",
            win_i + 1,
            len(anchors),
            anchor,
            seed,
        )
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
            num_inference_steps=num_steps,
            seed=seed,
        )
        pred_native = _postprocess_actions(
            pred_action_norm,
            norm_state,
            dataset_id=args.dataset,
            output_transforms=output_transforms,
        )
        if pred_native.shape[-1] != native_dim:
            raise ValueError(
                f"native action dim mismatch: got {pred_native.shape[-1]}, expected {native_dim}"
            )

        next_boundary = anchors[win_i + 1] if win_i + 1 < len(anchors) else t_len
        for i in range(horizon):
            idx = anchor + i
            if idx >= next_boundary or idx >= t_len:
                break
            pred_action_native[idx] = pred_native[i]
            covered_action[idx] = True

    if not np.all(covered_action):
        missing = np.flatnonzero(~covered_action)
        raise RuntimeError(f"Uncovered action steps after stitch: {missing[:20]}...")

    gt_native = episode["action"].astype(np.float64)
    out_dir = args.output_dir / f"episode_{episode['episode_index']:05d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    traj_info = _save_native_trajectories(
        out_dir,
        dataset_id=args.dataset,
        gt_native=gt_native,
        pred_native=pred_action_native,
    )

    meta = {
        "model": "hpt",
        "dataset": args.dataset,
        "split": args.split,
        "episode_index": episode["episode_index"],
        "split_ordinal": ordinal,
        "num_frames": t_len,
        "task": episode["task"],
        "prompt": prompt,
        "action_horizon": horizon,
        "stride": stride,
        "anchors": anchors,
        "num_windows": len(anchors),
        "num_inference_steps": num_steps,
        "image_resolution": list(config.model.image_resolution),
        "checkpoint": str(ckpt_path),
        "config_name": args.config_name,
        "head_mode": head_mode,
        "gt_trajectory": f"gt_action_native_{traj_info['native_dim']}d.npy",
        "pred_trajectory": f"pred_action_native_{traj_info['native_dim']}d.npy",
        "overlay": f"action_native_{traj_info['native_dim']}d_overlay.png",
        "trajectory_mae": traj_info["mae"],
        "dim_names": traj_info["dim_names"],
        "elapsed_sec": time.perf_counter() - t0,
        "note": "Action-only HPT episode stitch; no video.",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "output_dir": str(out_dir),
                "episode_index": episode["episode_index"],
                "num_frames": t_len,
                "num_windows": len(anchors),
                "trajectory_mae": traj_info["mae"],
                "elapsed_sec": meta["elapsed_sec"],
                "checkpoint": str(ckpt_path),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    logging.info(
        "Done episode %s -> %s (T=%d windows=%d mae=%.4f %.1fs)",
        episode["episode_index"],
        out_dir,
        t_len,
        len(anchors),
        traj_info["mae"],
        meta["elapsed_sec"],
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
