#!/usr/bin/env python
"""Standalone FastWAM val metrics via ``openpi.cotrain.fastwam_eval.run_eval``.

Uses the same RLDS val loaders + metric keys as training-time eval in ``train_fastwam.py``.

Example (piper30 / 576×512 ckpt)::

  cd /data/zjyang/Atom-0 && source scripts/atom0_env.sh
  export DIFFSYNTH_MODEL_BASE_PATH=\"$(pwd)/checkpoints/fastwam\"
  PYTHONPATH=src .venv/bin/python scripts/run_fastwam_eval.py \\
    --config-name wam-cross-piper \\
    --checkpoint checkpoints/wam-cross-piper/fw-wam-cross-piper-v1-8gpu-b208-30k/29999 \\
    --dataset piper30 \\
    --image-resolution 576,512 \\
    --output-dir tmp/fastwam_eval/piper-v1-29999-piper30
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import time
from pathlib import Path

import tyro


def init_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


@dataclasses.dataclass
class Args:
    config_name: str
    """Cotrain FastWAM config (architecture / assets / data mixture)."""

    checkpoint: Path
    """Step dir, exp dir, or ``model.safetensors`` path."""

    dataset: str | None = None
    """If set, only evaluate this dataset uid (e.g. ``piper30``)."""

    image_resolution: str | None = None
    """Override model compose size as ``H,W`` (e.g. ``576,512``). Match the ckpt's train res."""

    output_dir: Path = Path("tmp/fastwam_eval")
    """Writes ``metrics.json`` and ``run_summary.json`` here."""

    device: str = "cuda:0"
    assets_base_dir: Path | None = None
    rlds_data_dir: Path | None = None

    num_val_batches: int = 20
    num_action_mse_batches: int = 5
    run_action_mse: bool = True
    val_flow_loss_mode: str = "fixed_seed"
    val_batch_size: int = 4
    val_seed: int = 0


def _parse_hw(text: str) -> tuple[int, int]:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 2:
        raise ValueError(f"--image-resolution must be H,W got {text!r}")
    return int(parts[0]), int(parts[1])


def _configure_rlds_root(args: Args) -> None:
    if args.rlds_data_dir is not None:
        os.environ["RLDS_DATA_DIR"] = str(args.rlds_data_dir)
    elif "RLDS_DATA_DIR" not in os.environ:
        os.environ["RLDS_DATA_DIR"] = "/mnt/bos/bo23lu"


def _filter_loaders(loaders: dict, dataset: str | None) -> dict:
    if not dataset:
        return loaders
    out: dict = {}
    for label, by_name in loaders.items():
        if dataset in by_name:
            out[label] = {dataset: by_name[dataset]}
    if not out:
        raise ValueError(
            f"Dataset {dataset!r} not found in val loaders. "
            f"Available: { {lab: sorted(d) for lab, d in loaders.items()} }"
        )
    return out


def main(args: Args) -> None:
    init_logging()
    _configure_rlds_root(args)

    # Import after RLDS_DATA_DIR is set (builder paths are captured at import time).
    import torch

    import openpi.cotrain.config as cotrain_config
    import openpi.cotrain.data_loader as cotrain_data_loader
    import openpi.cotrain.fastwam_eval as fastwam_eval
    from openpi.cotrain.fastwam_checkpoint import load_fastwam_checkpoint
    from openpi.models.fastwam_config import FastWAMConfig

    config = cotrain_config.get_config(args.config_name)
    if not isinstance(config.model, FastWAMConfig):
        raise TypeError(f"{args.config_name} is not a FastWAM config")

    model_cfg = config.model
    if args.image_resolution:
        hw = _parse_hw(args.image_resolution)
        model_cfg = dataclasses.replace(model_cfg, image_resolution=hw)
        logging.info("Override image_resolution=%s", hw)

    # Avoid re-downloading Wan pretrain weights; ckpt provides full MoT/VAE.
    model_cfg = dataclasses.replace(
        model_cfg,
        skip_dit_load_from_pretrain=True,
        skip_vae_load_from_pretrain=True,
    )

    replace_kw: dict = {
        "model": model_cfg,
        "exp_name": "eval",
        "wandb_enabled": False,
        "eval_interval": 1,
        "num_val_batches": args.num_val_batches,
        "num_action_mse_batches": args.num_action_mse_batches,
        "run_action_mse": args.run_action_mse,
        "val_flow_loss_mode": args.val_flow_loss_mode,
        "val_batch_size": args.val_batch_size,
        "val_seed": args.val_seed,
        "val_max_datasets": None,
    }
    if args.assets_base_dir is not None:
        replace_kw["assets_base_dir"] = str(args.assets_base_dir)
    if args.dataset is not None:
        keep = tuple(ds for ds in config.data.datasets if ds.uid == args.dataset)
        if not keep:
            raise ValueError(
                f"dataset={args.dataset!r} not in {args.config_name}; "
                f"have {[d.uid for d in config.data.datasets]}"
            )
        weight_sum = sum(ds.weight for ds in keep) or 1.0
        keep = tuple(dataclasses.replace(ds, weight=ds.weight / weight_sum) for ds in keep)
        replace_kw["data"] = dataclasses.replace(config.data, datasets=keep)

    config = dataclasses.replace(config, **replace_kw)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        logging.warning("CUDA unavailable; running on CPU (very slow).")

    logging.info(
        "Building model res=%s cameras=%s concat=%s device=%s",
        config.model.image_resolution,
        config.model.camera_keys,
        config.model.concat_multi_camera,
        device,
    )
    model = config.model.create_pytorch(device=str(device))
    model.freeze_encoders()
    model.to(device)
    ckpt_path = load_fastwam_checkpoint(model, args.checkpoint, strict=True)
    model.eval()
    logging.info("Loaded weights from %s", ckpt_path)

    logging.info("Building validation loaders (single-process)...")
    val_loaders = cotrain_data_loader.build_val_loaders(
        config,
        framework="pytorch",
        single_process=True,
    )
    val_loaders = _filter_loaders(val_loaders, args.dataset)
    train_weights = cotrain_data_loader.dataset_train_weights(config)
    action_masks = cotrain_data_loader.dataset_action_masks(config)
    logging.info(
        "Val loaders: %s | flow_mode=%s run_action_mse=%s num_val_batches=%s",
        {lab: list(d) for lab, d in val_loaders.items()},
        config.val_flow_loss_mode,
        config.run_action_mse,
        config.num_val_batches,
    )

    t0 = time.perf_counter()
    metrics = fastwam_eval.run_eval(
        val_loaders,
        model,
        device=device,
        flow_mode=config.val_flow_loss_mode,
        run_action_mse=config.run_action_mse,
        num_val_batches=config.num_val_batches,
        num_action_mse_batches=config.num_action_mse_batches,
        val_flow_loss_num_samples=config.val_flow_loss_num_samples,
        action_mse_num_denoise_steps=config.action_mse_num_denoise_steps,
        val_seed=config.val_seed,
        train_weights=train_weights,
        action_masks=action_masks,
    )
    elapsed = time.perf_counter() - t0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "config_name": args.config_name,
        "checkpoint": str(ckpt_path),
        "dataset_filter": args.dataset,
        "image_resolution": list(config.model.image_resolution),
        "device": str(device),
        "elapsed_sec": elapsed,
        "val_flow_loss_mode": config.val_flow_loss_mode,
        "run_action_mse": config.run_action_mse,
        "num_val_batches": config.num_val_batches,
        "num_action_mse_batches": config.num_action_mse_batches,
        "val_batch_size": cotrain_data_loader.resolve_val_batch_size(config),
        "metrics": metrics,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    logging.info("Eval done in %.1fs -> %s", elapsed, args.output_dir)
    for key in sorted(metrics):
        logging.info("  %s = %.6f", key, metrics[key])


if __name__ == "__main__":
    # Parse CLI before heavy imports so --help is fast; RLDS root is applied in main().
    main(tyro.cli(Args))
