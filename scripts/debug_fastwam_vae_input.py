#!/usr/bin/env python
"""Sample cotrain RLDS batches and save FastWAM VAE input video frames (e.g. 576×512 robot_wrist).

Does **not** start training or load Wan/VAE weights — only reuses the training data pipeline
and the same ``_images_to_video`` compose path as ``FastWAMPytorch.observation_to_sample``.

Usage::

  cd Atom-0
  source scripts/atom0_env.sh   # sets RLDS_DATA_DIR=/mnt/bos/bo23lu
  PYTHONPATH=src .venv/bin/python scripts/debug_fastwam_vae_input.py wam-cross-robot \\
      --num-batches 8 --samples-per-batch 2 --output-dir tmp/fastwam_vae_input

Or explicitly::

  RLDS_DATA_DIR=/mnt/bos/bo23lu PYTHONPATH=src .venv/bin/python scripts/debug_fastwam_vae_input.py ...
"""

from __future__ import annotations

import dataclasses
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import tyro
from tyro.conf import Positional


def init_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


@dataclasses.dataclass
class Args:
    config_name: Positional[str]
    """Cotrain FastWAM config, e.g. ``wam-cross-robot``."""

    rlds_data_dir: Path | None = None
    """RLDS root (default: env ``RLDS_DATA_DIR``, else ``/mnt/bos/bo23lu``). Must be set before config import."""

    output_dir: Path | None = None
    """Directory for PNG outputs (default: ``tmp/fastwam_vae_input/<config>/<timestamp>``)."""

    num_batches: int = 8
    """How many training batches to inspect after ``skip_batches``."""

    samples_per_batch: int = 2
    """Random samples per batch to save."""

    skip_batches: int = 4
    """Skip initial batches so TF shuffle buffer can mix datasets."""

    seed: int = 0
    shuffle_buffer_size: int | None = None
    """Override config shuffle buffer (default: use config value)."""

    batch_size: int | None = None
    """Override global batch size for faster local debug (default: config value)."""


def _configure_rlds_root(args: Args) -> Path:
    if args.rlds_data_dir is not None:
        os.environ["RLDS_DATA_DIR"] = str(args.rlds_data_dir)
    elif "RLDS_DATA_DIR" not in os.environ:
        os.environ["RLDS_DATA_DIR"] = "/mnt/bos/bo23lu"
    root = Path(os.environ["RLDS_DATA_DIR"])
    if not root.is_dir():
        raise FileNotFoundError(
            f"RLDS_DATA_DIR not found: {root}\n"
            "Set RLDS_DATA_DIR or pass --rlds-data-dir (e.g. /mnt/bos/bo23lu)."
        )
    return root


def _observation_from_raw_batch(batch: dict, model) -> tuple:
    batch = dict(batch)
    prompts = batch.pop("prompt", None)
    is_ego = batch.pop("is_ego", None)
    observation = model.Observation.from_dict(batch)
    if prompts is not None:
        if isinstance(prompts, np.ndarray):
            prompt_list = [str(p) for p in prompts.tolist()]
        else:
            prompt_list = [str(p) for p in prompts]
        object.__setattr__(observation, "_fastwam_prompts", prompt_list)
    if is_ego is not None:
        if isinstance(is_ego, torch.Tensor):
            is_ego_t = is_ego.to(dtype=torch.bool).reshape(-1)
        else:
            is_ego_t = torch.as_tensor(np.asarray(is_ego), dtype=torch.bool).reshape(-1)
        object.__setattr__(observation, "_fastwam_is_ego", is_ego_t)
    return observation, batch


def main(args: Args) -> None:
    init_logging()
    rlds_root = _configure_rlds_root(args)
    logging.info("RLDS_DATA_DIR=%s", rlds_root)

    # Import cotrain after RLDS_DATA_DIR is fixed (config builder paths resolve at import time).
    import openpi.cotrain.config as cotrain_config
    import openpi.cotrain.data_loader as cotrain_data_loader
    import openpi.models.model as _model
    from openpi.cotrain import fastwam_vae_input_debug
    from openpi.cotrain.data_loader import resolve_train_image_resize_hw
    from openpi.cotrain.data_loader import resolve_train_image_resize_hw_by_slot

    train_config = cotrain_config.get_config(args.config_name)
    if train_config.model.model_type != _model.ModelType.FASTWAM:
        raise TypeError(f"Config {args.config_name!r} is not a FastWAM config.")

    batch_size = train_config.batch_size if args.batch_size is None else args.batch_size
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    out_dir = args.output_dir
    if out_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("tmp") / "fastwam_vae_input" / args.config_name / stamp

    shuffle_buffer = (
        train_config.shuffle_buffer_size
        if args.shuffle_buffer_size is None
        else args.shuffle_buffer_size
    )

    logging.info(
        "Building cotrain loader: config=%s batch_size=%s shuffle_buffer=%s (no model / no VAE load)",
        args.config_name,
        batch_size,
        shuffle_buffer,
    )
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    video_num_frames = int(getattr(train_config.model, "video_num_frames", 9))
    action_video_freq_ratio = int(getattr(train_config.model, "action_video_freq_ratio", 4))
    loader_impl = cotrain_data_loader.create_cotrain_rlds_data_loader(
        data_config,
        action_horizon=train_config.model.action_horizon,
        batch_size=batch_size,
        split_label="train",
        shuffle=True,
        shuffle_buffer_size=shuffle_buffer,
        num_parallel_reads=train_config.data_num_parallel_reads,
        num_parallel_calls=train_config.data_num_parallel_calls,
        pad_action_dim=train_config.model.action_dim,
        image_resize_hw=resolve_train_image_resize_hw(train_config.model),
        image_resize_hw_by_slot=resolve_train_image_resize_hw_by_slot(train_config.model),
        video_num_frames=video_num_frames,
        action_video_freq_ratio=action_video_freq_ratio,
        framework="pytorch",
        partition_builders_by_rank=False,
        single_process=True,
    )
    raw_loader = loader_impl._data_loader

    collected: list[tuple[dict, object, torch.Tensor | np.ndarray]] = []

    data_iter = iter(raw_loader)
    for _ in range(args.skip_batches):
        next(data_iter)
        logging.info("Skipped batch for shuffle warmup")

    t0 = time.perf_counter()
    while len(collected) < args.num_batches:
        try:
            raw_batch = next(data_iter)
        except StopIteration:
            logging.warning("Data iterator exhausted after %s batches", len(collected))
            break
        raw_batch = dict(raw_batch)
        actions = raw_batch.get("actions")
        observation, meta_batch = _observation_from_raw_batch(raw_batch, _model)
        collected.append((meta_batch, observation, actions))
        logging.info("Collected batch %s/%s", len(collected), args.num_batches)

    if not collected:
        sys.exit("No batches collected; increase skip_batches or check RLDS paths.")

    debug_cfg = fastwam_vae_input_debug.VaeInputDebugConfig(
        output_dir=out_dir,
        num_batches=args.num_batches,
        samples_per_batch=args.samples_per_batch,
        skip_batches=args.skip_batches,
        seed=args.seed,
    )
    summary = fastwam_vae_input_debug.save_random_vae_inputs_from_batches(
        collected,
        train_config.model,
        debug_cfg,
    )
    elapsed = time.perf_counter() - t0
    logging.info(
        "Saved %s PNGs from %s batches -> %s (%.1fs)",
        summary["num_png"],
        summary["num_batches_saved"],
        out_dir,
        elapsed,
    )
    logging.info("Index: %s", out_dir / "index.json")


if __name__ == "__main__":
    main(tyro.cli(Args))
