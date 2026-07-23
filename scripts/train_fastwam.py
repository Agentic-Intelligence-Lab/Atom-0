"""
PyTorch FastWAM training on Atom-0 **cotrain RLDS** data pipeline.

Uses ``openpi.cotrain`` (multi-dataset RLDS, restructure registry, per-dataset
delta/normalize) with FastWAM video-window chunking, then trains the PyTorch MoT.

Usage::

  export DIFFSYNTH_MODEL_BASE_PATH=\"$(pwd)/checkpoints/fastwam\"
  uv run --group rlds scripts/train_fastwam.py fastwam_cotrain_piper30 \\
      --exp_name=fw_piper30 --overwrite

See ``docs/fastwam_algorithm.md`` and ``docs/cotrain_技术文档.md``.
"""

from __future__ import annotations

import dataclasses
import gc
import logging
import os
import shutil
import time

import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
import torch.nn.parallel
import tqdm
import wandb

import openpi.models.fastwam_config as fastwam_config
import openpi.shared.normalize as _normalize
import openpi.cotrain.config as cotrain_config
import openpi.cotrain.data_loader as cotrain_data_loader


def init_logging():
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        logger.handlers[0].setFormatter(formatter)


def setup_ddp():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    if use_ddp and not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend, init_method="env://")
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    return use_ddp, local_rank, device


def cleanup_ddp():
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def set_seed(seed: int, local_rank: int):
    torch.manual_seed(seed + local_rank)
    np.random.seed(seed + local_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + local_rank)


def trainable_parameters(model: torch.nn.Module):
    root = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    params = list(root.dit.parameters())
    proprio = getattr(root.fastwam, "proprio_encoder", None)
    if proprio is not None:
        params.extend(list(proprio.parameters()))
    return [p for p in params if p.requires_grad]


def save_checkpoint(model, optimizer, global_step, config, is_main, data_config):
    if not is_main:
        return
    if not (
        (global_step % config.save_interval == 0 and global_step > 0)
        or global_step == config.num_train_steps - 1
    ):
        return

    final_ckpt_dir = config.checkpoint_dir / f"{global_step}"
    tmp_ckpt_dir = config.checkpoint_dir / f"tmp_{global_step}"
    if tmp_ckpt_dir.exists():
        shutil.rmtree(tmp_ckpt_dir)
    tmp_ckpt_dir.mkdir(parents=True, exist_ok=True)

    model_to_save = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    safetensors.torch.save_model(model_to_save, tmp_ckpt_dir / "model.safetensors")
    torch.save(optimizer.state_dict(), tmp_ckpt_dir / "optimizer.pt")
    torch.save({"global_step": global_step, "timestamp": time.time()}, tmp_ckpt_dir / "metadata.pt")

    # Per-dataset cotrain norm stats live under assets/<uid>; copy nothing here if absent.
    if data_config.norm_stats is not None and data_config.asset_id is not None:
        _normalize.save(tmp_ckpt_dir / "assets" / data_config.asset_id, data_config.norm_stats)

    if final_ckpt_dir.exists():
        shutil.rmtree(final_ckpt_dir)
    tmp_ckpt_dir.rename(final_ckpt_dir)
    logging.info(f"Saved checkpoint at step {global_step} -> {final_ckpt_dir}")


def jax_tree_map_to_device(observation, fn):
    prompts = getattr(observation, "_fastwam_prompts", None)
    as_dict = observation.to_dict()

    def _map(tree):
        if tree is None:
            return None
        if isinstance(tree, dict):
            return {k: _map(v) for k, v in tree.items()}
        return fn(tree)

    mapped = _map(as_dict)
    out = type(observation).from_dict(mapped)
    if prompts is not None:
        object.__setattr__(out, "_fastwam_prompts", prompts)
    return out


def train_loop(config: cotrain_config.CotrainTrainConfig):
    use_ddp, local_rank, device = setup_ddp()
    is_main = (not use_ddp) or (dist.get_rank() == 0)
    set_seed(config.seed, local_rank)

    if not isinstance(config.model, fastwam_config.FastWAMConfig):
        raise TypeError(
            f"train_fastwam.py requires FastWAMConfig, got {type(config.model)}. "
            "Use a cotrain config such as `fastwam_cotrain_piper30`."
        )

    if config.overwrite and config.checkpoint_dir.exists() and not config.resume:
        shutil.rmtree(config.checkpoint_dir)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if is_main and config.wandb_enabled:
        wandb.init(name=config.exp_name, config=dataclasses.asdict(config), project=config.project_name)

    world_size = dist.get_world_size() if use_ddp else 1
    if config.batch_size % world_size != 0:
        raise ValueError(f"batch_size={config.batch_size} not divisible by world_size={world_size}")

    logging.info("Building cotrain RLDS loader (FastWAM video window)...")
    loader = cotrain_data_loader.create_cotrain_data_loader(
        config,
        split_label="train",
        shuffle=True,
        shuffle_buffer_size=config.shuffle_buffer_size,
        framework="pytorch",
    )
    data_config = loader.data_config()

    logging.info("Creating FastWAM model (may download Wan2.2 weights on first run)...")
    model = config.model.create_pytorch(device=str(device))
    model.freeze_encoders()
    model.to(device)
    model.train()

    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            find_unused_parameters=True,
        )

    params = trainable_parameters(model)
    optimizer = torch.optim.AdamW(params, lr=config.lr_schedule.peak_lr, weight_decay=config.optimizer.weight_decay)
    logging.info(f"Trainable parameter tensors: {len(params)}")

    global_step = 0
    pbar = tqdm.tqdm(total=config.num_train_steps, disable=not is_main, desc="fastwam-rlds")
    data_iter = iter(loader)

    while global_step < config.num_train_steps:
        try:
            observation, actions = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            observation, actions = next(data_iter)

        def _to_device(x):
            if isinstance(x, torch.Tensor):
                return x.to(device=device, non_blocking=True)
            if isinstance(x, np.ndarray):
                return torch.as_tensor(x, device=device)
            return x

        observation = jax_tree_map_to_device(observation, _to_device)
        actions = _to_device(actions)

        optimizer.zero_grad(set_to_none=True)
        raw = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        losses = raw.compute_loss(observation, actions, train=True)
        loss = losses["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()

        if is_main and global_step % config.log_interval == 0:
            payload = {
                "loss": float(loss.detach().cpu()),
                "loss_video": float(losses["loss_video"].detach().cpu()),
                "loss_action": float(losses["loss_action"].detach().cpu()),
                "step": global_step,
            }
            logging.info(
                f"step={global_step} loss={payload['loss']:.4f} "
                f"video={payload['loss_video']:.4f} action={payload['loss_action']:.4f}"
            )
            if config.wandb_enabled:
                wandb.log(payload, step=global_step)

        save_checkpoint(model, optimizer, global_step, config, is_main, data_config)
        global_step += 1
        pbar.update(1)

        if global_step % 50 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    pbar.close()
    if is_main and config.wandb_enabled:
        wandb.finish()
    cleanup_ddp()


def main():
    init_logging()
    config = cotrain_config.cli()
    train_loop(config)


if __name__ == "__main__":
    main()
