"""
PyTorch HPT training on Atom-0 **cotrain RLDS** data pipeline (same as pi05).

Usage::

  uv run --group rlds scripts/train_hpt.py hpt_cotrain_fk_eef_plus_piper_ego \\
      --exp-name=hpt_mix --overwrite

See ``docs/HPT.md``.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import os
import shutil
import time
from datetime import timedelta

import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
import tqdm
import wandb

import openpi.models.hpt_config as hpt_config
import openpi.shared.normalize as _normalize
import openpi.cotrain.config as cotrain_config
import openpi.cotrain.data_loader as cotrain_data_loader
import openpi.cotrain.hpt_eval as hpt_eval


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
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    if use_ddp and not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        timeout_hours = int(os.environ.get("DDP_TIMEOUT_HOURS", "2"))
        init_kwargs: dict = {
            "backend": backend,
            "init_method": "env://",
            "timeout": timedelta(hours=timeout_hours),
        }
        if backend == "nccl":
            init_kwargs["device_id"] = device
        torch.distributed.init_process_group(**init_kwargs)
    return use_ddp, local_rank, device


def ddp_barrier(use_ddp: bool, device: torch.device) -> None:
    if not use_ddp:
        return
    if device.type == "cuda":
        dist.barrier(device_ids=[device.index])
    else:
        dist.barrier()


def cleanup_ddp(use_ddp: bool, device: torch.device):
    if torch.distributed.is_initialized():
        ddp_barrier(use_ddp, device)
        torch.distributed.destroy_process_group()


def set_seed(seed: int, local_rank: int):
    torch.manual_seed(seed + local_rank)
    np.random.seed(seed + local_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + local_rank)


def save_checkpoint(model, optimizer, global_step, config, is_main, data_config, ema_state=None):
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
    # Always persist live weights; when EMA is on, also write EMA as the primary
    # ``model.safetensors`` used by inference (matches JAX pi05 ckpt convention).
    safetensors.torch.save_model(model_to_save, tmp_ckpt_dir / "model_live.safetensors")
    if ema_state is not None:
        live_backup = {k: v.detach().clone() for k, v in model_to_save.state_dict().items()}
        model_to_save.load_state_dict(ema_state, strict=True)
        safetensors.torch.save_model(model_to_save, tmp_ckpt_dir / "model.safetensors")
        model_to_save.load_state_dict(live_backup, strict=True)
        torch.save({"ema_decay": float(config.ema_decay)}, tmp_ckpt_dir / "ema.pt")
    else:
        safetensors.torch.save_model(model_to_save, tmp_ckpt_dir / "model.safetensors")
    torch.save(optimizer.state_dict(), tmp_ckpt_dir / "optimizer.pt")
    torch.save({"global_step": global_step, "timestamp": time.time()}, tmp_ckpt_dir / "metadata.pt")

    if data_config.norm_stats is not None and data_config.asset_id is not None:
        _normalize.save(tmp_ckpt_dir / "assets" / data_config.asset_id, data_config.norm_stats)

    if final_ckpt_dir.exists():
        shutil.rmtree(final_ckpt_dir)
    tmp_ckpt_dir.rename(final_ckpt_dir)
    logging.info("Saved checkpoint at step %s -> %s", global_step, final_ckpt_dir)


def _init_ema_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


@torch.no_grad()
def _update_ema_state(ema_state: dict[str, torch.Tensor], model: torch.nn.Module, decay: float) -> None:
    for name, param in model.state_dict().items():
        if name not in ema_state:
            ema_state[name] = param.detach().clone()
            continue
        ema_state[name].mul_(decay).add_(param.detach(), alpha=1.0 - decay)


@torch.no_grad()
def _swap_weights(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Load ``state`` into ``model``; return a clone of the previous weights."""
    backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state, strict=True)
    return backup


def jax_tree_map_to_device(observation, fn):
    prompts = getattr(observation, "_fastwam_prompts", None)
    is_ego = getattr(observation, "_fastwam_is_ego", None)
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
    if is_ego is not None:
        object.__setattr__(
            out, "_fastwam_is_ego", fn(is_ego) if isinstance(is_ego, (torch.Tensor, np.ndarray)) else is_ego
        )
    return out


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def _local_batch_domain_counts(observation) -> tuple[float, float]:
    """Return (ego_sample_count, robot_sample_count) for the local batch."""
    is_ego = getattr(observation, "_fastwam_is_ego", None)
    if is_ego is None:
        return 0.0, 1.0
    if isinstance(is_ego, torch.Tensor):
        ego_count = float(is_ego.to(dtype=torch.float32).sum().item())
        batch_n = float(is_ego.numel())
    else:
        arr = np.asarray(is_ego, dtype=np.float32).reshape(-1)
        ego_count = float(arr.sum())
        batch_n = float(arr.size)
    return ego_count, max(batch_n - ego_count, 0.0)


def _aggregate_ddp_log_payload(
    losses: dict,
    loss: torch.Tensor,
    observation,
    *,
    device: torch.device,
    use_ddp: bool,
) -> dict[str, float]:
    """All-rank metrics for wandb: global losses averaged; domain losses weighted by counts."""
    ego_n, robot_n = _local_batch_domain_counts(observation)

    def _scalar(value) -> float:
        return float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)

    loss_val = _scalar(loss)
    ego_action = _scalar(losses["loss_ego_action"])
    robot_action = _scalar(losses["loss_robot_action"])
    ego_world = _scalar(losses["loss_ego_world"])
    robot_world = _scalar(losses["loss_robot_world"])
    action = _scalar(losses["loss_action"])
    world = _scalar(losses["loss_world"])
    action_smooth = _scalar(losses["loss_action_smooth"])
    ego_smooth = _scalar(losses["loss_ego_smooth"])
    robot_smooth = _scalar(losses["loss_robot_smooth"])

    stats = torch.tensor(
        [
            loss_val,
            ego_action * ego_n,
            robot_action * robot_n,
            ego_world * ego_n,
            robot_world * robot_n,
            action,
            world,
            action_smooth,
            ego_smooth * ego_n,
            robot_smooth * robot_n,
            ego_n,
            robot_n,
        ],
        device=device,
        dtype=torch.float64,
    )
    if use_ddp:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)

    world_size = dist.get_world_size() if use_ddp else 1
    total_ego = stats[10].item()
    total_robot = stats[11].item()
    total_batch = total_ego + total_robot

    return {
        "loss": stats[0].item() / world_size,
        "loss_ego_action": stats[1].item() / max(total_ego, 1.0),
        "loss_robot_action": stats[2].item() / max(total_robot, 1.0),
        "loss_ego_world": stats[3].item() / max(total_ego, 1.0),
        "loss_robot_world": stats[4].item() / max(total_robot, 1.0),
        "loss_action": stats[5].item() / world_size,
        "loss_world": stats[6].item() / world_size,
        "loss_action_smooth": stats[7].item() / world_size,
        "loss_ego_smooth": stats[8].item() / max(total_ego, 1.0),
        "loss_robot_smooth": stats[9].item() / max(total_robot, 1.0),
        "batch_ego_frac": total_ego / max(total_batch, 1.0),
        "batch_ego_count": total_ego,
        "batch_robot_count": total_robot,
    }


def linear_warmup_cosine_decay_lr(
    step: int,
    *,
    num_train_steps: int,
    peak_lr: float,
    decay_lr: float,
    warmup_steps: int,
) -> float:
    """Linear warmup to ``peak_lr``, then cosine decay to ``decay_lr`` (same as FastWAM / JAX)."""
    total = max(1, int(num_train_steps))
    warmup_steps = min(max(1, int(warmup_steps)), total)
    step = max(0, int(step))
    if step < warmup_steps:
        init_lr = peak_lr / (warmup_steps + 1)
        return float(init_lr + (peak_lr - init_lr) * step / warmup_steps)
    if total <= warmup_steps + 1:
        return float(decay_lr if step >= total - 1 else peak_lr)
    denom = max(1, (total - 1) - warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / denom))
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(decay_lr + (peak_lr - decay_lr) * cos)


def train_loop(config: cotrain_config.CotrainTrainConfig):
    use_ddp, local_rank, device = setup_ddp()
    is_main = (not use_ddp) or (dist.get_rank() == 0)
    set_seed(config.seed, local_rank)

    if not isinstance(config.model, hpt_config.HPTConfig):
        raise TypeError(
            f"train_hpt.py requires HPTConfig, got {type(config.model)}. "
            "Use a cotrain config such as `hpt_cotrain_fk_eef_plus_piper_ego`."
        )

    pretrained_trunk = os.environ.get("PRETRAINED_TRUNK_PATH") or config.model.pretrained_trunk_path
    if pretrained_trunk:
        config = dataclasses.replace(
            config,
            model=dataclasses.replace(config.model, pretrained_trunk_path=pretrained_trunk),
        )
        logging.info("HPT trunk warm-start: %s", pretrained_trunk)

    if config.overwrite and not config.resume:
        if is_main and config.checkpoint_dir.exists():
            shutil.rmtree(config.checkpoint_dir)
        if use_ddp:
            ddp_barrier(use_ddp, device)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if is_main and config.wandb_enabled:
        wandb.init(name=config.exp_name, config=dataclasses.asdict(config), project=config.project_name)

    world_size = dist.get_world_size() if use_ddp else 1
    if config.batch_size % world_size != 0:
        raise ValueError(f"batch_size={config.batch_size} not divisible by world_size={world_size}")

    logging.info(
        "Creating HPT model (embed_dim=%s, ema_decay=%s)...",
        config.model.embed_dim,
        config.ema_decay,
    )
    model = config.model.create_pytorch(device=str(device))
    if config.model.freeze_encoders:
        model.freeze_encoders()
    model.to(device)
    model.train()

    if config.pytorch_weight_path:
        from openpi.models_pytorch.hpt.model import load_hpt_weights

        load_hpt_weights(model, config.pytorch_weight_path, device=str(device))
        logging.info("Loaded weights from %s", config.pytorch_weight_path)

    if config.model.train_mode == "finetune":
        model.apply_finetune_freeze()

    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            find_unused_parameters=True,
        )

    root = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    params = root.trainable_parameters()
    peak_lr = float(config.lr_schedule.peak_lr)
    decay_lr = float(getattr(config.lr_schedule, "decay_lr", peak_lr))
    warmup_steps = min(max(1, int(config.lr_schedule.warmup_steps)), max(1, int(config.num_train_steps)))

    def lr_at(step: int) -> float:
        return linear_warmup_cosine_decay_lr(
            step,
            num_train_steps=config.num_train_steps,
            peak_lr=peak_lr,
            decay_lr=decay_lr,
            warmup_steps=warmup_steps,
        )

    optimizer = torch.optim.AdamW(params, lr=lr_at(0), weight_decay=config.optimizer.weight_decay)
    logging.info("Trainable parameter tensors: %s", len(params))
    logging.info(
        "LR schedule: linear warmup %d steps (%.1f%% of %d) %.3e -> peak %.3e, then cosine -> %.3e",
        warmup_steps,
        100.0 * warmup_steps / max(1, config.num_train_steps),
        config.num_train_steps,
        lr_at(0),
        peak_lr,
        decay_lr,
    )

    ema_decay = config.ema_decay
    ema_state: dict[str, torch.Tensor] | None = None
    if ema_decay is not None:
        ema_state = _init_ema_state(root)
        logging.info("EMA enabled: decay=%s (eval_on_ema=%s)", ema_decay, config.eval_on_ema)

    stagger_sec = float(os.environ.get("RLDS_RANK_STAGGER_SEC", "3"))
    if stagger_sec > 0 and use_ddp:
        time.sleep(local_rank * stagger_sec)
        ddp_barrier(use_ddp, device)

    logging.info("Building cotrain RLDS loader (HPT current+future frames)...")
    loader = cotrain_data_loader.create_cotrain_data_loader(
        config,
        split_label="train",
        shuffle=True,
        shuffle_buffer_size=config.shuffle_buffer_size,
        framework="pytorch",
    )
    data_config = loader.data_config()

    val_loaders = None
    action_masks: dict[str, tuple[bool, ...]] | None = None
    val_train_weights: dict[str, float] | None = None
    if is_main and config.eval_interval:
        val_loaders = cotrain_data_loader.build_val_loaders(
            config, framework="pytorch", single_process=True
        )
        action_masks = cotrain_data_loader.dataset_action_masks(config)
        val_uids = getattr(config, "val_dataset_uids", None)
        if val_uids:
            allowed = set(val_uids)
            val_train_weights = {
                ds.uid: ds.weight
                for ds in data_config.datasets
                if ds.uid in allowed
            }

    data_iter = iter(loader)
    start = time.perf_counter()
    for global_step in tqdm.trange(config.num_train_steps, disable=not is_main):
        observation, actions = next(data_iter)
        observation = jax_tree_map_to_device(observation, lambda x: x.to(device) if torch.is_tensor(x) else x)
        actions = actions.to(device) if torch.is_tensor(actions) else torch.as_tensor(actions, device=device)

        optimizer.zero_grad(set_to_none=True)
        # Match train_fastwam: compute_loss on the unwrapped module (find_unused_parameters=True).
        raw = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        losses = raw.compute_loss(observation, actions, train=True)
        loss = losses["loss"]
        loss.backward()
        if config.optimizer.clip_gradient_norm is not None:
            torch.nn.utils.clip_grad_norm_(params, config.optimizer.clip_gradient_norm)
        set_optimizer_lr(optimizer, lr_at(global_step))
        optimizer.step()
        if ema_state is not None:
            _update_ema_state(ema_state, raw, float(ema_decay))

        if global_step % config.log_interval == 0:
            payload = _aggregate_ddp_log_payload(
                losses,
                loss,
                observation,
                device=device,
                use_ddp=use_ddp,
            )
            if is_main:
                elapsed = time.perf_counter() - start
                payload["lr"] = optimizer.param_groups[0]["lr"]
                payload["steps_per_sec"] = (global_step + 1) / max(elapsed, 1e-6)
                logging.info("step=%s %s", global_step, payload)
                if config.wandb_enabled:
                    wandb.log(payload, step=global_step)

        if is_main and config.eval_interval and global_step > 0 and global_step % config.eval_interval == 0:
            live_backup = None
            if ema_state is not None and config.eval_on_ema:
                live_backup = _swap_weights(raw, ema_state)
            metrics = hpt_eval.run_eval(
                val_loaders,
                model,
                device=device,
                num_val_batches=config.num_val_batches,
                run_action_mse=config.run_action_mse,
                num_action_mse_batches=config.num_action_mse_batches,
                action_mse_num_denoise_steps=config.action_mse_num_denoise_steps,
                val_seed=config.val_seed,
                action_masks=action_masks,
                train_weights=val_train_weights,
                max_datasets=getattr(config, "val_max_datasets", None),
            )
            if live_backup is not None:
                _swap_weights(raw, live_backup)
            logging.info("eval@%s %s", global_step, metrics)
            if config.wandb_enabled and metrics:
                wandb.log(metrics, step=global_step)

        save_checkpoint(model, optimizer, global_step, config, is_main, data_config, ema_state=ema_state)

    if is_main and config.wandb_enabled:
        wandb.finish()
    cleanup_ddp(use_ddp, device)


def main():
    init_logging()
    config = cotrain_config.cli()
    train_loop(config)


if __name__ == "__main__":
    main()
