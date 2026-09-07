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
from datetime import timedelta
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
import openpi.cotrain.fastwam_eval as fastwam_eval
from openpi.cotrain.fastwam_checkpoint import find_latest_resume_dir
from openpi.cotrain.fastwam_checkpoint import load_fastwam_checkpoint


def linear_warmup_cosine_decay_lr(
    step: int,
    *,
    num_train_steps: int,
    peak_lr: float,
    decay_lr: float,
    warmup_steps: int | None = None,
    warmup_frac: float = 0.05,
) -> float:
    """Linear warmup to ``peak_lr``, then cosine decay to ``decay_lr``.

    Warmup length comes from ``warmup_steps`` when set; otherwise
    ``round(warmup_frac * num_train_steps)``. After warmup, LR follows a cosine
    from ``peak_lr`` down to ``decay_lr``, reaching the end value at the last step.
    """
    total = max(1, int(num_train_steps))
    if warmup_steps is None:
        warmup_steps = max(1, int(round(total * warmup_frac)))
    else:
        warmup_steps = max(1, int(warmup_steps))
    warmup_steps = min(warmup_steps, total)
    step = max(0, int(step))

    if step < warmup_steps:
        # Small start (same init convention as train_pytorch / JAX CosineDecaySchedule).
        init_lr = peak_lr / (warmup_steps + 1)
        return float(init_lr + (peak_lr - init_lr) * step / warmup_steps)

    if total <= warmup_steps + 1:
        return float(decay_lr if step >= total - 1 else peak_lr)

    # Cosine decay: step==warmup_steps → peak_lr; step==total-1 → decay_lr.
    denom = max(1, (total - 1) - warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / denom))
    cos = 0.5 * (1.0 + np.cos(np.pi * progress))
    return float(decay_lr + (peak_lr - decay_lr) * cos)


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


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
    """Short NCCL barrier for startup/checkpoint sync (not for long eval waits)."""
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


def trainable_parameters(model: torch.nn.Module):
    root = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    params = list(root.dit.parameters())
    proprio = getattr(root.fastwam, "proprio_encoder", None)
    if proprio is not None:
        params.extend(list(proprio.parameters()))
    return [p for p in params if p.requires_grad]


def _global_param_norm(params: list[torch.nn.Parameter]) -> float:
    """L2 norm over all trainable parameter values (MoT + proprio encoder)."""
    total_sq = 0.0
    for param in params:
        if param.requires_grad:
            total_sq += param.detach().float().pow(2).sum().item()
    return total_sq**0.5


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
        object.__setattr__(out, "_fastwam_is_ego", fn(is_ego) if isinstance(is_ego, (torch.Tensor, np.ndarray)) else is_ego)
    return out


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
    """All-rank metrics for wandb: domain losses weighted by sample counts."""
    ego_n, robot_n = _local_batch_domain_counts(observation)

    def _scalar(value) -> float:
        return float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)

    loss_val = _scalar(loss)
    ego_v = _scalar(losses["loss_ego_video"])
    ego_a = _scalar(losses["loss_ego_action"])
    robot_v = _scalar(losses["loss_robot_video"])
    robot_a = _scalar(losses["loss_robot_action"])
    video = _scalar(losses["loss_video"])
    action = _scalar(losses["loss_action"])
    video_raw = _scalar(losses.get("loss_video_raw", float("nan")))
    video_weighted = _scalar(losses.get("loss_video_weighted", float("nan")))
    video_sigma = _scalar(losses.get("video_sigma", float("nan")))
    video_fm_weight = _scalar(losses.get("video_fm_weight", float("nan")))

    stats = torch.tensor(
        [
            loss_val,
            ego_v * ego_n,
            ego_a * ego_n,
            robot_v * robot_n,
            robot_a * robot_n,
            video,
            action,
            ego_n,
            robot_n,
            video_raw,
            video_weighted,
            video_sigma,
            video_fm_weight,
        ],
        device=device,
        dtype=torch.float64,
    )
    if use_ddp:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)

    world = dist.get_world_size() if use_ddp else 1
    total_ego = stats[7].item()
    total_robot = stats[8].item()
    total_batch = total_ego + total_robot

    return {
        "loss": stats[0].item() / world,
        "loss_ego_video": stats[1].item() / max(total_ego, 1.0),
        "loss_ego_action": stats[2].item() / max(total_ego, 1.0),
        "loss_robot_video": stats[3].item() / max(total_robot, 1.0),
        "loss_robot_action": stats[4].item() / max(total_robot, 1.0),
        "loss_video": stats[5].item() / world,
        "loss_action": stats[6].item() / world,
        "loss_video_raw": stats[9].item() / world,
        "loss_video_weighted": stats[10].item() / world,
        "video_sigma": stats[11].item() / world,
        "video_fm_weight": stats[12].item() / world,
        "batch_ego_frac": total_ego / max(total_batch, 1.0),
        "batch_ego_count": total_ego,
        "batch_robot_count": total_robot,
    }


def _next_batch_with_heartbeat(data_iter, *, rank: int, label: str, interval_sec: float = 30.0):
    """Block on ``next(data_iter)`` but emit periodic logs (first batch can take many minutes)."""
    import queue
    import threading

    result_q: queue.Queue = queue.Queue(maxsize=1)
    error_q: queue.Queue = queue.Queue(maxsize=1)

    def _worker():
        try:
            result_q.put(next(data_iter))
        except Exception as exc:  # noqa: BLE001 - surface loader failures on the main thread
            error_q.put(exc)

    threading.Thread(target=_worker, daemon=True).start()
    started = time.perf_counter()
    last_log = started
    while True:
        if not result_q.empty():
            return result_q.get()
        if not error_q.empty():
            raise error_q.get()
        now = time.perf_counter()
        if now - last_log >= interval_sec:
            logging.info(
                "%s still waiting (rank=%s, elapsed=%.0fs)...",
                label,
                rank,
                now - started,
            )
            last_log = now
        time.sleep(1.0)


def train_loop(config: cotrain_config.CotrainTrainConfig):
    use_ddp, local_rank, device = setup_ddp()
    is_main = (not use_ddp) or (dist.get_rank() == 0)
    set_seed(config.seed, local_rank)

    if not isinstance(config.model, fastwam_config.FastWAMConfig):
        raise TypeError(
            f"train_fastwam.py requires FastWAMConfig, got {type(config.model)}. "
            "Use a cotrain config such as `wam-cross-robot` or `wam-cross-fix`."
        )

    init_weight_path = config.pytorch_weight_path or os.environ.get("PYTORCH_WEIGHT_PATH") or os.environ.get(
        "INIT_CHECKPOINT"
    )
    if config.model.skip_dit_load_from_pretrain and not (init_weight_path or config.resume):
        raise ValueError(
            "This preset skips pretrained DiT loading; explicitly provide a matching "
            "--pytorch-weight-path / INIT_CHECKPOINT or use --resume."
        )

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
        "DDP world_size=%s local_rank=%s global_batch=%s per_rank_batch=%s",
        world_size,
        local_rank,
        config.batch_size,
        config.batch_size // world_size,
    )

    logging.info("Creating FastWAM model (may download Wan2.2 weights on first run)...")
    model = config.model.create_pytorch(device=str(device))
    model.freeze_encoders()
    model.to(device)
    model.train()

    # Fine-tune init: load weights from an external checkpoint (model only; fresh optimizer / step=0).
    # Resume: continue the same exp from its latest step (model + optimizer + global_step).
    resume_dir = find_latest_resume_dir(config.checkpoint_dir) if config.resume else None
    if config.resume and resume_dir is None:
        raise FileNotFoundError(
            f"--resume set but no checkpoint found under {config.checkpoint_dir}. "
            "For fine-tuning from another run, pass --pytorch-weight-path instead."
        )
    if init_weight_path and resume_dir is not None:
        raise ValueError(
            "Cannot combine --resume with --pytorch-weight-path / INIT_CHECKPOINT. "
            "Use resume for the same exp, or pytorch_weight_path for fine-tune init."
        )

    if init_weight_path:
        load_fastwam_checkpoint(model, init_weight_path, strict=True)
        logging.info("Fine-tune init from %s (optimizer/step reset)", init_weight_path)
    elif resume_dir is not None:
        load_fastwam_checkpoint(model, resume_dir, strict=True)

    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            find_unused_parameters=True,
        )

    params = trainable_parameters(model)
    peak_lr = float(config.lr_schedule.peak_lr)
    decay_lr = float(config.lr_schedule.decay_lr)
    warmup_steps = max(1, int(config.lr_schedule.warmup_steps))
    warmup_steps = min(warmup_steps, max(1, int(config.num_train_steps)))

    def lr_at(step: int) -> float:
        return linear_warmup_cosine_decay_lr(
            step,
            num_train_steps=config.num_train_steps,
            peak_lr=peak_lr,
            decay_lr=decay_lr,
            warmup_steps=warmup_steps,
        )

    optimizer = torch.optim.AdamW(params, lr=lr_at(0), weight_decay=config.optimizer.weight_decay)
    logging.info(f"Trainable parameter tensors: {len(params)}")
    logging.info(
        "LR schedule: linear warmup %d steps (%.1f%% of %d) %.3e -> peak %.3e, "
        "then cosine decay -> %.3e (from config.lr_schedule.warmup_steps)",
        warmup_steps,
        100.0 * warmup_steps / max(1, config.num_train_steps),
        config.num_train_steps,
        lr_at(0),
        peak_lr,
        decay_lr,
    )

    global_step = 0
    if resume_dir is not None:
        opt_path = resume_dir / "optimizer.pt"
        meta_path = resume_dir / "metadata.pt"
        if opt_path.is_file():
            optimizer.load_state_dict(torch.load(opt_path, map_location="cpu", weights_only=False))
            logging.info("Restored optimizer from %s", opt_path)
        else:
            logging.warning("Resume missing optimizer.pt under %s; continuing with fresh optimizer", resume_dir)
        if meta_path.is_file():
            meta = torch.load(meta_path, map_location="cpu", weights_only=False)
            global_step = int(meta.get("global_step", int(resume_dir.name)))
        else:
            global_step = int(resume_dir.name)
        # Checkpoints are saved at the completed step index; continue from the next step.
        global_step = global_step + 1
        logging.info("Resuming from %s at global_step=%s", resume_dir, global_step)
    # Always re-apply schedule for the current step (resume may have stale param-group lr).
    set_optimizer_lr(optimizer, lr_at(global_step))

    if use_ddp:
        ddp_barrier(use_ddp, device)

    stagger_sec = float(os.environ.get("RLDS_RANK_STAGGER_SEC", "3"))
    if stagger_sec > 0 and use_ddp:
        logging.info("RLDS rank stagger: sleeping %.1fs (rank=%s)", local_rank * stagger_sec, local_rank)
        time.sleep(local_rank * stagger_sec)
        ddp_barrier(use_ddp, device)

    logging.info("Building cotrain RLDS loader (FastWAM video window)...")
    loader = cotrain_data_loader.create_cotrain_data_loader(
        config,
        split_label="train",
        shuffle=True,
        shuffle_buffer_size=config.shuffle_buffer_size,
        framework="pytorch",
    )
    data_config = loader.data_config()
    logging.info(
        "RLDS loader ready (rank=%s, datasets=%s, shuffle_buffer=%s)",
        dist.get_rank() if use_ddp else 0,
        len(data_config.datasets),
        config.shuffle_buffer_size,
    )

    val_loaders = None
    train_weights = None
    action_masks = None
    if is_main and config.eval_interval:
        logging.info("Building validation loaders (rank 0, single-process)...")
        val_loaders = cotrain_data_loader.build_val_loaders(
            config,
            framework="pytorch",
            single_process=True,
        )
        train_weights = cotrain_data_loader.dataset_train_weights(config)
        action_masks = cotrain_data_loader.dataset_action_masks(config)
        logging.info(
            "Validation loaders ready: %s",
            {label: list(loaders) for label, loaders in val_loaders.items()},
        )

    def _run_eval(step: int) -> None:
        if val_loaders is None:
            return
        eval_flag = config.checkpoint_dir / f".eval_done_{step}"
        if is_main:
            if eval_flag.exists():
                eval_flag.unlink()
            eval_t0 = time.perf_counter()
            try:
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
                elapsed = time.perf_counter() - eval_t0
                summary = ", ".join(f"{k}={v:.4f}" for k, v in sorted(metrics.items()))
                logging.info("[eval] step %s (%.1fs): %s", step, elapsed, summary)
                if config.wandb_enabled:
                    wandb.log(metrics, step=step)
            finally:
                eval_flag.touch()
        elif use_ddp:
            poll_sec = float(os.environ.get("FASTWAM_EVAL_POLL_SEC", "5"))
            logging.info("Waiting for rank-0 eval at step %s (rank=%s)...", step, dist.get_rank())
            while not eval_flag.exists():
                time.sleep(poll_sec)
            logging.info("Rank-0 eval finished at step %s (rank=%s)", step, dist.get_rank())

    pbar = tqdm.tqdm(
        total=config.num_train_steps,
        initial=global_step,
        disable=not is_main,
        desc="fastwam-rlds",
    )
    data_iter = iter(loader)
    rank = dist.get_rank() if use_ddp else 0
    logging.info(
        "Waiting for first RLDS batch (rank=%s, shuffle_buffer=%s, datasets=%s, start_step=%s)...",
        rank,
        config.shuffle_buffer_size,
        len(data_config.datasets),
        global_step,
    )

    waiting_first_batch = True
    overfit_batch = None  # (observation, actions) when overfit_fixed_batch
    overfit_noise = None  # {"noise_video", "noise_action"} once shapes known
    while global_step < config.num_train_steps:
        batch_t0 = time.perf_counter()
        if config.overfit_fixed_batch and overfit_batch is not None:
            observation, actions = overfit_batch
        elif waiting_first_batch:
            observation, actions = _next_batch_with_heartbeat(
                data_iter,
                rank=rank,
                label="First RLDS batch",
            )
            logging.info(
                "First RLDS batch ready in %.1fs (rank=%s)",
                time.perf_counter() - batch_t0,
                rank,
            )
            waiting_first_batch = False
            if config.overfit_fixed_batch:
                # Share rank-0's batch so all ranks overfit the same sample.
                if use_ddp:
                    payload = [observation, actions] if is_main else [None, None]
                    dist.broadcast_object_list(payload, src=0)
                    observation, actions = payload
                overfit_batch = (observation, actions)
                logging.info(
                    "Overfit mode: freezing first batch (fixed_video_sigma=%s fixed_action_sigma=%s fixed_noise=%s)",
                    config.fixed_video_sigma,
                    config.fixed_action_sigma,
                    config.overfit_fixed_noise,
                )
        else:
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
        if config.overfit_fixed_batch and overfit_batch is not None:
            # Keep device-resident cache for subsequent steps.
            overfit_batch = (observation, actions)

        optimizer.zero_grad(set_to_none=True)
        raw = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model

        loss_kwargs = {}
        if config.fixed_video_sigma is not None:
            loss_kwargs["video_sigma"] = float(config.fixed_video_sigma)
        if config.fixed_action_sigma is not None:
            loss_kwargs["action_sigma"] = float(config.fixed_action_sigma)
        if config.overfit_fixed_batch and config.overfit_fixed_noise:
            if overfit_noise is None:
                with torch.no_grad():
                    sample0 = raw.observation_to_sample(observation, actions)
                    inputs0 = raw.fastwam.build_inputs(sample0)
                    nv = torch.randn_like(inputs0["input_latents"])
                    na = torch.randn_like(inputs0["action"])
                    if use_ddp:
                        dist.broadcast(nv, src=0)
                        dist.broadcast(na, src=0)
                    overfit_noise = {"noise_video": nv, "noise_action": na}
                    logging.info(
                        "Overfit mode: fixed noise allocated (video=%s action=%s)",
                        tuple(nv.shape),
                        tuple(na.shape),
                    )
            loss_kwargs["noise_video"] = overfit_noise["noise_video"]
            loss_kwargs["noise_action"] = overfit_noise["noise_action"]

        losses = raw.compute_loss(observation, actions, train=True, **loss_kwargs)
        loss = losses["loss"]
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=config.optimizer.clip_gradient_norm)
        current_lr = lr_at(global_step)
        set_optimizer_lr(optimizer, current_lr)
        optimizer.step()

        if global_step % config.log_interval == 0:
            payload = _aggregate_ddp_log_payload(
                losses,
                loss,
                observation,
                device=device,
                use_ddp=use_ddp,
            )
            payload["step"] = global_step
            payload["learning_rate"] = float(current_lr)
            payload["grad_norm"] = float(grad_norm)
            payload["params_norm"] = _global_param_norm(params)
            if is_main:
                logging.info(
                    f"step={global_step} loss={payload['loss']:.4f} "
                    f"lr={current_lr:.3e} "
                    f"v_raw={payload['loss_video_raw']:.4f} v_w={payload['loss_video_weighted']:.4f} "
                    f"sigma={payload['video_sigma']:.3f} w_t={payload['video_fm_weight']:.3f} "
                    f"ego_v={payload['loss_ego_video']:.4f} ego_a={payload['loss_ego_action']:.4f} "
                    f"robot_v={payload['loss_robot_video']:.4f} robot_a={payload['loss_robot_action']:.4f} "
                    f"grad_norm={payload['grad_norm']:.4f} params_norm={payload['params_norm']:.2f} "
                    f"ego_frac={payload['batch_ego_frac']:.2f}"
                )
                if config.wandb_enabled:
                    wandb.log(payload, step=global_step)

        save_checkpoint(model, optimizer, global_step, config, is_main, data_config)

        if config.eval_interval and global_step > 0 and global_step % config.eval_interval == 0:
            _run_eval(global_step)

        global_step += 1
        pbar.update(1)

        if global_step % 50 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    pbar.close()
    if is_main and config.wandb_enabled:
        wandb.finish()
    cleanup_ddp(use_ddp, device)


def main():
    init_logging()
    config = cotrain_config.cli()
    train_loop(config)


if __name__ == "__main__":
    main()
