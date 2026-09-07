"""Co-training entrypoint: multi-dataset RLDS training with train/val splits + eval.

Self-contained fork of `scripts/train.py`. It does NOT modify any existing openpi
file; the train_step / init helpers are copied verbatim, and the additions are:

  * train loader built from `openpi.cotrain.data_loader` (multi-dataset, split="train")
  * per-dataset validation loaders (split="val")
  * periodic eval: val flow-matching loss (fixed-seed + multi-sample) and action MSE,
    logged per-dataset and aggregated, via `openpi.cotrain.eval`

Run with this module's own config registry, e.g.:
    uv run python scripts/train_cotrain.py cotrain_droid_sanity \
        --exp_name=my_run --data.rlds_data_dir=/path/to/rlds
"""

import dataclasses
import functools
import logging
import os
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental.multihost_utils as multihost_utils
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.cotrain.action_space as cotrain_action_space
import openpi.cotrain.config as cotrain_config
import openpi.cotrain.data_loader as cotrain_data_loader
import openpi.cotrain.eval as cotrain_eval
import openpi.models.model as _model
import openpi.models.tokenizer as _tokenizer
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def init_logging():
    """Custom logging format for better readability."""
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
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    # Only the primary process logs to wandb; other hosts disable it (their wandb.log become
    # no-ops) so a multi-host run produces a single wandb run instead of one per process.
    if not enabled or jax.process_index() != 0:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        model = config.model.create(model_rng)
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    train_state = jax.jit(
        init,
        donate_argnums=(1,),
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


def _ki_grad_norm_split(grads) -> dict[str, at.Array]:
    vlm_sq = jnp.zeros(())
    act_sq = jnp.zeros(())
    for k, v in grads.flat_state().items():
        path_str = "/".join(str(p) for p in k)
        g = v.value
        sq = jnp.sum(jnp.square(g.astype(jnp.float32)))
        if "llm" in path_str and "_1" not in path_str:
            vlm_sq = vlm_sq + sq
        elif "_1" in path_str:
            act_sq = act_sq + sq
    return {"grad_norm_vlm": jnp.sqrt(vlm_sq), "grad_norm_action": jnp.sqrt(act_sq)}


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        out = model.compute_loss(rng, observation, actions, train=True)
        if isinstance(out, dict):
            ki_alpha = getattr(model, "ki_alpha", 1.0)
            total = jnp.mean(out["flow"])
            aux = {"flow_loss": jnp.mean(out["flow"])}
            # ---- 新增：分域统计 ----
            if "flow_robot" in out and "flow_ego" in out:
                # out["flow_*"] 一般是 [B, action_horizon]，先对 horizon 平均 → [B]
                robot_per = jnp.mean(out["flow_robot"], axis=-1)
                ego_per = jnp.mean(out["flow_ego"], axis=-1)
                if observation.domain_mask is None:
                    # 没有 mask 时退化为全 batch 平均（单域数据）
                    aux["robot_loss"] = jnp.mean(robot_per)
                    aux["ego_loss"] = jnp.mean(ego_per)
                    aux["ego_frac"] = jnp.asarray(0.0)
                else:
                    is_ego = observation.domain_mask.astype(jnp.float32)       # [B], 1=ego
                    is_robot = 1.0 - is_ego
                    n_ego = jnp.sum(is_ego)
                    n_robot = jnp.sum(is_robot)
                    aux["robot_loss"] = jnp.where(
                        n_robot > 0,
                        jnp.sum(robot_per * is_robot) / jnp.clip(n_robot, 1.0),
                        jnp.nan,
                    )
                    aux["ego_loss"] = jnp.where(
                        n_ego > 0,
                        jnp.sum(ego_per * is_ego) / jnp.clip(n_ego, 1.0),
                        jnp.nan,
                    )
                    aux["ego_frac"] = jnp.mean(is_ego)
            # ---- 新增结束 ----
            if "ki_fast" in out:
                total = total + ki_alpha * jnp.mean(out["ki_fast"])
                aux["ki_fast_loss"] = jnp.mean(out["ki_fast"])
            if "ot" in out:
                ot_alpha = getattr(model, "ot_alpha", 0.0)
                total = total + ot_alpha * out["ot"]
                aux["ot_loss"] = out["ot"]
                if "ot_hz" in out:
                    aux["ot_hz"] = out["ot_hz"]
                if "ot_sz" in out:
                    aux["ot_sz"] = out["ot_sz"]
            return total, aux
        return jnp.mean(out), {}

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, aux), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
        model, train_rng, observation, actions
    )

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
        **aux,
    }
    if getattr(model, "ki_enabled", False):
        info.update(_ki_grad_norm_split(grads))

    return new_state, info


def _maybe_init_jax_distributed():
    """Initialize JAX multi-host (e.g. PAI DLC 16-GPU = 2 nodes x 8). No-op for single host.

    Must run BEFORE any jax device call. Reads standard distributed env vars (DLC/torchrun
    style: WORLD_SIZE / RANK / MASTER_ADDR / MASTER_PORT); override the coordinator with
    JAX_COORDINATOR_ADDRESS if needed. After this, jax.device_count() is GLOBAL (16),
    jax.local_device_count() is per-node (8), and checkpoint/data paths key off process_index.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return
    coordinator = os.environ.get("JAX_COORDINATOR_ADDRESS")
    if coordinator is None:
        coordinator = f"{os.environ['MASTER_ADDR']}:{os.environ.get('MASTER_PORT', '1234')}"
    jax.distributed.initialize(
        coordinator_address=coordinator,
        num_processes=world_size,
        process_id=int(os.environ["RANK"]),
    )
    logging.info(
        f"JAX distributed initialized: process {jax.process_index()}/{jax.process_count()}, "
        f"local_devices={jax.local_device_count()}, global_devices={jax.device_count()}"
    )


def _init_checkpoint_dir_multihost(config: cotrain_config.CotrainTrainConfig):
    """Multi-host-safe checkpoint dir init (PAI DLC 16-GPU = 2 nodes x 8).

    openpi's `initialize_checkpoint_dir` is NOT multi-host safe: with --overwrite every
    process races to rmtree the same NAS dir (FileNotFound '_METADATA'); without a flag,
    rank0 mkdirs the dir and the other ranks then see it exists -> FileExistsError. Here
    only process 0 wipes/creates the dir, all processes barrier, then everyone opens it
    with overwrite=False/resume=True so no rank rmtrees or raises. An empty dir is treated
    as a fresh run (openpi downgrades resume->False when there are 0 checkpoints); a dir
    with real checkpoints resumes from the latest.
    """
    checkpoint_dir = epath.Path(config.checkpoint_dir).resolve()
    if jax.process_index() == 0:
        if config.overwrite and checkpoint_dir.exists():
            checkpoint_dir.rmtree()
            logging.info(f"Wiped checkpoint directory {checkpoint_dir}")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # All ranks wait until rank0 has wiped/created the (shared NAS) dir.
    multihost_utils.sync_global_devices("cotrain_ckpt_dir_ready")
    return _checkpoints.initialize_checkpoint_dir(
        checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=False,
        resume=True,
    )


def main(config: cotrain_config.CotrainTrainConfig):
    init_logging()
    _maybe_init_jax_distributed()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    val_batch_size = cotrain_data_loader.resolve_val_batch_size(config)
    if val_batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Validation batch size {val_batch_size} must be divisible by the number of devices "
            f"{jax.device_count()}."
        )

    # Training pods share /data but their home directories are ephemeral.  Honour the
    # host/job-provided cache location so recompilations can be reused across restarts.
    compilation_cache_dir = os.environ.get(
        "JAX_COMPILATION_CACHE_DIR", str(epath.Path("~/.cache/jax").expanduser())
    )
    jax.config.update("jax_compilation_cache_dir", compilation_cache_dir)

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _init_checkpoint_dir_multihost(config)
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    # --- Train loader (multi-dataset weighted mixture, split="train") -------------------
    data_loader = cotrain_data_loader.create_cotrain_data_loader(
        config,
        split_label="train",
        sharding=data_sharding,
        shuffle=True,
        shuffle_buffer_size=config.shuffle_buffer_size,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized train data loader:\n{training_utils.array_tree_to_info(batch)}")

    # --- Validation loaders (one per dataset, split="val") ------------------------------
    val_loaders = cotrain_data_loader.build_val_loaders(config, sharding=data_sharding)
    # val_loaders = {}
    train_weights = cotrain_data_loader.dataset_train_weights(config)
    action_masks = cotrain_data_loader.dataset_action_masks(config)
    logging.info(f"Initialized validation loaders by label: { {label: list(d) for label, d in val_loaders.items()} }")

    # Sanity-check the language prompt of the first train batch.
    # NOTE: in multi-host the batch is a globally-sharded jax.Array, so np.array()
    # would try to fetch non-addressable shards and crash. Concatenate only this
    # process's local shards instead — enough to decode a few sample prompts.
    def _local_np(arr):
        shards = getattr(arr, "addressable_shards", None)
        if shards:
            return np.concatenate([np.asarray(s.data) for s in shards], axis=0)
        return np.asarray(arr)

    if batch[0].tokenized_prompt is not None:
        _prompt_tok = _tokenizer.PaligemmaTokenizer()
        _tok = _local_np(batch[0].tokenized_prompt)
        _tok_mask = _local_np(batch[0].tokenized_prompt_mask)
        for _i in range(min(3, _tok.shape[0])):
            _ids = _tok[_i][_tok_mask[_i]].astype(int).tolist()
            logging.info(f"[prompt-check] sample {_i}: {_prompt_tok._tokenizer.decode(_ids)!r}")
    else:
        logging.warning("[prompt-check] batch has no tokenized_prompt — language conditioning is OFF!")

    def _current_frame(arr):
        if arr.ndim == 4:
            arr = arr[-1]
        return arr

    # Gather only this process's local image shards (multi-host safe), then index locally.
    _local_imgs = {k: _local_np(v) for k, v in batch[0].images.items()}
    _n_local = min(5, len(next(iter(_local_imgs.values()))))
    images_to_log = [
        wandb.Image(np.concatenate([_current_frame(img[i]) for img in _local_imgs.values()], axis=1))
        for i in range(_n_local)
    ]
    if jax.process_index() == 0:
        wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    # --- Jitted eval steps (static config baked via closures) ---------------------------
    val_flow_step = cotrain_eval.make_val_flow_step(
        num_samples=config.val_flow_loss_num_samples,
        mode=config.val_flow_loss_mode,
        use_ema=config.eval_on_ema,
    )
    # One action-MSE step per dataset (the mask is per-dataset, label-independent).
    val_action_mse_steps = {
        name: cotrain_eval.make_val_action_mse_step(
            num_denoise_steps=config.action_mse_num_denoise_steps,
            fallback_mask=action_masks[name],
            use_ema=config.eval_on_ema,
        )
        for name in action_masks
    }
    # Shared predicted/gt action-chunk step for trajectory visualization.
    val_action_pred_step = cotrain_eval.make_val_action_pred_step(
        num_denoise_steps=config.action_mse_num_denoise_steps,
        use_ema=config.eval_on_ema,
    )

    def _log_action_traj(step: int):
        try:
            for label, loaders in val_loaders.items():
                for name, loader in loaders.items():
                    try:
                        batch = next(iter(loader))
                    except StopIteration:
                        continue
                    rng = jax.random.fold_in(jax.random.key(config.val_seed), 0)
                    with sharding.set_mesh(mesh):
                        out_sharded = val_action_pred_step(rng, train_state, batch)
                    # pred/gt are sharded across the batch axis over all (16) devices; jax.device_get
                    # would fetch non-addressable shards and crash in multi-host. process_allgather
                    # (collective; every rank must call it, which they do — same loader order) rebuilds
                    # the full global arrays as numpy on each host. tiled=True concatenates along the
                    # existing sharded axis instead of adding a new process axis.
                    out = {k: multihost_utils.process_allgather(v, tiled=True) for k, v in out_sharded.items()}
                    fig = cotrain_eval.plot_action_trajectories(
                        out["pred"],
                        out["gt"],
                        action_mask=action_masks[name],
                        slot_names=cotrain_action_space.UNIFIED_SLOT_NAMES,
                        n_samples=config.viz_num_samples,
                        title=f"{name} [{label}] step {step}: pred (--) vs gt",
                    )
                    wandb.log({f"val/{label}/{name}/traj_pred_vs_gt": wandb.Image(fig)}, step=step)
                    import matplotlib.pyplot as plt

                    plt.close(fig)
        except ImportError:
            logging.warning("[eval] matplotlib not available; skipping trajectory visualization.")

    def _run_eval(step: int):
        with sharding.set_mesh(mesh):
            metrics = cotrain_eval.run_eval(
                val_loaders,
                train_state,
                val_flow_step=val_flow_step,
                val_action_mse_steps=val_action_mse_steps,
                flow_mode=config.val_flow_loss_mode,
                run_action_mse=config.run_action_mse,
                num_val_batches=config.num_val_batches,
                num_action_mse_batches=config.num_action_mse_batches,
                val_seed=config.val_seed,
                train_weights=train_weights,
            )
        if metrics:
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in sorted(metrics.items()))
            logging.info(f"[eval] step {step}: {info_str}")
            wandb.log(metrics, step=step)
        if config.viz_action_traj:
            _log_action_traj(step)

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            # nanmean: robot_loss/ego_loss may be NaN on steps with no samples of that domain.
            reduced_info = jax.device_get(jax.tree.map(jnp.nanmean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []
        batch = next(data_iter)

        # Periodic validation (skip step 0, where weights are still the init).
        if config.eval_interval and step > start_step and (
            step % config.eval_interval == 0 or step == config.num_train_steps - 1
        ):
            _run_eval(step)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(cotrain_config.cli())
