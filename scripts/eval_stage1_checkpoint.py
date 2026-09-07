#!/usr/bin/env python3
"""Read-only offline evaluation for an EgoScale Stage-1 checkpoint."""

import argparse
import dataclasses
import json
import logging
import os
import pathlib
import platform
import time

import jax

import openpi.cotrain.config as cotrain_config
import openpi.cotrain.data_loader as cotrain_data_loader
import openpi.cotrain.eval as cotrain_eval
import openpi.training.sharding as sharding
import openpi.training.weight_loaders as weight_loaders
import train_cotrain


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", required=True)
    parser.add_argument("--assets-base-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fsdp-devices", type=int, default=8)
    return parser.parse_args()


def save_json(path: pathlib.Path, payload: dict) -> None:
    if jax.process_index() != 0:
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def main() -> None:
    args = parse_args()
    train_cotrain.init_logging()
    train_cotrain._maybe_init_jax_distributed()

    params = pathlib.Path(args.params)
    assets_base_dir = pathlib.Path(args.assets_base_dir)
    output_dir = pathlib.Path(args.output_dir)
    if not params.is_dir():
        raise FileNotFoundError(params)
    if not (params / "manifest.ocdbt").is_file():
        raise FileNotFoundError(params / "manifest.ocdbt")
    if not (assets_base_dir / "egoscale_stage1_ego_cartesian_clean").is_dir():
        raise FileNotFoundError(assets_base_dir / "egoscale_stage1_ego_cartesian_clean")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = dataclasses.replace(
        cotrain_config.get_config("egoscale_stage1_ego"),
        assets_base_dir=str(assets_base_dir),
        weight_loader=weight_loaders.CheckpointWeightLoader(str(params)),
        fsdp_devices=args.fsdp_devices,
    )
    if config.val_batch_size != 96:
        raise ValueError(f"Unexpected native val_batch_size: {config.val_batch_size}")
    if config.num_val_batches != 10 or config.num_action_mse_batches != 2:
        raise ValueError(
            "Unexpected native eval coverage: "
            f"flow={config.num_val_batches}, mse={config.num_action_mse_batches}"
        )

    started = time.time()
    status = {
        "status": "initializing",
        "host": platform.node(),
        "params": str(params),
        "dataset_root": os.environ.get("ATOM_RLDS_ROOT"),
        "config": config.name,
        "val_batch_size": config.val_batch_size,
        "num_val_batches": config.num_val_batches,
        "num_action_mse_batches": config.num_action_mse_batches,
        "val_flow_loss_mode": config.val_flow_loss_mode,
        "val_flow_loss_num_samples": config.val_flow_loss_num_samples,
        "action_mse_num_denoise_steps": config.action_mse_num_denoise_steps,
        "metric_space": "normalized Unified80, masked to the 12 EgoVerse Cartesian slots",
        "started_unix": started,
        "completed": [],
        "metrics": {},
    }
    save_json(output_dir / "status.json", status)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS)
    )
    loaders = cotrain_data_loader.build_val_loaders(config, sharding=data_sharding)
    train_weights = cotrain_data_loader.dataset_train_weights(config)
    action_masks = cotrain_data_loader.dataset_action_masks(config)

    _, init_rng = jax.random.split(jax.random.key(config.seed))
    state, _ = train_cotrain.init_train_state(config, init_rng, mesh, resume=False)
    jax.block_until_ready(state)
    logging.info("Checkpoint restored successfully from %s", params)

    flow_step = cotrain_eval.make_val_flow_step(
        num_samples=config.val_flow_loss_num_samples,
        mode=config.val_flow_loss_mode,
        use_ema=config.eval_on_ema,
    )
    mse_steps = {
        name: cotrain_eval.make_val_action_mse_step(
            num_denoise_steps=config.action_mse_num_denoise_steps,
            fallback_mask=mask,
            use_ema=config.eval_on_ema,
        )
        for name, mask in action_masks.items()
    }

    first_label = "seen"
    first_name = next(iter(loaders[first_label]))
    logging.info("Starting one-batch smoke: %s/%s", first_label, first_name)
    with sharding.set_mesh(mesh):
        smoke = cotrain_eval.run_eval(
            {first_label: {first_name: loaders[first_label][first_name]}},
            state,
            val_flow_step=flow_step,
            val_action_mse_steps=mse_steps,
            flow_mode=config.val_flow_loss_mode,
            run_action_mse=False,
            num_val_batches=1,
            num_action_mse_batches=0,
            val_seed=config.val_seed,
            train_weights=train_weights,
        )
    status["smoke"] = smoke
    status["status"] = "running"
    save_json(output_dir / "status.json", status)
    logging.info("Smoke passed: %s", smoke)

    for label in ("seen", "unseen"):
        for name, loader in loaders[label].items():
            logging.info("Evaluating %s/%s", label, name)
            with sharding.set_mesh(mesh):
                result = cotrain_eval.run_eval(
                    {label: {name: loader}},
                    state,
                    val_flow_step=flow_step,
                    val_action_mse_steps=mse_steps,
                    flow_mode=config.val_flow_loss_mode,
                    run_action_mse=config.run_action_mse,
                    num_val_batches=config.num_val_batches,
                    num_action_mse_batches=config.num_action_mse_batches,
                    val_seed=config.val_seed,
                    train_weights=train_weights,
                )
            status["metrics"].update(result)
            status["completed"].append(f"{label}/{name}")
            status["elapsed_seconds"] = time.time() - started
            save_json(output_dir / "status.json", status)
            logging.info("Completed %s/%s: %s", label, name, result)

    for label in ("seen", "unseen"):
        for suffix in ("flow_loss_fixed", "flow_loss_multi", "action_mse"):
            per_dataset = {
                name: status["metrics"][f"val/{label}/{name}/{suffix}"]
                for name in loaders[label]
                if f"val/{label}/{name}/{suffix}" in status["metrics"]
            }
            cotrain_eval._add_aggregate(
                status["metrics"],
                f"val/{label}/agg/{suffix}",
                per_dataset,
                train_weights,
            )

    status["status"] = "complete"
    status["elapsed_seconds"] = time.time() - started
    status["finished_unix"] = time.time()
    save_json(output_dir / "status.json", status)
    save_json(output_dir / "metrics.json", status["metrics"])
    logging.info("FORMAL_EVAL_COMPLETE %s", json.dumps(status["metrics"], sort_keys=True))


if __name__ == "__main__":
    main()
