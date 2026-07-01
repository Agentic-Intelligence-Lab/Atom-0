"""Compute full lightweight norm stats for AgiBot only.

This is a probe script for estimating the cost of full-dataset norm statistics.
Unlike compute_cotrain_norm_stats_light.py, it does not repeat the train split and
does not stop at 1,000,000 frames by default.
"""

import dataclasses
import json
import time
from pathlib import Path

import tqdm
import tyro

import compute_cotrain_norm_stats_light as light
import openpi.cotrain.config as cotrain_config
import openpi.shared.normalize as normalize


def main(
    config_name: str = "cotrain_full_all",
    exp_name: str = "cotrain_full_all_agibot_full_norm_probe",
    output_assets_name: str = "cotrain_full_all_agibot_full_norm_probe",
    dataset_id: str = "agibot",
    rlds_data_dir: str | None = None,
    overwrite: bool = False,
    num_parallel_reads: int = 1,
    num_parallel_calls: int = 1,
) -> None:
    config = cotrain_config.get_config(config_name)
    config = dataclasses.replace(config, name=output_assets_name, exp_name=exp_name)
    if rlds_data_dir is not None:
        config = dataclasses.replace(config, data=dataclasses.replace(config.data, rlds_data_dir=rlds_data_dir))

    data_config = config.data.create(config.assets_dirs, config.model)
    matches = [ds for ds in data_config.datasets if ds.uid == dataset_id]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one dataset for dataset_id={dataset_id!r}, got {len(matches)}.")
    dataset_cfg = matches[0]

    out_dir = config.assets_dirs / dataset_cfg.uid
    if out_dir.exists() and not overwrite:
        try:
            normalize.load(out_dir)
            raise FileExistsError(f"Norm stats already exist at {out_dir}; pass --overwrite to recompute.")
        except FileNotFoundError:
            pass

    dataset = light._create_light_dataset(
        data_config,
        dataset_cfg,
        config.model.action_horizon,
        config.batch_size,
        split_label="train",
        shuffle=False,
        repeat=False,
        num_parallel_reads=num_parallel_reads,
        num_parallel_calls=num_parallel_calls,
    )

    print(f"=== Computing FULL LIGHT norm stats for dataset '{dataset_cfg.uid}' ===")
    print(f"  output: {out_dir}")
    print(f"  builder_dir: {dataset_cfg.builder_dir}")
    print(f"  batch_size: {config.batch_size}")
    print("  repeat: False")

    start_time = time.time()
    stats = light._empty_stats()
    n_frames = 0
    n_batches = 0
    for batch in tqdm.tqdm(dataset.as_numpy_iterator(), desc=dataset_cfg.name):
        state, actions = light._state_actions_from_light_batch(batch, dataset_cfg)
        light._update_stats(stats, state, actions)
        n_frames += int(state.shape[0])
        n_batches += 1

    if n_frames == 0:
        raise RuntimeError(f"No frames read for dataset '{dataset_cfg.uid}'.")

    elapsed_sec = time.time() - start_time
    norm_stats = light._finalize_stats(stats)
    normalize.save(out_dir, norm_stats)

    meta = {
        "config_name": config_name,
        "exp_name": exp_name,
        "output_assets_name": output_assets_name,
        "dataset_id": dataset_cfg.uid,
        "builder_dir": dataset_cfg.builder_dir,
        "split": dataset_cfg.train_split,
        "batch_size": config.batch_size,
        "num_batches": n_batches,
        "num_frames": n_frames,
        "elapsed_sec": elapsed_sec,
        "frames_per_sec": n_frames / elapsed_sec if elapsed_sec > 0 else None,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "norm_stats_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    print(f"  accumulated {n_frames} frames in {elapsed_sec / 3600:.2f} h ({meta['frames_per_sec']:.2f} frames/s)")
    print(f"Saved norm stats and metadata to {out_dir}")


if __name__ == "__main__":
    tyro.cli(main)
