"""Compute PER-DATASET normalization statistics for a co-training config.

For each dataset in the chosen cotrain config, this iterates that dataset's standardized
RLDS (train split), accumulates mean/std/quantiles of the native (un-padded) state and
action vectors, and saves them to `<assets_dirs>/<dataset_name>/`. At training time
`DispatchNormalize` loads these per-dataset stats and applies them by `dataset_id`.

Usage:
    uv run python scripts/compute_cotrain_norm_stats.py cotrain_sanity \
        --data.rlds_data_dir=/path/to/standardized_rlds --max-frames=1000000
"""

import dataclasses
from itertools import islice

import numpy as np
import tqdm
import tyro

import openpi.cotrain.config as cotrain_config
import openpi.cotrain.data_loader as cotrain_data_loader
import openpi.shared.normalize as normalize
from openpi.training.data_loader import IterableTransformedDataset


def main(config: cotrain_config.CotrainTrainConfig, max_frames: int = 1_000_000, overwrite: bool = False):
    data_config = config.data.create(config.assets_dirs, config.model)
    batch_size = config.batch_size
    num_batches = max(1, max_frames // batch_size)

    for ds in data_config.datasets:
        out_dir = config.assets_dirs / ds.uid
        # Skip datasets whose stats already exist (unless --overwrite). Lets you re-run the
        # command to fill in only the missing datasets without recomputing the rest.
        if not overwrite:
            try:
                normalize.load(out_dir)
                print(f"\n=== Skipping '{ds.uid}': norm stats already exist at {out_dir} (use --overwrite to redo) ===")
                continue
            except FileNotFoundError:
                pass

        print(f"\n=== Computing norm stats for dataset '{ds.uid}' (split='{ds.train_split}') ===")
        single_dc = dataclasses.replace(data_config, datasets=(dataclasses.replace(ds, weight=1.0),))

        # No shuffle for stats: avoids a huge shuffle buffer (OOM) and deterministically
        # sweeps the whole split in shard order, which is plenty representative for stats.
        dataset = cotrain_data_loader.create_cotrain_rlds_dataset(
            single_dc, config.model.action_horizon, batch_size, split_label="train", shuffle=False
        )
        # Apply repack + data transforms (StandardizedInputs + DispatchNormalize). With no
        # stats present yet, DispatchNormalize is a no-op, so state/actions stay un-normalized.
        dataset = IterableTransformedDataset(
            dataset,
            [*single_dc.repack_transforms.inputs, *single_dc.data_transforms.inputs],
            is_batched=True,
        )

        stats = {"state": normalize.RunningStats(), "actions": normalize.RunningStats()}
        n_frames = 0
        for batch in tqdm.tqdm(islice(iter(dataset), num_batches), total=num_batches, desc=ds.name):
            for key in ("state", "actions"):
                x = np.asarray(batch[key])
                stats[key].update(x.reshape(-1, x.shape[-1]))
            n_frames += int(np.asarray(batch["state"]).shape[0])

        if n_frames == 0:
            raise RuntimeError(
                f"No frames read for dataset '{ds.uid}' (split '{ds.train_split}'). Check the "
                f"RLDS path / split name / restructure field mapping."
            )
        print(f"  accumulated {n_frames} frames")
        norm_stats = {k: s.get_statistics() for k, s in stats.items()}
        normalize.save(out_dir, norm_stats)
        print(f"Saved norm stats for '{ds.uid}' to {out_dir}")


def cli_main(
    config_name: str,
    exp_name: str,
    max_frames: int = 1_000_000,
    rlds_data_dir: str | None = None,
    overwrite: bool = False,
) -> None:
    config = cotrain_config.get_config(config_name)
    config = dataclasses.replace(config, exp_name=exp_name)
    if rlds_data_dir is not None:
        config = dataclasses.replace(config, data=dataclasses.replace(config.data, rlds_data_dir=rlds_data_dir))
    main(config, max_frames, overwrite=overwrite)


if __name__ == "__main__":
    tyro.cli(cli_main)
