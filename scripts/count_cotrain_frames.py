"""Count exact train-split FRAME counts per dataset in a cotrain config.

RLDS stores per-shard EPISODE counts, not frames. But the infidata schema keeps
`episode_metadata/num_frames` per episode, so we can sum that WITHOUT decoding any
images (cheap). Use the printed frames to set sampling weights for an "N-epoch" run.

Usage:
    uv run --group rlds python scripts/count_cotrain_frames.py --config-name cotrain_full_all_full_norm
"""

import numpy as np
import tensorflow_datasets as tfds
import tyro

import openpi.cotrain.config as cotrain_config


def cli_main(config_name: str) -> None:
    config = cotrain_config.get_config(config_name)
    datasets = config.data.datasets

    print(f"{'dataset_id':16s} {'episodes':>10s} {'frames':>14s} {'mean_len':>10s}")
    print("-" * 56)
    totals = {}
    for ds in datasets:
        builder = tfds.builder_from_directory(ds.builder_dir)
        split = ds.train_split
        # Read ONLY episode_metadata/num_frames (no steps -> no image decode).
        dset = builder.as_dataset(split=split, shuffle_files=False)
        frames = 0
        eps = 0
        for ep in tfds.as_numpy(dset):
            nf = ep["episode_metadata"]["num_frames"]
            frames += int(np.asarray(nf))
            eps += 1
        totals[ds.uid] = frames
        mean_len = frames / max(eps, 1)
        print(f"{ds.uid:16s} {eps:10d} {frames:14d} {mean_len:10.1f}")

    print("-" * 56)
    print(f"{'TOTAL':16s} {'':>10s} {sum(totals.values()):14d}")
    print("\nframes dict (copy for weight computation):")
    print({k: v for k, v in totals.items()})


if __name__ == "__main__":
    tyro.cli(cli_main)
