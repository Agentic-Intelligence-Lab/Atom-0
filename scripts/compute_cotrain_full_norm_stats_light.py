"""Compute full lightweight cotrain norm stats without a max-frame cap.

This script is meant for production recomputation after the 1M-frame probe. It
does not repeat the train split, so each selected dataset is consumed exactly
once. It can skip datasets that already have full stats, such as a previously
computed AgiBot probe.
"""

import dataclasses
import json
from pathlib import Path
import time

import compute_cotrain_norm_stats_light as light
import tqdm
import tyro

import openpi.cotrain.config as cotrain_config
import openpi.shared.normalize as normalize


def _split_csv(value: str | None) -> set[str]:
    if value is None or value.strip() == "":
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def _copy_existing_stats(src_assets_name: str, dst_assets_dir: Path, dataset_id: str) -> bool:
    candidates = [
        Path("assets") / src_assets_name / dataset_id,
        Path("/mnt/data/xule/pi07_reproduction/assets") / src_assets_name / dataset_id,
    ]
    src_dir = next((path for path in candidates if (path / "norm_stats.json").exists()), candidates[0])
    if not (src_dir / "norm_stats.json").exists():
        return False
    dst_dir = dst_assets_dir / dataset_id
    dst_dir.mkdir(parents=True, exist_ok=True)
    (dst_dir / "norm_stats.json").write_text((src_dir / "norm_stats.json").read_text())
    meta = src_dir / "norm_stats_meta.json"
    if meta.exists():
        (dst_dir / "norm_stats_meta.json").write_text(meta.read_text())
    return True


def _train_split_info(dataset_cfg) -> tuple[int | None, int | None]:
    info_path = Path(dataset_cfg.builder_dir) / "dataset_info.json"
    if not info_path.exists():
        return None, None
    info = json.loads(info_path.read_text())
    split_name = dataset_cfg.resolve_split("train")
    split_info = next((split for split in info["splits"] if split["name"] == split_name), None)
    if split_info is None:
        return None, None
    num_episodes = sum(int(length) for length in split_info["shardLengths"])
    num_bytes = int(split_info["numBytes"])
    return num_episodes, num_bytes


def _load_agibot_bytes_per_frame(copy_from_assets_name: str | None) -> float | None:
    if copy_from_assets_name is None:
        return None
    candidates = [
        Path("assets") / copy_from_assets_name / "agibot",
        Path("/mnt/data/xule/pi07_reproduction/assets") / copy_from_assets_name / "agibot",
    ]
    src_dir = next((path for path in candidates if (path / "norm_stats_meta.json").exists()), None)
    if src_dir is None:
        return None
    meta = json.loads((src_dir / "norm_stats_meta.json").read_text())
    _, num_bytes = _train_split_info(
        type(
            "DatasetCfg",
            (),
            {
                "builder_dir": meta["builder_dir"],
                "resolve_split": lambda self, split: "train",
            },
        )()
    )
    if not num_bytes or not meta.get("num_frames"):
        return None
    return num_bytes / meta["num_frames"]


def _estimated_batches(
    dataset_cfg, batch_size: int, bytes_per_frame: float | None
) -> tuple[int | None, int | None, int | None]:
    num_episodes, num_bytes = _train_split_info(dataset_cfg)
    if num_bytes is None or bytes_per_frame is None or bytes_per_frame <= 0:
        return num_episodes, num_bytes, None
    estimated_frames = max(batch_size, int(round(num_bytes / bytes_per_frame)))
    return num_episodes, num_bytes, max(1, estimated_frames // batch_size)


def main(
    config_name: str = "cotrain_full_all_full_norm",
    output_assets_dir: str = "/mnt/workspace/wudi/Atom-0/assets/cotrain_full_all_full_norm",
    dataset_id: str | None = None,
    skip_dataset_ids: str = "",
    copy_from_assets_name: str | None = None,
    rlds_data_dir: str | None = None,
    overwrite: bool = False,
    num_parallel_reads: int = 1,
    num_parallel_calls: int = 1,
) -> None:
    config = cotrain_config.get_config(config_name)
    if rlds_data_dir is not None:
        config = dataclasses.replace(config, data=dataclasses.replace(config.data, rlds_data_dir=rlds_data_dir))
    # Norm computation needs only the resolved dataset mappings. Do not create the full
    # training transforms here: that would preload existing norm files and could make
    # --overwrite fail on exactly the stale mapping metadata it is meant to replace.
    data_config = light._resolve_light_data_config(config)

    requested = _split_csv(dataset_id)
    skipped = _split_csv(skip_dataset_ids)
    datasets = [ds for ds in data_config.datasets if (not requested or ds.uid in requested) and ds.uid not in skipped]
    if not datasets:
        raise ValueError("No datasets selected.")

    output_root = Path(output_assets_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    run_meta = {
        "config_name": config_name,
        "output_assets_dir": str(output_root),
        "dataset_id": dataset_id,
        "skip_dataset_ids": sorted(skipped),
        "copy_from_assets_name": copy_from_assets_name,
        "datasets": [],
    }
    bytes_per_frame = _load_agibot_bytes_per_frame(copy_from_assets_name)
    if bytes_per_frame is None:
        print("Progress ETA uses no total because AgiBot full metadata was not found.")
    else:
        print(f"Progress ETA is estimated from AgiBot bytes/frame calibration: {bytes_per_frame:.2f} bytes/frame.")

    for dataset_cfg in datasets:
        out_dir = output_root / dataset_cfg.uid
        if (
            copy_from_assets_name is not None
            and dataset_cfg.uid == "agibot"
            and dataset_cfg.unified_action_spec is None
        ):
            if overwrite and _copy_existing_stats(copy_from_assets_name, output_root, dataset_cfg.uid):
                print(f"\n=== Copied full AgiBot stats from assets/{copy_from_assets_name} to {out_dir} ===")
                continue

        if out_dir.exists() and not overwrite:
            try:
                normalize.load(out_dir)
                if (out_dir / "norm_stats_meta.json").exists():
                    print(f"\n=== Skipping '{dataset_cfg.uid}': full norm stats already exist at {out_dir} ===")
                    continue
                print(f"\n=== Recomputing '{dataset_cfg.uid}': existing stats have no full-run metadata ===")
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
            drop_remainder=False,
            num_parallel_reads=num_parallel_reads,
            num_parallel_calls=num_parallel_calls,
        )

        print(f"\n=== Computing FULL LIGHT norm stats for dataset '{dataset_cfg.uid}' ===")
        print(f"  output: {out_dir}")
        print(f"  builder_dir: {dataset_cfg.builder_dir}")
        print(f"  batch_size: {config.batch_size}")
        print("  repeat: False")
        num_episodes, num_bytes, estimated_total_batches = _estimated_batches(
            dataset_cfg, config.batch_size, bytes_per_frame
        )
        if estimated_total_batches is not None:
            print(
                f"  estimated progress total: ~{estimated_total_batches * config.batch_size:,} frames "
                f"from {num_bytes:,} train bytes"
            )
        if num_episodes is not None:
            print(f"  train episodes: {num_episodes:,}")

        start_time = time.time()
        stats = light._empty_stats()
        n_frames = 0
        n_batches = 0
        progress = tqdm.tqdm(
            dataset.as_numpy_iterator(),
            total=estimated_total_batches,
            desc=dataset_cfg.uid,
            unit="batch",
            dynamic_ncols=True,
        )
        for batch in progress:
            state, actions = light._state_actions_from_light_batch(batch, dataset_cfg)
            light._update_stats(stats, state, actions)
            n_frames += int(state.shape[0])
            n_batches += 1
            if progress.total is not None and n_batches > progress.total:
                progress.total = n_batches + 1
                progress.refresh()
            progress.set_postfix(frames=f"{n_frames:,}")

        if n_frames == 0:
            raise RuntimeError(f"No frames read for dataset '{dataset_cfg.uid}'.")

        elapsed_sec = time.time() - start_time
        norm_stats = light._finalize_stats(stats, dataset_cfg)
        normalize.save(out_dir, norm_stats)
        if dataset_cfg.unified_action_spec is not None:
            light.cotrain_action_space.write_metadata(out_dir, dataset_cfg.unified_action_spec)

        meta = {
            "config_name": config_name,
            "dataset_id": dataset_cfg.uid,
            "builder_dir": dataset_cfg.builder_dir,
            "split": dataset_cfg.train_split,
            "batch_size": config.batch_size,
            "num_batches": n_batches,
            "num_frames": n_frames,
            "estimated_total_batches": estimated_total_batches,
            "train_episodes": num_episodes,
            "train_bytes": num_bytes,
            "elapsed_sec": elapsed_sec,
            "frames_per_sec": n_frames / elapsed_sec if elapsed_sec > 0 else None,
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "norm_stats_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        run_meta["datasets"].append(meta)

        print(f"  accumulated {n_frames} frames in {elapsed_sec / 3600:.2f} h ({meta['frames_per_sec']:.2f} frames/s)")
        print(f"Saved norm stats and metadata to {out_dir}")

    (output_root / "full_norm_run_meta.json").write_text(json.dumps(run_meta, indent=2, ensure_ascii=False))
    print(f"\nSaved run metadata to {output_root / 'full_norm_run_meta.json'}")


if __name__ == "__main__":
    tyro.cli(main)
