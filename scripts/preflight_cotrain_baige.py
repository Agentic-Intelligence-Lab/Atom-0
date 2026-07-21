#!/usr/bin/env python
"""Fail-fast validation for the Baige co-training configurations."""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from openpi.cotrain import action_space
from openpi.cotrain import config
from openpi.shared import normalize

FIX_EXCLUDED_DATASET_IDS = {
    "robocoin_leju_robot_s54_a54",
    "robocoin_agilex_decoupled_magic_s14_a14_fps50",
    "robocoin_agilex_decoupled_magic_s26_a26",
}


def validate(config_name: str, assets_base: Path, params_path: Path) -> None:
    cfg = config.get_config(config_name)
    datasets = cfg.data.datasets
    ids = [dataset.uid for dataset in datasets]
    assert "piper30" in ids
    assert "piper2" in ids
    expected_counts = {
        "cotrain_real_only": 2,
        "cotrain_real_only_legacy32": 2,
        "cotrain_real_robot": 37,
        "cotrain_real_robot_fix": 34,
        "cotrain_full_all_full_norm": 39,
    }
    expected_count = expected_counts[config_name]
    assert len(ids) == expected_count, (config_name, len(ids), expected_count)
    if config_name == "cotrain_full_all_full_norm":
        assert sum(dataset_id.startswith("egoverse_") for dataset_id in ids) == 5
    else:
        assert not any(dataset_id.startswith("egoverse_") for dataset_id in ids)
    if config_name in {"cotrain_real_robot_fix", "cotrain_full_all_full_norm"}:
        assert set(ids).isdisjoint(FIX_EXCLUDED_DATASET_IDS)

    for marker in ("_CHECKPOINT_METADATA", "manifest.ocdbt"):
        assert (params_path / marker).is_file(), params_path / marker

    total_frames = 0
    degenerate = []
    for dataset in datasets:
        builder_dir = Path(dataset.builder_dir)
        assert (builder_dir / "dataset_info.json").is_file(), builder_dir
        source_config = cfg.data.norm_stats_source_config or config_name
        directory = assets_base / source_config / dataset.uid
        for filename in ("norm_stats.json", "norm_stats_meta.json", "unified_action_space.json"):
            assert (directory / filename).is_file(), directory / filename

        spec = action_space.UNIFIED_ACTION_SPECS[dataset.uid]
        action_space.validate_metadata(directory, spec)
        stats = normalize.load(directory)
        if not cfg.data.unified_action_space:
            stats = config.project_unified_norm_stats_to_native(stats, dataset.uid)
        meta = json.loads((directory / "norm_stats_meta.json").read_text())
        assert Path(meta["builder_dir"]) == builder_dir, (dataset.uid, meta["builder_dir"], builder_dir)
        assert meta["num_frames"] > 0, dataset.uid
        total_frames += int(meta["num_frames"])

        if cfg.data.unified_action_space:
            state_mask = np.zeros(action_space.UNIFIED_ACTION_DIM, dtype=bool)
            state_mask[list(spec.state_target_slots)] = True
            masks = (("state", state_mask), ("actions", np.asarray(spec.action_mask, dtype=bool)))
        else:
            masks = (
                ("state", np.ones(dataset.action_dim, dtype=bool)),
                ("actions", np.ones(dataset.action_dim, dtype=bool)),
            )
        for key, active in masks:
            value = stats[key]
            arrays = {field: np.asarray(getattr(value, field)) for field in ("mean", "std", "q01", "q99")}
            expected_dim = action_space.UNIFIED_ACTION_DIM if cfg.data.unified_action_space else dataset.action_dim
            assert all(array.shape == (expected_dim,) for array in arrays.values())
            assert all(np.isfinite(array).all() for array in arrays.values())
            inactive = ~active
            if inactive.any():
                assert np.allclose(arrays["mean"][inactive], 0)
                assert np.allclose(arrays["std"][inactive], 1)
                assert np.allclose(arrays["q01"][inactive], -1)
                assert np.allclose(arrays["q99"][inactive], 1)
            bad = np.flatnonzero(active & (arrays["q99"] <= arrays["q01"]))
            if bad.size:
                degenerate.append(f"{dataset.uid}:{key}:{bad.tolist()}")

    print(f"PASS {config_name}: datasets={len(ids)}, source_frames={total_frames:,}")
    for item in degenerate:
        print(f"WARN degenerate active quantile (smoke test must remain finite): {item}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        choices=(
            "cotrain_real_only",
            "cotrain_real_only_legacy32",
            "cotrain_real_robot",
            "cotrain_real_robot_fix",
            "cotrain_full_all_full_norm",
        ),
    )
    parser.add_argument("--assets-base", type=Path, default=Path("assets"))
    parser.add_argument("--params-path", type=Path, default=Path(os.environ["PARAMS_PATH"]))
    args = parser.parse_args()
    validate(args.config, args.assets_base.resolve(), args.params_path.resolve())


if __name__ == "__main__":
    main()
