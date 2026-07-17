#!/usr/bin/env python
"""Fail-fast validation for the two Baige co-training configurations."""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from openpi.cotrain import action_space, config
from openpi.shared import normalize


def validate(config_name: str, assets_base: Path, params_path: Path) -> None:
    cfg = config.get_config(config_name)
    datasets = cfg.data.datasets
    ids = [dataset.uid for dataset in datasets]
    assert "piper30" in ids and "piper2" in ids
    assert not any(dataset_id.startswith("egoverse_") for dataset_id in ids)
    expected_count = 2 if config_name == "cotrain_real_only" else 37
    assert len(ids) == expected_count, (config_name, len(ids), expected_count)

    for marker in ("_CHECKPOINT_METADATA", "manifest.ocdbt"):
        assert (params_path / marker).is_file(), params_path / marker

    total_frames = 0
    degenerate = []
    for dataset in datasets:
        builder_dir = Path(dataset.builder_dir)
        assert (builder_dir / "dataset_info.json").is_file(), builder_dir
        directory = assets_base / config_name / dataset.uid
        for filename in ("norm_stats.json", "norm_stats_meta.json", "unified_action_space.json"):
            assert (directory / filename).is_file(), directory / filename

        spec = action_space.UNIFIED_ACTION_SPECS[dataset.uid]
        action_space.validate_metadata(directory, spec)
        stats = normalize.load(directory)
        meta = json.loads((directory / "norm_stats_meta.json").read_text())
        assert Path(meta["builder_dir"]) == builder_dir, (dataset.uid, meta["builder_dir"], builder_dir)
        assert meta["num_frames"] > 0, dataset.uid
        total_frames += int(meta["num_frames"])

        state_mask = np.zeros(action_space.UNIFIED_ACTION_DIM, dtype=bool)
        state_mask[list(spec.state_target_slots)] = True
        for key, active in (("state", state_mask), ("actions", np.asarray(spec.action_mask, dtype=bool))):
            value = stats[key]
            arrays = {field: np.asarray(getattr(value, field)) for field in ("mean", "std", "q01", "q99")}
            assert all(array.shape == (action_space.UNIFIED_ACTION_DIM,) for array in arrays.values())
            assert all(np.isfinite(array).all() for array in arrays.values())
            inactive = ~active
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
    parser.add_argument("config", choices=("cotrain_real_only", "cotrain_real_robot"))
    parser.add_argument("--assets-base", type=Path, default=Path("assets"))
    parser.add_argument("--params-path", type=Path, default=Path(os.environ["PARAMS_PATH"]))
    args = parser.parse_args()
    validate(args.config, args.assets_base.resolve(), args.params_path.resolve())


if __name__ == "__main__":
    main()
