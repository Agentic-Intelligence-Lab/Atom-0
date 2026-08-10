#!/usr/bin/env python3
"""Preflight a staged EgoScale-inspired run before allocating GPUs."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import sys


def _params_look_valid(path: Path) -> bool:
    if path.is_file():
        if path.suffix != ".npz":
            return False
        import numpy as np

        with np.load(path, allow_pickle=False) as checkpoint:
            keys = checkpoint.files
            return any(key.startswith("params/img/") for key in keys) and any(
                key.startswith("params/llm/") for key in keys
            )
    if not path.is_dir():
        return False
    markers = ("_CHECKPOINT_METADATA", "manifest.ocdbt", "_METADATA")
    return any((path / marker).exists() for marker in markers) or any(path.iterdir())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--assets-base-dir", default="./assets")
    parser.add_argument("--params-path")
    parser.add_argument("--allow-missing-norm-stats", action="store_true")
    args = parser.parse_args()

    # Import after ATOM_RLDS_ROOT is set; config paths are resolved at import time.
    from openpi.cotrain import action_space
    from openpi.cotrain import config as cotrain_config

    config = cotrain_config.get_config(args.config_name)
    datasets = cotrain_config._resolve_unified_datasets(config.data.datasets, config.model)
    config = dataclasses.replace(config, assets_base_dir=args.assets_base_dir)
    assets_root = Path(config.assets_dirs)
    failures: list[str] = []

    print(f"config={config.name}")
    print(f"ATOM_RLDS_ROOT={os.environ.get('ATOM_RLDS_ROOT', '/mnt/data/RLDS')}")
    print(f"datasets={len(datasets)} assets={assets_root}")
    for dataset in datasets:
        builder = Path(dataset.builder_dir) if dataset.builder_dir is not None else None
        builder_ok = (
            builder is not None
            and (builder / "dataset_info.json").exists()
            and (builder / "features.json").exists()
        )
        stats_dir = assets_root / dataset.uid
        stats_ok = (stats_dir / "norm_stats.json").exists()
        mapping_ok = (stats_dir / "unified_action_space.json").exists()
        status = "OK" if builder_ok and stats_ok and mapping_ok else "MISSING"
        print(
            f"[{status}] {dataset.uid}: builder={builder} "
            f"norm_stats={stats_ok} mapping_meta={mapping_ok}"
        )
        if not builder_ok:
            failures.append(f"missing/invalid TFDS builder: {builder}")
        if not args.allow_missing_norm_stats:
            if not stats_ok:
                failures.append(f"missing norm stats: {stats_dir / 'norm_stats.json'}")
            elif not mapping_ok:
                failures.append(f"missing mapping metadata: {stats_dir / 'unified_action_space.json'}")
            else:
                try:
                    action_space.validate_metadata(stats_dir, dataset.unified_action_spec)
                except ValueError as exc:
                    failures.append(str(exc))

    precomputed_datasets = [dataset for dataset in datasets if dataset.precomputed_action_chunk]
    if precomputed_datasets and not args.allow_missing_norm_stats:
        metadata_path = assets_root / "action_chunk_metadata.json"
        if not metadata_path.exists():
            failures.append(f"missing precomputed action metadata: {metadata_path}")
        else:
            metadata = json.loads(metadata_path.read_text())
            if metadata.get("version") == 1:
                # Backward compatibility for the committed Stage 1 EgoVerse stats.
                expected = {
                    "version": 1,
                    "action_source": "actions_cartesian",
                    "source_action_horizon": 100,
                    "model_action_horizon": config.model.action_horizon,
                    "resampling": "uniform_full_window",
                    "dataset_ids": sorted(dataset.uid for dataset in precomputed_datasets),
                }
            elif metadata.get("version") in (2, 3, 4):
                expected = {
                    "model_action_horizon": config.model.action_horizon,
                    "resampling": "uniform_full_window",
                    "datasets": {
                        dataset.uid: {
                            "action_source": dataset.precomputed_action_source,
                            "source_action_horizon": dataset.precomputed_action_horizon,
                        }
                        for dataset in sorted(precomputed_datasets, key=lambda item: item.uid)
                    },
                }
            else:
                failures.append(
                    f"unsupported precomputed action metadata version at {metadata_path}: "
                    f"{metadata.get('version')!r}"
                )
                expected = None

            actual = None
            if expected is not None:
                actual = {key: metadata.get(key) for key in expected}
                if metadata.get("version") in (3, 4) and isinstance(actual.get("datasets"), dict):
                    # Later versions record additional physical-time, frame, and gripper
                    # semantics. Validate the training-critical v2 subset while
                    # preserving those provenance fields.
                    actual["datasets"] = {
                        uid: {
                            key: values.get(key)
                            for key in ("action_source", "source_action_horizon")
                        }
                        for uid, values in actual["datasets"].items()
                    }
            if expected is not None and actual != expected:
                failures.append(
                    f"precomputed action metadata mismatch at {metadata_path}: "
                    f"expected {expected}, got {actual}"
                )

    params_path = args.params_path
    if params_path is None and args.config_name == "egoscale_stage1_ego":
        params_path = os.environ.get("ATOM_PI05_BASE_PARAMS")
    if params_path and not params_path.startswith("gs://"):
        resolved_params = Path(params_path).expanduser().resolve()
        print(f"params={resolved_params}")
        if not _params_look_valid(resolved_params):
            failures.append(f"missing/invalid params source: {resolved_params}")
    elif args.config_name != "egoscale_stage1_ego" and not params_path:
        failures.append("later stages require --params-path pointing to the previous stage's .../<step>/params")

    if failures:
        print("\nPreflight FAILED:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print("\nPreflight OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
