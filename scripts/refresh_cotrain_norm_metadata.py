#!/usr/bin/env python
"""Refresh unified_action_space.json for existing cotrain norm stats.

Legacy assets may fail training with:
  Unified norm stats mapping mismatch ... unified_action_space.json

when the 80D mapping registry was updated but norm_stats.json values are still
valid. This script rewrites mapping metadata from the current config registry
without recomputing statistics.

For a full stats refresh (recommended before large production runs), use:
  uv run --group rlds python scripts/compute_cotrain_full_norm_stats_light.py \\
      --config-name cotrain_real_robot_ego_fix \\
      --output-assets-dir assets/cotrain_real_robot_ego_fix \\
      --overwrite
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import tyro

import openpi.cotrain.action_space as cotrain_action_space
import openpi.cotrain.config as cotrain_config


def main(
    config_name: str = "cotrain_real_robot_ego_fix",
    assets_dir: str | None = None,
    dataset_id: str | None = None,
    dry_run: bool = False,
) -> None:
    config = cotrain_config.get_config(config_name)
    if assets_dir is not None:
        config = dataclasses.replace(config, assets_name=Path(assets_dir).name)
    assets_root = Path(assets_dir or config.assets_dirs)
    datasets = cotrain_config._resolve_unified_datasets(config.data.datasets, config.model)

    updated = 0
    skipped = 0
    for ds in datasets:
        if dataset_id is not None and ds.uid != dataset_id:
            continue
        out_dir = assets_root / ds.uid
        if not (out_dir / "norm_stats.json").exists():
            skipped += 1
            continue
        spec = ds.unified_action_spec
        assert spec is not None
        meta_path = out_dir / "unified_action_space.json"
        if dry_run:
            print(f"[dry-run] would refresh {meta_path} -> fingerprint={spec.fingerprint[:12]}...")
        else:
            cotrain_action_space.write_metadata(out_dir, spec)
            print(f"Refreshed {meta_path}")
        updated += 1

    print(f"Done: updated={updated}, skipped_missing_stats={skipped}, assets_root={assets_root}")


if __name__ == "__main__":
    tyro.cli(main)
