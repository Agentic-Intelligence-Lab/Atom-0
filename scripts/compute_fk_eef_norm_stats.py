#!/usr/bin/env python
"""Compute the two FK-EEF norm variants requested for joint2eef.

1) ``cotrain_fk_eef_plus_piper_ego``
   URDF-validated FK-filled datasets + piper2 + piper30 + EgoVerse.

2) ``cotrain_full_all_full_norm``
   Full audited mixture: FK-filled where available, untouched otherwise.

FK validation is separate: run ``scripts/validate_joint2eef_fk.py`` first.

Examples:
  export RLDS_DATA_DIR=/mnt/workspace/RLDS
  PYTHONPATH=src uv run --group rlds python scripts/compute_fk_eef_norm_stats.py \\
      --variant anchor --output-assets-dir assets/cotrain_fk_eef_plus_piper_ego

  PYTHONPATH=src uv run --group rlds python scripts/compute_fk_eef_norm_stats.py \\
      --variant full --output-assets-dir assets/cotrain_full_all_full_norm
"""

from __future__ import annotations

import argparse
from pathlib import Path

import compute_cotrain_full_norm_stats_light as full_norm

from openpi.cotrain import fk_eef


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=("anchor", "full"),
        required=True,
        help="anchor=fk+piper+ego; full=all datasets with FK where available",
    )
    parser.add_argument("--output-assets-dir", type=Path, default=None)
    parser.add_argument("--rlds-data-dir", type=str, default=None)
    parser.add_argument("--dataset-id", type=str, default=None)
    parser.add_argument(
        "--skip-dataset-ids",
        type=str,
        default="",
        help="Comma-separated dataset uids to skip (e.g. piper2,piper30)",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-validate", action="store_true")
    args = parser.parse_args()

    if not args.skip_validate:
        print("=== Validating joint2eef FK registry ===")
        n_ok = 0
        n_fail = 0
        for dataset_id, spec in fk_eef.FK_EEF_SPECS.items():
            result = fk_eef.validate_fk_spec(spec)
            status = "OK" if result.ok else "FAIL"
            print(f"  [{status}] {dataset_id} ({spec.urdf_file}) map={result.mapped_arm_dofs} urdf={result.urdf_arm_dofs}")
            for message in result.messages:
                print(f"      - {message}")
            n_ok += int(result.ok)
            n_fail += int(not result.ok)
        print(f"FK enabled={n_ok}, skipped/fail={n_fail}")
        print("Enabled:", ", ".join(fk_eef.enabled_fk_dataset_ids()) or "(none)")

    if args.variant == "anchor":
        config_name = "cotrain_fk_eef_plus_piper_ego"
        default_out = Path("assets/cotrain_fk_eef_plus_piper_ego")
    else:
        config_name = "cotrain_full_all_full_norm"
        default_out = Path("assets/cotrain_full_all_full_norm")

    output = args.output_assets_dir or default_out
    print(f"\n=== Computing full light norm stats for {config_name} -> {output} ===")
    full_norm.main(
        config_name=config_name,
        output_assets_dir=str(output),
        dataset_id=args.dataset_id,
        skip_dataset_ids=args.skip_dataset_ids,
        rlds_data_dir=args.rlds_data_dir,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
