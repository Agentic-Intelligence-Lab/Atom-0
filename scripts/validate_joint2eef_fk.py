#!/usr/bin/env python
"""Validate joint2eef URDF <-> dataset mappings before FK fill / norm.

Checks for each entry in ``openpi.cotrain.fk_eef.FK_EEF_SPECS``:
  * URDF file exists under assets/urdf
  * listed arm joint names exist and are actuated
  * joint count matches the unified 80D arm mapping DOF
  * zero-configuration FK runs and returns a finite pose

Usage:
  PYTHONPATH=src python scripts/validate_joint2eef_fk.py
  PYTHONPATH=src python scripts/validate_joint2eef_fk.py --urdf-dir assets/urdf --dataset-id droid
"""

from __future__ import annotations

import argparse
from pathlib import Path

from openpi.cotrain import fk_eef


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf-dir", type=Path, default=Path("assets/urdf"))
    parser.add_argument("--dataset-id", type=str, default=None)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when any registry entry fails joint-count / URDF checks.",
    )
    args = parser.parse_args()

    specs = fk_eef.FK_EEF_SPECS
    if args.dataset_id is not None:
        if args.dataset_id not in specs:
            raise SystemExit(f"Unknown dataset_id: {args.dataset_id}")
        specs = {args.dataset_id: specs[args.dataset_id]}

    n_ok = 0
    n_fail = 0
    print(f"{'dataset_id':48s} {'urdf':22s} {'map_dof':10s} {'urdf_dof':10s} status")
    print("-" * 110)
    for dataset_id, spec in specs.items():
        result = fk_eef.validate_fk_spec(spec, urdf_dir=args.urdf_dir)
        status = "OK" if result.ok else "FAIL"
        if result.ok:
            n_ok += 1
        else:
            n_fail += 1
        print(
            f"{dataset_id:48s} {spec.urdf_file:22s} "
            f"{str(result.mapped_arm_dofs):10s} {str(result.urdf_arm_dofs):10s} {status}"
        )
        for message in result.messages:
            print(f"  - {message}")

    print("-" * 110)
    print(f"enabled (OK): {n_ok}")
    print(f"disabled/fail: {n_fail}")
    print("Enabled dataset ids:")
    for dataset_id in fk_eef.enabled_fk_dataset_ids(str(args.urdf_dir.resolve())):
        print(f"  {dataset_id}")
    if args.strict and n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
