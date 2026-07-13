#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from evaluate_validation import DEFAULT_DATASET_DIR, DEFAULT_TARGET_ROOT, ensure_anchor_manifest


def main() -> int:
    p = argparse.ArgumentParser(description="Generate deterministic validation anchor manifest.")
    p.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    p.add_argument("--split", default="seen_test")
    p.add_argument("--horizon", type=int, required=True)
    p.add_argument("--anchors-per-episode", type=int, default=20)
    p.add_argument("--episodes", type=int, default=0)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    rows = ensure_anchor_manifest(
        args.output,
        args.dataset_dir,
        args.split,
        args.horizon,
        args.anchors_per_episode,
        args.episodes if args.episodes > 0 else None,
    )
    print(f"wrote {len(rows)} anchors to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
