#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate_validation import generate_report, parse_args


def main() -> int:
    p = argparse.ArgumentParser(description="Regenerate Chinese Markdown/HTML report from saved manifests.")
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    run_config = json.loads((args.output_dir / "run_config.json").read_text(encoding="utf-8"))
    metadata = {
        "environment": json.loads((args.output_dir / "environment_manifest.json").read_text(encoding="utf-8")),
        "checkpoint_format": json.loads((args.output_dir / "checkpoint_manifest.json").read_text(encoding="utf-8")),
    }
    metadata_path = Path(__file__).resolve().parents[1] / "config" / "training_metadata_0629.json"
    if metadata_path.exists():
        metadata.update(json.loads(metadata_path.read_text(encoding="utf-8")))
    summaries = {}
    for name in ("open_loop_summary", "flow_loss_summary"):
        path = args.output_dir / f"{name}.json"
        summaries[name] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    validation_path = args.output_dir / "validation_status.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8")) if validation_path.exists() else {"ok": False, "errors": ["validation_status.json missing"]}
    cli_args = []
    for key, value in run_config.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                cli_args.append(flag)
        elif value is not None:
            cli_args.extend([flag, str(value)])
    ns = parse_args(cli_args)
    generate_report(args.output_dir, ns, metadata, summaries, validation)
    print(args.output_dir / "report.md")
    print(args.output_dir / "report.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
