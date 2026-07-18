#!/usr/bin/env python3
"""Extract action-space metadata from the RLDS datasets used by cotrain_full_all.

The TFDS builders store one episode per TFRecord Example. Episodes can be hundreds of
megabytes because they include image sequences, so this script scans the serialized
protobuf with mmap and decodes only selected scalar metadata features.
"""

from __future__ import annotations

import argparse
import json
import mmap
from pathlib import Path
from typing import Any

DATASET_ROOTS = (
    "AgiBot",
    "DROID",
    "EgoVerse_full",
    "realworld_piper",
    "realworld_piper_2",
    "RoboCOIN",
    "RoboMIND_full",
)

METADATA_FIELDS = (
    "action_is_delta",
    "action_representation",
    "cartesian_frame",
    "control_mode",
    "eef_pose_coordinate_frame",
    "raw_episode_metadata_json",
    "robot_schema_key",
    "robot_type",
    "state_action_schema_json",
)


def _read_varint(data: mmap.mmap, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, offset
        shift += 7
        if shift >= 70:
            raise ValueError("invalid protobuf varint")


def _iter_fields(data: mmap.mmap, start: int, end: int):
    offset = start
    while offset < end:
        tag, offset = _read_varint(data, offset)
        field_number = tag >> 3
        wire_type = tag & 0x07
        if wire_type == 0:
            _, value_end = _read_varint(data, offset)
            yield field_number, wire_type, offset, value_end
            offset = value_end
        elif wire_type == 1:
            yield field_number, wire_type, offset, offset + 8
            offset += 8
        elif wire_type == 2:
            size, value_start = _read_varint(data, offset)
            value_end = value_start + size
            yield field_number, wire_type, value_start, value_end
            offset = value_end
        elif wire_type == 5:
            yield field_number, wire_type, offset, offset + 4
            offset += 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wire_type}")


def _scalar_bytes_features(data: mmap.mmap, requested_keys: set[str]) -> dict[str, bytes]:
    """Read scalar bytes_list features from the first TFRecord Example."""
    record_size = int.from_bytes(data[0:8], "little")
    example_start = 12  # uint64 length + masked CRC32C
    example_end = example_start + record_size
    example_fields = list(_iter_fields(data, example_start, example_end))
    features_field = next(field for field in example_fields if field[0] == 1 and field[1] == 2)

    values: dict[str, bytes] = {}
    for field_number, wire_type, entry_start, entry_end in _iter_fields(data, features_field[2], features_field[3]):
        if field_number != 1 or wire_type != 2:
            continue
        entry_fields = list(_iter_fields(data, entry_start, entry_end))
        key_field = next((field for field in entry_fields if field[0] == 1 and field[1] == 2), None)
        value_field = next((field for field in entry_fields if field[0] == 2 and field[1] == 2), None)
        if key_field is None or value_field is None:
            continue
        key = bytes(data[key_field[2] : key_field[3]]).decode("utf-8")
        if key not in requested_keys:
            continue

        # Feature.field_1(bytes_list) -> BytesList.field_1(value).
        feature_fields = list(_iter_fields(data, value_field[2], value_field[3]))
        bytes_list = next((field for field in feature_fields if field[0] == 1 and field[1] == 2), None)
        if bytes_list is None:
            continue
        bytes_fields = list(_iter_fields(data, bytes_list[2], bytes_list[3]))
        scalar = next((field for field in bytes_fields if field[0] == 1 and field[1] == 2), None)
        if scalar is not None:
            values[key] = bytes(data[scalar[2] : scalar[3]])
    return values


def _first_train_shard(builder_dir: Path) -> Path:
    shards = sorted(builder_dir.glob("*-train.tfrecord-*"))
    if not shards:
        raise FileNotFoundError(f"no train TFRecord shard under {builder_dir}")
    return shards[0]


def _read_builder_metadata(root: Path, builder_dir: Path) -> dict[str, Any]:
    shard = _first_train_shard(builder_dir)
    with shard.open("rb") as file:
        data = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            requested_keys = {f"episode_metadata/{field}" for field in METADATA_FIELDS}
            scalar_features = _scalar_bytes_features(data, requested_keys)
            metadata: dict[str, Any] = {}
            for field in METADATA_FIELDS:
                raw = scalar_features.get(f"episode_metadata/{field}")
                if raw is None:
                    continue
                value = raw.decode("utf-8")
                if field.endswith("_json"):
                    try:
                        value = json.loads(value)
                    except json.JSONDecodeError:
                        pass
                metadata[field] = value
        finally:
            data.close()

    return {
        "builder_dir": str(builder_dir.relative_to(root)),
        "first_train_shard": shard.name,
        "metadata": metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rlds-root", type=Path, default=Path("/mnt/workspace/RLDS"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    builders = []
    for dataset_root in DATASET_ROOTS:
        builders.extend(sorted((args.rlds_root / dataset_root).rglob("features.json")))

    records = [_read_builder_metadata(args.rlds_root, path.parent) for path in builders]
    output = json.dumps(records, ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        print(output, end="")
    else:
        args.output.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
