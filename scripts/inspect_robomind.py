"""Inspect a RoboMIND (or any) RLDS to confirm assumptions made by the cotrain loader.

Run this ON THE SERVER THAT HAS THE DATA. It loads the dataset through the SAME path the
cotrain loader uses (dlimp `DLataset.from_rlds`) and reports:

  1. Images: encoded (bytes/string) vs already-decoded (uint8 HWC). This decides whether the
     loader's `tf.io.decode_image` step is correct or must be removed.
  2. Action representation: delta vs absolute -- from episode metadata
     (action_is_delta / action_representation / control_mode / state_action_schema_json) and
     from action value statistics (deltas hover near 0; absolute joint angles do not).
  3. Cameras: whether every inspected episode has all 3 expected cameras, plus camera_mapping_json.

Usage (on the data server):
    uv run python scripts/inspect_robomind.py \
        --data_dir /mnt/workspace/RLDS/RoboMIND --name robomind_infidata --version 1.1.0 \
        --split train --num_episodes 5
"""

import argparse
import json

import numpy as np

EXPECTED_CAMS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
META_KEYS = (
    "action_is_delta",
    "action_representation",
    "state_representation",
    "control_mode",
    "state_action_schema_json",
    "camera_mapping_json",
    "robot_type",
    "robots_json",
)


def _describe_tree(x, prefix="", out=None):
    """Recursively print dtype/shape (+ a sample for strings) of a nested numpy dict."""
    if out is None:
        out = []
    if isinstance(x, dict):
        for k in sorted(x):
            _describe_tree(x[k], f"{prefix}/{k}", out)
    else:
        a = np.asarray(x)
        line = f"  {prefix:52s} dtype={str(a.dtype):10s} shape={tuple(a.shape)}"
        if a.dtype.kind in ("S", "O", "U") and a.size:
            v = a.reshape(-1)[0]
            line += f"  sample={str(v)[:70]!r}"
        out.append(line)
    return out


def _collect(x, names, prefix="", found=None):
    """Find leaves whose key name is in `names`, anywhere in the tree."""
    if found is None:
        found = {}
    if isinstance(x, dict):
        for k, v in x.items():
            if k in names:
                found[f"{prefix}/{k}"] = v
            _collect(v, names, f"{prefix}/{k}", found)
    return found


def _is_encoded_image(arr) -> bool:
    a = np.asarray(arr)
    # Encoded -> bytes/string leaves (one encoded blob per frame). Decoded -> uint8 with HWC.
    if a.dtype.kind in ("S", "O", "U"):
        return True
    return not (a.dtype == np.uint8 and a.ndim >= 3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="/mnt/workspace/RLDS/RoboMIND")
    p.add_argument("--name", default="robomind_infidata")
    p.add_argument("--version", default="1.1.0")
    p.add_argument("--split", default="train")
    p.add_argument("--num_episodes", type=int, default=5)
    args = p.parse_args()

    import dlimp as dl
    import tensorflow as tf
    import tensorflow_datasets as tfds

    tf.config.set_visible_devices([], "GPU")

    builder = tfds.builder(args.name, data_dir=args.data_dir, version=args.version)
    print(f"\n=== Available splits ===")
    for sname, info in builder.info.splits.items():
        print(f"  {sname}: {info.num_examples} examples")

    ds = dl.DLataset.from_rlds(builder, split=args.split, shuffle=False)

    first = True
    for ep_idx, traj in enumerate(ds.take(args.num_episodes).as_numpy_iterator()):
        if first:
            print(f"\n=== Full structure of first trajectory (split='{args.split}') ===")
            print("\n".join(_describe_tree(traj)))
            first = False

        print(f"\n=== Episode {ep_idx} ===")

        # --- Point 1 & 3: images ---
        imgs = _collect(traj, set(EXPECTED_CAMS))
        if not imgs:
            print("  [images] could not locate cam_* keys -- check structure dump above.")
        for path, arr in imgs.items():
            a = np.asarray(arr)
            enc = _is_encoded_image(a)
            kind = "ENCODED(bytes)" if enc else "DECODED(uint8 array)"
            extra = ""
            if not enc and a.dtype == np.uint8 and a.ndim >= 3:
                # report whether frames look like all-zero placeholders (missing camera)
                nonzero = float(np.mean(a != 0))
                extra = f"  frac_nonzero={nonzero:.3f}"
            print(f"  [image] {path:40s} {kind:22s} dtype={a.dtype} shape={tuple(a.shape)}{extra}")
        present = {c for c in EXPECTED_CAMS if any(c in pth for pth in imgs)}
        missing = set(EXPECTED_CAMS) - present
        print(f"  [cameras] present={sorted(present)} missing={sorted(missing)}")

        # --- Point 2: action representation ---
        meta = _collect(traj, set(META_KEYS))
        for path, v in meta.items():
            a = np.asarray(v)
            val = a.reshape(-1)[0] if a.size else None
            if isinstance(val, bytes):
                val = val.decode("utf-8", "replace")
            print(f"  [meta] {path:48s} = {str(val)[:120]}")

        act = _collect(traj, {"action"})
        for path, v in act.items():
            a = np.asarray(v, dtype=np.float32)
            if a.ndim == 2:
                print(
                    f"  [action] {path} shape={a.shape}\n"
                    f"           per-dim mean={np.array2string(a.mean(0), precision=3, max_line_width=200)}\n"
                    f"           per-dim std ={np.array2string(a.std(0), precision=3, max_line_width=200)}\n"
                    f"           global |mean|={np.abs(a.mean()):.4f} (near 0 => likely DELTA; large => ABSOLUTE)\n"
                    f"           first row   ={np.array2string(a[0], precision=3, max_line_width=200)}"
                )

    print("\nDone. Interpretation:")
    print("  * If images are ENCODED -> the loader's decode step is correct (keep it).")
    print("  * If images are DECODED uint8 -> remove tf.io.decode_image in the standardized path.")
    print("  * action_is_delta / |mean| near 0 -> delta; otherwise absolute (consider delta conversion).")
    print("  * Any 'missing' cameras -> set that slot's image_mask=False instead of all-True.")


if __name__ == "__main__":
    main()
