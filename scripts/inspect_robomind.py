"""Inspect a RoboMIND (or any) RLDS to confirm assumptions made by the cotrain loader.

Run this ON THE SERVER THAT HAS THE DATA. Uses plain `tensorflow_datasets` (no dlimp needed)
and reports:

  1. Images: stored ENCODED (tfds Image feature) vs raw decoded Tensor. Our loader uses
     dlimp `from_rlds` which SKIPS decoding, so an Image feature => the loader receives
     ENCODED bytes => the pipeline's `tf.io.decode_image` step is correct. A raw Tensor =>
     remove that decode step.
  2. Action representation: delta vs absolute -- from episode metadata
     (action_is_delta / action_representation / control_mode / state_action_schema_json) and
     from action value statistics (deltas hover near 0; absolute joint angles do not).
  3. Cameras: whether every inspected episode has all 3 expected cameras (+ camera_mapping_json).

Usage (on the data server):
    python scripts/inspect_robomind.py \
        --data_dir /mnt/workspace/RLDS/RoboMIND --name robomind_infidata --version 1.1.0 \
        --split train --num_episodes 5
"""

import argparse

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


def _to_str(v):
    a = np.asarray(v)
    x = a.reshape(-1)[0] if a.size else a
    return x.decode("utf-8", "replace") if isinstance(x, bytes) else str(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="/mnt/workspace/RLDS/RoboMIND")
    p.add_argument("--name", default="robomind_infidata")
    p.add_argument("--version", default="1.1.0")
    p.add_argument("--split", default="train")
    p.add_argument("--num_episodes", type=int, default=5)
    args = p.parse_args()

    import tensorflow as tf
    import tensorflow_datasets as tfds

    tf.config.set_visible_devices([], "GPU")

    builder = tfds.builder(args.name, data_dir=args.data_dir, version=args.version)

    print("\n=== Available splits ===")
    for sname, info in builder.info.splits.items():
        print(f"  {sname}: {info.num_examples} examples")

    # --- Point 1: image storage (encoded vs raw) via feature types ---
    print("\n=== Image feature types (Point 1: encoded vs decoded) ===")
    try:
        steps_feat = builder.info.features["steps"]
        inner = getattr(steps_feat, "feature", steps_feat)  # Sequence -> inner FeaturesDict
        img_feats = inner["observation"]["images"]
        for c in EXPECTED_CAMS:
            if c in img_feats:
                fcls = type(img_feats[c]).__name__
                verdict = (
                    "ENCODED  -> keep tf.io.decode_image in the loader"
                    if fcls == "Image"
                    else "RAW Tensor -> REMOVE the decode step"
                )
                print(f"  {c:20s} feature={fcls:10s}  {verdict}")
            else:
                print(f"  {c:20s} NOT FOUND in features")
    except Exception as e:  # noqa: BLE001
        print(f"  could not introspect features ({e}); see structure dump below.")

    # --- Load episodes (default decoding -> images come back as uint8 arrays) ---
    ds = builder.as_dataset(split=args.split, shuffle_files=False)

    for ep_idx, ep in enumerate(tfds.as_numpy(ds.take(args.num_episodes))):
        print(f"\n=== Episode {ep_idx} ===")
        meta = ep.get("episode_metadata", {})
        steps = ep["steps"]  # already a list/array of per-step dicts under tfds.as_numpy

        # materialize steps into stacked arrays
        step_list = list(steps) if not isinstance(steps, dict) else None

        # --- Point 2: action representation ---
        for k in META_KEYS:
            if k in meta:
                print(f"  [meta] {k:28s} = {_to_str(meta[k])[:140]}")

        if step_list:
            try:
                act = np.stack([np.asarray(s["action"], np.float32) for s in step_list])
                print(
                    f"  [action] shape={act.shape}\n"
                    f"           per-dim mean={np.array2string(act.mean(0), precision=3, max_line_width=200)}\n"
                    f"           per-dim std ={np.array2string(act.std(0), precision=3, max_line_width=200)}\n"
                    f"           global |mean|={np.abs(act.mean()):.4f}  (near 0 => likely DELTA; large => ABSOLUTE)\n"
                    f"           first row   ={np.array2string(act[0], precision=3, max_line_width=200)}"
                )
            except Exception as e:  # noqa: BLE001
                print(f"  [action] could not stack ({e})")

            # --- Point 3: cameras present + non-trivial ---
            s0 = step_list[0]
            obs_imgs = s0.get("observation", {}).get("images", {})
            present, missing = [], []
            for c in EXPECTED_CAMS:
                if c in obs_imgs:
                    a = np.asarray(obs_imgs[c])
                    nz = float(np.mean(a != 0)) if a.dtype == np.uint8 else None
                    present.append(c)
                    print(f"  [image] {c:18s} dtype={a.dtype} shape={tuple(a.shape)} frac_nonzero={nz}")
                else:
                    missing.append(c)
            print(f"  [cameras] present={present} missing={missing}")

            if ep_idx == 0:
                print(f"  [first step keys] {sorted(s0.keys())}")
                print(f"  [observation keys] {sorted(s0.get('observation', {}).keys())}")

    print("\nInterpretation:")
    print("  * Image feature 'Image' => ENCODED in storage; dlimp from_rlds yields bytes => keep decode step.")
    print("  * action_is_delta / |mean| near 0 => delta; otherwise absolute (maybe add delta conversion).")
    print("  * Any 'missing' camera or frac_nonzero==0 => set that slot's image_mask=False (not all-True).")


if __name__ == "__main__":
    main()
