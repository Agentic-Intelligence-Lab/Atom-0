#!/usr/bin/env python3
"""Full-episode FastWAM joint inference (slide windows, stitch results).

Unlike ``run_fastwam_joint_sample.py`` (one action/video chunk), this loads a
complete RLDS episode, runs ``infer_joint`` on sliding windows, and writes:

  gt_video.mp4 / pred_video.mp4
  gt_action_native_14d.{npy,png} / pred_action_native_14d.{npy,png}
  action_native_14d_overlay.png

Example (shortest seen_test piper2 episode, ckpt 29999)::

  cd /data/zjyang/Atom-0 && source scripts/atom0_env.sh
  export DIFFSYNTH_MODEL_BASE_PATH=\"$(pwd)/checkpoints/fastwam\"
  PYTHONPATH=src .venv/bin/python scripts/run_fastwam_joint_episode.py \\
    --checkpoint checkpoints/wam-cross-piper/fw-wam-cross-piper-v1-8gpu-b208-30k/29999 \\
    --dataset piper2 --split seen_test --prefer-short \\
    --image-resolution 576,512 \\
    --output-dir tmp/fastwam_joint/piper2-ep-full
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro

_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from run_fastwam_joint_sample import (  # noqa: E402
    _build_output_transforms,
    _parse_hw,
    _postprocess_actions,
    _save_mp4,
    _save_native_trajectories,
    _video_frame_to_uint8_hwc,
    init_logging,
)


CAMERA_SLOTS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
RAW_CAM_KEYS = {
    "base_0_rgb": "cam_high",
    "left_wrist_0_rgb": "cam_left_wrist",
    "right_wrist_0_rgb": "cam_right_wrist",
}
PROMPT_PREFIX = "Action Mode: joint. "


@dataclasses.dataclass
class Args:
    checkpoint: Path
    config_name: str = "wam-cross-piper-ft"
    dataset: str = "piper2"
    split: str = "seen_test"
    """TFDS split name: train / seen_test / unseen_test."""

    episode_index: int | None = None
    """If set, load this episode_index. Else first matching episode (or shortest if --prefer-short)."""

    prefer_short: bool = False
    """Pick the shortest episode with length >= action_horizon (smoke-friendly)."""

    stride: int | None = None
    """Action stitch stride. Default = action_horizon (non-overlapping)."""

    image_resolution: str | None = "576,512"
    output_dir: Path = Path("tmp/fastwam_joint/episode")
    device: str = "cuda:0"
    num_inference_steps: int = 20
    seed: int = 42
    rlds_data_dir: Path | None = None
    assets_base_dir: Path | None = None
    video_fps: int | None = None
    """Override mp4 fps; default uses episode metadata fps."""


def _configure_rlds_root(args: Args) -> None:
    if args.rlds_data_dir is not None:
        os.environ["RLDS_DATA_DIR"] = str(args.rlds_data_dir)
    elif "RLDS_DATA_DIR" not in os.environ:
        os.environ["RLDS_DATA_DIR"] = "/mnt/bos/bo23lu"


def _decode_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "numpy"):
        value = value.numpy()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _builder_dir_for_dataset(dataset: str) -> Path:
    import openpi.cotrain.config as cotrain_config

    if dataset == "piper2":
        return Path(cotrain_config._PIPER2_BUILDER_DIR)
    if dataset == "piper30":
        return Path(cotrain_config._PIPER30_BUILDER_DIR)
    raise ValueError(f"Supported datasets: piper2, piper30; got {dataset!r}")


def _select_episode_ordinal(
    builder_dir: Path,
    split: str,
    *,
    episode_index: int | None,
    prefer_short: bool,
    min_frames: int,
) -> tuple[int, dict[str, Any]]:
    """Return (ordinal_in_split, meta_row). Scans metadata only."""
    import tensorflow_datasets as tfds

    builder = tfds.builder_from_directory(str(builder_dir))
    ds = builder.as_dataset(split=split, shuffle_files=False)

    rows: list[dict[str, Any]] = []
    for ordinal, ep in enumerate(ds):
        meta = ep["episode_metadata"]
        row = {
            "ordinal": ordinal,
            "episode_index": int(meta["episode_index"].numpy()),
            "num_frames": int(meta["num_frames"].numpy()),
            "task": _decode_text(meta["task"]),
            "fps": int(meta["fps"].numpy()) if "fps" in meta else 30,
        }
        rows.append(row)
        if episode_index is not None and row["episode_index"] == episode_index:
            if row["num_frames"] < min_frames:
                raise ValueError(
                    f"episode_index={episode_index} has {row['num_frames']} frames < min {min_frames}"
                )
            return ordinal, row

    if episode_index is not None:
        raise ValueError(f"episode_index={episode_index} not found in {split}")

    candidates = [r for r in rows if r["num_frames"] >= min_frames]
    if not candidates:
        raise RuntimeError(f"No episode in {split} with >= {min_frames} frames")
    chosen = min(candidates, key=lambda r: r["num_frames"]) if prefer_short else candidates[0]
    return int(chosen["ordinal"]), chosen


def _load_full_episode(builder_dir: Path, split: str, ordinal: int) -> dict[str, Any]:
    import tensorflow_datasets as tfds

    builder = tfds.builder_from_directory(str(builder_dir))
    ds = builder.as_dataset(split=split, shuffle_files=False)
    ep = None
    for i, example in enumerate(ds):
        if i == ordinal:
            ep = example
            break
    if ep is None:
        raise RuntimeError(f"Failed to load split ordinal {ordinal}")

    meta = ep["episode_metadata"]
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    images = {slot: [] for slot in CAMERA_SLOTS}
    prompts: list[str] = []

    for step in ep["steps"].as_numpy_iterator():
        states.append(np.asarray(step["observation"]["state"], dtype=np.float32))
        actions.append(np.asarray(step["action"], dtype=np.float32))
        prompts.append(_decode_text(step["task"]))
        obs_imgs = step["observation"]["images"]
        for slot, raw_key in RAW_CAM_KEYS.items():
            images[slot].append(np.asarray(obs_imgs[raw_key]))

    t_len = len(actions)
    return {
        "episode_index": int(meta["episode_index"].numpy()),
        "num_frames": t_len,
        "fps": int(meta["fps"].numpy()) if "fps" in meta else 30,
        "task": _decode_text(meta["task"]),
        "state": np.stack(states, axis=0),
        "action": np.stack(actions, axis=0),
        "images": {slot: np.stack(frames, axis=0) for slot, frames in images.items()},
        "prompt": prompts[0] if prompts else _decode_text(meta["task"]),
    }


def _make_anchors(num_frames: int, horizon: int, stride: int) -> list[int]:
    if num_frames < horizon:
        raise ValueError(f"episode length {num_frames} < action_horizon {horizon}")
    last = num_frames - horizon
    anchors = list(range(0, last + 1, stride))
    if anchors[-1] != last:
        anchors.append(last)
    return anchors


def _uint8_hwc_to_model(img: np.ndarray) -> np.ndarray:
    """HWC uint8 -> HWC float32 in [-1, 1]."""
    return img.astype(np.float32) / 255.0 * 2.0 - 1.0


def _compose_frame(
    images_t: dict[str, np.ndarray],
    *,
    model_cfg,
    device: torch.device,
) -> np.ndarray:
    """Compose one timestep of raw uint8 cams -> HWC uint8 robot_wrist frame."""
    from openpi.models_pytorch import fastwam_pytorch as fw_pt

    torch_imgs = {}
    for key in model_cfg.camera_keys:
        arr = _uint8_hwc_to_model(images_t[key])[None, None, ...]  # B,T,H,W,C
        torch_imgs[key] = torch.from_numpy(arr).to(device=device, dtype=torch.float32)
    video = fw_pt._images_to_video(
        torch_imgs,
        model_cfg.camera_keys,
        model_cfg.concat_multi_camera,
        image_resolution=model_cfg.image_resolution,
    )
    return _video_frame_to_uint8_hwc(video[0, :, 0])


def _normalize_state(
    native_state: np.ndarray,
    *,
    dataset_id: str,
    norm_stats: dict,
    use_quantiles: bool,
) -> np.ndarray:
    """Native 14D -> unified80 -> quantile-normalized."""
    from openpi import transforms as pi_transforms
    from openpi.cotrain import action_space

    spec = action_space.UNIFIED_ACTION_SPECS[dataset_id]
    unified = action_space.map_array(native_state, spec.state_mapping)
    return pi_transforms.Normalize(
        {"state": norm_stats["state"]},
        use_quantiles=use_quantiles,
    )({"state": unified})["state"]


def _infer_window(
    *,
    model,
    model_cfg,
    device: torch.device,
    episode: dict[str, Any],
    anchor: int,
    offsets: list[int],
    prompt: str,
    normalized_state: np.ndarray,
    action_mask: np.ndarray,
    num_inference_steps: int,
    seed: int,
) -> tuple[list, np.ndarray]:
    """Run infer_joint at ``anchor``. Returns (pred_video_frames, pred_action_norm [H,80])."""
    from openpi.models_pytorch import fastwam_pytorch as fw_pt

    t_len = episode["num_frames"]
    torch_imgs = {}
    for key in model_cfg.camera_keys:
        frames = []
        for off in offsets:
            idx = min(anchor + off, t_len - 1)
            frames.append(_uint8_hwc_to_model(episode["images"][key][idx]))
        stacked = np.stack(frames, axis=0)[None, ...]  # B,T,H,W,C
        torch_imgs[key] = torch.from_numpy(stacked).to(device=device, dtype=torch.float32)

    video = fw_pt._images_to_video(
        torch_imgs,
        model_cfg.camera_keys,
        model_cfg.concat_multi_camera,
        image_resolution=model_cfg.image_resolution,
    )
    input_image = video[0, :, 0].to(dtype=torch.bfloat16)
    proprio = torch.as_tensor(
        normalized_state[: model_cfg.proprio_dim],
        device=device,
        dtype=torch.bfloat16,
    )
    mask = torch.as_tensor(action_mask, device=device, dtype=torch.bool)

    out = model.fastwam.infer_joint(
        prompt=prompt,
        input_image=input_image,
        num_video_frames=model_cfg.video_num_frames,
        action_horizon=model_cfg.action_horizon,
        proprio=proprio,
        action_mask=mask,
        num_inference_steps=num_inference_steps,
        seed=seed,
        test_action_with_infer_action=False,
    )
    pred_action = out["action"].detach().cpu().numpy()
    return out["video"], pred_action


def main(args: Args) -> None:
    init_logging()
    _configure_rlds_root(args)

    import openpi.cotrain.config as cotrain_config
    from openpi.cotrain import action_space
    from openpi.cotrain.config import load_per_dataset_norm_stats
    from openpi.cotrain.fastwam_checkpoint import load_fastwam_checkpoint
    from openpi.models.fastwam_config import FastWAMConfig

    if args.dataset not in ("piper2", "piper30"):
        raise ValueError(f"Unsupported dataset={args.dataset!r}; use piper2 or piper30")

    config = cotrain_config.get_config(args.config_name)
    if not isinstance(config.model, FastWAMConfig):
        raise TypeError(f"{args.config_name} is not a FastWAM config")

    model_cfg = config.model
    if args.image_resolution:
        hw = _parse_hw(args.image_resolution)
        model_cfg = dataclasses.replace(model_cfg, image_resolution=hw)
    model_cfg = dataclasses.replace(
        model_cfg,
        skip_dit_load_from_pretrain=True,
        skip_vae_load_from_pretrain=True,
    )

    keep = tuple(ds for ds in config.data.datasets if ds.uid == args.dataset)
    if not keep:
        raise ValueError(f"dataset={args.dataset!r} not in {args.config_name}")
    replace_kw: dict = {
        "model": model_cfg,
        "exp_name": "joint_episode",
        "wandb_enabled": False,
        "data": dataclasses.replace(config.data, datasets=keep),
    }
    if args.assets_base_dir is not None:
        replace_kw["assets_base_dir"] = str(args.assets_base_dir)
    config = dataclasses.replace(config, **replace_kw)

    horizon = int(config.model.action_horizon)
    video_frames = int(config.model.video_num_frames)
    freq_ratio = int(config.model.action_video_freq_ratio)
    offsets = list(range(0, horizon + 1, freq_ratio))
    if len(offsets) != video_frames:
        raise ValueError(
            f"video offsets {offsets} length != video_num_frames={video_frames}"
        )
    stride = int(args.stride) if args.stride is not None else horizon

    builder_dir = _builder_dir_for_dataset(args.dataset)
    logging.info("Selecting episode from %s split=%s", builder_dir, args.split)
    ordinal, meta_row = _select_episode_ordinal(
        builder_dir,
        args.split,
        episode_index=args.episode_index,
        prefer_short=args.prefer_short,
        min_frames=horizon,
    )
    logging.info(
        "Selected episode_index=%s ordinal=%s T=%s task=%r",
        meta_row["episode_index"],
        ordinal,
        meta_row["num_frames"],
        meta_row["task"],
    )

    logging.info("Loading full episode frames...")
    episode = _load_full_episode(builder_dir, args.split, ordinal)
    t_len = episode["num_frames"]
    anchors = _make_anchors(t_len, horizon, stride)
    logging.info("Anchors (%d, stride=%d): %s", len(anchors), stride, anchors)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        logging.warning("CUDA unavailable; joint inference will be very slow on CPU.")

    logging.info(
        "Model: res=%s concat=%s action_horizon=%s video_frames=%s",
        config.model.image_resolution,
        config.model.concat_multi_camera,
        horizon,
        video_frames,
    )
    model = config.model.create_pytorch(device=str(device))
    model.freeze_encoders()
    model.to(device)
    ckpt_path = load_fastwam_checkpoint(model, args.checkpoint, strict=True)
    model.eval()
    logging.info("Loaded checkpoint %s", ckpt_path)

    data_config = config.data.create(config.assets_dirs, config.model)
    output_transforms = _build_output_transforms(config, data_config)
    per_dataset_stats = load_per_dataset_norm_stats(config.assets_dirs, data_config.datasets)
    norm_stats = per_dataset_stats[args.dataset]
    spec = action_space.UNIFIED_ACTION_SPECS[args.dataset]
    action_mask = np.asarray(spec.action_mask, dtype=np.bool_)
    prompt = f"{PROMPT_PREFIX}{episode['prompt']}"

    pred_action_native = np.full((t_len, 14), np.nan, dtype=np.float64)
    pred_video = [None] * t_len  # type: ignore[list-item]
    covered_action = np.zeros(t_len, dtype=bool)

    t0 = time.perf_counter()
    for win_i, anchor in enumerate(anchors):
        seed = args.seed + win_i
        state_t = episode["state"][anchor]
        norm_state = _normalize_state(
            state_t,
            dataset_id=args.dataset,
            norm_stats=norm_stats,
            use_quantiles=data_config.use_quantile_norm,
        )
        logging.info(
            "Window %d/%d anchor=%d seed=%d",
            win_i + 1,
            len(anchors),
            anchor,
            seed,
        )
        pred_frames, pred_action_norm = _infer_window(
            model=model,
            model_cfg=config.model,
            device=device,
            episode=episode,
            anchor=anchor,
            offsets=offsets,
            prompt=prompt,
            normalized_state=norm_state,
            action_mask=action_mask,
            num_inference_steps=args.num_inference_steps,
            seed=seed,
        )
        pred_native = _postprocess_actions(
            pred_action_norm,
            norm_state,
            dataset_id=args.dataset,
            output_transforms=output_transforms,
        )

        next_boundary = anchors[win_i + 1] if win_i + 1 < len(anchors) else t_len
        for i in range(horizon):
            idx = anchor + i
            if idx >= next_boundary or idx >= t_len:
                break
            pred_action_native[idx] = pred_native[i]
            covered_action[idx] = True

        for k, off in enumerate(offsets):
            idx = min(anchor + off, t_len - 1)
            frame = pred_frames[k]
            arr = np.asarray(frame.convert("RGB")) if hasattr(frame, "convert") else np.asarray(frame)
            # Skip re-writing the conditioning frame if already filled.
            if off == 0 and pred_video[idx] is not None:
                continue
            pred_video[idx] = arr

    # Fill any missing pred video frames by holding the last prediction (or GT later).
    logging.info("Composing full GT video (%d frames)...", t_len)
    gt_video = []
    for t in range(t_len):
        gt_video.append(
            _compose_frame(
                {slot: episode["images"][slot][t] for slot in CAMERA_SLOTS},
                model_cfg=config.model,
                device=device,
            )
        )

    last_pred = gt_video[0]
    filled_pred = []
    for t in range(t_len):
        if pred_video[t] is not None:
            last_pred = pred_video[t]
        filled_pred.append(np.asarray(last_pred))

    if not np.all(covered_action):
        missing = np.flatnonzero(~covered_action)
        raise RuntimeError(f"Uncovered action steps after stitch: {missing[:20]}...")

    gt_native = episode["action"].astype(np.float64)
    fps = int(args.video_fps) if args.video_fps is not None else int(episode["fps"])

    out_dir = args.output_dir / f"episode_{episode['episode_index']:05d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_mp4(gt_video, out_dir / "gt_video.mp4", fps=fps)
    _save_mp4(filled_pred, out_dir / "pred_video.mp4", fps=fps)
    traj_info = _save_native_trajectories(
        out_dir,
        dataset_id=args.dataset,
        gt_native=gt_native,
        pred_native=pred_action_native,
    )

    meta = {
        "dataset": args.dataset,
        "split": args.split,
        "episode_index": episode["episode_index"],
        "split_ordinal": ordinal,
        "num_frames": t_len,
        "task": episode["task"],
        "prompt": prompt,
        "fps": fps,
        "action_horizon": horizon,
        "video_num_frames": video_frames,
        "video_frame_offsets": offsets,
        "stride": stride,
        "anchors": anchors,
        "num_windows": len(anchors),
        "image_resolution": list(config.model.image_resolution),
        "concat_multi_camera": config.model.concat_multi_camera,
        "checkpoint": str(ckpt_path),
        "gt_video": "gt_video.mp4",
        "pred_video": "pred_video.mp4",
        "gt_trajectory": f"gt_action_native_{traj_info['native_dim']}d.npy",
        "pred_trajectory": f"pred_action_native_{traj_info['native_dim']}d.npy",
        "trajectory_mae": traj_info["mae"],
        "dim_names": traj_info["dim_names"],
        "elapsed_sec": time.perf_counter() - t0,
        "note": (
            "pred_video is stitched from joint windows at offsets "
            f"{offsets}; in-between frames hold the last predicted frame."
        ),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "output_dir": str(out_dir),
                "episode_index": episode["episode_index"],
                "num_frames": t_len,
                "num_windows": len(anchors),
                "trajectory_mae": traj_info["mae"],
                "elapsed_sec": meta["elapsed_sec"],
                "checkpoint": str(ckpt_path),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    logging.info(
        "Done episode %s -> %s (T=%d windows=%d mae=%.4f %.1fs)",
        episode["episode_index"],
        out_dir,
        t_len,
        len(anchors),
        traj_info["mae"],
        meta["elapsed_sec"],
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
