"""Debug helpers for FastWAM VAE input video (composed ``image_resolution`` for robot_wrist).

Standalone tooling only — training does not import this module unless explicitly enabled
via ``scripts/debug_fastwam_vae_input.py``.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

import openpi.models.fastwam_config as fastwam_config
import openpi.models.model as _model
from openpi.models_pytorch import fastwam_pytorch


@dataclass(frozen=True)
class VaeInputDebugConfig:
    output_dir: Path
    num_batches: int = 8
    samples_per_batch: int = 2
    skip_batches: int = 4
    seed: int = 0
    frame_indices: tuple[int, ...] | None = None  # None -> first, middle, last


def _decode_prompt(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        return str(value)
    arr = np.asarray(value)
    if arr.ndim == 0:
        return str(arr.item())
    if arr.size == 1:
        return str(arr.reshape(-1)[0])
    return None


def _decode_dataset_id(batch: dict[str, Any]) -> str | None:
    for key in ("_cotrain_dataset_id", "dataset_id"):
        if key not in batch:
            continue
        raw = batch[key]
        if isinstance(raw, np.ndarray):
            flat = raw.reshape(-1)
            if flat.size == 0:
                continue
            return str(flat[0])
        return str(raw)
    return None


def compose_vae_video_from_observation(
    observation: _model.Observation,
    model_config: fastwam_config.FastWAMConfig,
) -> torch.Tensor:
    """Return ``(B, 3, T, H, W)`` video tensor exactly as ``FastWAMPytorch`` builds for VAE."""
    images = {
        key: torch.as_tensor(np.asarray(value), dtype=torch.float32)
        for key, value in observation.images.items()
    }
    return fastwam_pytorch._images_to_video(
        images,
        model_config.camera_keys,
        model_config.concat_multi_camera,
        image_resolution=model_config.image_resolution,
    )


def video_frame_to_uint8_hwc(frame_chw: torch.Tensor) -> np.ndarray:
    """Map a single ``(3, H, W)`` frame in ``[-1, 1]`` to ``uint8 HWC``."""
    if frame_chw.ndim != 3 or frame_chw.shape[0] != 3:
        raise ValueError(f"Expected frame (3, H, W), got {tuple(frame_chw.shape)}")
    x = frame_chw.detach().float().cpu().clamp(-1.0, 1.0)
    x = ((x + 1.0) * 127.5).round().to(torch.uint8)
    return x.permute(1, 2, 0).numpy()


def _pick_frame_indices(num_frames: int, requested: tuple[int, ...] | None) -> tuple[int, ...]:
    if num_frames <= 0:
        return ()
    if requested is not None:
        return tuple(idx for idx in requested if 0 <= idx < num_frames)
    if num_frames == 1:
        return (0,)
    mid = num_frames // 2
    return tuple(dict.fromkeys((0, mid, num_frames - 1)))


def save_vae_input_sample(
    video_b3thw: torch.Tensor,
    batch_idx: int,
    sample_idx: int,
    output_dir: Path,
    *,
    frame_indices: tuple[int, ...] | None = None,
    meta: dict[str, Any] | None = None,
) -> list[Path]:
    """Save selected temporal frames for one batch element; returns written PNG paths."""
    if video_b3thw.ndim != 5:
        raise ValueError(f"Expected video (B, 3, T, H, W), got {tuple(video_b3thw.shape)}")
    if sample_idx >= video_b3thw.shape[0]:
        raise IndexError(f"sample_idx={sample_idx} out of range for batch size {video_b3thw.shape[0]}")

    output_dir.mkdir(parents=True, exist_ok=True)
    num_frames = int(video_b3thw.shape[2])
    height = int(video_b3thw.shape[3])
    width = int(video_b3thw.shape[4])
    picked = _pick_frame_indices(num_frames, frame_indices)

    written: list[Path] = []
    prefix = f"batch{batch_idx:03d}_sample{sample_idx:02d}"
    for frame_idx in picked:
        png_path = output_dir / f"{prefix}_t{frame_idx:02d}_{height}x{width}.png"
        arr = video_frame_to_uint8_hwc(video_b3thw[sample_idx, :, frame_idx])
        Image.fromarray(arr).save(png_path)
        written.append(png_path)

    meta_path = output_dir / f"{prefix}_meta.json"
    payload = {
        "batch_idx": batch_idx,
        "sample_idx": sample_idx,
        "video_shape": list(video_b3thw.shape),
        "saved_frames": list(picked),
        **(meta or {}),
    }
    meta_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    written.append(meta_path)
    return written


def save_random_vae_inputs_from_batches(
    batches: list[tuple[dict[str, Any], _model.Observation, torch.Tensor | np.ndarray]],
    model_config: fastwam_config.FastWAMConfig,
    debug_cfg: VaeInputDebugConfig,
) -> dict[str, Any]:
    """Compose and persist VAE-bound video frames from pre-collected training batches."""
    rng = random.Random(debug_cfg.seed)
    output_dir = debug_cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    index: list[dict[str, Any]] = []
    total_png = 0

    for batch_idx, (raw_batch, observation, _actions) in enumerate(batches):
        video = compose_vae_video_from_observation(observation, model_config)
        batch_size = int(video.shape[0])
        k = min(debug_cfg.samples_per_batch, batch_size)
        sample_indices = sorted(rng.sample(range(batch_size), k=k))

        prompts = raw_batch.get("prompt")
        is_ego = raw_batch.get("is_ego")
        dataset_id = _decode_dataset_id(raw_batch)

        for sample_idx in sample_indices:
            prompt = None
            if prompts is not None:
                if isinstance(prompts, np.ndarray) and prompts.ndim >= 1:
                    prompt = _decode_prompt(prompts[sample_idx])
                else:
                    prompt = _decode_prompt(prompts)

            ego_flag = None
            if is_ego is not None:
                ego_arr = np.asarray(is_ego).reshape(-1)
                if sample_idx < ego_arr.size:
                    ego_flag = bool(ego_arr[sample_idx])

            meta = {
                "dataset_id": dataset_id,
                "prompt": prompt,
                "is_ego": ego_flag,
                "concat_multi_camera": model_config.concat_multi_camera,
                "camera_keys": list(model_config.camera_keys),
            }
            paths = save_vae_input_sample(
                video,
                batch_idx,
                sample_idx,
                output_dir,
                frame_indices=debug_cfg.frame_indices,
                meta=meta,
            )
            png_paths = [str(p) for p in paths if p.suffix == ".png"]
            total_png += len(png_paths)
            index.append(
                {
                    "batch_idx": batch_idx,
                    "sample_idx": sample_idx,
                    "png_paths": png_paths,
                    **meta,
                }
            )

    summary = {
        "output_dir": str(output_dir),
        "num_batches_saved": len(batches),
        "num_png": total_png,
        "samples": index,
        "model": {
            "concat_multi_camera": model_config.concat_multi_camera,
            "camera_keys": list(model_config.camera_keys),
            "video_num_frames": model_config.video_num_frames,
            "image_resolution": list(model_config.image_resolution),
        },
    }
    (output_dir / "index.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary
