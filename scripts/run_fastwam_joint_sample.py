#!/usr/bin/env python3
"""Joint FastWAM inference (video gen + action chunk) on cotrain val samples.

Video: ``video_num_frames=9`` composed frames (robot_wrist layout), denoised with
``num_inference_steps`` (default 20). Action: ``action_horizon=32`` steps.

Per-sample tmp outputs (complete sample window only)::

  gt_video.mp4              # full GT compose video for this sample
  pred_video.mp4            # full predicted video
  gt_action_native_14d.npy  # full native GT trajectory [T, Da]
  pred_action_native_14d.npy
  gt_action_native_14d.png  # 14-dim trajectory plot
  pred_action_native_14d.png
  action_native_14d_overlay.png  # GT vs pred overlay (for reading diffs)

Example (piper2 seen val, robot_wrist compose, ckpt step 29999)::

  cd /data/zjyang/Atom-0 && source scripts/atom0_env.sh
  export DIFFSYNTH_MODEL_BASE_PATH=\"$(pwd)/checkpoints/fastwam\"
  PYTHONPATH=src .venv/bin/python scripts/run_fastwam_joint_sample.py \\
    --checkpoint checkpoints/wam-cross-piper/fw-wam-cross-piper-v1-8gpu-b208-30k/29999 \\
    --dataset piper2 --val-split seen --num-samples 1 \\
    --image-resolution 576,512 \\
    --output-dir tmp/fastwam_joint/piper2-v1-29999-seen-full
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro


def init_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


@dataclasses.dataclass
class Args:
    checkpoint: Path
    """Step dir, exp dir, or ``model.safetensors``."""

    config_name: str = "wam-cross-piper"
    dataset: str = "piper2"
    val_split: str = "seen"
    """Val label key: ``seen`` or ``unseen`` (maps to seen_test / unseen_test)."""

    num_samples: int = 1
    image_resolution: str | None = "576,512"
    """Compose H,W for ``robot_wrist``. Default 576×512 layout; wam-cross train config uses 288×256 — match your ckpt."""

    output_dir: Path = Path("tmp/fastwam_joint")
    device: str = "cuda:0"
    num_inference_steps: int = 20
    seed: int = 42
    val_batch_size: int = 4
    rlds_data_dir: Path | None = None
    assets_base_dir: Path | None = None


def _parse_hw(text: str) -> tuple[int, int]:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 2:
        raise ValueError(f"--image-resolution must be H,W got {text!r}")
    return int(parts[0]), int(parts[1])


def _configure_rlds_root(args: Args) -> None:
    if args.rlds_data_dir is not None:
        os.environ["RLDS_DATA_DIR"] = str(args.rlds_data_dir)
    elif "RLDS_DATA_DIR" not in os.environ:
        os.environ["RLDS_DATA_DIR"] = "/mnt/bos/bo23lu"


def _tensor_to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _native_dim_names(dataset_id: str) -> list[str] | None:
    if dataset_id in ("piper2", "piper30"):
        return (
            [f"left_joint_{i}" for i in range(1, 7)]
            + ["left_gripper"]
            + [f"right_joint_{i}" for i in range(1, 7)]
            + ["right_gripper"]
        )
    return None


def _array_json_payload(arr: np.ndarray, *, dim_names: list[str] | None = None) -> dict[str, Any]:
    arr = np.asarray(arr, dtype=np.float64)
    payload: dict[str, Any] = {
        "shape": list(arr.shape),
        "values": arr.tolist(),
    }
    if dim_names is not None and arr.ndim >= 1 and arr.shape[-1] == len(dim_names):
        payload["dim_names"] = dim_names
        if arr.ndim == 1:
            payload["by_dim"] = {name: float(v) for name, v in zip(dim_names, arr, strict=True)}
        elif arr.ndim == 2:
            payload["steps"] = [
                {"t": t, "by_dim": {name: float(v) for name, v in zip(dim_names, row, strict=True)}}
                for t, row in enumerate(arr)
            ]
    return payload


def _save_array_json(path: Path, arr: np.ndarray, *, dim_names: list[str] | None = None) -> None:
    path.write_text(
        json.dumps(_array_json_payload(arr, dim_names=dim_names), indent=2) + "\n",
        encoding="utf-8",
    )


def _frame_to_uint8_hwc(frame) -> np.ndarray:
    """PIL / HWC ndarray / CHW tensor -> HWC uint8."""
    if hasattr(frame, "convert"):
        return np.asarray(frame.convert("RGB"))
    if isinstance(frame, torch.Tensor):
        return _video_frame_to_uint8_hwc(frame)
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))
    return arr


def _video_frame_to_uint8_hwc(frame_chw: torch.Tensor) -> np.ndarray:
    """CHW float (typically [-1,1]) -> HWC uint8."""
    x = frame_chw.detach().float().cpu()
    if x.min() >= 0.0 and x.max() <= 1.0:
        arr = (x.clamp(0, 1) * 255.0).byte()
    else:
        arr = (((x.clamp(-1, 1) + 1.0) * 127.5).round()).byte()
    return arr.permute(1, 2, 0).numpy()


def _save_mp4(frames: list, path: Path, fps: int = 8) -> None:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise ValueError("No frames to save")
    if hasattr(frames[0], "convert"):  # PIL
        arrays = [np.asarray(f.convert("RGB")) for f in frames]
    else:
        arrays = [np.asarray(f) for f in frames]
    imageio.mimwrite(str(path), arrays, fps=fps)


def _save_sample_videos(
    gt_frames: list,
    pred_frames: list,
    sample_dir: Path,
    *,
    fps: int = 8,
) -> dict[str, Any]:
    """Write complete GT / pred videos for the sample window (no compare grid)."""
    gt = [_frame_to_uint8_hwc(f) for f in gt_frames]
    pred = [_frame_to_uint8_hwc(f) for f in pred_frames]
    if not gt or not pred:
        raise ValueError("Empty GT/pred video frames")
    n = min(len(gt), len(pred))
    gt, pred = gt[:n], pred[:n]
    h, w = gt[0].shape[:2]
    _save_mp4(gt, sample_dir / "gt_video.mp4", fps=fps)
    _save_mp4(pred, sample_dir / "pred_video.mp4", fps=fps)
    return {"num_frames": n, "frame_size_hw": [h, w], "fps": fps}


def _plot_native_trajectory(
    traj: np.ndarray,
    *,
    dim_names: list[str] | None,
    title: str,
    color: str,
    path: Path,
) -> None:
    """Save one complete native trajectory figure [T, Da]."""
    import math

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    traj = np.asarray(traj, dtype=np.float64)
    if traj.ndim != 2:
        raise ValueError(f"traj must be [T, D], got {traj.shape}")
    t_len, ad = traj.shape
    names = dim_names if dim_names is not None and len(dim_names) == ad else [f"dim_{i}" for i in range(ad)]
    ncols = 4
    nrows = math.ceil(ad / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.2 * nrows), squeeze=False)
    x = np.arange(t_len)
    for dim in range(ad):
        ax = axes[dim // ncols][dim % ncols]
        ax.plot(x, traj[:, dim], color=color, lw=1.6)
        ax.set_title(names[dim], fontsize=9)
        ax.set_xlabel("t", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25)
    for k in range(ad, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    fig.suptitle(f"{title}  shape=[{t_len},{ad}]", fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_native_overlay(
    gt: np.ndarray,
    pred: np.ndarray,
    *,
    dim_names: list[str] | None,
    title: str,
    path: Path,
) -> None:
    """GT (solid) vs pred (dashed) for all native dims."""
    import math

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gt = np.asarray(gt, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    t_len = min(gt.shape[0], pred.shape[0])
    ad = min(gt.shape[1], pred.shape[1])
    gt, pred = gt[:t_len, :ad], pred[:t_len, :ad]
    names = dim_names if dim_names is not None and len(dim_names) == ad else [f"dim_{i}" for i in range(ad)]
    ncols = 4
    nrows = math.ceil(ad / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.2 * nrows), squeeze=False)
    x = np.arange(t_len)
    for dim in range(ad):
        ax = axes[dim // ncols][dim % ncols]
        ax.plot(x, gt[:, dim], color="tab:green", lw=1.5, label="gt")
        ax.plot(x, pred[:, dim], color="tab:red", ls="--", lw=1.5, label="pred")
        ax.set_title(names[dim], fontsize=9)
        ax.set_xlabel("t", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25)
    for k in range(ad, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    axes[0][0].legend(fontsize=8)
    fig.suptitle(f"{title}  shape=[{t_len},{ad}]", fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_native_trajectories(
    sample_dir: Path,
    *,
    dataset_id: str,
    gt_native: np.ndarray,
    pred_native: np.ndarray,
) -> dict[str, Any]:
    """Complete native 14D (or Da) trajectories: npy + plots + compact json."""
    dim_names = _native_dim_names(dataset_id)
    gt_native = np.asarray(gt_native, dtype=np.float64)
    pred_native = np.asarray(pred_native, dtype=np.float64)
    da = gt_native.shape[-1]
    stem = f"action_native_{da}d"

    np.save(sample_dir / f"gt_{stem}.npy", gt_native)
    np.save(sample_dir / f"pred_{stem}.npy", pred_native)
    _save_array_json(sample_dir / f"gt_{stem}.json", gt_native, dim_names=dim_names)
    _save_array_json(sample_dir / f"pred_{stem}.json", pred_native, dim_names=dim_names)

    _plot_native_trajectory(
        gt_native,
        dim_names=dim_names,
        title="GT native trajectory",
        color="tab:green",
        path=sample_dir / f"gt_{stem}.png",
    )
    _plot_native_trajectory(
        pred_native,
        dim_names=dim_names,
        title="Pred native trajectory",
        color="tab:red",
        path=sample_dir / f"pred_{stem}.png",
    )
    _plot_native_overlay(
        gt_native,
        pred_native,
        dim_names=dim_names,
        title="GT vs Pred native trajectory",
        path=sample_dir / f"{stem}_overlay.png",
    )
    mae = float(np.mean(np.abs(gt_native - pred_native)))
    return {
        "native_dim": da,
        "horizon": int(gt_native.shape[0]),
        "mae": mae,
        "dim_names": dim_names,
    }


def _build_output_transforms(train_config, data_config):
    from openpi.cotrain.config import load_per_dataset_norm_stats
    from openpi.cotrain import transforms as cotrain_transforms

    datasets = data_config.datasets
    per_dataset_stats = load_per_dataset_norm_stats(train_config.assets_dirs, datasets)
    delta_masks = {ds.uid: ds.unified_action_spec.delta_mask for ds in datasets}
    specs = {ds.uid: ds.unified_action_spec for ds in datasets}
    native_dims = {ds.uid: ds.action_dim for ds in datasets if ds.action_dim > 0}
    return [
        cotrain_transforms.DispatchUnnormalize(
            norm_stats_by_dataset=per_dataset_stats,
            use_quantiles=data_config.use_quantile_norm,
        ),
        cotrain_transforms.DispatchAbsoluteActions(masks_by_dataset=delta_masks),
        cotrain_transforms.DispatchStandardizedOutputs(
            specs_by_dataset=specs,
            native_action_dims_by_dataset=native_dims,
        ),
    ]


def _postprocess_actions(
    actions: np.ndarray,
    state: np.ndarray,
    *,
    dataset_id: str,
    output_transforms: list,
) -> np.ndarray:
    """Unified normalized deltas [T,80] -> native absolute [T, Da]."""
    data: dict[str, Any] = {
        "actions": actions,
        "state": state,
        "_cotrain_dataset_id": dataset_id,
    }
    for transform in output_transforms:
        data = transform(data)
    return np.asarray(data["actions"])


def _prepare_infer_inputs(
    observation,
    *,
    model_cfg,
    device: torch.device,
    index: int,
):
    from openpi.models_pytorch import fastwam_pytorch as fw_pt

    images = {}
    for key, img in observation.images.items():
        t = img[index : index + 1]
        if t.ndim == 4:
            t = t.unsqueeze(1)
        images[key] = t.to(device=device, dtype=torch.float32)

    video = fw_pt._images_to_video(
        images,
        model_cfg.camera_keys,
        model_cfg.concat_multi_camera,
        image_resolution=model_cfg.image_resolution,
    )
    input_image = video[0, :, 0].to(dtype=torch.bfloat16)
    gt_video_frames = [_video_frame_to_uint8_hwc(video[0, :, t]) for t in range(video.shape[2])]

    state = observation.state[index].to(device=device, dtype=torch.bfloat16)
    if state.ndim == 1:
        proprio = state[: model_cfg.proprio_dim]
    else:
        proprio = state[-1, : model_cfg.proprio_dim]

    prompts = getattr(observation, "_fastwam_prompts", None)
    prompt = prompts[index] if prompts is not None else ""

    context = context_mask = None
    if observation.context is not None and observation.context_mask is not None:
        context = observation.context[index : index + 1].to(device=device, dtype=torch.bfloat16)
        context_mask = observation.context_mask[index : index + 1].to(device=device, dtype=torch.bool)

    return {
        "input_image": input_image,
        "proprio": proprio,
        "prompt": prompt,
        "context": context,
        "context_mask": context_mask,
        "input_video_frame": video[0, :, 0],
        "gt_video_frames": gt_video_frames,
    }


def main(args: Args) -> None:
    init_logging()
    _configure_rlds_root(args)

    import openpi.cotrain.config as cotrain_config
    import openpi.cotrain.data_loader as cotrain_data_loader
    from openpi.cotrain.fastwam_checkpoint import load_fastwam_checkpoint
    from openpi.models.fastwam_config import FastWAMConfig

    config = cotrain_config.get_config(args.config_name)
    if not isinstance(config.model, FastWAMConfig):
        raise TypeError(f"{args.config_name} is not a FastWAM config")

    model_cfg = config.model
    if args.image_resolution:
        hw = _parse_hw(args.image_resolution)
        model_cfg = dataclasses.replace(model_cfg, image_resolution=hw)
        logging.info("Using image_resolution=%s (robot_wrist compose)", hw)
    else:
        logging.info("Using config image_resolution=%s", model_cfg.image_resolution)

    model_cfg = dataclasses.replace(
        model_cfg,
        skip_dit_load_from_pretrain=True,
        skip_vae_load_from_pretrain=True,
    )

    keep = tuple(ds for ds in config.data.datasets if ds.uid == args.dataset)
    if not keep:
        raise ValueError(
            f"dataset={args.dataset!r} not in {args.config_name}; "
            f"have {[d.uid for d in config.data.datasets]}"
        )
    replace_kw: dict = {
        "model": model_cfg,
        "exp_name": "joint_sample",
        "wandb_enabled": False,
        "eval_interval": 0,
        "num_val_batches": max(1, (args.num_samples + args.val_batch_size - 1) // args.val_batch_size),
        "val_batch_size": args.val_batch_size,
        "data": dataclasses.replace(config.data, datasets=keep),
    }
    if args.assets_base_dir is not None:
        replace_kw["assets_base_dir"] = str(args.assets_base_dir)
    config = dataclasses.replace(config, **replace_kw)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        logging.warning("CUDA unavailable; joint inference will be very slow on CPU.")

    logging.info(
        "Model: res=%s concat=%s action_horizon=%s video_frames=%s",
        config.model.image_resolution,
        config.model.concat_multi_camera,
        config.model.action_horizon,
        config.model.video_num_frames,
    )
    model = config.model.create_pytorch(device=str(device))
    model.freeze_encoders()
    model.to(device)
    ckpt_path = load_fastwam_checkpoint(model, args.checkpoint, strict=True)
    model.eval()
    logging.info("Loaded checkpoint %s", ckpt_path)

    data_config = config.data.create(config.assets_dirs, config.model)
    output_transforms = _build_output_transforms(config, data_config)

    val_loaders = cotrain_data_loader.build_val_loaders(
        config,
        framework="pytorch",
        single_process=True,
    )
    if args.val_split not in val_loaders or args.dataset not in val_loaders[args.val_split]:
        available = {k: sorted(v.keys()) for k, v in val_loaders.items()}
        raise ValueError(
            f"No val loader for split={args.val_split!r} dataset={args.dataset!r}. Available: {available}"
        )
    loader = val_loaders[args.val_split][args.dataset]
    logging.info(
        "Val loader: dataset=%s split=%s batch_size=%s",
        args.dataset,
        args.val_split,
        cotrain_data_loader.resolve_val_batch_size(config),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples_saved = 0
    t0 = time.perf_counter()

    for batch_idx, (observation, actions) in enumerate(loader):
        batch_size = _tensor_to_numpy(observation.state).shape[0]
        for i in range(batch_size):
            if samples_saved >= args.num_samples:
                break

            sample_seed = args.seed + samples_saved
            infer_inputs = _prepare_infer_inputs(
                observation,
                model_cfg=config.model,
                device=device,
                index=i,
            )
            prompt = infer_inputs["prompt"]
            action_mask_i = None
            if observation.action_mask is not None:
                action_mask_i = observation.action_mask[i].to(device=device)

            logging.info(
                "Sample %s/%s batch=%s idx=%s prompt=%r",
                samples_saved + 1,
                args.num_samples,
                batch_idx,
                i,
                prompt[:120],
            )

            out = model.fastwam.infer_joint(
                prompt=prompt if infer_inputs["context"] is None else None,
                input_image=infer_inputs["input_image"],
                num_video_frames=config.model.video_num_frames,
                action_horizon=config.model.action_horizon,
                proprio=infer_inputs["proprio"],
                context=infer_inputs["context"],
                context_mask=infer_inputs["context_mask"],
                action_mask=action_mask_i,
                num_inference_steps=args.num_inference_steps,
                seed=sample_seed,
                test_action_with_infer_action=False,
            )

            pred_action = _tensor_to_numpy(out["action"])
            gt_action = _tensor_to_numpy(actions[i])
            state_np = _tensor_to_numpy(observation.state[i])

            pred_native = _postprocess_actions(
                pred_action,
                state_np,
                dataset_id=args.dataset,
                output_transforms=output_transforms,
            )
            gt_native = _postprocess_actions(
                gt_action,
                state_np,
                dataset_id=args.dataset,
                output_transforms=output_transforms,
            )

            sample_dir = args.output_dir / f"sample_{samples_saved:02d}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            video_info = _save_sample_videos(
                infer_inputs["gt_video_frames"],
                out["video"],
                sample_dir,
            )
            traj_info = _save_native_trajectories(
                sample_dir,
                dataset_id=args.dataset,
                gt_native=gt_native,
                pred_native=pred_native,
            )

            meta = {
                "dataset": args.dataset,
                "val_split": args.val_split,
                "batch_idx": batch_idx,
                "batch_index": i,
                "prompt": prompt,
                "seed": sample_seed,
                "num_inference_steps": args.num_inference_steps,
                "image_resolution": list(config.model.image_resolution),
                "concat_multi_camera": config.model.concat_multi_camera,
                "action_horizon": config.model.action_horizon,
                "video_num_frames": config.model.video_num_frames,
                "gt_video": "gt_video.mp4",
                "pred_video": "pred_video.mp4",
                "video_num_frames_saved": video_info["num_frames"],
                "video_frame_size_hw": video_info["frame_size_hw"],
                "video_fps": video_info["fps"],
                "gt_trajectory": f"gt_action_native_{traj_info['native_dim']}d.npy",
                "pred_trajectory": f"pred_action_native_{traj_info['native_dim']}d.npy",
                "native_dim": traj_info["native_dim"],
                "trajectory_horizon": traj_info["horizon"],
                "trajectory_mae": traj_info["mae"],
                "dim_names": traj_info["dim_names"],
                "pred_action_unified_shape": list(pred_action.shape),
                "state_unified_shape": list(state_np.shape),
                "checkpoint": str(ckpt_path),
            }
            (sample_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
            logging.info(
                "Saved %s (video T=%s @%sfps | native traj T=%s D=%s mae=%.4f)",
                sample_dir,
                video_info["num_frames"],
                video_info["fps"],
                traj_info["horizon"],
                traj_info["native_dim"],
                traj_info["mae"],
            )
            samples_saved += 1

        if samples_saved >= args.num_samples:
            break

    summary = {
        "config_name": args.config_name,
        "checkpoint": str(ckpt_path),
        "dataset": args.dataset,
        "val_split": args.val_split,
        "num_samples": samples_saved,
        "output_dir": str(args.output_dir),
        "elapsed_sec": time.perf_counter() - t0,
    }
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    logging.info("Done: %s samples -> %s (%.1fs)", samples_saved, args.output_dir, summary["elapsed_sec"])


if __name__ == "__main__":
    main(tyro.cli(Args))
