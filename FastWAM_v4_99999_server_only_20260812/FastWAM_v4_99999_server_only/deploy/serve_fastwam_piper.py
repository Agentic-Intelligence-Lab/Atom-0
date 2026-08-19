#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


PIPER_DIM = 14
PIPER_STATE_SLOTS = (0, 1, 2, 3, 4, 5, 16, 29, 30, 31, 32, 33, 34, 45)
ROBOTWIN_IMAGE_RESOLUTION = (288, 256)
ROBOTWIN_POLICY_CAMERA_KEYS = ("base_0_rgb",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a FastWAM cotrain checkpoint for Piper.")
    parser.add_argument("--openpi-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--norm-stats-path", type=Path, required=True)
    parser.add_argument("--config-name", default="fastwam_cotrain_real_robot_ego_fix")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--actions-per-inference", type=int, default=8)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rtc-enabled", action="store_true")
    parser.add_argument("--rtc-delay-steps", type=int, default=3)
    parser.add_argument("--rtc-execution-horizon", type=int, default=8)
    parser.add_argument("--rtc-soft-mask-decay", type=float, default=0.6)
    parser.add_argument("--rtc-guidance-scale", type=float, default=1.0)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.rtc_enabled:
        parser.error("RTC is not supported by the B200 wam-cross-piper v4 runtime.")
    return args


def add_openpi_paths(root: Path) -> None:
    for path in (root / "src", root / "packages" / "openpi-client" / "src"):
        sys.path.insert(0, str(path))


def observation_value(obs: dict[str, Any], flat_key: str, nested_key: str) -> Any:
    if flat_key in obs:
        return obs[flat_key]
    images = obs.get("images")
    if isinstance(images, dict) and nested_key in images:
        return images[nested_key]
    raise KeyError(f"Missing image {nested_key!r}; expected {flat_key!r} or images/{nested_key}")


def compose_robotwin_image(head: Any, left_wrist: Any, right_wrist: Any) -> np.ndarray:
    """Compose head over left|right wrists into one uint8 288x256 RGB image."""
    import torch

    from openpi.models_pytorch.fastwam_pytorch import _compose_robot_wrist_video

    frames = []
    for name, image in (("head", head), ("left_wrist", left_wrist), ("right_wrist", right_wrist)):
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError(f"Expected {name} RGB image as HWC, got {array.shape}")
        if array.dtype != np.uint8:
            raise ValueError(f"Expected {name} RGB image dtype uint8, got {array.dtype}")
        tensor = torch.from_numpy(np.ascontiguousarray(array))
        frames.append(tensor.permute(2, 0, 1).unsqueeze(0).unsqueeze(2))

    with torch.no_grad():
        composed = _compose_robot_wrist_video(
            frames,
            image_resolution=ROBOTWIN_IMAGE_RESOLUTION,
        )
    image = composed[0, :, 0].permute(1, 2, 0).contiguous().cpu().numpy()
    if image.shape != (*ROBOTWIN_IMAGE_RESOLUTION, 3):
        raise RuntimeError(f"RobotWin image shape mismatch: {image.shape}")
    return image.astype(np.uint8, copy=False)


@dataclasses.dataclass(frozen=True)
class RtcConfig:
    enabled: bool
    delay_steps: int
    execution_horizon: int
    soft_mask_decay: float
    guidance_scale: float


class PiperFastWAMPolicy:
    def __init__(
        self,
        policy: Any,
        *,
        prompt: str,
        actions_per_inference: int,
        rtc_config: RtcConfig,
        seed: int,
    ) -> None:
        self._policy = policy
        self._prompt = prompt
        self._actions_per_inference = actions_per_inference
        self._rtc_config = rtc_config
        self._seed = seed
        horizon = int(policy._model.action_horizon)
        if not 1 <= actions_per_inference <= horizon:
            raise ValueError(f"actions_per_inference must be in [1, {horizon}]")
        model_config = getattr(policy._model, "config", None)
        if model_config is not None:
            camera_keys = tuple(getattr(model_config, "camera_keys", ()))
            concat_mode = getattr(model_config, "concat_multi_camera", None)
            if camera_keys != ROBOTWIN_POLICY_CAMERA_KEYS or concat_mode != "none":
                raise ValueError(
                    "Server-composed RobotWin input requires camera_keys="
                    f"{ROBOTWIN_POLICY_CAMERA_KEYS} and concat_multi_camera='none'; "
                    f"got camera_keys={camera_keys}, concat_multi_camera={concat_mode!r}"
                )

    @property
    def metadata(self) -> dict[str, Any]:
        metadata = dict(self._policy.metadata)
        metadata.update(
            {
                "rtc_supported": False,
                "rtc_enabled": self._rtc_config.enabled,
                "rtc_mode": "fastwam_flow_inpainting" if self._rtc_config.enabled else "off",
                "inference_seed": self._seed,
                "image_layout": "robotwin_head_over_left_right_wrists",
                "image_resolution": list(ROBOTWIN_IMAGE_RESOLUTION),
                "server_composed_image": True,
            }
        )
        return metadata

    @staticmethod
    def _read_state(obs: dict[str, Any]) -> np.ndarray:
        for key in ("active_state", "observation.state", "state"):
            if key in obs:
                state = np.asarray(obs[key], dtype=np.float32)
                break
        else:
            raise KeyError("Missing Piper state: active_state, observation.state, or state")
        if state.shape != (PIPER_DIM,):
            raise ValueError(f"Expected Piper state shape (14,), got {state.shape}")
        return state

    def _standardize(self, obs: dict[str, Any]) -> dict[str, Any]:
        native_state = self._read_state(obs)
        state = np.zeros(int(self._policy._model.action_dim), dtype=np.float32)
        state[np.asarray(PIPER_STATE_SLOTS)] = native_state
        prompt = str(obs.get("prompt") or self._prompt)
        action_mask = np.zeros(int(self._policy._model.action_dim), dtype=bool)
        action_mask[np.asarray(PIPER_STATE_SLOTS)] = True
        composed_image = compose_robotwin_image(
            observation_value(obs, "observation.images.head", "head"),
            observation_value(obs, "observation.images.left_wrist", "left_wrist"),
            observation_value(obs, "observation.images.right_wrist", "right_wrist"),
        )
        return {
            "state": state,
            "action_mask": action_mask,
            "image": {
                "base_0_rgb": composed_image,
                "left_wrist_0_rgb": np.zeros((1, 1, 3), dtype=np.uint8),
                "right_wrist_0_rgb": np.zeros((1, 1, 3), dtype=np.uint8),
            },
            "image_mask": {
                "base_0_rgb": np.asarray(True),
                "left_wrist_0_rgb": np.asarray(False),
                "right_wrist_0_rgb": np.asarray(False),
            },
            "prompt": prompt,
            "dataset_id": "piper30",
        }

    def _absolute_piper_to_model_actions(
        self,
        standardized: dict[str, Any],
        absolute_actions: np.ndarray,
    ) -> np.ndarray:
        absolute = np.asarray(absolute_actions, dtype=np.float32)
        if absolute.ndim != 2 or absolute.shape[1] != PIPER_DIM:
            raise ValueError(f"RTC target actions must have shape (H, 14), got {absolute.shape}")
        native = np.zeros(
            (absolute.shape[0], int(self._policy._model.action_dim)),
            dtype=np.float32,
        )
        native[:, np.asarray(PIPER_STATE_SLOTS)] = absolute
        with_actions = dict(standardized)
        with_actions["actions"] = native
        transformed = self._policy._input_transform(with_actions)
        return np.asarray(transformed["actions"], dtype=np.float32)

    def _build_rtc_target(
        self,
        standardized: dict[str, Any],
        state: np.ndarray,
        rtc_context: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        previous = np.asarray(rtc_context.get("previous_full_actions"), dtype=np.float32)
        if previous.ndim != 2 or previous.shape[1] != PIPER_DIM:
            raise ValueError(f"RTC previous_full_actions must have shape (H, 14), got {previous.shape}")

        horizon = int(self._policy._model.action_horizon)
        action_dim = int(self._policy._model.action_dim)
        start_index = max(0, int(rtc_context.get("previous_action_start_index", 0)))
        switch_steps = max(1, int(rtc_context.get("switch_steps", 1)))
        delay_steps = max(
            switch_steps,
            int(rtc_context.get("delay_steps", self._rtc_config.delay_steps)),
        )
        execution_horizon = max(
            0,
            int(rtc_context.get("execution_horizon", self._rtc_config.execution_horizon)),
        )
        soft_decay = float(
            np.clip(
                rtc_context.get("soft_mask_decay", self._rtc_config.soft_mask_decay),
                0.0,
                1.0,
            )
        )
        guidance = float(
            np.clip(
                rtc_context.get("guidance_scale", self._rtc_config.guidance_scale),
                0.0,
                1.0,
            )
        )

        target_absolute = np.repeat(state[None, :], horizon, axis=0).astype(np.float32)
        step_weights = np.zeros(horizon, dtype=np.float32)
        valid_steps = 0
        for step in range(horizon):
            if execution_horizon > 0 and step >= execution_horizon:
                continue
            previous_index = start_index + step
            if previous_index >= previous.shape[0]:
                continue
            target_absolute[step] = previous[previous_index]
            valid_steps += 1
            if step < delay_steps:
                step_weights[step] = 1.0
            else:
                step_weights[step] = soft_decay ** (step - delay_steps + 1)

        target_model = self._absolute_piper_to_model_actions(standardized, target_absolute)
        if target_model.shape != (horizon, action_dim):
            raise ValueError(
                f"RTC transformed target shape {target_model.shape} does not match "
                f"model shape {(horizon, action_dim)}"
            )
        mask = np.zeros_like(target_model, dtype=np.float32)
        mask[:, np.asarray(PIPER_STATE_SLOTS)] = step_weights[:, None] * guidance
        hard_steps = int(np.count_nonzero(step_weights >= 1.0 - 1e-6))
        soft_steps = int(np.count_nonzero((step_weights > 0.0) & (step_weights < 1.0 - 1e-6)))
        return target_model, mask, {
            "rtc_enabled": True,
            "rtc_mode": "fastwam_flow_inpainting",
            "rtc_skip_steps": switch_steps,
            "rtc_delay_steps": delay_steps,
            "rtc_switch_steps": switch_steps,
            "rtc_valid_steps": valid_steps,
            "rtc_execution_horizon": execution_horizon,
            "rtc_prior_horizon": int(previous.shape[0]),
            "rtc_hard_steps": hard_steps,
            "rtc_soft_steps": soft_steps,
            "rtc_free_steps": int(horizon - hard_steps - soft_steps),
            "rtc_step_weights": step_weights.tolist(),
            "rtc_mask_max": float(mask.max(initial=0.0)),
            "rtc_mask_sum": float(mask.sum()),
        }

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        requested = int(obs.get("actions_per_inference", self._actions_per_inference))
        horizon = int(self._policy._model.action_horizon)
        if not 1 <= requested <= horizon:
            raise ValueError(f"actions_per_inference must be in [1, {horizon}], got {requested}")
        state = self._read_state(obs)
        standardized = self._standardize(obs)
        rtc_context = obs.get("rtc_context")
        if self._rtc_config.enabled and isinstance(rtc_context, dict):
            target_actions, target_mask, rtc_info = self._build_rtc_target(
                standardized,
                state,
                rtc_context,
            )
            result = dict(
                self._policy.infer(
                    standardized,
                    sample_kwargs_override={
                        "rtc_target_action": target_actions,
                        "rtc_target_mask": target_mask,
                    },
                )
            )
        else:
            result = dict(self._policy.infer(standardized))
            rtc_info = {"rtc_enabled": False, "rtc_skip_steps": 0}
        full_actions = np.asarray(result["actions"], dtype=np.float32)
        if full_actions.shape != (horizon, PIPER_DIM):
            raise RuntimeError(
                f"FastWAM output contract mismatch: expected {(horizon, PIPER_DIM)}, "
                f"got {full_actions.shape}"
            )
        if not np.isfinite(full_actions).all():
            raise RuntimeError("FastWAM returned non-finite Piper actions")
        result["actions"] = full_actions[:requested].copy()
        result["full_actions"] = full_actions
        result["rtc"] = rtc_info
        return result


def create_policy(args: argparse.Namespace) -> PiperFastWAMPolicy:
    from openpi.cotrain import config as cotrain_config
    from openpi.models_pytorch.fastwam.runtime_env import configure_fastwam_runtime_env
    from openpi.policies import policy_config

    configure_fastwam_runtime_env(repo_root=args.openpi_root, offline=True)
    train_config = cotrain_config.get_config(args.config_name)
    train_config = dataclasses.replace(
        train_config,
        model=dataclasses.replace(
            train_config.model,
            camera_keys=ROBOTWIN_POLICY_CAMERA_KEYS,
            concat_multi_camera="none",
            image_resolution=ROBOTWIN_IMAGE_RESOLUTION,
            skip_dit_load_from_pretrain=True,
            skip_vae_load_from_pretrain=False,
        ),
        assets_base_dir=str((args.openpi_root / "assets").resolve()),
    )
    effective_norm_stats_path = (Path(train_config.assets_dirs) / "piper30" / "norm_stats.json").resolve()
    requested_norm_stats_path = args.norm_stats_path.resolve()
    if requested_norm_stats_path != effective_norm_stats_path:
        raise ValueError(
            "Configured norm stats do not match the piper30 stats used by the co-training policy: "
            f"configured={requested_norm_stats_path}, effective={effective_norm_stats_path}"
        )
    policy = policy_config.create_trained_policy(
        train_config,
        args.checkpoint_dir,
        default_prompt=args.prompt,
        denormalize_outputs=True,
        pytorch_device=args.device,
        sample_kwargs={"num_inference_steps": args.num_inference_steps, "seed": args.seed},
    )
    return PiperFastWAMPolicy(
        policy,
        prompt=args.prompt,
        actions_per_inference=args.actions_per_inference,
        seed=args.seed,
        rtc_config=RtcConfig(
            enabled=args.rtc_enabled,
            delay_steps=args.rtc_delay_steps,
            execution_horizon=args.rtc_execution_horizon,
            soft_mask_decay=args.rtc_soft_mask_decay,
            guidance_scale=args.rtc_guidance_scale,
        ),
    )


def validate_contract(policy: PiperFastWAMPolicy, args: argparse.Namespace) -> None:
    synthetic = {
        "active_state": np.zeros(PIPER_DIM, dtype=np.float32),
        "images": {
            "head": np.zeros((224, 224, 3), dtype=np.uint8),
            "left_wrist": np.zeros((224, 224, 3), dtype=np.uint8),
            "right_wrist": np.zeros((224, 224, 3), dtype=np.uint8),
        },
        "prompt": args.prompt or "validation only",
    }
    transformed = policy._policy._input_transform(policy._standardize(synthetic))
    dataset_id = transformed.get("_cotrain_dataset_id")
    fake_output = {
        "state": np.asarray(transformed["state"]),
        "actions": np.zeros(
            (policy._policy._model.action_horizon, policy._policy._model.action_dim),
            dtype=np.float32,
        ),
        "_cotrain_dataset_id": dataset_id,
    }
    native = policy._policy._output_transform(fake_output)["actions"]
    expected = (int(policy._policy._model.action_horizon), PIPER_DIM)
    if np.asarray(native).shape != expected:
        raise RuntimeError(f"Output transform mismatch: expected {expected}, got {np.asarray(native).shape}")
    rtc_validation = None
    if policy._rtc_config.enabled:
        previous = np.ones(expected, dtype=np.float32)
        target_actions, target_mask, rtc_validation = policy._build_rtc_target(
            policy._standardize(synthetic),
            np.zeros(PIPER_DIM, dtype=np.float32),
            {
                "previous_full_actions": previous,
                "previous_action_start_index": 0,
                "switch_steps": 2,
            },
        )
        if target_actions.shape != (expected[0], int(policy._policy._model.action_dim)):
            raise RuntimeError("RTC model-space target validation failed")
        if not np.allclose(
            target_mask[: policy._rtc_config.delay_steps, np.asarray(PIPER_STATE_SLOTS)],
            policy._rtc_config.guidance_scale,
        ):
            raise RuntimeError("RTC hard-mask validation failed")
        if rtc_validation["rtc_prior_horizon"] != expected[0]:
            raise RuntimeError("RTC prior horizon validation failed")
    print(
        json.dumps(
            {
                "status": "VALIDATE_ONLY_OK",
                "config_name": args.config_name,
                "checkpoint": str(args.checkpoint_dir),
                "model_action_dim": int(policy._policy._model.action_dim),
                "action_horizon": int(policy._policy._model.action_horizon),
                "inference_seed": args.seed,
                "piper_output_shape": list(expected),
                "camera_keys": list(policy._policy._model.config.camera_keys),
                "dataset_id": dataset_id,
                "rtc": rtc_validation or {"rtc_enabled": False},
                "robot_motion": False,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def main() -> None:
    args = parse_args()
    add_openpi_paths(args.openpi_root)
    os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(args.openpi_root / "checkpoints" / "fastwam"))
    os.environ.setdefault("HF_HOME", str(args.openpi_root / "checkpoints" / "fastwam" / "hf_cache"))
    os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")
    policy = create_policy(args)
    if args.validate_only:
        validate_contract(policy, args)
        return

    from openpi.serving import websocket_policy_server

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=policy.metadata,
    )
    logging.info("Serving FastWAM Piper policy on %s:%d", args.host, args.port)
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
