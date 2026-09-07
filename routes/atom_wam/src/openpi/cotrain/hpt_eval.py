"""HPT validation metrics: total loss + action MSE (piper-only or filtered subsets)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

logger = logging.getLogger("openpi")


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    return model


def _to_device(observation, actions, device):
    def _map(x):
        if x is None:
            return None
        if isinstance(x, dict):
            return {k: _map(v) for k, v in x.items()}
        if isinstance(x, (str, bytes)):
            return x
        if isinstance(x, np.ndarray) and (
            x.dtype == object or np.issubdtype(x.dtype, np.str_) or np.issubdtype(x.dtype, np.bytes_)
        ):
            return x
        t = torch.as_tensor(x) if not isinstance(x, torch.Tensor) else x
        return t.to(device)

    prompts = getattr(observation, "_fastwam_prompts", None)
    is_ego = getattr(observation, "_fastwam_is_ego", None)
    obs = type(observation).from_dict(_map(observation.to_dict()))
    if prompts is not None:
        object.__setattr__(obs, "_fastwam_prompts", prompts)
    if is_ego is not None:
        object.__setattr__(obs, "_fastwam_is_ego", _map(is_ego) if not isinstance(is_ego, list) else is_ego)
    return obs, _map(actions)


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _nanmean(xs: list[float]) -> float:
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")


def _simple_mean(per_dataset: dict[str, float]) -> float:
    return sum(per_dataset.values()) / len(per_dataset)


def _add_aggregate(
    metrics: dict[str, float],
    key: str,
    per_dataset: dict[str, float],
    train_weights: dict[str, float] | None,
) -> None:
    if not per_dataset:
        return
    if train_weights:
        num = sum(per_dataset[n] * train_weights.get(n, 0.0) for n in per_dataset)
        den = sum(train_weights.get(n, 0.0) for n in per_dataset)
        metrics[key] = num / den if den > 0 else _simple_mean(per_dataset)
    else:
        metrics[key] = _simple_mean(per_dataset)


@torch.no_grad()
def val_action_mse(
    model: torch.nn.Module,
    observation,
    actions: torch.Tensor,
    *,
    device: torch.device,
    num_inference_steps: int,
    seed: int,
    fallback_mask: tuple[bool, ...] | None,
) -> float:
    _set_seed(seed)
    pred = model.sample_actions(device, observation, num_steps=num_inference_steps)
    err2 = (pred.float() - actions.float()) ** 2

    mask = observation.action_mask
    if mask is None:
        if fallback_mask is None:
            return float(err2.mean().cpu())
        mask = torch.tensor(fallback_mask, device=pred.device, dtype=torch.bool).unsqueeze(0).expand(
            pred.shape[0], -1
        )
    else:
        mask = mask.to(device=pred.device, dtype=torch.bool)

    mask = mask.unsqueeze(-2)
    denom = float(mask.sum().item()) * actions.shape[-2]
    denom = max(denom, 1.0)
    return float((err2 * mask).sum().cpu() / denom)


@torch.no_grad()
def run_eval(
    val_loaders: dict[str, dict[str, Any]],
    model: torch.nn.Module,
    *,
    device: torch.device,
    num_val_batches: int = 2,
    run_action_mse: bool = True,
    num_action_mse_batches: int = 2,
    action_mse_num_denoise_steps: int = 10,
    val_seed: int = 0,
    action_masks: dict[str, tuple[bool, ...]] | None = None,
    train_weights: dict[str, float] | None = None,
    max_datasets: int | None = None,
) -> dict[str, float]:
    """Run per-(label, dataset) validation; metrics keyed for wandb."""
    root = _unwrap_model(model)
    was_training = root.training
    root.eval()
    action_masks = action_masks or {}

    metrics: dict[str, float] = {}
    try:
        for label, loaders in val_loaders.items():
            items = list(loaders.items())
            if max_datasets is not None and max_datasets > 0:
                items = items[:max_datasets]

            per_dataset_loss: dict[str, float] = {}
            per_dataset_mse: dict[str, float] = {}

            for ds_name, loader in items:
                loss_vals: list[float] = []
                it = iter(loader)
                for bi in range(num_val_batches):
                    try:
                        obs, acts = next(it)
                    except StopIteration:
                        break
                    obs, acts = _to_device(obs, acts, device)
                    out = root.compute_loss(obs, acts, train=False)
                    loss_vals.append(float(out["loss"].detach().cpu()))

                if loss_vals:
                    v = _nanmean(loss_vals)
                    metrics[f"val/{label}/{ds_name}/loss"] = v
                    per_dataset_loss[ds_name] = v

                if run_action_mse:
                    mse_vals: list[float] = []
                    it2 = iter(loader)
                    for bi in range(num_action_mse_batches):
                        try:
                            obs, acts = next(it2)
                        except StopIteration:
                            break
                        obs, acts = _to_device(obs, acts, device)
                        mse_vals.append(
                            val_action_mse(
                                root,
                                obs,
                                acts,
                                device=device,
                                num_inference_steps=action_mse_num_denoise_steps,
                                seed=val_seed + bi,
                                fallback_mask=action_masks.get(ds_name),
                            )
                        )
                    if mse_vals:
                        v = sum(mse_vals) / len(mse_vals)
                        metrics[f"val/{label}/{ds_name}/action_mse"] = v
                        per_dataset_mse[ds_name] = v

            _add_aggregate(metrics, f"val/{label}/agg/loss", per_dataset_loss, train_weights)
            _add_aggregate(metrics, f"val/{label}/agg/action_mse", per_dataset_mse, train_weights)
    finally:
        root.train(was_training)

    return metrics
