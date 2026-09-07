"""PyTorch FastWAM validation metrics: flow-matching loss + action MSE.

Mirrors ``openpi.cotrain.eval`` key naming so W&B curves align with ``train_cotrain.py``:
  * ``val/{seen|unseen}/{dataset}/flow_loss_fixed``
  * ``val/{seen|unseen}/{dataset}/flow_loss_multi``
  * ``val/{seen|unseen}/{dataset}/action_mse``
  * ``val/{seen|unseen}/agg/...``

FastWAM also logs ``loss_video`` / ``loss_action`` breakdown alongside total flow loss.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import openpi.models.model as _model

logger = logging.getLogger(__name__)


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    return model


def _to_device_observation(observation: _model.Observation, device: torch.device) -> _model.Observation:
    def _map(x):
        if isinstance(x, torch.Tensor):
            return x.to(device=device, non_blocking=True)
        return x

    prompts = getattr(observation, "_fastwam_prompts", None)
    is_ego = getattr(observation, "_fastwam_is_ego", None)
    mapped = type(observation).from_dict({k: _map(v) for k, v in observation.to_dict().items()})
    if prompts is not None:
        object.__setattr__(mapped, "_fastwam_prompts", prompts)
    if is_ego is not None:
        object.__setattr__(
            mapped,
            "_fastwam_is_ego",
            _map(is_ego) if isinstance(is_ego, torch.Tensor) else is_ego,
        )
    return mapped


def _flow_components(model: torch.nn.Module, observation, actions) -> dict[str, float]:
    losses = model.compute_loss(observation, actions, train=False)
    keys = ("loss", "loss_video", "loss_action")
    return {k: float(losses[k].detach().cpu()) for k in keys}


def val_flow_loss_fixed(
    model: torch.nn.Module,
    observation,
    actions: torch.Tensor,
    *,
    seed: int,
) -> dict[str, float]:
    _set_seed(seed)
    return _flow_components(model, observation, actions)


def val_flow_loss_multi(
    model: torch.nn.Module,
    observation,
    actions: torch.Tensor,
    *,
    num_samples: int,
    seed: int,
) -> dict[str, float]:
    totals: dict[str, float] = {}
    for i in range(num_samples):
        _set_seed(seed + i)
        comp = _flow_components(model, observation, actions)
        for key, value in comp.items():
            totals[key] = totals.get(key, 0.0) + value
    return {k: v / num_samples for k, v in totals.items()}


def val_action_mse(
    model: torch.nn.Module,
    observation,
    actions: torch.Tensor,
    *,
    device: torch.device,
    num_denoise_steps: int,
    seed: int,
    fallback_mask: tuple[bool, ...] | None,
) -> float:
    _set_seed(seed)
    pred = model.sample_actions(device, observation, num_inference_steps=num_denoise_steps, seed=seed)
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


def _nanmean(xs: list[float]) -> float:
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")


def _simple_mean(per_dataset: dict[str, float]) -> float:
    return sum(per_dataset.values()) / len(per_dataset)


def _add_aggregate(metrics: dict[str, float], key: str, per_dataset: dict[str, float], train_weights):
    if not per_dataset:
        return
    if train_weights:
        num = sum(per_dataset[n] * train_weights.get(n, 0.0) for n in per_dataset)
        den = sum(train_weights.get(n, 0.0) for n in per_dataset)
        metrics[key] = num / den if den > 0 else _simple_mean(per_dataset)
    else:
        metrics[key] = _simple_mean(per_dataset)


def run_eval(
    val_loaders_by_label: dict,
    model: torch.nn.Module,
    *,
    device: torch.device,
    flow_mode: str,
    run_action_mse: bool,
    num_val_batches: int,
    num_action_mse_batches: int,
    val_flow_loss_num_samples: int,
    action_mse_num_denoise_steps: int,
    val_seed: int,
    train_weights: dict[str, float] | None,
    action_masks: dict[str, tuple[bool, ...]],
) -> dict[str, float]:
    """Run per-(label, dataset) validation and return wandb-ready metrics."""
    model = _unwrap_model(model)
    was_training = model.training
    model.eval()

    metrics: dict[str, float] = {}
    try:
        for label, loaders in val_loaders_by_label.items():
            per_dataset_fixed: dict[str, float] = {}
            per_dataset_multi: dict[str, float] = {}
            per_dataset_mse: dict[str, float] = {}
            per_dataset_video: dict[str, float] = {}
            per_dataset_action: dict[str, float] = {}

            for name, loader in loaders.items():
                fixed_vals: list[float] = []
                multi_vals: list[float] = []
                video_vals: list[float] = []
                action_vals: list[float] = []

                it = iter(loader)
                for bi in range(num_val_batches):
                    try:
                        observation, actions = next(it)
                    except StopIteration:
                        break
                    observation = _to_device_observation(observation, device)
                    actions = actions.to(device=device, non_blocking=True)
                    seed = val_seed + bi

                    if flow_mode in ("fixed_seed", "both"):
                        out = val_flow_loss_fixed(model, observation, actions, seed=seed)
                        fixed_vals.append(out["loss"])
                        video_vals.append(out["loss_video"])
                        action_vals.append(out["loss_action"])

                    if flow_mode in ("multi_sample", "both"):
                        out = val_flow_loss_multi(
                            model,
                            observation,
                            actions,
                            num_samples=val_flow_loss_num_samples,
                            seed=seed,
                        )
                        multi_vals.append(out["loss"])

                if flow_mode in ("fixed_seed", "both") and fixed_vals:
                    v = _nanmean(fixed_vals)
                    metrics[f"val/{label}/{name}/flow_loss_fixed"] = v
                    per_dataset_fixed[name] = v
                    metrics[f"val/{label}/{name}/loss_video"] = _nanmean(video_vals)
                    metrics[f"val/{label}/{name}/loss_action"] = _nanmean(action_vals)
                    per_dataset_video[name] = metrics[f"val/{label}/{name}/loss_video"]
                    per_dataset_action[name] = metrics[f"val/{label}/{name}/loss_action"]

                if flow_mode in ("multi_sample", "both") and multi_vals:
                    v = _nanmean(multi_vals)
                    metrics[f"val/{label}/{name}/flow_loss_multi"] = v
                    per_dataset_multi[name] = v

                if run_action_mse:
                    mse_vals: list[float] = []
                    it2 = iter(loader)
                    for bi in range(num_action_mse_batches):
                        try:
                            observation, actions = next(it2)
                        except StopIteration:
                            break
                        observation = _to_device_observation(observation, device)
                        actions = actions.to(device=device, non_blocking=True)
                        mse_vals.append(
                            val_action_mse(
                                model,
                                observation,
                                actions,
                                device=device,
                                num_denoise_steps=action_mse_num_denoise_steps,
                                seed=val_seed + bi,
                                fallback_mask=action_masks.get(name),
                            )
                        )
                    if mse_vals:
                        v = sum(mse_vals) / len(mse_vals)
                        metrics[f"val/{label}/{name}/action_mse"] = v
                        per_dataset_mse[name] = v

            _add_aggregate(metrics, f"val/{label}/agg/flow_loss_fixed", per_dataset_fixed, train_weights)
            _add_aggregate(metrics, f"val/{label}/agg/flow_loss_multi", per_dataset_multi, train_weights)
            _add_aggregate(metrics, f"val/{label}/agg/action_mse", per_dataset_mse, train_weights)
            _add_aggregate(metrics, f"val/{label}/agg/loss_video", per_dataset_video, train_weights)
            _add_aggregate(metrics, f"val/{label}/agg/loss_action", per_dataset_action, train_weights)
    finally:
        model.train(was_training)

    return metrics
