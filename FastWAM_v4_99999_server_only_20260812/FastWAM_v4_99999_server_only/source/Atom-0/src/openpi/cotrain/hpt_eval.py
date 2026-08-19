"""Lightweight validation for HPT co-training (flow + world losses)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

logger = logging.getLogger("openpi")


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


@torch.no_grad()
def run_eval(
    val_loaders: dict[str, dict[str, Any]],
    model: torch.nn.Module,
    *,
    device: torch.device,
    num_val_batches: int = 2,
    max_datasets: int | None = None,
) -> dict[str, float]:
    root = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    was_training = root.training
    root.eval()

    metrics: dict[str, float] = {}
    for label, loaders in val_loaders.items():
        items = list(loaders.items())
        if max_datasets is not None:
            items = items[:max_datasets]
        for ds_name, loader in items:
            losses = []
            for i, (obs, acts) in enumerate(loader):
                if i >= num_val_batches:
                    break
                obs, acts = _to_device(obs, acts, device)
                out = root.compute_loss(obs, acts, train=False)
                losses.append(float(out["loss"].detach().cpu()))
            if losses:
                metrics[f"val/{label}/{ds_name}/loss"] = float(np.mean(losses))

    root.train(was_training)
    return metrics
