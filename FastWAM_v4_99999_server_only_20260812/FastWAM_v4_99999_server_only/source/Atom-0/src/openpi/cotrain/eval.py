"""Validation metrics for co-training: flow-matching loss + action MSE.

Two flow-loss estimators are provided (selectable / both):
  * fixed-seed:  single `compute_loss` call with a fixed rng -> identical every eval,
                 cheap, low bias but higher variance.
  * multi-sample: average `compute_loss` over K noise/timestep draws on the same batch
                 -> lower-variance estimate of the flow loss, K x more compute.

Action MSE (a la EgoVerse offline metric): run the full flow-matching sampler
(`sample_actions`) and compare the predicted action chunk to the ground-truth chunk,
masked to the sample's valid unified action dimensions, in the model's normalized space.

All metrics are computed with `model.eval()` (no dropout) and a fixed rng so the curves
are comparable across checkpoints. Static config (num samples, denoise steps, valid
dims, ema) is baked via closures in the `make_*` factories so the returned step
functions can be `jax.jit`-ed without static-argnum bookkeeping.
"""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.training.utils as training_utils


def _select_params(state: training_utils.TrainState, use_ema: bool):
    if use_ema and state.ema_params is not None:
        return state.ema_params
    return state.params


def _flow_loss(model: _model.BaseModel, rng: at.KeyArrayLike, observation, actions) -> at.Array:
    out = model.compute_loss(rng, observation, actions, train=False)
    flow = out["flow"] if isinstance(out, dict) else out
    return jnp.mean(flow)


def make_val_flow_step(*, num_samples: int, mode: str, use_ema: bool):
    """Build a jitted step returning {'fixed': ..., 'multi': ...} flow-loss estimates.

    `mode` in {"fixed_seed", "multi_sample", "both"} controls which estimates are
    actually computed; the others are returned as NaN so the dict shape is static.
    """

    def step(rng, state, batch):
        params = _select_params(state, use_ema)
        model = nnx.merge(state.model_def, params)
        model.eval()
        observation, actions = batch

        nan = jnp.float32(jnp.nan)
        fixed = nan
        multi = nan

        if mode in ("fixed_seed", "both"):
            fixed = _flow_loss(model, rng, observation, actions)

        if mode in ("multi_sample", "both"):
            rngs = jax.random.split(rng, num_samples)
            total = jnp.float32(0.0)
            for i in range(num_samples):  # static unroll (num_samples is a Python int)
                total = total + _flow_loss(model, rngs[i], observation, actions)
            multi = total / num_samples

        return {"fixed": fixed, "multi": multi}

    return jax.jit(step)


def make_val_action_mse_step(*, num_denoise_steps: int, fallback_mask, use_ema: bool):
    """Build a jitted step returning the masked action MSE for one batch."""

    def step(rng, state, batch):
        params = _select_params(state, use_ema)
        model = nnx.merge(state.model_def, params)
        model.eval()
        observation, actions = batch

        pred = model.sample_actions(rng, observation, num_steps=num_denoise_steps)
        err2 = (pred - actions) ** 2  # [B, H, Ad]

        mask = observation.action_mask
        if mask is None:
            mask = jnp.broadcast_to(
                jnp.asarray(fallback_mask, dtype=jnp.bool_), actions.shape[:-2] + actions.shape[-1:]
            )
        mask = jnp.expand_dims(mask, axis=-2)
        denom = jnp.sum(mask) * actions.shape[-2]
        return jnp.sum(err2 * mask) / jnp.clip(denom, 1)

    return jax.jit(step)


def make_val_action_pred_step(*, num_denoise_steps: int, use_ema: bool):
    """Build a jitted step returning sampled (pred) and ground-truth (gt) action chunks."""

    def step(rng, state, batch):
        params = _select_params(state, use_ema)
        model = nnx.merge(state.model_def, params)
        model.eval()
        observation, actions = batch
        pred = model.sample_actions(rng, observation, num_steps=num_denoise_steps)
        return {"pred": pred, "gt": actions}

    return jax.jit(step)


def plot_action_trajectories(pred, gt, *, action_mask=None, slot_names=None, n_samples: int = 1, title: str = ""):
    """Per-dim predicted-vs-GT action-chunk trajectories. Returns a matplotlib Figure.

    pred/gt are [B, H, Ad] in the model's normalized(+delta) space (same space as the MSE).
    Plots the first `n_samples` examples; x-axis is the action-chunk step (0..H-1).
    """
    import math

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pred = np.asarray(pred)
    gt = np.asarray(gt)
    b, h, ad = pred.shape
    active_slots = np.flatnonzero(np.ones(ad, dtype=bool) if action_mask is None else np.asarray(action_mask)[:ad])
    if not len(active_slots):
        raise ValueError("action_mask has no active slots")
    n = min(n_samples, b)
    ncols = min(len(active_slots), 4)
    nrows = math.ceil(len(active_slots) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2 * nrows), squeeze=False)
    x = np.arange(h)
    for plot_index, dim in enumerate(active_slots):
        ax = axes[plot_index // ncols][plot_index % ncols]
        for s in range(n):
            ax.plot(x, gt[s, :, dim], color="tab:green", alpha=0.8, lw=1.2, label="gt" if s == 0 else None)
            ax.plot(x, pred[s, :, dim], color="tab:red", ls="--", alpha=0.8, lw=1.2, label="pred" if s == 0 else None)
        name = slot_names[dim] if slot_names is not None else f"dim_{dim}"
        ax.set_title(f"U{dim + 1} {name}", fontsize=8)
        ax.tick_params(labelsize=6)
    for k in range(len(active_slots), nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    axes[0][0].legend(fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


def run_eval(
    val_loaders_by_label: dict,
    train_state: training_utils.TrainState,
    *,
    val_flow_step,
    val_action_mse_steps: dict,
    flow_mode: str,
    run_action_mse: bool,
    num_val_batches: int,
    num_action_mse_batches: int,
    val_seed: int,
    train_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Run per-(label,dataset) + per-label-aggregate validation.

    `val_loaders_by_label` = {val_label: {dataset_name: loader}} (e.g. label in
    {"seen", "unseen"}). Returns a flat dict of wandb-ready floats keyed
    `val/{label}/{dataset}/...` and `val/{label}/agg/...`. The rng is fixed (derived from
    `val_seed` + batch index) so metrics are deterministic and comparable across evals.
    """
    base_rng = jax.random.key(val_seed)
    metrics: dict[str, float] = {}

    for label, loaders in val_loaders_by_label.items():
        per_dataset_fixed: dict[str, float] = {}
        per_dataset_multi: dict[str, float] = {}
        per_dataset_mse: dict[str, float] = {}

        for name, loader in loaders.items():
            fixed_vals, multi_vals = [], []
            it = iter(loader)
            for bi in range(num_val_batches):
                try:
                    batch = next(it)
                except StopIteration:
                    break
                rng = jax.random.fold_in(base_rng, bi)
                out = jax.device_get(val_flow_step(rng, train_state, batch))
                fixed_vals.append(float(out["fixed"]))
                multi_vals.append(float(out["multi"]))

            if flow_mode in ("fixed_seed", "both") and fixed_vals:
                v = _nanmean(fixed_vals)
                metrics[f"val/{label}/{name}/flow_loss_fixed"] = v
                per_dataset_fixed[name] = v
            if flow_mode in ("multi_sample", "both") and multi_vals:
                v = _nanmean(multi_vals)
                metrics[f"val/{label}/{name}/flow_loss_multi"] = v
                per_dataset_multi[name] = v

            if run_action_mse and name in val_action_mse_steps:
                mse_step = val_action_mse_steps[name]
                mse_vals = []
                it2 = iter(loader)
                for bi in range(num_action_mse_batches):
                    try:
                        batch = next(it2)
                    except StopIteration:
                        break
                    rng = jax.random.fold_in(base_rng, bi)
                    mse_vals.append(float(jax.device_get(mse_step(rng, train_state, batch))))
                if mse_vals:
                    v = sum(mse_vals) / len(mse_vals)
                    metrics[f"val/{label}/{name}/action_mse"] = v
                    per_dataset_mse[name] = v

        # Aggregate across datasets within this label (weighted by train mixture weight).
        _add_aggregate(metrics, f"val/{label}/agg/flow_loss_fixed", per_dataset_fixed, train_weights)
        _add_aggregate(metrics, f"val/{label}/agg/flow_loss_multi", per_dataset_multi, train_weights)
        _add_aggregate(metrics, f"val/{label}/agg/action_mse", per_dataset_mse, train_weights)

    return metrics


def _nanmean(xs: list[float]) -> float:
    xs = [x for x in xs if x == x]  # drop NaNs
    return sum(xs) / len(xs) if xs else float("nan")


def _add_aggregate(metrics, key, per_dataset, train_weights):
    if not per_dataset:
        return
    if train_weights:
        num = sum(per_dataset[n] * train_weights.get(n, 0.0) for n in per_dataset)
        den = sum(train_weights.get(n, 0.0) for n in per_dataset)
        metrics[key] = num / den if den > 0 else _simple_mean(per_dataset)
    else:
        metrics[key] = _simple_mean(per_dataset)


def _simple_mean(per_dataset: dict[str, float]) -> float:
    return sum(per_dataset.values()) / len(per_dataset)
