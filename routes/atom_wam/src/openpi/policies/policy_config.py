import logging
import os
import pathlib
from typing import Any

import jax.numpy as jnp

import openpi.cotrain.transforms as cotrain_transforms
import openpi.models.model as _model
import openpi.models.tokenizer as _tokenizer
import openpi.policies.policy as _policy
import openpi.shared.download as download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
import openpi.transforms as transforms


def resolve_train_config(config_name: str) -> _config.TrainConfig:
    """Resolve a training config from openpi or cotrain registries."""
    try:
        return _config.get_config(config_name)
    except ValueError:
        pass

    from openpi.cotrain import config as cotrain_config

    try:
        return cotrain_config.get_config(config_name)
    except ValueError as exc:
        openpi_names = list(_config._CONFIGS_DICT)  # noqa: SLF001
        cotrain_names = list(cotrain_config._COTRAIN_CONFIGS_DICT)  # noqa: SLF001
        raise ValueError(
            f"Config '{config_name}' not found in openpi ({len(openpi_names)} configs) "
            f"or cotrain ({len(cotrain_names)} configs)."
        ) from exc


def is_cotrain_train_config(train_config: _config.TrainConfig) -> bool:
    from openpi.cotrain.config import CotrainDataConfig

    return isinstance(train_config.data, CotrainDataConfig)


def _build_cotrain_output_transforms(
    data_config: _config.DataConfig,
    assets_dirs: pathlib.Path,
    *,
    denormalize_outputs: bool,
) -> list[transforms.DataTransformFn]:
    if not denormalize_outputs:
        return []

    datasets = getattr(data_config, "datasets", None) or ()
    if not datasets:
        return []

    from openpi.cotrain.config import load_per_dataset_norm_stats

    per_dataset_stats = load_per_dataset_norm_stats(assets_dirs, datasets)
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


def create_cotrain_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    denormalize_outputs: bool = False,
    pytorch_device: str | None = None,
) -> _policy.Policy:
    """Create a co-training policy without duplicate global Normalize/Unnormalize."""
    repack_transforms = repack_transforms or transforms.Group()
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))

    weight_path = os.path.join(checkpoint_dir, "model.safetensors")
    is_pytorch = os.path.exists(weight_path)

    logging.info("Loading co-training model...")
    if is_pytorch:
        model = train_config.model.load_pytorch(train_config, weight_path)
        if hasattr(model, "paligemma_with_expert"):
            model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    else:
        model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))

    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)

    if is_pytorch and pytorch_device is None:
        try:
            import torch

            pytorch_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pytorch_device = "cpu"

    output_transforms = [
        *data_config.model_transforms.outputs,
        *_build_cotrain_output_transforms(
            data_config,
            train_config.assets_dirs,
            denormalize_outputs=denormalize_outputs,
        ),
        *data_config.data_transforms.outputs,
        *repack_transforms.outputs,
    ]

    return _policy.Policy(
        model,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            *(
                [transforms.HistoryBufferTransform(train_config.model.history_length)]
                if getattr(train_config.model, "history_length", 1) > 1
                else []
            ),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=output_transforms,
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
        is_pytorch=is_pytorch,
        pytorch_device=pytorch_device if is_pytorch else None,
    )


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    denormalize_outputs: bool = False,
    pytorch_device: str | None = None,
) -> _policy.Policy:
    """Create a policy from a trained checkpoint.

    Args:
        train_config: The training config to use to create the model.
        checkpoint_dir: The directory to load the model from.
        repack_transforms: Optional transforms that will be applied before any other transforms.
        sample_kwargs: The kwargs to pass to the `sample_actions` method. If not provided, the default
            kwargs will be used.
        default_prompt: The default prompt to use for the policy. Will inject the prompt into the input
            data if it doesn't already exist.
        norm_stats: The norm stats to use for the policy. If not provided, the norm stats will be loaded
            from the checkpoint directory.
        pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda", "cuda:0").
                      If None and is_pytorch=True, will use "cuda" if available, otherwise "cpu".

    Note:
        The function automatically detects whether the model is PyTorch-based by checking for the
        presence of "model.safensors" in the checkpoint directory.
    """
    if is_cotrain_train_config(train_config):
        return create_cotrain_trained_policy(
            train_config,
            checkpoint_dir,
            repack_transforms=repack_transforms,
            sample_kwargs=sample_kwargs,
            default_prompt=default_prompt,
            denormalize_outputs=denormalize_outputs,
            pytorch_device=pytorch_device,
        )

    repack_transforms = repack_transforms or transforms.Group()
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))

    # Check if this is a PyTorch model by looking for model.safetensors
    weight_path = os.path.join(checkpoint_dir, "model.safetensors")
    is_pytorch = os.path.exists(weight_path)

    logging.info("Loading model...")
    if is_pytorch:
        model = train_config.model.load_pytorch(train_config, weight_path)
        # PI0 PyTorch path has an extra precision helper; FastWAM / others skip it.
        if hasattr(model, "paligemma_with_expert"):
            model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    else:
        model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

    # Determine the device to use for PyTorch models
    if is_pytorch and pytorch_device is None:
        try:
            import torch

            pytorch_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pytorch_device = "cpu"

    return _policy.Policy(
        model,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            *(
                [transforms.HistoryBufferTransform(train_config.model.history_length)]
                if getattr(train_config.model, "history_length", 1) > 1
                else []
            ),
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
        is_pytorch=is_pytorch,
        pytorch_device=pytorch_device if is_pytorch else None,
    )
