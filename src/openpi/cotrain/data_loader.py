"""Data-loader construction for co-training with train/val splits.

Thin layer on top of `openpi.cotrain.rlds_dataset.CotrainRldsDataset`. It reuses the
unchanged openpi helpers `transform_iterable_dataset`, `RLDSDataLoader`, and
`DataLoaderImpl` so we do not duplicate the transform/sharding machinery.
"""

import dataclasses
import logging

import jax

import openpi.training.config as _config
# Reuse the unchanged openpi pieces.
from openpi.training.data_loader import DataLoaderImpl, RLDSDataLoader, transform_iterable_dataset

from openpi.cotrain.rlds_dataset import CotrainRldsDataset, Split

# Common image size for mixed-resolution batching, matching the model's ResizeImages target
# (openpi ModelTransformFactory hardcodes ResizeImages(224, 224)). Images are resize_with_pad'd
# to this in the TF pipeline before batching; the later model-transform resize is then idempotent.
_MODEL_IMAGE_HW = (224, 224)


def create_cotrain_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    split_label: Split = "train",
    shuffle: bool = False,
    shuffle_buffer_size: int = 250_000,
    pad_action_dim: int | None = None,
    image_resize_hw: tuple[int, int] | None = None,
) -> CotrainRldsDataset:
    if data_config.rlds_data_dir is None:
        raise ValueError("rlds_data_dir must be set for the co-training RLDS loader.")
    return CotrainRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        datasets=data_config.datasets,
        split_label=split_label,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        shuffle_buffer_size=shuffle_buffer_size,
        pad_action_dim=pad_action_dim,
        image_resize_hw=image_resize_hw,
    )


def create_cotrain_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    split_label: Split = "train",
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    shuffle_buffer_size: int = 250_000,
    pad_action_dim: int | None = None,
    image_resize_hw: tuple[int, int] | None = None,
) -> DataLoaderImpl:
    dataset = create_cotrain_rlds_dataset(
        data_config,
        action_horizon,
        batch_size,
        split_label=split_label,
        shuffle=shuffle,
        shuffle_buffer_size=shuffle_buffer_size,
        pad_action_dim=pad_action_dim,
        image_resize_hw=image_resize_hw,
    )
    # The built-in openpi Normalize is disabled (skip_norm_stats=True) because per-dataset
    # normalization is handled by DispatchNormalize inside data_transforms (keyed by dataset_id).
    # `skip_norm_stats` arg here is accepted for API symmetry but the built-in stays off.
    del skip_norm_stats
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=True, is_batched=True)
    data_loader = RLDSDataLoader(dataset, sharding=sharding, num_batches=num_batches)
    return DataLoaderImpl(data_config, data_loader)


def create_cotrain_data_loader(
    config: _config.TrainConfig,
    *,
    split_label: Split = "train",
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    shuffle_buffer_size: int = 250_000,
) -> DataLoaderImpl:
    """Build the (mixed, weighted) train or val loader from a TrainConfig."""
    data_config = config.data.create(config.assets_dirs, config.model)
    return create_cotrain_rlds_data_loader(
        data_config,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        split_label=split_label,
        sharding=sharding,
        skip_norm_stats=skip_norm_stats,
        shuffle=shuffle,
        num_batches=num_batches,
        shuffle_buffer_size=shuffle_buffer_size,
        pad_action_dim=config.model.action_dim,
        image_resize_hw=_MODEL_IMAGE_HW,
    )


def build_val_loaders(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
) -> dict[str, dict[str, DataLoaderImpl]]:
    """Per-label, per-dataset validation loaders.

    Returns {val_label: {dataset_name: loader}} where val_label is a key of the dataset's
    `val_splits` (e.g. "seen", "unseen"). Each loader is single-dataset (weight=1.0),
    finite and deterministic. Only datasets that expose a given label appear under it.
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    loaders: dict[str, dict[str, DataLoaderImpl]] = {}
    for ds in data_config.datasets:
        single = dataclasses.replace(ds, weight=1.0)
        dc = dataclasses.replace(data_config, datasets=(single,))
        for label in ds.val_labels():
            logging.info(f"Building val loader: dataset='{ds.uid}' label='{label}' split='{ds.val_splits[label]}'")
            loaders.setdefault(label, {})[ds.uid] = create_cotrain_rlds_data_loader(
                dc,
                action_horizon=config.model.action_horizon,
                batch_size=config.batch_size,
                split_label=label,
                sharding=sharding,
                skip_norm_stats=skip_norm_stats,
                shuffle=False,
                num_batches=config.num_val_batches,
                shuffle_buffer_size=1,
                pad_action_dim=config.model.action_dim,
                image_resize_hw=_MODEL_IMAGE_HW,
            )
    return loaders


def dataset_train_weights(config: _config.TrainConfig) -> dict[str, float]:
    """Map dataset name -> configured train mixture weight (for aggregate metrics)."""
    data_config = config.data.create(config.assets_dirs, config.model)
    return {ds.uid: ds.weight for ds in data_config.datasets}


def dataset_action_dims(config: _config.TrainConfig) -> dict[str, int]:
    """Map dataset name -> native action dim (for the per-dataset action-MSE mask).

    0 means "use all dims" (no mask). Falls back to 0 if a dataset entry lacks action_dim.
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    return {ds.uid: getattr(ds, "action_dim", 0) or None for ds in data_config.datasets}
