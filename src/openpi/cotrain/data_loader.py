"""Data-loader construction for co-training with train/val splits.

Thin layer on top of `openpi.cotrain.rlds_dataset.CotrainRldsDataset`. It reuses the
unchanged openpi helpers `transform_iterable_dataset`, `RLDSDataLoader`, and
`DataLoaderImpl` so we do not duplicate the transform/sharding machinery.
"""

import dataclasses
import logging
from typing import Literal

import jax
import numpy as np
import torch

from openpi.cotrain.rlds_dataset import CotrainRldsDataset
from openpi.cotrain.rlds_dataset import Split
import openpi.training.config as _config

# Reuse the unchanged openpi pieces.
from openpi.training.data_loader import DataLoaderImpl
from openpi.training.data_loader import RLDSDataLoader
from openpi.training.data_loader import transform_iterable_dataset
import openpi.models.model as _model

# Common image size for mixed-resolution batching, matching the model's ResizeImages target
# (openpi ModelTransformFactory hardcodes ResizeImages(224, 224)). Images are resize_with_pad'd
# to this in the TF pipeline before batching; the later model-transform resize is then idempotent.
_MODEL_IMAGE_HW = (224, 224)


def resolve_val_batch_size(config: _config.TrainConfig) -> int:
    """Return the configured global validation batch size with legacy fallback."""
    configured = getattr(config, "val_batch_size", None)
    val_batch_size = config.batch_size if configured is None else configured
    if val_batch_size <= 0:
        raise ValueError(f"val_batch_size must be positive, got {val_batch_size}.")
    return val_batch_size


class CotrainRLDSDataLoader(RLDSDataLoader):
    """openpi RLDSDataLoader, but without the hard `process_count() > 1` block.

    The base class raises NotImplementedError for multi-process, yet its `__iter__` already
    assembles a global sharded array via `jax.make_array_from_process_local_data` (which IS the
    multi-host primitive). Each host feeds its OWN local_batch_size slice of DIFFERENT data
    (per-process split sharding lives in CotrainRldsDataset). So we just reimplement __init__
    to skip the guard; __iter__ is inherited unchanged. (No openpi file is modified.)
    """

    def __init__(self, dataset, *, sharding: jax.sharding.Sharding | None = None, num_batches: int | None = None):
        self._dataset = dataset
        if sharding is None:
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._sharding = sharding
        self._num_batches = num_batches


def create_cotrain_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    split_label: Split = "train",
    shuffle: bool = False,
    shuffle_buffer_size: int = 250_000,
    num_parallel_reads: int = -1,
    num_parallel_calls: int = -1,
    pad_action_dim: int | None = None,
    image_resize_hw: tuple[int, int] | None = None,
    video_num_frames: int | None = None,
    action_video_freq_ratio: int = 4,
) -> CotrainRldsDataset:
    if data_config.rlds_data_dir is None:
        raise ValueError("rlds_data_dir must be set for the co-training RLDS loader.")
    # Multi-host: each process produces its OWN local_batch_size slice; the loader assembles the
    # global batch via make_array_from_process_local_data. Single-host -> process_count=1 -> unchanged.
    process_count = jax.process_count()
    if batch_size % process_count != 0:
        raise ValueError(f"batch_size ({batch_size}) must be divisible by process_count ({process_count}).")
    local_batch_size = batch_size // process_count
    return CotrainRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=local_batch_size,
        datasets=data_config.datasets,
        split_label=split_label,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        shuffle_buffer_size=shuffle_buffer_size,
        num_parallel_reads=num_parallel_reads,
        num_parallel_calls=num_parallel_calls,
        pad_action_dim=pad_action_dim,
        image_resize_hw=image_resize_hw,
        video_num_frames=video_num_frames,
        action_video_freq_ratio=action_video_freq_ratio,
        process_count=process_count,
        process_index=jax.process_index(),
    )


class CotrainRLDSNumpyDataLoader:
    """Cotrain RLDS loader for PyTorch training (no JAX sharding)."""

    def __init__(self, dataset, *, num_batches: int | None = None):
        self._dataset = dataset
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break
                num_items += 1
                batch = dict(batch)

                def _to_tensor(x):
                    if isinstance(x, (str, bytes)):
                        return x
                    if isinstance(x, np.ndarray) and (
                        x.dtype == object
                        or np.issubdtype(x.dtype, np.str_)
                        or np.issubdtype(x.dtype, np.bytes_)
                    ):
                        return x
                    return torch.as_tensor(x)

                yield jax.tree.map(_to_tensor, batch)


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
    num_parallel_reads: int = -1,
    num_parallel_calls: int = -1,
    pad_action_dim: int | None = None,
    image_resize_hw: tuple[int, int] | None = None,
    video_num_frames: int | None = None,
    action_video_freq_ratio: int = 4,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoaderImpl:
    dataset = create_cotrain_rlds_dataset(
        data_config,
        action_horizon,
        batch_size,
        split_label=split_label,
        shuffle=shuffle,
        shuffle_buffer_size=shuffle_buffer_size,
        num_parallel_reads=num_parallel_reads,
        num_parallel_calls=num_parallel_calls,
        pad_action_dim=pad_action_dim,
        image_resize_hw=image_resize_hw,
        video_num_frames=video_num_frames,
        action_video_freq_ratio=action_video_freq_ratio,
    )
    # Per-dataset normalization is handled by DispatchNormalize inside data_transforms.
    del skip_norm_stats
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=True, is_batched=True)
    if framework == "pytorch":
        data_loader = CotrainRLDSNumpyDataLoader(dataset, num_batches=num_batches)
    else:
        data_loader = CotrainRLDSDataLoader(dataset, sharding=sharding, num_batches=num_batches)
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
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoaderImpl:
    """Build the (mixed, weighted) train or val loader from a TrainConfig."""
    data_config = config.data.create(config.assets_dirs, config.model)
    video_num_frames = None
    action_video_freq_ratio = 4
    if getattr(config.model, "model_type", None) == _model.ModelType.FASTWAM:
        video_num_frames = int(getattr(config.model, "video_num_frames", 9))
        action_video_freq_ratio = int(getattr(config.model, "action_video_freq_ratio", 4))
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
        num_parallel_reads=config.data_num_parallel_reads,
        num_parallel_calls=config.data_num_parallel_calls,
        pad_action_dim=config.model.action_dim,
        image_resize_hw=_MODEL_IMAGE_HW,
        video_num_frames=video_num_frames,
        action_video_freq_ratio=action_video_freq_ratio,
        framework=framework,
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
    val_batch_size = resolve_val_batch_size(config)
    logging.info(f"Building validation loaders with global batch size {val_batch_size}.")
    loaders: dict[str, dict[str, DataLoaderImpl]] = {}
    for ds in data_config.datasets:
        single = dataclasses.replace(ds, weight=1.0)
        dc = dataclasses.replace(data_config, datasets=(single,))
        for label in ds.val_labels():
            logging.info(f"Building val loader: dataset='{ds.uid}' label='{label}' split='{ds.val_splits[label]}'")
            loaders.setdefault(label, {})[ds.uid] = create_cotrain_rlds_data_loader(
                dc,
                action_horizon=config.model.action_horizon,
                batch_size=val_batch_size,
                split_label=label,
                sharding=sharding,
                skip_norm_stats=skip_norm_stats,
                shuffle=False,
                num_batches=config.num_val_batches,
                shuffle_buffer_size=1,
                num_parallel_reads=config.data_num_parallel_reads,
                num_parallel_calls=config.data_num_parallel_calls,
                pad_action_dim=config.model.action_dim,
                image_resize_hw=_MODEL_IMAGE_HW,
            )
    return loaders


def dataset_train_weights(config: _config.TrainConfig) -> dict[str, float]:
    """Map dataset name -> configured train mixture weight (for aggregate metrics)."""
    data_config = config.data.create(config.assets_dirs, config.model)
    return {ds.uid: ds.weight for ds in data_config.datasets}


def dataset_action_masks(config: _config.TrainConfig) -> dict[str, tuple[bool, ...]]:
    """Map dataset name to its model-width action mask."""
    data_config = config.data.create(config.assets_dirs, config.model)
    masks = {}
    for ds in data_config.datasets:
        if ds.unified_action_spec is not None:
            masks[ds.uid] = ds.unified_action_spec.action_mask
            continue
        native_dim = getattr(ds, "action_dim", 0) or config.model.action_dim
        masks[ds.uid] = tuple(index < native_dim for index in range(config.model.action_dim))
    return masks
