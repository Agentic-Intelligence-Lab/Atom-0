"""Transforms for the hybrid co-training pipeline.

Hybrid design:
  * Schema is standardized OFFLINE (Option 1): every dataset is converted to a common
    RLDS schema (see `rlds_dataset._standardized_restructure` for the contract), so after
    `sample_from_datasets` all samples already share one structure -> a single GENERIC
    inputs transform (`StandardizedInputs`) works for every dataset.
  * Normalization is dispatched at RUNTIME by `dataset_id` (Option 2): `DispatchNormalize`
    looks at each sample's `dataset_id` tag and applies that dataset's own norm stats.
    This lets us re-tune / re-compute normalization without regenerating the RLDS data.

Action SPACES are NOT unified across robot datasets (pi0/pi05 don't either): each dataset
keeps its native state/action vector placed at the front and zero-padded to action_dim=32
downstream. The model disambiguates via observation/proprioception conditioning. The only
genuinely per-dataset runtime step is normalization.
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms as _transforms
from openpi.models import model as _model

# Canonical image slots for pi0 / pi05 (3 slots; missing cameras -> zeros + mask=False).
_IMAGE_SLOTS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    elif image.ndim == 4 and image.shape[1] == 3:
        image = einops.rearrange(image, "t c h w -> t h w c")
    return image


def _decode_str(value) -> str:
    arr = np.asarray(value)
    item = arr.item() if arr.ndim == 0 else arr.reshape(-1)[0]
    return item.decode("utf-8") if isinstance(item, bytes) else str(item)


@dataclasses.dataclass(frozen=True)
class StandardizedInputs(_transforms.DataTransformFn):
    """Generic inputs transform for the standardized co-training schema.

    Expects (nested) keys produced by the standardized restructure:
        state:      float[Ds]            native proprio (un-padded, un-normalized)
        actions:    float[H, Da]         native action chunk (un-padded, un-normalized)
        image:      {base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb}
        image_mask: {base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb}  (bool)
        prompt:     str
        dataset_id: str   (passed through for DispatchNormalize, popped there)
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["state"])
        current_state = state[-1] if state.ndim == 2 else state

        images = data["image"]
        masks = data["image_mask"]
        out_images = {slot: _parse_image(images[slot]) for slot in _IMAGE_SLOTS}
        out_masks = {slot: np.asarray(masks[slot]).astype(bool) for slot in _IMAGE_SLOTS}

        inputs: dict = {
            "state": current_state,
            "image": out_images,
            "image_mask": out_masks,
        }
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])
        if "prompt" in data:
            inputs["prompt"] = _decode_str(data["prompt"])
        # Carry the dataset tag forward so DispatchNormalize can pick per-dataset stats.
        if "dataset_id" in data:
            inputs["dataset_id"] = _decode_str(data["dataset_id"])
        return inputs


@dataclasses.dataclass(frozen=True)
class StandardizedOutputs(_transforms.DataTransformFn):
    """Inference-time outputs: slice the padded action vector back to native dims.

    `action_dim` is the dataset's native action dimensionality (e.g. 8 for DROID).
    """

    action_dim: int

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[..., : self.action_dim]}


@dataclasses.dataclass(frozen=True)
class DispatchNormalize(_transforms.DataTransformFn):
    """Per-dataset normalization, dispatched by the sample's `dataset_id` tag.

    `norm_stats_by_dataset` maps dataset name -> {"state": NormStats, "actions": NormStats}.
    Reuses openpi's `Normalize` math for the looked-up dataset. Pops `dataset_id` so it
    never reaches JAX sharding (strings are not shardable).
    """

    norm_stats_by_dataset: dict
    use_quantiles: bool = True

    def __call__(self, data: dict) -> dict:
        ds = data.pop("dataset_id", None)
        if ds is not None:
            ds_name = _decode_str(ds)
            stats = self.norm_stats_by_dataset.get(ds_name)
            if stats:
                data = _transforms.Normalize(stats, use_quantiles=self.use_quantiles)(data)
        return data
