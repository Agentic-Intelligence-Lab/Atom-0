"""Transforms for the hybrid co-training pipeline.

Hybrid design:
  * Schema is standardized OFFLINE (Option 1): every dataset is converted to a common
    RLDS schema (see `rlds_dataset._standardized_restructure` for the contract), so after
    `sample_from_datasets` all samples already share one structure -> a single GENERIC
    inputs transform (`StandardizedInputs`) works for every dataset.
  * Normalization is dispatched at RUNTIME by `dataset_id` (Option 2): `DispatchNormalize`
    looks at each sample's `dataset_id` tag and applies that dataset's own norm stats.
    This lets us re-tune / re-compute normalization without regenerating the RLDS data.

The full-all configuration maps every dataset into a fixed 80D physical layout before
mixing. Legacy configurations retain native-prefix padding for compatibility.
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms as _transforms
from openpi.cotrain import action_space as cotrain_action_space
from openpi.models import model as _model
from openpi.shared import normalize as _normalize

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
        prompt_prefix: str   optional text prepended before tokenizer's "Task:" prefix
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
            # Writable COPY (not a read-only tf view): DeltaActions mutates actions in place.
            inputs["actions"] = np.array(data["actions"])
        if "action_mask" in data:
            inputs["action_mask"] = np.asarray(data["action_mask"], dtype=bool)
        if "prompt" in data:
            inputs["prompt"] = _decode_str(data["prompt"])
        if "prompt_prefix" in data:
            inputs["prompt_prefix"] = _decode_str(data["prompt_prefix"])
        # Carry the dataset tag forward so DispatchNormalize can pick per-dataset stats.
        if "dataset_id" in data:
            inputs["dataset_id"] = _decode_str(data["dataset_id"])
        return inputs


@dataclasses.dataclass(frozen=True)
class StandardizedOutputs(_transforms.DataTransformFn):
    """Inference-time outputs: restore the dataset's native action layout.

    `action_dim` is the dataset's native action dimensionality (e.g. 8 for DROID).
    """

    action_dim: int
    unified_action_spec: cotrain_action_space.UnifiedActionSpec | None = None

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        if self.unified_action_spec is not None:
            actions = cotrain_action_space.unmap_array(
                actions, self.unified_action_spec.action_mapping, self.action_dim
            )
        else:
            actions = actions[..., : self.action_dim]
        return {"actions": actions}


@dataclasses.dataclass(frozen=True)
class DispatchDeltaActions(_transforms.DataTransformFn):
    """Per-dataset absolute->delta action conversion, dispatched by `dataset_id`.

    `masks_by_dataset` maps dataset name -> boolean mask (from `make_bool_mask`) selecting
    which action dims become deltas relative to the current state (e.g. joints delta, gripper
    absolute). Runs BEFORE DispatchNormalize and does NOT pop `dataset_id` (normalize needs it).
    Datasets absent from the map (or with a None mask) are left as absolute actions.
    """

    masks_by_dataset: dict

    def __call__(self, data: dict) -> dict:
        ds = data.get("dataset_id")
        if ds is not None and "actions" in data:
            mask = self.masks_by_dataset.get(_decode_str(ds))
            if mask is not None:
                data["actions"] = cotrain_action_space.apply_delta(data["state"], data["actions"], mask)
        return data


@dataclasses.dataclass(frozen=True)
class DispatchNormalize(_transforms.DataTransformFn):
    """Per-dataset normalization, dispatched by the sample's `dataset_id` tag.

    `norm_stats_by_dataset` maps dataset name -> {"state": NormStats, "actions": NormStats}.
    Reuses openpi's `Normalize` math for the looked-up dataset. Pops `dataset_id` so it
    never reaches JAX sharding (strings are not shardable).

    Also emits `domain_mask` (bool): True = human → ego head; False = robot head.
    Human: EgoVerse non-eva (aria/human/mecka/scale/rl2_human) + aligned_*_human_*.
    Robot: egoverse_eva / egoverse_rl2_eva + aligned_*_robot_* + agibot/droid/piper/robocoin/robomind.
    """

    norm_stats_by_dataset: dict
    use_quantiles: bool = True

    def __call__(self, data: dict) -> dict:
        ds = data.pop("dataset_id", None)
        if ds is not None:
            ds_name = _decode_str(ds)
            ROBOT_EGOVERSE_IDS = frozenset({"egoverse_eva", "egoverse_rl2_eva"})
            is_ego = (
                (ds_name.startswith("egoverse_") and ds_name not in ROBOT_EGOVERSE_IDS)
                or (ds_name.startswith("aligned_") and "_human" in ds_name)
            )
            data["domain_mask"] = np.asarray(is_ego, dtype=bool)
            # OT group for AtomAligned bridges
            if ds_name == "aligned_hangzhou_human_right":
                ot_g = 1
            elif ds_name == "aligned_hangzhou_robot_right":
                ot_g = 2
            elif ds_name == "aligned_shenzhen_human_bimanual":
                ot_g = 3
            elif ds_name == "aligned_shenzhen_robot_bimanual":
                ot_g = 4
            else:
                ot_g = 0
            data["ot_group"] = np.asarray(ot_g, dtype=np.int32)
            stats = self.norm_stats_by_dataset.get(ds_name)
            if stats:
                # Stats are computed at NATIVE dim (e.g. 14); but in the train/val pipeline the
                # state/action are padded to the model action_dim (e.g. 40) so heterogeneous
                # datasets can be batched together. openpi's Normalize only slices stats DOWN to
                # the data width, never up -> pad each NormStats up to the data width with neutral
                # values (mean 0 / std 1 / q01 -1 / q99 1) so padded dims normalize to ~0.
                stats = self._pad_stats_to_data(stats, data)
                data = _transforms.Normalize(stats, use_quantiles=self.use_quantiles)(data)
        else:
            data["domain_mask"] = np.asarray(False, dtype=bool)
            data["ot_group"] = np.asarray(0, dtype=np.int32)
        return data

    @staticmethod
    def _pad_stats_to_data(stats: dict, data: dict) -> dict:
        out = {}
        for key, ns in stats.items():
            mean = np.asarray(ns.mean)
            target = np.asarray(data[key]).shape[-1] if key in data else mean.shape[-1]
            pad = target - mean.shape[-1]
            if pad <= 0:
                out[key] = ns
                continue
            out[key] = _normalize.NormStats(
                mean=np.concatenate([mean, np.zeros(pad, mean.dtype)], axis=-1),
                std=np.concatenate([np.asarray(ns.std), np.ones(pad, np.asarray(ns.std).dtype)], axis=-1),
                q01=(None if ns.q01 is None else np.concatenate([np.asarray(ns.q01), -np.ones(pad)], axis=-1)),
                q99=(None if ns.q99 is None else np.concatenate([np.asarray(ns.q99), np.ones(pad)], axis=-1)),
            )
        return out