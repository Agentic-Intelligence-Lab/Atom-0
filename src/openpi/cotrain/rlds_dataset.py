"""Multi-dataset RLDS loader with per-dataset train/val splits.

Forked from `openpi.training.droid_rlds_dataset.DroidRldsDataset` and generalized:

  * Each dataset entry (`CotrainRLDSDataset`) carries its own `train_split` /
    `val_split` TFDS split names, so the train/val split decision lives in the data
    (baked at RLDS-generation time) rather than being decided at train time.
  * A `split` argument selects which split to materialize.
  * Validation pipelines do NOT `.repeat()` and (by default) do NOT shuffle, so the
    metric is computed over a deterministic, finite set of batches.
  * `restructure` is looked up from a per-dataset registry keyed by `restructure_name`,
    so heterogeneous schemas can be added later without touching this file's core.

For now only the DROID schema restructure is registered (`"droid"`). Adding a new
dataset schema = register a new restructure fn in `RESTRUCTURE_FNS`.
"""

from collections.abc import Sequence
import dataclasses
import json
import logging
from pathlib import Path
from typing import Literal

import tqdm

import openpi.shared.download as download

# Reuse the action-space enum unchanged from the original DROID loader.
from openpi.training.droid_rlds_dataset import DroidActionSpace

Split = str  # a TFDS split label: "train" or any key of `val_splits` (e.g. "seen", "unseen")


def _default_val_splits() -> dict:
    return {"val": "val"}


@dataclasses.dataclass(frozen=True)
class CotrainRLDSDataset:
    """One dataset in the co-training mixture.

    `train_split` is the TFDS split used for training. `val_splits` maps a logical label
    (e.g. "seen", "unseen") to the TFDS split name (e.g. "seen_test", "unseen_test"), so a
    dataset can expose multiple validation sets. These splits must already exist in the
    built RLDS (the split decision is made once, at generation/collection time).
    """

    name: str
    version: str
    weight: float
    train_split: str = "train"
    # label -> TFDS split name. E.g. RoboMIND: {"seen": "seen_test", "unseen": "unseen_test"}.
    val_splits: dict = dataclasses.field(default_factory=_default_val_splits)
    filter_dict_path: str | None = None
    # Which restructure to use: "standardized" (offline common schema), "robomind" (raw
    # RoboMIND schema, mapped at runtime), or "droid" (raw DROID schema).
    restructure_name: str = "standardized"
    # Native (un-padded) action dimensionality, used for the per-dataset action-MSE mask
    # and for slicing model outputs back to native dims at inference. 0 -> use all dims.
    action_dim: int = 0
    # Args to `make_bool_mask` selecting which action dims become deltas (relative to current
    # state) for absolute-action datasets. None -> keep absolute. E.g. RoboMIND (dual ALOHA,
    # absolute joint): (6, -1, 6, -1) = 6 joints delta + gripper absolute, per arm.
    delta_action_mask_dims: tuple | None = None

    def resolve_split(self, label: str) -> str:
        """Resolve a split label to the underlying TFDS split name."""
        if label == "train":
            return self.train_split
        if label in self.val_splits:
            return self.val_splits[label]
        raise KeyError(f"Dataset '{self.name}' has no val split labeled '{label}' (have {list(self.val_splits)})")

    def val_labels(self) -> list:
        return list(self.val_splits)


def _droid_restructure(traj, action_space: DroidActionSpace, filter_table):
    """DROID-schema restructure (identical logic to the original DROID loader)."""
    import tensorflow as tf

    actions = tf.concat(
        (
            (
                traj["action_dict"]["joint_position"]
                if action_space == DroidActionSpace.JOINT_POSITION
                else traj["action_dict"]["joint_velocity"]
            ),
            traj["action_dict"]["gripper_position"],
        ),
        axis=-1,
    )
    exterior_img = tf.cond(
        tf.random.uniform(shape=[]) > 0.5,
        lambda: traj["observation"]["exterior_image_1_left"],
        lambda: traj["observation"]["exterior_image_2_left"],
    )
    wrist_img = traj["observation"]["wrist_image_left"]
    instruction = tf.random.shuffle(
        [traj["language_instruction"], traj["language_instruction_2"], traj["language_instruction_3"]]
    )[0]

    traj_len = tf.shape(traj["action"])[0]
    indices = tf.as_string(tf.range(traj_len))
    step_id = (
        traj["traj_metadata"]["episode_metadata"]["recording_folderpath"]
        + "--"
        + traj["traj_metadata"]["episode_metadata"]["file_path"]
        + "--"
        + indices
    )
    passes_filter = filter_table.lookup(step_id)

    return {
        "actions": actions,
        "observation": {
            "image": exterior_img,
            "wrist_image": wrist_img,
            "joint_position": traj["observation"]["joint_position"],
            "gripper_position": traj["observation"]["gripper_position"],
        },
        "prompt": instruction,
        "step_id": step_id,
        "passes_filter": passes_filter,
    }


# Registry: restructure_name -> fn(traj, action_space, filter_table) -> restructured traj.
RESTRUCTURE_FNS = {
    "droid": _droid_restructure,
}


# Canonical image slots in the standardized schema (must match cotrain.transforms._IMAGE_SLOTS).
_STD_IMAGE_SLOTS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def _standardized_restructure(traj, dataset_name: str):
    """Restructure for the common (offline-standardized) co-training schema.

    Contract -- the offline RLDS generation MUST write, per frame (leading time dim T):
        state:              float32[T, Ds]   native proprio (un-padded, un-normalized)
        actions:            float32[T, Da]   native per-frame action (un-padded, un-normalized)
        image_base:         encoded image    (required)
        image_left_wrist:   encoded image    (placeholder zeros if camera absent)
        image_right_wrist:  encoded image    (placeholder zeros if camera absent)
        image_mask_base / _left_wrist / _right_wrist:  bool[T]
        prompt:             string[T]

    `dataset_id` is injected here as a constant (= dataset name), so it does NOT need to be
    stored in the data. It is used downstream by DispatchNormalize to pick per-dataset stats.
    """
    import tensorflow as tf

    n = tf.shape(traj["actions"])[0]
    return {
        "actions": traj["actions"],
        "state": traj["state"],
        "image": {
            "base_0_rgb": traj["image_base"],
            "left_wrist_0_rgb": traj["image_left_wrist"],
            "right_wrist_0_rgb": traj["image_right_wrist"],
        },
        "image_mask": {
            "base_0_rgb": traj["image_mask_base"],
            "left_wrist_0_rgb": traj["image_mask_left_wrist"],
            "right_wrist_0_rgb": traj["image_mask_right_wrist"],
        },
        "prompt": traj["prompt"],
        "dataset_id": tf.fill([n], dataset_name),
    }


def _robomind_restructure(traj, dataset_name: str):
    """Map the raw RoboMIND (robomind_infidata) RLDS schema -> common co-training keys.

    RoboMIND is already a clean RLDS, so no offline regeneration is needed -- this runs at
    load time. It is dual-arm: 14-dim state/action kept NATIVE (no remap; per-dataset
    normalization handles scale). Three cameras map directly to our canonical slots.

    Raw fields (after dlimp from_rlds lifts `steps` to top level):
        action                -> actions   [T, 14]
        observation/state     -> state     [T, 14]
        observation/images/cam_high        -> base_0_rgb
        observation/images/cam_left_wrist  -> left_wrist_0_rgb
        observation/images/cam_right_wrist -> right_wrist_0_rgb
        task (text)           -> prompt
    """
    import tensorflow as tf

    n = tf.shape(traj["action"])[0]
    true_mask = tf.fill([n], True)
    imgs = traj["observation"]["images"]
    return {
        "actions": traj["action"],
        "state": traj["observation"]["state"],
        "image": {
            "base_0_rgb": imgs["cam_high"],
            "left_wrist_0_rgb": imgs["cam_left_wrist"],
            "right_wrist_0_rgb": imgs["cam_right_wrist"],
        },
        "image_mask": {
            "base_0_rgb": true_mask,
            "left_wrist_0_rgb": true_mask,
            "right_wrist_0_rgb": true_mask,
        },
        "prompt": traj["task"],
        "dataset_id": tf.fill([n], dataset_name),
    }


# Standardized-style restructures: signature (traj, dataset_name) -> common nested schema.
# All feed the same prepare path (chunk + decode). The images they emit are encoded; the
# prepare path decodes them. Add new clean datasets (e.g. "agibot", "egoverse") here.
STD_RESTRUCTURE_FNS = {
    "standardized": _standardized_restructure,
    "robomind": _robomind_restructure,
}


class CotrainRldsDataset:
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        datasets: Sequence[CotrainRLDSDataset],
        *,  # Force keyword-only arguments
        split_label: str = "train",
        shuffle: bool = True,
        repeat: bool | None = None,
        action_chunk_size: int = 16,
        action_space: DroidActionSpace = DroidActionSpace.JOINT_POSITION,
        max_loaded_steps_per_episode: int = 100,
        shuffle_buffer_size: int = 250_000,
        num_parallel_reads: int = -1,  # -1 == tf.data.AUTOTUNE
        num_parallel_calls: int = -1,  # -1 == tf.data.AUTOTUNE
    ):
        import dlimp as dl
        import tensorflow as tf
        import tensorflow_datasets as tfds

        tf.config.set_visible_devices([], "GPU")

        is_train = split_label == "train"
        # Validation defaults to a finite, deterministic pass: no repeat.
        if repeat is None:
            repeat = is_train

        # Mixture weights only need to sum to 1.0 for the (train) sampling step. For a
        # single-dataset validation loader this is trivially satisfied (weight == 1.0).
        assert abs(sum(d.weight for d in datasets) - 1.0) < 1e-6, "Dataset weights must sum to 1.0"

        def _chunk_actions(traj):
            traj_len = tf.shape(traj["actions"])[0]
            action_chunk_indices = tf.broadcast_to(
                tf.range(action_chunk_size)[None],
                [traj_len, action_chunk_size],
            ) + tf.broadcast_to(
                tf.range(traj_len)[:, None],
                [traj_len, action_chunk_size],
            )
            # Cap to length of the sequence -> final chunks repeat the last action.
            action_chunk_indices = tf.minimum(action_chunk_indices, traj_len - 1)
            traj["actions"] = tf.gather(traj["actions"], action_chunk_indices)
            return traj

        def decode_std_images(frame):
            for slot in _STD_IMAGE_SLOTS:
                frame["image"][slot] = tf.io.decode_image(
                    frame["image"][slot], expand_animations=False, dtype=tf.uint8
                )
            return frame

        def _prepare_standardized(dataset, dataset_cfg: CotrainRLDSDataset):
            # Standardized-style schema (incl. RoboMIND): no DROID-specific success filter /
            # step_id / filter_dict. The restructure maps raw fields -> common nested keys.
            # NOTE: images are left ENCODED here; they are decoded AFTER the shuffle buffer
            # (see below) so the buffer holds small encoded bytes, not huge raw frames.
            restructure_fn = STD_RESTRUCTURE_FNS[dataset_cfg.restructure_name]
            if repeat:
                dataset = dataset.repeat()
            dataset = dataset.traj_map(
                lambda traj: restructure_fn(traj, dataset_cfg.name), num_parallel_calls
            )
            dataset = dataset.traj_map(_chunk_actions, num_parallel_calls)
            return dataset.flatten(num_parallel_calls=num_parallel_calls)

        def prepare_single_dataset(dataset_cfg: CotrainRLDSDataset):
            split_name = dataset_cfg.resolve_split(split_label)
            builder = tfds.builder(dataset_cfg.name, data_dir=data_dir, version=dataset_cfg.version)
            dataset = dl.DLataset.from_rlds(
                builder, split=split_name, shuffle=shuffle, num_parallel_reads=num_parallel_reads
            )

            if dataset_cfg.restructure_name in STD_RESTRUCTURE_FNS:
                return _prepare_standardized(dataset, dataset_cfg)

            # --- Legacy DROID-schema path below ---
            # Filter out any unsuccessful trajectories -- we use the file name to check this.
            dataset = dataset.filter(
                lambda traj: tf.strings.regex_full_match(
                    traj["traj_metadata"]["episode_metadata"]["file_path"][0], ".*success.*"
                )
            )

            # Only repeat for training; validation should terminate so it is a finite eval set.
            if repeat:
                dataset = dataset.repeat()

            # Optional per-frame filter dictionary (episode key -> kept frame ranges).
            filter_dict_path = dataset_cfg.filter_dict_path
            if filter_dict_path is not None:
                cached_filter_dict_path = download.maybe_download(filter_dict_path)
                with Path(cached_filter_dict_path).open("r") as f:
                    filter_dict = json.load(f)
                logging.info(f"Using filter dictionary with {len(filter_dict)} episodes")

                keys_tensor = []
                values_tensor = []
                for episode_key, ranges in tqdm.tqdm(filter_dict.items(), desc="Creating idle filter hash table..."):
                    for start, end in ranges:
                        for t in range(start, end):
                            keys_tensor.append(f"{episode_key}--{t}")
                            values_tensor.append(True)
                filter_table = tf.lookup.StaticHashTable(
                    tf.lookup.KeyValueTensorInitializer(keys_tensor, values_tensor), default_value=False
                )
                logging.info("Filter hash table initialized")
            else:
                filter_table = tf.lookup.StaticHashTable(
                    tf.lookup.KeyValueTensorInitializer([""], [True]), default_value=True
                )

            restructure_fn = RESTRUCTURE_FNS[dataset_cfg.restructure_name]

            def restructure(traj):
                return restructure_fn(traj, action_space, filter_table)

            dataset = dataset.traj_map(restructure, num_parallel_calls)

            def chunk_actions(traj):
                traj_len = tf.shape(traj["actions"])[0]
                action_chunk_indices = tf.broadcast_to(
                    tf.range(action_chunk_size)[None],
                    [traj_len, action_chunk_size],
                ) + tf.broadcast_to(
                    tf.range(traj_len)[:, None],
                    [traj_len, action_chunk_size],
                )
                # Cap to length of the sequence -> final chunks repeat the last action.
                action_chunk_indices = tf.minimum(action_chunk_indices, traj_len - 1)
                traj["actions"] = tf.gather(traj["actions"], action_chunk_indices)
                return traj

            dataset = dataset.traj_map(chunk_actions, num_parallel_calls)
            dataset = dataset.flatten(num_parallel_calls=num_parallel_calls)

            def filter_from_dict(frame):
                return frame["passes_filter"]

            dataset = dataset.filter(filter_from_dict)

            def remove_passes_filter(frame):
                frame.pop("passes_filter")
                return frame

            dataset = dataset.map(remove_passes_filter)

            def decode_images(traj):
                traj["observation"]["image"] = tf.io.decode_image(
                    traj["observation"]["image"], expand_animations=False, dtype=tf.uint8
                )
                traj["observation"]["wrist_image"] = tf.io.decode_image(
                    traj["observation"]["wrist_image"], expand_animations=False, dtype=tf.uint8
                )
                return traj

            return dataset.frame_map(decode_images, num_parallel_calls)

        logging.info(f"Preparing {len(datasets)} dataset(s) for split_label='{split_label}' (repeat={repeat})...")
        logging.info("-" * 50)
        for d in datasets:
            logging.info(f"    {d.name}:{d.version} [{d.resolve_split(split_label)}] weight={d.weight:.2f}")
        logging.info("-" * 50)

        all_datasets = [prepare_single_dataset(d) for d in datasets]
        weights = [d.weight for d in datasets]

        final_dataset = dl.DLataset.sample_from_datasets(all_datasets, weights=weights)
        # Only shuffle when requested (train). Validation stays deterministic.
        if shuffle:
            final_dataset = final_dataset.shuffle(shuffle_buffer_size)
        # Decode images AFTER the shuffle buffer for standardized-style datasets, so the
        # buffer holds small encoded bytes (not raw uint8 frames -> avoids OOM). The legacy
        # DROID path decodes inside prepare_single_dataset (unchanged).
        std_mode = all(d.restructure_name in STD_RESTRUCTURE_FNS for d in datasets)
        if std_mode:
            final_dataset = final_dataset.frame_map(decode_std_images, num_parallel_calls)
        final_dataset = final_dataset.batch(batch_size)
        final_dataset = final_dataset.with_ram_budget(1)

        self.dataset = final_dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.split_label = split_label
        self.repeat = repeat

    def __iter__(self):
        yield from self.dataset.as_numpy_iterator()

    def __len__(self):
        # Approximate; only TorchDataLoader uses __len__, and we go through RLDSDataLoader.
        return 20_000_000
