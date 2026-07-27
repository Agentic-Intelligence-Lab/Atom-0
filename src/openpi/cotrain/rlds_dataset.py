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

import tqdm

from openpi.cotrain import action_space as cotrain_action_space
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
    # Native action width, used for legacy prefix masks and restoring model outputs. Unified
    # datasets derive their train/eval mask from unified_action_spec instead.
    action_dim: int = 0
    # Optional index selections applied after restructure and before padding/chunking. These
    # let schema-rich datasets (notably RoboCOIN) crop/reorder raw proprio state into the same
    # semantic order as action, and drop unnamed action tail dims when metadata cannot identify
    # them.
    state_indices: tuple[int, ...] | None = None
    action_indices: tuple[int, ...] | None = None
    # Optional source camera keys for datasets whose camera names vary per TFDS builder.
    # Order is (base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb); None means fill a blank
    # image and set that slot's mask to False.
    camera_keys: tuple[str | None, str | None, str | None] | None = None
    # Args to `make_bool_mask` selecting which action dims become deltas (relative to current
    # state) for absolute-action datasets. None -> keep absolute. E.g. RoboMIND (dual ALOHA,
    # absolute joint): (6, -1, 6, -1) = 6 joints delta + gripper absolute, per arm.
    delta_action_mask_dims: tuple[int, ...] | None = None
    # Optional source-to-80D mapping. When present, the loader scatters state/action into
    # fixed unified slots instead of selecting native dims and padding them as a prefix.
    unified_action_spec: cotrain_action_space.UnifiedActionSpec | None = None
    # Full path to the TFDS *version* directory (the dir containing dataset_info.json /
    # features.json, e.g. ".../egoverse_infidata/1.0.0"). When set, the loader uses
    # tfds.builder_from_directory(builder_dir) directly -- this sidesteps the single global
    # data_dir and the tfds name<->dir matching (needed because several datasets live under
    # different parent dirs and even share a tfds `name`). When None, falls back to the legacy
    # tfds.builder(name, data_dir, version) lookup.
    builder_dir: str | None = None
    # Unique logical id for this dataset entry. MUST be unique across the mixture (the tfds
    # `name` is NOT, e.g. eva & mecka are both "egoverse_infidata"). Injected as `dataset_id`
    # for per-dataset normalization / delta dispatch, and used as the norm-stats subdir and the
    # key for per-dataset val loaders / weights / action dims. Defaults to `name`.
    dataset_id: str = ""
    # True when restructure emits an already time-aligned action chunk [T, source_horizon, D].
    # The loader uniformly resamples that source horizon to the model horizon instead of
    # gathering consecutive trajectory frames a second time.
    precomputed_action_chunk: bool = False
    # Auditable description of the precomputed source. These fields are used by
    # staged-run preflight checks; they do not alter the runtime tensor path.
    precomputed_action_source: str | None = None
    precomputed_action_horizon: int | None = None

    @property
    def uid(self) -> str:
        return self.dataset_id or self.name

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
        "prompt_prefix": tf.fill([traj_len], _action_prompt_prefix("joint")),
        "step_id": step_id,
        "passes_filter": passes_filter,
    }


# Registry: restructure_name -> fn(traj, action_space, filter_table) -> restructured traj.
RESTRUCTURE_FNS = {
    "droid": _droid_restructure,
}


# Canonical image slots in the standardized schema (must match cotrain.transforms._IMAGE_SLOTS).
_STD_IMAGE_SLOTS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def _action_prompt_prefix(action_mode: str, eef_frame=None):
    """Build the action-metadata text prepended before the tokenizer's ``Task:`` text."""
    import tensorflow as tf

    prefix = tf.strings.join(["Action Mode: ", tf.convert_to_tensor(action_mode, tf.string), ". "])
    if eef_frame is None:
        return prefix

    frame = tf.strings.strip(tf.convert_to_tensor(eef_frame, tf.string))
    return tf.strings.join([prefix, "EEF Frame: ", frame, ". "])


def _fill_action_prompt_prefix(n, action_mode: str, eef_frame=None):
    import tensorflow as tf

    return tf.fill([n], _action_prompt_prefix(action_mode, eef_frame))


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


def _aligned_parallel_gripper_restructure(traj, dataset_name: str):
    """Aligned human/robot play data with a shared 14D EEF+gripper interface.

    This uses the same image/prompt fields as ``standardized`` and requires:

      state[T,14]   = [L xyz, L ypr, L grip, R xyz, R ypr, R grip]
      actions[T,14] = [L absolute xyz, L absolute ypr, L grip,
                       R absolute xyz, R absolute ypr, R grip]

    EEF poses are kept in the source dataset's documented frame and are not
    differenced, matching the project's final unified-action-space design.
    Gripper values are absolute, normalized to 0=open and 1=closed.
    ``eef_frame`` is a scalar episode string included in prompt metadata.
    """
    import tensorflow as tf

    n = tf.shape(traj["actions"])[0]
    tf.debugging.assert_equal(tf.shape(traj["state"])[-1], 14)
    tf.debugging.assert_equal(tf.shape(traj["actions"])[-1], 14)
    eef_frame = traj.get("eef_frame", tf.constant("chunk_start_local"))
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
        "prompt_prefix": _fill_action_prompt_prefix(n, "eef", eef_frame),
        "dataset_id": tf.fill([n], dataset_name),
    }


def _egomimic_restructure(traj, dataset_name: str):
    """EgoMimic RLDS converted from the public robomimic-style HDF5 files.

    The offline converter writes the common image/state fields and a rank-three
    ``actions[T,100,D]`` tensor. Human source width is 3 (right-hand XYZ) or 6
    (left/right XYZ). Robot source width is 10 (right 6 joints + gripper + XYZ)
    or 20 (bimanual joints/grippers + left/right XYZ).

    EgoMimic XYZ values are already expressed in the current egocentric camera
    frame. No Euler orientation or human gripper label exists in the public
    data, so the registered 80D mapping leaves those slots masked out.
    """
    import tensorflow as tf

    n = tf.shape(traj["actions"])[0]
    is_human = dataset_name.endswith("_human")
    action_mode = "eef_xyz" if is_human else "joint_gripper_and_eef_xyz"
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
        "prompt_prefix": _fill_action_prompt_prefix(n, action_mode, "current_egocentric_camera"),
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
        "prompt_prefix": _fill_action_prompt_prefix(n, "joint"),
        "dataset_id": tf.fill([n], dataset_name),
    }


def _three_cam_task_restructure(traj, dataset_id: str):
    """Generic dual-arm, 3-camera infidata schema (high + left/right wrist), prompt = `task`.

    Covers realworld_piper (14-dim joint) and RoboCOIN (36-dim joint). Identical in shape to
    `_robomind_restructure`; kept separate only so the registry name documents the source.
    The native action dim and the absolute->delta mask are set per-dataset in the config, not
    here -- this fn just maps raw fields to the common nested keys and injects `dataset_id`.
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
        "prompt_prefix": _fill_action_prompt_prefix(n, "joint"),
        "dataset_id": tf.fill([n], dataset_id),
    }


def _piper2_restructure(traj, dataset_id: str):
    """Map the second real-world Piper RLDS drop into the common co-training schema.

    The actual builder metadata declares both state and action as
    ``left_joint_1..6, left_gripper, right_joint_1..6, right_gripper``. Empirically,
    ``action[t]`` equals ``state[t + 1]`` exactly, so this function preserves the raw
    absolute targets; the unified-action transform later converts only the 12 arm-joint
    slots to deltas. The two gripper slots stay absolute.

    Four cameras are present. The canonical three model slots use the head/high camera and
    both wrist cameras; ``cam_front`` is intentionally unused, matching the original Piper
    adapter. ``task`` is the per-step language instruction.
    """
    import tensorflow as tf

    actions = tf.ensure_shape(traj["action"], [None, 14])
    state = tf.ensure_shape(traj["observation"]["state"], [None, 14])
    n = tf.shape(actions)[0]
    true_mask = tf.fill([n], True)
    imgs = traj["observation"]["images"]
    return {
        "actions": actions,
        "state": state,
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
        "prompt_prefix": _fill_action_prompt_prefix(n, "joint"),
        "dataset_id": tf.fill([n], dataset_id),
    }


def _agibot_restructure(traj, dataset_id: str):
    """AgiBotWorld beta mobile dual-arm schema -> common co-training keys.

    The dlimp RLDS loader keeps image features encoded, so this can feed the
    common post-shuffle decode path directly. Actions/state are 20-dim:
        joint14 + effector2 + head2 + waist2.
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
        "prompt_prefix": _fill_action_prompt_prefix(n, "joint"),
        "dataset_id": tf.fill([n], dataset_id),
    }


def _egoverse_eva_restructure(traj, dataset_id: str):
    """EgoVerse eva (bimanual robot teleop): 12-dim absolute cartesian EE pose, 3 cameras.

    base = front_1, plus real left/right wrist cameras. prompt = `prompt` (not `task`).
    """
    import tensorflow as tf

    n = tf.shape(traj["action"])[0]
    true_mask = tf.fill([n], True)
    imgs = traj["observation"]["images"]
    return {
        "actions": traj["action"],
        "state": traj["observation"]["state"],
        "image": {
            "base_0_rgb": imgs["front_1"],
            "left_wrist_0_rgb": imgs["left_wrist"],
            "right_wrist_0_rgb": imgs["right_wrist"],
        },
        "image_mask": {
            "base_0_rgb": true_mask,
            "left_wrist_0_rgb": true_mask,
            "right_wrist_0_rgb": true_mask,
        },
        "prompt": traj["prompt"],
        "prompt_prefix": _action_prompt_prefix("eef", traj["cartesian_frame"]),
        "dataset_id": tf.fill([n], dataset_id),
    }


def _egoverse_mecka_restructure(traj, dataset_id: str):
    """EgoVerse mecka (human egocentric): 12-dim absolute cartesian EE pose, ONLY front_1 cam.

    The two wrist slots are filled with a blank JPEG and masked OFF (image_mask=False) so the
    model does not attend to non-existent cameras. front_1 is 360x640 (resized downstream).
    """
    import tensorflow as tf

    n = tf.shape(traj["action"])[0]
    true_mask = tf.fill([n], True)
    false_mask = tf.fill([n], False)
    # One blank encoded JPEG, broadcast over the trajectory (decoded after the shuffle buffer
    # like every other slot). Cheap: a single encode op, then tf.fill replicates the bytes.
    blank = tf.fill([n], tf.io.encode_jpeg(tf.zeros([360, 640, 3], tf.uint8)))
    return {
        "actions": traj["action"],
        "state": traj["observation"]["state"],
        "image": {
            "base_0_rgb": traj["observation"]["images"]["front_1"],
            "left_wrist_0_rgb": blank,
            "right_wrist_0_rgb": blank,
        },
        "image_mask": {
            "base_0_rgb": true_mask,
            "left_wrist_0_rgb": false_mask,
            "right_wrist_0_rgb": false_mask,
        },
        "prompt": traj["prompt"],
        "prompt_prefix": _action_prompt_prefix("eef", traj["cartesian_frame"]),
        "dataset_id": tf.fill([n], dataset_id),
    }


def _egoverse_full_restructure(traj, dataset_id: str):
    """EgoVerse_full schema: 12-dim absolute cartesian EE pose, front camera plus optional wrists.

    The full EgoVerse drop is split into multiple TFDS builder dirs with the same TFDS name.
    Some subsets provide only `front_1`; eva additionally provides left/right wrist cameras.
    Missing wrist slots are filled with a blank JPEG and masked off.
    """
    import tensorflow as tf

    n = tf.shape(traj["action"])[0]
    true_mask = tf.fill([n], True)
    false_mask = tf.fill([n], False)
    imgs = traj["observation"]["images"]

    if "left_wrist" in imgs and "right_wrist" in imgs:
        left_wrist = imgs["left_wrist"]
        right_wrist = imgs["right_wrist"]
        left_mask = true_mask
        right_mask = true_mask
    else:
        blank = tf.fill([n], tf.io.encode_jpeg(tf.zeros([360, 640, 3], tf.uint8)))
        left_wrist = blank
        right_wrist = blank
        left_mask = false_mask
        right_mask = false_mask

    return {
        "actions": traj["action"],
        "state": traj["observation"]["state"],
        "image": {
            "base_0_rgb": imgs["front_1"],
            "left_wrist_0_rgb": left_wrist,
            "right_wrist_0_rgb": right_wrist,
        },
        "image_mask": {
            "base_0_rgb": true_mask,
            "left_wrist_0_rgb": left_mask,
            "right_wrist_0_rgb": right_mask,
        },
        "prompt": traj["prompt"],
        "prompt_prefix": _action_prompt_prefix("eef", traj["cartesian_frame"]),
        "dataset_id": tf.fill([n], dataset_id),
    }


def _egoverse_cartesian_chunk_restructure(traj, dataset_id: str):
    """EgoVerse schema using its official aligned future Cartesian trajectory.

    ``actions_cartesian[t]`` is a 100-step future trajectory aligned to frame
    ``t``. Its first element equals ``action[t]``. Keeping this as rank three
    prevents the generic loader from incorrectly rebuilding the horizon from
    adjacent episode frames, whose poses may use different moving head frames.
    """
    output = _egoverse_full_restructure(traj, dataset_id)
    output["actions"] = traj["actions_cartesian"]
    return output


def _robocoin_restructure(traj, dataset_id: str):
    """RoboCOIN schema -> common co-training keys.

    RoboCOIN_full is split into many robot-schema TFDS builders. They share the same basic
    fields (`action`, `observation/state`, `observation/images`, `task`), but a few subsets
    only have `cam_high`. Missing wrist slots are filled with a blank JPEG and masked off.
    """
    import tensorflow as tf

    n = tf.shape(traj["action"])[0]
    true_mask = tf.fill([n], True)
    false_mask = tf.fill([n], False)
    imgs = traj["observation"]["images"]

    if "cam_left_wrist" in imgs and "cam_right_wrist" in imgs:
        left_wrist = imgs["cam_left_wrist"]
        right_wrist = imgs["cam_right_wrist"]
        left_mask = true_mask
        right_mask = true_mask
    else:
        blank = tf.fill([n], tf.io.encode_jpeg(tf.zeros([480, 640, 3], tf.uint8)))
        left_wrist = blank
        right_wrist = blank
        left_mask = false_mask
        right_mask = false_mask

    return {
        "actions": traj["action"],
        "state": traj["observation"]["state"],
        "image": {
            "base_0_rgb": imgs["cam_high"],
            "left_wrist_0_rgb": left_wrist,
            "right_wrist_0_rgb": right_wrist,
        },
        "image_mask": {
            "base_0_rgb": true_mask,
            "left_wrist_0_rgb": left_mask,
            "right_wrist_0_rgb": right_mask,
        },
        "prompt": traj["task"],
        "prompt_prefix": _fill_action_prompt_prefix(n, "joint"),
        "dataset_id": tf.fill([n], dataset_id),
    }


def _robomind_full_restructure(
    traj,
    dataset_id: str,
    camera_keys: tuple[str | None, str | None, str | None] | None,
):
    """RoboMIND_full schema -> common co-training keys.

    RoboMIND_full has one TFDS builder per robot/camera layout. State/action are already
    aligned joint-position vectors; camera names vary, so the per-builder config supplies
    the source keys for our three canonical image slots.
    """
    import tensorflow as tf

    if camera_keys is None:
        camera_keys = ("cam_high", "cam_left_wrist", "cam_right_wrist")

    n = tf.shape(traj["action"])[0]
    true_mask = tf.fill([n], True)
    false_mask = tf.fill([n], False)
    blank = tf.fill([n], tf.io.encode_jpeg(tf.zeros([480, 640, 3], tf.uint8)))
    imgs = traj["observation"]["images"]

    def image_or_blank(key):
        if key is None:
            return blank, false_mask
        return imgs[key], true_mask

    base_img, base_mask = image_or_blank(camera_keys[0])
    left_img, left_mask = image_or_blank(camera_keys[1])
    right_img, right_mask = image_or_blank(camera_keys[2])

    return {
        "actions": traj["action"],
        "state": traj["observation"]["state"],
        "image": {
            "base_0_rgb": base_img,
            "left_wrist_0_rgb": left_img,
            "right_wrist_0_rgb": right_img,
        },
        "image_mask": {
            "base_0_rgb": base_mask,
            "left_wrist_0_rgb": left_mask,
            "right_wrist_0_rgb": right_mask,
        },
        "prompt": traj["task"],
        "prompt_prefix": _fill_action_prompt_prefix(n, "joint"),
        "dataset_id": tf.fill([n], dataset_id),
    }


# Standardized-style restructures: signature (traj, dataset_id) -> common nested schema.
# All feed the same prepare path (chunk + decode). The images they emit are encoded; the
# prepare path decodes them. Add new clean datasets here.
STD_RESTRUCTURE_FNS = {
    "aligned_parallel_gripper": _aligned_parallel_gripper_restructure,
    "egomimic": _egomimic_restructure,
    "standardized": _standardized_restructure,
    "agibot": _agibot_restructure,
    "robomind": _robomind_restructure,
    "three_cam_task": _three_cam_task_restructure,  # realworld_piper, RoboCOIN
    "piper2": _piper2_restructure,
    "egoverse_eva": _egoverse_eva_restructure,
    "egoverse_mecka": _egoverse_mecka_restructure,
    "egoverse_full": _egoverse_full_restructure,
    "egoverse_cartesian_chunk": _egoverse_cartesian_chunk_restructure,
    "robocoin": _robocoin_restructure,
    "robomind_full": _robomind_full_restructure,
}


def resample_precomputed_action_chunk(traj: dict, action_chunk_size: int) -> dict:
    """Uniformly resample ``[T, source_horizon, D]`` actions to model horizon.

    EgoVerse stores 100 samples over its complete physical prediction window.
    For pi0's 50-step horizon we retain both endpoints and sample across that
    entire window, rather than taking only its first half.
    """
    if action_chunk_size <= 0:
        raise ValueError(f"action_chunk_size must be positive, got {action_chunk_size}")
    import tensorflow as tf

    actions = tf.convert_to_tensor(traj["actions"])
    tf.debugging.assert_rank(
        actions,
        3,
        message="precomputed_action_chunk requires actions shaped [T, source_horizon, D]",
    )
    source_horizon = tf.shape(actions)[1]
    tf.debugging.assert_positive(source_horizon, message="precomputed action horizon must be non-empty")
    indices = tf.cast(
        tf.round(tf.linspace(0.0, tf.cast(source_horizon - 1, tf.float32), action_chunk_size)),
        tf.int32,
    )
    traj["actions"] = tf.gather(actions, indices, axis=1)
    traj["actions"].set_shape([actions.shape[0], action_chunk_size, actions.shape[-1]])
    return traj


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
        # Legacy configs zero-pad native vectors to this width. Unified configs require width 80
        # and map before mixing/batching. None keeps the mapped/native width for norm-stat jobs.
        pad_action_dim: int | None = None,
        # If set (h, w), resize_with_pad all decoded images to this size BEFORE mixing/batching,
        # so datasets with different native resolutions (e.g. mecka 360x640 vs others 480x640)
        # share one element spec. Match the model's ResizeImages target (224x224) so the later
        # model-transform resize is idempotent. None -> keep native (norm-stats is single-dataset
        # so its images are already uniform and need no resize).
        image_resize_hw: tuple[int, int] | None = None,
        # Multi-host (JAX distributed / DLC): each process reads a DIFFERENT 1/process_count
        # slice of every dataset's split (via tfds.even_splits), so data-parallel hosts see
        # disjoint data. With process_count=1 this is a no-op (single-host behavior unchanged).
        process_count: int = 1,
        process_index: int = 0,
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

        def _pad_state_actions(traj):
            # state/actions are [T, D] here (pre-chunk); pad the last dim up to pad_action_dim.
            def _pad2d(x):
                cur = tf.shape(x)[-1]
                return tf.pad(x, [[0, 0], [0, tf.maximum(pad_action_dim - cur, 0)]])

            traj["state"] = _pad2d(traj["state"])
            traj["actions"] = _pad2d(traj["actions"])
            return traj

        def _select_state_actions(traj, dataset_cfg: CotrainRLDSDataset):
            if dataset_cfg.state_indices is not None:
                traj["state"] = tf.gather(traj["state"], tf.constant(dataset_cfg.state_indices, tf.int32), axis=-1)
            if dataset_cfg.action_indices is not None:
                traj["actions"] = tf.gather(traj["actions"], tf.constant(dataset_cfg.action_indices, tf.int32), axis=-1)
            return traj

        def decode_std_images(frame):
            for slot in _STD_IMAGE_SLOTS:
                img = tf.io.decode_image(frame["image"][slot], expand_animations=False, dtype=tf.uint8)
                if image_resize_hw is not None:
                    # resize_with_pad preserves aspect ratio (pads), matching the model's
                    # ResizeImages; cast back to uint8 (resize returns float32).
                    img = tf.cast(
                        tf.round(tf.image.resize_with_pad(img, image_resize_hw[0], image_resize_hw[1])),
                        tf.uint8,
                    )
                frame["image"][slot] = img
            return frame

        def _prepare_standardized(dataset, dataset_cfg: CotrainRLDSDataset):
            # Standardized-style schema (incl. RoboMIND): no DROID-specific success filter /
            # step_id / filter_dict. The restructure maps raw fields -> common nested keys.
            # NOTE: images are left ENCODED here; they are decoded AFTER the shuffle buffer
            # (see below) so the buffer holds small encoded bytes, not huge raw frames.
            restructure_fn = STD_RESTRUCTURE_FNS[dataset_cfg.restructure_name]
            if repeat:
                dataset = dataset.repeat()
            if dataset_cfg.restructure_name == "robomind_full":
                dataset = dataset.traj_map(
                    lambda traj: restructure_fn(traj, dataset_cfg.uid, dataset_cfg.camera_keys), num_parallel_calls
                )
            else:
                dataset = dataset.traj_map(lambda traj: restructure_fn(traj, dataset_cfg.uid), num_parallel_calls)
            if dataset_cfg.unified_action_spec is not None:
                if pad_action_dim is not None and pad_action_dim != cotrain_action_space.UNIFIED_ACTION_DIM:
                    raise ValueError(
                        f"Unified dataset '{dataset_cfg.uid}' requires model action_dim="
                        f"{cotrain_action_space.UNIFIED_ACTION_DIM}, got {pad_action_dim}."
                    )
                dataset = dataset.traj_map(
                    lambda traj: cotrain_action_space.map_trajectory_tensorflow(traj, dataset_cfg.unified_action_spec),
                    num_parallel_calls,
                )
            elif dataset_cfg.state_indices is not None or dataset_cfg.action_indices is not None:
                dataset = dataset.traj_map(lambda traj: _select_state_actions(traj, dataset_cfg), num_parallel_calls)
            # Pad native state/action to the model width BEFORE chunk/mix/batch (if requested),
            # so heterogeneous-dim datasets share one element spec.
            if pad_action_dim is not None and dataset_cfg.unified_action_spec is None:
                dataset = dataset.traj_map(_pad_state_actions, num_parallel_calls)
            if dataset_cfg.precomputed_action_chunk:
                dataset = dataset.traj_map(
                    lambda traj: resample_precomputed_action_chunk(traj, action_chunk_size),
                    num_parallel_calls,
                )
            else:
                dataset = dataset.traj_map(_chunk_actions, num_parallel_calls)
            return dataset.flatten(num_parallel_calls=num_parallel_calls)

        def prepare_single_dataset(dataset_cfg: CotrainRLDSDataset):
            split_name = dataset_cfg.resolve_split(split_label)
            # Multi-host: give each process a disjoint 1/process_count slice of this split so
            # data-parallel hosts don't read identical data. even_splits partitions by episode.
            if process_count > 1:
                split_name = tfds.even_splits(split_name, n=process_count)[process_index]
            # Prefer an explicit version-dir (handles datasets under different parent dirs and
            # datasets that share a tfds `name`); fall back to the global data_dir lookup.
            if dataset_cfg.builder_dir is not None:
                builder = tfds.builder_from_directory(dataset_cfg.builder_dir)
            else:
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
        # drop_remainder=True so EVERY yielded batch is exactly batch_size. Required for multi-host:
        # the global batch (local_batch_size * process_count) must stay divisible by the device mesh
        # (16 here); otherwise val's finite final partial batch (e.g. 116) fails make_array_from_process_local_data.
        # Train (repeat=True) is an infinite stream so it never hits a partial batch anyway.
        final_dataset = final_dataset.batch(batch_size, drop_remainder=True)
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
