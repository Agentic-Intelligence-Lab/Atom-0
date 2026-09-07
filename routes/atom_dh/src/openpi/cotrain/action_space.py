"""Unified 80D state/action layouts for RLDS co-training."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np

UNIFIED_ACTION_DIM = 80

LEFT_ARM = 0
LEFT_EEF_POSITION = 7
LEFT_EEF_EULER = 10
LEFT_GRIPPER = 16
LEFT_HAND = 17
RIGHT_ARM = 29
RIGHT_EEF_POSITION = 36
RIGHT_EEF_EULER = 39
RIGHT_GRIPPER = 45
RIGHT_HAND = 46
LEFT_LEG = 58
RIGHT_LEG = 64
HEAD = 70
WAIST = 72
OTHER_BODY = 74

DimMapping = tuple[tuple[int, int], ...]


def _slot_names() -> tuple[str, ...]:
    names = [f"reserved_{index + 1}" for index in range(UNIFIED_ACTION_DIM)]
    groups = (
        (LEFT_ARM, 7, "left_arm_joint"),
        (LEFT_EEF_POSITION, 3, "left_eef_position"),
        (LEFT_EEF_EULER, 3, "left_eef_euler"),
        (LEFT_GRIPPER, 1, "left_gripper"),
        (LEFT_HAND, 12, "left_hand_joint"),
        (RIGHT_ARM, 7, "right_arm_joint"),
        (RIGHT_EEF_POSITION, 3, "right_eef_position"),
        (RIGHT_EEF_EULER, 3, "right_eef_euler"),
        (RIGHT_GRIPPER, 1, "right_gripper"),
        (RIGHT_HAND, 12, "right_hand_joint"),
        (LEFT_LEG, 6, "left_leg_joint"),
        (RIGHT_LEG, 6, "right_leg_joint"),
        (HEAD, 2, "head_joint"),
        (WAIST, 2, "waist_joint"),
        (OTHER_BODY, 6, "other_body"),
    )
    for start, count, label in groups:
        for offset in range(count):
            names[start + offset] = label if count == 1 else f"{label}_{offset + 1}"
    return tuple(names)


UNIFIED_SLOT_NAMES = _slot_names()


def dims(source_start: int, target_start: int, count: int) -> DimMapping:
    """Map a contiguous source range to a contiguous unified range."""
    return tuple((source_start + i, target_start + i) for i in range(count))


def slots(start: int, count: int) -> tuple[int, ...]:
    return tuple(range(start, start + count))


@dataclasses.dataclass(frozen=True)
class UnifiedActionSpec:
    """Source-to-unified mappings and temporal semantics for one RLDS builder.

    Indices are zero-based. ``absolute_to_delta_slots`` identifies source-absolute
    targets that are converted to deltas after state/action mapping. Slots listed in
    ``already_delta_slots`` are already relative in the source and must never be
    differenced again. All other mapped action slots remain absolute.
    """

    state_mapping: DimMapping
    action_mapping: DimMapping
    absolute_to_delta_slots: tuple[int, ...] = ()
    already_delta_slots: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        self._validate_mapping("state", self.state_mapping)
        self._validate_mapping("action", self.action_mapping)

        action_targets = set(self.action_target_slots)
        state_targets = set(self.state_target_slots)
        absolute_to_delta = set(self.absolute_to_delta_slots)
        already_delta = set(self.already_delta_slots)
        if absolute_to_delta & already_delta:
            raise ValueError("absolute_to_delta_slots and already_delta_slots overlap")
        for name, temporal_slots in (
            ("absolute_to_delta_slots", absolute_to_delta),
            ("already_delta_slots", already_delta),
        ):
            missing_action = temporal_slots - action_targets
            if missing_action:
                raise ValueError(f"{name} contains unmapped action slots: {sorted(missing_action)}")
            missing_state = temporal_slots - state_targets
            if missing_state:
                raise ValueError(f"{name} contains slots without mapped state: {sorted(missing_state)}")

    @staticmethod
    def _validate_mapping(name: str, mapping: DimMapping) -> None:
        sources = [source for source, _ in mapping]
        targets = [target for _, target in mapping]
        if any(index < 0 for index in sources):
            raise ValueError(f"{name} mapping contains a negative source index")
        if any(index < 0 or index >= UNIFIED_ACTION_DIM for index in targets):
            raise ValueError(f"{name} mapping target is outside 0..{UNIFIED_ACTION_DIM - 1}")
        if len(sources) != len(set(sources)):
            raise ValueError(f"{name} mapping contains duplicate source indices")
        if len(targets) != len(set(targets)):
            raise ValueError(f"{name} mapping contains duplicate target slots")

    @property
    def state_target_slots(self) -> tuple[int, ...]:
        return tuple(target for _, target in self.state_mapping)

    @property
    def action_target_slots(self) -> tuple[int, ...]:
        return tuple(target for _, target in self.action_mapping)

    @property
    def action_mask(self) -> tuple[bool, ...]:
        targets = set(self.action_target_slots)
        return tuple(index in targets for index in range(UNIFIED_ACTION_DIM))

    @property
    def delta_mask(self) -> tuple[bool, ...]:
        targets = set(self.absolute_to_delta_slots)
        return tuple(index in targets for index in range(UNIFIED_ACTION_DIM))

    def validate_source_dims(self, state_dim: int, action_dim: int) -> None:
        if self.state_mapping and max(source for source, _ in self.state_mapping) >= state_dim:
            raise ValueError(f"state mapping requires source dim beyond state width {state_dim}")
        if self.action_mapping and max(source for source, _ in self.action_mapping) >= action_dim:
            raise ValueError(f"action mapping requires source dim beyond action width {action_dim}")

    @property
    def fingerprint(self) -> str:
        payload = dataclasses.asdict(self)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


def map_array(array: np.ndarray, mapping: DimMapping) -> np.ndarray:
    """Scatter the final axis of a NumPy array into the unified 80D layout."""
    array = np.asarray(array)
    output = np.zeros((*array.shape[:-1], UNIFIED_ACTION_DIM), dtype=array.dtype)
    if mapping:
        sources, targets = zip(*mapping, strict=True)
        output[..., targets] = array[..., sources]
    return output


def unmap_array(array: np.ndarray, mapping: DimMapping, source_dim: int) -> np.ndarray:
    """Gather unified slots back into their original source indices."""
    array = np.asarray(array)
    output = np.zeros((*array.shape[:-1], source_dim), dtype=array.dtype)
    if mapping:
        sources, targets = zip(*mapping, strict=True)
        output[..., sources] = array[..., targets]
    return output


def apply_delta(state: np.ndarray, actions: np.ndarray, mask) -> np.ndarray:
    """Convert only masked source-absolute slots to deltas against current state."""
    state = np.asarray(state)
    output = np.array(actions, copy=True)
    mask = np.asarray(mask, dtype=bool)
    dims = mask.shape[-1]
    output[..., :dims] -= np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
    return output


# def map_trajectory_tensorflow(traj: dict, spec: UnifiedActionSpec) -> dict:
#     """Map trajectory-level TensorFlow state/actions and attach the action mask."""
#     import tensorflow as tf  # noqa: PLC0415

#     def map_tensor(tensor, mapping: DimMapping):
#         if not mapping:
#             return tf.zeros([tf.shape(tensor)[0], UNIFIED_ACTION_DIM], tensor.dtype)
#         sources, targets = zip(*mapping, strict=True)
#         tf.debugging.assert_less(max(sources), tf.shape(tensor)[-1])
#         selected = tf.gather(tensor, tf.constant(sources, tf.int32), axis=-1)
#         projection = tf.one_hot(targets, UNIFIED_ACTION_DIM, dtype=tensor.dtype)
#         mapped = tf.linalg.matmul(selected, projection)
#         mapped.set_shape([None, UNIFIED_ACTION_DIM])
#         return mapped

#     traj["state"] = map_tensor(traj["state"], spec.state_mapping)
#     traj["actions"] = map_tensor(traj["actions"], spec.action_mapping)
#     traj["action_mask"] = tf.broadcast_to(
#         tf.constant(spec.action_mask, tf.bool),
#         [tf.shape(traj["actions"])[0], UNIFIED_ACTION_DIM],
#     )
#     return traj

def map_trajectory_tensorflow(traj: dict, spec: UnifiedActionSpec) -> dict:
    """Map trajectory-level TensorFlow state/actions and attach the action mask.

    Supports:
      - state/actions rank-2: [T, D]  -> [T, 80]   (robot / open-source)
      - actions rank-3:       [T, H, D] -> [T, H, 80]  (EgoVerse actions_cartesian)
    Scatter is always on the last axis; leading dims are preserved.
    """
    import tensorflow as tf  # noqa: PLC0415

    def map_tensor(tensor, mapping: DimMapping):
        tensor = tf.convert_to_tensor(tensor)
        if not mapping:
            lead = tf.shape(tensor)[:-1]
            return tf.zeros(tf.concat([lead, [UNIFIED_ACTION_DIM]], axis=0), dtype=tensor.dtype)

        sources, targets = zip(*mapping, strict=True)
        source_idx = tf.constant(sources, dtype=tf.int32)
        target_idx = tf.constant(targets, dtype=tf.int32)
        tf.debugging.assert_less(tf.reduce_max(source_idx), tf.shape(tensor)[-1])

        selected = tf.gather(tensor, source_idx, axis=-1)  # [..., K]
        lead_shape = tf.shape(selected)[:-1]
        flat = tf.reshape(selected, [-1, tf.shape(selected)[-1]])  # [N, K]
        projection = tf.one_hot(target_idx, UNIFIED_ACTION_DIM, dtype=tensor.dtype)  # [K, 80]
        mapped_flat = tf.matmul(flat, projection)  # [N, 80]
        return tf.reshape(
            mapped_flat,
            tf.concat([lead_shape, [UNIFIED_ACTION_DIM]], axis=0),
        )

    traj["state"] = map_tensor(traj["state"], spec.state_mapping)
    traj["actions"] = map_tensor(traj["actions"], spec.action_mapping)
    # mask 仍按时间维 T 广播成 [T, 80]（与 horizon 无关）
    traj["action_mask"] = tf.broadcast_to(
        tf.constant(spec.action_mask, tf.bool),
        [tf.shape(traj["state"])[0], UNIFIED_ACTION_DIM],
    )
    return traj


def write_metadata(directory: str | Path, spec: UnifiedActionSpec) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "unified_action_space.json").write_text(
        json.dumps(
            {"version": 1, "width": UNIFIED_ACTION_DIM, "fingerprint": spec.fingerprint},
            indent=2,
        )
        + "\n"
    )


def validate_metadata(directory: str | Path, spec: UnifiedActionSpec) -> None:
    path = Path(directory) / "unified_action_space.json"
    if not path.exists():
        raise ValueError(f"Unified norm stats are missing mapping metadata: {path}")
    metadata = json.loads(path.read_text())
    expected = {"version": 1, "width": UNIFIED_ACTION_DIM, "fingerprint": spec.fingerprint}
    if metadata != expected:
        raise ValueError(f"Unified norm stats mapping mismatch at {path}: expected {expected}, got {metadata}")


def _same(mapping: DimMapping, *, delta: tuple[int, ...] = ()) -> UnifiedActionSpec:
    return UnifiedActionSpec(mapping, mapping, absolute_to_delta_slots=delta)


def _dual_arm(arm_dof: int, *, left_source: int = 0, right_source: int | None = None) -> DimMapping:
    right_source = arm_dof if right_source is None else right_source
    return dims(left_source, LEFT_ARM, arm_dof) + dims(right_source, RIGHT_ARM, arm_dof)


def _single_right(arm_dof: int, gripper_source: int | None = None) -> UnifiedActionSpec:
    mapping = dims(0, RIGHT_ARM, arm_dof)
    if gripper_source is not None:
        mapping += dims(gripper_source, RIGHT_GRIPPER, 1)
    return _same(mapping, delta=slots(RIGHT_ARM, arm_dof))


_EGO_MAPPING = (
    dims(0, LEFT_EEF_POSITION, 3)
    + dims(3, LEFT_EEF_EULER, 3)
    + dims(6, RIGHT_EEF_POSITION, 3)
    + dims(9, RIGHT_EEF_EULER, 3)
)

# eva native 14D -> 80D eef槽 + gripper槽
# layout: [L_xyz3, L_ypr3, R_xyz3, R_ypr3, L_grip1, R_grip1]
_EGO_EVA_14_MAPPING = (
    _EGO_MAPPING
    + dims(12, LEFT_GRIPPER, 1)
    + dims(13, RIGHT_GRIPPER, 1)
)

_AGIBOT_MAPPING = (
    _dual_arm(7) + dims(14, LEFT_GRIPPER, 1) + dims(15, RIGHT_GRIPPER, 1) + dims(16, HEAD, 2) + dims(18, WAIST, 2)
)

_PIPER_MAPPING = dims(0, LEFT_ARM, 6) + dims(6, LEFT_GRIPPER, 1) + dims(7, RIGHT_ARM, 6) + dims(13, RIGHT_GRIPPER, 1)

# AtomAligned 自采对齐：绝对 EEF(ypr) + gripper∈[0,1]
# 单右臂 7D: [R_xyz, R_ypr, R_grip]
# 双臂 14D: [L_xyz, L_ypr, L_grip, R_xyz, R_ypr, R_grip]
_ALIGNED_PARALLEL_GRIPPER_MAPPING = (
    dims(0, LEFT_EEF_POSITION, 3)
    + dims(3, LEFT_EEF_EULER, 3)
    + dims(6, LEFT_GRIPPER, 1)
    + dims(7, RIGHT_EEF_POSITION, 3)
    + dims(10, RIGHT_EEF_EULER, 3)
    + dims(13, RIGHT_GRIPPER, 1)
)
_ALIGNED_SINGLE_RIGHT_MAPPING = dims(0, RIGHT_EEF_POSITION, 6) + dims(6, RIGHT_GRIPPER, 1)

_ALIGNED_PARALLEL_NO_GRIPPER_MAPPING = (
    dims(0, LEFT_EEF_POSITION, 3)
    + dims(3, LEFT_EEF_EULER, 3)
    + dims(6, RIGHT_EEF_POSITION, 3)
    + dims(9, RIGHT_EEF_EULER, 3)
)
_ALIGNED_SINGLE_RIGHT_NO_GRIPPER_MAPPING = dims(0, RIGHT_EEF_POSITION, 6)

UNIFIED_ACTION_SPECS: dict[str, UnifiedActionSpec] = {
    "agibot": _same(_AGIBOT_MAPPING, delta=slots(LEFT_ARM, 7) + slots(RIGHT_ARM, 7)),
    "droid": _single_right(7, 7),
    "egoverse_aria": _same(_EGO_MAPPING),
    "egoverse_eva": _same(_EGO_EVA_14_MAPPING),
    "egoverse_human": _same(_EGO_MAPPING),
    "egoverse_mecka": _same(_EGO_MAPPING),
    "egoverse_scale": _same(_EGO_MAPPING),
    # 以后单独数据集族 EgoVerse_rl2（不要并进 full 子集）
    "egoverse_rl2_eva": _same(_EGO_EVA_14_MAPPING),
    "egoverse_rl2_human": _same(_EGO_MAPPING),  # legacy; mixture 不再使用
    "egoverse_rl2_indomain": _same(_EGO_MAPPING),
    "egoverse_rl2_diverse": _same(_EGO_MAPPING),
    "aligned_hangzhou_human_right": _same(_ALIGNED_SINGLE_RIGHT_NO_GRIPPER_MAPPING),
    "aligned_hangzhou_robot_right": _same(_ALIGNED_SINGLE_RIGHT_MAPPING),
    "aligned_shenzhen_human_bimanual": _same(_ALIGNED_PARALLEL_NO_GRIPPER_MAPPING),
    "aligned_shenzhen_robot_bimanual": _same(_ALIGNED_PARALLEL_GRIPPER_MAPPING),
    "piper30": _same(_PIPER_MAPPING, delta=slots(LEFT_ARM, 6) + slots(RIGHT_ARM, 6)),
    "piper2": _same(_PIPER_MAPPING, delta=slots(LEFT_ARM, 6) + slots(RIGHT_ARM, 6)),
}


def _register_robocoin() -> None:
    specs = UNIFIED_ACTION_SPECS
    dual6_grippers = (
        dims(0, LEFT_ARM, 6) + dims(6, LEFT_GRIPPER, 1) + dims(7, RIGHT_ARM, 6) + dims(13, RIGHT_GRIPPER, 1)
    )
    dual7_grippers = (
        dims(0, LEFT_ARM, 7) + dims(7, LEFT_GRIPPER, 1) + dims(8, RIGHT_ARM, 7) + dims(15, RIGHT_GRIPPER, 1)
    )
    mixed6 = dims(0, LEFT_ARM, 6) + dims(6, LEFT_GRIPPER, 1) + dims(13, RIGHT_ARM, 6) + dims(19, RIGHT_GRIPPER, 1)
    mixed7_right_first = (
        dims(0, RIGHT_ARM, 7) + dims(7, RIGHT_GRIPPER, 1) + dims(14, LEFT_ARM, 7) + dims(21, LEFT_GRIPPER, 1)
    )
    dual6_delta = slots(LEFT_ARM, 6) + slots(RIGHT_ARM, 6)
    dual7_delta = slots(LEFT_ARM, 7) + slots(RIGHT_ARM, 7)

    specs.update(
        {
            "robocoin_agilex_cobot_magic_s26_a26": _same(mixed6, delta=dual6_delta),
            "robocoin_airbot_mmk2_s36_a36": _same(
                _dual_arm(6) + dims(12, LEFT_HAND, 12) + dims(24, RIGHT_HAND, 12), delta=dual6_delta
            ),
            "robocoin_galaxea_r1_lite_upper_s14_a14": _same(
                _dual_arm(6) + dims(12, LEFT_GRIPPER, 1) + dims(13, RIGHT_GRIPPER, 1), delta=dual6_delta
            ),
            "robocoin_realman_rmc_aida_l_s28_a28": _same(mixed7_right_first, delta=dual7_delta),
            "robocoin_unitree_g1_dex3_s28_a28": _same(
                _dual_arm(7) + dims(14, LEFT_HAND, 7) + dims(21, RIGHT_HAND, 7), delta=dual7_delta
            ),
            "robocoin_agilex_decoupled_magic_s14_a14_fps30": _same(dual6_grippers, delta=dual6_delta),
            "robocoin_agilex_decoupled_magic_s14_a14_fps50": _same(dual6_grippers, delta=dual6_delta),
            "robocoin_agilex_decoupled_magic_s26_a26": _same(mixed6, delta=dual6_delta),
            "robocoin_aloha_s26_a26": _same(
                dims(0, LEFT_ARM, 6) + dims(12, LEFT_GRIPPER, 1) + dims(13, RIGHT_ARM, 6) + dims(25, RIGHT_GRIPPER, 1),
                delta=dual6_delta,
            ),
            "robocoin_alpha_bot_2_s28_a28": _same(
                dims(0, LEFT_ARM, 7) + dims(13, RIGHT_ARM, 7) + dims(26, LEFT_GRIPPER, 1) + dims(27, RIGHT_GRIPPER, 1),
                delta=dual7_delta,
            ),
            "robocoin_discover_aitbot_mmk2_s36_a36": _same(
                _dual_arm(6) + dims(12, LEFT_HAND, 12) + dims(24, RIGHT_HAND, 12), delta=dual6_delta
            ),
            "robocoin_galaxea_r1_lite_s14_a14": _same(dual6_grippers, delta=dual6_delta),
            "robocoin_galaxea_r1_lite_s16_a18": _same(
                _dual_arm(7) + dims(14, LEFT_GRIPPER, 1) + dims(15, RIGHT_GRIPPER, 1), delta=dual7_delta
            ),
            "robocoin_leju_robot_s118_a54": _same(
                _dual_arm(7)
                + dims(14, LEFT_LEG, 6)
                + dims(20, RIGHT_LEG, 6)
                + dims(26, LEFT_HAND, 6)
                + dims(32, RIGHT_HAND, 6)
                + dims(38, HEAD, 2),
                delta=dual7_delta,
            ),
            "robocoin_leju_robot_s54_a54": _same(
                _dual_arm(7)
                + dims(14, LEFT_LEG, 6)
                + dims(20, RIGHT_LEG, 6)
                + dims(26, LEFT_HAND, 6)
                + dims(32, RIGHT_HAND, 6)
                + dims(38, HEAD, 2),
                delta=dual7_delta,
            ),
            "robocoin_realman_rmc_aidal_s28_a28": _same(mixed7_right_first, delta=dual7_delta),
            "robocoin_ruantong_a2d_s17_a17": _same(
                dims(0, OTHER_BODY, 1)
                + dims(1, LEFT_ARM, 7)
                + dims(8, RIGHT_ARM, 7)
                + dims(15, LEFT_GRIPPER, 1)
                + dims(16, RIGHT_GRIPPER, 1),
                delta=dual7_delta,
            ),
            "robocoin_ruantong_a2d_s41_a34": _same(
                _dual_arm(7)
                + dims(28, WAIST, 2)
                + dims(30, HEAD, 2)
                + dims(32, LEFT_GRIPPER, 1)
                + dims(33, RIGHT_GRIPPER, 1),
                delta=dual7_delta,
            ),
            "robocoin_unitree_g1_s28_a28_high": _same(
                _dual_arm(7) + dims(14, LEFT_HAND, 7) + dims(21, RIGHT_HAND, 7), delta=dual7_delta
            ),
            "robocoin_unitree_g1_s28_a28": _same(
                _dual_arm(7) + dims(14, LEFT_HAND, 7) + dims(21, RIGHT_HAND, 7), delta=dual7_delta
            ),
            "robocoin_unknown_s30_a30_high": _same(
                _dual_arm(7) + dims(14, LEFT_HAND, 7) + dims(21, RIGHT_HAND, 7), delta=dual7_delta
            ),
        }
    )

    yinhe_action = dual7_grippers
    yinhe_state = dims(5, LEFT_ARM, 7) + dims(12, LEFT_GRIPPER, 1) + dims(13, RIGHT_ARM, 7) + dims(20, RIGHT_GRIPPER, 1)
    specs["robocoin_yinhe_s49_a16"] = UnifiedActionSpec(
        state_mapping=yinhe_state,
        action_mapping=yinhe_action,
        absolute_to_delta_slots=dual7_delta,
    )


def _register_robomind() -> None:
    specs = UNIFIED_ACTION_SPECS
    dual6_grippers = (
        dims(0, LEFT_ARM, 6) + dims(6, LEFT_GRIPPER, 1) + dims(7, RIGHT_ARM, 6) + dims(13, RIGHT_GRIPPER, 1)
    )
    dual7_grippers = (
        dims(0, LEFT_ARM, 7) + dims(7, LEFT_GRIPPER, 1) + dims(8, RIGHT_ARM, 7) + dims(15, RIGHT_GRIPPER, 1)
    )
    dual7_delta = slots(LEFT_ARM, 7) + slots(RIGHT_ARM, 7)
    specs.update(
        {
            "robomind_agilex_cobot_magic_s14_a14": _same(
                dual6_grippers, delta=slots(LEFT_ARM, 6) + slots(RIGHT_ARM, 6)
            ),
            "robomind_franka_fr3_dual_s16_a16": _same(dual7_grippers, delta=dual7_delta),
            "robomind_franka_panda_s8_a8": _single_right(7, 7),
            "robomind_franka_sim_franka_s8_a8": _single_right(7, 7),
            "robomind_franka_sim_simulation_s8_a8": _single_right(7, 7),
            "robomind_franka_sim_simulation_no_front_s8_a8": _single_right(7, 7),
            "robomind_franka_sim_none_s8_a8": _single_right(7, 7),
            "robomind_tienkung_gello_s16_a16": _same(dual7_grippers, delta=dual7_delta),
            "robomind_tienkung_prod1_gello_s16_a16": _same(dual7_grippers, delta=dual7_delta),
            "robomind_tienkung_xsens_s14_a14": _same(_dual_arm(7), delta=dual7_delta),
            "robomind_tienkung_sim_s38_a38": _same(
                dims(0, LEFT_ARM, 7) + dims(7, LEFT_HAND, 12) + dims(19, RIGHT_ARM, 7) + dims(26, RIGHT_HAND, 12),
                delta=dual7_delta,
            ),
            "robomind_tienkung_real_s38_a38": _same(
                dims(0, LEFT_ARM, 7) + dims(7, LEFT_HAND, 12) + dims(19, RIGHT_ARM, 7) + dims(26, RIGHT_HAND, 12),
                delta=dual7_delta,
            ),
            "robomind_ur5e_s7_a7": _single_right(6, 6),
        }
    )


_register_robocoin()
_register_robomind()
