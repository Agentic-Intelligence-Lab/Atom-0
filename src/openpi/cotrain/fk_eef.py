"""URDF forward-kinematics helpers that fill unified 80D EEF slots.

Only datasets listed in ``FK_EEF_SPECS`` (from ``docs/joint2eef.md``) are filled.
Datasets without a URDF mapping are left unchanged. EEF pose is absolute
``xyz + yaw/pitch/roll`` (matches the cotrain unified-action design).
"""

from __future__ import annotations

import dataclasses
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from openpi.cotrain.action_space import (
    LEFT_ARM,
    LEFT_EEF_EULER,
    LEFT_EEF_POSITION,
    RIGHT_ARM,
    RIGHT_EEF_EULER,
    RIGHT_EEF_POSITION,
    UNIFIED_ACTION_DIM,
    slots,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_URDF_DIR = _REPO_ROOT / "assets" / "urdf"

LEFT_EEF_SLOTS = slots(LEFT_EEF_POSITION, 3) + slots(LEFT_EEF_EULER, 3)
RIGHT_EEF_SLOTS = slots(RIGHT_EEF_POSITION, 3) + slots(RIGHT_EEF_EULER, 3)


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = map(float, rpy)
    return Rotation.from_euler("xyz", [roll, pitch, yaw]).as_matrix()


def _make_transform(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _rpy_matrix(rpy)
    transform[:3, 3] = xyz
    return transform


def _parse_xyz_rpy(element: ET.Element | None) -> tuple[np.ndarray, np.ndarray]:
    if element is None:
        return np.zeros(3), np.zeros(3)
    xyz = np.fromstring(element.attrib.get("xyz", "0 0 0"), sep=" ", dtype=np.float64)
    rpy = np.fromstring(element.attrib.get("rpy", "0 0 0"), sep=" ", dtype=np.float64)
    if xyz.size != 3:
        xyz = np.zeros(3)
    if rpy.size != 3:
        rpy = np.zeros(3)
    return xyz, rpy


@dataclasses.dataclass(frozen=True)
class UrdfJoint:
    name: str
    joint_type: str
    parent: str
    child: str
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray
    axis: np.ndarray


@dataclasses.dataclass(frozen=True)
class UrdfModel:
    path: Path
    joints: dict[str, UrdfJoint]
    child_to_joint: dict[str, str]

    @property
    def actuated_joint_names(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, joint in self.joints.items()
            if joint.joint_type in {"revolute", "continuous", "prismatic"}
        )


def load_urdf(path: str | Path) -> UrdfModel:
    path = Path(path)
    root = ET.parse(path).getroot()
    joints: dict[str, UrdfJoint] = {}
    child_to_joint: dict[str, str] = {}
    for joint_el in root.findall("joint"):
        name = joint_el.attrib["name"]
        joint_type = joint_el.attrib["type"]
        parent = joint_el.find("parent").attrib["link"]
        child = joint_el.find("child").attrib["link"]
        origin_xyz, origin_rpy = _parse_xyz_rpy(joint_el.find("origin"))
        axis_el = joint_el.find("axis")
        if axis_el is None:
            axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            axis = np.fromstring(axis_el.attrib.get("xyz", "1 0 0"), sep=" ", dtype=np.float64)
            if axis.size != 3:
                axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            norm = np.linalg.norm(axis)
            axis = axis / norm if norm > 0 else np.array([1.0, 0.0, 0.0], dtype=np.float64)
        joint = UrdfJoint(
            name=name,
            joint_type=joint_type,
            parent=parent,
            child=child,
            origin_xyz=origin_xyz,
            origin_rpy=origin_rpy,
            axis=axis,
        )
        joints[name] = joint
        child_to_joint[child] = name
    return UrdfModel(path=path, joints=joints, child_to_joint=child_to_joint)


def _joint_motion_matrix(joint: UrdfJoint, q: float) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    if joint.joint_type in {"revolute", "continuous"}:
        transform[:3, :3] = Rotation.from_rotvec(joint.axis * q).as_matrix()
    elif joint.joint_type == "prismatic":
        transform[:3, 3] = joint.axis * q
    # fixed / floating / planar: identity motion
    return transform


def fk_link_pose(model: UrdfModel, ee_link: str, q_by_name: dict[str, float]) -> np.ndarray:
    """Return 4x4 transform of ``ee_link`` in the URDF root frame."""
    chain = _joint_chain(model, ee_link)
    pose = np.eye(4, dtype=np.float64)
    for joint in chain:
        pose = pose @ _make_transform(joint.origin_xyz, joint.origin_rpy)
        if joint.joint_type == "fixed":
            continue
        if joint.name not in q_by_name:
            q = 0.0
        else:
            q = float(q_by_name[joint.name])
        pose = pose @ _joint_motion_matrix(joint, q)
    return pose


def _joint_chain(model: UrdfModel, ee_link: str) -> tuple[UrdfJoint, ...]:
    chain: list[UrdfJoint] = []
    link = ee_link
    seen: set[str] = set()
    while link in model.child_to_joint:
        if link in seen:
            raise ValueError(f"Cycle while walking URDF parents to {ee_link}")
        seen.add(link)
        joint = model.joints[model.child_to_joint[link]]
        chain.append(joint)
        link = joint.parent
    chain.reverse()
    return tuple(chain)


@dataclasses.dataclass(frozen=True)
class _ArmFkPlanStep:
    origin: np.ndarray  # (4, 4)
    joint_type: str
    q_index: int | None  # index into arm joint vector; None -> q=0
    axis: np.ndarray | None


@dataclasses.dataclass(frozen=True)
class _ArmFkPlan:
    steps: tuple[_ArmFkPlanStep, ...]


@lru_cache(maxsize=64)
def _arm_fk_plan(model_path: str, ee_link: str, joint_names: tuple[str, ...]) -> _ArmFkPlan:
    model = _cached_urdf(model_path)
    name_to_idx = {name: index for index, name in enumerate(joint_names)}
    steps: list[_ArmFkPlanStep] = []
    for joint in _joint_chain(model, ee_link):
        q_index = name_to_idx.get(joint.name)
        steps.append(
            _ArmFkPlanStep(
                origin=_make_transform(joint.origin_xyz, joint.origin_rpy),
                joint_type=joint.joint_type,
                q_index=q_index,
                axis=None if joint.joint_type == "fixed" else joint.axis.copy(),
            )
        )
    return _ArmFkPlan(tuple(steps))


def _fk_arm_batch(q: np.ndarray, plan: _ArmFkPlan) -> np.ndarray:
    """Map arm joint configs ``q`` (N, dof) to EEF xyz+yaw/pitch/roll (N, 6)."""
    q = np.asarray(q, dtype=np.float64)
    if q.ndim == 1:
        q = q[np.newaxis, :]
    n = q.shape[0]
    pose = np.tile(np.eye(4, dtype=np.float64), (n, 1, 1))
    for step in plan.steps:
        pose = np.matmul(pose, step.origin)
        if step.joint_type == "fixed":
            continue
        qi = np.zeros(n, dtype=np.float64) if step.q_index is None else q[:, step.q_index]
        motion = np.tile(np.eye(4, dtype=np.float64), (n, 1, 1))
        if step.joint_type in {"revolute", "continuous"}:
            assert step.axis is not None
            motion[:, :3, :3] = Rotation.from_rotvec(step.axis * qi[:, np.newaxis]).as_matrix()
        elif step.joint_type == "prismatic":
            assert step.axis is not None
            motion[:, :3, 3] = step.axis * qi[:, np.newaxis]
        pose = np.matmul(pose, motion)
    xyz = pose[:, :3, 3]
    ypr = Rotation.from_matrix(pose[:, :3, :3]).as_euler("zyx")
    return np.concatenate([xyz, ypr], axis=-1)


def pose_to_xyz_yaw_pitch_roll(pose: np.ndarray) -> np.ndarray:
    xyz = pose[:3, 3]
    # zyx intrinsic == extrinsic xyz reversed; EgoVerse / design use yaw/pitch/roll.
    yaw, pitch, roll = Rotation.from_matrix(pose[:3, :3]).as_euler("zyx")
    return np.array([xyz[0], xyz[1], xyz[2], yaw, pitch, roll], dtype=np.float64)


@dataclasses.dataclass(frozen=True)
class FkArmConfig:
    """One arm chain: URDF joints mapped from contiguous unified arm slots."""

    joint_names: tuple[str, ...]
    ee_link: str
    arm_slot_start: int  # LEFT_ARM or RIGHT_ARM
    eef_position_start: int
    eef_euler_start: int

    @property
    def dof(self) -> int:
        return len(self.joint_names)

    @property
    def eef_slots(self) -> tuple[int, ...]:
        return slots(self.eef_position_start, 3) + slots(self.eef_euler_start, 3)


@dataclasses.dataclass(frozen=True)
class FkEefSpec:
    """Per-dataset FK fill config (only datasets with a known URDF)."""

    dataset_id: str
    urdf_file: str
    arms: tuple[FkArmConfig, ...]
    expected_arm_dofs: tuple[int, ...]  # for the validate script
    notes: str = ""

    @property
    def eef_slots(self) -> tuple[int, ...]:
        out: list[int] = []
        for arm in self.arms:
            out.extend(arm.eef_slots)
        return tuple(out)


def _left_arm(joint_names: tuple[str, ...], ee_link: str) -> FkArmConfig:
    return FkArmConfig(joint_names, ee_link, LEFT_ARM, LEFT_EEF_POSITION, LEFT_EEF_EULER)


def _right_arm(joint_names: tuple[str, ...], ee_link: str) -> FkArmConfig:
    return FkArmConfig(joint_names, ee_link, RIGHT_ARM, RIGHT_EEF_POSITION, RIGHT_EEF_EULER)


def _spec(
    dataset_id: str,
    urdf_file: str,
    arms: tuple[FkArmConfig, ...],
    *,
    notes: str = "",
) -> FkEefSpec:
    return FkEefSpec(
        dataset_id=dataset_id,
        urdf_file=urdf_file,
        arms=arms,
        expected_arm_dofs=tuple(arm.dof for arm in arms),
        notes=notes,
    )


# Piper real-robot datasets share one 6-DOF arm URDF; left/right read unified slots 0-5 / 29-34.
_PIPER_FK_ARMS = (
    _left_arm(tuple(f"joint{i}" for i in range(1, 7)), "link6"),
    _right_arm(tuple(f"joint{i}" for i in range(1, 7)), "link6"),
)

# Registry derived from docs/joint2eef.md. dataset_id uses cotrain uids.
FK_EEF_SPECS: dict[str, FkEefSpec] = {
    "agibot": _spec(
        "agibot",
        "agibot_G2.urdf",
        (
            _left_arm(
                tuple(f"idx2{i}_arm_l_joint{i}" for i in range(1, 8)),
                "arm_l_end_link",
            ),
            _right_arm(
                tuple(f"idx6{i}_arm_r_joint{i}" for i in range(1, 8)),
                "arm_r_end_link",
            ),
        ),
    ),
    "droid": _spec(
        "droid",
        "panda.urdf",
        (_right_arm(tuple(f"panda_joint{i}" for i in range(1, 8)), "panda_hand"),),
    ),
    "piper30": _spec("piper30", "piper_gripper.urdf", _PIPER_FK_ARMS),
    "piper2": _spec("piper2", "piper_gripper.urdf", _PIPER_FK_ARMS),
    "robocoin_airbot_mmk2_s36_a36": _spec(
        "robocoin_airbot_mmk2_s36_a36",
        "mmk2_s_g2.urdf",
        (
            _left_arm(tuple(f"left_joint{i}" for i in range(1, 7)), "left_flange"),
            _right_arm(tuple(f"right_joint{i}" for i in range(1, 7)), "right_flange"),
        ),
    ),
    "robocoin_discover_aitbot_mmk2_s36_a36": _spec(
        "robocoin_discover_aitbot_mmk2_s36_a36",
        "mmk2_s_g2.urdf",
        (
            _left_arm(tuple(f"left_joint{i}" for i in range(1, 7)), "left_flange"),
            _right_arm(tuple(f"right_joint{i}" for i in range(1, 7)), "right_flange"),
        ),
    ),
    "robocoin_galaxea_r1_lite_upper_s14_a14": _spec(
        "robocoin_galaxea_r1_lite_upper_s14_a14",
        "r1_v2_1_0.urdf",
        (
            _left_arm(tuple(f"left_arm_joint{i}" for i in range(1, 7)), "left_arm_link6"),
            _right_arm(tuple(f"right_arm_joint{i}" for i in range(1, 7)), "right_arm_link6"),
        ),
    ),
    "robocoin_galaxea_r1_lite_s14_a14": _spec(
        "robocoin_galaxea_r1_lite_s14_a14",
        "r1_v2_1_0.urdf",
        (
            _left_arm(tuple(f"left_arm_joint{i}" for i in range(1, 7)), "left_arm_link6"),
            _right_arm(tuple(f"right_arm_joint{i}" for i in range(1, 7)), "right_arm_link6"),
        ),
    ),
    "robocoin_galaxea_r1_lite_s16_a18": _spec(
        "robocoin_galaxea_r1_lite_s16_a18",
        "r1_v2_1_0.urdf",
        (
            # Dataset maps 7 arm joints; URDF only exposes 6. Validate script flags this.
            _left_arm(tuple(f"left_arm_joint{i}" for i in range(1, 7)), "left_arm_link6"),
            _right_arm(tuple(f"right_arm_joint{i}" for i in range(1, 7)), "right_arm_link6"),
        ),
        notes="URDF arm dof=6 but unified mapping uses 7; FK disabled until remapped.",
    ),
    "robocoin_unitree_g1_s28_a28": _spec(
        "robocoin_unitree_g1_s28_a28",
        "g1_body29_hand14.urdf",
        (
            _left_arm(
                (
                    "left_shoulder_pitch_joint",
                    "left_shoulder_roll_joint",
                    "left_shoulder_yaw_joint",
                    "left_elbow_joint",
                    "left_wrist_roll_joint",
                    "left_wrist_pitch_joint",
                    "left_wrist_yaw_joint",
                ),
                "left_wrist_yaw_link",
            ),
            _right_arm(
                (
                    "right_shoulder_pitch_joint",
                    "right_shoulder_roll_joint",
                    "right_shoulder_yaw_joint",
                    "right_elbow_joint",
                    "right_wrist_roll_joint",
                    "right_wrist_pitch_joint",
                    "right_wrist_yaw_joint",
                ),
                "right_wrist_yaw_link",
            ),
        ),
    ),
    "robocoin_unitree_g1_s28_a28_high": _spec(
        "robocoin_unitree_g1_s28_a28_high",
        "g1_body29_hand14.urdf",
        (
            _left_arm(
                (
                    "left_shoulder_pitch_joint",
                    "left_shoulder_roll_joint",
                    "left_shoulder_yaw_joint",
                    "left_elbow_joint",
                    "left_wrist_roll_joint",
                    "left_wrist_pitch_joint",
                    "left_wrist_yaw_joint",
                ),
                "left_wrist_yaw_link",
            ),
            _right_arm(
                (
                    "right_shoulder_pitch_joint",
                    "right_shoulder_roll_joint",
                    "right_shoulder_yaw_joint",
                    "right_elbow_joint",
                    "right_wrist_roll_joint",
                    "right_wrist_pitch_joint",
                    "right_wrist_yaw_joint",
                ),
                "right_wrist_yaw_link",
            ),
        ),
    ),
    "robocoin_unitree_g1_dex3_s28_a28": _spec(
        "robocoin_unitree_g1_dex3_s28_a28",
        "g1_body29_hand14.urdf",
        (
            _left_arm(
                (
                    "left_shoulder_pitch_joint",
                    "left_shoulder_roll_joint",
                    "left_shoulder_yaw_joint",
                    "left_elbow_joint",
                    "left_wrist_roll_joint",
                    "left_wrist_pitch_joint",
                    "left_wrist_yaw_joint",
                ),
                "left_wrist_yaw_link",
            ),
            _right_arm(
                (
                    "right_shoulder_pitch_joint",
                    "right_shoulder_roll_joint",
                    "right_shoulder_yaw_joint",
                    "right_elbow_joint",
                    "right_wrist_roll_joint",
                    "right_wrist_pitch_joint",
                    "right_wrist_yaw_joint",
                ),
                "right_wrist_yaw_link",
            ),
        ),
    ),
    "robocoin_leju_robot_s118_a54": _spec(
        "robocoin_leju_robot_s118_a54",
        "leju_s54.urdf",
        (
            _left_arm(tuple(f"zarm_l{i}_joint" for i in range(1, 8)), "zarm_l7_end_effector"),
            _right_arm(tuple(f"zarm_r{i}_joint" for i in range(1, 8)), "zarm_r7_end_effector"),
        ),
    ),
    "robocoin_leju_robot_s54_a54": _spec(
        "robocoin_leju_robot_s54_a54",
        "leju_s54.urdf",
        (
            _left_arm(tuple(f"zarm_l{i}_joint" for i in range(1, 8)), "zarm_l7_end_effector"),
            _right_arm(tuple(f"zarm_r{i}_joint" for i in range(1, 8)), "zarm_r7_end_effector"),
        ),
    ),
    "robocoin_yinhe_s49_a16": _spec(
        "robocoin_yinhe_s49_a16",
        "galbot_one_golf.urdf",
        (
            _left_arm(tuple(f"left_arm_joint{i}" for i in range(1, 8)), "left_gripper_tcp_link"),
            _right_arm(tuple(f"right_arm_joint{i}" for i in range(1, 8)), "right_gripper_tcp_link"),
        ),
    ),
    "robomind_franka_fr3_dual_s16_a16": _spec(
        "robomind_franka_fr3_dual_s16_a16",
        "dual_fr3.urdf",
        (
            _left_arm(tuple(f"panda_1_joint{i}" for i in range(1, 8)), "panda_1_hand_tcp"),
            _right_arm(tuple(f"panda_2_joint{i}" for i in range(1, 8)), "panda_2_hand_tcp"),
        ),
    ),
    "robomind_franka_panda_s8_a8": _spec(
        "robomind_franka_panda_s8_a8",
        "panda.urdf",
        (_right_arm(tuple(f"panda_joint{i}" for i in range(1, 8)), "panda_hand"),),
    ),
    "robomind_franka_sim_franka_s8_a8": _spec(
        "robomind_franka_sim_franka_s8_a8",
        "panda.urdf",
        (_right_arm(tuple(f"panda_joint{i}" for i in range(1, 8)), "panda_hand"),),
    ),
    "robomind_franka_sim_simulation_s8_a8": _spec(
        "robomind_franka_sim_simulation_s8_a8",
        "panda.urdf",
        (_right_arm(tuple(f"panda_joint{i}" for i in range(1, 8)), "panda_hand"),),
    ),
    "robomind_franka_sim_simulation_no_front_s8_a8": _spec(
        "robomind_franka_sim_simulation_no_front_s8_a8",
        "panda.urdf",
        (_right_arm(tuple(f"panda_joint{i}" for i in range(1, 8)), "panda_hand"),),
    ),
    "robomind_franka_sim_none_s8_a8": _spec(
        "robomind_franka_sim_none_s8_a8",
        "panda.urdf",
        (_right_arm(tuple(f"panda_joint{i}" for i in range(1, 8)), "panda_hand"),),
    ),
    "robomind_tienkung_gello_s16_a16": _spec(
        "robomind_tienkung_gello_s16_a16",
        "tienkung2_lite.urdf",
        (
            # URDF only has 4 arm joints; unified mapping uses 7. Validate flags mismatch.
            _left_arm(
                (
                    "shoulder_pitch_l_joint",
                    "shoulder_roll_l_joint",
                    "shoulder_yaw_l_joint",
                    "elbow_pitch_l_joint",
                ),
                "elbow_pitch_l_link",
            ),
            _right_arm(
                (
                    "shoulder_pitch_r_joint",
                    "shoulder_roll_r_joint",
                    "shoulder_yaw_r_joint",
                    "elbow_pitch_r_joint",
                ),
                "elbow_pitch_r_link",
            ),
        ),
        notes="URDF arm dof=4 but unified mapping uses 7; FK disabled until full arm URDF is available.",
    ),
    "robomind_tienkung_prod1_gello_s16_a16": _spec(
        "robomind_tienkung_prod1_gello_s16_a16",
        "tienkung2_lite.urdf",
        (
            _left_arm(
                (
                    "shoulder_pitch_l_joint",
                    "shoulder_roll_l_joint",
                    "shoulder_yaw_l_joint",
                    "elbow_pitch_l_joint",
                ),
                "elbow_pitch_l_link",
            ),
            _right_arm(
                (
                    "shoulder_pitch_r_joint",
                    "shoulder_roll_r_joint",
                    "shoulder_yaw_r_joint",
                    "elbow_pitch_r_joint",
                ),
                "elbow_pitch_r_link",
            ),
        ),
        notes="URDF arm dof=4 but unified mapping uses 7; FK disabled until full arm URDF is available.",
    ),
    "robomind_ur5e_s7_a7": _spec(
        "robomind_ur5e_s7_a7",
        "ur5e.urdf",
        (
            _right_arm(
                (
                    "shoulder_pan_joint",
                    "shoulder_lift_joint",
                    "elbow_joint",
                    "wrist_1_joint",
                    "wrist_2_joint",
                    "wrist_3_joint",
                ),
                "tool0",
            ),
        ),
    ),
}

# Fix agibot joint naming: actual names are idx21..idx27 and idx61..idx67
FK_EEF_SPECS["agibot"] = _spec(
    "agibot",
    "agibot_G2.urdf",
    (
        _left_arm(
            (
                "idx21_arm_l_joint1",
                "idx22_arm_l_joint2",
                "idx23_arm_l_joint3",
                "idx24_arm_l_joint4",
                "idx25_arm_l_joint5",
                "idx26_arm_l_joint6",
                "idx27_arm_l_joint7",
            ),
            "arm_l_end_link",
        ),
        _right_arm(
            (
                "idx61_arm_r_joint1",
                "idx62_arm_r_joint2",
                "idx63_arm_r_joint3",
                "idx64_arm_r_joint4",
                "idx65_arm_r_joint5",
                "idx66_arm_r_joint6",
                "idx67_arm_r_joint7",
            ),
            "arm_r_end_link",
        ),
    ),
)


@dataclasses.dataclass(frozen=True)
class FkValidationResult:
    dataset_id: str
    urdf_file: str
    ok: bool
    messages: tuple[str, ...]
    mapped_arm_dofs: tuple[int, ...]
    urdf_arm_dofs: tuple[int, ...]


def validate_fk_spec(spec: FkEefSpec, *, urdf_dir: Path | None = None) -> FkValidationResult:
    """Check URDF exists, joint names exist, and DOF matches the unified arm mapping."""
    from openpi.cotrain import action_space as cotrain_action_space

    urdf_dir = Path(urdf_dir) if urdf_dir is not None else _DEFAULT_URDF_DIR
    urdf_path = urdf_dir / spec.urdf_file
    messages: list[str] = []
    ok = True

    if not urdf_path.is_file():
        return FkValidationResult(
            dataset_id=spec.dataset_id,
            urdf_file=spec.urdf_file,
            ok=False,
            messages=(f"missing URDF: {urdf_path}",),
            mapped_arm_dofs=(),
            urdf_arm_dofs=(),
        )

    model = load_urdf(urdf_path)
    unified = cotrain_action_space.UNIFIED_ACTION_SPECS.get(spec.dataset_id)
    if unified is None:
        ok = False
        messages.append("dataset_id not in UNIFIED_ACTION_SPECS")
        mapped_dofs: list[int] = []
    else:
        mapped_dofs = []
        for arm in spec.arms:
            n_mapped = sum(
                1
                for _, target in unified.action_mapping
                if arm.arm_slot_start <= target < arm.arm_slot_start + 7
            )
            mapped_dofs.append(n_mapped)
            if n_mapped != arm.dof:
                ok = False
                messages.append(
                    f"arm@{arm.arm_slot_start}: unified mapping dof={n_mapped} != URDF chain dof={arm.dof}"
                )

    urdf_dofs: list[int] = []
    for arm in spec.arms:
        missing = [name for name in arm.joint_names if name not in model.joints]
        if missing:
            ok = False
            messages.append(f"missing joints in URDF: {missing}")
        for name in arm.joint_names:
            joint = model.joints.get(name)
            if joint is not None and joint.joint_type not in {"revolute", "continuous", "prismatic"}:
                ok = False
                messages.append(f"joint {name} has non-actuated type {joint.joint_type}")
        if arm.ee_link not in model.child_to_joint and arm.ee_link not in {
            j.parent for j in model.joints.values()
        }:
            # ee may be a root-side link; also accept if any joint references it
            links = {j.parent for j in model.joints.values()} | {j.child for j in model.joints.values()}
            if arm.ee_link not in links:
                ok = False
                messages.append(f"ee_link not found: {arm.ee_link}")
        urdf_dofs.append(arm.dof)
        # Smoke FK at zero configuration.
        try:
            pose = fk_link_pose(model, arm.ee_link, {name: 0.0 for name in arm.joint_names})
            if not np.isfinite(pose).all():
                ok = False
                messages.append(f"non-finite FK pose for {arm.ee_link}")
        except Exception as exc:  # noqa: BLE001 - validation should surface any FK failure
            ok = False
            messages.append(f"FK failed for {arm.ee_link}: {exc}")

    if spec.notes:
        messages.append(f"note: {spec.notes}")

    return FkValidationResult(
        dataset_id=spec.dataset_id,
        urdf_file=spec.urdf_file,
        ok=ok,
        messages=tuple(messages),
        mapped_arm_dofs=tuple(mapped_dofs),
        urdf_arm_dofs=tuple(urdf_dofs),
    )


@lru_cache(maxsize=8)
def enabled_fk_dataset_ids(urdf_dir: str | None = None) -> tuple[str, ...]:
    """Return dataset ids whose URDF/joint-count validation passes."""
    directory = Path(urdf_dir) if urdf_dir is not None else _DEFAULT_URDF_DIR
    enabled = []
    for dataset_id, spec in FK_EEF_SPECS.items():
        result = validate_fk_spec(spec, urdf_dir=directory)
        if result.ok:
            enabled.append(dataset_id)
    return tuple(enabled)


@lru_cache(maxsize=1)
def _default_fk_enabled_ids() -> frozenset[str]:
    return frozenset(enabled_fk_dataset_ids(None))


def fk_enabled(dataset_id: str, *, urdf_dir: Path | None = None) -> bool:
    """Fast membership check for hot paths (norm / training loops)."""
    if urdf_dir is not None and Path(urdf_dir).resolve() != _DEFAULT_URDF_DIR.resolve():
        return dataset_id in enabled_fk_dataset_ids(str(Path(urdf_dir).resolve()))
    return dataset_id in _default_fk_enabled_ids()


@lru_cache(maxsize=64)
def _cached_urdf(path: str) -> UrdfModel:
    return load_urdf(path)


def _fill_vector(vector: np.ndarray, spec: FkEefSpec, urdf_path: Path) -> np.ndarray:
    """Fill EEF slots in a single [80] vector from its arm-joint slots."""
    return fill_eef_vectors_batch(vector, spec, urdf_dir=urdf_path.parent)[..., :]


def fill_eef_vectors_batch(
    vectors: np.ndarray,
    spec: FkEefSpec,
    *,
    urdf_dir: Path | None = None,
) -> np.ndarray:
    """Batch FK fill for ``[..., 80]`` vectors (any leading batch/time dimensions)."""
    vectors = np.asarray(vectors)
    if vectors.shape[-1] != UNIFIED_ACTION_DIM:
        raise ValueError(f"Expected last dim {UNIFIED_ACTION_DIM}, got {vectors.shape[-1]}")
    orig_dtype = vectors.dtype
    orig_shape = vectors.shape
    flat = vectors.reshape(-1, UNIFIED_ACTION_DIM).astype(np.float64, copy=False)
    if flat.size == 0:
        return np.asarray(vectors, copy=True)
    urdf_dir = Path(urdf_dir) if urdf_dir is not None else _DEFAULT_URDF_DIR
    urdf_path = urdf_dir / spec.urdf_file
    model_path = str(urdf_path)
    out = flat.copy()
    for arm in spec.arms:
        plan = _arm_fk_plan(model_path, arm.ee_link, arm.joint_names)
        q = flat[:, arm.arm_slot_start : arm.arm_slot_start + arm.dof]
        eef = _fk_arm_batch(q, plan)
        out[:, arm.eef_position_start : arm.eef_position_start + 3] = eef[:, :3]
        out[:, arm.eef_euler_start : arm.eef_euler_start + 3] = eef[:, 3:6]
    return out.reshape(orig_shape).astype(orig_dtype, copy=False)


def fill_eef_from_fk(
    state: np.ndarray,
    actions: np.ndarray | None,
    spec: FkEefSpec,
    *,
    urdf_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Fill unified EEF slots from absolute arm joints (before delta conversion)."""
    state_out = fill_eef_vectors_batch(state, spec, urdf_dir=urdf_dir)
    if actions is None:
        return state_out, None
    actions_out = fill_eef_vectors_batch(actions, spec, urdf_dir=urdf_dir)
    return state_out, actions_out


def fill_batch_dict(batch: dict, dataset_id: str, *, urdf_dir: Path | None = None) -> dict:
    """In-place friendly helper for numpy batches tagged by dataset_id."""
    spec = FK_EEF_SPECS.get(dataset_id)
    if spec is None:
        return batch
    if not fk_enabled(dataset_id, urdf_dir=urdf_dir):
        return batch
    state, actions = fill_eef_from_fk(batch["state"], batch.get("actions"), spec, urdf_dir=urdf_dir)
    batch["state"] = state
    if actions is not None:
        batch["actions"] = actions
    return batch
