from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_validation as ev


def test_piper_14d_to_unified_80d_slots():
    assert ev.UNIFIED_ACTION_DIM == 80
    assert ev.UNIFIED_PIPER_DIMS == (
        0,
        1,
        2,
        3,
        4,
        5,
        16,
        29,
        30,
        31,
        32,
        33,
        34,
        45,
    )


def test_piper_joint_delta_dims_are_arm_joints_only():
    assert ev.JOINT_DELTA_DIMS == (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12)
    assert ev.GRIPPER_DIMS == (6, 13)
