from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_validation import compute_uniform_anchors, build_anchor_records, normal_swapped_mae


def test_uniform_anchors_use_valid_interval_and_include_endpoint():
    assert compute_uniform_anchors(trajectory_length=10, horizon=3, limit=4) == [0, 2, 5, 7]


def test_uniform_anchors_short_trajectory_has_no_padding():
    assert compute_uniform_anchors(trajectory_length=5, horizon=8, limit=20) == []
    assert compute_uniform_anchors(trajectory_length=8, horizon=8, limit=20) == [0]


def test_anchor_records_are_exclusive_end_and_deterministic():
    episodes = [
        {"episode_index": 3, "task": "task a", "trajectory_length": 12},
        {"episode_index": 4, "task": "task b", "trajectory_length": 9},
    ]
    records = build_anchor_records(episodes, split="seen_test", horizon=8, anchors_per_episode=3)
    assert [(r["episode_index"], r["anchor_index"], r["target_start"], r["target_end"]) for r in records] == [
        (3, 0, 0, 8),
        (3, 2, 2, 10),
        (3, 4, 4, 12),
        (4, 0, 0, 8),
        (4, 1, 1, 9),
    ]


def test_normal_swapped_mae_reports_arm_mapping_without_changing_outputs():
    target = [[0, 0, 0, 0, 0, 0, 0, 10, 10, 10, 10, 10, 10, 10]]
    pred_swapped = [[10, 10, 10, 10, 10, 10, 10, 0, 0, 0, 0, 0, 0, 0]]
    out = normal_swapped_mae(pred_swapped, target)
    assert out["normal_mae"] == 10.0
    assert out["swapped_mae"] == 0.0
