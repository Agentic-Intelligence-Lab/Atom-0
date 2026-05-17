"""DCC Stage 1: context prompt/dropout transform checks."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

import numpy as np

from openpi import transforms


def _make_sample():
    return {
        "prompt": "put candies in bowl",
        "subtask": "grasp or manipulate the object",
        "quality": 5,
        "speed_bin": "normal",
        "mistake": False,
        "success": True,
        "control_mode": "joint",
        "subgoal_image": {
            "base_0_rgb": np.zeros((224, 224, 3), dtype=np.uint8),
        },
        "subgoal_image_mask": {
            "base_0_rgb": np.True_,
        },
    }


def main():
    np.random.seed(0)

    # With no dropout, all InfiData-style fields become a stable structured text prompt.
    sample = _make_sample()
    sample = transforms.ApplyDiverseContextDropout(
        subgoal_keep_prob=1.0,
        subtask_drop_when_subgoal=0.0,
        metadata_drop_prob=0.0,
        metadata_field_drop_prob=0.0,
    )(sample)
    sample = transforms.BuildDiverseContextPrompt()(sample)
    assert "Task: put candies in bowl" in sample["prompt"]
    assert "Subtask: grasp or manipulate the object" in sample["prompt"]
    assert "Metadata: quality=5; speed=normal; mistake=False; success=True" in sample["prompt"]
    assert "Control: joint" in sample["prompt"]
    assert "subtask" not in sample

    # With forced dropout, the subgoal mask is retained structurally but disabled.
    sample = transforms.ApplyDiverseContextDropout(
        subgoal_keep_prob=0.0,
        metadata_drop_prob=1.0,
    )(_make_sample())
    assert not bool(sample["subgoal_image_mask"]["base_0_rgb"])
    sample = transforms.BuildDiverseContextPrompt()(sample)
    assert "Metadata: quality=none; speed=none; mistake=none; success=none" in sample["prompt"]

    # Statistical sanity check for per-sample subgoal dropout.
    np.random.seed(1)
    kept = []
    transform = transforms.ApplyDiverseContextDropout(subgoal_keep_prob=0.25)
    for _ in range(5000):
        out = transform(_make_sample())
        kept.append(bool(out["_dcc_subgoal_kept"]))
    keep_rate = np.mean(kept)
    assert abs(keep_rate - 0.25) < 0.03, keep_rate

    print("PASS: DCC context transforms")


if __name__ == "__main__":
    main()
