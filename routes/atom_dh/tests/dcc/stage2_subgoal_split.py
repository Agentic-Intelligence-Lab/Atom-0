"""DCC Stage 2: split future subgoal frame from loaded image deltas."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

import numpy as np

from openpi import transforms


def main():
    image = np.arange(7 * 4 * 4 * 3, dtype=np.uint8).reshape(7, 4, 4, 3)
    data = {
        "image": {
            "base_0_rgb": image.copy(),
            "left_wrist_0_rgb": image.copy() + 1,
        },
        "image_mask": {
            "base_0_rgb": np.ones((7,), dtype=np.bool_),
            "left_wrist_0_rgb": np.ones((7,), dtype=np.bool_),
        },
    }

    out = transforms.SplitSubgoalFromHistory(image_keys=("base_0_rgb", "left_wrist_0_rgb"))(data)

    assert out["image"]["base_0_rgb"].shape == (6, 4, 4, 3)
    assert out["subgoal_image"]["base_0_rgb"].shape == (4, 4, 3)
    np.testing.assert_array_equal(out["image"]["base_0_rgb"][-1], image[-2])
    np.testing.assert_array_equal(out["subgoal_image"]["base_0_rgb"], image[-1])
    assert out["image_mask"]["base_0_rgb"].shape == (6,)
    assert bool(out["subgoal_image_mask"]["base_0_rgb"])

    print("PASS: DCC subgoal split")


if __name__ == "__main__":
    main()
