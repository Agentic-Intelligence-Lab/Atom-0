"""MEM Stage 2: policy-side history buffer sanity."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

import numpy as np

from openpi import transforms


def main():
    transform = transforms.HistoryBufferTransform(history_length=4)
    for value in [1, 2, 3, 4, 5]:
        out = transform(
            {
                "state": np.array([value], dtype=np.float32),
                "image": {"base_0_rgb": np.full((2, 2, 3), value, dtype=np.uint8)},
                "image_mask": {"base_0_rgb": np.True_},
            }
        )
    assert out["state_history"][:, 0].tolist() == [2.0, 3.0, 4.0, 5.0]
    assert out["image"]["base_0_rgb"][:, 0, 0, 0].tolist() == [2, 3, 4, 5]
    print("PASS: MEM policy history buffer")


if __name__ == "__main__":
    main()
