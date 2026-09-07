"""DCC Stage 1: context prompt/dropout transform checks."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

import numpy as np

from openpi import transforms


class FakeTokenizer:
    def __init__(self, max_len: int):
        self._max_len = max_len
        self.seen: list[str] = []

    def tokenize(self, prompt: str, state=None):
        del state
        self.seen.append(prompt)
        token_count = min(len(prompt.split()), self._max_len)
        tokens = np.arange(self._max_len, dtype=np.int32)
        mask = np.zeros((self._max_len,), dtype=np.bool_)
        mask[:token_count] = True
        return tokens, mask


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

    # With no dropout, all InfiData-style fields become independent token segments.
    sample = _make_sample()
    sample = transforms.ApplyDiverseContextDropout(
        subgoal_keep_prob=1.0,
        subtask_drop_when_subgoal=0.0,
        metadata_drop_prob=0.0,
        metadata_field_drop_prob=0.0,
    )(sample)
    metadata_tokenizer = FakeTokenizer(16)
    control_tokenizer = FakeTokenizer(8)
    subtask_tokenizer = FakeTokenizer(12)
    sample = transforms.TokenizeDiverseContextSegments(
        metadata_tokenizer=metadata_tokenizer,
        control_tokenizer=control_tokenizer,
        subtask_tokenizer=subtask_tokenizer,
    )(sample)
    assert sample["prompt"] == "put candies in bowl"
    assert metadata_tokenizer.seen == ["Metadata: quality=5; speed=normal; mistake=False; success=True"]
    assert control_tokenizer.seen == ["Control: joint"]
    assert subtask_tokenizer.seen == ["Subtask: grasp or manipulate the object"]
    assert sample["dcc_metadata_tokens"].shape == (16,)
    assert sample["dcc_control_tokens"].shape == (8,)
    assert sample["dcc_subtask_tokens"].shape == (12,)
    assert "subtask" not in sample

    # With forced dropout, the subgoal mask is retained structurally but disabled, and metadata segment is empty.
    sample = transforms.ApplyDiverseContextDropout(
        subgoal_keep_prob=0.0,
        metadata_drop_prob=1.0,
    )(_make_sample())
    assert not bool(sample["subgoal_image_mask"]["base_0_rgb"])
    metadata_tokenizer = FakeTokenizer(16)
    sample = transforms.TokenizeDiverseContextSegments(
        metadata_tokenizer=metadata_tokenizer,
        control_tokenizer=FakeTokenizer(8),
        subtask_tokenizer=FakeTokenizer(12),
    )(sample)
    assert metadata_tokenizer.seen == ["Metadata: quality=none; speed=none; mistake=none; success=none"]

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
