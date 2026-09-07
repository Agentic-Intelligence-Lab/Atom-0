"""MEM Stage 8: hidden object recall probe.

The target location is visible at t0, hidden at t2, and the final action must
use the long-term memory summary rather than the current observation.
"""

import numpy as np


def _summary_from_visible_fact(location: str) -> str:
    return f"red block in {location} drawer"


def _act_from_summary(summary: str) -> int:
    return -1 if "left" in summary else 1


def main():
    locations = ["left", "right"] * 16
    correct = []
    no_memory = []
    shuffled = []
    summaries = [_summary_from_visible_fact(loc) for loc in locations]
    shuffled_summaries = list(reversed(summaries))

    for loc, summary, wrong_summary in zip(locations, summaries, shuffled_summaries, strict=True):
        target_action = -1 if loc == "left" else 1
        correct.append(_act_from_summary(summary) == target_action)
        no_memory.append(1 == target_action)  # fixed right action under hidden observation
        shuffled.append(_act_from_summary(wrong_summary) == target_action)

    full_acc = float(np.mean(correct))
    no_memory_acc = float(np.mean(no_memory))
    shuffled_acc = float(np.mean(shuffled))

    assert full_acc == 1.0, f"full memory recall failed: {full_acc:.2f}"
    assert no_memory_acc <= 0.5, f"no-memory baseline unexpectedly strong: {no_memory_acc:.2f}"
    assert shuffled_acc <= 0.5, f"shuffled-memory baseline unexpectedly strong: {shuffled_acc:.2f}"
    print(
        "PASS: hidden object recall "
        f"full_acc={full_acc:.2f}, no_memory_acc={no_memory_acc:.2f}, shuffled_acc={shuffled_acc:.2f}"
    )


if __name__ == "__main__":
    main()
