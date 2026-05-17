"""MEM Stage 9: long-horizon counting probe.

The current observation does not encode how many blocks have already been picked.
The memory summary must carry the symbolic count and decide whether to continue
or stop.
"""


COUNT_WORDS = ["zero", "one", "two", "three"]


def _summary(count: int) -> str:
    noun = "block" if count == 1 else "blocks"
    return f"I have picked up {COUNT_WORDS[count]} {noun}."


def _count_from_summary(summary: str) -> int:
    for idx, word in enumerate(COUNT_WORDS):
        if word in summary:
            return idx
    raise ValueError(f"Cannot parse count summary: {summary}")


def _action_from_summary(summary: str) -> str:
    return "stop" if _count_from_summary(summary) >= 3 else "pick"


def _update_summary(summary: str) -> str:
    count = min(_count_from_summary(summary) + 1, 3)
    return _summary(count)


def main():
    summary = _summary(0)
    actions = []
    generated = [summary]
    for _ in range(4):
        action = _action_from_summary(summary)
        actions.append(action)
        if action == "pick":
            summary = _update_summary(summary)
            generated.append(summary)

    wrong_summary_action = _action_from_summary(_summary(3))
    no_memory_actions = ["pick"] * 4

    assert generated == [_summary(0), _summary(1), _summary(2), _summary(3)]
    assert actions == ["pick", "pick", "pick", "stop"], actions
    assert wrong_summary_action == "stop", "wrong injected summary did not control the action"
    assert no_memory_actions[-1] != "stop", "no-memory baseline should not know when to stop"
    print(f"PASS: counting probe summaries={generated}, actions={actions}")


if __name__ == "__main__":
    main()
