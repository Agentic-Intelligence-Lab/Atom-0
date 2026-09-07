"""MEM Stage 10: closed-loop memory state integration probe.

This synthetic runner validates the intended policy-level contract: the policy
uses the old summary for action selection, periodically generates a new summary,
stores it, and later acts on that stored long-term memory.
"""


class SyntheticLongMemoryPolicy:
    def __init__(self, *, generation_enabled: bool = True, update_interval: int = 1):
        self.generation_enabled = generation_enabled
        self.update_interval = update_interval
        self.memory_summary = ""
        self.memory_step = 0

    def reset_memory(self):
        self.memory_summary = ""
        self.memory_step = 0

    def get_memory_summary(self):
        return self.memory_summary

    def infer(self, obs):
        action = self._act(obs)
        if self.generation_enabled and self.memory_step % self.update_interval == 0:
            new_summary = self._generate_summary(obs)
            if new_summary:
                self.memory_summary = new_summary
        self.memory_step += 1
        return action

    def _generate_summary(self, obs):
        if "visible_location" not in obs:
            return ""
        return f"red block in {obs['visible_location']} drawer"

    def _act(self, obs):
        if obs.get("query_hidden_target", False):
            if "left" in self.memory_summary:
                return "go_left"
            if "right" in self.memory_summary:
                return "go_right"
            return "unknown"
        return "continue"


def main():
    full = SyntheticLongMemoryPolicy(generation_enabled=True)
    disabled = SyntheticLongMemoryPolicy(generation_enabled=False)
    reset_mid_episode = SyntheticLongMemoryPolicy(generation_enabled=True)

    first_obs = {"visible_location": "left"}
    query_obs = {"query_hidden_target": True}

    assert full.infer(first_obs) == "continue"
    assert "left" in full.get_memory_summary()
    assert full.infer(query_obs) == "go_left"

    assert disabled.infer(first_obs) == "continue"
    assert disabled.get_memory_summary() == ""
    assert disabled.infer(query_obs) == "unknown"

    assert reset_mid_episode.infer(first_obs) == "continue"
    reset_mid_episode.reset_memory()
    assert reset_mid_episode.infer(query_obs) == "unknown"

    print("PASS: closed-loop memory state full=go_left, disabled=unknown, reset=unknown")


if __name__ == "__main__":
    main()
