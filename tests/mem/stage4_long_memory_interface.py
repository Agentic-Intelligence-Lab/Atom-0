"""MEM Stage 4: long-term memory summary prompt interface."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from openpi import transforms
from openpi.training import config as _config


def main():
    config = _config.get_config("pi05_mem_long_interface")
    assert config.model.long_memory_enabled
    assert config.model.max_token_len >= 320
    transform = transforms.PrependMemorySummaryToPrompt()
    out = transform({"prompt": "clean the table", "memory_summary": "I put the plate away."})
    assert out["prompt"] == "Memory: I put the plate away.\nTask: clean the table"
    print("PASS: MEM long-memory prompt interface")


if __name__ == "__main__":
    main()
