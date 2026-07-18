#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" tests/mem/stage1_short_memory_forward.py
"${PYTHON_BIN}" tests/mem/stage2_policy_history_buffer.py
"${PYTHON_BIN}" tests/mem/stage3_train_step_smoke.py
"${PYTHON_BIN}" tests/mem/stage4_long_memory_interface.py
"${PYTHON_BIN}" tests/mem/stage5_single_batch_overfit.py

echo "MEM short-term local validation passed."
