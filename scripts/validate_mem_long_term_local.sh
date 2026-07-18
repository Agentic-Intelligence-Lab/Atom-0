#!/usr/bin/env bash
set -euo pipefail

python -m pytest src/openpi/models/model_test.py src/openpi/models/pi0_mem_test.py src/openpi/transforms_test.py -q
python tests/mem/stage1_short_memory_forward.py
python tests/mem/stage2_policy_history_buffer.py
python tests/mem/stage3_train_step_smoke.py
python tests/mem/stage4_long_memory_interface.py
python tests/mem/stage5_single_batch_overfit.py
python tests/mem/stage6_summary_ce_overfit.py
python tests/mem/stage7_memory_action_counterfactual.py
python tests/mem/stage8_hidden_object_recall.py
python tests/mem/stage9_counting_probe.py
python tests/mem/stage10_closed_loop_memory_state.py
