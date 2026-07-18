#!/usr/bin/env bash
set -euo pipefail

python -m py_compile \
  src/openpi/models/model.py \
  src/openpi/models/pi0.py \
  src/openpi/models/pi0_config.py \
  src/openpi/transforms.py \
  src/openpi/policies/libero_policy.py \
  src/openpi/training/config.py \
  src/openpi/training/data_loader.py

python tests/dcc/stage1_context_transforms.py
python tests/dcc/stage2_subgoal_split.py
python tests/dcc/stage3_train_step_smoke.py

git diff --check
