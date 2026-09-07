# Atom-DH: domain-specific action heads

This is an independent OpenPI project. Run commands from `routes/atom_dh` and
use this directory's `.venv`; do not mix it with the root Atom-CL environment.

## Implementation

- [`src/openpi/models/pi0.py`](src/openpi/models/pi0.py): shared backbone and separately routed ego/robot action projections.
- [`src/openpi/cotrain/ot_loss.py`](src/openpi/cotrain/ot_loss.py): trajectory-aware Soft-DTW and Sinkhorn alignment.
- [`src/openpi/cotrain/config.py`](src/openpi/cotrain/config.py): data recipes and dual-head/OT options.
- [`scripts/train_cotrain.py`](scripts/train_cotrain.py): JAX training and validation.
- [`assets`](assets/): small normalization and action-space metadata, without model weights.

`cotrain_real_robot_ego_fix` enables the ego action head. In this imported
snapshot it also enables OT by default. Use `OT_ENABLED=0` with the launcher
below for dual-head co-training without OT. At robot deployment, the robot
projection provides the control output.

## Installation and inputs

```bash
cd routes/atom_dh
GIT_LFS_SKIP_SMUDGE=1 uv sync --python 3.11 --group rlds
export RLDS_DATA_DIR=/path/to/rlds
export ASSETS_BASE_DIR="$PWD/assets"
export CHECKPOINT_BASE_DIR=/path/to/output/atom-dh
export PARAMS_PATH=/path/to/paligemma-pt-224.npz
```

The paper's pre-training initializes from the PaliGemma VLM `.npz`. The code
also accepts compatible Orbax parameters. Its shape-safe loader reports loaded,
missing, and mismatched parameters; inspect those counts when transferring
between a dual-head model and a robot-only fine-tuning configuration.

## Co-training

```bash
# Short single-GPU smoke run; use a new experiment name for each run.
CONFIG_NAME=cotrain_real_robot_ego_fix EXP_NAME=atom_dh_smoke \
MODE=smoke NPROC_PER_NODE=1 FSDP_DEVICES=1 BATCH_SIZE=2 \
VAL_BATCH_SIZE=2 OT_ENABLED=0 \
bash scripts/train_cotrain_baige.sh
```

For full training, set `MODE=train`, the hardware/device topology, batch size,
training updates, and learning-rate schedule explicitly. The launcher retains
the original distributed configuration and computes its default update count
from the configured sample count and global batch size.

Enable the OT variant with `OT_ENABLED=1`. Its fixed-quota data mixture can be
controlled with `MIX_MODE`, `P_OTHER`, `LAB_IN_ALIGN`, `O_EV_HUMAN`, and
`ALIGN_BY_DURATION`; the regularizer uses `OT_ALPHA`. OT requires enough valid
human and robot examples in the global batch. The two-sample smoke example
above does not validate OT training.

## Piper post-training

```bash
export PARAMS_PATH=/path/to/pretrained/checkpoint/params
CONFIG_NAME=cotrain_real_only_unified80_aliyun_recipe \
EXP_NAME=atom_dh_piper_ft MODE=train NPROC_PER_NODE=8 \
FSDP_DEVICES=4 BATCH_SIZE=512 NUM_TRAIN_STEPS=20000 \
WARMUP_STEPS=1000 DECAY_STEPS=30000 \
PEAK_LR=2.5e-5 DECAY_LR=2.5e-6 \
EVAL_INTERVAL=1000 SAVE_INTERVAL=5000 OT_ENABLED=0 WANDB_ENABLED=0 \
bash scripts/train_cotrain_baige.sh
```

Post-training uses Piper30/Piper2 and the robot control output. The explicit
learning-rate overrides matter: the general launcher otherwise defaults to
the pre-training learning rate. Supply your own dataset builders and weights.

## Inference integration

The model's `sample_actions` implementation and policy/client infrastructure
are included. The upstream `scripts/serve_policy.py` does not resolve the
co-training configuration registry. A Piper server must explicitly load the
co-training configuration and apply its matching per-dataset normalization and
native/Unified80 mapping. A turnkey co-training server is not supplied here.

This snapshot is not a released checkpoint or a guarantee of reproducing a
specific manuscript row. See [release notes](../../docs/release.md).
