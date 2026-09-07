# Robot-only baselines

The manuscript uses one robot-only baseline per architecture family. Both VLA
and WAM training implementations are included; exact corpus/run reproduction
remains subject to the [paper alignment audit](../docs/paper_alignment.md).

## VLA baseline

Use the root project with `cotrain_real_robot_fix` for robot-only pre-training,
followed by `cotrain_real_only_unified80_aliyun_recipe` for Piper post-training.
The paper initializes the pre-training stage from base PaliGemma VLM weights.

```bash
# From the repository root, with the root environment installed.
export ATOM_RLDS_ROOT=/path/to/rlds
export ASSETS_BASE_DIR="$PWD/assets"
export CHECKPOINT_BASE_DIR=/path/to/output/baseline
export PARAMS_PATH=/path/to/paligemma-pt-224.npz

CONFIG_NAME=cotrain_real_robot_fix EXP_NAME=vla_robot_only \
MODE=train FSDP_DEVICES=4 BATCH_SIZE=1536 NUM_TRAIN_STEPS=97728 \
WARMUP_STEPS=5000 DECAY_STEPS=97728 PEAK_LR=1e-6 DECAY_LR=1e-7 \
EVAL_INTERVAL=1000 SAVE_INTERVAL=10000 WANDB_ENABLED=0 \
bash scripts/train_cotrain_baige.sh

export PARAMS_PATH=/path/to/robot-pretrain/checkpoint/params
CONFIG_NAME=cotrain_real_only_unified80_aliyun_recipe \
EXP_NAME=vla_robot_only_piper_ft MODE=train \
FSDP_DEVICES=4 BATCH_SIZE=512 NUM_TRAIN_STEPS=20000 \
WARMUP_STEPS=1000 DECAY_STEPS=30000 PEAK_LR=2.5e-5 DECAY_LR=2.5e-6 \
EVAL_INTERVAL=1000 SAVE_INTERVAL=5000 WANDB_ENABLED=0 \
bash scripts/train_cotrain_baige.sh
```

Configure the distributed environment for the intended hardware: the paper
reports 3×8 B200 GPUs for robot pre-training and 1×8 for post-training. The
commands specify the recipe, not cluster provisioning. Check dataset paths,
normalization provenance, and initialization loading reports before training.

The [auxiliary VLA baseline](../experiments/auxiliary_vla/baseline/) belongs to the
KI/MEM/DCC experiment ladder. It is not the paper's robot-only pre-training
baseline. The legacy32 Piper configurations are separate action-space ablations.

## WAM baseline

Use the independent [Atom-WAM project](../routes/atom_wam/README.md): select
`wam-cross-robot` instead of `wam-cross-fix`, then `wam-cross-piper-ft` initialized
from the resulting paper-aligned robot-only checkpoint. Keep the model,
normalization protocol and downstream evaluation fixed for the ego-data
comparison. The source robot-only preset contains 15 datasets; the manuscript's
exact final corpus selection and WAM run schedule still require confirmation.
Old 32-step/proprio-conditioned checkpoints are not this modified 50-step model.
