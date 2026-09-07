#!/usr/bin/env bash
set -Eeuo pipefail

# Reproduce the historical 20k-step Piper Stage 4 fine-tune from the Stage 3
# step-60000 parameters. This entrypoint intentionally cannot resume or
# overwrite an existing experiment.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_LOG="${STAGE4_DLC_LOG:-${REPO_DIR}/stage4_dlc_fresh20k.log}"
exec > >(tee -a "${TRAIN_LOG}") 2>&1
trap 'rc=$?; echo "FAILED rc=${rc} line=${LINENO} command=${BASH_COMMAND}"; exit "${rc}"' ERR

echo "===== Stage 4 DLC fresh 20k $(date --iso-8601=seconds) ====="
cd "${REPO_DIR}"
EXPECTED_GIT_COMMIT=ebaf24fdac58171f382c86ae276a8daba8d7e925
ACTUAL_GIT_COMMIT="$(git rev-parse HEAD)"
echo "commit=${ACTUAL_GIT_COMMIT}"
if [[ "${ACTUAL_GIT_COMMIT}" != "${EXPECTED_GIT_COMMIT}" ]]; then
  echo "Historical Stage 4 used ${EXPECTED_GIT_COMMIT}; refusing code drift ${ACTUAL_GIT_COMMIT}." >&2
  exit 2
fi
nvidia-smi -L

# Pin the historical data roots/counts instead of inheriting stale job variables.
export ATOM_RLDS_ROOT=/mnt/data/RLDS
unset RLDS_DATA_DIR
unset REALWORLD_PIPER30_BUILDER_DIR REALWORLD_PIPER_2_BUILDER_DIR
unset REALWORLD_PIPER30_TRAIN_EPISODES REALWORLD_PIPER_2_TRAIN_EPISODES
export CHECKPOINT_BASE_DIR="${REPO_DIR}/checkpoints"
export JAX_COMPILATION_CACHE_DIR=/path/to/atom-workspace/cache/jax

if [[ "${RESUME:-0}" != "0" ]]; then
  echo "This launcher is fresh-only; RESUME must be 0." >&2
  exit 2
fi
if [[ "${OVERWRITE:-0}" != "0" ]]; then
  echo "This launcher protects existing checkpoints; OVERWRITE must be 0." >&2
  exit 2
fi

if [[ ! -x .venv/bin/python ]]; then
  echo "Venv missing; rebuilding it in the DLC container."
  bash scripts/setup_aliyun_dsw_env.sh
fi

.venv/bin/python - <<'PY'
import jax

from openpi.cotrain.config import get_config

devices = jax.devices()
print("JAX devices:", devices, flush=True)
if len(devices) != 8:
    raise RuntimeError(f"Expected 8 GPUs, got {len(devices)}: {devices}")

config = get_config("egoscale_stage4_piper_finetune")
dataset_ids = tuple(dataset.uid for dataset in config.data.datasets)
dataset_weights = tuple(dataset.weight for dataset in config.data.datasets)
schedule = config.lr_schedule
expected_schedule = (1_000, 2.5e-5, 30_000, 2.5e-6)
actual_schedule = (
    schedule.warmup_steps,
    schedule.peak_lr,
    schedule.decay_steps,
    schedule.decay_lr,
)
if dataset_ids != ("piper30", "piper2"):
    raise RuntimeError(f"Historical dataset contract changed: {dataset_ids}")
expected_weights = (4927 / 5829, 902 / 5829)
if any(abs(actual - expected) > 1e-12 for actual, expected in zip(dataset_weights, expected_weights, strict=True)):
    raise RuntimeError(f"Historical dataset weights changed: {dataset_weights}")
if config.model.action_dim != 80 or config.model.max_token_len != 384:
    raise RuntimeError(f"Historical Unified80 model contract changed: {config.model}")
if actual_schedule != expected_schedule:
    raise RuntimeError(f"Historical LR schedule changed: {actual_schedule}")
if config.seed != 42 or type(config.freeze_filter).__name__ != "Nothing":
    raise RuntimeError(
        f"Historical trainability contract changed: seed={config.seed}, "
        f"freeze_filter={config.freeze_filter}"
    )
print(
    "Historical Stage 4 contract verified:",
    dataset_ids,
    actual_schedule,
    "seed=42 full-parameter",
    flush=True,
)
PY

# Match the deleted historical Stage 4 run exactly: load only model params from
# Stage 3 step 60000, then create a new optimizer/EMA state at training step 0.
DEFAULT_MODEL_PARAMS_PATH="${REPO_DIR}/checkpoints/egoscale_stage3_real_robot_fix/stage3_real_robot_fix_from_stage2_20000_b200_3x8_b1536_20260812_v1/60000/params"
export MODEL_PARAMS_PATH="${DEFAULT_MODEL_PARAMS_PATH}"
export PARAMS_PATH="${MODEL_PARAMS_PATH}"

test -f "${PARAMS_PATH}/manifest.ocdbt"
test -f "${PARAMS_PATH}/_METADATA"

: "${WANDB_API_KEY:?Inject WANDB_API_KEY through a DLC Secret environment variable}"
export WANDB_ENTITY="${WANDB_ENTITY:-your-wandb-team}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_ENABLED=1

export EXP_NAME=stage4_piper_finetune_from_stage3_60000_dlc_1x8_b512_repro20k_20260820_v1
export FSDP_DEVICES=4
export BATCH_SIZE=512
export VAL_BATCH_SIZE=96
export NUM_TRAIN_STEPS=20000
export LOG_INTERVAL=100
export SAVE_INTERVAL=5000
export EVAL_INTERVAL=1000
export NUM_VAL_BATCHES=10
export NUM_ACTION_MSE_BATCHES=2
export RUN_ACTION_MSE=1
export SHUFFLE_BUFFER_SIZE=50000
export DATA_NUM_PARALLEL_READS=1
export DATA_NUM_PARALLEL_CALLS=2
export CHECKPOINT_PARAMS_ONLY=0
export RESUME=0
export OVERWRITE=0

EXPERIMENT_DIR="${CHECKPOINT_BASE_DIR}/egoscale_stage4_piper_finetune/${EXP_NAME}"
if [[ -e "${EXPERIMENT_DIR}" ]]; then
  echo "Refusing to reuse existing experiment directory: ${EXPERIMENT_DIR}" >&2
  exit 2
fi

echo "Fresh model params: ${PARAMS_PATH}"
echo "New experiment: ${EXPERIMENT_DIR}"
echo "Launching ${EXP_NAME} for exactly ${NUM_TRAIN_STEPS} fresh optimizer steps"

export STAGE=stage4_piper_finetune
exec bash scripts/run_egoscale_stage.sh \
  --val-batch-size "${VAL_BATCH_SIZE}" \
  --val-flow-loss-mode fixed_seed \
  --seed 42 \
  --keep-period 19999
