#!/usr/bin/env bash
set -Eeuo pipefail

# Fresh Stage 2 aligned-data fine-tune initialized from the Pi0.5 all-robot
# checkpoint at step 97727. Submit this script as a single-node, 8-GPU DLC
# command. It never resumes or overwrites an existing experiment.

REPO_DIR=/path/to/Atom-0
TRAIN_LOG="${STAGE2_DLC_LOG:-/path/to/atom-workspace/logs/stage2_pi05_all_robot_dlc.log}"
mkdir -p "$(dirname "${TRAIN_LOG}")"
exec > >(tee -a "${TRAIN_LOG}") 2>&1
trap 'rc=$?; echo "FAILED rc=${rc} line=${LINENO} command=${BASH_COMMAND}"; exit "${rc}"' ERR

echo "===== Stage 2 Pi0.5 all-robot DLC $(date --iso-8601=seconds) ====="
cd "${REPO_DIR}"

EXPECTED_GIT_COMMIT=ebaf24fdac58171f382c86ae276a8daba8d7e925
ACTUAL_GIT_COMMIT="$(git rev-parse HEAD)"
echo "commit=${ACTUAL_GIT_COMMIT}"
if [[ "${ACTUAL_GIT_COMMIT}" != "${EXPECTED_GIT_COMMIT}" ]]; then
  echo "Expected ${EXPECTED_GIT_COMMIT}; refusing code drift ${ACTUAL_GIT_COMMIT}." >&2
  exit 2
fi

GPU_COUNT="$(nvidia-smi -L | wc -l)"
if [[ "${GPU_COUNT}" != "8" ]]; then
  echo "Expected exactly 8 GPUs in this DLC node, got ${GPU_COUNT}." >&2
  exit 2
fi
nvidia-smi -L

export ATOM_RLDS_ROOT=/mnt/data/RLDS
export ATOM_SELF_COLLECTED_ALIGNED_RLDS_ROOT=/mnt/data/RLDS/AtomAligned_full
export ASSETS_BASE_DIR="${REPO_DIR}/assets"
export CHECKPOINT_BASE_DIR="${REPO_DIR}/checkpoints"
export JAX_COMPILATION_CACHE_DIR=/path/to/atom-workspace/cache/jax
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export TF_FORCE_GPU_ALLOW_GROWTH=true
mkdir -p "${JAX_COMPILATION_CACHE_DIR}"

DATASET_IDS=(
  aligned_hangzhou_human_right
  aligned_shenzhen_human_bimanual
  aligned_hangzhou_robot_right
  aligned_shenzhen_robot_bimanual
)
for dataset_id in "${DATASET_IDS[@]}"; do
  builder_dir="${ATOM_SELF_COLLECTED_ALIGNED_RLDS_ROOT}/atom_aligned_rlds/${dataset_id}/2.0.0"
  test -f "${builder_dir}/dataset_info.json"
  test -f "${builder_dir}/features.json"
  test -f "${ASSETS_BASE_DIR}/egoscale_stage2_self_collected_aligned/${dataset_id}/norm_stats.json"
  test -f "${ASSETS_BASE_DIR}/egoscale_stage2_self_collected_aligned/${dataset_id}/unified_action_space.json"
done
test -f "${ASSETS_BASE_DIR}/egoscale_stage2_self_collected_aligned/action_chunk_metadata.json"

export MODEL_PARAMS_PATH=/path/to/Atom-0/checkpoints/cotrain_real_robot_fix/cotrain_real_robot_fix_b200_0719/97727/params
export PARAMS_PATH="${MODEL_PARAMS_PATH}"
test -f "${PARAMS_PATH}/manifest.ocdbt"
test -f "${PARAMS_PATH}/_METADATA"

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
export PYTHON_BIN="${REPO_DIR}/.venv/bin/python"

export WANDB_ENTITY="${WANDB_ENTITY:-your-wandb-team}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_ENABLED=1
if [[ "${PREFLIGHT_ONLY:-0}" != "1" ]]; then
  : "${WANDB_API_KEY:?Inject WANDB_API_KEY through a DLC Secret environment variable}"
fi

export STAGE=stage2_aligned
export EXP_NAME=pi05_all_robot
export FSDP_DEVICES=8
export BATCH_SIZE=256
export NUM_TRAIN_STEPS=50000
export LOG_INTERVAL=100
export SAVE_INTERVAL=5000
export EVAL_INTERVAL=1000
export NUM_VAL_BATCHES=10
export NUM_ACTION_MSE_BATCHES=2
export RUN_ACTION_MSE=1
export SHUFFLE_BUFFER_SIZE=50000
export DATA_NUM_PARALLEL_READS=4
export DATA_NUM_PARALLEL_CALLS=8
export CHECKPOINT_PARAMS_ONLY=0
export RESUME=0
export OVERWRITE=0

EXPERIMENT_DIR="${CHECKPOINT_BASE_DIR}/egoscale_stage2_aligned/${EXP_NAME}"
if [[ -e "${EXPERIMENT_DIR}" ]]; then
  echo "Refusing to reuse existing experiment directory: ${EXPERIMENT_DIR}" >&2
  exit 2
fi

echo "Fresh model params: ${PARAMS_PATH}"
echo "Aligned RLDS root: ${ATOM_SELF_COLLECTED_ALIGNED_RLDS_ROOT}"
echo "New experiment: ${EXPERIMENT_DIR}"

exec bash scripts/run_egoscale_stage.sh \
  --project-name Atom0 \
  --val-batch-size 96 \
  --val-flow-loss-mode fixed_seed \
  --seed 42
