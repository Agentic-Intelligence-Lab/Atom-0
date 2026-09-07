#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DEFAULTS=${ROOT_DIR}/config/eval_defaults.env
if [[ -f "${DEFAULTS}" ]]; then
  set -a
  source "${DEFAULTS}"
  set +a
fi

: "${CHECKPOINT_ROOT:?Set CHECKPOINT_ROOT to a checkpoint trained with the task-disjoint Piper30 builder}"
CHECKPOINT_STEP=${CHECKPOINT_STEP:-20000}
OPENPI_ROOT=${OPENPI_ROOT:-/path/to/Atom-0}
NORM_STATS_PATH=${NORM_STATS_PATH:-${OPENPI_ROOT}/assets/cotrain_real_only/piper30}
DATASET_DIR=${DATASET_DIR:-/mnt/data/RLDS/realworld_piper_task_split/piper_s14_a14_fps30_c4_ee_pose_cam_front_cam_high_cam_left_wrist_cam_right_wrist/realworld_piper_infidata/1.1.0}
CONFIG_NAME=${CONFIG_NAME:-cotrain_real_only}
SPLIT=${SPLIT:-seen_test}
EPISODES=${EPISODES:-0}
ANCHORS_PER_EPISODE=${ANCHORS_PER_EPISODE:-20}
ACTIONS_PER_INFERENCE=${ACTIONS_PER_INFERENCE:-8}
FLOW_LOSS_SAMPLES=${FLOW_LOSS_SAMPLES:-4}
SEED=${SEED:-42}
DEVICE=${DEVICE:-cuda}
OUTPUT_ROOT=${OUTPUT_ROOT:-${ROOT_DIR}/results}
PYTHON_BIN=${PYTHON_BIN:-/path/to/Atom-0/.venv/bin/python}

STEP_PADDED=$(printf "%06d" "${CHECKPOINT_STEP}")
CHECKPOINT_DIR=${CHECKPOINT_ROOT}/${CHECKPOINT_STEP}
OUTPUT_DIR=${OUTPUT_ROOT}/step_${STEP_PADDED}
ANCHOR_MANIFEST=${ROOT_DIR}/manifests/${SPLIT}_open_loop_h${ACTIONS_PER_INFERENCE}_seed${SEED}.jsonl

mkdir -p "${OUTPUT_DIR}"
exec "${PYTHON_BIN}" "${ROOT_DIR}/scripts/evaluate_validation.py" \
  --openpi-root "${OPENPI_ROOT}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --norm-stats-path "${NORM_STATS_PATH}" \
  --dataset-dir "${DATASET_DIR}" \
  --config-name "${CONFIG_NAME}" \
  --split "${SPLIT}" \
  --episodes "${EPISODES}" \
  --anchors-per-episode "${ANCHORS_PER_EPISODE}" \
  --actions-per-inference "${ACTIONS_PER_INFERENCE}" \
  --flow-loss-samples "${FLOW_LOSS_SAMPLES}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --anchor-manifest "${ANCHOR_MANIFEST}" \
  --output-dir "${OUTPUT_DIR}" \
  --generate-report \
  "$@"
