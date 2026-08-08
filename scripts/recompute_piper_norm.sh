#!/usr/bin/env bash
# Recompute per-dataset 80D norm stats for wam-cross-piper (piper30 / piper2).
#
# Uses the lightweight full-split pipeline (no image decode):
#   RLDS -> unified 80D scatter -> delta actions -> RunningStats
#
# Output layout (per dataset uid):
#   assets/cotrain_real_robot_ego_fix/<uid>/norm_stats.json
#   assets/cotrain_real_robot_ego_fix/<uid>/norm_stats_meta.json
#   assets/cotrain_real_robot_ego_fix/<uid>/unified_action_space.json
#
# Examples:
#   # Recompute piper30 only (1.1.0 task_split train split)
#   bash scripts/recompute_piper_norm.sh
#
#   # Recompute both piper30 + piper2
#   DATASET_ID=piper30,piper2 bash scripts/recompute_piper_norm.sh
#
#   # Quick smoke (first ~1M frames, no meta for full preflight)
#   MODE=probe MAX_FRAMES=1000000 bash scripts/recompute_piper_norm.sh
#
# After full recompute, verify:
#   .venv/bin/python scripts/preflight_cotrain_baige.py wam-cross-piper --assets-base assets

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

# --- paths (override on your host if needed) ---
export RLDS_DATA_DIR="${RLDS_DATA_DIR:-/mnt/bos/bo23lu}"
export REALWORLD_PIPER_2_BUILDER_DIR="${REALWORLD_PIPER_2_BUILDER_DIR:-${RLDS_DATA_DIR}/realworld_piper_2/realworld_piper_infidata/1.0.0}"
export REALWORLD_PIPER_2_TRAIN_EPISODES="${REALWORLD_PIPER_2_TRAIN_EPISODES:-902}"

CONFIG_NAME="${CONFIG_NAME:-wam-cross-piper}"
OUTPUT_ASSETS_DIR="${OUTPUT_ASSETS_DIR:-${ROOT}/assets/cotrain_real_robot_ego_fix}"
DATASET_ID="${DATASET_ID:-piper30}"          # comma-separated, e.g. piper30,piper2
MODE="${MODE:-full}"                          # full | probe
MAX_FRAMES="${MAX_FRAMES:-1000000}"
NUM_PARALLEL_READS="${NUM_PARALLEL_READS:-1}"
NUM_PARALLEL_CALLS="${NUM_PARALLEL_CALLS:-2}"
OVERWRITE="${OVERWRITE:-1}"                   # 1 = --overwrite, 0 = skip existing

PYTHON="${PYTHON:-${ROOT}/.venv/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="$(command -v python3)"
fi

echo "ROOT=${ROOT}"
echo "CONFIG_NAME=${CONFIG_NAME}"
echo "OUTPUT_ASSETS_DIR=${OUTPUT_ASSETS_DIR}"
echo "DATASET_ID=${DATASET_ID}"
echo "MODE=${MODE}"
echo "RLDS_DATA_DIR=${RLDS_DATA_DIR}"
echo "REALWORLD_PIPER_2_BUILDER_DIR=${REALWORLD_PIPER_2_BUILDER_DIR}"

PIPER30_BUILDER="${RLDS_DATA_DIR}/realworld_piper_task_split/piper_s14_a14_fps30_c4_ee_pose_cam_front_cam_high_cam_left_wrist_cam_right_wrist/realworld_piper_infidata/1.1.0"
for required in \
  "${PIPER30_BUILDER}/dataset_info.json" \
  "${REALWORLD_PIPER_2_BUILDER_DIR}/dataset_info.json"
do
  if [[ ! -f "${required}" ]]; then
    echo "ERROR: missing ${required}" >&2
    exit 1
  fi
done

OVERWRITE_FLAG=()
if [[ "${OVERWRITE}" == "1" ]]; then
  OVERWRITE_FLAG=(--overwrite)
fi

if [[ "${MODE}" == "full" ]]; then
  "${PYTHON}" scripts/compute_cotrain_full_norm_stats_light.py \
    --config-name "${CONFIG_NAME}" \
    --output-assets-dir "${OUTPUT_ASSETS_DIR}" \
    --dataset-id "${DATASET_ID}" \
    --rlds-data-dir "${RLDS_DATA_DIR}" \
    --num-parallel-reads "${NUM_PARALLEL_READS}" \
    --num-parallel-calls "${NUM_PARALLEL_CALLS}" \
    "${OVERWRITE_FLAG[@]}"
else
  # Faster probe: capped frames; does not write norm_stats_meta.json.
  IFS=',' read -ra IDS <<< "${DATASET_ID}"
  for uid in "${IDS[@]}"; do
    "${PYTHON}" scripts/compute_cotrain_norm_stats_light.py \
      --config-name "${CONFIG_NAME}" \
      --exp-name "recompute-${uid}-probe" \
      --dataset-id "${uid}" \
      --max-frames "${MAX_FRAMES}" \
      --rlds-data-dir "${RLDS_DATA_DIR}" \
      "${OVERWRITE_FLAG[@]}"
  done
  echo "NOTE: probe mode writes norm_stats.json only (no norm_stats_meta.json)."
  echo "      Run MODE=full for production stats + preflight-compatible metadata."
fi

echo "Done. Stats under: ${OUTPUT_ASSETS_DIR}/{${DATASET_ID//,/,}}"
