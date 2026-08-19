#!/usr/bin/env bash
set -euo pipefail

DEPLOY_ROOT=/home/ps/Documents/zhengdongchen/b200_wam_cross_piper_v4_99999
OPENPI_ROOT="${OPENPI_ROOT:-$DEPLOY_ROOT/source/Atom-0}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$DEPLOY_ROOT/checkpoint/99999}"
NORM_STATS_PATH="${NORM_STATS_PATH:-$OPENPI_ROOT/assets/cotrain_real_robot_ego_fix/piper30/norm_stats.json}"
CONFIG_NAME="${CONFIG_NAME:-wam-cross-piper}"
SERVER_SCRIPT="$DEPLOY_ROOT/deploy/serve_fastwam_piper.py"
PYTHON_BIN="${OPENPI_PYTHON_BIN:-/home/ps/Documents/kai0/.venv/bin/python}"

export PYTHONPATH="$OPENPI_ROOT/src:$OPENPI_ROOT/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
export DIFFSYNTH_MODEL_BASE_PATH="$OPENPI_ROOT/checkpoints/fastwam"
export HF_HOME="$DIFFSYNTH_MODEL_BASE_PATH/hf_cache"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export FASTWAM_SKIP_PRETRAIN_INIT=true

args=(
  "$SERVER_SCRIPT"
  --openpi-root "$OPENPI_ROOT"
  --checkpoint-dir "$CHECKPOINT_DIR"
  --norm-stats-path "$NORM_STATS_PATH"
  --config-name "$CONFIG_NAME"
  --prompt "${PROMPT:-}"
  --host "${POLICY_HOST:-127.0.0.1}"
  --port "${POLICY_PORT:-8011}"
  --device "${DEVICE:-cuda}"
  --actions-per-inference "${ACTIONS_PER_INFERENCE:-32}"
  --num-inference-steps "${NUM_INFERENCE_STEPS:-20}"
  --seed "${INFERENCE_SEED:-42}"
  --rtc-delay-steps "${RTC_DELAY_STEPS:-3}"
  --rtc-execution-horizon "${RTC_EXECUTION_HORIZON:-32}"
  --rtc-soft-mask-decay "${RTC_SOFT_MASK_DECAY:-0.6}"
  --rtc-guidance-scale "${RTC_GUIDANCE_SCALE:-1.0}"
)
[[ "${RTC_ENABLED:-false}" == "true" ]] && args+=(--rtc-enabled)
[[ "${VALIDATE_ONLY:-false}" == "true" ]] && args+=(--validate-only)

exec "$PYTHON_BIN" "${args[@]}" "$@"
