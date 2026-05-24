#!/bin/bash
# Evaluate KI-V3 checkpoints on LIBERO suites.
#
# Defaults target the 10k-step checkpoints:
#   KI:    /mnt/data/xule/openpi/checkpoints/pi05_ki_libero/test1_ki_v3_full_2h100_b8_s0/10000
#   no-KI: /mnt/data/xule/openpi/checkpoints/pi05_no_ki_libero/no_ki_v3_full_2h100_b8_s0/10000
#
# Environment overrides:
#   PROJECT_DIR=/mnt/data/xule/openpi
#   STEP=10000
#   NUM_TRIALS=50
#   SEED=7
#   SAVE_VIDEO=1
#   REPLAN_STEPS=5
#   SUITES="libero_spatial libero_object libero_goal libero_10"
#   RUNS="ki no_ki"

set -euo pipefail

export PATH="/root/.local/bin:/root/miniforge3/bin:$PATH"

PROJECT_DIR="${PROJECT_DIR:-/mnt/data/xule/openpi}"
CACHE_DIR="${CACHE_DIR:-/mnt/data/cache}"
STEP="${STEP:-10000}"
NUM_TRIALS="${NUM_TRIALS:-50}"
SEED="${SEED:-7}"
PORT="${PORT:-8000}"
SAVE_VIDEO="${SAVE_VIDEO:-1}"
REPLAN_STEPS="${REPLAN_STEPS:-5}"
RUNS="${RUNS:-ki no_ki}"
SUITES="${SUITES:-libero_spatial libero_object libero_goal libero_10}"

KI_RUN_NAME="${KI_RUN_NAME:-test1_ki_v3_full_2h100_b8_s0}"
NO_KI_RUN_NAME="${NO_KI_RUN_NAME:-no_ki_v3_full_2h100_b8_s0}"

cd "$PROJECT_DIR"

mkdir -p /root/.cache
rm -rf /root/.cache/huggingface /root/.cache/openpi /root/.cache/uv
ln -s "$CACHE_DIR/huggingface" /root/.cache/huggingface
ln -s "$CACHE_DIR/openpi"      /root/.cache/openpi
ln -s "$CACHE_DIR/uv"          /root/.cache/uv

source examples/libero/.venv/bin/activate
export PYTHONPATH="${PYTHONPATH:-}:$PWD/third_party/libero"

OUT_ROOT="data/libero/ki_v3_eval_step${STEP}"
mkdir -p "$OUT_ROOT"

SERVER_PID=""
cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

run_eval() {
  local run_key="$1"
  local config_name
  local ckpt_dir
  local out_dir

  case "$run_key" in
    ki)
      config_name="pi05_ki_libero"
      ckpt_dir="checkpoints/pi05_ki_libero/${KI_RUN_NAME}/${STEP}"
      out_dir="${OUT_ROOT}/ki"
      ;;
    no_ki)
      config_name="pi05_no_ki_libero"
      ckpt_dir="checkpoints/pi05_no_ki_libero/${NO_KI_RUN_NAME}/${STEP}"
      out_dir="${OUT_ROOT}/no_ki"
      ;;
    *)
      echo "[ERROR] Unknown run key: ${run_key}. Expected ki or no_ki."
      exit 1
      ;;
  esac

  if [[ ! -d "$ckpt_dir" ]]; then
    echo "[ERROR] Checkpoint directory not found: $ckpt_dir"
    exit 1
  fi

  mkdir -p "$out_dir"

  echo "[INFO] Starting policy server for ${run_key}"
  echo "[INFO] config=${config_name}"
  echo "[INFO] ckpt=${ckpt_dir}"

  uv run scripts/serve_policy.py \
    --port="${PORT}" \
    policy:checkpoint \
    --policy.config="${config_name}" \
    --policy.dir="${ckpt_dir}" \
    > "${out_dir}/server.log" 2>&1 &
  SERVER_PID=$!

  until grep -q "Creating server" "${out_dir}/server.log" 2>/dev/null; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      echo "[ERROR] Policy server exited early. Log:"
      cat "${out_dir}/server.log"
      exit 1
    fi
    sleep 5
  done
  echo "[INFO] Server ready for ${run_key}."

  for SUITE in ${SUITES}; do
    echo "[INFO] Running ${run_key} on ${SUITE} ..."
    mkdir -p "${out_dir}/videos_${SUITE}"
    video_arg="--args.save-video"
    if [[ "${SAVE_VIDEO}" == "0" || "${SAVE_VIDEO}" == "false" || "${SAVE_VIDEO}" == "False" ]]; then
      video_arg="--args.no-save-video"
    fi
    MUJOCO_GL=glx xvfb-run -a -e "${out_dir}/${SUITE}_xvfb.log" -s "-screen 0 1024x768x24" \
    python examples/libero/main.py \
      --args.host 0.0.0.0 \
      --args.port "${PORT}" \
      --args.task-suite-name "${SUITE}" \
      --args.num-trials-per-task "${NUM_TRIALS}" \
      --args.replan-steps "${REPLAN_STEPS}" \
      --args.video-out-path "${out_dir}/videos_${SUITE}" \
      --args.seed "${SEED}" \
      "${video_arg}" \
      2>&1 | tee "${out_dir}/${SUITE}.log"
    echo "[INFO] ${run_key} ${SUITE} done."
  done

  cleanup
  SERVER_PID=""
  echo "[INFO] Finished all suites for ${run_key}."
}

for RUN_KEY in ${RUNS}; do
  run_eval "${RUN_KEY}"
done

echo "[INFO] All requested evaluations completed."
echo "[INFO] Result logs: ${OUT_ROOT}"
echo "[INFO] Summary:"
grep -R "Total success rate\|Total episodes" "${OUT_ROOT}" || true
