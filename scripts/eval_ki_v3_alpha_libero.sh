#!/bin/bash
# Evaluate KI-V3 alpha ablation checkpoints on LIBERO suites.
#
# Defaults target the experiment names previously used for alpha ablations:
#   alpha01 -> checkpoints/pi05_ki_libero/ki_v3_alpha01_2h100_b16_s0/<STEP>
#   alpha03 -> checkpoints/pi05_ki_libero/ki_v3_alpha03_2h100_b16_s0/<STEP>
#   alpha05 -> checkpoints/pi05_ki_libero/ki_v3_alpha05_2h100_b16_s0/<STEP>
#
# Environment overrides:
#   PROJECT_DIR=/mnt/data/xule/openpi
#   STEP=30000
#   NUM_TRIALS=50
#   SEED=7
#   SAVE_VIDEO=0
#   REPLAN_STEPS=5
#   SUITES="libero_spatial libero_object libero_goal libero_10"
#   RUNS="alpha01 alpha03 alpha05"

set -euo pipefail

export PATH="/root/.local/bin:/root/miniforge3/bin:$PATH"

PROJECT_DIR="${PROJECT_DIR:-/mnt/data/xule/openpi}"
CACHE_DIR="${CACHE_DIR:-/mnt/data/cache}"
STEP="${STEP:-30000}"
NUM_TRIALS="${NUM_TRIALS:-50}"
SEED="${SEED:-7}"
PORT="${PORT:-8000}"
SAVE_VIDEO="${SAVE_VIDEO:-0}"
REPLAN_STEPS="${REPLAN_STEPS:-5}"
RUNS="${RUNS:-alpha01 alpha03 alpha05}"
SUITES="${SUITES:-libero_spatial libero_object libero_goal libero_10}"

ALPHA01_RUN_NAME="${ALPHA01_RUN_NAME:-ki_v3_alpha01_2h100_b16_s0}"
ALPHA03_RUN_NAME="${ALPHA03_RUN_NAME:-ki_v3_alpha03_2h100_b16_s0}"
ALPHA05_RUN_NAME="${ALPHA05_RUN_NAME:-ki_v3_alpha05_2h100_b16_s0}"

cd "$PROJECT_DIR"

mkdir -p /root/.cache
rm -rf /root/.cache/huggingface /root/.cache/openpi /root/.cache/uv
ln -s "$CACHE_DIR/huggingface" /root/.cache/huggingface
ln -s "$CACHE_DIR/openpi"      /root/.cache/openpi
ln -s "$CACHE_DIR/uv"          /root/.cache/uv

source examples/libero/.venv/bin/activate
export PYTHONPATH="${PYTHONPATH:-}:$PWD/third_party/libero"

OUT_ROOT="data/libero/ki_v3_alpha_eval_step${STEP}"
mkdir -p "$OUT_ROOT"

SERVER_PID=""
cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

video_arg="--args.save-video"
if [[ "${SAVE_VIDEO}" == "0" || "${SAVE_VIDEO}" == "false" || "${SAVE_VIDEO}" == "False" ]]; then
  video_arg="--args.no-save-video"
fi

run_eval() {
  local run_key="$1"
  local run_name
  local alpha_label
  local ckpt_dir
  local out_dir

  case "$run_key" in
    alpha01|a01|0.1)
      run_name="${ALPHA01_RUN_NAME}"
      alpha_label="alpha01"
      ;;
    alpha03|a03|0.3)
      run_name="${ALPHA03_RUN_NAME}"
      alpha_label="alpha03"
      ;;
    alpha05|a05|0.5)
      run_name="${ALPHA05_RUN_NAME}"
      alpha_label="alpha05"
      ;;
    *)
      echo "[ERROR] Unknown run key: ${run_key}. Expected alpha01, alpha03, or alpha05."
      exit 1
      ;;
  esac

  ckpt_dir="checkpoints/pi05_ki_libero/${run_name}/${STEP}"
  out_dir="${OUT_ROOT}/${alpha_label}"

  if [[ ! -d "$ckpt_dir" ]]; then
    echo "[ERROR] Checkpoint directory not found: $ckpt_dir"
    exit 1
  fi

  mkdir -p "$out_dir"

  echo "[INFO] Starting policy server for ${alpha_label}"
  echo "[INFO] config=pi05_ki_libero"
  echo "[INFO] ckpt=${ckpt_dir}"

  uv run scripts/serve_policy.py \
    --port="${PORT}" \
    policy:checkpoint \
    --policy.config=pi05_ki_libero \
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
  echo "[INFO] Server ready for ${alpha_label}."

  for SUITE in ${SUITES}; do
    echo "[INFO] Running ${alpha_label} on ${SUITE} ..."
    mkdir -p "${out_dir}/videos_${SUITE}"
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
    echo "[INFO] ${alpha_label} ${SUITE} done."
  done

  cleanup
  SERVER_PID=""
  echo "[INFO] Finished all suites for ${alpha_label}."
}

for RUN_KEY in ${RUNS}; do
  run_eval "${RUN_KEY}"
done

echo "[INFO] All requested alpha evaluations completed."
echo "[INFO] Result logs: ${OUT_ROOT}"
echo "[INFO] Summary:"
grep -R "Total success rate\|Total episodes" "${OUT_ROOT}" || true
