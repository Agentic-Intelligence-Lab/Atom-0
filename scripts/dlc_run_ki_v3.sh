#!/bin/bash
# KI-V3 training launcher for Aliyun DLC.
#
# This script follows the same DLC path/cache style as scripts/eval_libero.sh,
# but launches training instead of LIBERO evaluation.
#
# Examples:
#   # 1k smoke on one GPU.
#   bash scripts/dlc_run_ki_v3.sh --run ki --steps 1000 --batch-size 1 --fsdp-devices 1
#
#   # Main A/B jobs. Prefer submitting no_ki and ki as two independent DLC jobs.
#   bash scripts/dlc_run_ki_v3.sh --run no_ki --steps 30000 --batch-size 32 --fsdp-devices 8 --seed 0
#   bash scripts/dlc_run_ki_v3.sh --run ki    --steps 30000 --batch-size 32 --fsdp-devices 8 --seed 0
#
# Useful environment overrides:
#   PROJECT_DIR=/mnt/data/xule/pi07_reproduction
#   CACHE_DIR=/mnt/data/cache
#   NORM_STATS_SRC=/mnt/data/xule/openpi/assets/pi05_libero/physical-intelligence/libero/norm_stats.json

set -euo pipefail

export PATH="/root/.local/bin:/root/miniforge3/bin:$PATH"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"

PROJECT_DIR="${PROJECT_DIR:-/mnt/data/xule/pi07_reproduction}"
CACHE_DIR="${CACHE_DIR:-/mnt/data/cache}"
NORM_STATS_SRC="${NORM_STATS_SRC:-/mnt/data/xule/openpi/assets/pi05_libero/physical-intelligence/libero/norm_stats.json}"

RUN="ki"
STEPS=1000
BATCH_SIZE=1
FSDP_DEVICES=1
NUM_WORKERS=2
LOG_INTERVAL=20
SAVE_INTERVAL=1000
KEEP_PERIOD=1000
SEED=0
EXP_TAG="v3_smoke_dlc"
WANDB_MODE=""

usage() {
    cat <<EOF
Usage:
  bash scripts/dlc_run_ki_v3.sh [options]

Options:
  --run <ki|no_ki|all>        Which run to launch. Default: ki
  --steps <N>                 Number of train steps. Default: 1000
  --batch-size <N>            Global batch size. Default: 1
  --fsdp-devices <N>          FSDP devices. Default: 1
  --num-workers <N>           Dataloader workers. Default: 2
  --log-interval <N>          Log interval. Default: 20
  --save-interval <N>         Save interval. Default: 1000
  --keep-period <N>           Checkpoint keep period. Default: 1000
  --seed <N>                  Random seed. Default: 0
  --exp-tag <TAG>             Experiment tag. Default: v3_smoke_dlc
  --wandb-mode <online|offline|disabled>
                              Optional WANDB_MODE override.
  -h, --help                  Show this message.

Environment:
  PROJECT_DIR                 Repo path. Default: /mnt/data/xule/pi07_reproduction
  CACHE_DIR                   Shared cache path. Default: /mnt/data/cache
  NORM_STATS_SRC              Existing LIBERO norm_stats.json to copy.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run)           RUN="$2";          shift 2 ;;
        --steps)         STEPS="$2";        shift 2 ;;
        --batch-size)    BATCH_SIZE="$2";   shift 2 ;;
        --fsdp-devices)  FSDP_DEVICES="$2"; shift 2 ;;
        --num-workers)   NUM_WORKERS="$2";  shift 2 ;;
        --log-interval)  LOG_INTERVAL="$2"; shift 2 ;;
        --save-interval) SAVE_INTERVAL="$2"; shift 2 ;;
        --keep-period)   KEEP_PERIOD="$2";  shift 2 ;;
        --seed)          SEED="$2";         shift 2 ;;
        --exp-tag)       EXP_TAG="$2";      shift 2 ;;
        --wandb-mode)    WANDB_MODE="$2";   shift 2 ;;
        -h|--help)       usage; exit 0 ;;
        *) echo "[ERROR] Unknown argument: $1"; usage; exit 1 ;;
    esac
done

if [[ "$RUN" != "ki" && "$RUN" != "no_ki" && "$RUN" != "all" ]]; then
    echo "[ERROR] --run must be one of: ki, no_ki, all"
    exit 1
fi

cd "$PROJECT_DIR"

echo "[INFO] Project dir: $PROJECT_DIR"
echo "[INFO] Cache dir:   $CACHE_DIR"

# DLC images sometimes do not include the local rerun stub directory referenced
# by pyproject.toml. Creating the directory is enough when rerun-sdk is already
# available in the lock/environment; otherwise uv will surface the real package
# resolution error next.
mkdir -p third_party/rerun-stub/dist

mkdir -p /root/.cache
for name in huggingface openpi uv; do
    if [[ -e "/root/.cache/$name" && ! -L "/root/.cache/$name" ]]; then
        echo "[WARN] /root/.cache/$name exists and is not a symlink; leaving it untouched."
    elif [[ ! -e "/root/.cache/$name" ]]; then
        mkdir -p "$CACHE_DIR/$name"
        ln -s "$CACHE_DIR/$name" "/root/.cache/$name"
    fi
done

if [[ -n "$WANDB_MODE" ]]; then
    export WANDB_MODE="$WANDB_MODE"
    echo "[INFO] WANDB_MODE=$WANDB_MODE"
fi

copy_norm_stats() {
    local config_name="$1"
    local dst_dir="assets/${config_name}/physical-intelligence/libero"
    if [[ ! -f "$NORM_STATS_SRC" ]]; then
        echo "[ERROR] Norm stats not found: $NORM_STATS_SRC"
        echo "[ERROR] Set NORM_STATS_SRC or run scripts/compute_norm_stats.py first."
        exit 1
    fi
    mkdir -p "$dst_dir"
    cp "$NORM_STATS_SRC" "$dst_dir/norm_stats.json"
    echo "[INFO] Norm stats ready: $dst_dir/norm_stats.json"
}

copy_norm_stats "pi05_ki_libero"
copy_norm_stats "pi05_no_ki_libero"

mkdir -p logs
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

run_train() {
    local config_name="$1"
    local prefix="$2"
    local exp_name="${prefix}_${EXP_TAG}_s${SEED}_${TIMESTAMP}"
    local log_path="logs/${exp_name}.log"

    echo "[INFO] Launching ${config_name}"
    echo "[INFO] exp_name=${exp_name}"
    echo "[INFO] log=${log_path}"

    uv run scripts/train.py "${config_name}" \
        --exp-name "${exp_name}" \
        --num-train-steps "${STEPS}" \
        --batch-size "${BATCH_SIZE}" \
        --fsdp-devices "${FSDP_DEVICES}" \
        --num-workers "${NUM_WORKERS}" \
        --log-interval "${LOG_INTERVAL}" \
        --save-interval "${SAVE_INTERVAL}" \
        --keep-period "${KEEP_PERIOD}" \
        --seed "${SEED}" \
        2>&1 | tee "${log_path}"

    echo "[INFO] Finished ${config_name}: ${exp_name}"
}

case "$RUN" in
    ki)
        run_train "pi05_ki_libero" "ki"
        ;;
    no_ki)
        run_train "pi05_no_ki_libero" "no_ki"
        ;;
    all)
        run_train "pi05_no_ki_libero" "no_ki"
        run_train "pi05_ki_libero" "ki"
        ;;
esac

echo "[INFO] Done."
echo "[INFO] Logs:        ${PROJECT_DIR}/logs"
echo "[INFO] Checkpoints: ${PROJECT_DIR}/checkpoints"
