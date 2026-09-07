#!/bin/bash
# KI-V3: A vs B training comparison on LIBERO (Aliyun DLC / H20).
#
# 在 DLC 上提交时，建议将 A（no-KI）和 B（KI）分别提交为两个独立 Job，
# 分别传入 --run a / --run b，这样两条线可以并行跑在不同节点上。
#
# Usage（单节点顺序跑）:
#   bash scripts/run_ki_v3.sh --run a         # 只跑 no-KI baseline
#   bash scripts/run_ki_v3.sh --run b         # 只跑 KI
#   bash scripts/run_ki_v3.sh --run all       # 先 A 后 B
#
# 可选参数:
#   --exp-tag <tag>       区分 wandb run 的标签（默认 v3）
#   --fsdp-devices <N>    FSDP 分片数（默认 8，适合单节点 8x H20）
#   --num-workers <N>     dataloader worker 数（默认 4）
#   --batch-size <N>      全局 batch size（默认 32）

set -e

# ── 环境 ───────────────────────────────────────────────────
export PATH="/root/.local/bin:/root/miniforge3/bin:$PATH"

# 项目根目录（DLC 上按实际路径修改）
PROJECT_DIR="${PROJECT_DIR:-/path/to/Atom-0}"
cd "$PROJECT_DIR"

# 将 /mnt/data/cache 下的 cache 软链接到 ~/.cache，避免重复下载
mkdir -p /root/.cache
for name in huggingface openpi uv; do
    if [[ ! -e "/root/.cache/$name" ]]; then
        ln -s "/mnt/data/cache/$name" "/root/.cache/$name"
    fi
done

# ── 参数解析 ───────────────────────────────────────────────
RUN="all"
EXP_TAG="v3"
FSDP_DEVICES=8
NUM_WORKERS=4
BATCH_SIZE=32

while [[ $# -gt 0 ]]; do
    case $1 in
        --run)           RUN="$2";            shift 2 ;;
        --exp-tag)       EXP_TAG="$2";        shift 2 ;;
        --fsdp-devices)  FSDP_DEVICES="$2";   shift 2 ;;
        --num-workers)   NUM_WORKERS="$2";    shift 2 ;;
        --batch-size)    BATCH_SIZE="$2";     shift 2 ;;
        *) echo "[ERROR] Unknown argument: $1"; exit 1 ;;
    esac
done

EXP_A="no_ki_${EXP_TAG}"
EXP_B="ki_${EXP_TAG}"
mkdir -p logs

# ── 训练函数 ───────────────────────────────────────────────
run_train() {
    local CONFIG_NAME="$1"
    local EXP_NAME="$2"
    echo "[INFO] Starting: ${CONFIG_NAME}  exp_name=${EXP_NAME}"
    # config 名是位置参数（tyro overridable_config_cli），不是 --config 选项
    uv run scripts/train.py "${CONFIG_NAME}" \
        --exp-name "${EXP_NAME}" \
        --fsdp-devices "${FSDP_DEVICES}" \
        --num-workers "${NUM_WORKERS}" \
        --batch-size "${BATCH_SIZE}" \
        2>&1 | tee "logs/${EXP_NAME}.log"
    echo "[INFO] Finished: ${CONFIG_NAME}"
}

# ── 执行 ────────────────────────────────────────────────────
case "$RUN" in
    a|no_ki)
        run_train "pi05_no_ki_libero" "${EXP_A}"
        ;;
    b|ki)
        run_train "pi05_ki_libero" "${EXP_B}"
        ;;
    all)
        run_train "pi05_no_ki_libero" "${EXP_A}"
        run_train "pi05_ki_libero"    "${EXP_B}"
        ;;
    *)
        echo "[ERROR] --run must be one of: a, b, all"; exit 1
        ;;
esac

echo "[INFO] Done. Logs: logs/  Checkpoints: checkpoints/"
echo "[INFO] WandB: filter by exp_name '${EXP_A}' vs '${EXP_B}'"
