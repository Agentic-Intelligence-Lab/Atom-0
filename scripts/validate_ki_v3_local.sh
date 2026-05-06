#!/bin/bash
# 本地 A100 验证脚本：在正式提交 DLC/H20 之前，确认 KI-V3 代码可以跑通。
#
# 验证内容（全程 < 10 分钟）：
#   Step 1  Stage 1-3 单元测试（CPU，~2 分钟）
#             - Stage 1: 梯度路径断言（stop_gradient 是否生效）
#             - Stage 2: 前向传播 sanity check
#             - Stage 3: 单 batch overfit（flow_loss 和 ki_fast_loss 都下降）
#   Step 2  debug_pi05_ki 端到端训练（GPU，~3 分钟）
#             - 使用 dummy 模型 + FakeDataConfig，跑 20 步
#             - 验证 train_step 不崩、log 里出现 grad_norm_vlm / grad_norm_action
#   Step 3  config 加载验证（CPU，< 1 分钟）
#             - 确认 pi05_ki_libero / pi05_no_ki_libero 的 repo_id 已正确填写
#
# Usage:
#   bash scripts/validate_ki_v3_local.sh
#
# 如需跳过某一步：
#   SKIP_STAGE_TESTS=1  bash scripts/validate_ki_v3_local.sh
#   SKIP_GPU_TRAIN=1    bash scripts/validate_ki_v3_local.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

PASS=0
FAIL=0

log()  { echo "[$(date '+%H:%M:%S')] $*"; }
ok()   { echo "  ✓ $*"; PASS=$((PASS+1)); }
fail() { echo "  ✗ $*"; FAIL=$((FAIL+1)); }

# ══════════════════════════════════════════════════════════════
# Step 1: Stage 1-3 单元测试（CPU 即可）
# ══════════════════════════════════════════════════════════════
if [[ "${SKIP_STAGE_TESTS:-0}" != "1" ]]; then
    log "Step 1: Running Stage 1-3 unit tests (CPU)..."

    for STAGE in 1 2 3; do
        TEST_FILE="tests/ki/stage${STAGE}_$(
            case $STAGE in
                1) echo "gradient_paths" ;;
                2) echo "forward_sanity" ;;
                3) echo "overfit" ;;
            esac
        ).py"
        log "  Running ${TEST_FILE} ..."
        if JAX_PLATFORMS=cpu uv run python "${TEST_FILE}" 2>&1 | tee /tmp/ki_stage${STAGE}.log | tail -3; then
            if grep -q "failed" /tmp/ki_stage${STAGE}.log; then
                fail "Stage ${STAGE}: $(grep 'Results:' /tmp/ki_stage${STAGE}.log)"
            else
                ok "Stage ${STAGE}: $(grep 'Results:' /tmp/ki_stage${STAGE}.log | tr -d '\n')"
            fi
        else
            fail "Stage ${STAGE}: script exited with error"
        fi
    done
else
    log "Step 1: Skipped (SKIP_STAGE_TESTS=1)"
fi

# ══════════════════════════════════════════════════════════════
# Step 2: debug_pi05_ki 端到端训练（GPU）
# ══════════════════════════════════════════════════════════════
if [[ "${SKIP_GPU_TRAIN:-0}" != "1" ]]; then
    log "Step 2: Running debug_pi05_ki end-to-end on GPU..."

    # 用一个临时目录存 checkpoint，避免污染正式目录
    TMP_CKPT_DIR="$(mktemp -d /tmp/ki_validate_XXXXXX)"
    trap "rm -rf ${TMP_CKPT_DIR}" EXIT

    # 跑 20 步（debug_pi05_ki 默认只跑 10 步；这里 override 到 20 步以确保 log 输出）
    TRAIN_LOG="${TMP_CKPT_DIR}/train.log"
    log "  Checkpoints: ${TMP_CKPT_DIR}"

    set +e
    # config 名是位置参数（tyro overridable_config_cli），不是 --config 选项
    uv run scripts/train.py debug_pi05_ki \
        --checkpoint-base-dir "${TMP_CKPT_DIR}" \
        --num-train-steps 20 \
        --log-interval 5 \
        --num-workers 0 \
        2>&1 | tee "${TRAIN_LOG}"
    TRAIN_EXIT=$?
    set -e

    if [[ $TRAIN_EXIT -ne 0 ]]; then
        fail "GPU train: exited with code ${TRAIN_EXIT}"
    else
        ok "GPU train: completed 20 steps without error"
    fi

    # 检查 log 里是否有 grad_norm_vlm 和 grad_norm_action
    # debug_pi05_ki 使用 FakeDataConfig，不含 ki_fast_tokens，
    # 所以 flow_loss / ki_fast_loss 不会出现（只有真实数据才触发该路径）。
    # 但 ki_enabled=True 使 _ki_grad_norm_split 一定执行，两个梯度范数必须出现。
    if grep -q "grad_norm_vlm" "${TRAIN_LOG}"; then
        ok "Logging: 'grad_norm_vlm' found in training log"
    else
        fail "Logging: 'grad_norm_vlm' NOT found in training log"
    fi

    if grep -q "grad_norm_action" "${TRAIN_LOG}"; then
        ok "Logging: 'grad_norm_action' found in training log"
    else
        fail "Logging: 'grad_norm_action' NOT found in training log"
    fi

    # 基础指标必须存在
    if grep -q "loss=" "${TRAIN_LOG}"; then
        ok "Logging: 'loss' found in training log"
    else
        fail "Logging: 'loss' NOT found in training log"
    fi
else
    log "Step 2: Skipped (SKIP_GPU_TRAIN=1)"
fi

# ══════════════════════════════════════════════════════════════
# Step 3: 正式训练 config 加载验证（CPU）
# ══════════════════════════════════════════════════════════════
log "Step 3: Verifying production config fields..."

CONFIG_CHECK_LOG="$(JAX_PLATFORMS=cpu uv run python - <<'PYEOF' 2>&1
import sys
sys.path.insert(0, "src")
from openpi.training.config import get_config

errors = []

for name in ("pi05_ki_libero", "pi05_no_ki_libero"):
    c = get_config(name)
    if c.data.repo_id != "physical-intelligence/libero":
        errors.append(f"{name}: repo_id={c.data.repo_id!r} (expected 'physical-intelligence/libero')")
    if c.data.base_config is None or not getattr(c.data.base_config, "prompt_from_task", False):
        errors.append(f"{name}: base_config.prompt_from_task is not True")

ki  = get_config("pi05_ki_libero")
nki = get_config("pi05_no_ki_libero")
for attr in ("batch_size", "num_train_steps", "log_interval", "save_interval"):
    if getattr(ki, attr) != getattr(nki, attr):
        errors.append(f"hyperparams differ: {attr}: ki={getattr(ki,attr)}  no_ki={getattr(nki,attr)}")

if ki.model.ki_enabled is not True:
    errors.append("pi05_ki_libero: ki_enabled should be True")
if nki.model.ki_enabled is not False:
    errors.append("pi05_no_ki_libero: ki_enabled should be False")

if errors:
    print("FAIL")
    for e in errors:
        print(" ", e)
else:
    print("OK")
    print(f"  pi05_ki_libero:    repo_id={ki.data.repo_id!r}, ki_enabled={ki.model.ki_enabled}, batch_size={ki.batch_size}")
    print(f"  pi05_no_ki_libero: repo_id={nki.data.repo_id!r}, ki_enabled={nki.model.ki_enabled}, batch_size={nki.batch_size}")
PYEOF
)"

if echo "$CONFIG_CHECK_LOG" | grep -q "^OK"; then
    while IFS= read -r line; do
        [[ -n "$line" && "$line" != "OK" ]] && ok "Config: $line"
    done <<< "$CONFIG_CHECK_LOG"
    ok "Config: both production configs load correctly"
else
    fail "Config check failed:"
    echo "$CONFIG_CHECK_LOG"
fi

# ══════════════════════════════════════════════════════════════
# 汇总
# ══════════════════════════════════════════════════════════════
echo ""
echo "══════════════════════════════════════"
echo "  Validation result: ${PASS} passed, ${FAIL} failed"
echo "══════════════════════════════════════"

if [[ $FAIL -gt 0 ]]; then
    echo "[FAIL] Fix the issues above before submitting to DLC."
    exit 1
else
    echo "[PASS] All checks passed. Safe to submit to Aliyun DLC."
fi
