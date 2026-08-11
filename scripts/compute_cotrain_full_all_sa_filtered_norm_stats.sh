#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SA_FILTERED_RLDS_DATA_DIR="${SA_FILTERED_RLDS_DATA_DIR:-/data/wudi/RLDS_SA_Filtered}"
OUTPUT_ASSETS_DIR="${OUTPUT_ASSETS_DIR:-${REPO_DIR}/assets/cotrain_full_all_sa_filtered}"

test -f "${SA_FILTERED_RLDS_DATA_DIR}/_QUALITY_FILTER_BATCH_SUMMARY.json"

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}/packages/openpi-client/src:${PYTHONPATH:-}"

exec .venv/bin/python -u scripts/compute_cotrain_full_norm_stats_light.py \
  --config-name cotrain_full_all_sa_filtered \
  --output-assets-dir "${OUTPUT_ASSETS_DIR}" \
  --num-parallel-reads "${NUM_PARALLEL_READS:-1}" \
  --num-parallel-calls "${NUM_PARALLEL_CALLS:-2}" \
  "$@"
