#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_ASSETS_DIR="${REPO_DIR}/assets/cotrain_full_all_full_norm"
OUTPUT_ASSETS_DIR="${OUTPUT_ASSETS_DIR:-${REPO_DIR}/assets/cotrain_full_all_atom_aligned_rl2}"

test -d "${SOURCE_ASSETS_DIR}"
mkdir -p "${OUTPUT_ASSETS_DIR}"

# The first 39 builders are byte-for-byte the A-3 inputs, so their audited per-dataset
# statistics and Unified80 fingerprints are reusable. The Python job skips these copied
# directories and computes only the four AtomAligned and two RL2 builders.
cp -a "${SOURCE_ASSETS_DIR}/." "${OUTPUT_ASSETS_DIR}/"

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}/packages/openpi-client/src:${PYTHONPATH:-}"

exec .venv/bin/python -u scripts/compute_cotrain_full_norm_stats_light.py \
  --config-name cotrain_full_all_atom_aligned_rl2 \
  --output-assets-dir "${OUTPUT_ASSETS_DIR}" \
  --copy-from-assets-name cotrain_full_all_full_norm \
  --num-parallel-reads "${NUM_PARALLEL_READS:-1}" \
  --num-parallel-calls "${NUM_PARALLEL_CALLS:-2}" \
  "$@"
