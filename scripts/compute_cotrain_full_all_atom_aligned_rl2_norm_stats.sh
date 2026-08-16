#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_ASSETS_DIR="${REPO_DIR}/assets/cotrain_full_all_full_norm"
OUTPUT_ASSETS_DIR="${OUTPUT_ASSETS_DIR:-${REPO_DIR}/assets/cotrain_full_all_atom_aligned_rl2}"

test -d "${SOURCE_ASSETS_DIR}"
mkdir -p "${OUTPUT_ASSETS_DIR}"

# Reuse only the 34 non-Ego A-3 builders. A-4 uses official precomputed chunks for the
# four retained EgoVerse-full builders, so those four plus AtomAligned/RL2 must be computed.
for source_dir in "${SOURCE_ASSETS_DIR}"/*; do
  dataset_id="$(basename "${source_dir}")"
  if [[ -d "${source_dir}" && "${dataset_id}" != egoverse_* ]]; then
    cp -a "${source_dir}" "${OUTPUT_ASSETS_DIR}/"
  fi
done

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}/packages/openpi-client/src:${PYTHONPATH:-}"

exec .venv/bin/python -u scripts/compute_cotrain_full_norm_stats_light.py \
  --config-name cotrain_full_all_atom_aligned_rl2 \
  --output-assets-dir "${OUTPUT_ASSETS_DIR}" \
  --copy-from-assets-name cotrain_full_all_full_norm \
  --num-parallel-reads "${NUM_PARALLEL_READS:-1}" \
  --num-parallel-calls "${NUM_PARALLEL_CALLS:-2}" \
  "$@"
