#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DEFAULTS="${ROOT_DIR}/config/eval_defaults.env"
if [[ -f "${DEFAULTS}" ]]; then
  set -a
  source "${DEFAULTS}"
  set +a
fi

mkdir -p "${RESULT_ROOT}"

if [[ "${1:-}" == "--validate-only" ]]; then
  exec "${PYTHON_BIN}" "${ROOT_DIR}/scripts/evaluate_validation.py" --validate-only
fi

if [[ "${1:-}" == "--metadata-only" ]]; then
  exec "${PYTHON_BIN}" "${ROOT_DIR}/scripts/evaluate_validation.py" --metadata-only
fi

OUTPUT_JSON="${OUTPUT_JSON:-${RESULT_ROOT}/b200_9999_${SPLIT}_e${EPISODES}_a${ANCHORS_PER_EPISODE}.json}"
OUTPUT_CSV="${OUTPUT_CSV:-${RESULT_ROOT}/b200_9999_${SPLIT}_e${EPISODES}_a${ANCHORS_PER_EPISODE}.csv}"
export OUTPUT_JSON OUTPUT_CSV

exec "${PYTHON_BIN}" "${ROOT_DIR}/scripts/evaluate_validation.py" "$@"
