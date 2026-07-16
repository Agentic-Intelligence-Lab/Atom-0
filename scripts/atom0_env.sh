#!/usr/bin/env bash

# Source this file before running Atom-0 commands on the new host:
#   source scripts/atom0_env.sh

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "This script must be sourced: source scripts/atom0_env.sh" >&2
  exit 1
fi

ATOM0_REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ATOM0_STATE_ROOT="${ATOM0_STATE_ROOT:-$(dirname "${ATOM0_REPO_DIR}")}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-${ATOM0_STATE_ROOT}/.cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-${ATOM0_STATE_ROOT}/.local/share/uv/python}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${ATOM0_STATE_ROOT}/.cache}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-${XDG_CACHE_HOME}/jax}"
export PYTHONPATH="${ATOM0_REPO_DIR}/src:${ATOM0_REPO_DIR}/packages/openpi-client/src:${PYTHONPATH:-}"

export HF_HOME="${HF_HOME:-${ATOM0_STATE_ROOT}/cache/huggingface}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${ATOM0_STATE_ROOT}/cache/openpi}"
export OPENPI_MODEL_HOME="${OPENPI_MODEL_HOME:-/data/models/openpi}"
export RLDS_DATA_DIR="${RLDS_DATA_DIR:-/mnt/bos/bo23lu}"
export PARAMS_PATH="${PARAMS_PATH:-${OPENPI_MODEL_HOME}/openpi-assets/checkpoints/pi05_base/params}"
export LOG_DIR="${LOG_DIR:-${ATOM0_REPO_DIR}/logs}"

export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-1}"

mkdir -p "${HF_HOME}" "${OPENPI_DATA_HOME}" "${OPENPI_MODEL_HOME}" "${JAX_COMPILATION_CACHE_DIR}" "${LOG_DIR}"

if [[ -n "${MASTER_ADDR:-}" && -z "${JAX_COORDINATOR_ADDRESS:-}" ]]; then
  export JAX_COORDINATOR_ADDRESS="${MASTER_ADDR}:29500"
fi

echo "Atom-0 environment loaded from ${ATOM0_REPO_DIR}"
echo "RLDS_DATA_DIR=${RLDS_DATA_DIR}"
echo "PARAMS_PATH=${PARAMS_PATH}"
