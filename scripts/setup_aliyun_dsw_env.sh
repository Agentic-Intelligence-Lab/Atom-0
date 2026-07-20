#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"
export PATH="${HOME}/.local/bin:${PATH}"
export UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-https://mirrors.aliyun.com/pypi/simple}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${HOME}/.cache/uv}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

if ! command -v python3.11 >/dev/null 2>&1; then
  echo "Python 3.11 is required. Select a PAI image tagged py311." >&2
  exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
  python3.11 -m pip install --user uv
fi

export GIT_LFS_SKIP_SMUDGE=1
# pyproject.toml keeps an optional local wheel source for environments that ship
# a lightweight rerun stub. Fresh Git clones do not contain this ignored directory;
# uv requires every find-links path to exist even when it resolves rerun-sdk from PyPI.
mkdir -p third_party/rerun-stub/dist

# Keep the environment on the instance-local overlay. Installing hundreds of
# Python package files directly into the NAS-mounted repository is much slower.
VENV_DIR="${ATOM_VENV_DIR:-${HOME}/.cache/atom0/venvs/Atom-0-py311}"
export UV_PROJECT_ENVIRONMENT="${VENV_DIR}"
if [[ -d .venv && ! -L .venv && "$(realpath .venv)" != "$(realpath -m "${VENV_DIR}")" ]]; then
  echo "Found a NAS-backed .venv directory at ${REPO_DIR}/.venv." >&2
  echo "Move it aside once (for example: mv .venv .venv.nfs-partial) and rerun." >&2
  exit 2
fi
mkdir -p "$(dirname "${VENV_DIR}")"

if [[ "${UV_VENV_CLEAR:-0}" == "1" ]]; then
  uv venv "${VENV_DIR}" --clear --python 3.11
elif [[ -x "${VENV_DIR}/bin/python" ]]; then
  VENV_PYTHON_VERSION="$("${VENV_DIR}/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  if [[ "${VENV_PYTHON_VERSION}" != "3.11" ]]; then
    echo "${VENV_DIR} uses Python ${VENV_PYTHON_VERSION}; rerun with UV_VENV_CLEAR=1" >&2
    exit 1
  fi
  echo "Reusing existing Python ${VENV_PYTHON_VERSION} environment at ${VENV_DIR}"
else
  uv venv "${VENV_DIR}" --python 3.11
fi

if [[ ! -e .venv && ! -L .venv ]]; then
  ln -s "${VENV_DIR}" .venv
elif [[ -L .venv && "$(realpath -m .venv)" != "$(realpath -m "${VENV_DIR}")" ]]; then
  echo ".venv points to $(readlink .venv), expected ${VENV_DIR}" >&2
  exit 1
fi

SYNC_ARGS=(--frozen --group rlds)
if [[ "${INSTALL_DEV:-0}" == "1" ]]; then
  SYNC_ARGS+=(--group dev)
fi
uv sync "${SYNC_ARGS[@]}"

"${VENV_DIR}/bin/python" - <<'PY'
import jax
import tensorflow as tf
import torch

print("jax", jax.__version__, "devices", jax.devices())
print("tensorflow", tf.__version__)
print("torch", torch.__version__, "cuda", torch.version.cuda)
if not any(device.platform == "gpu" for device in jax.devices()):
    raise SystemExit("JAX did not detect a GPU")
PY

echo "DSW environment is ready: ${REPO_DIR}/.venv"
