#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"
export PATH="${HOME}/.local/bin:${PATH}"

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
if [[ "${UV_VENV_CLEAR:-0}" == "1" ]]; then
  uv venv --clear --python 3.11
elif [[ -x .venv/bin/python ]]; then
  VENV_PYTHON_VERSION="$(.venv/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  if [[ "${VENV_PYTHON_VERSION}" != "3.11" ]]; then
    echo ".venv uses Python ${VENV_PYTHON_VERSION}; rerun with UV_VENV_CLEAR=1" >&2
    exit 1
  fi
  echo "Reusing existing Python ${VENV_PYTHON_VERSION} environment at ${REPO_DIR}/.venv"
else
  uv venv --python 3.11
fi
uv sync --frozen --group dev --group rlds
uv pip install -e .

.venv/bin/python - <<'PY'
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
