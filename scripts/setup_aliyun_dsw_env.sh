#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

if ! command -v python3.11 >/dev/null 2>&1; then
  echo "Python 3.11 is required. Select a PAI image tagged py311." >&2
  exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
  python3.11 -m pip install --user uv
  export PATH="${HOME}/.local/bin:${PATH}"
fi

export GIT_LFS_SKIP_SMUDGE=1
uv venv --python 3.11
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
