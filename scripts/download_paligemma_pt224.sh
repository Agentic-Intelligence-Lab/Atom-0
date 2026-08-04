#!/usr/bin/env bash
set -euo pipefail

# Download the exact big_vision JAX checkpoint expected by
# LocalPaliGemmaWeightLoader. This is not the PyTorch safetensors checkpoint.
#
# Usage:
#   export HF_TOKEN=hf_...
#   bash scripts/download_paligemma_pt224.sh [destination_directory]
#
# Default destination:
#   /data/models/paligemma
#
# The Hugging Face account behind HF_TOKEN must already have access to:
#   google/paligemma-3b-pt-224-jax

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEST_DIR="${1:-/data/models/paligemma}"

MODEL_REPO="google/paligemma-3b-pt-224-jax"
REMOTE_FILENAME="paligemma-3b-pt-224.npz"
LOCAL_FILENAME="pt_224.npz"
HF_MIRROR_ENDPOINT="https://hf-mirror.com"

# Do not use a VPN or HTTP(S) proxy for either the mirror request or redirected
# object-storage downloads.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY
unset http_proxy https_proxy all_proxy

export HF_ENDPOINT="${HF_MIRROR_ENDPOINT}"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"
export HF_HOME="${HF_HOME:-${DEST_DIR}/.hf_cache}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set." >&2
  echo "Accept access to ${MODEL_REPO}, then export a read token:" >&2
  echo "  export HF_TOKEN=hf_..." >&2
  exit 2
fi

HF_CLI="${REPO_DIR}/.venv/bin/huggingface-cli"
if [[ ! -x "${HF_CLI}" ]]; then
  HF_CLI="$(command -v huggingface-cli || true)"
fi
if [[ -z "${HF_CLI}" || ! -x "${HF_CLI}" ]]; then
  echo "huggingface-cli was not found. Install the Atom-0 environment first." >&2
  exit 2
fi

mkdir -p "${DEST_DIR}" "${HF_HOME}"

echo "Downloading ${MODEL_REPO}/${REMOTE_FILENAME}"
echo "Mirror: ${HF_ENDPOINT}"
echo "Proxy variables: cleared"
echo "Destination: ${DEST_DIR}"

# huggingface_hub resumes an interrupted download from its local metadata/cache.
"${HF_CLI}" download \
  "${MODEL_REPO}" \
  "${REMOTE_FILENAME}" \
  --repo-type model \
  --local-dir "${DEST_DIR}"

DOWNLOADED_PATH="${DEST_DIR}/${REMOTE_FILENAME}"
LOADER_PATH="${DEST_DIR}/${LOCAL_FILENAME}"

if [[ ! -s "${DOWNLOADED_PATH}" ]]; then
  echo "Download did not produce a non-empty file: ${DOWNLOADED_PATH}" >&2
  exit 1
fi

if [[ -e "${LOADER_PATH}" && ! -L "${LOADER_PATH}" ]]; then
  echo "Refusing to replace existing regular file: ${LOADER_PATH}" >&2
  echo "The downloaded checkpoint is available at: ${DOWNLOADED_PATH}" >&2
  exit 2
fi
ln -sfn "${REMOTE_FILENAME}" "${LOADER_PATH}"

# Read only the NPZ directory and parameter names; arrays are not loaded into RAM.
"${REPO_DIR}/.venv/bin/python" - "${LOADER_PATH}" <<'PY'
import pathlib
import sys

import numpy as np

path = pathlib.Path(sys.argv[1])
with np.load(path, allow_pickle=False) as checkpoint:
    keys = checkpoint.files
    has_image = any(key.startswith("params/img/") for key in keys)
    has_llm = any(key.startswith("params/llm/") for key in keys)
    if not has_image or not has_llm:
        raise SystemExit(
            f"Unexpected PaliGemma NPZ layout: image={has_image}, llm={has_llm}, "
            f"num_keys={len(keys)}"
        )
    print(f"NPZ layout OK: {len(keys)} arrays; image=True; llm=True")
PY

du -h "${DOWNLOADED_PATH}"

if [[ "${VERIFY_SHA256:-0}" == "1" ]]; then
  sha256sum "${DOWNLOADED_PATH}" | tee "${DOWNLOADED_PATH}.sha256"
fi

echo "Download complete."
echo "Use this path for VLM initialization:"
echo "  ${LOADER_PATH}"
