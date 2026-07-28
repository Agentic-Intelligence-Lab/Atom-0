#!/usr/bin/env bash
# Import a ready-made Atom-0 venv from shared PFS (Aliyun / wudi machine).
# Use this when local `uv sync` fails due to GitHub/network limits.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_VENV="${SOURCE_VENV:-/data/wudi/Atom-0/.venv}"
SOURCE_UV_CACHE="${SOURCE_UV_CACHE:-/data/wudi/.cache/uv}"
SOURCE_UV_PYTHON="${SOURCE_UV_PYTHON:-/data/wudi/.local/share/uv/python}"

if [[ ! -x "${SOURCE_VENV}/bin/python" ]]; then
  echo "Missing source venv: ${SOURCE_VENV}/bin/python" >&2
  exit 1
fi

cd "${REPO_DIR}"
rm -rf .venv
ln -sf "${SOURCE_VENV}" .venv
mkdir -p third_party/rerun-stub/dist

mkdir -p "${SOURCE_UV_CACHE}" "${SOURCE_UV_PYTHON}" 2>/dev/null || true

cat <<EOF
Imported Aliyun/wudi environment:
  .venv -> ${SOURCE_VENV}
  third_party/rerun-stub/dist created

Recommended env overrides (add to your shell or source atom0_env.sh after):
  export UV_CACHE_DIR=${SOURCE_UV_CACHE}
  export UV_PYTHON_INSTALL_DIR=${SOURCE_UV_PYTHON}
  export UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/

Verify:
  source scripts/atom0_env.sh
  .venv/bin/python --version
  .venv/bin/python scripts/preflight_cotrain_baige.py fastwam_cotrain_real_robot_ego_fix_debug
EOF

.venv/bin/python --version
