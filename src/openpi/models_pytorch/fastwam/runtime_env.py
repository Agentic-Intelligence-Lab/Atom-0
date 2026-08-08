"""Default DiffSynth / HuggingFace paths for FastWAM (train + serve)."""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_DIFFSYNTH_BASE = _REPO_ROOT / "checkpoints" / "fastwam"


def configure_fastwam_runtime_env(
    *,
    repo_root: Path | str | None = None,
    offline: bool = False,
) -> Path:
    """Apply FastWAM runtime defaults unless already set in the environment.

    Mirrors ``scripts/train_fastwam_baige.sh`` so ``serve_policy`` does not need
    manual ``export DIFFSYNTH_*`` before each launch.

    Args:
        repo_root: Atom-0 repo root. Defaults to the package checkout.
        offline: When True (default for serve), skip Hub/ModelScope downloads and
            use only files under ``DIFFSYNTH_MODEL_BASE_PATH``.
    """
    root = Path(repo_root) if repo_root is not None else _REPO_ROOT
    base = Path(os.environ.get("DIFFSYNTH_MODEL_BASE_PATH", root / "checkpoints" / "fastwam"))
    if "DIFFSYNTH_MODEL_BASE_PATH" not in os.environ:
        os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(base)
    if "HF_HOME" not in os.environ:
        os.environ["HF_HOME"] = str(base / "hf_cache")
    if "DIFFSYNTH_DOWNLOAD_SOURCE" not in os.environ:
        os.environ["DIFFSYNTH_DOWNLOAD_SOURCE"] = "huggingface"
    if offline and "DIFFSYNTH_SKIP_DOWNLOAD" not in os.environ:
        os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "true"

    base.mkdir(parents=True, exist_ok=True)
    (base / "hf_cache").mkdir(parents=True, exist_ok=True)
    logger.info(
        "FastWAM runtime env: DIFFSYNTH_MODEL_BASE_PATH=%s HF_HOME=%s "
        "DIFFSYNTH_DOWNLOAD_SOURCE=%s DIFFSYNTH_SKIP_DOWNLOAD=%s",
        os.environ["DIFFSYNTH_MODEL_BASE_PATH"],
        os.environ["HF_HOME"],
        os.environ.get("DIFFSYNTH_DOWNLOAD_SOURCE"),
        os.environ.get("DIFFSYNTH_SKIP_DOWNLOAD"),
    )
    return base
