"""FastWAM checkpoint path helpers and weight loading (train / fine-tune)."""

from __future__ import annotations

import logging
from pathlib import Path

import safetensors.torch
import torch
import torch.nn.parallel


def resolve_model_safetensors(path: str | Path) -> Path:
    """Accept a step dir, exp dir, or direct ``model.safetensors`` path."""
    p = Path(path).expanduser().resolve()
    if p.is_file() and p.name == "model.safetensors":
        return p
    if p.is_dir():
        direct = p / "model.safetensors"
        if direct.is_file():
            return direct
        step_dirs = sorted(
            (d for d in p.iterdir() if d.is_dir() and d.name.isdigit() and (d / "model.safetensors").is_file()),
            key=lambda d: int(d.name),
        )
        if step_dirs:
            return step_dirs[-1] / "model.safetensors"
    raise FileNotFoundError(
        f"No model.safetensors under {path!s}. Pass a step dir, exp dir, or the safetensors file."
    )


def find_latest_resume_dir(checkpoint_dir: Path) -> Path | None:
    if not checkpoint_dir.is_dir():
        return None
    step_dirs = sorted(
        (
            d
            for d in checkpoint_dir.iterdir()
            if d.is_dir()
            and d.name.isdigit()
            and (d / "model.safetensors").is_file()
            and (d / "metadata.pt").is_file()
        ),
        key=lambda d: int(d.name),
    )
    return step_dirs[-1] if step_dirs else None


def resolve_resume_dir(
    checkpoint_dir: Path,
    *,
    resume_from_step: int | None = None,
) -> Path | None:
    """Pick a step dir for ``--resume`` (latest, or an explicit step index)."""
    if resume_from_step is not None:
        step_dir = checkpoint_dir / str(int(resume_from_step))
        if step_dir.is_dir() and (step_dir / "model.safetensors").is_file():
            return step_dir
        raise FileNotFoundError(
            f"resume_from_step={resume_from_step} but no model.safetensors under {step_dir}"
        )
    return find_latest_resume_dir(checkpoint_dir)


def load_fastwam_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    *,
    strict: bool = True,
) -> Path:
    """Load weights into FastWAMPytorch (or DDP-wrapped). Returns resolved safetensors path."""
    model_path = resolve_model_safetensors(checkpoint_path)
    root = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    missing, unexpected = safetensors.torch.load_model(root, str(model_path), strict=strict)
    if missing:
        logging.warning("Missing keys when loading %s: %s", model_path, missing[:20])
    if unexpected:
        logging.warning("Unexpected keys when loading %s: %s", model_path, unexpected[:20])
    logging.info("Loaded FastWAM weights from %s", model_path)
    return model_path
