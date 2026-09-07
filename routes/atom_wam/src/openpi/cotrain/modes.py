"""Shared enums for co-training supervision and prompt contracts."""

from __future__ import annotations

import enum


class ActionSupervisionMode(enum.Enum):
    """Which unified 80D slots participate in the action loss."""

    JOINT = "joint"
    EEF = "eef"
    JOINT_AND_EEF = "joint_and_eef"


class PromptActionMode(enum.Enum):
    """How to set the tokenizer ``prompt_prefix`` at runtime (RLDS is unchanged)."""

    NATIVE = "native"
    JOINT = "joint"
    EEF = "eef"


def format_prompt_prefix(action_mode: str, eef_frame: str | None = None) -> str:
    """Build the action-metadata prefix prepended before ``Task:``."""
    prefix = f"Action Mode: {action_mode}. "
    if action_mode != "eef" or eef_frame is None:
        return prefix
    frame = str(eef_frame).strip()
    if not frame:
        return prefix
    return f"{prefix}EEF Frame: {frame}. "


def resolve_prompt_prefix(
    *,
    mode: PromptActionMode,
    native_prefix: str | None,
    eef_frame: str | None = None,
) -> str | None:
    """Resolve the prefix to feed the tokenizer."""
    if mode == PromptActionMode.NATIVE:
        return native_prefix
    if mode == PromptActionMode.JOINT:
        return format_prompt_prefix("joint")
    if mode == PromptActionMode.EEF:
        return format_prompt_prefix("eef", eef_frame)
    raise ValueError(f"Unsupported PromptActionMode: {mode!r}")
