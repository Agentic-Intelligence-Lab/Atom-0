"""Action supervision helpers for co-training."""

from __future__ import annotations

from openpi.cotrain import action_space as cotrain_action_space
from openpi.cotrain.modes import ActionSupervisionMode

UnifiedActionSpec = cotrain_action_space.UnifiedActionSpec

_EEF_SLOT_INDICES = frozenset(
    cotrain_action_space.slots(cotrain_action_space.LEFT_EEF_POSITION, 3)
    + cotrain_action_space.slots(cotrain_action_space.LEFT_EEF_EULER, 3)
    + cotrain_action_space.slots(cotrain_action_space.RIGHT_EEF_POSITION, 3)
    + cotrain_action_space.slots(cotrain_action_space.RIGHT_EEF_EULER, 3)
)

# Native EgoVerse-style layouts may also carry parallel gripper scalars (e.g. eva 14D).
_NATIVE_EEF_SLOT_INDICES = _EEF_SLOT_INDICES | frozenset(
    {
        cotrain_action_space.LEFT_GRIPPER,
        cotrain_action_space.RIGHT_GRIPPER,
    }
)


def is_native_eef_spec(spec: UnifiedActionSpec) -> bool:
    """True when every mapped action slot is EEF pose and/or gripper (no joints)."""
    if not spec.action_target_slots:
        return False
    return all(index in _NATIVE_EEF_SLOT_INDICES for index in spec.action_target_slots)


def supervised_action_mask(
    spec: UnifiedActionSpec,
    mode: ActionSupervisionMode,
) -> tuple[bool, ...]:
    """Return the per-dimension loss mask for a unified spec and supervision mode."""
    if mode == ActionSupervisionMode.JOINT:
        targets = set(spec.action_target_slots)
    elif mode == ActionSupervisionMode.EEF:
        if spec.fk_eef_slots:
            targets = set(spec.fk_eef_slots)
        elif is_native_eef_spec(spec):
            targets = set(spec.action_target_slots)
        else:
            targets = set()
    elif mode == ActionSupervisionMode.JOINT_AND_EEF:
        targets = set(spec.action_target_slots) | set(spec.fk_eef_slots)
    else:
        raise ValueError(f"Unsupported ActionSupervisionMode: {mode!r}")

    width = cotrain_action_space.UNIFIED_ACTION_DIM
    return tuple(index in targets for index in range(width))
