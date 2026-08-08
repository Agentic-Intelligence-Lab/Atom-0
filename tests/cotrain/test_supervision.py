import dataclasses

import pytest

from openpi.cotrain import action_space
from openpi.cotrain import config
from openpi.cotrain import fk_eef
from openpi.cotrain import supervision
from openpi.cotrain.modes import ActionSupervisionMode, PromptActionMode, resolve_prompt_prefix


def test_supervised_action_mask_joint_and_eef_includes_fk_slots():
    spec = dataclasses.replace(
        action_space.UNIFIED_ACTION_SPECS["robocoin_airbot_mmk2_s36_a36"],
        fk_eef_slots=(7, 8, 9, 10, 11, 12, 36, 37, 38, 39, 40, 41),
        supervision_mode=ActionSupervisionMode.JOINT_AND_EEF,
    )
    mask = spec.action_mask
    assert any(mask[index] for index in spec.fk_eef_slots)
    assert any(mask[index] for index in spec.action_target_slots)


def test_supervised_action_mask_eef_only_excludes_joints():
    spec = dataclasses.replace(
        action_space.UNIFIED_ACTION_SPECS["robocoin_airbot_mmk2_s36_a36"],
        fk_eef_slots=(7, 8, 9, 10, 11, 12, 36, 37, 38, 39, 40, 41),
        supervision_mode=ActionSupervisionMode.EEF,
    )
    mask = spec.action_mask
    assert all(not mask[index] for index in spec.action_target_slots)
    assert all(mask[index] for index in spec.fk_eef_slots)


def test_supervised_action_mask_joint_excludes_fk_eef():
    spec = dataclasses.replace(
        action_space.UNIFIED_ACTION_SPECS["robocoin_airbot_mmk2_s36_a36"],
        fk_eef_slots=(7, 8, 9, 10, 11, 12, 36, 37, 38, 39, 40, 41),
        supervision_mode=ActionSupervisionMode.JOINT,
    )
    mask = spec.action_mask
    assert all(not mask[index] for index in spec.fk_eef_slots)
    assert any(mask[index] for index in spec.action_target_slots)


def test_native_eef_spec_egoverse():
    spec = action_space.UNIFIED_ACTION_SPECS["egoverse_eva"]
    assert supervision.is_native_eef_spec(spec)
    eef_mask = supervision.supervised_action_mask(
        dataclasses.replace(spec, supervision_mode=ActionSupervisionMode.EEF),
        ActionSupervisionMode.EEF,
    )
    assert sum(eef_mask) == len(spec.action_target_slots)


def test_piper_has_no_eef_supervision_without_fk(monkeypatch):
    monkeypatch.setattr(fk_eef, "enabled_fk_dataset_ids", lambda urdf_dir=None: ())
    with pytest.raises(ValueError, match="no EEF supervision slots"):
        config._resolve_unified_spec("piper30", supervision_mode=ActionSupervisionMode.EEF)


def test_piper_has_eef_supervision_when_fk_enabled():
    spec = config._resolve_unified_spec("piper30", supervision_mode=ActionSupervisionMode.EEF)
    assert spec.fk_eef_slots == fk_eef.FK_EEF_SPECS["piper30"].eef_slots
    assert any(spec.action_mask)


def test_supervision_mode_does_not_change_fingerprint():
    base = action_space.UNIFIED_ACTION_SPECS["piper30"]
    with_mode = dataclasses.replace(base, supervision_mode=ActionSupervisionMode.EEF)
    assert base.fingerprint == with_mode.fingerprint


def test_fk_eef_slots_do_not_change_fingerprint():
    base = action_space.UNIFIED_ACTION_SPECS["piper30"]
    with_fk = dataclasses.replace(base, fk_eef_slots=(7, 8, 9, 10, 11, 12))
    assert base.fingerprint == with_fk.fingerprint


def test_prompt_prefix_override():
    assert resolve_prompt_prefix(
        mode=PromptActionMode.JOINT,
        native_prefix="Action Mode: eef. EEF Frame: foo. ",
    ) == "Action Mode: joint. "
    assert resolve_prompt_prefix(
        mode=PromptActionMode.EEF,
        native_prefix="Action Mode: joint. ",
        eef_frame="obs_head_pose",
    ) == "Action Mode: eef. EEF Frame: obs_head_pose. "
    assert (
        resolve_prompt_prefix(
            mode=PromptActionMode.NATIVE,
            native_prefix="Action Mode: joint. ",
        )
        == "Action Mode: joint. "
    )


def test_dispatch_prompt_prefix_transform():
    from openpi.cotrain import transforms as cotrain_transforms

    transform = cotrain_transforms.DispatchPromptPrefix(prompt_action_mode=PromptActionMode.EEF)
    out = transform({"prompt_prefix": "Action Mode: joint. ", "eef_frame": "source_pose_frame"})
    assert out["prompt_prefix"] == "Action Mode: eef. EEF Frame: source_pose_frame. "


def test_eef_only_config_registered():
    train_config = config.get_config("fastwam_cotrain_fk_eef_plus_piper_ego_eef_only")
    assert train_config.assets_name == "cotrain_fk_eef_plus_piper_ego"
    assert train_config.data.action_supervision_mode == ActionSupervisionMode.EEF
    assert train_config.data.prompt_action_mode == PromptActionMode.EEF


def test_eef_only_mix_excludes_piper_when_fk_disabled(monkeypatch):
    monkeypatch.setattr(fk_eef, "enabled_fk_dataset_ids", lambda urdf_dir=None: ())
    datasets = config._fk_eef_plus_piper_ego_eef_only_datasets()
    ids = {ds.uid for ds in datasets}
    assert "piper30" not in ids
    assert "piper2" not in ids
    assert ids  # egoverse subsets remain
