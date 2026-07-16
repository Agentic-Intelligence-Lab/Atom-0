from __future__ import annotations

import json

import numpy as np
import pytest

from openpi.cotrain import action_space

EXPECTED_DATASET_IDS = {
    "agibot",
    "droid",
    "egoverse_aria",
    "egoverse_eva",
    "egoverse_human",
    "egoverse_mecka",
    "egoverse_scale",
    "piper30",
    "piper2",
    "robocoin_agilex_cobot_magic_s26_a26",
    "robocoin_airbot_mmk2_s36_a36",
    "robocoin_galaxea_r1_lite_upper_s14_a14",
    "robocoin_realman_rmc_aida_l_s28_a28",
    "robocoin_unitree_g1_dex3_s28_a28",
    "robocoin_agilex_decoupled_magic_s14_a14_fps30",
    "robocoin_agilex_decoupled_magic_s14_a14_fps50",
    "robocoin_agilex_decoupled_magic_s26_a26",
    "robocoin_aloha_s26_a26",
    "robocoin_alpha_bot_2_s28_a28",
    "robocoin_discover_aitbot_mmk2_s36_a36",
    "robocoin_galaxea_r1_lite_s14_a14",
    "robocoin_galaxea_r1_lite_s16_a18",
    "robocoin_leju_robot_s118_a54",
    "robocoin_leju_robot_s54_a54",
    "robocoin_realman_rmc_aidal_s28_a28",
    "robocoin_ruantong_a2d_s17_a17",
    "robocoin_ruantong_a2d_s41_a34",
    "robocoin_unitree_g1_s28_a28_high",
    "robocoin_unitree_g1_s28_a28",
    "robocoin_unknown_s30_a30_high",
    "robocoin_yinhe_s49_a16",
    "robomind_agilex_cobot_magic_s14_a14",
    "robomind_franka_fr3_dual_s16_a16",
    "robomind_franka_panda_s8_a8",
    "robomind_franka_sim_franka_s8_a8",
    "robomind_franka_sim_simulation_s8_a8",
    "robomind_franka_sim_simulation_no_front_s8_a8",
    "robomind_franka_sim_none_s8_a8",
    "robomind_tienkung_gello_s16_a16",
    "robomind_tienkung_prod1_gello_s16_a16",
    "robomind_tienkung_xsens_s14_a14",
    "robomind_tienkung_sim_s38_a38",
    "robomind_tienkung_real_s38_a38",
    "robomind_ur5e_s7_a7",
}


def test_registry_covers_all_documented_builders() -> None:
    assert set(action_space.UNIFIED_ACTION_SPECS) == EXPECTED_DATASET_IDS
    assert len(EXPECTED_DATASET_IDS) == 44


@pytest.mark.parametrize("dataset_id", sorted(EXPECTED_DATASET_IDS))
def test_masks_are_80d_and_temporal_slots_are_mapped(dataset_id: str) -> None:
    spec = action_space.UNIFIED_ACTION_SPECS[dataset_id]
    assert len(spec.action_mask) == action_space.UNIFIED_ACTION_DIM
    assert len(spec.delta_mask) == action_space.UNIFIED_ACTION_DIM
    assert sum(spec.action_mask) == len(spec.action_mapping)
    assert set(spec.absolute_to_delta_slots) <= set(spec.action_target_slots)
    assert set(spec.absolute_to_delta_slots) <= set(spec.state_target_slots)
    assert not spec.already_delta_slots


def test_only_egoverse_maps_eef_slots() -> None:
    eef_slots = set(range(action_space.LEFT_EEF_POSITION, action_space.LEFT_EEF_EULER + 3))
    eef_slots |= set(range(action_space.RIGHT_EEF_POSITION, action_space.RIGHT_EEF_EULER + 3))
    users = {
        dataset_id
        for dataset_id, spec in action_space.UNIFIED_ACTION_SPECS.items()
        if set(spec.action_target_slots) & eef_slots
    }
    assert users == {
        "egoverse_aria",
        "egoverse_eva",
        "egoverse_human",
        "egoverse_mecka",
        "egoverse_scale",
    }


def test_robocoin_mixed_eef_sources_are_dropped() -> None:
    expected_dropped = {
        "robocoin_agilex_cobot_magic_s26_a26": set(range(7, 13)) | set(range(20, 26)),
        "robocoin_realman_rmc_aida_l_s28_a28": set(range(8, 14)) | set(range(22, 28)),
        "robocoin_agilex_decoupled_magic_s26_a26": set(range(7, 13)) | set(range(20, 26)),
        "robocoin_aloha_s26_a26": set(range(6, 12)) | set(range(19, 25)),
        "robocoin_alpha_bot_2_s28_a28": set(range(7, 13)) | set(range(20, 26)),
        "robocoin_realman_rmc_aidal_s28_a28": set(range(8, 14)) | set(range(22, 28)),
        "robocoin_ruantong_a2d_s41_a34": set(range(14, 28)),
    }
    for dataset_id, dropped in expected_dropped.items():
        mapped_sources = {source for source, _ in action_space.UNIFIED_ACTION_SPECS[dataset_id].action_mapping}
        assert mapped_sources.isdisjoint(dropped)


def test_yinhe_state_reorder_is_explicit() -> None:
    spec = action_space.UNIFIED_ACTION_SPECS["robocoin_yinhe_s49_a16"]
    assert tuple(source for source, _ in spec.state_mapping) == tuple(range(5, 21))
    assert tuple(source for source, _ in spec.action_mapping) == tuple(range(16))


def test_invalid_temporal_spec_is_rejected() -> None:
    with pytest.raises(ValueError, match="without mapped state"):
        action_space.UnifiedActionSpec(
            state_mapping=action_space.dims(0, action_space.LEFT_ARM, 1),
            action_mapping=action_space.dims(0, action_space.RIGHT_ARM, 1),
            absolute_to_delta_slots=(action_space.RIGHT_ARM,),
        )


def test_map_array_scatters_and_zero_fills() -> None:
    spec = action_space.UNIFIED_ACTION_SPECS["piper30"]
    source = np.arange(14, dtype=np.float32)
    mapped = action_space.map_array(source, spec.action_mapping)
    assert mapped.shape == (action_space.UNIFIED_ACTION_DIM,)
    np.testing.assert_array_equal(mapped[:6], source[:6])
    assert mapped[action_space.LEFT_GRIPPER] == source[6]
    np.testing.assert_array_equal(mapped[action_space.RIGHT_ARM : action_space.RIGHT_ARM + 6], source[7:13])
    assert mapped[action_space.RIGHT_GRIPPER] == source[13]
    mask = np.asarray(spec.action_mask)
    assert np.count_nonzero(mapped[mask]) == 13  # source a1 is intentionally zero
    assert np.all(mapped[~mask] == 0)


def test_unmap_array_restores_mapped_source_indices() -> None:
    spec = action_space.UNIFIED_ACTION_SPECS["robocoin_aloha_s26_a26"]
    source = np.arange(26, dtype=np.float32)
    unified = action_space.map_array(source, spec.action_mapping)
    restored = action_space.unmap_array(unified, spec.action_mapping, source_dim=26)
    mapped_sources = np.asarray([source_index for source_index, _ in spec.action_mapping])
    dropped_sources = np.asarray(sorted(set(range(26)) - set(mapped_sources)))
    np.testing.assert_array_equal(restored[mapped_sources], source[mapped_sources])
    np.testing.assert_array_equal(restored[dropped_sources], 0)


def test_unified_slot_names_cover_all_80_slots() -> None:
    assert len(action_space.UNIFIED_SLOT_NAMES) == action_space.UNIFIED_ACTION_DIM
    assert action_space.UNIFIED_SLOT_NAMES[action_space.LEFT_ARM] == "left_arm_joint_1"
    assert action_space.UNIFIED_SLOT_NAMES[action_space.RIGHT_GRIPPER] == "right_gripper"


def test_mapping_metadata_rejects_stale_stats(tmp_path) -> None:
    spec = action_space.UNIFIED_ACTION_SPECS["droid"]
    action_space.write_metadata(tmp_path, spec)
    action_space.validate_metadata(tmp_path, spec)

    path = tmp_path / "unified_action_space.json"
    metadata = json.loads(path.read_text())
    metadata["width"] = 64
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="mapping mismatch"):
        action_space.validate_metadata(tmp_path, spec)


def test_tensorflow_trajectory_mapping() -> None:
    tf = pytest.importorskip("tensorflow")
    spec = action_space.UNIFIED_ACTION_SPECS["egoverse_eva"]
    source = np.arange(24, dtype=np.float32).reshape(2, 12)
    mapped = action_space.map_trajectory_tensorflow(
        {"state": tf.constant(source), "actions": tf.constant(source + 100)}, spec
    )
    np.testing.assert_array_equal(mapped["state"].numpy(), action_space.map_array(source, spec.state_mapping))
    np.testing.assert_array_equal(mapped["actions"].numpy(), action_space.map_array(source + 100, spec.action_mapping))
    np.testing.assert_array_equal(mapped["action_mask"].numpy(), np.broadcast_to(spec.action_mask, (2, 80)))


def test_delta_is_applied_once_only_to_declared_slots() -> None:
    spec = action_space.UNIFIED_ACTION_SPECS["piper30"]
    state = np.arange(80, dtype=np.float32)
    actions = np.broadcast_to(state + 3, (4, 80)).copy()
    converted = action_space.apply_delta(state, actions, spec.delta_mask)
    delta_mask = np.asarray(spec.delta_mask)
    np.testing.assert_array_equal(converted[:, delta_mask], 3)
    np.testing.assert_array_equal(converted[:, ~delta_mask], actions[:, ~delta_mask])


def test_second_piper_drop_uses_audited_physical_layout() -> None:
    spec = action_space.UNIFIED_ACTION_SPECS["piper2"]
    expected_mapping = (
        action_space.dims(0, action_space.LEFT_ARM, 6)
        + action_space.dims(6, action_space.LEFT_GRIPPER, 1)
        + action_space.dims(7, action_space.RIGHT_ARM, 6)
        + action_space.dims(13, action_space.RIGHT_GRIPPER, 1)
    )
    expected_delta_slots = action_space.slots(action_space.LEFT_ARM, 6) + action_space.slots(
        action_space.RIGHT_ARM, 6
    )

    assert spec.state_mapping == expected_mapping
    assert spec.action_mapping == expected_mapping
    assert spec.absolute_to_delta_slots == expected_delta_slots
    assert spec.already_delta_slots == ()
