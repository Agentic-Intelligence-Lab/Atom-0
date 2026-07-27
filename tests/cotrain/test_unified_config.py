import dataclasses

import pytest

from openpi.cotrain import action_space
from openpi.cotrain import config
from openpi.cotrain import data_loader
from openpi.cotrain.rlds_dataset import CotrainRLDSDataset


def test_all_registered_cotrain_configs_resolve_to_unified_80d() -> None:
    assert {train_config.name for train_config in config._COTRAIN_CONFIGS} == {
        "cotrain_real_only",
        "cotrain_real_robot",
        "cotrain_real_robot_fix",
        "cotrain_full_all_full_norm",
        "egoscale_stage1_ego",
        "egoscale_stage2_robot",
        "egoscale_stage2_aligned",
        "egoscale_stage2_egomimic",
        "egoscale_stage2_egomimic_all",
        "egoscale_stage3_robot",
    }
    for train_config in config._COTRAIN_CONFIGS:
        assert train_config.model.action_dim == action_space.UNIFIED_ACTION_DIM
        datasets = config._resolve_unified_datasets(train_config.data.datasets, train_config.model)
        assert datasets
        assert all(
            dataset.unified_action_spec is action_space.UNIFIED_ACTION_SPECS[dataset.uid] for dataset in datasets
        )


def test_validation_batch_size_is_independent_with_legacy_fallback() -> None:
    train_config = config.get_config("cotrain_real_robot_fix")
    assert train_config.batch_size == 32
    assert data_loader.resolve_val_batch_size(train_config) == 96

    legacy = dataclasses.replace(train_config, batch_size=512, val_batch_size=None)
    assert data_loader.resolve_val_batch_size(legacy) == 512

    invalid = dataclasses.replace(train_config, val_batch_size=0)
    with pytest.raises(ValueError, match="val_batch_size must be positive"):
        data_loader.resolve_val_batch_size(invalid)


def test_cotrain_rejects_non_80d_model() -> None:
    model = dataclasses.replace(config._UNIFIED_PI05_MODEL, action_dim=32)
    with pytest.raises(ValueError, match="require action_dim=80"):
        config._resolve_unified_datasets(config._PIPER30_DATA.datasets, model)


def test_cotrain_rejects_dataset_without_mapping() -> None:
    dataset = CotrainRLDSDataset(name="new_builder", dataset_id="new_builder", version="1.0.0", weight=1.0)
    with pytest.raises(ValueError, match="no registered unified 80D action mapping"):
        config._resolve_unified_datasets((dataset,), config._UNIFIED_PI05_MODEL)


def test_real_only_contains_both_in_house_piper_datasets() -> None:
    assert {dataset.uid for dataset in config._REAL_ONLY_DATA.datasets} == {"piper30", "piper2"}
    assert sum(dataset.weight for dataset in config._REAL_ONLY_DATA.datasets) == pytest.approx(1.0)


def test_real_robot_contains_public_robot_data_but_no_egoverse() -> None:
    dataset_ids = {dataset.uid for dataset in config._REAL_ROBOT_DATA.datasets}
    assert {"piper30", "piper2", "agibot", "droid"} <= dataset_ids
    assert any(dataset_id.startswith("robocoin_") for dataset_id in dataset_ids)
    assert any(dataset_id.startswith("robomind_") for dataset_id in dataset_ids)
    assert not any(dataset_id.startswith("egoverse_") for dataset_id in dataset_ids)
    assert dataset_ids.isdisjoint(config._FULL_ALL_EXCLUDED_DATASET_IDS)
    assert sum(dataset.weight for dataset in config._REAL_ROBOT_DATA.datasets) == pytest.approx(1.0)


def test_real_robot_fix_excludes_audited_risky_datasets_and_renormalizes() -> None:
    original_ids = {dataset.uid for dataset in config._REAL_ROBOT_DATA.datasets}
    fixed_ids = {dataset.uid for dataset in config._REAL_ROBOT_FIX_DATA.datasets}

    assert fixed_ids == original_ids - config._REAL_ROBOT_FIX_EXCLUDED_DATASET_IDS
    assert len(fixed_ids) == 34
    assert sum(dataset.weight for dataset in config._REAL_ROBOT_FIX_DATA.datasets) == pytest.approx(1.0)
    assert config.get_config("cotrain_real_robot_fix").data is config._REAL_ROBOT_FIX_DATA


def test_full_all_full_norm_adds_egoverse_to_audited_real_robot_data() -> None:
    real_robot_fix_ids = {dataset.uid for dataset in config._REAL_ROBOT_FIX_DATA.datasets}
    full_ids = {dataset.uid for dataset in config._FULL_ALL_FIX_DATA.datasets}
    ego_ids = {dataset.uid for dataset in config._EGOVERSE_FULL_DATA.datasets}

    assert full_ids == real_robot_fix_ids | ego_ids
    assert len(full_ids) == 39
    assert full_ids.isdisjoint(config._FULL_ALL_EXCLUDED_DATASET_IDS)
    assert full_ids.isdisjoint(config._REAL_ROBOT_FIX_EXCLUDED_DATASET_IDS)
    assert sum(dataset.weight for dataset in config._FULL_ALL_FIX_DATA.datasets) == pytest.approx(1.0)
    assert config.get_config("cotrain_full_all_full_norm").data is config._FULL_ALL_FIX_DATA


def test_robomind_weights_use_actual_train_split_episode_counts() -> None:
    expected = {
        "robomind_agilex_cobot_magic_s14_a14": 9_855,
        "robomind_franka_fr3_dual_s16_a16": 1_685,
        "robomind_franka_panda_s8_a8": 14_956,
        "robomind_franka_sim_franka_s8_a8": 8_445,
        "robomind_franka_sim_simulation_s8_a8": 8_662,
        "robomind_franka_sim_simulation_no_front_s8_a8": 150,
        "robomind_franka_sim_none_s8_a8": 211,
        "robomind_tienkung_gello_s16_a16": 5_402,
        "robomind_tienkung_prod1_gello_s16_a16": 2_811,
        "robomind_tienkung_xsens_s14_a14": 5_775,
        "robomind_tienkung_sim_s38_a38": 3_767,
        "robomind_tienkung_real_s38_a38": 139,
        "robomind_ur5e_s7_a7": 25_061,
    }
    actual = {dataset_id: train_episodes for dataset_id, _, train_episodes, _, _, _ in config._ROBOMIND_FULL_REPOS}
    assert actual == expected
    assert config._ROBOMIND_FULL_EPISODES == 86_919


def test_robot_stage_excludes_all_egoverse_datasets() -> None:
    robot_ids = {dataset.uid for dataset in config._ROBOT_ALL_DATA.datasets}
    assert robot_ids
    assert robot_ids.isdisjoint(config._EGOVERSE_DATASET_IDS)
    assert abs(sum(dataset.weight for dataset in config._ROBOT_ALL_DATA.datasets) - 1.0) < 1e-6


def test_staged_configs_use_expected_data_and_strict_checkpoint_loader() -> None:
    stage1_datasets = config._EGOSCALE_STAGE1_EGO.data.datasets
    assert {dataset.uid for dataset in stage1_datasets} == {
        "egoverse_aria",
        "egoverse_eva",
        "egoverse_human",
        "egoverse_mecka",
    }
    assert {dataset.uid for dataset in stage1_datasets}.isdisjoint(
        config._EGOSCALE_STAGE1_EXCLUDED_DATASET_IDS
    )
    assert sum(dataset.weight for dataset in stage1_datasets) == pytest.approx(1.0)
    assert all(dataset.restructure_name == "egoverse_cartesian_chunk" for dataset in stage1_datasets)
    assert all(dataset.precomputed_action_chunk for dataset in stage1_datasets)
    assert all(dataset.precomputed_action_source == "actions_cartesian" for dataset in stage1_datasets)
    assert all(dataset.precomputed_action_horizon == 100 for dataset in stage1_datasets)
    assert config._EGOSCALE_STAGE1_EGO.norm_stats_assets_name == "egoscale_stage1_ego_cartesian_clean"
    assert config._EGOSCALE_STAGE2_ROBOT.data is config._ROBOT_ALL_DATA
    assert config._EGOSCALE_STAGE2_ALIGNED.data is config._ALIGNED_PARALLEL_GRIPPER_DATA
    assert config._EGOSCALE_STAGE2_EGOMIMIC.data is config._EGOMIMIC_GROCERIES_DATA
    assert {dataset.uid for dataset in config._EGOSCALE_STAGE2_EGOMIMIC.data.datasets} == {
        "egomimic_groceries_human",
        "egomimic_groceries_robot",
    }
    assert {
        dataset.uid: dataset.action_dim
        for dataset in config._EGOSCALE_STAGE2_EGOMIMIC.data.datasets
    } == {
        "egomimic_groceries_human": 6,
        "egomimic_groceries_robot": 20,
    }
    assert {
        dataset.uid: dataset.val_splits
        for dataset in config._EGOSCALE_STAGE2_EGOMIMIC.data.datasets
    } == {
        "egomimic_groceries_human": {"seen": "seen_test", "unseen": "unseen_test"},
        "egomimic_groceries_robot": {"seen": "train", "unseen": "train"},
    }
    assert all(dataset.precomputed_action_chunk for dataset in config._EGOSCALE_STAGE2_EGOMIMIC.data.datasets)
    assert config._EGOSCALE_STAGE2_EGOMIMIC_ALL.data is config._EGOMIMIC_ALL_DATA
    assert {
        dataset.uid: dataset.action_dim
        for dataset in config._EGOSCALE_STAGE2_EGOMIMIC_ALL.data.datasets
    } == {
        "egomimic_bowlplace_human": 3,
        "egomimic_bowlplace_robot": 10,
        "egomimic_groceries_human": 6,
        "egomimic_groceries_robot": 20,
        "egomimic_smallclothfold_human": 6,
        "egomimic_smallclothfold_robot": 20,
    }
    assert all(
        dataset.weight == pytest.approx(1 / 6)
        for dataset in config._EGOSCALE_STAGE2_EGOMIMIC_ALL.data.datasets
    )
    assert {
        dataset.uid: dataset.val_splits
        for dataset in config._EGOSCALE_STAGE2_EGOMIMIC_ALL.data.datasets
    } == {
        "egomimic_bowlplace_human": {"seen": "seen_test", "unseen": "unseen_test"},
        "egomimic_bowlplace_robot": {"seen": "train", "unseen": "train"},
        "egomimic_groceries_human": {"seen": "seen_test", "unseen": "unseen_test"},
        "egomimic_groceries_robot": {"seen": "train", "unseen": "train"},
        "egomimic_smallclothfold_human": {"seen": "seen_test", "unseen": "unseen_test"},
        "egomimic_smallclothfold_robot": {"seen": "seen_test", "unseen": "unseen_test"},
    }
    assert all(
        dataset.precomputed_action_chunk
        for dataset in config._EGOSCALE_STAGE2_EGOMIMIC_ALL.data.datasets
    )
    assert config._EGOSCALE_STAGE3_ROBOT.data is config._ROBOT_ALL_DATA
    for staged in (
        config._EGOSCALE_STAGE2_ROBOT,
        config._EGOSCALE_STAGE2_ALIGNED,
        config._EGOSCALE_STAGE2_EGOMIMIC,
        config._EGOSCALE_STAGE2_EGOMIMIC_ALL,
        config._EGOSCALE_STAGE3_ROBOT,
    ):
        assert staged.weight_loader.__class__.__name__ == "CheckpointWeightLoader"


def test_stage2_freeze_filter_keeps_action_expert_and_vision_trainable() -> None:
    freeze = config._freeze_vlm_language_filter()
    assert freeze(("PaliGemma", "llm", "layers", "attn"), object())
    assert not freeze(("PaliGemma", "llm", "layers", "attn_1"), object())
    assert not freeze(("PaliGemma", "img", "encoderblock", "attn"), object())
