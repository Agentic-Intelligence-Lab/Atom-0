import dataclasses

import pytest

from openpi.cotrain import action_space
from openpi.cotrain import config
from openpi.cotrain import data_loader
from openpi.cotrain import transforms as cotrain_transforms
from openpi.cotrain import weight_loaders
from openpi.cotrain.rlds_dataset import CotrainRLDSDataset


def test_registered_cotrain_configs_include_controlled_legacy32_ablation() -> None:
    assert {train_config.name for train_config in config._COTRAIN_CONFIGS} == {
        "cotrain_real_only",
        "cotrain_real_only_legacy32",
        "cotrain_piper30_legacy32_aliyun_replay",
        "cotrain_real_only_legacy32_aliyun_recipe",
        "cotrain_real_only_unified80_aliyun_recipe",
        "cotrain_real_robot",
        "cotrain_real_robot_fix",
        "cotrain_full_all_full_norm",
    }
    for train_config in config._COTRAIN_CONFIGS:
        if train_config.name in {
            "cotrain_real_only_legacy32",
            "cotrain_piper30_legacy32_aliyun_replay",
            "cotrain_real_only_legacy32_aliyun_recipe",
        }:
            continue
        assert train_config.model.action_dim == action_space.UNIFIED_ACTION_DIM
        datasets = config._resolve_unified_datasets(train_config.data.datasets, train_config.model)
        assert datasets
        assert all(
            dataset.unified_action_spec is action_space.UNIFIED_ACTION_SPECS[dataset.uid] for dataset in datasets
        )


def test_all_cotrain_configs_support_params_path_auto_detection() -> None:
    assert all(
        isinstance(train_config.weight_loader, weight_loaders.ShapeSafeCheckpointWeightLoader)
        for train_config in config._COTRAIN_CONFIGS
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


def test_piper30_uses_task_disjoint_validation_builder_as_direct_replacement() -> None:
    dataset = config._PIPER30_DATA.datasets[0]
    assert dataset.uid == "piper30"
    assert dataset.version == "1.1.0"
    assert dataset.builder_dir.endswith("/realworld_piper_infidata/1.1.0")
    assert "/realworld_piper_task_split/" in dataset.builder_dir
    assert dataset.train_split == "train"
    assert dataset.val_splits == {"seen": "seen_test", "unseen": "unseen_test"}
    assert config._PIPER30_TRAIN_EPISODES == 4_927


def test_legacy32_is_a_single_variable_action_space_ablation() -> None:
    unified = config.get_config("cotrain_real_only")
    legacy = config.get_config("cotrain_real_only_legacy32")

    assert legacy.model.action_dim == 32
    assert legacy.model.max_token_len == unified.model.max_token_len
    assert legacy.data.datasets == unified.data.datasets
    assert legacy.data.unified_action_space is False
    assert legacy.data.norm_stats_source_config == "cotrain_real_only"
    assert legacy.lr_schedule == unified.lr_schedule
    assert legacy.optimizer == unified.optimizer
    assert legacy.batch_size == unified.batch_size
    assert legacy.num_train_steps == unified.num_train_steps

    resolved = config._resolve_legacy32_datasets(legacy.data.datasets, legacy.model)
    assert {dataset.uid for dataset in resolved} == {"piper30", "piper2"}
    assert all(dataset.unified_action_spec is None for dataset in resolved)


def test_legacy32_norm_projection_restores_native_piper_order() -> None:
    legacy = config.get_config("cotrain_real_only_legacy32")
    source_dir = legacy.assets_dirs.parent / legacy.data.norm_stats_source_config
    datasets = config._resolve_legacy32_datasets(legacy.data.datasets, legacy.model)
    projected = config.load_per_dataset_norm_stats(
        source_dir,
        datasets,
        project_unified_to_native=True,
    )

    for dataset in datasets:
        spec = action_space.UNIFIED_ACTION_SPECS[dataset.uid]
        native_targets = [target for _, target in sorted(spec.action_mapping)]
        unified = config.load_per_dataset_norm_stats(
            source_dir, (dataclasses.replace(dataset, unified_action_spec=spec),)
        )
        assert projected[dataset.uid]["state"].mean.shape == (14,)
        assert projected[dataset.uid]["actions"].mean.shape == (14,)
        assert projected[dataset.uid]["actions"].mean.tolist() == pytest.approx(
            unified[dataset.uid]["actions"].mean[native_targets].tolist()
        )


def test_aliyun_replay_restores_confirmed_historical_training_contract() -> None:
    replay = config.get_config("cotrain_piper30_legacy32_aliyun_replay")

    assert [dataset.uid for dataset in replay.data.datasets] == ["piper30"]
    assert replay.data.unified_action_space is False
    assert replay.data.include_action_prompt_prefix is False
    assert replay.model.action_dim == 32
    assert replay.model.max_token_len == 200
    assert replay.num_train_steps == 20_000
    assert replay.lr_schedule.warmup_steps == 1_000
    assert replay.lr_schedule.peak_lr == pytest.approx(2.5e-5)
    assert replay.lr_schedule.decay_steps == 30_000
    assert replay.lr_schedule.decay_lr == pytest.approx(2.5e-6)
    assert replay.save_interval == 5_000

    data_config = replay.data.create(replay.assets_dirs, replay.model)
    assert any(
        isinstance(transform, cotrain_transforms.DropPromptPrefix) for transform in data_config.data_transforms.inputs
    )

    resolved = config._resolve_legacy32_datasets(replay.data.datasets, replay.model)
    assert [dataset.uid for dataset in resolved] == ["piper30"]
    assert resolved[0].unified_action_spec is None


def test_aliyun_recipe_dataset_comparison_only_adds_piper2() -> None:
    baseline = config.get_config("cotrain_piper30_legacy32_aliyun_replay")
    add_piper2 = config.get_config("cotrain_real_only_legacy32_aliyun_recipe")

    assert [dataset.uid for dataset in baseline.data.datasets] == ["piper30"]
    assert {dataset.uid for dataset in add_piper2.data.datasets} == {"piper30", "piper2"}
    for field in dataclasses.fields(baseline):
        if field.name not in {"name", "data"}:
            assert getattr(add_piper2, field.name) == getattr(baseline, field.name), field.name
    for field in dataclasses.fields(baseline.data):
        if field.name not in {"rlds_data_dir", "datasets"}:
            assert getattr(add_piper2.data, field.name) == getattr(baseline.data, field.name), field.name
    assert add_piper2.data.unified_action_space is False
    assert add_piper2.data.include_action_prompt_prefix is False
    assert add_piper2.data.norm_stats_source_config == "cotrain_real_only"


def test_unified80_aliyun_recipe_only_changes_formal_training_1_recipe() -> None:
    formal = config.get_config("cotrain_real_only")
    comparison = config.get_config("cotrain_real_only_unified80_aliyun_recipe")
    legacy_comparison = config.get_config("cotrain_real_only_legacy32_aliyun_recipe")

    for field in dataclasses.fields(formal):
        if field.name not in {"name", "data", "lr_schedule", "num_train_steps", "save_interval"}:
            assert getattr(comparison, field.name) == getattr(formal, field.name), field.name
    for field in dataclasses.fields(formal.data):
        if field.name != "norm_stats_source_config":
            assert getattr(comparison.data, field.name) == getattr(formal.data, field.name), field.name
    assert comparison.data.norm_stats_source_config == formal.name

    assert comparison.lr_schedule == legacy_comparison.lr_schedule
    assert comparison.num_train_steps == legacy_comparison.num_train_steps == 20_000
    assert comparison.save_interval == legacy_comparison.save_interval == 5_000
    assert comparison.lr_schedule.warmup_steps == 1_000
    assert comparison.lr_schedule.peak_lr == pytest.approx(2.5e-5)
    assert comparison.lr_schedule.decay_steps == 30_000
    assert comparison.lr_schedule.decay_lr == pytest.approx(2.5e-6)


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
