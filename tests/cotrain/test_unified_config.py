import dataclasses
import pathlib

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
        "cotrain_real_robot_ego_fix",
        "cotrain_full_all_full_norm",
        "cotrain_fk_eef_plus_piper_ego",
        "cotrain_fk_eef_plus_piper_ego_eef_only",
        "wam-cross-robot",
        "wam-cross-fix",
        "wam-cross-piper-ft",
        "hpt_cotrain_real_only",
        "hpt_cotrain_real_robot_ego_fix",
        "hpt_cotrain_smoke",
    }
    for train_config in config._COTRAIN_CONFIGS:
        assert train_config.model.action_dim == action_space.UNIFIED_ACTION_DIM
        datasets = config._resolve_unified_datasets(train_config.data.datasets, train_config.model)
        assert datasets
        for dataset in datasets:
            base = action_space.UNIFIED_ACTION_SPECS[dataset.uid]
            assert dataset.unified_action_spec is not None
            assert dataset.unified_action_spec.state_mapping == base.state_mapping
            assert dataset.unified_action_spec.action_mapping == base.action_mapping
            assert dataset.unified_action_spec.absolute_to_delta_slots == base.absolute_to_delta_slots
            # Native mapping only — never invent FK EEF slots from URDF presence.
            assert dataset.unified_action_spec.fk_eef_slots == ()
            assert set(dataset.unified_action_spec.action_target_slots) == set(base.action_target_slots)


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
    ego_ids = {dataset.uid for dataset in config._EGOVERSE_FULL_DATA.datasets} | {
        dataset.uid for dataset in config._EGOVERSE_RL2_DATA.datasets
    }

    assert full_ids == real_robot_fix_ids | ego_ids
    assert len(full_ids) == 41
    assert full_ids.isdisjoint(config._FULL_ALL_EXCLUDED_DATASET_IDS)
    assert full_ids.isdisjoint(config._REAL_ROBOT_FIX_EXCLUDED_DATASET_IDS)
    assert sum(dataset.weight for dataset in config._FULL_ALL_FIX_DATA.datasets) == pytest.approx(1.0)
    assert config.get_config("cotrain_full_all_full_norm").data is config._FULL_ALL_FIX_DATA


def test_real_robot_ego_fix_matches_legacy_norm_assets_mixture() -> None:
    real_robot_ids = {dataset.uid for dataset in config._REAL_ROBOT_DATA.datasets}
    ego_fix_ids = {dataset.uid for dataset in config._REAL_ROBOT_EGO_FIX_DATA.datasets}
    ego_subset = {
        "egoverse_aria",
        "egoverse_eva",
        "egoverse_human",
        "egoverse_mecka",
        "egoverse_rl2_eva",
        "egoverse_rl2_human",
    }

    aligned_subset = {
        "aligned_hangzhou_human_right",
        "aligned_hangzhou_robot_right",
        "aligned_shenzhen_human_bimanual",
        "aligned_shenzhen_robot_bimanual",
    }

    assert ego_fix_ids == real_robot_ids | ego_subset | aligned_subset
    assert "egoverse_scale" not in ego_fix_ids
    assert len(ego_fix_ids) == 47
    assert ego_fix_ids.isdisjoint(config._FULL_ALL_EXCLUDED_DATASET_IDS)
    assert sum(dataset.weight for dataset in config._REAL_ROBOT_EGO_FIX_DATA.datasets) == pytest.approx(1.0)
    assert config.get_config("cotrain_real_robot_ego_fix").data is config._REAL_ROBOT_EGO_FIX_DATA
    assert config.get_config("cotrain_real_robot_ego_fix").rlds_partition_builders_by_rank is True


def test_wam_cross_robot_mixture_and_model() -> None:
    assert {dataset.uid for dataset in config._WAM_CROSS_ROBOT_DATA.datasets} == config._WAM_CROSS_ROBOT_DATASET_IDS
    assert len(config._WAM_CROSS_ROBOT_DATA.datasets) == 15
    assert sum(dataset.weight for dataset in config._WAM_CROSS_ROBOT_DATA.datasets) == pytest.approx(1.0)
    assert not any(dataset.uid.startswith("egoverse_") for dataset in config._WAM_CROSS_ROBOT_DATA.datasets)
    assert "robocoin_unitree_g1_dex3_s28_a28" not in {d.uid for d in config._WAM_CROSS_ROBOT_DATA.datasets}

    fastwam = config.get_config("wam-cross-robot")
    assert fastwam.assets_name == "cotrain_real_robot_ego_fix"
    assert fastwam.data is config._WAM_CROSS_ROBOT_DATA
    assert fastwam.model.concat_multi_camera == "robot_wrist"
    assert fastwam.model.camera_keys == ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    assert fastwam.model.image_resolution == (288, 256)
    assert fastwam.lr_schedule.peak_lr == pytest.approx(1.0e-4)
    assert fastwam.lr_schedule.decay_lr == pytest.approx(1.0e-6)

    by_slot = data_loader.resolve_train_image_resize_hw_by_slot(fastwam.model)
    assert by_slot == {
        "base_0_rgb": (192, 256),
        "left_wrist_0_rgb": (96, 128),
        "right_wrist_0_rgb": (96, 128),
    }
    assert data_loader.resolve_train_image_resize_hw(fastwam.model) is None


def test_wam_cross_fix_mixture_and_model() -> None:
    ego_fix_ids = {dataset.uid for dataset in config._REAL_ROBOT_EGO_FIX_DATA.datasets}
    assert len(ego_fix_ids) == 47
    assert sum(uid.startswith("egoverse_") for uid in ego_fix_ids) == 6
    assert sum(uid.startswith("aligned_") for uid in ego_fix_ids) == 4

    fastwam = config.get_config("wam-cross-fix")
    assert fastwam.assets_name == "cotrain_real_robot_ego_fix"
    assert fastwam.data is config._REAL_ROBOT_EGO_FIX_DATA
    assert fastwam.model.concat_multi_camera == "robot_wrist"
    assert fastwam.model.camera_keys == ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    assert fastwam.model.image_resolution == (288, 256)
    assert fastwam.model.loss["lambda_ego_video"] == pytest.approx(0.60)
    assert fastwam.model.loss["lambda_robot_action"] == pytest.approx(1.00)
    assert fastwam.rlds_partition_builders_by_rank is True


def test_wam_cross_piper_ft_mixture_eval_and_ddp() -> None:
    piper_ids = {dataset.uid for dataset in config._WAM_CROSS_PIPER_DATA.datasets}
    assert piper_ids == {"piper2", "piper30"}
    assert sum(dataset.weight for dataset in config._WAM_CROSS_PIPER_DATA.datasets) == pytest.approx(1.0)

    ft = config.get_config("wam-cross-piper-ft")
    assert ft.data is config._WAM_CROSS_PIPER_DATA
    assert ft.rlds_partition_builders_by_rank is False
    assert ft.eval_interval == 1_000
    assert ft.run_action_mse is True
    assert ft.val_max_datasets is None
    assert ft.model.freeze_video_expert is True
    assert ft.model.skip_dit_load_from_pretrain is True
    assert ft.pytorch_weight_path is not None


def test_partition_datasets_for_rank_splits_41_builders_across_16_ranks() -> None:
    from openpi.cotrain.rlds_dataset import CotrainRLDSDataset
    from openpi.cotrain.rlds_dataset import partition_datasets_for_rank

    datasets = tuple(
        CotrainRLDSDataset(name=f"ds{i}", dataset_id=f"ds{i}", version="1.0.0", weight=1.0) for i in range(41)
    )
    counts: dict[int, int] = {}
    owned: list[str] = []
    for rank in range(16):
        subset = partition_datasets_for_rank(datasets, process_index=rank, process_count=16)
        counts[rank] = len(subset)
        owned.extend(ds.uid for ds in subset)
        assert sum(ds.weight for ds in subset) == pytest.approx(1.0)
    assert sum(counts.values()) == 41
    assert counts[0] == 3
    assert counts[8] == 3
    assert counts[9] == 2
    assert counts[15] == 2
    assert owned == [f"ds{i}" for i in range(41)]


def test_fk_eef_plus_piper_ego_matches_norm_assets_mixture() -> None:
    assets_root = pathlib.Path(__file__).resolve().parents[2] / "assets" / "cotrain_fk_eef_plus_piper_ego"
    asset_ids = {path.name for path in assets_root.iterdir() if path.is_dir()}
    fk_ids = {dataset.uid for dataset in config._FK_EEF_PLUS_PIPER_EGO_DATA.datasets}

    # EgoVerse_rl2 uids are registered in the mixture; assets are filled after norm recompute.
    assert asset_ids <= fk_ids
    assert len(fk_ids) == 24
    assert {"egoverse_rl2_eva", "egoverse_rl2_human"} <= fk_ids
    assert "agibot" not in fk_ids
    assert "egoverse_scale" not in fk_ids
    assert sum(dataset.weight for dataset in config._FK_EEF_PLUS_PIPER_EGO_DATA.datasets) == pytest.approx(1.0)
    pi05 = config.get_config("cotrain_fk_eef_plus_piper_ego")
    assert pi05.assets_name == "cotrain_fk_eef_plus_piper_ego"
    assert pi05.data is config._FK_EEF_PLUS_PIPER_EGO_DATA
