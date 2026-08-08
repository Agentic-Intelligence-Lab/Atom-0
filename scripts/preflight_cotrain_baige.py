#!/usr/bin/env python
"""Fail-fast validation for the Baige co-training configurations."""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from openpi.cotrain import action_space
from openpi.cotrain import config
from openpi.shared import normalize

FIX_EXCLUDED_DATASET_IDS = {
    "robocoin_leju_robot_s54_a54",
    "robocoin_agilex_decoupled_magic_s14_a14_fps50",
    "robocoin_agilex_decoupled_magic_s26_a26",
}

FASTWAM_CONFIGS = {
    "fastwam_cotrain_real_robot_ego_fix",
    "fastwam_cotrain_real_robot_ego_fix_debug",
    "wam-cross-robot",
    "wam-cross-piper",
    "wam-cross-piper-ft",
    "wam-cross-piper-overfit",
    "wam-cross-piper-overfit-rndnoise",
    "wam-cross-piper-overfit-rndall",
    "wam-cross-robot-ego",
}

HPT_CONFIGS = {
    "hpt_cotrain_real_only",
    "hpt_cotrain_real_robot_ego_fix",
    "hpt_cotrain_smoke",
}

EXPECTED_DATASET_COUNTS = {
    "cotrain_real_only": 2,
    "cotrain_real_robot": 37,
    "cotrain_real_robot_fix": 34,
    "cotrain_real_robot_ego_fix": 43,
    "cotrain_full_all_full_norm": 41,
    "fastwam_cotrain_real_robot_ego_fix": 43,
    "fastwam_cotrain_real_robot_ego_fix_debug": 1,
    "wam-cross-robot": 15,
    "wam-cross-piper": 2,
    "wam-cross-piper-ft": 2,
    "wam-cross-piper-overfit": 2,
    "wam-cross-piper-overfit-rndnoise": 2,
    "wam-cross-piper-overfit-rndall": 2,
    "wam-cross-robot-ego": 20,
    "hpt_cotrain_real_only": 2,
    "hpt_cotrain_real_robot_ego_fix": 43,
    "hpt_cotrain_smoke": 1,
}


def _assets_subdir(cfg: config.CotrainTrainConfig, assets_base: Path) -> Path:
    return assets_base / (cfg.assets_name or cfg.name)


def validate(config_name: str, assets_base: Path, params_path: Path | None) -> None:
    cfg = config.get_config(config_name)
    datasets = cfg.data.datasets
    ids = [dataset.uid for dataset in datasets]
    expected_count = EXPECTED_DATASET_COUNTS[config_name]
    assert len(ids) == expected_count, (config_name, len(ids), expected_count)

    if config_name in {
        "cotrain_real_only",
        "cotrain_real_robot",
        "cotrain_real_robot_fix",
        "cotrain_real_robot_ego_fix",
    } | FASTWAM_CONFIGS | HPT_CONFIGS:
        assert "piper30" in ids
    if config_name not in {
        "cotrain_piper30_legacy32_aliyun_replay",
        "fastwam_cotrain_real_robot_ego_fix_debug",
        "hpt_cotrain_smoke",
    }:
        assert "piper2" in ids

    if config_name == "cotrain_full_all_full_norm":
        assert sum(dataset_id.startswith("egoverse_") for dataset_id in ids) == 7
    elif config_name == "cotrain_real_robot_ego_fix":
        assert sum(dataset_id.startswith("egoverse_") for dataset_id in ids) == 6
        assert "egoverse_scale" not in ids
        assert {"egoverse_aria", "egoverse_eva", "egoverse_human", "egoverse_mecka"} <= set(ids)
        assert {"egoverse_rl2_eva", "egoverse_rl2_human"} <= set(ids)
    elif config_name == "fastwam_cotrain_real_robot_ego_fix":
        assert sum(dataset_id.startswith("egoverse_") for dataset_id in ids) == 6
        assert "egoverse_scale" not in ids
        assert cfg.assets_name == "cotrain_real_robot_ego_fix"
    elif config_name == "wam-cross-robot":
        assert not any(dataset_id.startswith("egoverse_") for dataset_id in ids)
        assert cfg.assets_name == "cotrain_real_robot_ego_fix"
        assert cfg.model.concat_multi_camera == "robot_wrist"
        assert cfg.model.image_resolution == (288, 256)
        assert set(ids) == config._WAM_CROSS_ROBOT_DATASET_IDS
    elif config_name == "wam-cross-piper":
        assert set(ids) == {"piper2", "piper30"}
        assert cfg.rlds_partition_builders_by_rank is False
        assert cfg.eval_interval == 1_000
        assert cfg.run_action_mse is True
        assert cfg.val_max_datasets is None
        assert cfg.model.concat_multi_camera == "robot_wrist"
        assert cfg.model.image_resolution == (288, 256)
        assert cfg.assets_name == "cotrain_real_robot_ego_fix"
    elif config_name == "wam-cross-piper-ft":
        assert set(ids) == {"piper2", "piper30"}
        assert cfg.rlds_partition_builders_by_rank is False
        assert cfg.eval_interval == 1_000
        assert cfg.run_action_mse is True
        assert cfg.val_max_datasets is None
        assert cfg.pytorch_weight_path is not None
        assert abs(cfg.lr_schedule.peak_lr - 1.0e-5) < 1e-12
    elif config_name == "wam-cross-piper-overfit":
        assert set(ids) == {"piper2", "piper30"}
        assert cfg.overfit_fixed_batch is True
        assert cfg.overfit_fixed_noise is True
        assert cfg.fixed_video_sigma == 0.5
        assert cfg.fixed_action_sigma == 0.5
        assert cfg.batch_size == 8
        assert cfg.rlds_partition_builders_by_rank is False
        assert cfg.assets_name == "cotrain_real_robot_ego_fix"
    elif config_name == "wam-cross-piper-overfit-rndnoise":
        assert set(ids) == {"piper2", "piper30"}
        assert cfg.overfit_fixed_batch is True
        assert cfg.overfit_fixed_noise is False
        assert cfg.fixed_video_sigma == 0.5
        assert cfg.fixed_action_sigma == 0.5
        assert cfg.batch_size == 112
        assert cfg.num_train_steps == 3_000
        assert cfg.rlds_partition_builders_by_rank is False
        assert cfg.assets_name == "cotrain_real_robot_ego_fix"
    elif config_name == "wam-cross-piper-overfit-rndall":
        assert set(ids) == {"piper2", "piper30"}
        assert cfg.overfit_fixed_batch is True
        assert cfg.overfit_fixed_noise is False
        assert cfg.fixed_video_sigma is None
        assert cfg.fixed_action_sigma is None
        assert cfg.batch_size == 112
        assert cfg.num_train_steps == 3_000
        assert cfg.rlds_partition_builders_by_rank is False
        assert cfg.assets_name == "cotrain_real_robot_ego_fix"
    elif config_name == "wam-cross-robot-ego":
        ego_ids = {
            "egoverse_aria",
            "egoverse_eva",
            "egoverse_human",
            "egoverse_rl2_eva",
            "egoverse_rl2_human",
        }
        assert ego_ids <= set(ids)
        assert "egoverse_mecka" not in ids
        assert cfg.assets_name == "cotrain_real_robot_ego_fix"
        assert cfg.model.concat_multi_camera == "robot_wrist"
        assert cfg.model.image_resolution == (288, 256)
        assert set(ids) == config._WAM_CROSS_ROBOT_EGO_DATASET_IDS
        assert cfg.model.loss["lambda_ego_video"] > 0.0
        assert cfg.rlds_partition_builders_by_rank is True
    elif config_name == "fastwam_cotrain_real_robot_ego_fix_debug":
        assert ids == ["piper30"]
    elif config_name == "hpt_cotrain_real_only":
        assert set(ids) == {"piper30", "piper2"}
        assert cfg.assets_name == "cotrain_real_only"
        # Default: random-init full train. True FT = TRAIN_MODE=finetune + checkpoint.
        assert cfg.model.train_mode == "pretrain"
        assert cfg.rlds_partition_builders_by_rank is False
        assert cfg.eval_interval > 0
        assert cfg.wandb_enabled is True
    elif config_name == "hpt_cotrain_real_robot_ego_fix":
        assert sum(dataset_id.startswith("egoverse_") for dataset_id in ids) == 6
        assert "egoverse_scale" not in ids
        assert cfg.assets_name == "cotrain_real_robot_ego_fix"
        assert cfg.model.train_mode == "pretrain"
        assert cfg.rlds_partition_builders_by_rank is False
        assert cfg.wandb_enabled is True
    elif config_name == "hpt_cotrain_smoke":
        assert ids == ["piper30"]
    elif config_name not in FASTWAM_CONFIGS | HPT_CONFIGS:
        assert not any(dataset_id.startswith("egoverse_") for dataset_id in ids)

    if config_name in {"cotrain_real_robot_fix", "cotrain_full_all_full_norm"}:
        assert set(ids).isdisjoint(FIX_EXCLUDED_DATASET_IDS)

    if params_path is not None:
        for marker in ("_CHECKPOINT_METADATA", "manifest.ocdbt"):
            assert (params_path / marker).is_file(), params_path / marker

    total_frames = 0
    degenerate = []
    for dataset in datasets:
        builder_dir = Path(dataset.builder_dir)
        assert (builder_dir / "dataset_info.json").is_file(), builder_dir
        directory = _assets_subdir(cfg, assets_base) / dataset.uid
        for filename in ("norm_stats.json", "norm_stats_meta.json", "unified_action_space.json"):
            assert (directory / filename).is_file(), directory / filename

        spec = action_space.UNIFIED_ACTION_SPECS[dataset.uid]
        action_space.validate_metadata(directory, spec)
        stats = normalize.load(directory)
        meta = json.loads((directory / "norm_stats_meta.json").read_text())
        assert Path(meta["builder_dir"]) == builder_dir, (dataset.uid, meta["builder_dir"], builder_dir)
        assert meta["num_frames"] > 0, dataset.uid
        total_frames += int(meta["num_frames"])

        state_mask = np.zeros(action_space.UNIFIED_ACTION_DIM, dtype=bool)
        state_mask[list(spec.state_target_slots)] = True
        for key, active in (("state", state_mask), ("actions", np.asarray(spec.action_mask, dtype=bool))):
            value = stats[key]
            arrays = {field: np.asarray(getattr(value, field)) for field in ("mean", "std", "q01", "q99")}
            assert all(array.shape == (action_space.UNIFIED_ACTION_DIM,) for array in arrays.values())
            assert all(np.isfinite(array).all() for array in arrays.values())
            inactive = ~active
            assert np.allclose(arrays["mean"][inactive], 0)
            assert np.allclose(arrays["std"][inactive], 1)
            assert np.allclose(arrays["q01"][inactive], -1)
            assert np.allclose(arrays["q99"][inactive], 1)
            bad = np.flatnonzero(active & (arrays["q99"] <= arrays["q01"]))
            if bad.size:
                degenerate.append(f"{dataset.uid}:{key}:{bad.tolist()}")

    print(f"PASS {config_name}: datasets={len(ids)}, source_frames={total_frames:,}")
    for item in degenerate:
        print(f"WARN degenerate active quantile (smoke test must remain finite): {item}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        choices=tuple(EXPECTED_DATASET_COUNTS),
    )
    parser.add_argument("--assets-base", type=Path, default=Path("assets"))
    parser.add_argument(
        "--params-path",
        type=Path,
        default=None,
        help="Required for pi05 cotrain configs; optional for FastWAM (NoOpWeightLoader).",
    )
    args = parser.parse_args()
    params_path = args.params_path
    if params_path is None and "PARAMS_PATH" in os.environ:
        params_path = Path(os.environ["PARAMS_PATH"])
    if params_path is None and args.config not in FASTWAM_CONFIGS | HPT_CONFIGS:
        parser.error("Set --params-path or export PARAMS_PATH for pi05 cotrain configs.")
    validate(
        args.config,
        args.assets_base.resolve(),
        params_path.resolve() if params_path is not None else None,
    )


if __name__ == "__main__":
    main()
