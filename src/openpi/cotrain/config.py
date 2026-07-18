"""Config for multi-dataset RLDS co-training with train/val splits.

Subclasses the frozen `openpi.training.config.TrainConfig` to add validation-eval
knobs, and provides a `CotrainDataConfig` factory mirroring `RLDSDroidDataConfig` but
backed by `CotrainRLDSDataset` (per-dataset train/val splits). Nothing in openpi is
modified; this module only imports from it.
"""

import dataclasses
import logging
import os
import pathlib
from typing import Literal

from typing_extensions import override
import tyro

from openpi.cotrain import action_space as cotrain_action_space
from openpi.cotrain.rlds_dataset import CotrainRLDSDataset
import openpi.cotrain.transforms as cotrain_transforms
import openpi.cotrain.weight_loaders as cotrain_weight_loaders
import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.optimizer as _optimizer
import openpi.transforms as _transforms

logger = logging.getLogger(__name__)


def _resolve_unified_datasets(datasets, model_config: _model.BaseModelConfig):
    if model_config.action_dim != cotrain_action_space.UNIFIED_ACTION_DIM:
        raise ValueError(
            f"All co-training configs require action_dim={cotrain_action_space.UNIFIED_ACTION_DIM}, "
            f"got {model_config.action_dim}."
        )

    resolved = []
    for ds in datasets:
        try:
            spec = cotrain_action_space.UNIFIED_ACTION_SPECS[ds.uid]
        except KeyError as exc:
            raise ValueError(f"Dataset '{ds.uid}' has no registered unified 80D action mapping.") from exc
        if ds.unified_action_spec is not None and ds.unified_action_spec != spec:
            raise ValueError(f"Dataset '{ds.uid}' overrides its registered unified 80D action mapping.")
        resolved.append(dataclasses.replace(ds, unified_action_spec=spec))
    return tuple(resolved)


def load_per_dataset_norm_stats(assets_dirs: pathlib.Path, datasets) -> dict:
    """Load per-dataset norm stats from `<assets_dirs>/<dataset_name>` (skip if missing).

    Returns {dataset_name: {"state": NormStats, "actions": NormStats}} for the DispatchNormalize.
    """
    stats: dict = {}
    for ds in datasets:
        try:
            d = str(pathlib.Path(assets_dirs) / ds.uid)
            resolved = pathlib.Path(_download.maybe_download(d))
            loaded = _normalize.load(resolved)
            if ds.unified_action_spec is not None:
                cotrain_action_space.validate_metadata(resolved, ds.unified_action_spec)
                for key in ("state", "actions"):
                    if key not in loaded or len(loaded[key].mean) != cotrain_action_space.UNIFIED_ACTION_DIM:
                        raise ValueError(
                            f"Unified norm stats for '{ds.uid}' key '{key}' must be "
                            f"{cotrain_action_space.UNIFIED_ACTION_DIM}D."
                        )
            stats[ds.uid] = loaded
            logger.info(f"Loaded per-dataset norm stats for '{ds.uid}' from {d}")
        except FileNotFoundError:
            logger.warning(f"Norm stats for dataset '{ds.uid}' not found under {assets_dirs}; skipping (no norm).")
    return stats


@dataclasses.dataclass(frozen=True)
class CotrainDataConfig(_config.DataConfigFactory):
    """Multi-dataset RLDS data config with per-dataset train/val splits.

    Currently assumes the DROID schema for repack/data transforms (the framework
    milestone starts from DROID-format data). Heterogeneous schemas plug in via the
    restructure registry in `rlds_dataset.py` plus per-dataset repack transforms here.
    """

    # RLDS path does not use a LeRobot repo_id; give it a default so it isn't a required CLI
    # arg. Per-dataset norm stats live under <assets_dirs>/<dataset_name>, not under repo_id.
    repo_id: str = "cotrain"
    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    datasets: tuple[CotrainRLDSDataset, ...] = ()

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> _config.DataConfig:
        assert self.rlds_data_dir is not None, "Need to set rlds_data_dir for the co-training RLDS loader."
        assert len(self.datasets) > 0, "Need at least one dataset in `datasets`."
        datasets = _resolve_unified_datasets(self.datasets, model_config)
        if getattr(model_config, "ki_enabled", False):
            raise NotImplementedError("KI FAST-token supervision does not yet support per-dimension action masks.")

        base = self.create_base_config(assets_dirs, model_config)

        # Per-dataset absolute->delta action conversion (e.g. RoboMIND absolute joint).
        delta_masks = {}
        for ds in datasets:
            delta_masks[ds.uid] = ds.unified_action_spec.delta_mask
        dispatch_delta = cotrain_transforms.DispatchDeltaActions(masks_by_dataset=delta_masks)

        # Per-dataset normalization (dispatched at runtime by dataset_id). Quantile norm for
        # pi05 (use_quantile_norm is True for non-PI0 models in create_base_config).
        per_dataset_stats = load_per_dataset_norm_stats(assets_dirs, datasets)
        dispatch_norm = cotrain_transforms.DispatchNormalize(
            norm_stats_by_dataset=per_dataset_stats,
            use_quantiles=base.use_quantile_norm,
        )

        # Generic inputs (uniform schema across datasets) -> per-dataset delta -> per-dataset
        # normalization. No per-dataset repack needed (StandardizedInputs reads the nested
        # standardized keys directly). Delta MUST precede normalization (stats are on deltas).
        data_transforms = _transforms.Group(
            inputs=[
                cotrain_transforms.StandardizedInputs(model_type=model_config.model_type),
                dispatch_delta,
                dispatch_norm,
            ],
        )

        model_transforms = _config.ModelTransformFactory()(model_config)

        return dataclasses.replace(
            base,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
            datasets=datasets,
        )


@dataclasses.dataclass(frozen=True)
class CotrainTrainConfig(_config.TrainConfig):
    """TrainConfig + validation-eval knobs."""

    # How often (in steps) to run validation.
    eval_interval: int = 1000
    # Number of val batches per dataset for the (cheap) flow-loss pass.
    num_val_batches: int = 20
    # Whether to also run the (expensive) action-MSE sampling pass.
    run_action_mse: bool = True
    # Number of val batches per dataset for the action-MSE pass.
    num_action_mse_batches: int = 5
    # Denoising steps used by sample_actions during action-MSE eval.
    action_mse_num_denoise_steps: int = 10
    # Action MSE uses the per-sample binary mask carried by the RLDS pipeline.
    # Flow-loss estimator(s): "fixed_seed", "multi_sample", or "both".
    val_flow_loss_mode: Literal["fixed_seed", "multi_sample", "both"] = "both"
    # K for the multi-sample flow-loss estimator.
    val_flow_loss_num_samples: int = 8
    # Fixed rng seed for all validation metrics (deterministic, comparable curves).
    val_seed: int = 0
    # Evaluate on EMA params instead of live params.
    eval_on_ema: bool = False
    # Log predicted-vs-GT action-chunk trajectory plots to wandb at each eval.
    viz_action_traj: bool = True
    # Number of samples per dataset to draw in the trajectory plot.
    viz_num_samples: int = 1
    # Shuffle buffer size for the TRAIN loader. Images are buffered ENCODED, but this still
    # costs ~buffer_size * (encoded image bytes); lower it if you hit OOM. (val uses ~1.)
    shuffle_buffer_size: int = 50_000
    # tf.data parallelism for RLDS reading and mapping. -1 keeps TensorFlow AUTOTUNE.
    data_num_parallel_reads: int = -1
    data_num_parallel_calls: int = -1


# ---------------------------------------------------------------------------
# Config registry (separate from openpi's _CONFIGS; selected via this module's cli()).
# ---------------------------------------------------------------------------
# Every config in this registry uses the fixed 80D state/action layout. pi05_base has a
# 32D projection/head, so checkpoint-start configs use the shape-safe loader and randomly
# initialize only parameters whose shapes changed.

_RLDS_ROOT = os.environ.get("RLDS_DATA_DIR", "/mnt/bos/bo23lu")

_PIPER30_ROOT = (
    f"{_RLDS_ROOT}/realworld_piper/"
    "piper_s14_a14_fps30_c4_ee_pose_cam_front_cam_high_cam_left_wrist_cam_right_wrist"
)
_PIPER30_BUILDER_DIR = f"{_PIPER30_ROOT}/realworld_piper_infidata/1.0.0"
_PIPER30_TRAIN_EPISODES = 5_307

# Second in-house Piper RLDS drop. Its on-host builder was audited at
# /mnt/bos/bo23lu/realworld_piper_2/realworld_piper_infidata/1.0.0. Keep an override for
# hosts whose RLDS root is mounted elsewhere.
_PIPER2_ROOT = f"{_RLDS_ROOT}/realworld_piper_2"
_PIPER2_BUILDER_DIR = os.environ.get(
    "REALWORLD_PIPER_2_BUILDER_DIR",
    f"{_PIPER2_ROOT}/realworld_piper_infidata/1.0.0",
)
_PIPER2_TRAIN_EPISODES = int(os.environ.get("REALWORLD_PIPER_2_TRAIN_EPISODES", "902"))

_DROID_ROOT = f"{_RLDS_ROOT}/DROID"
_DROID_BUILDER_DIR = f"{_DROID_ROOT}/droid_infidata/1.1.0"
_DROID_TRAIN_EPISODES = 64_124

_EGOVERSE_FULL_ROOT = f"{_RLDS_ROOT}/EgoVerse_full"
_EGOVERSE_FULL_TRAIN_EPISODES = 910 + 2_813 + 770 + 39_530 + 16_223

_ROBOCOIN_ROOT = f"{_RLDS_ROOT}/RoboCOIN"
# RoboCOIN tuple format:
#   dataset_id, repo dir, train episodes, effective action_dim, delta mask dims,
#   optional state_indices override.
#
# The raw per-robot schemas expose `state_action_schema_json`. For most RoboCOIN builders,
# action names match the state prefix, so we crop state/action to the named action width and
# apply delta only to arm joint-position dims. Yinhe has body/head joints before arm joints in
# state, so it needs an explicit state-index reorder.
_ROBOCOIN_REPOS = (
    (
        "robocoin_agilex_cobot_magic_s26_a26",
        "Agilex_Cobot_Magic_s26_a26_fps30_cam_high_cam_left_wrist_cam_right_wrist__episodes_8284",
        7_870,
        26,
        (6, -7, 6, -7),
        None,
    ),
    (
        "robocoin_airbot_mmk2_s36_a36",
        "Airbot_MMK2_s36_a36_fps30_cam_high_cam_left_wrist_cam_right_wrist__episodes_10532",
        10_005,
        36,
        (12, -24),
        None,
    ),
    (
        "robocoin_galaxea_r1_lite_upper_s14_a14",
        "Galaxea_R1_Lite_s14_a14_fps30_cam_high_cam_left_wrist_cam_right_wrist__episodes_3650",
        1_337,
        14,
        (12, -2),
        None,
    ),
    (
        "robocoin_realman_rmc_aida_l_s28_a28",
        "Realman_RMC-AIDA-L_s28_a28_fps30_cam_high_cam_left_wrist_cam_right_wrist__episodes_693",
        658,
        28,
        (7, -7, 7, -7),
        None,
    ),
    (
        "robocoin_unitree_g1_dex3_s28_a28",
        "Unitree_G1_Dex3_phecda_s28_a28_fps30_cam_high_cam_left_wrist_cam_right_wrist__episodes_1411",
        1_340,
        28,
        (14, -14),
        None,
    ),
    (
        "robocoin_agilex_decoupled_magic_s14_a14_fps30",
        "agilex_cobot_decoupled_magic_s14_a14_fps30_cam_high_cam_left_wrist_cam_right_wrist__episodes_7778",
        7_389,
        14,
        (6, -1, 6, -1),
        None,
    ),
    (
        "robocoin_agilex_decoupled_magic_s14_a14_fps50",
        "agilex_cobot_decoupled_magic_s14_a14_fps50_cam_high_cam_left_wrist_cam_right_wrist__episodes_3397",
        3_227,
        14,
        (6, -1, 6, -1),
        None,
    ),
    (
        "robocoin_agilex_decoupled_magic_s26_a26",
        "agilex_cobot_decoupled_magic_s26_a26_fps30_cam_high_cam_left_wrist_cam_right_wrist__episodes_23712",
        18_406,
        26,
        (6, -7, 6, -7),
        None,
    ),
    (
        "robocoin_aloha_s26_a26",
        "aloha_s26_a26_fps30_cam_high_cam_left_wrist_cam_right_wrist__episodes_4879",
        4_634,
        26,
        (6, -7, 6, -7),
        None,
    ),
    (
        "robocoin_alpha_bot_2_s28_a28",
        "alpha_bot_2_s28_a28_fps30_cam_high_cam_left_wrist_cam_right_wrist__episodes_857",
        814,
        28,
        (7, -6, 7, -8),
        None,
    ),
    (
        "robocoin_discover_aitbot_mmk2_s36_a36",
        "discover_robotics_aitbot_mmk2_s36_a36_fps30_cam_high_cam_left_wrist_cam_right_wrist__episodes_5747",
        5_460,
        36,
        (12, -24),
        None,
    ),
    (
        "robocoin_galaxea_r1_lite_s14_a14",
        "galaxea_r1_lite_s14_a14_fps30_cam_high_cam_left_wrist_cam_right_wrist__episodes_5167",
        4_886,
        14,
        (6, -1, 6, -1),
        None,
    ),
    (
        "robocoin_galaxea_r1_lite_s16_a18",
        "galaxea_r1_lite_s16_a18_fps30_cam_high_cam_left_wrist_cam_right_wrist__episodes_970",
        922,
        # The raw action tensor is 18-wide, but schema names only identify 16 dims
        # (14 arm joints + 2 grippers). Drop the unnamed tail dims.
        16,
        (14, -2),
        None,
    ),
    (
        "robocoin_leju_robot_s118_a54",
        "leju_robot_s118_a54_fps30_cam_high_cam_left_wrist_cam_right_wrist__episodes_17897",
        17_002,
        54,
        (14, -40),
        None,
    ),
    (
        "robocoin_leju_robot_s54_a54",
        "leju_robot_s54_a54_fps30_cam_high_cam_left_wrist_cam_right_wrist__episodes_394",
        374,
        54,
        (14, -40),
        None,
    ),
    (
        "robocoin_realman_rmc_aidal_s28_a28",
        "realman_rmc_aidal_s28_a28_fps30_cam_high_cam_left_wrist_cam_right_wrist__episodes_18412",
        17_481,
        28,
        (7, -7, 7, -7),
        None,
    ),
    (
        "robocoin_ruantong_a2d_s17_a17",
        "ruantong_a2d_s17_a17_fps30_cam_high_cam_left_wrist_cam_right_wrist__episodes_1719",
        1_633,
        17,
        (-1, 14, -2),
        None,
    ),
    (
        "robocoin_ruantong_a2d_s41_a34",
        "ruantong_a2d_s41_a34_fps30_cam_high_cam_left_wrist_cam_right_wrist__episodes_6459",
        6_136,
        34,
        (14, -20),
        None,
    ),
    (
        "robocoin_unitree_g1_s28_a28_high",
        "unitree_g1_s28_a28_fps30_cam_high__episodes_227",
        216,
        28,
        (14, -14),
        None,
    ),
    (
        "robocoin_unitree_g1_s28_a28",
        "unitree_g1_s28_a28_fps30_cam_high_cam_left_wrist_cam_right_wrist__episodes_931",
        884,
        28,
        (14, -14),
        None,
    ),
    (
        "robocoin_unknown_s30_a30_high",
        "unknown_s30_a30_fps30_cam_high__episodes_891",
        846,
        # The raw action tensor is 30-wide, but schema names only identify 28 dims.
        28,
        (14, -14),
        None,
    ),
    (
        "robocoin_yinhe_s49_a16",
        "yinhe_s49_a16_fps30_cam_high_cam_left_wrist_cam_right_wrist__episodes_5452",
        5_179,
        16,
        (7, -1, 7, -1),
        tuple(range(5, 21)),
    ),
)
_ROBOCOIN_TRAIN_EPISODES = sum(train_episodes for _, _, train_episodes, _, _, _ in _ROBOCOIN_REPOS)

_AGIBOT_ROOT = (
    f"{_RLDS_ROOT}/AgiBot/"
    "agibot_world_robot_agibot_world_beta_mobile_dual_arm_joint_absolute_position_real_s20_a20_fps30_"
    "cam_high_cam_left_wrist_cam_right_wrist__episodes_22986"
)
_AGIBOT_BUILDER_DIR = f"{_AGIBOT_ROOT}/agibot_infidata/1.0.0"
_AGIBOT_TRAIN_EPISODES = 21_837

_AGIBOT_DATA = CotrainDataConfig(
    rlds_data_dir=_AGIBOT_ROOT,
    datasets=(
        CotrainRLDSDataset(
            name="agibot_infidata",
            dataset_id="agibot",
            version="1.0.0",
            builder_dir=_AGIBOT_BUILDER_DIR,
            weight=1.0,
            train_split="train",
            val_splits={"seen": "seen_test", "unseen": "unseen_test"},
            restructure_name="agibot",
            action_dim=20,
            # AgiBot action/state layout: joint14 + effector2 + head2 + waist2.
            # Keep non-joint command dims absolute; train joint targets as deltas.
            delta_action_mask_dims=(14, -2, -2, -2),
        ),
    ),
)

_DROID_DATA = CotrainDataConfig(
    rlds_data_dir=_DROID_ROOT,
    datasets=(
        CotrainRLDSDataset(
            name="droid_infidata",
            dataset_id="droid",
            version="1.1.0",
            builder_dir=_DROID_BUILDER_DIR,
            weight=1.0,
            train_split="train",
            val_splits={"seen": "seen_test", "unseen": "unseen_test"},
            restructure_name="three_cam_task",
            action_dim=8,
            # DROID layout: 7 Franka joint positions + gripper. Match openpi's
            # full-DROID recipe: train joint deltas and keep the gripper absolute.
            delta_action_mask_dims=(7, -1),
        ),
    ),
)

_EGOVERSE_FULL_DATA = CotrainDataConfig(
    rlds_data_dir=_EGOVERSE_FULL_ROOT,
    datasets=(
        CotrainRLDSDataset(
            name="ego_verse_infidata",
            dataset_id="egoverse_aria",
            version="1.0.0",
            builder_dir=f"{_EGOVERSE_FULL_ROOT}/aria_bimanual_front_1/ego_verse_infidata/1.0.0",
            weight=910 / _EGOVERSE_FULL_TRAIN_EPISODES,
            train_split="train",
            val_splits={"seen": "seen_test", "unseen": "unseen_test"},
            restructure_name="egoverse_full",
            action_dim=12,
            delta_action_mask_dims=None,
        ),
        CotrainRLDSDataset(
            name="ego_verse_infidata",
            dataset_id="egoverse_eva",
            version="1.0.0",
            builder_dir=f"{_EGOVERSE_FULL_ROOT}/eva_bimanual_front_1_left_wrist_right_wrist/ego_verse_infidata/1.0.0",
            weight=2_813 / _EGOVERSE_FULL_TRAIN_EPISODES,
            train_split="train",
            val_splits={"seen": "seen_test", "unseen": "unseen_test"},
            restructure_name="egoverse_full",
            action_dim=12,
            delta_action_mask_dims=None,
        ),
        CotrainRLDSDataset(
            name="ego_verse_infidata",
            dataset_id="egoverse_human",
            version="1.0.0",
            builder_dir=f"{_EGOVERSE_FULL_ROOT}/human_bimanual_front_1/ego_verse_infidata/1.0.0",
            weight=770 / _EGOVERSE_FULL_TRAIN_EPISODES,
            train_split="train",
            val_splits={"seen": "seen_test", "unseen": "unseen_test"},
            restructure_name="egoverse_full",
            action_dim=12,
            delta_action_mask_dims=None,
        ),
        CotrainRLDSDataset(
            name="ego_verse_infidata",
            dataset_id="egoverse_mecka",
            version="1.0.0",
            builder_dir=f"{_EGOVERSE_FULL_ROOT}/mecka_bimanual_front_1/ego_verse_infidata/1.0.0",
            weight=39_530 / _EGOVERSE_FULL_TRAIN_EPISODES,
            train_split="train",
            val_splits={"seen": "seen_test", "unseen": "unseen_test"},
            restructure_name="egoverse_full",
            action_dim=12,
            delta_action_mask_dims=None,
        ),
        CotrainRLDSDataset(
            name="ego_verse_infidata",
            dataset_id="egoverse_scale",
            version="1.0.0",
            builder_dir=f"{_EGOVERSE_FULL_ROOT}/scale_front_1/ego_verse_infidata/1.0.0",
            weight=16_223 / _EGOVERSE_FULL_TRAIN_EPISODES,
            train_split="train",
            val_splits={"seen": "seen_test", "unseen": "unseen_test"},
            restructure_name="egoverse_full",
            action_dim=12,
            delta_action_mask_dims=None,
        ),
    ),
)


def _make_robocoin_dataset(
    dataset_id: str,
    repo: str,
    train_episodes: int,
    action_dim: int,
    delta_action_mask_dims: tuple[int, ...],
    state_indices: tuple[int, ...] | None,
) -> CotrainRLDSDataset:
    # Most RoboCOIN schemas put action-aligned proprio first in `observation/state`. A few
    # builders have leading body/head state dims or unnamed action tail dims; those are handled
    # by explicit per-builder index selections in `_ROBOCOIN_REPOS`.
    if state_indices is None:
        state_indices = tuple(range(action_dim))
    action_indices = tuple(range(action_dim))
    return CotrainRLDSDataset(
        name="robocoin_infidata",
        dataset_id=dataset_id,
        version="1.0.0",
        builder_dir=f"{_ROBOCOIN_ROOT}/{repo}/robocoin_infidata/1.0.0",
        weight=train_episodes / _ROBOCOIN_TRAIN_EPISODES,
        train_split="train",
        val_splits={"seen": "seen_test", "unseen": "unseen_test"},
        restructure_name="robocoin",
        action_dim=action_dim,
        state_indices=state_indices,
        action_indices=action_indices,
        delta_action_mask_dims=delta_action_mask_dims,
    )


_ROBOCOIN_DATA = CotrainDataConfig(
    rlds_data_dir=_ROBOCOIN_ROOT,
    datasets=tuple(_make_robocoin_dataset(*repo_cfg) for repo_cfg in _ROBOCOIN_REPOS),
)

_ROBOMIND_FULL_ROOT = f"{_RLDS_ROOT}/RoboMIND_full"
# RoboMIND_full tuple format:
#   dataset_id, repo dir, episodes, action_dim, delta mask dims, camera keys
# where camera keys are (base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb).
_ROBOMIND_FULL_REPOS = (
    (
        "robomind_agilex_cobot_magic_s14_a14",
        "agilex_cobot_magic_agilex_dual_arm_h5_agilex_3rgb_real_s14_a14_fps30_cam_high_cam_left_wrist_cam_right_wrist__episodes_10374",
        10_374,
        14,
        (6, -1, 6, -1),
        ("cam_high", "cam_left_wrist", "cam_right_wrist"),
    ),
    (
        "robomind_franka_fr3_dual_s16_a16",
        "franka_fr3_dual_master_puppet_joint_position_h5_franka_fr3_dual_real_s16_a16_fps30_cam_high_cam_left_cam_right_cam_top__episodes_1774",
        1_774,
        16,
        (7, -1, 7, -1),
        ("cam_high", "cam_left", "cam_right"),
    ),
    (
        "robomind_franka_panda_s8_a8",
        "franka_panda_master_puppet_joint_position_h5_franka_3rgb_real_s8_a8_fps30_cam_left_cam_right_cam_top__episodes_17219",
        17_219,
        8,
        (7, -1),
        ("cam_top", "cam_left", "cam_right"),
    ),
    (
        "robomind_franka_sim_franka_s8_a8",
        "franka_sim_simulation_franka_joint_position_h5_sim_franka_3rgb_sim_s8_a8_fps30_cam_front_external_cam_handeye_cam_left_external_cam_right_external__episodes_14488",
        14_488,
        8,
        (7, -1),
        ("cam_front_external", "cam_handeye", "cam_right_external"),
    ),
    (
        "robomind_franka_sim_simulation_s8_a8",
        "franka_sim_simulation_franka_joint_position_h5_simulation_sim_s8_a8_fps30_cam_front_external_cam_handeye_cam_left_external_cam_right_external__episodes_11422",
        11_422,
        8,
        (7, -1),
        ("cam_front_external", "cam_handeye", "cam_right_external"),
    ),
    (
        "robomind_franka_sim_simulation_no_front_s8_a8",
        "franka_sim_simulation_franka_joint_position_h5_simulation_sim_s8_a8_fps30_cam_handeye_cam_left_external_cam_right_external__episodes_158",
        158,
        8,
        (7, -1),
        ("cam_left_external", "cam_handeye", "cam_right_external"),
    ),
    (
        "robomind_franka_sim_none_s8_a8",
        "franka_sim_simulation_franka_joint_position_none_sim_s8_a8_fps30_cam_front_external_cam_handeye_cam_left_external_cam_right_external__episodes_222",
        222,
        8,
        (7, -1),
        ("cam_front_external", "cam_handeye", "cam_right_external"),
    ),
    (
        "robomind_tienkung_gello_s16_a16",
        "tienkung_humanoid_master_puppet_joint_position_h5_tienkung_gello_1rgb_real_s16_a16_fps30_cam_top__episodes_6626",
        6_626,
        16,
        # arm7 delta + hand-closure absolute, per side.
        (7, -1, 7, -1),
        ("cam_top", None, None),
    ),
    (
        "robomind_tienkung_prod1_gello_s16_a16",
        "tienkung_humanoid_master_puppet_joint_position_h5_tienkung_prod1_gello_1rgb_real_s16_a16_fps30_cam_top__episodes_2959",
        2_959,
        16,
        # arm7 delta + hand-closure absolute, per side.
        (7, -1, 7, -1),
        ("cam_top", None, None),
    ),
    (
        "robomind_tienkung_xsens_s14_a14",
        "tienkung_humanoid_master_puppet_joint_position_h5_tienkung_xsens_1rgb_real_s14_a14_fps30_cam_top__episodes_6126",
        6_126,
        14,
        (14,),
        ("cam_top", None, None),
    ),
    (
        "robomind_tienkung_sim_s38_a38",
        "tienkung_humanoid_tiangong_joint_position_h5_sim_tienkung_1rgb_sim_s38_a38_fps30_cam_chest_cam_head__episodes_3965",
        3_965,
        38,
        # arm7 delta + dex-hand12 absolute, per side.
        (7, -12, 7, -12),
        ("cam_chest", "cam_head", None),
    ),
    (
        "robomind_tienkung_real_s38_a38",
        "tienkung_humanoid_tiangong_joint_position_none_real_s38_a38_fps30_cam_chest_cam_head__episodes_146",
        146,
        38,
        # arm7 delta + dex-hand12 absolute, per side.
        (7, -12, 7, -12),
        ("cam_chest", "cam_head", None),
    ),
    (
        "robomind_ur5e_s7_a7",
        "ur5e_master_puppet_joint_position_h5_ur_1rgb_real_s7_a7_fps30_cam_top__episodes_26380",
        26_380,
        7,
        (6, -1),
        ("cam_top", None, None),
    ),
)
_ROBOMIND_FULL_EPISODES = sum(episodes for _, _, episodes, _, _, _ in _ROBOMIND_FULL_REPOS)


def _make_robomind_full_dataset(
    dataset_id: str,
    repo: str,
    episodes: int,
    action_dim: int,
    delta_action_mask_dims: tuple[int, ...],
    camera_keys: tuple[str | None, str | None, str | None],
) -> CotrainRLDSDataset:
    return CotrainRLDSDataset(
        name="robomind_full_infidata",
        dataset_id=dataset_id,
        version="1.0.0",
        builder_dir=f"{_ROBOMIND_FULL_ROOT}/{repo}/robomind_full_infidata/1.0.0",
        weight=episodes / _ROBOMIND_FULL_EPISODES,
        train_split="train",
        val_splits={"seen": "seen_test", "unseen": "unseen_test"},
        restructure_name="robomind_full",
        action_dim=action_dim,
        camera_keys=camera_keys,
        delta_action_mask_dims=delta_action_mask_dims,
    )


_ROBOMIND_FULL_DATA = CotrainDataConfig(
    rlds_data_dir=_ROBOMIND_FULL_ROOT,
    datasets=tuple(_make_robomind_full_dataset(*repo_cfg) for repo_cfg in _ROBOMIND_FULL_REPOS),
)

_PIPER30_DATA = CotrainDataConfig(
    rlds_data_dir=_PIPER30_ROOT,
    datasets=(
        CotrainRLDSDataset(
            name="realworld_piper_infidata",
            dataset_id="piper30",
            version="1.0.0",
            builder_dir=_PIPER30_BUILDER_DIR,
            weight=1.0,
            train_split="train",
            val_splits={"seen": "seen_test", "unseen": "unseen_test"},
            restructure_name="three_cam_task",
            action_dim=14,
            # Piper stores absolute joint targets: delta the 6 arm joints on each side,
            # keep both gripper dims absolute.
            delta_action_mask_dims=(6, -1, 6, -1),
        ),
    ),
)

_PIPER2_DATA = CotrainDataConfig(
    rlds_data_dir=_PIPER2_ROOT,
    datasets=(
        CotrainRLDSDataset(
            name="realworld_piper_infidata",
            dataset_id="piper2",
            version="1.0.0",
            builder_dir=_PIPER2_BUILDER_DIR,
            weight=1.0,
            train_split="train",
            val_splits={"seen": "seen_test", "unseen": "unseen_test"},
            restructure_name="piper2",
            action_dim=14,
            # Actual metadata and samples agree on this layout:
            #   left_joint_1..6, left_gripper, right_joint_1..6, right_gripper.
            # action[t] is exactly state[t+1]. Arm targets become relative to the current
            # observation while both gripper targets remain absolute.
            delta_action_mask_dims=(6, -1, 6, -1),
        ),
    ),
)


_ALL_TRAIN_EPISODES = (
    _AGIBOT_TRAIN_EPISODES
    + _DROID_TRAIN_EPISODES
    + _EGOVERSE_FULL_TRAIN_EPISODES
    + _PIPER30_TRAIN_EPISODES
    + _PIPER2_TRAIN_EPISODES
    + _ROBOCOIN_TRAIN_EPISODES
    + _ROBOMIND_FULL_EPISODES
)


def _scale_dataset_weights(datasets: tuple[CotrainRLDSDataset, ...], train_episodes: int):
    return tuple(
        dataclasses.replace(ds, weight=ds.weight * train_episodes / _ALL_TRAIN_EPISODES) for ds in datasets
    )


_FULL_ALL_EXCLUDED_DATASET_IDS = {
    "robocoin_unitree_g1_dex3_s28_a28",
    "robomind_tienkung_sim_s38_a38",
}


def _drop_dataset_ids_and_renormalize(
    datasets: tuple[CotrainRLDSDataset, ...], excluded_dataset_ids: set[str] | frozenset[str]
):
    kept = tuple(ds for ds in datasets if ds.uid not in excluded_dataset_ids)
    if not kept:
        raise ValueError("At least one co-training dataset must remain after exclusions.")
    total_weight = sum(ds.weight for ds in kept)
    return tuple(dataclasses.replace(ds, weight=ds.weight / total_weight) for ds in kept)


def _drop_excluded_and_renormalize(datasets: tuple[CotrainRLDSDataset, ...]):
    return _drop_dataset_ids_and_renormalize(datasets, _FULL_ALL_EXCLUDED_DATASET_IDS)


# In-house real-robot mixture. Weights are proportional to train episode counts.
_REAL_ONLY_DATA = CotrainDataConfig(
    rlds_data_dir=_RLDS_ROOT,
    datasets=_drop_excluded_and_renormalize(
        (
            *_scale_dataset_weights(_PIPER30_DATA.datasets, _PIPER30_TRAIN_EPISODES),
            *_scale_dataset_weights(_PIPER2_DATA.datasets, _PIPER2_TRAIN_EPISODES),
        )
    ),
)

# All in-house real data plus public robot datasets. EgoVerse is deliberately absent.
_REAL_ROBOT_DATA = CotrainDataConfig(
    rlds_data_dir=_RLDS_ROOT,
    datasets=_drop_excluded_and_renormalize(
        (
            *_scale_dataset_weights(_PIPER30_DATA.datasets, _PIPER30_TRAIN_EPISODES),
            *_scale_dataset_weights(_PIPER2_DATA.datasets, _PIPER2_TRAIN_EPISODES),
            *_scale_dataset_weights(_AGIBOT_DATA.datasets, _AGIBOT_TRAIN_EPISODES),
            *_scale_dataset_weights(_DROID_DATA.datasets, _DROID_TRAIN_EPISODES),
            *_scale_dataset_weights(_ROBOCOIN_DATA.datasets, _ROBOCOIN_TRAIN_EPISODES),
            *_scale_dataset_weights(_ROBOMIND_FULL_DATA.datasets, _ROBOMIND_FULL_EPISODES),
        )
    ),
)

# Audited production mixture. Keep the original cotrain_real_robot config immutable for
# reproducibility, and exclude datasets whose sparse/corrupt tails make quantile normalization
# unsafe or destroy state conditioning.
_REAL_ROBOT_FIX_EXCLUDED_DATASET_IDS = frozenset(
    {
        "robocoin_leju_robot_s54_a54",
        "robocoin_agilex_decoupled_magic_s14_a14_fps50",
        "robocoin_agilex_decoupled_magic_s26_a26",
    }
)
_REAL_ROBOT_FIX_DATA = dataclasses.replace(
    _REAL_ROBOT_DATA,
    datasets=_drop_dataset_ids_and_renormalize(
        _REAL_ROBOT_DATA.datasets,
        _REAL_ROBOT_FIX_EXCLUDED_DATASET_IDS,
    ),
)

# Production all-data mixture: the audited real+robot mixture plus EgoVerse. This
# includes both in-house Piper datasets and excludes the two globally disabled
# datasets as well as the three datasets rejected by the real-robot audit.
_FULL_ALL_FIX_DATA = CotrainDataConfig(
    rlds_data_dir=_RLDS_ROOT,
    datasets=_drop_dataset_ids_and_renormalize(
        (
            *_scale_dataset_weights(_PIPER30_DATA.datasets, _PIPER30_TRAIN_EPISODES),
            *_scale_dataset_weights(_PIPER2_DATA.datasets, _PIPER2_TRAIN_EPISODES),
            *_scale_dataset_weights(_AGIBOT_DATA.datasets, _AGIBOT_TRAIN_EPISODES),
            *_scale_dataset_weights(_DROID_DATA.datasets, _DROID_TRAIN_EPISODES),
            *_scale_dataset_weights(_EGOVERSE_FULL_DATA.datasets, _EGOVERSE_FULL_TRAIN_EPISODES),
            *_scale_dataset_weights(_ROBOCOIN_DATA.datasets, _ROBOCOIN_TRAIN_EPISODES),
            *_scale_dataset_weights(_ROBOMIND_FULL_DATA.datasets, _ROBOMIND_FULL_EPISODES),
        ),
        _FULL_ALL_EXCLUDED_DATASET_IDS | _REAL_ROBOT_FIX_EXCLUDED_DATASET_IDS,
    ),
)


_UNIFIED_PI05_MODEL = pi0_config.Pi0Config(
    pi05=True,
    action_dim=cotrain_action_space.UNIFIED_ACTION_DIM,
    max_token_len=384,
)
_PI05_BASE_SHAPE_SAFE_LOADER = cotrain_weight_loaders.ShapeSafeCheckpointWeightLoader(
    params_path="gs://openpi-assets/checkpoints/pi05_base/params",
)


_REAL_ONLY_PI05 = CotrainTrainConfig(
    name="cotrain_real_only",
    model=_UNIFIED_PI05_MODEL,
    data=_REAL_ONLY_DATA,
    weight_loader=_PI05_BASE_SHAPE_SAFE_LOADER,
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=10_000,
        peak_lr=1.0e-6,
        decay_steps=3_000_000,
        decay_lr=1.0e-7,
    ),
    optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
    batch_size=32,
    num_train_steps=30_000,
    log_interval=100,
    save_interval=2_000,
    eval_interval=1_000,
    num_val_batches=10,
    num_action_mse_batches=2,
    exp_name=tyro.MISSING,
)

_REAL_ROBOT_PI05 = dataclasses.replace(
    _REAL_ONLY_PI05,
    name="cotrain_real_robot",
    data=_REAL_ROBOT_DATA,
)

_REAL_ROBOT_FIX_PI05 = dataclasses.replace(
    _REAL_ROBOT_PI05,
    name="cotrain_real_robot_fix",
    data=_REAL_ROBOT_FIX_DATA,
)

_FULL_ALL_PI05_FULL_NORM = dataclasses.replace(
    _REAL_ONLY_PI05,
    name="cotrain_full_all_full_norm",
    data=_FULL_ALL_FIX_DATA,
)

_COTRAIN_CONFIGS = [
    _REAL_ONLY_PI05,
    _REAL_ROBOT_PI05,
    _REAL_ROBOT_FIX_PI05,
    _FULL_ALL_PI05_FULL_NORM,
]

if len({c.name for c in _COTRAIN_CONFIGS}) != len(_COTRAIN_CONFIGS):
    raise ValueError("Co-train config names must be unique.")
_COTRAIN_CONFIGS_DICT = {c.name: c for c in _COTRAIN_CONFIGS}


def cli() -> CotrainTrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _COTRAIN_CONFIGS_DICT.items()})


def get_config(config_name: str) -> CotrainTrainConfig:
    if config_name not in _COTRAIN_CONFIGS_DICT:
        raise ValueError(f"Co-train config '{config_name}' not found. Available: {list(_COTRAIN_CONFIGS_DICT)}")
    return _COTRAIN_CONFIGS_DICT[config_name]
