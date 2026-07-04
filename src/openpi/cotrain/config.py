"""Config for multi-dataset RLDS co-training with train/val splits.

Subclasses the frozen `openpi.training.config.TrainConfig` to add validation-eval
knobs, and provides a `CotrainDataConfig` factory mirroring `RLDSDroidDataConfig` but
backed by `CotrainRLDSDataset` (per-dataset train/val splits). Nothing in openpi is
modified; this module only imports from it.
"""

import dataclasses
import logging
import pathlib
from typing import Literal

from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

import openpi.cotrain.transforms as cotrain_transforms
import openpi.cotrain.weight_loaders as cotrain_weight_loaders
from openpi.cotrain.rlds_dataset import CotrainRLDSDataset

logger = logging.getLogger(__name__)


def load_per_dataset_norm_stats(assets_dirs: pathlib.Path, datasets) -> dict:
    """Load per-dataset norm stats from `<assets_dirs>/<dataset_name>` (skip if missing).

    Returns {dataset_name: {"state": NormStats, "actions": NormStats}} for the DispatchNormalize.
    """
    stats: dict = {}
    for ds in datasets:
        try:
            d = str(pathlib.Path(assets_dirs) / ds.uid)
            stats[ds.uid] = _normalize.load(_download.maybe_download(d))
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

        base = self.create_base_config(assets_dirs, model_config)

        # Per-dataset absolute->delta action conversion (e.g. RoboMIND absolute joint).
        delta_masks = {
            ds.uid: _transforms.make_bool_mask(*ds.delta_action_mask_dims)
            for ds in self.datasets
            if ds.delta_action_mask_dims is not None
        }
        dispatch_delta = cotrain_transforms.DispatchDeltaActions(masks_by_dataset=delta_masks)

        # Per-dataset normalization (dispatched at runtime by dataset_id). Quantile norm for
        # pi05 (use_quantile_norm is True for non-PI0 models in create_base_config).
        per_dataset_stats = load_per_dataset_norm_stats(assets_dirs, self.datasets)
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
            datasets=self.datasets,
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
    # NOTE: the number of valid (non-padded) action dims for the MSE mask is taken
    # per-dataset from each CotrainRLDSDataset.action_dim (0 -> all dims).
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
# This edited registry intentionally keeps ONLY the requested piper30 RLDS dataset:
#   /mnt/data/RLDS/realworld_piper/
#   piper_s14_a14_fps30_c4_ee_pose_cam_front_cam_high_cam_left_wrist_cam_right_wrist
#
# Initialization choices:
#   - cotrain_piper30_only / cotrain_all / cotrain_all_2ep:
#       fine-tune FROM pi05_base checkpoint (the choice we want).
#   - cotrain_piper30_only_paligemma:
#       initialize FROM raw PaliGemma VLM backbone only (action expert random-init), kept
#       as an explicit optional config so the two training starts remain selectable.
#
# For pi05 checkpoint compatibility, keep the model at the default pi05 action_dim
# (do NOT widen to 40; that was only needed for RoboCOIN in the old multi-dataset mix).

_PIPER30_ROOT = (
    "/mnt/data/RLDS/realworld_piper/"
    "piper_s14_a14_fps30_c4_ee_pose_cam_front_cam_high_cam_left_wrist_cam_right_wrist"
)
_PIPER30_BUILDER_DIR = f"{_PIPER30_ROOT}/realworld_piper_infidata/1.0.0"
_PIPER30_TRAIN_EPISODES = 5_307

_DROID_ROOT = "/mnt/data/RLDS/DROID"
_DROID_BUILDER_DIR = f"{_DROID_ROOT}/droid_infidata/1.1.0"
_DROID_TRAIN_EPISODES = 64_124

_EGOVERSE_FULL_ROOT = "/mnt/data/RLDS/EgoVerse_full"
_EGOVERSE_FULL_TRAIN_EPISODES = 910 + 2_813 + 770 + 39_530 + 16_223

_ROBOCOIN_ROOT = "/mnt/data/RLDS/RoboCOIN"
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
    "/mnt/data/RLDS/AgiBot/"
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

_ROBOMIND_FULL_ROOT = "/mnt/data/RLDS/RoboMIND_full"
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
        (16,),
        ("cam_top", None, None),
    ),
    (
        "robomind_tienkung_prod1_gello_s16_a16",
        "tienkung_humanoid_master_puppet_joint_position_h5_tienkung_prod1_gello_1rgb_real_s16_a16_fps30_cam_top__episodes_2959",
        2_959,
        16,
        (16,),
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
        (38,),
        ("cam_chest", "cam_head", None),
    ),
    (
        "robomind_tienkung_real_s38_a38",
        "tienkung_humanoid_tiangong_joint_position_none_real_s38_a38_fps30_cam_chest_cam_head__episodes_146",
        146,
        38,
        (38,),
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


_FULL_ALL_TRAIN_EPISODES = (
    _AGIBOT_TRAIN_EPISODES
    + _DROID_TRAIN_EPISODES
    + _EGOVERSE_FULL_TRAIN_EPISODES
    + _PIPER30_TRAIN_EPISODES
    + _ROBOCOIN_TRAIN_EPISODES
    + _ROBOMIND_FULL_EPISODES
)


def _scale_dataset_weights(datasets: tuple[CotrainRLDSDataset, ...], train_episodes: int):
    return tuple(
        dataclasses.replace(ds, weight=ds.weight * train_episodes / _FULL_ALL_TRAIN_EPISODES) for ds in datasets
    )


_FULL_ALL_DATA = CotrainDataConfig(
    rlds_data_dir="/mnt/data/RLDS",
    datasets=(
        *_scale_dataset_weights(_AGIBOT_DATA.datasets, _AGIBOT_TRAIN_EPISODES),
        *_scale_dataset_weights(_DROID_DATA.datasets, _DROID_TRAIN_EPISODES),
        *_scale_dataset_weights(_EGOVERSE_FULL_DATA.datasets, _EGOVERSE_FULL_TRAIN_EPISODES),
        *_scale_dataset_weights(_PIPER30_DATA.datasets, _PIPER30_TRAIN_EPISODES),
        *_scale_dataset_weights(_ROBOCOIN_DATA.datasets, _ROBOCOIN_TRAIN_EPISODES),
        *_scale_dataset_weights(_ROBOMIND_FULL_DATA.datasets, _ROBOMIND_FULL_EPISODES),
    ),
)


_PIPER30_ONLY_PI05 = CotrainTrainConfig(
    name="cotrain_piper30_only",
    model=pi0_config.Pi0Config(pi05=True),
    data=_PIPER30_DATA,
    # Fine-tune from the trained pi05 VLA checkpoint. This is the selected start point.
    # Public openpi checkpoint; includes the PaliGemma backbone plus the trained pi05 action expert.
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    batch_size=32,
    num_train_steps=30_000,
    log_interval=100,
    save_interval=5_000,
    keep_period=5_000,
    eval_interval=1_000,
    num_val_batches=10,
    num_action_mse_batches=2,
    exp_name=tyro.MISSING,
)

_DROID_ONLY_PI05 = CotrainTrainConfig(
    name="cotrain_droid",
    model=pi0_config.Pi0Config(pi05=True),
    data=_DROID_DATA,
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    batch_size=32,
    num_train_steps=30_000,
    log_interval=100,
    save_interval=2_000,
    eval_interval=1_000,
    num_val_batches=10,
    num_action_mse_batches=2,
    exp_name=tyro.MISSING,
)

_AGIBOT_ONLY_PI05 = CotrainTrainConfig(
    name="cotrain_agibot",
    model=pi0_config.Pi0Config(pi05=True),
    data=_AGIBOT_DATA,
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    batch_size=32,
    num_train_steps=30_000,
    log_interval=100,
    save_interval=2_000,
    eval_interval=1_000,
    num_val_batches=10,
    num_action_mse_batches=2,
    exp_name=tyro.MISSING,
)

_EGOVERSE_FULL_ONLY_PI05 = CotrainTrainConfig(
    name="cotrain_egoverse_full",
    model=pi0_config.Pi0Config(pi05=True),
    data=_EGOVERSE_FULL_DATA,
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    batch_size=32,
    num_train_steps=30_000,
    log_interval=100,
    save_interval=2_000,
    eval_interval=1_000,
    num_val_batches=10,
    num_action_mse_batches=2,
    exp_name=tyro.MISSING,
)

_ROBOCOIN_ONLY_PI05 = CotrainTrainConfig(
    name="cotrain_robocoin",
    # RoboCOIN_full includes many robot schemas. We crop each raw state/action to its named,
    # action-aligned proprio/action coordinates, then pad to a common model width. The widest
    # effective schema is Leju at 54 dims, so 64 is enough with headroom.
    model=pi0_config.Pi0Config(pi05=True, action_dim=64, max_token_len=384),
    data=_ROBOCOIN_DATA,
    # A 64-wide action/state head is not shape-compatible with pi05_base. Load the PaliGemma
    # VLM backbone and leave the action expert randomly initialized, matching the old widened
    # RoboCOIN-style setup.
    weight_loader=cotrain_weight_loaders.LocalPaliGemmaWeightLoader(
        npz_path="/mnt/data/cache/openpi/vertex-model-garden-paligemma-us/paligemma/pt_224.npz"
    ),
    batch_size=32,
    num_train_steps=30_000,
    log_interval=100,
    save_interval=2_000,
    eval_interval=1_000,
    num_val_batches=10,
    num_action_mse_batches=2,
    exp_name=tyro.MISSING,
)

_ROBOMIND_FULL_ONLY_PI05 = CotrainTrainConfig(
    name="cotrain_robomind_full",
    # RoboMIND_full reaches 38 native action/state dims, so pad to a 64-wide head.
    model=pi0_config.Pi0Config(pi05=True, action_dim=64, max_token_len=384),
    data=_ROBOMIND_FULL_DATA,
    weight_loader=cotrain_weight_loaders.LocalPaliGemmaWeightLoader(
        npz_path="/mnt/data/cache/openpi/vertex-model-garden-paligemma-us/paligemma/pt_224.npz"
    ),
    batch_size=32,
    num_train_steps=30_000,
    log_interval=100,
    save_interval=2_000,
    eval_interval=1_000,
    num_val_batches=10,
    num_action_mse_batches=2,
    exp_name=tyro.MISSING,
)

_FULL_ALL_PI05 = CotrainTrainConfig(
    name="cotrain_full_all",
    # Full co-training includes RoboCOIN (54 effective dims) and RoboMIND_full (38 dims), so
    # all native state/action vectors are padded to a shared 64-wide model head.
    model=pi0_config.Pi0Config(pi05=True, action_dim=64, max_token_len=384),
    data=_FULL_ALL_DATA,
    # Initialize from pi05_base for consistency with the piper30-only reproduction. The widened
    # 64-dim state/action projection/head is not shape-compatible with pi05_base's 32-dim head,
    # so the shape-safe loader skips only those mismatched keys and keeps their random init.
    weight_loader=cotrain_weight_loaders.ShapeSafeCheckpointWeightLoader(
        params_path="gs://openpi-assets/checkpoints/pi05_base/params",
    ),
    batch_size=32,
    num_train_steps=30_000,
    log_interval=100,
    save_interval=2_000,
    eval_interval=1_000,
    num_val_batches=10,
    num_action_mse_batches=2,
    exp_name=tyro.MISSING,
)

_FULL_ALL_PI05_FULL_NORM = dataclasses.replace(
    _FULL_ALL_PI05,
    name="cotrain_full_all_full_norm",
)

_PIPER30_ONLY_PALIGEMMA = dataclasses.replace(
    _PIPER30_ONLY_PI05,
    name="cotrain_piper30_only_paligemma",
    # Initialize from the raw PaliGemma VLM backbone only (action expert random-init).
    # Use this only if you intentionally want the PaliGemma-start baseline.
    weight_loader=cotrain_weight_loaders.LocalPaliGemmaWeightLoader(
        npz_path="/mnt/data/cache/openpi/vertex-model-garden-paligemma-us/paligemma/pt_224.npz"
    ),
)

_COTRAIN_CONFIGS = [
    _AGIBOT_ONLY_PI05,
    _DROID_ONLY_PI05,
    _EGOVERSE_FULL_ONLY_PI05,
    _ROBOCOIN_ONLY_PI05,
    _ROBOMIND_FULL_ONLY_PI05,
    _FULL_ALL_PI05,
    _FULL_ALL_PI05_FULL_NORM,
    # Clear explicit name for the intended training run.
    _PIPER30_ONLY_PI05,
    # Backward-compatible aliases: old launch commands will still train ONLY piper30 and
    # will now start from pi05, not from PaliGemma.
    dataclasses.replace(_PIPER30_ONLY_PI05, name="cotrain_all"),
    dataclasses.replace(_PIPER30_ONLY_PI05, name="cotrain_all_2ep"),
    # Optional baseline, selectable only by the explicit *_paligemma name.
    _PIPER30_ONLY_PALIGEMMA,
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
