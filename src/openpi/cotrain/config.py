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
from openpi.cotrain import fk_eef as cotrain_fk_eef
from openpi.cotrain.modes import ActionSupervisionMode, PromptActionMode
from openpi.cotrain.rlds_dataset import CotrainRLDSDataset
import openpi.cotrain.transforms as cotrain_transforms
import openpi.cotrain.weight_loaders as cotrain_weight_loaders
import openpi.models.fastwam_config as fastwam_config
import openpi.models.hpt_config as hpt_config
import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.training.weight_loaders as weight_loaders
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.optimizer as _optimizer
import openpi.transforms as _transforms

logger = logging.getLogger(__name__)


def _resolve_unified_spec(
    dataset_id: str,
    *,
    supervision_mode: ActionSupervisionMode,
) -> cotrain_action_space.UnifiedActionSpec:
    """Resolve the unified spec for loss masking.

    Mask / action mode follow the **native RLDS mapping** only. URDF availability must
    not invent extra FK EEF supervision slots.
    """
    try:
        spec = cotrain_action_space.UNIFIED_ACTION_SPECS[dataset_id]
    except KeyError as exc:
        raise ValueError(f"Dataset '{dataset_id}' has no registered unified 80D action mapping.") from exc

    # Always clear FK-only slots: supervision is the registered action_mapping only.
    spec = dataclasses.replace(spec, supervision_mode=supervision_mode, fk_eef_slots=())
    if supervision_mode == ActionSupervisionMode.EEF and not any(spec.action_mask):
        raise ValueError(
            f"Dataset '{dataset_id}' has no native EEF action slots for action_supervision_mode=EEF. "
            "Joint-mapped datasets (e.g. piper) cannot be supervised as EEF without FK fill, "
            "which is disabled."
        )
    return spec


def _resolve_unified_datasets(
    datasets,
    model_config: _model.BaseModelConfig,
    *,
    supervision_mode: ActionSupervisionMode = ActionSupervisionMode.JOINT,
):
    if model_config.action_dim != cotrain_action_space.UNIFIED_ACTION_DIM:
        raise ValueError(
            f"All co-training configs require action_dim={cotrain_action_space.UNIFIED_ACTION_DIM}, "
            f"got {model_config.action_dim}."
        )

    resolved = []
    for ds in datasets:
        spec = _resolve_unified_spec(ds.uid, supervision_mode=supervision_mode)
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
    # Which unified slots to supervise. Default JOINT = native ``action_mapping`` slots only
    # (for EgoVerse-style datasets those slots are already EEF). Do not invent FK EEF dims.
    action_supervision_mode: ActionSupervisionMode = ActionSupervisionMode.JOINT
    # NATIVE keeps each dataset's RLDS ``prompt_prefix`` (Action Mode: joint/eef).
    prompt_action_mode: PromptActionMode = PromptActionMode.NATIVE

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> _config.DataConfig:
        assert self.rlds_data_dir is not None, "Need to set rlds_data_dir for the co-training RLDS loader."
        assert len(self.datasets) > 0, "Need at least one dataset in `datasets`."
        datasets = _resolve_unified_datasets(
            self.datasets,
            model_config,
            supervision_mode=self.action_supervision_mode,
        )
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

        # Native actions only: no URDF FK EEF fill. Delta MUST precede normalization
        # (stats are computed on deltas).
        data_transforms = _transforms.Group(
            inputs=[
                cotrain_transforms.StandardizedInputs(model_type=model_config.model_type),
                cotrain_transforms.DispatchPromptPrefix(prompt_action_mode=self.prompt_action_mode),
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
    # Independent global validation batch size. None preserves the legacy behavior of
    # reusing the training batch size. Large co-training batches should set this explicitly
    # to avoid creating an enormous XLA graph for validation.
    val_batch_size: int | None = None
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
    # Cap datasets evaluated per val label (seen/unseen). None = all. Uses evenly spaced
    # names so aggregate metrics still cover the mixture without running every loader.
    val_max_datasets: int | None = None
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
    # Classic DDP: every rank opens the full RLDS builder set; data is sharded via
    # tfds.even_splits / sampler. Set True only to cut host RAM when dataset count >> GPUs.
    rlds_partition_builders_by_rank: bool = False
    # Optional override for assets/<subdir> when it differs from config name (e.g. FastWAM reuses
    # pi05 norm stats under assets/cotrain_fk_eef_plus_piper_ego).
    assets_name: str | None = None
    # Freeze the first train batch and reuse it every step (single-batch overfit probe).
    overfit_fixed_batch: bool = False
    # If set, use fixed continuous σ for video / action FM instead of sampling (e.g. 0.5).
    fixed_video_sigma: float | None = None
    fixed_action_sigma: float | None = None
    # When overfit_fixed_batch is True, also reuse the first-step noise tensors.
    overfit_fixed_noise: bool = True

    @property
    def assets_dirs(self) -> pathlib.Path:
        subdir = self.assets_name or self.name
        return (pathlib.Path(self.assets_base_dir) / subdir).resolve()


# ---------------------------------------------------------------------------
# Config registry (separate from openpi's _CONFIGS; selected via this module's cli()).
# ---------------------------------------------------------------------------
# Every config in this registry uses the fixed 80D state/action layout. pi05_base has a
# 32D projection/head, so checkpoint-start configs use the shape-safe loader and randomly
# initialize only parameters whose shapes changed.

_RLDS_ROOT = os.environ.get("RLDS_DATA_DIR", "/mnt/workspace/RLDS")

# piper30: use task-disjoint repartition (1.1.0) instead of the older
# realworld_piper/…/1.0.0 cut. Same schema; train/seen/unseen are re-split so
# unseen_test is task-disjoint and seen_test is trajectory-disjoint.
_PIPER30_ROOT = (
    f"{_RLDS_ROOT}/realworld_piper_task_split/"
    "piper_s14_a14_fps30_c4_ee_pose_cam_front_cam_high_cam_left_wrist_cam_right_wrist"
)
_PIPER30_BUILDER_DIR = f"{_PIPER30_ROOT}/realworld_piper_infidata/1.1.0"
_PIPER30_TRAIN_EPISODES = 4_927

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

_EGOVERSE_RL2_ROOT = f"{_RLDS_ROOT}/EgoVerse_rl2"
_EGOVERSE_RL2_TRAIN_EPISODES = 2_831 + 1_387

_ATOM_ALIGNED_ROOT = f"{_RLDS_ROOT}/AtomAligned_full"
_ATOM_ALIGNED_VERSION = "1.0.0"
# hz_h + hz_r + sz_h + sz_r on bo23lu (sz_robot=163)
_ATOM_ALIGNED_TRAIN_EPISODES = 387 + 90 + 656 + 163


def _make_atom_aligned_dataset(dataset_id: str, *, action_dim: int, weight: float) -> CotrainRLDSDataset:
    return CotrainRLDSDataset(
        name="atom_aligned_rlds",
        dataset_id=dataset_id,
        version=_ATOM_ALIGNED_VERSION,
        builder_dir=f"{_ATOM_ALIGNED_ROOT}/{dataset_id}/{_ATOM_ALIGNED_VERSION}",
        weight=weight,
        train_split="train",
        val_splits={"seen": "seen_test", "unseen": "unseen_test"},
        restructure_name="aligned_parallel_gripper",
        action_dim=action_dim,
    )


_ATOM_ALIGNED_DATA = CotrainDataConfig(
    rlds_data_dir=_ATOM_ALIGNED_ROOT,
    datasets=(
        _make_atom_aligned_dataset("aligned_hangzhou_human_right", action_dim=6, weight=387 / _ATOM_ALIGNED_TRAIN_EPISODES),
        _make_atom_aligned_dataset("aligned_hangzhou_robot_right", action_dim=7, weight=90 / _ATOM_ALIGNED_TRAIN_EPISODES),
        _make_atom_aligned_dataset("aligned_shenzhen_human_bimanual", action_dim=12, weight=656 / _ATOM_ALIGNED_TRAIN_EPISODES),
        _make_atom_aligned_dataset("aligned_shenzhen_robot_bimanual", action_dim=14, weight=163 / _ATOM_ALIGNED_TRAIN_EPISODES),
    ),
)

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
            # 12D EE + left/right gripper from source_float_vectors -> 14D.
            restructure_name="egoverse_eva",
            action_dim=14,
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

_EGOVERSE_RL2_DATA = CotrainDataConfig(
    rlds_data_dir=_EGOVERSE_RL2_ROOT,
    datasets=(
        CotrainRLDSDataset(
            name="ego_verse_infidata",
            dataset_id="egoverse_rl2_eva",
            version="1.0.0",
            builder_dir=(
                f"{_EGOVERSE_RL2_ROOT}/eva_bimanual_front_1_left_wrist_right_wrist/"
                "ego_verse_infidata/1.0.0"
            ),
            weight=2_831 / _EGOVERSE_RL2_TRAIN_EPISODES,
            train_split="train",
            val_splits={"seen": "seen_test", "unseen": "unseen_test"},
            restructure_name="egoverse_eva",
            action_dim=14,
            delta_action_mask_dims=None,
        ),
        CotrainRLDSDataset(
            name="ego_verse_infidata",
            dataset_id="egoverse_rl2_human",
            version="1.0.0",
            builder_dir=f"{_EGOVERSE_RL2_ROOT}/human_bimanual_front_1/ego_verse_infidata/1.0.0",
            weight=1_387 / _EGOVERSE_RL2_TRAIN_EPISODES,
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
            version="1.1.0",
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
    + _EGOVERSE_RL2_TRAIN_EPISODES
    + _ATOM_ALIGNED_TRAIN_EPISODES
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


def _keep_dataset_ids(
    keep_ids: frozenset[str],
    order: tuple[str, ...],
    *sources: tuple[CotrainRLDSDataset, ...],
) -> tuple[CotrainRLDSDataset, ...]:
    """Select a subset of co-training datasets (by uid) and renormalize mixture weights."""
    by_uid: dict[str, CotrainRLDSDataset] = {}
    for datasets in sources:
        for ds in datasets:
            if ds.uid in keep_ids:
                by_uid[ds.uid] = ds
    missing = keep_ids - by_uid.keys()
    if missing:
        raise ValueError(f"Co-training datasets not found in sources: {sorted(missing)}")
    extra = set(by_uid) - set(order)
    if extra:
        raise ValueError(f"Dataset order missing uids: {sorted(extra)}")
    kept = tuple(by_uid[uid] for uid in order)
    total_weight = sum(ds.weight for ds in kept)
    return tuple(dataclasses.replace(ds, weight=ds.weight / total_weight) for ds in kept)


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

# Production mixture aligned with assets/cotrain_real_robot_ego_fix: unaudited real+robot
# (same as cotrain_real_robot) plus EgoVerse aria/eva/human/mecka and EgoVerse_rl2
# (eva/human). egoverse_scale is intentionally excluded (no norm assets under that uid).
_EGOVERSE_EGO_FIX_TRAIN_EPISODES = 910 + 2_813 + 770 + 39_530
_EGOVERSE_EGO_FIX_DATASETS = tuple(ds for ds in _EGOVERSE_FULL_DATA.datasets if ds.uid != "egoverse_scale")
_REAL_ROBOT_EGO_FIX_DATA = CotrainDataConfig(
    rlds_data_dir=_RLDS_ROOT,
    datasets=_drop_excluded_and_renormalize(
        (
            *_scale_dataset_weights(_PIPER30_DATA.datasets, _PIPER30_TRAIN_EPISODES),
            *_scale_dataset_weights(_PIPER2_DATA.datasets, _PIPER2_TRAIN_EPISODES),
            *_scale_dataset_weights(_AGIBOT_DATA.datasets, _AGIBOT_TRAIN_EPISODES),
            *_scale_dataset_weights(_DROID_DATA.datasets, _DROID_TRAIN_EPISODES),
            *_scale_dataset_weights(_EGOVERSE_EGO_FIX_DATASETS, _EGOVERSE_EGO_FIX_TRAIN_EPISODES),
            *_scale_dataset_weights(_EGOVERSE_RL2_DATA.datasets, _EGOVERSE_RL2_TRAIN_EPISODES),
            *_scale_dataset_weights(_ATOM_ALIGNED_DATA.datasets, _ATOM_ALIGNED_TRAIN_EPISODES),
            *_scale_dataset_weights(_ROBOCOIN_DATA.datasets, _ROBOCOIN_TRAIN_EPISODES),
            *_scale_dataset_weights(_ROBOMIND_FULL_DATA.datasets, _ROBOMIND_FULL_EPISODES),
        ),
    ),
)

# Cross-robot FastWAM mixture (see docs/wam-cross.md). Per-dataset norm stats reuse
# assets/cotrain_real_robot_ego_fix/<uid>/.
_WAM_CROSS_ROBOT_DATASET_ORDER = (
    "agibot",
    "robocoin_agilex_cobot_magic_s26_a26",
    "robocoin_airbot_mmk2_s36_a36",
    "robocoin_realman_rmc_aida_l_s28_a28",
    "robocoin_agilex_decoupled_magic_s14_a14_fps50",
    "robocoin_agilex_decoupled_magic_s26_a26",
    "robocoin_aloha_s26_a26",
    "robocoin_alpha_bot_2_s28_a28",
    "robocoin_discover_aitbot_mmk2_s36_a36",
    "robocoin_realman_rmc_aidal_s28_a28",
    "robocoin_ruantong_a2d_s17_a17",
    "robocoin_unitree_g1_s28_a28",
    "robomind_agilex_cobot_magic_s14_a14",
    "piper2",
    "piper30",
)
_WAM_CROSS_ROBOT_DATASET_IDS = frozenset(_WAM_CROSS_ROBOT_DATASET_ORDER)

_WAM_CROSS_ROBOT_DATA = CotrainDataConfig(
    rlds_data_dir=_RLDS_ROOT,
    datasets=_keep_dataset_ids(
        _WAM_CROSS_ROBOT_DATASET_IDS,
        _WAM_CROSS_ROBOT_DATASET_ORDER,
        _REAL_ROBOT_EGO_FIX_DATA.datasets,
        _ROBOCOIN_DATA.datasets,
    ),
)

# wam-cross-fix uses the full real+ego mixture (_REAL_ROBOT_EGO_FIX_DATA, 47 datasets).

_WAM_CROSS_PIPER_DATA = CotrainDataConfig(
    rlds_data_dir=_RLDS_ROOT,
    datasets=_drop_excluded_and_renormalize(
        (
            *_scale_dataset_weights(_PIPER30_DATA.datasets, _PIPER30_TRAIN_EPISODES),
            *_scale_dataset_weights(_PIPER2_DATA.datasets, _PIPER2_TRAIN_EPISODES),
        ),
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
            *_scale_dataset_weights(_EGOVERSE_RL2_DATA.datasets, _EGOVERSE_RL2_TRAIN_EPISODES),
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
    val_batch_size=96,
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

_REAL_ROBOT_EGO_FIX_PI05 = dataclasses.replace(
    _REAL_ROBOT_PI05,
    name="cotrain_real_robot_ego_fix",
    data=_REAL_ROBOT_EGO_FIX_DATA,
    rlds_partition_builders_by_rank=True,
)

_FULL_ALL_PI05_FULL_NORM = dataclasses.replace(
    _REAL_ONLY_PI05,
    name="cotrain_full_all_full_norm",
    data=_FULL_ALL_FIX_DATA,
)

# Anchor-style mixture aligned with assets/cotrain_fk_eef_plus_piper_ego:
#   * datasets whose URDF joint-count validation passes (EEF filled by FK)
#   * plus piper2 / piper30 / EgoVerse (aria/eva/human/mecka only; no scale)
# agibot and egoverse_scale are intentionally excluded (no norm assets under those uids).
_FK_EEF_ANCHOR_EXTRA_IDS = frozenset(
    {
        "piper30",
        "piper2",
        "egoverse_aria",
        "egoverse_eva",
        "egoverse_human",
        "egoverse_mecka",
        "egoverse_rl2_eva",
        "egoverse_rl2_human",
    }
)
_FK_EEF_EXCLUDED_DATASET_IDS = frozenset({"agibot", "egoverse_scale"})


def _fk_eef_plus_piper_ego_datasets():
    # Historical FK-experiment mixture (matches assets/cotrain_fk_eef_plus_piper_ego).
    # Local URDF presence still selects *which robots are in this mixture*, but it no longer
    # changes action_mask / action_mode (see ``_resolve_unified_spec``).
    enabled = set(cotrain_fk_eef.enabled_fk_dataset_ids())
    keep = (enabled | _FK_EEF_ANCHOR_EXTRA_IDS) - _FK_EEF_EXCLUDED_DATASET_IDS
    return _drop_dataset_ids_and_renormalize(
        _FULL_ALL_FIX_DATA.datasets,
        {ds.uid for ds in _FULL_ALL_FIX_DATA.datasets if ds.uid not in keep},
    )


def _fk_eef_plus_piper_ego_eef_only_datasets():
    """Anchor mix with only **native** EEF-mapped datasets (EgoVerse / aligned cartesian).

    Joint-mapped robots (piper, etc.) are excluded: FK-from-URDF fill is disabled, so they
    have no EEF action slots in the native mapping.
    """
    from openpi.cotrain import supervision as cotrain_supervision

    excluded: set[str] = set()
    for ds in _fk_eef_plus_piper_ego_datasets():
        spec = cotrain_action_space.UNIFIED_ACTION_SPECS[ds.uid]
        if not cotrain_supervision.is_native_eef_spec(spec):
            excluded.add(ds.uid)
    if excluded:
        logger.warning(
            "EEF-only mixture excludes datasets without native EEF mapping: %s",
            ", ".join(sorted(excluded)),
        )
    anchor_ids = {ds.uid for ds in _fk_eef_plus_piper_ego_datasets()}
    drop = {ds.uid for ds in _FULL_ALL_FIX_DATA.datasets if ds.uid not in anchor_ids or ds.uid in excluded}
    return _drop_dataset_ids_and_renormalize(_FULL_ALL_FIX_DATA.datasets, drop)


_FK_EEF_PLUS_PIPER_EGO_DATA = CotrainDataConfig(
    rlds_data_dir=_RLDS_ROOT,
    datasets=_fk_eef_plus_piper_ego_datasets(),
)

_FK_EEF_PLUS_PIPER_EGO_EEF_ONLY_DATA = CotrainDataConfig(
    rlds_data_dir=_RLDS_ROOT,
    datasets=_fk_eef_plus_piper_ego_eef_only_datasets(),
    action_supervision_mode=ActionSupervisionMode.EEF,
    prompt_action_mode=PromptActionMode.EEF,
)

_FK_EEF_PLUS_PIPER_EGO_PI05 = dataclasses.replace(
    _REAL_ONLY_PI05,
    name="cotrain_fk_eef_plus_piper_ego",
    assets_name="cotrain_fk_eef_plus_piper_ego",
    data=_FK_EEF_PLUS_PIPER_EGO_DATA,
)

_FK_EEF_PLUS_PIPER_EGO_EEF_ONLY_PI05 = dataclasses.replace(
    _REAL_ONLY_PI05,
    name="cotrain_fk_eef_plus_piper_ego_eef_only",
    assets_name="cotrain_fk_eef_plus_piper_ego",
    data=_FK_EEF_PLUS_PIPER_EGO_EEF_ONLY_DATA,
)

_UNIFIED_FASTWAM = fastwam_config.FastWAMConfig(
    action_dim=cotrain_action_space.UNIFIED_ACTION_DIM,
    proprio_dim=cotrain_action_space.UNIFIED_ACTION_DIM,
    action_horizon=32,
    video_num_frames=9,
    action_video_freq_ratio=4,
    camera_keys=("base_0_rgb", "left_wrist_0_rgb"),
    concat_multi_camera="horizontal",
    # Four-way loss: ego/robot × world(video)/action. Tune per experiment as needed.
    loss={
        "lambda_ego_video": 0.60,
        "lambda_ego_action": 1.00,
        "lambda_robot_video": 0.60,
        "lambda_robot_action": 1.00,
    },
)

_WAM_CROSS_ROBOT_FASTWAM = dataclasses.replace(
    _UNIFIED_FASTWAM,
    camera_keys=("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"),
    concat_multi_camera="robot_wrist",
    # Half of original 576×512: VAE/DiT cheaper; TF decode→slot-resize before batch
    # keeps post-decode CPU RAM at compose targets (not native ~480×640).
    image_resolution=(288, 256),
    # Robot-only mixture: keep robot heads, zero ego (no egoverse datasets).
    loss={
        "lambda_ego_video": 0.0,
        "lambda_ego_action": 0.0,
        "lambda_robot_video": 0.60,
        "lambda_robot_action": 1.00,
    },
)

# FastWAM LR (enforced in ``scripts/train_fastwam.py``, not optax.create()):
#   - warmup_steps: passed through to train_fastwam.py (default in config = 5% mirror;
#     override via CLI / baige WARMUP_STEPS).
#   - peak_lr / decay_lr are read by the trainer.
#   - Curve: linear warmup to peak_lr, then cosine decay to decay_lr by the last step.
#   - decay_steps is set to num_train_steps (full horizon); the trainer does not use
#     this field for the curve, but keeps it aligned for logs / baige CLI mirrors.
_WAM_CROSS_ROBOT = CotrainTrainConfig(
    name="wam-cross-robot",
    assets_name="cotrain_real_robot_ego_fix",
    model=_WAM_CROSS_ROBOT_FASTWAM,
    data=_WAM_CROSS_ROBOT_DATA,
    weight_loader=weight_loaders.NoOpWeightLoader(),
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=15_000,  # = 5% * 300_000
        peak_lr=1.0e-4,
        decay_steps=300_000,  # = num_train_steps
        decay_lr=1.0e-6,
    ),
    optimizer=_optimizer.AdamW(clip_gradient_norm=1.0, weight_decay=0.01),
    batch_size=160,
    num_train_steps=300_000,
    log_interval=50,
    save_interval=10_000,
    eval_interval=0,
    val_batch_size=4,
    num_val_batches=1,
    num_action_mse_batches=1,
    run_action_mse=False,
    val_flow_loss_mode="fixed_seed",
    val_max_datasets=8,
    shuffle_buffer_size=256,
    data_num_parallel_reads=1,
    data_num_parallel_calls=1,
    rlds_partition_builders_by_rank=False,
    wandb_enabled=True,
    exp_name=tyro.MISSING,
)

# Full mixture pretrain (real robot + EgoVerse): same robot_wrist / 288×256 as wam-cross-robot.
# Inherits robot LR (1e-4 peak, 5% warmup, cosine → 1e-6) and weight_decay=0.01.
_WAM_CROSS_FIX_FASTWAM = dataclasses.replace(
    _WAM_CROSS_ROBOT_FASTWAM,
    loss={
        "lambda_ego_video": 0.60,
        "lambda_ego_action": 1.00,
        "lambda_robot_video": 0.60,
        "lambda_robot_action": 1.00,
    },
)

_WAM_CROSS_FIX = dataclasses.replace(
    _WAM_CROSS_ROBOT,
    name="wam-cross-fix",
    model=_WAM_CROSS_FIX_FASTWAM,
    data=_REAL_ROBOT_EGO_FIX_DATA,
    # 47 builders: partition by rank to avoid每个 rank 打开全部 TFDS graph (CPU OOM).
    rlds_partition_builders_by_rank=True,
)

# Piper fine-tune: init from wam-cross-robot ckpt; freeze Video DiT (Wan), train Action DiT + MoT.
# Same LR shape as robot/fix (linear 5% warmup → peak 1e-4 → cosine to 1e-6), shorter horizon.
_WAM_CROSS_PIPER_FT = dataclasses.replace(
    _WAM_CROSS_ROBOT,
    name="wam-cross-piper-ft",
    data=_WAM_CROSS_PIPER_DATA,
    model=dataclasses.replace(
        _WAM_CROSS_ROBOT_FASTWAM,
        skip_dit_load_from_pretrain=True,
        skip_vae_load_from_pretrain=True,
        freeze_video_expert=True,
    ),
    num_train_steps=20_000,
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=1_000,  # = 5% * 20_000
        peak_lr=1.0e-4,
        decay_steps=20_000,  # = num_train_steps
        decay_lr=1.0e-6,
    ),
    save_interval=5_000,
    eval_interval=1_000,
    val_batch_size=4,
    num_val_batches=10,
    num_action_mse_batches=2,
    run_action_mse=True,
    val_flow_loss_mode="fixed_seed",
    val_max_datasets=None,
    pytorch_weight_path=(
        "checkpoints/wam-cross-robot/fw-wam-cross-v3-8gpu-b208/20000"
    ),
    exp_name=tyro.MISSING,
)

# ---------------------------------------------------------------------------
# HPT (Heterogeneous Pre-trained Transformer) — PyTorch cotrain path
# ---------------------------------------------------------------------------
_UNIFIED_HPT_PRETRAIN = hpt_config.HPTConfig(
    action_dim=cotrain_action_space.UNIFIED_ACTION_DIM,
    proprio_dim=cotrain_action_space.UNIFIED_ACTION_DIM,
    action_horizon=50,
    video_num_frames=2,
    action_video_freq_ratio=50,
    observation_horizon=1,
    random_horizon_masking=False,
    embed_dim=256,
    ego_tokens=32,
    wrist_tokens=16,
    state_tokens=16,
    language_tokens=8,
    action_head_type="transformer_decoder",
    action_head_dim=128,
    action_head_blocks=6,
    action_head_heads=4,
    num_inference_steps=50,
    head_mode="action_world",
    train_mode="pretrain",
    loss={
        "lambda_ego_world": 1.0,
        "lambda_ego_action": 0.5,
        "lambda_robot_world": 0.5,
        "lambda_robot_action": 1.0,
        "lambda_action_smooth": 0.1,
    },
)
_UNIFIED_HPT_FINETUNE = dataclasses.replace(_UNIFIED_HPT_PRETRAIN, train_mode="finetune")

# Piper real-only (assets/cotrain_real_only): trunk warm-start + train stem/trunk/heads.
# Cosine 100k: warmup 5k (5%) to peak 1e-4, decay to 1e-5. Weight decay 1e-4 (EgoWAM / HPT transfer).
_HPT_REAL_ONLY = CotrainTrainConfig(
    name="hpt_cotrain_real_only",
    assets_name="cotrain_real_only",
    model=_UNIFIED_HPT_PRETRAIN,
    data=_REAL_ONLY_DATA,
    weight_loader=weight_loaders.NoOpWeightLoader(),
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=5_000,
        peak_lr=1.0e-4,
        decay_steps=100_000,
        decay_lr=1.0e-5,
    ),
    optimizer=_optimizer.AdamW(clip_gradient_norm=1.0, weight_decay=1e-4),
    ema_decay=None,
    eval_on_ema=False,
    batch_size=128,
    num_train_steps=100_000,
    log_interval=50,
    save_interval=10_000,
    eval_interval=1_000,
    val_batch_size=32,
    num_val_batches=10,
    num_action_mse_batches=2,
    run_action_mse=True,
    shuffle_buffer_size=10_000,
    data_num_parallel_reads=1,
    data_num_parallel_calls=2,
    rlds_partition_builders_by_rank=False,
    wandb_enabled=True,
    exp_name=tyro.MISSING,
)

# Large real+robot+ego mixture (assets/cotrain_real_robot_ego_fix): same cosine recipe as
# real_only, scaled to 300k (warmup 5% = 15k, peak 1e-4 → 1e-5, wd 1e-4).
_HPT_REAL_ROBOT_EGO_FIX = CotrainTrainConfig(
    name="hpt_cotrain_real_robot_ego_fix",
    assets_name="cotrain_real_robot_ego_fix",
    model=_UNIFIED_HPT_PRETRAIN,
    data=_REAL_ROBOT_EGO_FIX_DATA,
    weight_loader=weight_loaders.NoOpWeightLoader(),
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=15_000,
        peak_lr=1.0e-4,
        decay_steps=300_000,
        decay_lr=1.0e-5,
    ),
    optimizer=_optimizer.AdamW(clip_gradient_norm=1.0, weight_decay=1e-4),
    ema_decay=None,
    eval_on_ema=False,
    batch_size=128,
    num_train_steps=300_000,
    log_interval=50,
    save_interval=30_000,
    eval_interval=0,
    val_batch_size=16,
    num_val_batches=1,
    shuffle_buffer_size=256,
    data_num_parallel_reads=1,
    data_num_parallel_calls=1,
    rlds_partition_builders_by_rank=True,
    wandb_enabled=True,
    exp_name=tyro.MISSING,
)

_HPT_SMOKE = dataclasses.replace(
    _HPT_REAL_ONLY,
    name="hpt_cotrain_smoke",
    model=dataclasses.replace(_UNIFIED_HPT_PRETRAIN, load_encoders=False),
    data=_PIPER30_DATA,
    batch_size=2,
    num_train_steps=2,
    log_interval=1,
    save_interval=10,
    eval_interval=0,
    shuffle_buffer_size=64,
    wandb_enabled=False,
    exp_name="smoke",
)

_COTRAIN_CONFIGS = [
    _REAL_ONLY_PI05,
    _REAL_ROBOT_PI05,
    _REAL_ROBOT_FIX_PI05,
    _REAL_ROBOT_EGO_FIX_PI05,
    _FULL_ALL_PI05_FULL_NORM,
    _FK_EEF_PLUS_PIPER_EGO_PI05,
    _FK_EEF_PLUS_PIPER_EGO_EEF_ONLY_PI05,
    _WAM_CROSS_ROBOT,
    _WAM_CROSS_FIX,
    _WAM_CROSS_PIPER_FT,
    _HPT_REAL_ONLY,
    _HPT_REAL_ROBOT_EGO_FIX,
    _HPT_SMOKE,
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
