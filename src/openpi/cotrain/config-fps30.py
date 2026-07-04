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
    save_interval=2_000,
    eval_interval=1_000,
    num_val_batches=10,
    num_action_mse_batches=2,
    exp_name=tyro.MISSING,
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
