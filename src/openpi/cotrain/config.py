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
from openpi.cotrain.rlds_dataset import CotrainRLDSDataset

logger = logging.getLogger(__name__)


def load_per_dataset_norm_stats(assets_dirs: pathlib.Path, datasets) -> dict:
    """Load per-dataset norm stats from `<assets_dirs>/<dataset_name>` (skip if missing).

    Returns {dataset_name: {"state": NormStats, "actions": NormStats}} for the DispatchNormalize.
    """
    stats: dict = {}
    for ds in datasets:
        try:
            d = str(pathlib.Path(assets_dirs) / ds.name)
            stats[ds.name] = _normalize.load(_download.maybe_download(d))
            logger.info(f"Loaded per-dataset norm stats for '{ds.name}' from {d}")
        except FileNotFoundError:
            logger.warning(f"Norm stats for dataset '{ds.name}' not found under {assets_dirs}; skipping (no norm).")
    return stats


@dataclasses.dataclass(frozen=True)
class CotrainDataConfig(_config.DataConfigFactory):
    """Multi-dataset RLDS data config with per-dataset train/val splits.

    Currently assumes the DROID schema for repack/data transforms (the framework
    milestone starts from DROID-format data). Heterogeneous schemas plug in via the
    restructure registry in `rlds_dataset.py` plus per-dataset repack transforms here.
    """

    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    datasets: tuple[CotrainRLDSDataset, ...] = ()

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> _config.DataConfig:
        assert self.rlds_data_dir is not None, "Need to set rlds_data_dir for the co-training RLDS loader."
        assert len(self.datasets) > 0, "Need at least one dataset in `datasets`."

        base = self.create_base_config(assets_dirs, model_config)

        # Per-dataset normalization (dispatched at runtime by dataset_id). Quantile norm for
        # pi05 (use_quantile_norm is True for non-PI0 models in create_base_config).
        per_dataset_stats = load_per_dataset_norm_stats(assets_dirs, self.datasets)
        dispatch_norm = cotrain_transforms.DispatchNormalize(
            norm_stats_by_dataset=per_dataset_stats,
            use_quantiles=base.use_quantile_norm,
        )

        # Generic inputs (the offline-standardized schema is uniform across datasets), then
        # per-dataset normalization. No per-dataset repack needed (StandardizedInputs reads
        # the nested standardized keys directly).
        data_transforms = _transforms.Group(
            inputs=[
                cotrain_transforms.StandardizedInputs(model_type=model_config.model_type),
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


# ---------------------------------------------------------------------------
# Config registry (separate from openpi's _CONFIGS; selected via this module's cli()).
# ---------------------------------------------------------------------------

_COTRAIN_CONFIGS = [
    # RoboMIND single-dataset config. RoboMIND is a clean RLDS, so it is mapped at runtime
    # by the "robomind" restructure (no offline regeneration). It ships train / seen_test /
    # unseen_test splits, exposed here as val labels "seen" and "unseen".
    # Run norm stats first:
    #   uv run python scripts/compute_cotrain_norm_stats.py cotrain_robomind \
    #       --rlds-data-dir /export/pgs/xule/RLDS/RoboMIND
    CotrainTrainConfig(
        name="cotrain_robomind",
        model=pi0_config.Pi0Config(pi05=True),
        data=CotrainDataConfig(
            rlds_data_dir="/export/pgs/xule/RLDS/RoboMIND",
            datasets=(
                CotrainRLDSDataset(
                    name="robomind_infidata",
                    version="1.1.0",
                    weight=1.0,
                    train_split="train",
                    val_splits={"seen": "seen_test", "unseen": "unseen_test"},
                    restructure_name="robomind",
                    action_dim=14,  # dual-arm 14-dim native state/action
                ),
            ),
        ),
        weight_loader=weight_loaders.PaliGemmaWeightLoader(),
        batch_size=32,
        num_train_steps=30_000,
        log_interval=100,
        save_interval=1_000,
        eval_interval=1_000,
        num_val_batches=20,
        num_action_mse_batches=5,
        exp_name=tyro.MISSING,
    ),
    # Template for offline-standardized datasets (multiple datasets => weights sum to 1.0).
    CotrainTrainConfig(
        name="cotrain_sanity",
        model=pi0_config.Pi0Config(pi05=True),
        data=CotrainDataConfig(
            rlds_data_dir=tyro.MISSING,  # point at the OFFLINE-STANDARDIZED RLDS dir
            datasets=(
                CotrainRLDSDataset(
                    name="droid_std",
                    version="1.0.0",
                    weight=1.0,
                    train_split="train",
                    val_splits={"val": "val"},
                    restructure_name="standardized",
                    action_dim=8,
                ),
            ),
        ),
        weight_loader=weight_loaders.PaliGemmaWeightLoader(),
        batch_size=32,
        num_train_steps=30_000,
        log_interval=100,
        save_interval=1_000,
        eval_interval=1_000,
        num_val_batches=20,
        num_action_mse_batches=5,
        exp_name=tyro.MISSING,
    ),
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
