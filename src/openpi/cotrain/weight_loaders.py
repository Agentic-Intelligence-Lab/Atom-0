"""Weight loaders for co-training that read from explicit LOCAL paths.

Avoids openpi's gs:// download + cache-path guessing (which breaks across ephemeral pods
when OPENPI_DATA_HOME differs). Point directly at a file you already have on disk.
"""

import dataclasses

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download
from openpi.training.weight_loaders import _merge_params


@dataclasses.dataclass(frozen=True)
class LocalPaliGemmaWeightLoader:
    """Load the PaliGemma big_vision `.npz` from a local path (no GCS).

    Mirrors openpi's PaliGemmaWeightLoader but reads `npz_path` directly. The file must be
    the big_vision JAX export (flat keys with '/' separators under a 'params' subtree).
    """

    npz_path: str

    def load(self, params: at.Params) -> at.Params:
        with open(self.npz_path, "rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights (e.g. the randomly-initialized action expert).
        return _merge_params(loaded_params, params, missing_regex=".*")


@dataclasses.dataclass(frozen=True)
class ShapeSafeCheckpointWeightLoader:
    """Load a checkpoint, skipping keys whose shapes no longer match the target model.

    This is used when widening the co-training action/state width (e.g. pi05_base has a
    32-wide head while full-all uses 64). Matching pi05_base weights are loaded; widened
    projection/head parameters stay at the target model's random initialization.
    """

    params_path: str
    missing_regex: str = ".*"

    def load(self, params: at.Params) -> at.Params:
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
        flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")
        compatible = {
            key: value
            for key, value in flat_loaded.items()
            if key in flat_ref and getattr(value, "shape", None) == getattr(flat_ref[key], "shape", None)
        }
        return _merge_params(
            flax.traverse_util.unflatten_dict(compatible, sep="/"),
            params,
            missing_regex=self.missing_regex,
        )
