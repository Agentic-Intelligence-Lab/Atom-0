"""Weight loaders for co-training that read from explicit LOCAL paths.

Avoids openpi's gs:// download + cache-path guessing (which breaks across ephemeral pods
when OPENPI_DATA_HOME differs). Point directly at a file you already have on disk.
"""

import dataclasses

import flax.traverse_util
import numpy as np

import openpi.shared.array_typing as at
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
