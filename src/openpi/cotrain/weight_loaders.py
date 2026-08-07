# """Weight loaders for co-training that read from explicit LOCAL paths.

# Avoids openpi's gs:// download + cache-path guessing (which breaks across ephemeral pods
# when OPENPI_DATA_HOME differs). Point directly at a file you already have on disk.
# """

# import dataclasses

# import flax.traverse_util
# import numpy as np

# import openpi.models.model as _model
# import openpi.shared.array_typing as at
# import openpi.shared.download as download
# from openpi.training.weight_loaders import _merge_params


# @dataclasses.dataclass(frozen=True)
# class LocalPaliGemmaWeightLoader:
#     """Load the PaliGemma big_vision `.npz` from a local path (no GCS).

#     Mirrors openpi's PaliGemmaWeightLoader but reads `npz_path` directly. The file must be
#     the big_vision JAX export (flat keys with '/' separators under a 'params' subtree).
#     """

#     npz_path: str

#     def load(self, params: at.Params) -> at.Params:
#         with open(self.npz_path, "rb") as f:
#             flat_params = dict(np.load(f, allow_pickle=False))
#         loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
#         # Add all missing weights (e.g. the randomly-initialized action expert).
#         return _merge_params(loaded_params, params, missing_regex=".*")


# @dataclasses.dataclass(frozen=True)
# class ShapeSafeCheckpointWeightLoader:
#     """Load a checkpoint, skipping keys whose shapes no longer match the target model.

#     This is used when widening the co-training action/state width (e.g. pi05_base has a
#     32-wide head while full-all uses 80). Matching pi05_base weights are loaded; widened
#     projection/head parameters stay at the target model's random initialization.
#     """

#     params_path: str
#     missing_regex: str = ".*"

#     def load(self, params: at.Params) -> at.Params:
#         loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
#         flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
#         flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")
#         compatible = {
#             key: value
#             for key, value in flat_loaded.items()
#             if key in flat_ref and getattr(value, "shape", None) == getattr(flat_ref[key], "shape", None)
#         }
#         return _merge_params(
#             flax.traverse_util.unflatten_dict(compatible, sep="/"),
#             params,
#             missing_regex=self.missing_regex,
#         )

"""Weight loaders for co-training that read from explicit LOCAL paths.

Avoids openpi's gs:// download + cache-path guessing (which breaks across ephemeral pods
when OPENPI_DATA_HOME differs). Point directly at a file you already have on disk.
"""

import dataclasses
import logging
from pathlib import Path

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download
from openpi.training.weight_loaders import _merge_params

logger = logging.getLogger(__name__)


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
    """Load either an Orbax checkpoint or a local PaliGemma big_vision NPZ.

    PARAMS_PATH is intentionally the only initialization switch used by the co-training
    launchers:

    * An Orbax params directory loads every shape-compatible VLA parameter.
    * A ``.npz`` file loads only the PaliGemma image and language backbone.

    In both cases, missing or shape-incompatible target parameters retain their random
    initialization. For PaliGemma NPZ initialization this includes the entire action
    expert, timestep MLP, and action input/output projections.
    """

    params_path: str
    missing_regex: str = ".*"

    def load(self, params: at.Params) -> at.Params:
        resolved_path = download.maybe_download(self.params_path)
        if resolved_path.is_file():
            loaded_params, source_kind = self._load_paligemma_npz(resolved_path)
        else:
            loaded_params = _model.restore_params(resolved_path, restore_type=np.ndarray)
            source_kind = "orbax-checkpoint"

        flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
        flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")
        compatible = {
            key: value
            for key, value in flat_loaded.items()
            if key in flat_ref and getattr(value, "shape", None) == getattr(flat_ref[key], "shape", None)
        }
        mismatched = {
            key
            for key, value in flat_loaded.items()
            if key in flat_ref and getattr(value, "shape", None) != getattr(flat_ref[key], "shape", None)
        }
        logger.info(
            "Initialization source=%s path=%s: loaded=%d random_or_missing=%d shape_mismatch=%d",
            source_kind,
            resolved_path,
            len(compatible),
            len(flat_ref) - len(compatible),
            len(mismatched),
        )
        return _merge_params(
            flax.traverse_util.unflatten_dict(compatible, sep="/"),
            params,
            missing_regex=self.missing_regex,
        )

    @staticmethod
    def _load_paligemma_npz(path: Path) -> tuple[at.Params, str]:
        if path.suffix != ".npz":
            raise ValueError(
                f"Unsupported PARAMS_PATH file: {path}. Expected a PaliGemma big_vision .npz "
                "or an Orbax params directory."
            )
        with path.open("rb") as file:
            flat_params = dict(np.load(file, allow_pickle=False))
        try:
            paligemma_params = flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]
        except KeyError as error:
            raise ValueError(f"Invalid PaliGemma NPZ {path}: missing params/ root") from error
        if not {"img", "llm"}.issubset(paligemma_params):
            raise ValueError(f"Invalid PaliGemma NPZ {path}: expected params/img and params/llm")
        return {"PaliGemma": paligemma_params}, "paligemma-vlm-npz"