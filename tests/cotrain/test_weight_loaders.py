from pathlib import Path

import numpy as np
import pytest

from openpi.cotrain import weight_loaders
from openpi.models import model


def _target_params() -> dict:
    return {
        "PaliGemma": {
            "img": {"kernel": np.zeros((2, 2), dtype=np.float32)},
            "llm": {"kernel": np.zeros((2, 3), dtype=np.float32)},
        },
        "action_out_proj": {"kernel": np.full((3, 4), 7.0, dtype=np.float32)},
    }


def _write_paligemma_npz(path: Path, *, include_llm: bool = True) -> None:
    arrays = {
        "params/img/kernel": np.full((2, 2), 1.0, dtype=np.float32),
    }
    if include_llm:
        arrays["params/llm/kernel"] = np.full((2, 3), 2.0, dtype=np.float32)
    np.savez(path, **arrays)


def test_shape_safe_loader_auto_detects_paligemma_npz(tmp_path: Path) -> None:
    path = tmp_path / "pt_224.npz"
    _write_paligemma_npz(path)

    loaded = weight_loaders.ShapeSafeCheckpointWeightLoader(str(path)).load(_target_params())

    np.testing.assert_array_equal(loaded["PaliGemma"]["img"]["kernel"], np.full((2, 2), 1.0))
    np.testing.assert_array_equal(loaded["PaliGemma"]["llm"]["kernel"], np.full((2, 3), 2.0))
    # Parameters absent from the VLM checkpoint retain the target model's random initialization.
    np.testing.assert_array_equal(loaded["action_out_proj"]["kernel"], np.full((3, 4), 7.0))


def test_shape_safe_loader_retains_random_target_on_npz_shape_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "pt_224.npz"
    np.savez(
        path,
        **{
            "params/img/kernel": np.full((2, 2), 1.0, dtype=np.float32),
            "params/llm/kernel": np.full((9, 9), 2.0, dtype=np.float32),
        },
    )

    loaded = weight_loaders.ShapeSafeCheckpointWeightLoader(str(path)).load(_target_params())

    np.testing.assert_array_equal(loaded["PaliGemma"]["img"]["kernel"], np.full((2, 2), 1.0))
    np.testing.assert_array_equal(loaded["PaliGemma"]["llm"]["kernel"], np.zeros((2, 3)))


def test_shape_safe_loader_rejects_non_paligemma_npz(tmp_path: Path) -> None:
    path = tmp_path / "invalid.npz"
    _write_paligemma_npz(path, include_llm=False)

    with pytest.raises(ValueError, match="expected params/img and params/llm"):
        weight_loaders.ShapeSafeCheckpointWeightLoader(str(path)).load(_target_params())


def test_shape_safe_loader_preserves_orbax_checkpoint_behavior(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoint_dir = tmp_path / "params"
    checkpoint_dir.mkdir()
    checkpoint_params = _target_params()
    checkpoint_params["action_out_proj"]["kernel"] = np.full((3, 4), 5.0, dtype=np.float32)
    monkeypatch.setattr(
        model,
        "restore_params",
        lambda path, restore_type: checkpoint_params,
    )

    loaded = weight_loaders.ShapeSafeCheckpointWeightLoader(str(checkpoint_dir)).load(_target_params())

    np.testing.assert_array_equal(loaded["action_out_proj"]["kernel"], np.full((3, 4), 5.0))
