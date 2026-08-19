from pathlib import Path

import pytest

from openpi.cotrain import config
from openpi.cotrain import fastwam_checkpoint as ckpt


def test_resolve_model_safetensors_file(tmp_path: Path) -> None:
    model = tmp_path / "model.safetensors"
    model.write_bytes(b"x")
    assert ckpt.resolve_model_safetensors(model) == model.resolve()


def test_resolve_model_safetensors_step_dir(tmp_path: Path) -> None:
    step = tmp_path / "20000"
    step.mkdir()
    (step / "model.safetensors").write_bytes(b"x")
    assert ckpt.resolve_model_safetensors(step) == (step / "model.safetensors").resolve()


def test_resolve_model_safetensors_exp_dir_picks_latest(tmp_path: Path) -> None:
    for step in ("10000", "20000"):
        d = tmp_path / step
        d.mkdir()
        (d / "model.safetensors").write_bytes(b"x")
    assert ckpt.resolve_model_safetensors(tmp_path).parent.name == "20000"


def test_find_latest_resume_dir_requires_metadata(tmp_path: Path) -> None:
    incomplete = tmp_path / "10000"
    incomplete.mkdir()
    (incomplete / "model.safetensors").write_bytes(b"x")
    complete = tmp_path / "20000"
    complete.mkdir()
    (complete / "model.safetensors").write_bytes(b"x")
    (complete / "metadata.pt").write_bytes(b"x")
    assert ckpt.find_latest_resume_dir(tmp_path) == complete


def test_wam_cross_piper_ft_config() -> None:
    cfg = config.get_config("wam-cross-piper-ft")
    assert {d.uid for d in cfg.data.datasets} == {"piper2", "piper30"}
    assert cfg.rlds_partition_builders_by_rank is False
    assert cfg.pytorch_weight_path is not None
    assert cfg.model.skip_dit_load_from_pretrain is True
    assert cfg.model.freeze_video_expert is True
    assert cfg.lr_schedule.peak_lr == pytest.approx(1.0e-5)
