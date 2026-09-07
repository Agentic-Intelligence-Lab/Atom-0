import json
from pathlib import Path

import numpy as np
import torch

import openpi.models.fastwam_config as fastwam_config
import openpi.models.model as _model
from openpi.cotrain import fastwam_vae_input_debug


def _robot_wrist_config() -> fastwam_config.FastWAMConfig:
    return fastwam_config.FastWAMConfig(
        camera_keys=("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"),
        concat_multi_camera="robot_wrist",
        image_resolution=(576, 512),
        video_num_frames=9,
    )


def test_compose_vae_video_robot_wrist_shape() -> None:
    b, t = 2, 9
    images = {
        "base_0_rgb": torch.rand(b, t, 384, 512, 3) * 2 - 1,
        "left_wrist_0_rgb": torch.rand(b, t, 192, 256, 3) * 2 - 1,
        "right_wrist_0_rgb": torch.rand(b, t, 192, 256, 3) * 2 - 1,
    }
    observation = _model.Observation(
        images=images,
        image_masks={k: torch.ones(b, t, dtype=torch.bool) for k in images},
        state=torch.zeros(b, 80),
        tokenized_prompt=None,
        tokenized_prompt_mask=None,
        token_ar_mask=None,
        token_loss_mask=None,
    )
    video = fastwam_vae_input_debug.compose_vae_video_from_observation(
        observation,
        _robot_wrist_config(),
    )
    assert video.shape == (b, 3, t, 576, 512)


def test_save_vae_input_sample_writes_png_and_meta(tmp_path: Path) -> None:
    video = torch.rand(2, 3, 5, 576, 512) * 2 - 1
    paths = fastwam_vae_input_debug.save_vae_input_sample(
        video,
        batch_idx=0,
        sample_idx=1,
        output_dir=tmp_path,
        meta={"dataset_id": "piper30", "prompt": "pick cup"},
    )
    pngs = [p for p in paths if p.suffix == ".png"]
    assert pngs
    assert all(p.exists() for p in pngs)
    meta_path = tmp_path / "batch000_sample01_meta.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["dataset_id"] == "piper30"
    assert meta["saved_frames"]
