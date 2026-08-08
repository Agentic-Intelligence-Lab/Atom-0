import torch

from openpi.models_pytorch import fastwam_pytorch


def test_robot_wrist_compose_output_shape() -> None:
    b, t = 2, 9
    head = torch.randn(b, 3, t, 480, 640)
    wrist = torch.randn(b, 3, t, 480, 640)
    out = fastwam_pytorch._compose_robot_wrist_video([head, wrist, wrist])
    assert out.shape == (b, 3, t, 576, 512)


def test_robot_wrist_compose_half_resolution() -> None:
    b, t = 2, 9
    head = torch.randn(b, 3, t, 480, 640)
    wrist = torch.randn(b, 3, t, 480, 640)
    out = fastwam_pytorch._compose_robot_wrist_video(
        [head, wrist, wrist], image_resolution=(288, 256)
    )
    assert out.shape == (b, 3, t, 288, 256)


def test_images_to_video_robot_wrist() -> None:
    b, t = 1, 3
    images = {
        "base_0_rgb": torch.randn(b, t, 480, 640, 3),
        "left_wrist_0_rgb": torch.randn(b, t, 480, 640, 3),
        "right_wrist_0_rgb": torch.randn(b, t, 480, 640, 3),
    }
    keys = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    out = fastwam_pytorch._images_to_video(images, keys, "robot_wrist")
    assert out.shape == (b, 3, t, 576, 512)


def test_images_to_video_robot_wrist_pre_resized() -> None:
    """TF pipeline may already resize each camera to compose targets before batch."""
    b, t = 1, 3
    images = {
        "base_0_rgb": torch.randn(b, t, 384, 512, 3),
        "left_wrist_0_rgb": torch.randn(b, t, 192, 256, 3),
        "right_wrist_0_rgb": torch.randn(b, t, 192, 256, 3),
    }
    keys = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    out = fastwam_pytorch._images_to_video(images, keys, "robot_wrist")
    assert out.shape == (b, 3, t, 576, 512)


def test_images_to_video_robot_wrist_pre_resized_half() -> None:
    b, t = 1, 3
    images = {
        "base_0_rgb": torch.randn(b, t, 192, 256, 3),
        "left_wrist_0_rgb": torch.randn(b, t, 96, 128, 3),
        "right_wrist_0_rgb": torch.randn(b, t, 96, 128, 3),
    }
    keys = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    out = fastwam_pytorch._images_to_video(
        images, keys, "robot_wrist", image_resolution=(288, 256)
    )
    assert out.shape == (b, 3, t, 288, 256)
