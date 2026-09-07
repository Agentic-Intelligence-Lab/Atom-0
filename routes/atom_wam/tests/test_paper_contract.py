"""Paper-interface tests using tiny real DiTs and a test-only latent encoder.

No pretrained weights, datasets, GPU, or full VAE are required.
"""
from types import SimpleNamespace

import pytest
import torch

from openpi.models_pytorch.fastwam.wan22.action_dit import ActionDiT
from openpi.models_pytorch.fastwam.wan22.fastwam import FastWAM
from openpi.models_pytorch.fastwam.wan22.mot import MoT
from openpi.models_pytorch.fastwam.wan22.wan_video_dit import WanVideoDiT


class LatentEncoder(torch.nn.Module):
    temporal_downsample_factor = 4

    def encode(self, video, **kwargs):
        if isinstance(video, list):
            assert video[0].shape[1] == 1
            return [torch.zeros(4, 1, 4, 4)]
        return torch.zeros(video.shape[0], 4, (video.shape[2] - 1) // 4 + 1, 4, 4)

    def decode(self, *args, **kwargs):
        raise AssertionError("Action-only inference must not decode future video")


@pytest.fixture
def model():
    # Wan's three-axis RoPE requires an even allocation on each axis.
    shared = dict(hidden_dim=48, ffn_dim=64, text_dim=16, freq_dim=16,
                  eps=1e-6, num_heads=2, attn_head_dim=24, num_layers=2)
    video = WanVideoDiT(
        **shared, in_dim=4, out_dim=4, patch_size=(1, 2, 2),
        has_image_input=False, seperated_timestep=True,
        fuse_vae_embedding_in_latents=True, video_attention_mask_mode="first_frame_causal",
    )
    action = ActionDiT(**shared, action_dim=80)
    return FastWAM(video, action, MoT({"video": video, "action": action}, False),
                   LatentEncoder(), text_dim=16, proprio_dim=None)


def test_50_step_training_backward(model):
    sample = dict(video=torch.zeros(2, 3, 9, 32, 32), action=torch.randn(2, 50, 80),
                  context=torch.randn(2, 4, 16), context_mask=torch.ones(2, 4, dtype=torch.bool),
                  action_mask=torch.ones(2, 80, dtype=torch.bool), is_ego=torch.tensor([True, False]))
    loss, _ = model.training_loss(sample)
    loss.backward()
    assert torch.isfinite(loss)
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert model.proprio_encoder is None


def test_sampling_probability_is_not_rebalanced():
    values = torch.tensor([2., 4., 6., 20.], requires_grad=True)
    ego = torch.tensor([True, True, True, False])
    weights = torch.ones(4)
    ego_part = FastWAM._masked_domain_mean(values, ego, weights)
    robot_part = FastWAM._masked_domain_mean(values, ~ego, weights)
    torch.testing.assert_close(ego_part, torch.tensor(3.))
    torch.testing.assert_close(robot_part, torch.tensor(5.))
    torch.testing.assert_close(ego_part + robot_part, values.mean())
    (ego_part + robot_part).backward()
    torch.testing.assert_close(values.grad, torch.full((4,), .25))
    assert FastWAM._masked_domain_mean(values, torch.zeros(4, dtype=torch.bool), weights) == 0


def test_no_direct_or_multilayer_future_leakage(model):
    # Nine raw frames become three latent frames; four tokens per latent frame.
    mask = model._build_mot_attention_mask(12, 50, 4, torch.device("cpu"))
    assert mask[12:, :4].all()
    assert not mask[12:, 4:12].any()
    assert not mask[:4, 4:].any()  # Clean anchor cannot relay future or action tokens.
    reachability = mask.clone()
    for _ in range(3):
        reachability |= (reachability.float() @ mask.float()) > 0
    assert not reachability[12:, 4:12].any()
    assert mask[4:8, 12:37].all() and not mask[4:8, 37:].any()
    assert mask[8:12, 37:].all() and not mask[8:12, 12:37].any()


def test_inference_only_generates_50_by_80_actions(model):
    mask = torch.ones(80, dtype=torch.bool)
    mask[75:] = False
    result = model.infer_action(
        prompt=None, input_image=torch.zeros(3, 32, 32), action_horizon=50,
        context=torch.zeros(1, 4, 16), context_mask=torch.ones(1, 4, dtype=torch.bool),
        action_mask=mask, num_inference_steps=2, seed=0,
    )
    assert set(result) == {"action"}
    assert result["action"].shape == (50, 80)
    assert torch.isfinite(result["action"]).all()
    assert not result["action"][:, 75:].any()


def test_wrapper_ignores_state_and_uses_training_resolution(monkeypatch):
    from openpi.models.fastwam_config import FastWAMConfig
    from openpi.models_pytorch import fastwam_pytorch as wrapper

    class ActionOnly(torch.nn.Module):
        device = torch.device("cpu")
        torch_dtype = torch.float32

        def __init__(self):
            super().__init__()
            self.dit = torch.nn.Identity()
            self.calls = []

        def infer_action(self, **kwargs):
            self.calls.append(kwargs)
            assert kwargs["proprio"] is None
            return {"action": torch.zeros(50, 80)}

    backend = ActionOnly()
    monkeypatch.setattr(wrapper, "create_fastwam", lambda **kwargs: backend)
    cfg = FastWAMConfig(action_dim=80, action_horizon=50, proprio_dim=None,
                        camera_keys=("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"),
                        concat_multi_camera="robot_wrist", image_resolution=(288, 256))
    policy = wrapper.FastWAMPytorch(cfg, device="cpu")
    observation = SimpleNamespace(
        images={key: torch.zeros(1, 9, 32, 32, 3) for key in cfg.camera_keys},
        image_masks={}, state=None, action_mask=None,
        context=torch.zeros(1, 4, 16), context_mask=torch.ones(1, 4, dtype=torch.bool),
    )
    sample = policy.observation_to_sample(observation, torch.zeros(1, 50, 80))
    assert "proprio" not in sample
    assert sample["video"].shape == (1, 3, 9, 288, 256)
    first = policy.sample_actions("cpu", observation)
    observation.state = torch.full((1, 80), float("nan"))
    second = policy.sample_actions("cpu", observation)
    torch.testing.assert_close(first, second)
    assert first.shape == (1, 50, 80)
    assert all(call["input_image"].shape == (3, 288, 256) for call in backend.calls)
