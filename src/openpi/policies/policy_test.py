import flax.nnx as nnx
import jax.numpy as jnp
from openpi_client import action_chunk_broker
import numpy as np
import pytest

from openpi.policies import aloha_policy
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


class _DummyLongMemoryModel(nnx.Module):
    action_horizon = 1
    action_dim = 2
    long_memory_enabled = True
    memory_update_interval_steps = 100
    memory_generation_max_new_tokens = 1

    def sample_actions(self, rng, observation, **kwargs):
        return jnp.zeros((1, 1, 2), dtype=jnp.float32)


class _RejectResetKeysTransform:
    def __call__(self, data):
        assert "episode_start" not in data
        assert "reset_memory" not in data
        return data


def test_long_memory_resets_on_episode_boundary_signal():
    policy = _policy.Policy(
        _DummyLongMemoryModel(),
        transforms=(_RejectResetKeysTransform(),),
        output_transforms=(),
        memory_summary_tokenizer=None,
    )
    policy.memory_summary = "stale memory from previous episode"
    policy.memory_step = 42

    obs = {
        "episode_start": True,
        "image": {},
        "image_mask": {},
        "state": np.zeros((2,), dtype=np.float32),
    }
    result = policy.infer(obs)

    assert policy.get_memory_summary() == ""
    assert result["memory_summary"] == ""
    assert result["memory_step"] == 0


@pytest.mark.manual
def test_infer():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    example = aloha_policy.make_aloha_example()
    result = policy.infer(example)

    assert result["actions"].shape == (config.model.action_horizon, 14)


@pytest.mark.manual
def test_broker():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    broker = action_chunk_broker.ActionChunkBroker(
        policy,
        # Only execute the first half of the chunk.
        action_horizon=config.model.action_horizon // 2,
    )

    example = aloha_policy.make_aloha_example()
    for _ in range(config.model.action_horizon):
        outputs = broker.infer(example)
        assert outputs["actions"].shape == (14,)
