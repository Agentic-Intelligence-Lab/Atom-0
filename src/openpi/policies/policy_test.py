import os
from openpi_client import action_chunk_broker
import pytest

from openpi.policies import aloha_policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

# NOTE: Long-term memory episode-reset behavior moved from the action Policy to the
# high-level policy server (HLInferenceServer); its test moves there in Phase 6/7.


@pytest.mark.manual
@pytest.mark.skipif(os.environ.get("OPENPI_RUN_MANUAL_TESTS") != "1", reason="Set OPENPI_RUN_MANUAL_TESTS=1.")
def test_infer():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    example = aloha_policy.make_aloha_example()
    result = policy.infer(example)

    assert result["actions"].shape == (config.model.action_horizon, 14)


@pytest.mark.manual
@pytest.mark.skipif(os.environ.get("OPENPI_RUN_MANUAL_TESTS") != "1", reason="Set OPENPI_RUN_MANUAL_TESTS=1.")
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
