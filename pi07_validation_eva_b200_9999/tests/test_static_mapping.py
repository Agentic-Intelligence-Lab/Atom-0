from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_validation as ev


def test_piper_14d_to_unified_80d_slots():
    assert ev.UNIFIED_ACTION_DIM == 80
    assert ev.UNIFIED_PIPER_DIMS == (
        0,
        1,
        2,
        3,
        4,
        5,
        16,
        29,
        30,
        31,
        32,
        33,
        34,
        45,
    )


def test_piper_joint_delta_dims_are_arm_joints_only():
    assert ev.JOINT_DELTA_DIMS == (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12)
    assert ev.GRIPPER_DIMS == (6, 13)


def test_eval_mapping_matches_training_registry():
    from openpi.cotrain import action_space

    spec = action_space.UNIFIED_ACTION_SPECS["piper30"]
    assert tuple(target for _, target in spec.action_mapping) == ev.UNIFIED_PIPER_DIMS
    assert tuple(np.flatnonzero(spec.action_mask)) == ev.UNIFIED_PIPER_DIMS


def _episode_fixture():
    states = np.arange(28, dtype=np.float32).reshape(2, 14)
    return {
        "states": states,
        "actions": states + 10,
        "prompts": ["pick the cube", "pick the cube"],
        "images": {
            "cam_high": [np.zeros((2, 2, 3), dtype=np.uint8)] * 2,
            "cam_left_wrist": [np.zeros((2, 2, 3), dtype=np.uint8)] * 2,
            "cam_right_wrist": [np.zeros((2, 2, 3), dtype=np.uint8)] * 2,
        },
    }


def test_policy_observation_matches_piper_training_contract():
    obs = ev.standardized_policy_observation(_episode_fixture(), 0, ev.UNIFIED_ACTION_DIM)

    assert obs["state"].shape == (80,)
    np.testing.assert_array_equal(obs["state"][list(ev.UNIFIED_PIPER_DIMS)], np.arange(14, dtype=np.float32))
    assert obs["action_mask"].dtype == np.bool_
    assert tuple(np.flatnonzero(obs["action_mask"])) == ev.UNIFIED_PIPER_DIMS
    assert obs["prompt"] == "pick the cube"
    assert obs["prompt_prefix"] == "Action Mode: joint. "
    # The policy receives explicit piper30 norm stats. Keeping dataset_id out of inference
    # avoids DispatchNormalize followed by the generic policy Normalize a second time.
    assert "dataset_id" not in obs


def test_policy_state_is_normalized_exactly_once_like_training():
    from openpi import transforms
    from openpi.cotrain import action_space
    from openpi.cotrain import transforms as cotrain_transforms
    from openpi.models import model
    from openpi.shared import normalize

    repo_root = ROOT.parent
    norm_stats = normalize.deserialize_json(
        (repo_root / "assets/cotrain_real_only/piper30/norm_stats.json").read_text()
    )
    spec = action_space.UNIFIED_ACTION_SPECS["piper30"]
    training_inputs = [
        cotrain_transforms.StandardizedInputs(model_type=model.ModelType.PI05),
        cotrain_transforms.DispatchDeltaActions(masks_by_dataset={"piper30": spec.delta_mask}),
        cotrain_transforms.DispatchNormalize(
            norm_stats_by_dataset={"piper30": norm_stats},
            use_quantiles=True,
        ),
    ]

    episode = _episode_fixture()
    training_sample = ev.standardized_training_sample(episode, 0, horizon=1, policy_state_dim=80)
    training_pre_model = transforms.compose(training_inputs)(training_sample)

    policy_obs = ev.standardized_policy_observation(episode, 0, policy_state_dim=80)
    policy_pre_model = transforms.compose(
        [
            *training_inputs,
            transforms.Normalize(norm_stats, use_quantiles=True),
        ]
    )(policy_obs)

    np.testing.assert_allclose(policy_pre_model["state"], training_pre_model["state"])
    np.testing.assert_array_equal(policy_pre_model["action_mask"], training_pre_model["action_mask"])
    assert policy_pre_model["prompt"] == training_pre_model["prompt"]
    assert policy_pre_model["prompt_prefix"] == training_pre_model["prompt_prefix"]


def test_unified_prediction_restores_native_absolute_piper_action():
    state = np.arange(14, dtype=np.float32)
    native = np.arange(14, dtype=np.float32) / 10
    unified = ev.piper_actions_to_unified(native[None, :])

    restored = ev.native_to_absolute_actions(unified, state, actions_per_inference=1)[0]
    expected = native.copy()
    expected[list(ev.JOINT_DELTA_DIMS)] += state[list(ev.JOINT_DELTA_DIMS)]
    np.testing.assert_allclose(restored, expected)
    np.testing.assert_allclose(restored[list(ev.GRIPPER_DIMS)], native[list(ev.GRIPPER_DIMS)])


def test_anchor_indices_uniformly_cover_episode_instead_of_only_prefix():
    assert ev.anchor_indices(num_steps=101, stride=1, limit=5) == [0, 25, 50, 75, 100]
    assert ev.anchor_indices(num_steps=101, stride=2, limit=3) == [0, 50, 100]
    assert ev.anchor_indices(num_steps=100, stride=1, limit=1) == [50]
