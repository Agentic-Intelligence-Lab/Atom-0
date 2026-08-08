"""Unit tests for URDF FK EEF fill."""

import numpy as np

from openpi.cotrain import fk_eef


def test_validate_registry_reports_known_mismatches() -> None:
    ok = fk_eef.validate_fk_spec(fk_eef.FK_EEF_SPECS["droid"])
    assert ok.ok
    assert ok.mapped_arm_dofs == (7,)

    bad = fk_eef.validate_fk_spec(fk_eef.FK_EEF_SPECS["robomind_tienkung_gello_s16_a16"])
    assert not bad.ok
    assert bad.mapped_arm_dofs == (7, 7)
    assert bad.urdf_arm_dofs == (4, 4)


def test_fk_fill_writes_right_eef_for_droid() -> None:
    spec = fk_eef.FK_EEF_SPECS["droid"]
    state = np.zeros(80, dtype=np.float64)
    state[29:36] = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7]
    actions = np.zeros((3, 80), dtype=np.float64)
    actions[:] = state
    filled_state, filled_actions = fk_eef.fill_eef_from_fk(state, actions, spec)
    assert filled_actions is not None
    assert np.linalg.norm(filled_state[36:42]) > 1e-3
    np.testing.assert_allclose(filled_actions[0, 36:42], filled_state[36:42])
    # Arm joints unchanged.
    np.testing.assert_allclose(filled_state[29:36], state[29:36])


def test_enabled_set_excludes_dof_mismatches() -> None:
    enabled = set(fk_eef.enabled_fk_dataset_ids())
    assert "droid" in enabled
    assert "piper30" in enabled
    assert "piper2" in enabled
    assert "robomind_tienkung_gello_s16_a16" not in enabled
    assert "robocoin_galaxea_r1_lite_s16_a18" not in enabled


def test_validate_piper_registry() -> None:
    for dataset_id in ("piper30", "piper2"):
        result = fk_eef.validate_fk_spec(fk_eef.FK_EEF_SPECS[dataset_id])
        assert result.ok, (dataset_id, result.messages)


def test_fk_arm_batch_matches_scalar_fk() -> None:
    spec = fk_eef.FK_EEF_SPECS["droid"]
    arm = spec.arms[0]
    urdf_path = fk_eef._DEFAULT_URDF_DIR / spec.urdf_file
    model = fk_eef._cached_urdf(str(urdf_path))
    plan = fk_eef._arm_fk_plan(str(urdf_path), arm.ee_link, arm.joint_names)
    rng = np.random.default_rng(0)
    q = rng.normal(size=(16, arm.dof))
    batch_eef = fk_eef._fk_arm_batch(q, plan)
    for i in range(q.shape[0]):
        q_by_name = {name: float(q[i, j]) for j, name in enumerate(arm.joint_names)}
        pose = fk_eef.fk_link_pose(model, arm.ee_link, q_by_name)
        scalar = fk_eef.pose_to_xyz_yaw_pitch_roll(pose)
        np.testing.assert_allclose(batch_eef[i], scalar, rtol=1e-10, atol=1e-10)


def test_fill_eef_vectors_batch_matches_per_vector() -> None:
    spec = fk_eef.FK_EEF_SPECS["droid"]
    base = np.zeros(80, dtype=np.float64)
    base[29:36] = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7]
    single = fk_eef.fill_eef_vectors_batch(base, spec)
    stacked = np.stack([base, base * 0.5], axis=0)
    batched = fk_eef.fill_eef_vectors_batch(stacked, spec)
    np.testing.assert_allclose(batched[0], single)
    chunked = np.broadcast_to(base, (2, 3, 80)).copy()
    chunked[1, :, 29:36] *= 0.25
    filled = fk_eef.fill_eef_vectors_batch(chunked, spec)
    for b in range(2):
        for t in range(3):
            ref = fk_eef.fill_eef_vectors_batch(chunked[b, t], spec)
            np.testing.assert_allclose(filled[b, t], ref)
