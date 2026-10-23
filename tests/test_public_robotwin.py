"""Pure-data checks for importing the public RoboTwin source domain."""

import numpy as np
import pytest

from twingr00t_lab.public_robotwin import archive_member, canonical_vectors


def test_public_qpos_is_measured_state_with_next_frame_target() -> None:
    qpos = np.zeros((3, 14), dtype=np.float32)
    qpos[:, 6] = [1, 0, 1]
    qpos[:, 13] = [0, 1, 0]
    qpos[:, 0] = [0.1, 0.3, 0.6]
    state, action = canonical_vectors(qpos)
    assert state.shape == action.shape == (2, 14)
    np.testing.assert_allclose(state[:, 0], [0.1, 0.3])
    np.testing.assert_allclose(action[:, 0], [0.3, 0.6])
    np.testing.assert_allclose(state[:, 6], [0.048, 0.006])
    np.testing.assert_allclose(action[:, 6], [-0.010, 0.048])
    np.testing.assert_allclose(state[:, 13], [0.006, 0.048])
    np.testing.assert_allclose(action[:, 13], [0.048, -0.010])


@pytest.mark.parametrize("shape", [(1, 14), (3, 13), (3, 16)])
def test_public_qpos_bad_shapes_rejected(shape: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="14-D"):
        canonical_vectors(np.zeros(shape, dtype=np.float32))


def test_public_qpos_nonfinite_and_bad_gripper_rejected() -> None:
    values = np.zeros((2, 14), dtype=np.float32)
    values[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        canonical_vectors(values)
    values[0, 0] = 0
    values[0, 6] = 1.5
    with pytest.raises(ValueError, match="normalized"):
        canonical_vectors(values)


def test_archive_member_only_exact_data_or_instruction() -> None:
    assert archive_member(49, "data") == "aloha-agilex_clean_50/data/episode49.hdf5"
    assert archive_member(0, "instructions") == "aloha-agilex_clean_50/instructions/episode0.json"
    with pytest.raises(ValueError):
        archive_member(0, "_traj_data")
    with pytest.raises(ValueError):
        archive_member(50, "data")
