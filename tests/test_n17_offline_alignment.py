import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def offline_fixture(tmp_path, monkeypatch):
    h5py = pytest.importorskip("h5py")
    pytest.importorskip("PIL")
    spec = importlib.util.spec_from_file_location(
        "diagnose_n17_offline", Path(__file__).resolve().parents[1] / "scripts/diagnose_n17_offline.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "frame_rgb", lambda dataset, index, **kwargs: dataset[index])
    path = tmp_path / "episode.hdf5"
    with h5py.File(path, "w") as handle:
        policy = handle.create_group("observations/policy")
        for key, dim in (("left_arm_pos", 6), ("left_gripper_pos", 1), ("right_arm_pos", 6), ("right_gripper_pos", 1)):
            policy[key] = np.tile(np.arange(19)[:, None], (1, dim)).astype(np.float32)
        policy["actions"] = np.tile(np.arange(19)[:, None], (1, 16)).astype(np.float32)
        cameras = handle.create_group("observations/camera_obs")
        for camera in ("head_camera_rgb", "left_camera_rgb", "right_camera_rgb"):
            cameras[camera] = np.tile(np.arange(19, dtype=np.uint8)[:, None, None, None], (1, 2, 2, 3))

    class Policy:
        def __init__(self, offset):
            self.offset = offset
            self.calls = []

        def get_action(self, observation):
            index = int(observation["state"]["left_arm"][0, 0, 0])
            assert observation["video"]["head"][0, 0, 0, 0, 0] == index
            self.calls.append(index)
            values = index + self.offset + np.arange(16, dtype=np.float32)
            return {
                key: np.tile(values[None, :, None], (1, 1, dim))
                for key, dim in (("left_arm", 6), ("left_gripper", 1), ("right_arm", 6), ("right_gripper", 1))
            }, {}

    return module, path, Policy


def test_shifted_teacher_matches_nextaction_and_exact_end_boundary(offline_fixture):
    module, path, policy_type = offline_fixture
    policy = policy_type(1)
    rows = module.evaluate_episode(policy, path, "sort", [0, 2, 3, 19], observation_action_offset=1)
    assert policy.calls == [0, 2]
    assert [row["target_frame"] for row in rows] == [1, 3]
    assert all(row["mae_14d"] == 0 and row["compared_horizon"] == 16 for row in rows)
    assert rows[-1]["observation_action_offset"] == 1


def test_default_keeps_legacy_targets_and_reveals_unshifted_error(offline_fixture):
    module, path, policy_type = offline_fixture
    matched = module.evaluate_episode(policy_type(0), path, "sort", [3])
    mismatched = module.evaluate_episode(policy_type(1), path, "sort", [3])
    assert matched[0]["target_frame"] == 3
    assert matched[0]["mae_14d"] == 0
    assert mismatched[0]["mae_14d"] == 1


def test_default_repeat_does_not_add_calls_or_claim_variance_estimate(offline_fixture):
    module, path, policy_type = offline_fixture
    policy = policy_type(0)
    row = module.evaluate_episode(policy, path, "sort", [0])[0]
    assert policy.calls == [0]
    assert row["repeat_count"] == 1
    assert row["variance_estimated"] is False
    assert row["prediction_variance_14d"] == [0] * 14
    assert row["prediction_std_14d"] == [0] * 14


def test_grouped_errors_separate_units_and_first_step_from_horizon(offline_fixture):
    module, path, policy_type = offline_fixture

    class ErrorPolicy(policy_type):
        def get_action(self, observation):
            predictions, meta = super().get_action(observation)
            for name, values in predictions.items():
                values += 2 if "arm" in name else 0.25
                values[:, 0] += 4 if "arm" in name else 0.5
            return predictions, meta

    row = module.evaluate_episode(ErrorPolicy(0), path, "sort", [0])[0]
    assert row["arm_mae_rad"] == pytest.approx(2.25)
    assert row["gripper_mae_m"] == pytest.approx(0.28125)
    assert row["first_step_arm_mae_rad"] == pytest.approx(6)
    assert row["first_step_gripper_mae_m"] == pytest.approx(0.75)
    assert row["first_step_mae_14d"] == pytest.approx((6 * 12 + 0.75 * 2) / 14)
    assert row["first_step_abs_error_14d"] == [6] * 6 + [0.75] + [6] * 6 + [0.75]


def test_repeats_measure_spread_without_cancelling_opposite_errors(offline_fixture):
    module, path, policy_type = offline_fixture

    class VaryingPolicy(policy_type):
        def get_action(self, observation):
            predictions, meta = super().get_action(observation)
            sign = 1 if len(self.calls) % 2 else -1
            for name, values in predictions.items():
                values += sign * (2 if "arm" in name else 0.25)
            return predictions, meta

    policy = VaryingPolicy(1)
    row = module.evaluate_episode(policy, path, "sort", [0], observation_action_offset=1, repeats=2)[0]
    assert policy.calls == [0, 0]
    assert row["repeat_count"] == 2
    assert row["variance_estimated"] is True
    assert row["arm_mae_rad"] == pytest.approx(2)
    assert row["gripper_mae_m"] == pytest.approx(0.25)
    assert row["prediction_std_arm_rad"] == pytest.approx(2)
    assert row["prediction_std_gripper_m"] == pytest.approx(0.25)
    assert row["prediction_variance_14d"] == [4] * 6 + [0.0625] + [4] * 6 + [0.0625]
    assert row["first_step_prediction_std_14d"] == [2] * 6 + [0.25] + [2] * 6 + [0.25]
    assert row["first_prediction"] == [3] * 6 + [1.25] + [3] * 6 + [1.25]


@pytest.mark.parametrize("repeats", [0, -1, 1.5, True])
def test_invalid_repeats_never_call_policy(offline_fixture, repeats):
    module, path, policy_type = offline_fixture
    policy = policy_type(0)
    with pytest.raises(ValueError, match="positive integer"):
        module.evaluate_episode(policy, path, "sort", [0], repeats=repeats)
    assert policy.calls == []


@pytest.mark.parametrize(
    "indices,offset,error", [([0], -1, "non-negative"), ([-1], 0, "non-negative"), ([0], 19, "no paired")]
)
def test_invalid_offsets_and_indices_never_call_policy(offline_fixture, indices, offset, error):
    module, path, policy_type = offline_fixture
    policy = policy_type(0)
    with pytest.raises(ValueError, match=error):
        module.evaluate_episode(policy, path, "sort", indices, observation_action_offset=offset)
    assert policy.calls == []
