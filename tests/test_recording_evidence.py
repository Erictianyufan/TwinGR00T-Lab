import hashlib

import numpy as np
import pytest

from twingr00t_lab.dataset import collapse_isaac_action
from twingr00t_lab.policy_bridge import groot_action_to_isaac
from twingr00t_lab.recording_evidence import action_sha256, video_evidence


def test_raw_and_projected_hashes_are_distinct_and_logical_limits_roundtrip():
    prediction = {
        "left_arm": np.full((1, 8, 6), 3.0),
        "left_gripper": np.full((1, 8, 1), 0.07),
        "right_arm": np.full((1, 8, 6), -4.0),
        "right_gripper": np.full((1, 8, 1), -0.02),
    }
    raw = groot_action_to_isaac(prediction)[0]
    lower = np.asarray([-2.0] * 6 + [-0.01] * 2 + [-2.0] * 6 + [-0.01] * 2, dtype=np.float32)
    upper = np.asarray([2.0] * 6 + [0.048] * 2 + [2.0] * 6 + [0.048] * 2, dtype=np.float32)
    clipped = np.clip(raw, lower, upper)
    bounds = collapse_isaac_action(np.stack((lower, upper)))
    expanded = np.concatenate(
        (bounds[:, :6], np.repeat(bounds[:, 6:7], 2, axis=1), bounds[:, 7:13], np.repeat(bounds[:, 13:14], 2, axis=1)),
        axis=1,
    )
    np.testing.assert_array_equal(expanded, np.stack((lower, upper)))
    np.testing.assert_array_equal(np.clip(raw, expanded[0], expanded[1]), clipped)
    assert action_sha256(raw) != action_sha256(clipped)
    assert action_sha256(raw) == hashlib.sha256(raw.tobytes(order="C")).hexdigest()
    assert action_sha256(raw) == action_sha256(np.asfortranarray(raw, dtype=np.float64))


@pytest.mark.parametrize("steps,every,expected", [(1, 2, 1), (4, 2, 2), (5, 2, 3), (25, 4, 7)])
def test_video_frame_accounting_includes_partial_final_interval(tmp_path, steps, every, expected):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"recorded video bytes")
    result = video_evidence(video, steps=steps, record_every=every, record_fps=25.0, frames_written=expected)
    assert result["video_frame_count"] == result["expected_video_frames"] == expected
    assert result["expected_video_duration_s"] == expected / 25.0
    assert result["video_sha256"] == hashlib.sha256(video.read_bytes()).hexdigest()
    video.write_bytes(b"truncated")
    assert (
        video_evidence(video, steps=steps, record_every=every, record_fps=25.0, frames_written=expected)["video_sha256"]
        != result["video_sha256"]
    )


def test_incomplete_pipe_writes_fail_before_hashing(tmp_path):
    with pytest.raises(ValueError, match="Wrote 2 video frames but expected 3"):
        video_evidence(tmp_path / "does_not_exist", steps=5, record_every=2, record_fps=25.0, frames_written=2)
