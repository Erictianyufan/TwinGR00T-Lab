import numpy as np
import pytest

from twingr00t_lab.home_demo import DemoPhase, HomeDemoTimeline


def test_phase_boundaries_and_action_indices() -> None:
    timeline = HomeDemoTimeline()
    assert timeline.phase(0) is DemoPhase.ESTABLISH
    assert timeline.phase(timeline.transport_start) is DemoPhase.TRANSPORT
    assert timeline.phase(timeline.handoff_start) is DemoPhase.HANDOFF
    assert timeline.phase(timeline.stack_start) is DemoPhase.STACK
    assert timeline.phase(timeline.complete_start) is DemoPhase.COMPLETE
    assert timeline.aloha_action_index(timeline.stack_start) == 0
    assert timeline.aloha_action_index(timeline.complete_start - 1) == 599
    assert timeline.aloha_action_index(timeline.complete_start) == 599


def test_cargo_finishes_exactly_at_recorded_bowl_origin() -> None:
    timeline = HomeDemoTimeline()
    assert timeline.cargo_position(0)[2] == pytest.approx(0.0)
    assert timeline.cargo_position(timeline.transport_start)[2] == pytest.approx(0.24)
    final_handoff = timeline.stack_start - 1
    np.testing.assert_allclose(timeline.cargo_position(final_handoff), np.zeros(3), atol=1e-6)
    assert timeline.cargo_position(timeline.stack_start)[2] < -1.0


def test_humanoid_retreat_is_continuous() -> None:
    timeline = HomeDemoTimeline()
    before = timeline.humanoid_position(timeline.stack_start - 1)
    after = timeline.humanoid_position(timeline.stack_start)
    np.testing.assert_allclose(before, after, atol=1e-6)
    assert timeline.humanoid_position(timeline.stack_start + 80)[0] < before[0]


def test_invalid_frame_and_durations_are_rejected() -> None:
    timeline = HomeDemoTimeline()
    with pytest.raises(IndexError):
        timeline.phase(timeline.total_frames)
    with pytest.raises(ValueError):
        HomeDemoTimeline(transport_frames=0)


def test_camera_choreography_is_continuous_at_phase_boundaries() -> None:
    timeline = HomeDemoTimeline()
    for boundary in (timeline.handoff_start, timeline.stack_start, timeline.complete_start):
        eye_before, target_before = timeline.overview_camera_pose(boundary - 1)
        eye_after, target_after = timeline.overview_camera_pose(boundary)
        assert np.linalg.norm(eye_after - eye_before) < 0.1
        assert np.linalg.norm(target_after - target_before) < 0.1

    pickup_eye, _ = timeline.overview_camera_pose(0)
    stack_eye, _ = timeline.overview_camera_pose(timeline.stack_start + 100)
    assert pickup_eye[0] < stack_eye[0]


def test_pickup_has_reach_grip_lift_order_and_closeup() -> None:
    timeline = HomeDemoTimeline()
    early = timeline.pickup_progress(12)
    grip = timeline.pickup_progress(54)
    lifted = timeline.pickup_progress(timeline.establish_frames - 1)
    assert early == pytest.approx((0.0, 0.0, 0.0))
    assert grip[0] > grip[1] > grip[2]
    assert lifted == pytest.approx((1.0, 1.0, 1.0))
    assert timeline.pickup_camera_active(54)
    assert timeline.pickup_camera_active(100)
    assert not timeline.pickup_camera_active(timeline.transport_start)

    wide_eye, _ = timeline.overview_camera_pose(0)
    close_eye, _ = timeline.overview_camera_pose(54)
    carrier = timeline.humanoid_position(54)
    assert np.linalg.norm(close_eye - carrier) < np.linalg.norm(wide_eye - carrier)
