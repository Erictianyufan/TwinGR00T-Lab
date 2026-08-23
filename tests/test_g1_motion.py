import numpy as np
import pytest

from twingr00t_lab.g1_motion import G1MotionPlan, apply_named_offsets
from twingr00t_lab.home_demo import DemoPhase, HomeDemoTimeline
from twingr00t_lab.multi_robot import MultiRobotCoordinator, RobotRole


def test_g1_has_visible_opposed_leg_gait_during_transport() -> None:
    timeline = HomeDemoTimeline()
    plan = G1MotionPlan()
    samples = [plan.sample(timeline.transport_start + index, timeline) for index in range(20, 60)]
    hip_delta = [
        sample.joint_offsets["left_hip_pitch_joint"] - sample.joint_offsets["right_hip_pitch_joint"]
        for sample in samples
    ]
    assert max(hip_delta) - min(hip_delta) > 0.6
    assert all(sample.gait_active for sample in samples)


def test_g1_stands_still_after_completion() -> None:
    timeline = HomeDemoTimeline()
    sample = G1MotionPlan().sample(timeline.complete_start, timeline)
    assert not sample.gait_active
    assert sample.joint_offsets["left_hip_pitch_joint"] == pytest.approx(0.0)
    assert sample.root_position[2] == pytest.approx(0.74)


def test_named_offsets_are_joint_order_independent() -> None:
    result = apply_named_offsets(
        np.asarray((1.0, 2.0, 3.0)),
        ("knee", "hip", "ankle"),
        {"hip": 0.5, "missing": 100.0},
    )
    np.testing.assert_allclose(result, (1.0, 2.5, 3.0))
    with pytest.raises(ValueError):
        apply_named_offsets(np.zeros(2), ("only_one",), {})


def test_policy_coordinator_routes_continuous_task_by_embodiment() -> None:
    coordinator = MultiRobotCoordinator()
    assert coordinator.route(DemoPhase.TRANSPORT).role is RobotRole.UNITREE_G1
    assert coordinator.route(DemoPhase.HANDOFF).embodiment_tag == "real_g1_relative_eef_relative_joints"
    assert coordinator.route(DemoPhase.STACK).role is RobotRole.ALOHA
    assert coordinator.route(DemoPhase.COMPLETE).embodiment_tag == "new_embodiment"


def test_g1_closes_both_articulated_hands_before_lift() -> None:
    timeline = HomeDemoTimeline()
    plan = G1MotionPlan()
    open_hand = plan.sample(20, timeline).joint_offsets
    closed_hand = plan.sample(70, timeline).joint_offsets
    assert open_hand["left_three_joint"] == pytest.approx(0.0)
    assert closed_hand["left_three_joint"] > 0.4
    assert closed_hand["right_three_joint"] < -0.4
    assert closed_hand["left_elbow_roll_joint"] < 0.0
    assert closed_hand["right_elbow_roll_joint"] > 0.0
    np.testing.assert_allclose(plan.sample(70, timeline).root_quaternion_wxyz, (1.0, 0.0, 0.0, 0.0))
