import numpy as np
import pytest

from scripts.build_n17_phase_dataset import phase_segments
from scripts.verify_n17_recording import verify_language_routing
from twingr00t_lab.phase_router import ThreeBlockPhaseRouter, phase_prompts


def observation(blocks, *, right_grip=0.048, left_grip=0.048, right_eef=None):
    return {
        "block_positions": blocks,
        "block_linear_velocities": np.zeros((3, 3)),
        "eef_positions": np.stack((blocks[1], blocks[0] if right_eef is None else right_eef)),
        "gripper_positions": np.array([left_grip, right_grip]),
    }


def test_router_requires_sustained_pick_and_place_evidence():
    initial = np.array([[0.10, 0.0, 0.765], [-0.15, 0.0, 0.765], [0.25, 0.0, 0.765]])
    router = ThreeBlockPhaseRouter(initial)
    assert router.prompt == phase_prompts()[0]
    for step in range(100):
        assert not router.observe(**observation(initial), control_step=step)
    lifted = initial.copy()
    lifted[0, 2] += 0.04
    for step in range(4):
        assert not router.observe(**observation(lifted, right_grip=0.01), control_step=100 + step)
    assert router.observe(**observation(lifted, right_grip=0.01), control_step=104)
    assert router.prompt == phase_prompts()[1]
    # An object still in the air cannot count as released merely because the
    # gripper opens. A dropped but moving object is not yet stable either.
    for step in range(40):
        assert not router.observe(**observation(lifted), control_step=105 + step)
    placed = initial.copy()
    placed[0, 0] -= 0.08
    for step in range(30):
        assert not router.observe(**observation(placed), control_step=145 + step)
    assert router.observe(**observation(placed), control_step=175)
    assert router.prompt == phase_prompts()[2]
    assert [event["to_phase"] for event in router.events] == [1, 2]


def test_router_rejects_invalid_inputs_and_arm_plan():
    with pytest.raises(ValueError):
        phase_prompts(("right", "wrong", "left"))
    with pytest.raises(ValueError):
        ThreeBlockPhaseRouter(np.zeros((2, 3)))
    router = ThreeBlockPhaseRouter(np.zeros((3, 3)))
    with pytest.raises(ValueError):
        router.observe(
            block_positions=np.zeros((3, 3)),
            block_linear_velocities=np.zeros((3, 3)),
            eef_positions=np.zeros((2, 3)),
            gripper_positions=np.array([np.nan, 0.0]),
            control_step=0,
        )


def test_teacher_phase_boundaries_use_lift_and_release_not_elapsed_fractions():
    z = np.full((120, 3), 0.765, dtype=np.float32)
    for color, start, end in ((0, 10, 20), (1, 40, 50), (2, 70, 80)):
        z[start:end, color] += 0.04
    grippers = np.full((120, 2), 0.048, dtype=np.float32)
    grippers[5:25, 1] = -0.01
    grippers[35:55, 0] = -0.01
    grippers[65:85, 1] = -0.01
    assert phase_segments(z, grippers, frame_count=119) == [
        (0, 10),
        (10, 25),
        (25, 40),
        (40, 55),
        (55, 70),
        (70, 119),
    ]
    with pytest.raises(ValueError, match="one-step"):
        phase_segments(z, grippers, frame_count=120)


def test_recording_verifier_binds_policy_call_prompts_to_observed_transitions():
    prompts = phase_prompts()
    event = {
        "control_step": 17,
        "from_phase": 0,
        "to_phase": 1,
        "from_prompt": prompts[0],
        "to_prompt": prompts[1],
    }
    manifest = {
        "steps_executed": 48,
        "task_instruction": "Sort the red, green and blue blocks.",
        "phase_router": {
            "enabled": True,
            "prompts": list(prompts),
            "events": [event],
            "action_source": "GR00T_N1.7_model_inference_only",
        },
    }
    calls = [
        {"step": 0, "language_prompt": prompts[0]},
        {"step": 16, "language_prompt": prompts[0]},
        {"step": 32, "language_prompt": prompts[1]},
    ]
    assert verify_language_routing(manifest, calls) == {"language_router_verified": True, "phase_transitions": 1}
    bad = [dict(call) for call in calls]
    bad[-1]["language_prompt"] = prompts[0]
    with pytest.raises(ValueError, match="language"):
        verify_language_routing(manifest, bad)
