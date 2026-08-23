import numpy as np

from twingr00t_lab.experiment_metrics import G1EpisodeMetrics


def test_g1_episode_metrics_records_success_and_smoothness() -> None:
    metrics = G1EpisodeMetrics(seed=10, requested_steps=3)
    destination = np.array([1.0, 2.0, 0.5])
    for step, cargo in enumerate(
        (
            np.array([0.0, 0.0, 0.1]),
            np.array([0.5, 1.0, 0.4]),
            np.array([0.9, 1.95, 0.5]),
        )
    ):
        metrics.update(
            step=step,
            action=np.full(4, step, dtype=np.float32),
            robot_position=np.array([step, 0.0, 0.0]),
            cargo_position=cargo,
            destination_position=destination,
            inference_latency_ms=10.0 + step,
            terminated=step == 2,
            truncated=False,
        )

    result = metrics.summary()
    assert result["success"] is True
    assert result["success_step"] == 2
    assert result["terminal_step"] == 2
    assert result["completed_steps"] == 3
    assert result["max_cargo_lift_m"] == 0.4
    assert result["mean_action_delta_l2"] == 2.0


def test_g1_episode_metrics_records_failure() -> None:
    metrics = G1EpisodeMetrics(seed=None, requested_steps=1)
    metrics.update(
        step=0,
        action=np.zeros(2),
        robot_position=np.zeros(3),
        cargo_position=np.zeros(3),
        destination_position=np.ones(3),
        inference_latency_ms=5.0,
        terminated=False,
        truncated=True,
    )
    result = metrics.summary()
    assert result["success"] is False
    assert result["truncated_step"] == 0
    assert result["mean_action_delta_l2"] == 0.0


def test_g1_episode_metrics_ignores_autoreset_state() -> None:
    metrics = G1EpisodeMetrics(seed=10, requested_steps=2)
    for step, position, state_valid in (
        (0, np.array([2.0, 0.0, 0.2]), True),
        (1, np.zeros(3), False),
    ):
        metrics.update(
            step=step,
            action=np.zeros(2),
            robot_position=position,
            cargo_position=position,
            destination_position=np.ones(3),
            inference_latency_ms=5.0,
            terminated=step == 1,
            truncated=False,
            state_valid=state_valid,
        )
    result = metrics.summary()
    assert result["final_robot_position_w"] == [2.0, 0.0, 0.2]
    assert result["terminal_step"] == 1
