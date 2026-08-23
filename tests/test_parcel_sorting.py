import numpy as np
import pytest

from twingr00t_lab.parcel_sorting import (
    ParcelPhase,
    ParcelSortingTimeline,
    arched_transfer,
    clearance_transfer,
    interpolate,
    smooth_sequence,
    smoothstep,
)


def test_parcel_timeline_is_contiguous() -> None:
    timeline = ParcelSortingTimeline()
    assert timeline.feed_start == 45
    assert timeline.pick_start == 150
    assert timeline.sort_start == 240
    assert timeline.verify_start == 360
    assert timeline.total_frames == 420
    assert timeline.phase(0) is ParcelPhase.ESTABLISH
    assert timeline.phase(45) is ParcelPhase.CONVEYOR_FEED
    assert timeline.phase(150) is ParcelPhase.DUAL_PICK
    assert timeline.phase(240) is ParcelPhase.SORT
    assert timeline.phase(360) is ParcelPhase.VERIFY
    with pytest.raises(IndexError):
        timeline.phase(420)


def test_action_mapping_is_bounded_and_monotonic() -> None:
    timeline = ParcelSortingTimeline()
    indices = [timeline.action_index(frame, 101) for frame in range(timeline.total_frames)]
    assert indices[0] == 0
    assert indices[-1] == 100
    assert indices == sorted(indices)
    positions = [timeline.action_position(frame, 101) for frame in range(timeline.total_frames)]
    assert positions[0] == 0.0
    assert positions[-1] == 100.0
    assert positions == sorted(positions)


def test_smooth_interpolation_has_stationary_endpoints() -> None:
    assert smoothstep(0.0) == 0.0
    assert smoothstep(1.0) == 1.0
    assert smoothstep(-2.0) == 0.0
    assert smoothstep(3.0) == 1.0
    np.testing.assert_allclose(interpolate(np.zeros(3), np.ones(3), 0.5), np.full(3, 0.5))


def test_clearance_transfer_uses_lift_translate_lower_path() -> None:
    start = np.asarray((0.0, 0.0, 0.9))
    end = np.asarray((1.0, 1.0, 0.88))
    np.testing.assert_allclose(clearance_transfer(start, end, 0.0), start)
    np.testing.assert_allclose(clearance_transfer(start, end, 1.0), end)
    np.testing.assert_allclose(clearance_transfer(start, end, 0.25), (0.0, 0.0, 1.18))
    np.testing.assert_allclose(clearance_transfer(start, end, 0.75), (1.0, 1.0, 1.18))


def test_smooth_sequence_reduces_step_discontinuity() -> None:
    values = np.zeros((21, 2))
    values[10:, 0] = 1.0
    smoothed = smooth_sequence(values, radius=3)
    assert smoothed.shape == values.shape
    assert np.max(np.abs(np.diff(smoothed[:, 0]))) < 1.0
    np.testing.assert_allclose(smoothed[:, 1], 0.0)
    np.testing.assert_allclose(smooth_sequence(values, radius=0), values)


def test_arched_transfer_has_fixed_endpoints_and_positive_clearance() -> None:
    start = np.asarray((-0.4, 0.0, 0.9))
    end = np.asarray((0.4, 0.0, 0.9))
    np.testing.assert_allclose(arched_transfer(start, end, 0.0), start)
    np.testing.assert_allclose(arched_transfer(start, end, 1.0), end)
    midpoint = arched_transfer(start, end, 0.5, arc_height=0.2)
    np.testing.assert_allclose(midpoint, (0.0, 0.0, 1.1))
