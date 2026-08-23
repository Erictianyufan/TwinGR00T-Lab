import numpy as np
import pytest

from twingr00t_lab.parcel_sorting import (
    ParcelPhase,
    ParcelSortingTimeline,
    interpolate,
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


def test_smooth_interpolation_has_stationary_endpoints() -> None:
    assert smoothstep(0.0) == 0.0
    assert smoothstep(1.0) == 1.0
    assert smoothstep(-2.0) == 0.0
    assert smoothstep(3.0) == 1.0
    np.testing.assert_allclose(interpolate(np.zeros(3), np.ones(3), 0.5), np.full(3, 0.5))
