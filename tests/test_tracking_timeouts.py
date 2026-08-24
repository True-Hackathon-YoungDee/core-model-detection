from __future__ import annotations

import math

import pytest

from fall_detection.tracking import IdentityTracker


def test_identity_timeout_defaults_to_candidate_plus_observation_gap():
    tracker = IdentityTracker()

    assert tracker.max_unseen_s == pytest.approx(2.5)


def test_track_survives_many_result_calls_within_elapsed_timeout():
    tracker = IdentityTracker(max_unseen_s=2.5)
    assigned_id = tracker.assign([(0.25, 0.25)], now=0.0)[0]

    for index in range(1, 250):
        tracker.assign([], now=index / 100.0)

    assert tracker.active_ids == [assigned_id]


def test_track_expires_on_first_call_strictly_beyond_elapsed_timeout():
    lost_ids: list[int] = []
    tracker = IdentityTracker(max_unseen_s=2.5, on_lost=lost_ids.append)
    assigned_id = tracker.assign([(0.25, 0.25)], now=0.0)[0]

    tracker.assign([], now=2.5)
    assert tracker.active_ids == [assigned_id]
    assert lost_ids == []

    tracker.assign([], now=2.500001)
    tracker.assign([], now=20.0)

    assert tracker.active_ids == []
    assert lost_ids == [assigned_id]


def test_matched_observation_refreshes_last_seen_timestamp():
    tracker = IdentityTracker(max_unseen_s=2.5)
    assigned_id = tracker.assign([(0.25, 0.25)], now=0.0)[0]
    tracker.assign([], now=2.0)

    assert tracker.assign([(0.26, 0.25)], now=2.4) == [assigned_id]
    tracker.assign([], now=4.9)
    assert tracker.active_ids == [assigned_id]

    tracker.assign([], now=4.900001)
    assert tracker.active_ids == []


def test_detection_after_timeout_gets_new_id_instead_of_reviving_stale_track():
    lost_ids: list[int] = []
    tracker = IdentityTracker(max_unseen_s=2.5, on_lost=lost_ids.append)
    stale_id = tracker.assign([(0.25, 0.25)], now=0.0)[0]

    replacement_id = tracker.assign([(0.25, 0.25)], now=2.500001)[0]

    assert lost_ids == [stale_id]
    assert replacement_id != stale_id
    assert tracker.active_ids == [replacement_id]


def test_new_track_is_seen_on_its_creation_observation():
    lost_ids: list[int] = []
    tracker = IdentityTracker(max_unseen_s=0.001, on_lost=lost_ids.append)

    assigned_id = tracker.assign([(0.5, 0.5)], now=10.0)[0]

    assert tracker.active_ids == [assigned_id]
    assert lost_ids == []


@pytest.mark.parametrize("max_unseen_s", [0.0, -1.0, math.inf, math.nan, True])
def test_identity_tracker_rejects_invalid_elapsed_timeout(max_unseen_s):
    with pytest.raises(ValueError, match="max_unseen_s must be a positive finite number"):
        IdentityTracker(max_unseen_s=max_unseen_s)
