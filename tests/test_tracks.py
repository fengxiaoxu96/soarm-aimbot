import numpy as np

from soarm_aimbot.geometry.tracks import MultiObjectTracker, hungarian


def test_hungarian_prefers_the_crossing_assignment():
    pairs = dict(hungarian(np.array([[9.0, 1.0], [1.0, 9.0]])))
    assert pairs == {0: 1, 1: 0}


def test_hungarian_leaves_extra_detections_unmatched():
    pairs = hungarian(np.array([[0.0], [5.0]]))
    assert pairs == [(0, 0)]


def test_short_occlusion_keeps_the_id_and_a_long_gap_retires_it():
    tracker = MultiObjectTracker(match_m=0.08, max_speed_m_s=0.5, max_missed=2, confirm_hits=2, meas_std_m=0.001)
    first = tracker.update(np.array([[0.0, 0.0, 0.05]]), 0.0)
    assert first.tracks[0].confirmed is False
    second = tracker.update(np.array([[0.0, 0.0, 0.05]]), 1.0 / 30.0)
    assert second.tracks[0].confirmed is True
    track_id = second.tracks[0].track_id
    missed = tracker.update(np.zeros((0, 3)), 2.0 / 30.0)
    assert missed.tracks[0].track_id == track_id
    assert missed.tracks[0].missed_frames == 1
    still = tracker.update(np.zeros((0, 3)), 3.0 / 30.0)
    assert still.tracks[0].track_id == track_id
    gone = tracker.update(np.zeros((0, 3)), 4.0 / 30.0)
    assert gone.tracks == []
    arrived = tracker.update(np.array([[0.0, 0.0, 0.05]]), 5.0 / 30.0)
    assert arrived.tracks[0].track_id != track_id


def test_a_far_false_detection_does_not_take_over_the_track():
    tracker = MultiObjectTracker(match_m=0.05, max_speed_m_s=0.2, confirm_hits=1, meas_std_m=0.001)
    tracker.update(np.array([[0.0, 0.0, 0.04]]), 0.0)
    frame = tracker.update(np.array([[0.01, 0.0, 0.04], [1.0, 0.0, 0.04]]), 1.0 / 30.0)
    original = next(item for item in frame.tracks if item.track_id == 1)
    assert original.missed_frames == 0
    assert abs(original.position_m[0] - 0.01) < 0.02
    other = next(item for item in frame.tracks if item.track_id != 1)
    assert abs(other.position_m[0] - 1.0) < 0.05
