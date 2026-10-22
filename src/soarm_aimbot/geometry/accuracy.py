"""Compare overhead perception with ruler measurements and a known push direction.

Static points use the world-frame median position. Motion uses a track log from
``geometry.tracks``: one id should stay with the object, and its horizontal velocity
should follow the push.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StaticPoint:
    name: str
    expected_m: np.ndarray
    measured_m: np.ndarray
    error_m: np.ndarray
    planar_m: float
    height_m: float
    accepted: bool


@dataclass(frozen=True)
class MotionGrade:
    track_id: int
    travel_m: float
    max_jump_m: float
    speed_m_s: float
    angle_deg: float
    presence: float
    replaced_by: int | None
    id_held: bool
    continuous: bool
    direction_ok: bool
    moved: bool
    accepted: bool


def grade_static_point(
    name: str,
    expected_m: np.ndarray,
    measured_m: np.ndarray,
    *,
    planar_limit_m: float,
    height_limit_m: float,
) -> StaticPoint:
    """Signed error is measured minus expected. Height uses the absolute Z error."""

    expected = np.asarray(expected_m, dtype=np.float64).reshape(3)
    measured = np.asarray(measured_m, dtype=np.float64).reshape(3)
    error = measured - expected
    planar = float(np.linalg.norm(error[:2]))
    height = float(abs(error[2]))
    return StaticPoint(
        name=name,
        expected_m=expected,
        measured_m=measured,
        error_m=error,
        planar_m=planar,
        height_m=height,
        accepted=planar <= planar_limit_m and height <= height_limit_m,
    )


def median_position(samples_m: np.ndarray) -> np.ndarray:
    cloud = np.asarray(samples_m, dtype=np.float64)
    if cloud.ndim != 2 or cloud.shape[1] != 3 or len(cloud) == 0:
        raise ValueError(f"Static samples must be a non-empty (N, 3) array, got {cloud.shape}")
    return np.median(cloud, axis=0)


def grade_motion(
    frames: list[dict],
    direction_xy: np.ndarray,
    *,
    min_travel_m: float = 0.05,
    max_jump_m: float = 0.04,
    max_angle_deg: float = 25.0,
    min_speed_m_s: float = 0.05,
    min_presence: float = 0.8,
    handoff_gap_s: float = 0.5,
    handoff_distance_m: float = 0.08,
    track_id: int | None = None,
) -> MotionGrade:
    """Grade one pushed object inside a per-frame track log.

    Each frame dict needs ``timestamp_s`` and ``tracks``. A track needs
    ``id``, ``confirmed``, ``position_m``, ``velocity_m_s`` and ``missed_frames``.
    """

    direction = np.asarray(direction_xy, dtype=np.float64).reshape(2)
    if float(np.linalg.norm(direction)) < 1e-9:
        raise ValueError("The push direction must have non-zero length")
    direction = direction / np.linalg.norm(direction)
    series = _series_by_id(frames)
    if not series:
        raise ValueError("The track log has no confirmed track")
    chosen = track_id if track_id is not None else _longest_travel_id(series)
    if chosen not in series:
        raise ValueError(f"The log has no confirmed track with id {chosen}")
    samples = series[chosen]
    positions = np.stack([item["position_m"] for item in samples])
    travel = float(np.linalg.norm(positions[-1, :2] - positions[0, :2]))
    jumps = [
        float(np.linalg.norm(positions[index, :2] - positions[index - 1, :2]))
        for index in range(1, len(positions))
    ]
    max_jump = max(jumps) if jumps else 0.0
    velocities = [
        np.asarray(item["velocity_m_s"], dtype=np.float64)
        for item in samples
        if item["missed_frames"] == 0 and float(np.linalg.norm(item["velocity_m_s"][:2])) >= min_speed_m_s
    ]
    if velocities:
        mean_velocity = np.mean(np.stack(velocities), axis=0)
        speed = float(np.linalg.norm(mean_velocity[:2]))
        angle = _angle_deg(mean_velocity[:2], direction)
    else:
        speed = 0.0
        angle = 180.0
    presence = _presence(frames, chosen, samples[0]["timestamp_s"], samples[-1]["timestamp_s"])
    replaced_by = _replacement(
        series, chosen, samples[-1], handoff_gap_s=handoff_gap_s, handoff_distance_m=handoff_distance_m
    )
    id_held = presence >= min_presence and replaced_by is None
    continuous = max_jump <= max_jump_m
    direction_ok = angle <= max_angle_deg
    moved = travel >= min_travel_m and speed >= min_speed_m_s
    return MotionGrade(
        track_id=int(chosen),
        travel_m=travel,
        max_jump_m=max_jump,
        speed_m_s=speed,
        angle_deg=angle,
        presence=presence,
        replaced_by=replaced_by,
        id_held=id_held,
        continuous=continuous,
        direction_ok=direction_ok,
        moved=moved,
        accepted=id_held and continuous and direction_ok and moved,
    )


def _series_by_id(frames: list[dict]) -> dict[int, list[dict]]:
    series: dict[int, list[dict]] = {}
    for frame in frames:
        timestamp = float(frame["timestamp_s"])
        for track in frame.get("tracks", []):
            if not track.get("confirmed", False):
                continue
            track_id = int(track["id"])
            series.setdefault(track_id, []).append(
                {
                    "timestamp_s": timestamp,
                    "position_m": np.asarray(track["position_m"], dtype=np.float64),
                    "velocity_m_s": np.asarray(track["velocity_m_s"], dtype=np.float64),
                    "missed_frames": int(track.get("missed_frames", 0)),
                }
            )
    for samples in series.values():
        samples.sort(key=lambda item: item["timestamp_s"])
    return series


def _longest_travel_id(series: dict[int, list[dict]]) -> int:
    def travel(samples: list[dict]) -> float:
        return float(np.linalg.norm(samples[-1]["position_m"][:2] - samples[0]["position_m"][:2]))

    return max(series, key=lambda track_id: travel(series[track_id]))


def _presence(frames: list[dict], track_id: int, start_s: float, end_s: float) -> float:
    inside = [frame for frame in frames if start_s <= float(frame["timestamp_s"]) <= end_s]
    if not inside:
        return 0.0
    present = 0
    for frame in inside:
        ids = {int(track["id"]) for track in frame.get("tracks", []) if track.get("confirmed", False)}
        if track_id in ids:
            present += 1
    return float(present / len(inside))


def _replacement(
    series: dict[int, list[dict]],
    track_id: int,
    last: dict,
    *,
    handoff_gap_s: float,
    handoff_distance_m: float,
) -> int | None:
    first = series[track_id][0]
    for other_id, samples in series.items():
        if other_id == track_id or not samples:
            continue
        successor_gap = float(samples[0]["timestamp_s"] - last["timestamp_s"])
        successor_distance = float(np.linalg.norm(samples[0]["position_m"][:2] - last["position_m"][:2]))
        predecessor_gap = float(first["timestamp_s"] - samples[-1]["timestamp_s"])
        predecessor_distance = float(np.linalg.norm(samples[-1]["position_m"][:2] - first["position_m"][:2]))
        if 0.0 <= successor_gap <= handoff_gap_s and successor_distance <= handoff_distance_m:
            return int(other_id)
        if 0.0 <= predecessor_gap <= handoff_gap_s and predecessor_distance <= handoff_distance_m:
            return int(other_id)
    return None


def _angle_deg(velocity_xy: np.ndarray, direction_xy: np.ndarray) -> float:
    velocity = np.asarray(velocity_xy, dtype=np.float64)
    speed = float(np.linalg.norm(velocity))
    if speed < 1e-9:
        return 180.0
    cosine = float(np.clip(np.dot(velocity / speed, direction_xy), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))
