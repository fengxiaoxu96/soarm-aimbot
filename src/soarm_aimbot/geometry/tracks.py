"""Keep a stable id, position, and velocity for each table object.

Detections are positions in the world frame. Matching uses the Hungarian
algorithm. Each track is a constant-velocity Kalman filter. A track is
confirmed only after several hits, and a short gap does not retire its id.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrackState:
    """One track at one timestamp. Positions and velocities are in the world frame."""

    track_id: int
    position_m: np.ndarray
    velocity_m_s: np.ndarray
    measured_m: np.ndarray | None
    confirmed: bool
    hits: int
    missed_frames: int
    last_timestamp_s: float


@dataclass(frozen=True)
class TrackFrame:
    timestamp_s: float
    gate_m: float
    tracks: list[TrackState]


class MultiObjectTracker:
    """Associate 3D detections across frames and smooth them.

    ``match_m`` is the largest distance that may still belong to the same
    object. The per-frame gate is also limited by ``max_speed_m_s * dt``, so a
    slow frame does not pull in a neighbor, and a fast object is not split
    into a new id while it stays inside ``match_m``.
    """

    def __init__(
        self,
        *,
        match_m: float = 0.08,
        max_speed_m_s: float = 0.5,
        max_missed: int = 15,
        confirm_hits: int = 3,
        meas_std_m: float = 0.01,
        accel_std_m_s2: float = 5.0,
        min_gate_m: float = 0.02,
    ) -> None:
        if match_m <= 0 or max_speed_m_s <= 0 or max_missed < 0 or confirm_hits < 1:
            raise ValueError("Invalid tracking parameters")
        if meas_std_m <= 0 or accel_std_m_s2 <= 0 or min_gate_m <= 0:
            raise ValueError("Noise and minimum match distance must be positive")
        self.match_m = float(match_m)
        self.max_speed_m_s = float(max_speed_m_s)
        self.max_missed = int(max_missed)
        self.confirm_hits = int(confirm_hits)
        self.meas_std_m = float(meas_std_m)
        self.accel_std_m_s2 = float(accel_std_m_s2)
        self.min_gate_m = float(min_gate_m)
        self._tracks: list[_Track] = []
        self._next_id = 1
        self._time: float | None = None

    def update(self, positions_m: np.ndarray, timestamp_s: float) -> TrackFrame:
        detected = np.asarray(positions_m, dtype=np.float64)
        if detected.size == 0:
            detected = np.zeros((0, 3), dtype=np.float64)
        elif detected.ndim != 2 or detected.shape[1] != 3:
            raise ValueError(f"Detected positions must be (N, 3), got {detected.shape}")
        timestamp = float(timestamp_s)
        if self._time is None:
            measured = {}
            for position in detected:
                self._start(position, timestamp)
                measured[self._tracks[-1].track_id] = np.asarray(position, dtype=np.float64)
            self._time = timestamp
            return self._frame(timestamp, self.min_gate_m, measured)
        dt = timestamp - self._time
        if dt < 0:
            dt = 0.0
        for track in self._tracks:
            track.filter.predict(dt)
        gate = self.gate_m(dt)
        predicted = np.stack([track.filter.position for track in self._tracks]) if self._tracks else np.zeros((0, 3))
        pairs = _associate(predicted, detected, gate)
        matched_tracks = {track_index for track_index, _ in pairs}
        matched_detections = {detection_index for _, detection_index in pairs}
        measured: dict[int, np.ndarray] = {}
        for track_index, detection_index in pairs:
            track = self._tracks[track_index]
            measurement = detected[detection_index]
            track.filter.update(measurement)
            track.hits += 1
            track.missed = 0
            track.last_timestamp_s = timestamp
            measured[track.track_id] = measurement
        surviving: list[_Track] = []
        for index, track in enumerate(self._tracks):
            if index not in matched_tracks:
                track.missed += 1
            if track.missed <= self.max_missed:
                surviving.append(track)
        self._tracks = surviving
        for index, position in enumerate(detected):
            if index not in matched_detections:
                self._start(position, timestamp)
                measured[self._tracks[-1].track_id] = position
        self._time = timestamp
        return self._frame(timestamp, gate, measured)

    def gate_m(self, dt_s: float) -> float:
        speed_limit = self.max_speed_m_s * max(dt_s, 0.0) * 1.5
        return float(min(self.match_m, max(self.min_gate_m, speed_limit)))

    def _start(self, position: np.ndarray, timestamp: float) -> None:
        self._tracks.append(
            _Track(
                track_id=self._next_id,
                filter=_ConstantVelocityFilter(position, self.meas_std_m, self.accel_std_m_s2),
                hits=1,
                missed=0,
                last_timestamp_s=timestamp,
            )
        )
        self._next_id += 1

    def _frame(self, timestamp: float, gate: float, measured: dict[int, np.ndarray]) -> TrackFrame:
        states = [
            TrackState(
                track_id=track.track_id,
                position_m=track.filter.position,
                velocity_m_s=track.filter.velocity,
                measured_m=None if track.track_id not in measured else measured[track.track_id].copy(),
                confirmed=track.hits >= self.confirm_hits,
                hits=track.hits,
                missed_frames=track.missed,
                last_timestamp_s=track.last_timestamp_s,
            )
            for track in self._tracks
        ]
        states.sort(key=lambda item: item.track_id)
        return TrackFrame(timestamp_s=timestamp, gate_m=gate, tracks=states)


@dataclass
class _Track:
    track_id: int
    filter: "_ConstantVelocityFilter"
    hits: int
    missed: int
    last_timestamp_s: float


class _ConstantVelocityFilter:
    """State is ``(x, y, z, vx, vy, vz)``."""

    def __init__(self, position: np.ndarray, meas_std_m: float, accel_std_m_s2: float) -> None:
        self.state = np.zeros(6, dtype=np.float64)
        self.state[:3] = np.asarray(position, dtype=np.float64)
        self.covariance = np.eye(6, dtype=np.float64)
        self.covariance[3:, 3:] *= 10.0
        self.meas_std_m = float(meas_std_m)
        self.accel_std_m_s2 = float(accel_std_m_s2)

    def predict(self, dt_s: float) -> None:
        dt = max(float(dt_s), 0.0)
        transition = np.eye(6, dtype=np.float64)
        transition[0, 3] = dt
        transition[1, 4] = dt
        transition[2, 5] = dt
        process = _process_noise(dt, self.accel_std_m_s2)
        self.state = transition @ self.state
        self.covariance = transition @ self.covariance @ transition.T + process

    def update(self, measurement: np.ndarray) -> None:
        observation = np.zeros((3, 6), dtype=np.float64)
        observation[0, 0] = observation[1, 1] = observation[2, 2] = 1.0
        innovation = np.asarray(measurement, dtype=np.float64) - observation @ self.state
        innovation_cov = observation @ self.covariance @ observation.T + np.eye(3) * self.meas_std_m**2
        gain = self.covariance @ observation.T @ np.linalg.inv(innovation_cov)
        self.state = self.state + gain @ innovation
        self.covariance = (np.eye(6) - gain @ observation) @ self.covariance

    @property
    def position(self) -> np.ndarray:
        return self.state[:3].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.state[3:].copy()


def hungarian(cost: np.ndarray) -> list[tuple[int, int]]:
    """Minimum-cost assignment. Returns ``(row, column)`` pairs for a rectangular cost."""

    raw = np.asarray(cost, dtype=np.float64)
    if raw.ndim != 2:
        raise ValueError(f"The cost matrix must be two-dimensional, got {raw.shape}")
    n_rows, n_cols = raw.shape
    if n_rows == 0 or n_cols == 0:
        return []
    size = max(n_rows, n_cols)
    matrix = np.zeros((size, size), dtype=np.float64)
    matrix[:n_rows, :n_cols] = raw
    padding = float(np.max(np.abs(raw))) + 1.0
    if n_rows < size:
        matrix[n_rows:, :] = padding
    if n_cols < size:
        matrix[:, n_cols:] = padding
    row_potential = np.zeros(size + 1, dtype=np.float64)
    col_potential = np.zeros(size + 1, dtype=np.float64)
    column_of_row = np.zeros(size + 1, dtype=np.int64)
    predecessor = np.zeros(size + 1, dtype=np.int64)
    for row in range(1, size + 1):
        column_of_row[0] = row
        current_column = 0
        minimum = np.full(size + 1, np.inf)
        used = np.zeros(size + 1, dtype=bool)
        while True:
            used[current_column] = True
            current_row = int(column_of_row[current_column])
            delta = np.inf
            next_column = 0
            for column in range(1, size + 1):
                if used[column]:
                    continue
                value = matrix[current_row - 1, column - 1] - row_potential[current_row] - col_potential[column]
                if value < minimum[column]:
                    minimum[column] = value
                    predecessor[column] = current_column
                if minimum[column] < delta:
                    delta = float(minimum[column])
                    next_column = column
            for column in range(size + 1):
                if used[column]:
                    row_potential[column_of_row[column]] += delta
                    col_potential[column] -= delta
                else:
                    minimum[column] -= delta
            current_column = next_column
            if column_of_row[current_column] == 0:
                break
        while True:
            previous = int(predecessor[current_column])
            column_of_row[current_column] = column_of_row[previous]
            current_column = previous
            if current_column == 0:
                break
    pairs = []
    for column in range(1, size + 1):
        row = int(column_of_row[column]) - 1
        assigned_column = column - 1
        if row < n_rows and assigned_column < n_cols:
            pairs.append((row, assigned_column))
    return pairs


def _associate(predicted: np.ndarray, detected: np.ndarray, gate_m: float) -> list[tuple[int, int]]:
    if len(predicted) == 0 or len(detected) == 0:
        return []
    distance = np.linalg.norm(predicted[:, None, :] - detected[None, :, :], axis=2)
    blocked = float(gate_m) + 1.0 + float(np.max(distance))
    cost = np.where(distance <= gate_m, distance, blocked)
    return [(int(row), int(col)) for row, col in hungarian(cost) if distance[row, col] <= gate_m]


def _process_noise(dt_s: float, accel_std_m_s2: float) -> np.ndarray:
    dt = max(float(dt_s), 0.0)
    variance = accel_std_m_s2**2
    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt2 * dt2
    noise = np.zeros((6, 6), dtype=np.float64)
    for axis in range(3):
        noise[axis, axis] = 0.25 * dt4 * variance
        noise[axis, axis + 3] = 0.5 * dt3 * variance
        noise[axis + 3, axis] = 0.5 * dt3 * variance
        noise[axis + 3, axis + 3] = dt2 * variance
    return noise
