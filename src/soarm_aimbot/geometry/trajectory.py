"""Joint-space trajectory from the current pose to q-star.

Speed stays at or below the given limit. The fingertip of every sample is
checked against the work box before anything is sent to a servo.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from soarm_aimbot.geometry.urdf_fk import ARM_JOINTS, UrdfModel, forward_kinematics


@dataclass(frozen=True)
class JointTrajectory:
    times_s: np.ndarray
    joints_rad: np.ndarray
    period_s: float
    speed_limit_rad_s: np.ndarray


@dataclass(frozen=True)
class PlayResult:
    status: str
    held_joints_rad: np.ndarray | None
    sent_count: int


def interpolate(start_rad, goal_rad, speed_limit_rad_s, period_s: float) -> JointTrajectory:
    """Linear joint interpolation. Each step's speed is at most the limit."""

    start = np.asarray(start_rad, dtype=np.float64).reshape(5)
    goal = np.asarray(goal_rad, dtype=np.float64).reshape(5)
    limit = np.asarray(speed_limit_rad_s, dtype=np.float64).reshape(5)
    if period_s <= 0:
        raise ValueError("The control period must be positive")
    if np.any(limit <= 0):
        raise ValueError("The joint speed limit must be positive")
    delta = goal - start
    duration = float(np.max(np.abs(delta) / limit))
    if duration == 0.0:
        return JointTrajectory(np.array([0.0]), start.reshape(1, 5), period_s, limit)
    steps = int(np.ceil(duration / period_s))
    times = np.arange(steps + 1, dtype=np.float64) * period_s
    joints = start + (times / times[-1])[:, None] * delta
    joints[-1] = goal
    return JointTrajectory(times, joints, period_s, limit)


def segment_speeds(trajectory: JointTrajectory) -> np.ndarray:
    """Speed of each joint on each segment, shape ``(N-1, 5)``. Empty if one sample."""

    if len(trajectory.times_s) < 2:
        return np.zeros((0, 5), dtype=np.float64)
    dt = np.diff(trajectory.times_s)
    return np.diff(trajectory.joints_rad, axis=0) / dt[:, None]


def fingertips(model: UrdfModel, joints_rad: np.ndarray, tool_m: np.ndarray) -> np.ndarray:
    tool = np.asarray(tool_m, dtype=np.float64).reshape(3)
    points = []
    for row in np.asarray(joints_rad, dtype=np.float64).reshape(-1, 5):
        pose = forward_kinematics(model, dict(zip(ARM_JOINTS, row)))
        points.append(pose.translation + pose.rotation @ tool)
    return np.stack(points)


def outside_work_box(points_m: np.ndarray, work_box_m) -> list[int]:
    xmin, xmax, ymin, ymax, zmin = work_box_m
    bad = []
    for index, point in enumerate(np.asarray(points_m, dtype=np.float64).reshape(-1, 3)):
        if not (xmin <= point[0] <= xmax and ymin <= point[1] <= ymax and point[2] >= zmin):
            bad.append(index)
    return bad


def play(joints_rad, period_s: float, send, sleep, stop_requested) -> PlayResult:
    """Send each sample. A stop request holds the last sample already sent."""

    points = np.asarray(joints_rad, dtype=np.float64).reshape(-1, 5)
    held = None
    sent = 0
    for index, point in enumerate(points):
        if stop_requested():
            return PlayResult("estop", None if held is None else held.copy(), sent)
        try:
            send(point.copy())
            held = point.copy()
            sent += 1
            if index + 1 < len(points):
                sleep(period_s)
        except KeyboardInterrupt:
            return PlayResult("estop", None if held is None else held.copy(), sent)
    return PlayResult("done", None if held is None else held.copy(), sent)
