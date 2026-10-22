"""Closed-loop hover above a moving table object.

Each call consumes one already-tracked fix in the base frame and advances the
commanded joints by at most one control period. The arm is not told to finish
a trajectory before the aim point is allowed to move. Cameras and servos stay
outside this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from soarm_aimbot.geometry.hover import DOWN, NAMED_AXES
from soarm_aimbot.geometry.ik import solve_ik
from soarm_aimbot.geometry.refine import translate_target
from soarm_aimbot.geometry.trajectory import fingertips, outside_work_box
from soarm_aimbot.geometry.transforms import RigidTransform
from soarm_aimbot.geometry.urdf_fk import ARM_JOINTS, UrdfModel, forward_kinematics

ARM_LINKS = (
    "shoulder_link",
    "upper_arm_link",
    "lower_arm_link",
    "wrist_link",
    "gripper_link",
)


@dataclass(frozen=True)
class TrackFix:
    """One confirmed-or-not track already expressed in the base frame."""

    track_id: int
    position_m: np.ndarray
    velocity_m_s: np.ndarray
    confirmed: bool
    missed_frames: int


@dataclass(frozen=True)
class FollowTick:
    mode: str
    reason: str
    source: str
    note: str
    track_id: int | None
    measured_m: np.ndarray | None
    aim_m: np.ndarray | None
    reach_s: float | None
    joints_rad: np.ndarray
    fingertip_m: np.ndarray
    approach_angle_deg: float
    moved: bool


def wrist_improved(previous_m: float | None, current_m: float) -> bool:
    """A wrist residual is kept only when it falls to 70 percent or less."""

    if previous_m is None:
        return True
    return float(current_m) <= 0.7 * float(previous_m)


def arm_ignore_boxes(
    model: UrdfModel,
    joints_rad: np.ndarray,
    world_from_base: RigidTransform,
    fingertip_m: np.ndarray,
    radius_m: float = 0.04,
) -> list[tuple[float, float, float, float, float, float]]:
    """World-frame boxes around arm links. Points at or below the fingertip stay.

    These boxes stand in for a mesh mask. The object on the table is under the
    fingertip, so the box floor is not allowed to drop below that height.
    """

    if radius_m <= 0:
        raise ValueError("The arm ignore radius must be positive")
    joints = dict(zip(ARM_JOINTS, np.asarray(joints_rad, dtype=np.float64).reshape(5)))
    tip = world_from_base.apply(np.asarray(fingertip_m, dtype=np.float64).reshape(3))
    boxes = []
    for link in ARM_LINKS:
        center = world_from_base.apply(forward_kinematics(model, joints, tip_link=link).translation)
        zmin = max(float(center[2]) - radius_m, float(tip[2]) - 0.005)
        zmax = float(center[2]) + radius_m
        if zmax <= zmin + 1e-6:
            continue
        boxes.append(
            (
                float(center[0]) - radius_m,
                float(center[1]) - radius_m,
                zmin,
                float(center[0]) + radius_m,
                float(center[1]) + radius_m,
                zmax,
            )
        )
    return boxes


class FollowController:
    """Hold, acquire, or stream one joint step toward the latest aim point."""

    def __init__(
        self,
        model: UrdfModel,
        joints_rad: np.ndarray,
        *,
        tool_m: np.ndarray,
        approach_axis: str,
        height_m: float,
        work_box_m,
        speed_limit_rad_s: np.ndarray,
        period_s: float = 0.05,
        latency_s: float = 0.2,
        reach_max_s: float = 1.0,
        planar_tol_m: float = 0.015,
        still_speed_m_s: float = 0.02,
        slow_speed_m_s: float = 0.05,
        predict_acquire: bool = False,
    ) -> None:
        if approach_axis not in NAMED_AXES:
            raise ValueError(f"Unknown approach axis {approach_axis}")
        if height_m <= 0 or period_s <= 0 or latency_s < 0 or reach_max_s < period_s:
            raise ValueError("Invalid height, control period, or reach time")
        if planar_tol_m <= 0 or still_speed_m_s < 0 or slow_speed_m_s <= 0:
            raise ValueError("Invalid gate or speed threshold")
        limit = np.asarray(speed_limit_rad_s, dtype=np.float64).reshape(5)
        if np.any(limit <= 0):
            raise ValueError("The joint speed limit must be positive")
        self.model = model
        self.tool_m = np.asarray(tool_m, dtype=np.float64).reshape(3)
        self.approach_axis = approach_axis
        self.height_m = float(height_m)
        self.work_box_m = tuple(float(value) for value in work_box_m)
        self.speed_limit = limit
        self.period_s = float(period_s)
        self.latency_s = float(latency_s)
        self.reach_max_s = float(reach_max_s)
        self.planar_tol_m = float(planar_tol_m)
        self.still_speed_m_s = float(still_speed_m_s)
        self.slow_speed_m_s = float(slow_speed_m_s)
        self.near_m = 2.0 * self.planar_tol_m
        self.predict_acquire = bool(predict_acquire)
        self.q_cmd = np.asarray(joints_rad, dtype=np.float64).reshape(5).copy()
        self.phase = "acquire"
        self.selected_id: int | None = None
        self.locked_rotation: np.ndarray | None = None
        self.wrist_rejected = False
        self._wrist_gap_m: float | None = None
        self._stopped = False

    def stop(self) -> None:
        self._stopped = True

    def accepts_wrist(self, speed_m_s: float, horizontal_m: float) -> bool:
        """Wrist replaces xy only while following, slow, and already near the gate."""

        return (
            self.phase == "follow"
            and not self.wrist_rejected
            and float(speed_m_s) <= self.slow_speed_m_s
            and float(horizontal_m) <= self.near_m
        )

    def tick(self, fix: TrackFix | None, wrist_xy_m: np.ndarray | None = None, wrist_attempted: bool = False) -> FollowTick:
        if self._stopped:
            return self._hold("emergency stop")
        if fix is None:
            reason = "no confirmed track" if self.selected_id is None else "track lost"
            return self._hold(reason)
        if self.selected_id is not None and int(fix.track_id) != self.selected_id:
            return self._hold("track id changed", fix.track_id)
        if not fix.confirmed:
            return self._hold("track not confirmed yet", fix.track_id)
        if self.selected_id is None:
            self.selected_id = int(fix.track_id)
        if int(fix.missed_frames) > 0:
            return self._hold("track lost", fix.track_id)
        position = np.asarray(fix.position_m, dtype=np.float64).reshape(3).copy()
        velocity = np.asarray(fix.velocity_m_s, dtype=np.float64).reshape(3).copy()
        self._ensure_rotation()
        tip, angle = self._tip_and_angle()
        if angle <= 5.0 and abs(float(tip[2]) - (float(position[2]) + self.height_m)) <= self.planar_tol_m:
            self.phase = "follow"
        else:
            self.phase = "acquire"
        source = "overhead"
        note = ""
        speed = float(np.hypot(velocity[0], velocity[1]))
        gap = float(np.hypot(tip[0] - position[0], tip[1] - position[1]))
        if self.accepts_wrist(speed, gap):
            if wrist_xy_m is None:
                if wrist_attempted:
                    note = "wrist did not see it"
            else:
                wrist = np.asarray(wrist_xy_m, dtype=np.float64).reshape(2)
                residual = float(np.hypot(tip[0] - wrist[0], tip[1] - wrist[1]))
                if wrist_improved(self._wrist_gap_m, residual):
                    position[0] = float(wrist[0])
                    position[1] = float(wrist[1])
                    source = "wrist"
                    self._wrist_gap_m = residual
                else:
                    self.wrist_rejected = True
                    note = "wrist residual did not drop"
        elif speed > self.slow_speed_m_s:
            self.wrist_rejected = False
            self._wrist_gap_m = None
        if self.phase == "follow" and gap <= self.planar_tol_m and speed <= self.still_speed_m_s:
            return self._hold("already inside the gate", fix.track_id, source=source, note=note, measured=position, aim=position.copy())
        lead = self.phase == "follow" or self.predict_acquire
        status, solved, aim, reach = self._solve(position, velocity, lead)
        if status:
            return self._hold(status, fix.track_id, source=source, note=note, measured=position, aim=aim, reach=reach)
        step = np.clip(solved.joints_rad - self.q_cmd, -self.speed_limit * self.period_s, self.speed_limit * self.period_s)
        candidate = self.q_cmd + step
        candidate_tip = fingertips(self.model, candidate, self.tool_m)[0]
        if outside_work_box(candidate_tip, self.work_box_m):
            return self._hold("fingertips leave the work box", fix.track_id, source=source, note=note, measured=position, aim=aim, reach=reach)
        candidate_angle = _approach_angle(forward_kinematics(self.model, dict(zip(ARM_JOINTS, candidate))).rotation, self.approach_axis)
        if self.phase == "follow" and candidate_angle > 5.0:
            return self._hold("approach axis tilted away from down", fix.track_id, source=source, note=note, measured=position, aim=aim, reach=reach)
        self.q_cmd = candidate
        return FollowTick(
            self.phase,
            "",
            source,
            note,
            int(fix.track_id),
            position.copy(),
            None if aim is None else aim.copy(),
            reach,
            self.q_cmd.copy(),
            candidate_tip.copy(),
            candidate_angle,
            bool(np.any(np.abs(step) > 1e-12)),
        )

    def _ensure_rotation(self) -> None:
        if self.locked_rotation is not None:
            return
        current = forward_kinematics(self.model, dict(zip(ARM_JOINTS, self.q_cmd))).rotation
        self.locked_rotation = _aim_down(current, NAMED_AXES[self.approach_axis])

    def _solve(self, position, velocity, lead: bool):
        reach = 0.0
        if lead:
            reach = float(np.clip(self.latency_s, self.period_s, self.reach_max_s))
        solved = None
        aim = None
        for index in range(2 if lead else 1):
            if lead and index == 1 and solved is not None and solved.joints_rad is not None:
                travel = float(np.max(np.abs(solved.joints_rad - self.q_cmd) / self.speed_limit))
                reach = float(np.clip(self.latency_s + travel, self.period_s, self.reach_max_s))
            xy = position[:2] + (velocity[:2] * reach if lead else 0.0)
            target, fingertip = translate_target(self.locked_rotation, self.tool_m, xy, float(position[2]), self.height_m)
            aim = np.array([float(xy[0]), float(xy[1]), float(position[2])], dtype=np.float64)
            if not _point_in_box(fingertip, self.work_box_m) or not _point_in_box(target.translation, self.work_box_m):
                return "aim point outside the work box", None, aim, reach
            solved = solve_ik(self.model, self.q_cmd, target, self.approach_axis)
            if not solved.converged or solved.joints_rad is None:
                return ("solution at a joint limit" if solved.at_limit else "no solution"), None, aim, reach
        return "", solved, aim, reach

    def _tip_and_angle(self):
        pose = forward_kinematics(self.model, dict(zip(ARM_JOINTS, self.q_cmd)))
        tip = pose.translation + pose.rotation @ self.tool_m
        return tip, _approach_angle(pose.rotation, self.approach_axis)

    def _hold(
        self,
        reason: str,
        track_id: int | None = None,
        *,
        source: str = "none",
        note: str = "",
        measured: np.ndarray | None = None,
        aim: np.ndarray | None = None,
        reach: float | None = None,
    ) -> FollowTick:
        tip, angle = self._tip_and_angle()
        return FollowTick(
            "hold",
            reason,
            source,
            note,
            track_id,
            None if measured is None else np.asarray(measured, dtype=np.float64).copy(),
            None if aim is None else np.asarray(aim, dtype=np.float64).copy(),
            reach,
            self.q_cmd.copy(),
            tip.copy(),
            angle,
            False,
        )


def _aim_down(rotation: np.ndarray, approach: np.ndarray) -> np.ndarray:
    current = np.asarray(rotation, dtype=np.float64) @ np.asarray(approach, dtype=np.float64)
    return _rotation_between(current, DOWN) @ np.asarray(rotation, dtype=np.float64)


def _approach_angle(rotation: np.ndarray, approach_axis: str) -> float:
    axis = np.asarray(rotation, dtype=np.float64) @ NAMED_AXES[approach_axis]
    cosine = float(np.clip(np.dot(axis, DOWN), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _point_in_box(point: np.ndarray, work_box_m) -> bool:
    xmin, xmax, ymin, ymax, zmin = work_box_m
    return bool(xmin <= point[0] <= xmax and ymin <= point[1] <= ymax and point[2] >= zmin)


def _rotation_between(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    start = source / np.linalg.norm(source)
    finish = target / np.linalg.norm(target)
    cross = np.cross(start, finish)
    cosine = float(np.clip(np.dot(start, finish), -1.0, 1.0))
    sine = float(np.linalg.norm(cross))
    if sine < 1e-10:
        if cosine > 0:
            return np.eye(3)
        helper = np.array([1.0, 0.0, 0.0]) if abs(start[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(start, helper)
        return cv2.Rodrigues((axis / np.linalg.norm(axis)) * np.pi)[0]
    return cv2.Rodrigues(cross / sine * np.arctan2(sine, cosine))[0]
