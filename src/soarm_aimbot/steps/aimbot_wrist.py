"""Wrist-camera aimbot: track a table object and draw the gripper's aim line.

Run this after Steps 1-5. It reads every calibration file they write:

- ``calib/overhead_intrinsics.yaml``  (Step 1)
- ``calib/wrist_intrinsics.yaml``     (Step 2)
- ``calib/T_world_overhead.yaml``     (Step 3)
- ``calib/T_base_world.yaml`` and ``calib/servo_map.yaml`` (Step 4)
- ``calib/T_gripper_wrist.yaml``      (Step 5)

The overhead RealSense segments objects above the table and tracks them. The
arm runs in two stages:

Stage 1 - initial pose
    All five joints ramp at ``--init-speed-deg-s`` to a fixed joint pose
    (default shoulder_pan = shoulder_lift = elbow_flex = 0 deg, wrist_flex =
    +50 deg, wrist_roll = -90 deg, set with ``--init-pose-deg``). Once there
    the servos hold the pose under torque. Detection keeps running, so the
    object can be placed by hand while the overlay shows its position. Press
    **s** to enter Stage 2; the key is ignored until the pose is reached.

Stage 2 - base-joint trace of the object
    Joints 1-4 (shoulder_lift, elbow_flex, wrist_flex, wrist_roll) stay locked
    at the initial pose. Only shoulder_pan moves. The object's world position
    is transformed into base_link and its **y** component is the target. With
    joints 1-4 fixed the gripper moves on a circle about the pan axis, so

        y_EE(theta) = c + a*cos(theta) + b*sin(theta)

    a, b, c come from FK at theta = 0, +pi/2, -pi/2, which uses the pan axis
    exactly as the URDF defines it. Solving for theta:

        R = sqrt(a^2 + b^2),  psi = atan2(b, a)
        theta = psi +/- arccos((y_target - c) / R)

    Candidates outside the shoulder_pan limits are dropped, the one closest to
    the current angle is chosen and then speed-limited by shoulder_pan's URDF
    velocity times ``--speed-scale``. Press **s** again to return to Stage 1.

Wrist overlay
    The gripper's -Z axis (the approach direction the wrist camera looks
    along) is cast as a ray from the gripper_link origin. The first hit with a
    sphere around the tracked object or with the table plane is the aim point.
    The ray and the aim point are projected into the wrist image with the
    Step 5 hand-eye transform.

Keys
    q / Esc   quit
    s         Stage 1 -> Stage 2 (after the initial pose is reached), Stage 2 -> Stage 1
    r         release torque
    e         re-enable torque at the current position (re-reads the servos)

Usage::

    aimbot-wrist self-check
    aimbot-wrist run --preview
    aimbot-wrist run --port /dev/ttyACM0 --color-lower 0,120,60 --color-upper 15,255,255
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

from soarm_aimbot.cameras.base import CameraError
from soarm_aimbot.geometry.depth import spatial_median
from soarm_aimbot.geometry.follow import arm_ignore_boxes
from soarm_aimbot.geometry.hover import tcp_offset_m
from soarm_aimbot.geometry.intrinsics import PinholeIntrinsics, load_intrinsics
from soarm_aimbot.geometry.rgb_model import RgbCameraModel, load_rgb_intrinsics
from soarm_aimbot.geometry.table_objects import Segmentation, segment_depth
from soarm_aimbot.geometry.tracks import MultiObjectTracker, TrackFrame, TrackState
from soarm_aimbot.geometry.transforms import RigidTransform, load_transform
from soarm_aimbot.geometry.urdf_fk import ARM_JOINTS, UrdfModel, forward_kinematics, load_urdf
from soarm_aimbot.robots.sts3215 import radians_to_tick, tick_to_radians

DEFAULT_OVERHEAD = Path("calib/overhead_intrinsics.yaml")
DEFAULT_WRIST = Path("calib/wrist_intrinsics.yaml")
DEFAULT_WORLD = Path("calib/T_world_overhead.yaml")
DEFAULT_BASE = Path("calib/T_base_world.yaml")
DEFAULT_HAND = Path("calib/T_gripper_wrist.yaml")
DEFAULT_SERVO = Path("calib/servo_map.yaml")
DEFAULT_URDF = Path("assets/so101/so101_new_calib.urdf")
DEFAULT_OUTPUT = Path("output/aimbot")

# shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll (URDF degrees)
INIT_POSE_DEG = (0.0, 0.0, 0.0, 50.0, -90.0)

# Joint angles used for the arm ignore boxes when no servo bus is connected.
_REACHABLE = np.array([0.19951978, -0.22140632, 0.90907997, 0.88273936, 0.29894303])
_PALETTE = (
    (0, 180, 255),   # orange - also used for the selected track
    (0, 200, 0),     # green
    (255, 180, 0),   # blue
    (255, 0, 180),   # pink
    (180, 80, 255),  # purple
    (0, 220, 220),   # yellow
)


@dataclass(frozen=True)
class FollowCommand:
    mode: str
    reason: str
    joints_rad: np.ndarray
    target: RigidTransform | None
    track_id: int | None
    object_base_m: np.ndarray | None
    yaw_rad: float | None


# ---------------------------------------------------------------------------
# Aim-line geometry
# ---------------------------------------------------------------------------

def aim_line_endpoint(
    ee_base: np.ndarray,
    aim_dir_base: np.ndarray,
    base_from_world: RigidTransform,
    table_z_world_m: float,
    object_base_m: np.ndarray | None,
    object_radius_m: float = 0.04,
    max_reach_m: float = 0.6,
) -> tuple[np.ndarray, str]:
    """Return (hit_point_base, hit_type) for the aim ray.

    hit_type is one of ``"object"``, ``"table"``, or ``"miss"`` (max reach).
    Checks candidates in order - closest positive-t hit wins:
      1. Sphere of ``object_radius_m`` centred at ``object_base_m`` (base frame).
      2. Table plane at ``table_z_world_m`` in world frame.
    """
    direction = np.asarray(aim_dir_base, dtype=np.float64)
    direction = direction / np.linalg.norm(direction)
    ee = np.asarray(ee_base, dtype=np.float64)

    t_best = max_reach_m
    hit: str = "miss"

    # --- object sphere (base frame) ---
    if object_base_m is not None:
        d = ee - np.asarray(object_base_m, dtype=np.float64)
        b = 2.0 * float(np.dot(d, direction))
        c = float(np.dot(d, d)) - object_radius_m ** 2
        disc = b * b - 4.0 * c          # a == 1 (unit direction)
        if disc >= 0:
            sqrt_disc = np.sqrt(disc)
            for t_cand in ((-b - sqrt_disc) / 2.0, (-b + sqrt_disc) / 2.0):
                if 0.005 < t_cand < t_best:
                    t_best = t_cand
                    hit = "object"
                    break

    # --- table plane (world frame, Z = table_z_world_m) ---
    world_from_base = base_from_world.inverse()
    ee_world = world_from_base.apply(ee)
    dir_world = world_from_base.rotation @ direction
    if abs(dir_world[2]) > 1e-6:
        t_table = (table_z_world_m - float(ee_world[2])) / float(dir_world[2])
        if 0.005 < t_table < t_best:
            t_best = t_table
            hit = "table"

    return ee + t_best * direction, hit


# ---------------------------------------------------------------------------
# Wrist overlay
# ---------------------------------------------------------------------------

def draw_wrist_aimbot(
    color: np.ndarray,
    wrist_model: RgbCameraModel,
    urdf: UrdfModel,
    joints: np.ndarray,
    gripper_from_wrist: RigidTransform,
    command: FollowCommand,
    base_from_world: RigidTransform,
    table_z_world_m: float = 0.0,
    object_radius_m: float = 0.04,
    aim_samples: int = 40,
) -> np.ndarray:
    """Draw wrist view with EE indicator, gripper-axis aim line, and first-hit endpoint.

    Aim direction is gripper -Z (the optical/approach axis). Local +/-Y cannot
    be projected: +Y has cam_z = -0.572 (always behind focal point) and -Y
    asymptotically approaches v ~ 1078 px which exceeds the extended FOV bound.
    The EE (gripper_link origin) itself is ~25 mm behind the focal point and also
    cannot be projected; the first visible sample along the aim ray is used instead.
    """
    view = np.array(color, dtype=np.uint8, copy=True)
    pose = forward_kinematics(urdf, dict(zip(ARM_JOINTS, joints)))

    # T_base_wrist = T_base_gripper @ T_gripper_wrist
    base_from_wrist = RigidTransform(pose.matrix @ gripper_from_wrist.matrix)
    ee_base = pose.translation

    # Gripper -Z is the optical axis: the camera looks along this direction and
    # points on this ray enter the FOV at roughly 90 mm from the gripper origin.
    aim_dir_base = -pose.rotation[:, 2]

    endpoint_base, hit_type = aim_line_endpoint(
        ee_base, aim_dir_base, base_from_world, table_z_world_m,
        command.object_base_m, object_radius_m,
    )
    # red = table hit, blue = object hit, orange = miss
    aim_color: tuple[int, int, int] = (
        (0, 0, 255) if hit_type == "table"
        else (255, 80, 0) if hit_type == "object"
        else (0, 100, 255)
    )

    # --- EE indicator: scan forward to find first visible sample ---
    ee_indicator_px = None
    for k in range(1, 25):
        pt = ee_base + k * 0.01 * aim_dir_base   # 10 mm steps
        px = _project_base(pt, base_from_wrist, wrist_model)
        if px is not None:
            ee_indicator_px = px
            break

    if ee_indicator_px is not None:
        cv2.circle(view, _pixel(ee_indicator_px), 7, aim_color, 2)
        cv2.putText(
            view, "EE",
            (_pixel(ee_indicator_px)[0] + 9, _pixel(ee_indicator_px)[1]),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, aim_color, 1, cv2.LINE_AA,
        )
    else:
        cv2.putText(
            view, "EE (off-screen)",
            (12, 56),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, aim_color, 1, cv2.LINE_AA,
        )

    # --- aim line: sample N points from EE to endpoint ---
    prev_px = None
    for i in range(aim_samples + 1):
        frac = i / aim_samples
        pt = ee_base + frac * (endpoint_base - ee_base)
        px = _project_base(pt, base_from_wrist, wrist_model)
        if px is not None and prev_px is not None:
            cv2.line(view, _pixel(prev_px), _pixel(px), aim_color, 2)
        prev_px = px

    # --- endpoint dot + crosshair ---
    end_px = _project_base(endpoint_base, base_from_wrist, wrist_model)
    if end_px is not None:
        ex, ey = _pixel(end_px)
        cv2.circle(view, (ex, ey), 6, aim_color, -1)
        cv2.putText(
            view, "aim",
            (ex + 9, ey),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, aim_color, 1, cv2.LINE_AA,
        )
        # four-segment crosshair (screen space, always green)
        gap, seg = 12, 20
        cv2.line(view, (ex - gap - seg, ey), (ex - gap, ey), (0, 255, 0), 2)
        cv2.line(view, (ex + gap, ey), (ex + gap + seg, ey), (0, 255, 0), 2)
        cv2.line(view, (ex, ey - gap - seg), (ex, ey - gap), (0, 255, 0), 2)
        cv2.line(view, (ex, ey + gap), (ex, ey + gap + seg), (0, 255, 0), 2)

    # --- object target ---
    if command.object_base_m is not None:
        obj_px = _project_base(command.object_base_m, base_from_wrist, wrist_model)
        if obj_px is not None:
            cv2.circle(view, _pixel(obj_px), 10, (0, 180, 255), 2)
            cv2.putText(
                view, "target",
                (_pixel(obj_px)[0] + 12, _pixel(obj_px)[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 255), 1,
            )
        else:
            cv2.putText(
                view, "target outside",
                (12, view.shape[0] - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 180, 255), 1,
            )

    cv2.putText(view, "wrist aimbot", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return view


# ---------------------------------------------------------------------------
# Stage 1: initial pose
# ---------------------------------------------------------------------------

def init_step(
    model: UrdfModel,
    q_cmd: np.ndarray,
    init_pose_rad: np.ndarray,
    speed_rad_s: float,
    dt_s: float,
    track: TrackState | None,
    base_from_world: RigidTransform,
) -> FollowCommand:
    """One Stage-1 command: ramp every joint toward the fixed initial pose.

    Returns mode ``"init"`` while moving and ``"hold"`` once the commanded
    joints equal the initial pose. ``target`` is the FK of the initial pose
    so the HUD shows where the gripper is heading.
    """
    current = np.asarray(q_cmd, dtype=np.float64).reshape(5)
    goal = np.asarray(init_pose_rad, dtype=np.float64).reshape(5)
    target = forward_kinematics(model, dict(zip(ARM_JOINTS, goal)))
    usable = track is not None and track.confirmed and track.missed_frames == 0
    track_id = None if track is None else track.track_id
    object_base = base_from_world.apply(track.position_m) if usable else None
    if _at_pose(current, goal):
        return FollowCommand("hold", "at initial pose, press s for Stage 2", goal.copy(), target, track_id, object_base, None)
    joints = limit_joint_step(current, goal, np.full(5, float(speed_rad_s)), dt_s)
    return FollowCommand("init", "moving to initial pose", joints, target, track_id, object_base, None)


def _at_pose(joints: np.ndarray, pose: np.ndarray) -> bool:
    return bool(np.allclose(joints, pose, atol=1e-6))


def _init_pose_rad(model: UrdfModel, pose_deg: tuple[float, ...]) -> np.ndarray:
    pose = np.deg2rad(np.asarray(pose_deg, dtype=np.float64))
    for name, angle in zip(ARM_JOINTS, pose):
        joint = model.joints[name]
        if joint.lower is not None and not joint.lower <= float(angle) <= joint.upper:
            raise ValueError(
                f"initial pose {name}={np.rad2deg(angle):.1f} deg is outside the URDF limits "
                f"[{np.rad2deg(joint.lower):.1f}, {np.rad2deg(joint.upper):.1f}] deg"
            )
    return pose


# ---------------------------------------------------------------------------
# Stage 2: analytical shoulder_pan solver
# ---------------------------------------------------------------------------

def solve_base_for_y(
    model: UrdfModel,
    locked_joints_1_4: np.ndarray,
    current_base_rad: float,
    target_y_base_m: float,
    speed_limit_rad_s: float,
    dt_s: float,
) -> tuple[float | None, str]:
    """Return the speed-limited shoulder_pan angle that places EE base y at target.

    With joints 1-4 locked, y_EE(theta) = c + a*cos(theta) + b*sin(theta) for
    the revolute shoulder_pan joint; a, b, c are read from FK at theta = 0,
    +/-pi/2, so the URDF's axis direction and offset are handled exactly.

    Returns (angle_rad, reason). angle_rad is None when the target y is
    outside [c - R, c + R] or every solution violates the shoulder_pan limits.
    """
    locked = np.asarray(locked_joints_1_4, dtype=np.float64).reshape(4)

    def y_at(theta: float) -> float:
        joints = np.concatenate([[theta], locked])
        return float(forward_kinematics(model, dict(zip(ARM_JOINTS, joints))).translation[1])

    y0, y_plus, y_minus = y_at(0.0), y_at(np.pi / 2), y_at(-np.pi / 2)
    c = 0.5 * (y_plus + y_minus)
    a = y0 - c
    b = 0.5 * (y_plus - y_minus)
    amplitude = float(np.hypot(a, b))
    if amplitude < 1e-4:
        return None, "gripper is on the pan axis; turning the base does not change base y"
    ratio = (float(target_y_base_m) - c) / amplitude
    if abs(ratio) > 1.0:
        return None, (
            f"target base y {target_y_base_m * 1000:.0f} mm is outside the reachable range "
            f"[{(c - amplitude) * 1000:.0f}, {(c + amplitude) * 1000:.0f}] mm"
        )
    psi = float(np.arctan2(b, a))
    alpha = float(np.arccos(ratio))

    def _wrap(angle: float) -> float:
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)

    pan = model.joints[ARM_JOINTS[0]]
    candidates = [_wrap(psi + alpha), _wrap(psi - alpha)]
    if pan.lower is not None:
        candidates = [angle for angle in candidates if pan.lower <= angle <= pan.upper]
    if not candidates:
        return None, f"target base y {target_y_base_m * 1000:.0f} mm needs a pan angle outside the limits"
    current = float(current_base_rad)
    best = min(candidates, key=lambda angle: abs(angle - current))
    max_step = float(speed_limit_rad_s) * max(float(dt_s), 0.0)
    stepped = current + float(np.clip(best - current, -max_step, max_step))
    return stepped, "tracking object (base y)"


def stage2_step(
    model: UrdfModel,
    q_cmd: np.ndarray,
    track: TrackState | None,
    locked_joints_1_4: np.ndarray | None,
    base_from_world: RigidTransform,
    pan_speed_limit_rad_s: float,
    dt_s: float,
) -> FollowCommand:
    """One Stage-2 command: pan-only trace of the object's base-frame y.

    joints_rad always keeps the locked joints 1-4; only index 0
    (shoulder_pan) may differ from ``q_cmd``.
    """
    current = np.asarray(q_cmd, dtype=np.float64).reshape(5).copy()

    if locked_joints_1_4 is None:
        return FollowCommand("hold", "Stage 2 not initialised", current, None, None, None, None)

    locked = np.asarray(locked_joints_1_4, dtype=np.float64).reshape(4)

    if track is None or not track.confirmed or track.missed_frames > 0:
        reason = "no confirmed track" if track is None else "track lost"
        return FollowCommand("hold", reason, current, None,
                             None if track is None else track.track_id, None, None)

    object_base = base_from_world.apply(track.position_m)
    target_y = float(object_base[1])

    new_pan, reason = solve_base_for_y(
        model, locked, float(current[0]), target_y,
        pan_speed_limit_rad_s, dt_s,
    )
    if new_pan is None:
        return FollowCommand("hold", reason, current, None, track.track_id, object_base, None)

    new_joints = np.concatenate([[new_pan], locked])

    # FK of the commanded joints, shown on the HUD as the target.
    target_pose = forward_kinematics(model, dict(zip(ARM_JOINTS, new_joints)))

    return FollowCommand(
        mode="follow",
        reason=reason,
        joints_rad=new_joints,
        target=target_pose,
        track_id=track.track_id,
        object_base_m=object_base,
        yaw_rad=None,
    )


# ---------------------------------------------------------------------------
# Tracking and segmentation
# ---------------------------------------------------------------------------

def resolve_track(
    frame: TrackFrame,
    requested_id: int | None,
    locked_id: int | None,
) -> tuple[TrackState | None, int | None]:
    """Keep one confirmed id. A short miss holds the lock without returning a measurement."""

    present = {item.track_id: item for item in frame.tracks}
    if locked_id is not None and locked_id not in present:
        locked_id = None
    if locked_id is not None:
        track = present[locked_id]
        if track.confirmed and track.missed_frames == 0:
            return track, locked_id
        return None, locked_id
    ready = [item for item in frame.tracks if item.confirmed and item.missed_frames == 0]
    if requested_id is not None:
        match = [item for item in ready if item.track_id == requested_id]
        if not match:
            return None, None
        return match[0], match[0].track_id
    if not ready:
        return None, None
    chosen = min(ready, key=lambda item: item.track_id)
    return chosen, chosen.track_id


def _tracker(args: argparse.Namespace) -> MultiObjectTracker:
    return MultiObjectTracker(
        match_m=args.match_mm / 1000.0,
        max_speed_m_s=args.max_speed_mm_s / 1000.0,
        max_missed=args.max_missed,
        confirm_hits=args.confirm_hits,
        meas_std_m=args.meas_std_mm / 1000.0,
        accel_std_m_s2=args.accel_std,
    )


def _segment(args, depth, color, intrinsics, world_from_overhead, boxes) -> Segmentation:
    keep = _color_mask(color, args.color_lower, args.color_upper)
    # When color filtering is active the arm is not the target color, so the color
    # mask already excludes arm points from clustering. Applying arm ignore boxes
    # on top of that can block the target when the arm hovers directly above it.
    # Without color filtering we still need the boxes to suppress arm detections.
    effective_boxes = None if keep is not None else boxes
    return segment_depth(
        depth,
        intrinsics,
        world_from_overhead,
        pixel_keep=keep,
        ignore_rois=args.ignore_roi,
        min_height_m=args.min_height_mm / 1000.0,
        max_height_m=args.max_height_mm / 1000.0,
        cluster_m=args.cluster_mm / 1000.0,
        min_points=args.min_points,
        percentile=args.percentile,
        plane_distance_m=args.plane_mm / 1000.0,
        iterations=args.plane_iterations,
        xy_min=None if args.xy is None else args.xy[0],
        xy_max=None if args.xy is None else args.xy[1],
        ignore_boxes=effective_boxes,
    )


def _color_mask(color, lower, upper):
    if lower is None and upper is None:
        return None
    if lower is None or upper is None:
        raise ValueError("color filtering needs both --color-lower and --color-upper")
    if color is None:
        raise ValueError("color filtering needs a color image")
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8)) > 0


# ---------------------------------------------------------------------------
# Joint limits and servo I/O
# ---------------------------------------------------------------------------

def limit_joint_step(current_rad: np.ndarray, goal_rad: np.ndarray, speed_limit_rad_s: np.ndarray, dt_s: float) -> np.ndarray:
    """Move each joint toward the goal by at most ``speed * dt``."""

    current = np.asarray(current_rad, dtype=np.float64).reshape(5)
    goal = np.asarray(goal_rad, dtype=np.float64).reshape(5)
    limit = np.asarray(speed_limit_rad_s, dtype=np.float64).reshape(5) * max(float(dt_s), 0.0)
    if np.any(limit < 0):
        raise ValueError("joint step limit must not be negative")
    return current + np.clip(goal - current, -limit, limit)


def _speed_limit(model: UrdfModel, scale: float) -> np.ndarray:
    if not 0 < scale <= 1:
        raise ValueError("speed scale must be in (0, 1]")
    limits = []
    for name in ARM_JOINTS:
        velocity = model.joints[name].velocity
        if velocity is None or velocity <= 0:
            raise ValueError(f"{name} has no positive URDF velocity limit")
        limits.append(float(velocity) * scale)
    return np.array(limits)


def _load_matrix(path: Path, key: str) -> RigidTransform:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or key not in payload:
        raise ValueError(f"{path} has no {key}")
    return RigidTransform(np.array(payload[key], dtype=np.float64))


def _load_table_z(world_extrinsics_path: Path) -> float:
    """Read table_z_m from a T_world_overhead yaml; 0.0 if absent."""
    payload = yaml.safe_load(Path(world_extrinsics_path).read_text(encoding="utf-8"))
    value = payload.get("table_z_m") if isinstance(payload, dict) else None
    return 0.0 if value is None else float(value)


def _load_servo_map(path: Path) -> dict:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "joints" not in payload:
        raise ValueError(f"{path} has no joints")
    for name in ARM_JOINTS:
        item = payload["joints"][name]
        if int(item["sign"]) not in (-1, 1):
            raise ValueError(f"sign of {name} must be 1 or -1")
    return payload


def _ticks_for_joints(servo_map: dict, joints: np.ndarray) -> list[int]:
    per_rev = int(servo_map.get("ticks_per_revolution", 4096))
    ticks = []
    for name, angle in zip(ARM_JOINTS, np.asarray(joints, dtype=np.float64).reshape(5)):
        item = servo_map["joints"][name]
        ticks.append(radians_to_tick(float(angle), int(item["tick_at_urdf_zero"]), int(item["sign"]), per_rev))
    return ticks


def _radians_from_ticks(servo_map: dict, ticks: dict[str, int]) -> np.ndarray:
    per_rev = int(servo_map.get("ticks_per_revolution", 4096))
    return np.array(
        [
            tick_to_radians(
                ticks[name],
                int(servo_map["joints"][name]["tick_at_urdf_zero"]),
                int(servo_map["joints"][name]["sign"]),
                per_rev,
            )
            for name in ARM_JOINTS
        ]
    )


def _read_arm_radians(bus, servo_map: dict, model: UrdfModel) -> np.ndarray:
    ticks = {name: int(bus.read_position(int(servo_map["joints"][name]["id"]))) for name in ARM_JOINTS}
    joints = _radians_from_ticks(servo_map, ticks)
    for name, angle in zip(ARM_JOINTS, joints):
        joint = model.joints[name]
        if joint.lower is not None and not joint.lower - 0.1 <= float(angle) <= joint.upper + 0.1:
            raise ValueError(f"{name}={float(angle):.4f} rad is outside the URDF limits")
    return joints


def _enable_hold(bus, servo_map: dict, joints: np.ndarray) -> None:
    for name, tick in zip(ARM_JOINTS, _ticks_for_joints(servo_map, joints)):
        bus.write_goal(int(servo_map["joints"][name]["id"]), int(tick))
    for name in ARM_JOINTS:
        bus.set_torque(int(servo_map["joints"][name]["id"]), True)


def _release_torque(bus, servo_map: dict) -> None:
    for name in ARM_JOINTS:
        bus.set_torque(int(servo_map["joints"][name]["id"]), False)


def _write_joints(bus, servo_map: dict, joints: np.ndarray) -> None:
    for name, tick in zip(ARM_JOINTS, _ticks_for_joints(servo_map, joints)):
        bus.write_goal(int(servo_map["joints"][name]["id"]), int(tick))


# ---------------------------------------------------------------------------
# Overhead view
# ---------------------------------------------------------------------------

def _draw_overhead(color, intrinsics, world_from_overhead, segmentation, tracked, track, command, height_mm, preview) -> np.ndarray:
    view = np.array(color, dtype=np.uint8, copy=True) if color is not None else np.zeros((intrinsics.height, intrinsics.width, 3), np.uint8)
    selected_pos = None if track is None else track.measured_m

    # Map each detection to its tracker entry by matching positions.
    by_measure: dict = {}
    for item in tracked.tracks:
        if item.measured_m is not None:
            by_measure[tuple(np.round(item.measured_m, 4))] = item

    for obj in segmentation.objects:
        if len(obj.pixels) == 0:
            continue
        key = tuple(np.round(obj.position_m, 4))
        obj_track = by_measure.get(key)
        is_selected = selected_pos is not None and float(np.linalg.norm(obj.position_m - selected_pos)) < 1e-3
        if is_selected:
            paint = _PALETTE[0]
        elif obj_track is not None and obj_track.confirmed:
            paint = _PALETTE[obj_track.track_id % len(_PALETTE)]
        else:
            paint = (150, 150, 150)

        # Cluster mask (blended)
        view[obj.pixels[:, 1], obj.pixels[:, 0]] = (
            0.45 * view[obj.pixels[:, 1], obj.pixels[:, 0]] + 0.55 * np.array(paint)
        ).astype(np.uint8)

        # Bounding box
        x0, y0 = int(obj.pixels[:, 0].min()), int(obj.pixels[:, 1].min())
        x1, y1 = int(obj.pixels[:, 0].max()), int(obj.pixels[:, 1].max())
        cv2.rectangle(view, (x0, y0), (x1, y1), paint, 2)

        # Coordinate label: stable track id (if confirmed) + world XY + height above table
        pos_mm = obj.position_m * 1000.0
        h_mm = obj.height_above_table_m * 1000.0
        if obj_track is not None and obj_track.confirmed:
            label = f"id{obj_track.track_id} ({pos_mm[0]:.0f},{pos_mm[1]:.0f},{h_mm:.0f})mm"
        else:
            label = f"({pos_mm[0]:.0f},{pos_mm[1]:.0f},{h_mm:.0f})mm"
        cv2.putText(view, label, (x0, max(18, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, paint, 1, cv2.LINE_AA)

    lines = ["PREVIEW" if preview else command.mode.upper(), command.reason, "q=quit  s=stage  r=release  e=torque"]
    if track is not None:
        world_mm = track.position_m * 1000.0
        speed = float(np.linalg.norm(track.velocity_m_s[:2]) * 1000.0)
        lines.append(f"id {track.track_id}  world {world_mm[0]:.0f},{world_mm[1]:.0f},{world_mm[2]:.0f} mm  {speed:.0f} mm/s")
        pixel = _project_world(track.position_m, world_from_overhead, intrinsics)
        if pixel is not None:
            center = tuple(int(round(value)) for value in pixel)
            cv2.circle(view, center, 8, _PALETTE[0], 2)
    if command.object_base_m is not None:
        base_mm = command.object_base_m * 1000.0
        lines.append(f"base {base_mm[0]:.0f},{base_mm[1]:.0f},{base_mm[2]:.0f} mm  gripper z {height_mm:.0f} mm")
    if command.target is not None:
        grip_mm = command.target.translation * 1000.0
        lines.append(f"gripper {grip_mm[0]:.0f},{grip_mm[1]:.0f},{grip_mm[2]:.0f} mm")
    _text_block(view, lines)
    confirmed = [item.track_id for item in tracked.tracks if item.confirmed]
    if confirmed and track is None:
        _text_block(view, [f"confirmed {confirmed}"], origin=(12, view.shape[0] - 16))
    return view


def _draw_stage2_overlay(view: np.ndarray, locked_joints_1_4: np.ndarray | None) -> np.ndarray:
    """Add a STAGE 2 banner and locked-joint readout to the overhead view."""
    out = view.copy()
    h = out.shape[0]
    # Red semi-transparent banner at the top
    banner = out[0:36, :].astype(np.float32)
    banner[:] = banner * 0.4 + np.array([0, 0, 180], dtype=np.float32) * 0.6
    out[0:36, :] = banner.astype(np.uint8)
    cv2.putText(out, "STAGE 2 - BASE PAN TRACE (BASE Y)", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    if locked_joints_1_4 is not None:
        names = ARM_JOINTS[1:]
        parts = [f"{n}={np.rad2deg(v):.1f}" for n, v in zip(names, locked_joints_1_4)]
        label = "locked deg: " + "  ".join(parts)
        cv2.putText(out, label, (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
    return out


def _project_world(point_world: np.ndarray, world_from_overhead: RigidTransform, intrinsics: PinholeIntrinsics) -> np.ndarray | None:
    camera_point = world_from_overhead.inverse().apply(point_world)
    if float(camera_point[2]) <= 0.02:
        return None
    projected, _ = cv2.projectPoints(
        camera_point.reshape(1, 3),
        np.zeros(3),
        np.zeros(3),
        intrinsics.camera_matrix,
        intrinsics.distortion_coeffs,
    )
    return projected.reshape(2)


def _project_base(point_base: np.ndarray, base_from_camera: RigidTransform, model: RgbCameraModel) -> np.ndarray | None:
    camera_point = base_from_camera.inverse().apply(point_base)
    if float(camera_point[2]) <= 0.02:
        return None
    pixel = model.project(camera_point.reshape(1, 3), np.zeros(3), np.zeros(3))[0]
    if not np.isfinite(pixel).all():
        return None
    if not (-200 <= pixel[0] <= model.width + 200 and -200 <= pixel[1] <= model.height + 200):
        return None
    return pixel


def _pixel(point: np.ndarray) -> tuple[int, int]:
    return int(round(float(point[0]))), int(round(float(point[1])))


def _text_block(image: np.ndarray, lines: list[str], origin: tuple[int, int] = (12, 28)) -> None:
    for index, line in enumerate(lines):
        place = (origin[0], origin[1] + index * 22)
        cv2.putText(image, line, place, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Video, log and status output
# ---------------------------------------------------------------------------

def _ensure_writer(writer, path: Path, frame: np.ndarray, fps: float):
    if writer is not None:
        return writer
    height, width = frame.shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    for codec in ("avc1", "X264", "H264", "hvc1", "mp4v"):
        opened = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), float(fps), (width, height))
        if opened.isOpened():
            return opened
        opened.release()
    raise ValueError(f"cannot create video {path}; tried avc1/X264/H264/hvc1/mp4v")


def _reencode_h264(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return
    tmp = path.with_suffix(".h264.mp4")
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", str(path), "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-movflags", "+faststart", str(tmp)],
            capture_output=True,
        )
    except FileNotFoundError:
        result = None
    if result is not None and result.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0:
        path.unlink()
        tmp.rename(path)
    else:
        tmp.unlink(missing_ok=True)
        print(f"ffmpeg H.264 re-encode failed, keeping the original {path.name} (install ffmpeg to enable it)")


def _write_log(handle, now: float, command: FollowCommand, track: TrackState | None, joints: np.ndarray, current_pose: RigidTransform) -> None:
    record = {
        "time_s": round(float(now), 3),
        "mode": command.mode,
        "reason": command.reason,
        "track_id": None if track is None else int(track.track_id),
        "object_world_m": None if track is None else _plain(track.position_m),
        "object_base_m": _plain(command.object_base_m),
        "current_gripper_xyz_m": _plain(current_pose.translation),
        "current_gripper_rot": [_plain(current_pose.rotation[i]) for i in range(3)],
        "target_xyz_m": None if command.target is None else _plain(command.target.translation),
        "target_rot": None if command.target is None else [_plain(command.target.rotation[i]) for i in range(3)],
        "yaw_rad": None if command.yaw_rad is None else round(float(command.yaw_rad), 5),
        "joints_rad": _plain(joints),
    }
    handle.write(json.dumps(record) + "\n")
    handle.flush()


def _print_status(command: FollowCommand, track: TrackState | None, current_pose: RigidTransform | None = None) -> None:
    if track is None or command.object_base_m is None:
        print(f"{command.mode}: {command.reason}")
        return
    world_mm = track.position_m * 1000.0
    base_mm = command.object_base_m * 1000.0
    now_str = ""
    if current_pose is not None:
        t = current_pose.translation * 1000.0
        z = current_pose.rotation[:, 2]
        now_str = (
            f"  gripper ({t[0]:.0f},{t[1]:.0f},{t[2]:.0f})mm"
            f" +Z[{z[0]:.2f},{z[1]:.2f},{z[2]:.2f}]"
        )
    target_str = ""
    if command.target is not None:
        g = command.target.translation * 1000.0
        gz = command.target.rotation[:, 2]
        target_str = (
            f"  target ({g[0]:.0f},{g[1]:.0f},{g[2]:.0f})mm"
            f" +Z[{gz[0]:.2f},{gz[1]:.2f},{gz[2]:.2f}]"
        )
    print(
        f"{command.mode}: {command.reason}  id {track.track_id}  "
        f"obj world ({world_mm[0]:.0f},{world_mm[1]:.0f},{world_mm[2]:.0f})mm"
        f"  base ({base_mm[0]:.0f},{base_mm[1]:.0f},{base_mm[2]:.0f})mm"
        f"{now_str}{target_str}"
    )


def _plain(value) -> list[float] | None:
    if value is None:
        return None
    return [round(float(item), 6) for item in np.asarray(value, dtype=np.float64).reshape(-1)]


def _has_display() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except CameraError as exc:
        print(f"Camera error: {exc}")
        return 1
    except (ValueError, FileNotFoundError, KeyError, RuntimeError) as exc:
        print(f"Could not run the aimbot: {exc}")
        return 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aimbot-wrist",
        description=(
            "Track a table object with the overhead camera, pan the arm base to follow it, "
            "and draw the gripper's aim line on the wrist camera."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("self-check", help="check the pan solver and aim-line geometry with the URDF; no cameras or servos")
    check.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    check.set_defaults(func=_cmd_self_check)
    run = sub.add_parser("run", help="open both cameras, move to the initial pose, press s to start tracking")
    run.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    run.add_argument("--overhead-intrinsics", type=Path, default=DEFAULT_OVERHEAD)
    run.add_argument("--wrist-intrinsics", type=Path, default=DEFAULT_WRIST)
    run.add_argument("--world-extrinsics", type=Path, default=DEFAULT_WORLD)
    run.add_argument("--base-extrinsics", type=Path, default=DEFAULT_BASE)
    run.add_argument("--hand-eye", type=Path, default=DEFAULT_HAND)
    run.add_argument("--servo-map", type=Path, default=DEFAULT_SERVO)
    run.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    run.add_argument("--port", default=None, help="Feetech bus serial port, e.g. /dev/ttyACM0")
    run.add_argument("--baud", type=int, default=1000000)
    run.add_argument("--preview", action="store_true", help="display and record only; never enable torque or write the servos")
    run.add_argument(
        "--init-pose-deg",
        type=_pose_deg,
        default=INIT_POSE_DEG,
        help="Stage 1 joint angles pan,lift,elbow,wrist_flex,wrist_roll in URDF degrees (default 0,0,0,50,-90)",
    )
    run.add_argument("--init-speed-deg-s", type=float, default=30.0, help="max speed of each joint while moving to the initial pose, deg/s")
    run.add_argument("--track-id", type=int, default=None, help="track id to follow; default is the first confirmed track")
    run.add_argument("--device", type=int, default=None, help="/dev/video index of the wrist camera")
    run.add_argument("--serial", default=None, help="serial number of the overhead RealSense")
    run.add_argument("--name-substring", default="455")
    run.add_argument("--warmup", type=int, default=10)
    run.add_argument("--median-ksize", type=int, default=5)
    run.add_argument("--speed-scale", type=float, default=0.8, help="Stage 2 pan speed as a fraction of the URDF velocity limit")
    run.add_argument("--max-step-s", type=float, default=0.05, help="longest time used for one control step, avoids jumps after a stall")
    run.add_argument("--arm-radius-mm", type=float, default=45.0, help="radius of the arm links removed from the depth image")
    run.add_argument("--object-radius-mm", type=float, default=40.0, help="radius of the sphere around the target that the aim ray can hit")
    run.add_argument("--min-height-mm", type=float, default=5.0)
    run.add_argument("--max-height-mm", type=float, default=300.0)
    run.add_argument("--cluster-mm", type=float, default=15.0)
    run.add_argument("--min-points", type=int, default=80)
    run.add_argument("--percentile", type=float, default=2.0)
    run.add_argument("--plane-mm", type=float, default=8.0)
    run.add_argument("--plane-iterations", type=int, default=80)
    run.add_argument("--xy", type=_xy_bounds, default=None, help="table area xmin,ymin,xmax,ymax in world metres")
    run.add_argument("--ignore-roi", action="append", type=_roi, default=None, help="overhead pixel box x,y,w,h to ignore; repeatable")
    run.add_argument(
        "--color-lower",
        type=_hsv,
        default=None,
        help="HSV lower bound H,S,V (H 0-179, S/V 0-255); use with --color-upper, "
        "e.g. a red block: --color-lower 0,120,60 --color-upper 15,255,255",
    )
    run.add_argument("--color-upper", type=_hsv, default=None, help="HSV upper bound H,S,V; needs --color-lower")
    run.add_argument("--match-mm", type=float, default=80.0)
    run.add_argument("--max-speed-mm-s", type=float, default=500.0)
    run.add_argument("--max-missed", type=int, default=15)
    run.add_argument("--confirm-hits", type=int, default=3)
    run.add_argument("--meas-std-mm", type=float, default=10.0)
    run.add_argument("--accel-std", type=float, default=5.0)
    run.add_argument("--frames", type=int, default=None, help="exit after this many frames; required without a display")
    run.add_argument("--no-window", action="store_true")
    run.add_argument("--video-fps", type=float, default=20.0)
    run.set_defaults(func=_cmd_run)
    return parser


def _cmd_self_check(args: argparse.Namespace) -> int:
    model = load_urdf(args.urdf)
    init_pose = _init_pose_rad(model, INIT_POSE_DEG)
    identity = RigidTransform(np.eye(4))

    joints = _REACHABLE.copy()
    for _ in range(400):
        command = init_step(model, joints, init_pose, np.deg2rad(30.0), 0.05, None, identity)
        if command.mode == "hold":
            break
        joints = command.joints_rad
    else:
        print("Self-check failed: Stage 1 never reached the initial pose.")
        return 1

    locked = init_pose[1:]
    goal_pan = 0.3
    target_y = float(forward_kinematics(model, dict(zip(ARM_JOINTS, np.concatenate([[goal_pan], locked])))).translation[1])
    pan, reason = solve_base_for_y(model, locked, 0.0, target_y, 10.0, 1.0)
    if pan is None:
        print(f"Self-check failed: the pan solver found no angle ({reason}).")
        return 1
    reached_y = float(forward_kinematics(model, dict(zip(ARM_JOINTS, np.concatenate([[pan], locked])))).translation[1])
    if abs(reached_y - target_y) > 1e-6:
        print(f"Self-check failed: the pan solver missed base y by {abs(reached_y - target_y) * 1000:.3f} mm.")
        return 1
    stepped, _ = solve_base_for_y(model, locked, 0.0, target_y, 1.0, 0.05)
    if stepped is None or abs(stepped) > 0.05 + 1e-9:
        print("Self-check failed: one pan step exceeded the speed limit.")
        return 1
    unreachable, _ = solve_base_for_y(model, locked, 0.0, 5.0, 1.0, 0.05)
    if unreachable is not None:
        print("Self-check failed: an unreachable base y returned a pan angle.")
        return 1

    ee = np.array([0.2, 0.0, 0.2])
    down = np.array([0.0, 0.0, -1.0])
    table_hit, table_kind = aim_line_endpoint(ee, down, identity, 0.0, None)
    object_hit, object_kind = aim_line_endpoint(ee, down, identity, 0.0, np.array([0.2, 0.0, 0.05]))
    _, miss_kind = aim_line_endpoint(ee, -down, identity, 0.0, None)
    if table_kind != "table" or not np.allclose(table_hit, [0.2, 0.0, 0.0]):
        print(f"Self-check failed: the aim ray should hit the table at (200, 0, 0) mm, got {table_kind} {table_hit * 1000}.")
        return 1
    if object_kind != "object" or not np.allclose(object_hit, [0.2, 0.0, 0.09]):
        print(f"Self-check failed: the aim ray should hit the object sphere at z = 90 mm, got {object_kind} {object_hit * 1000}.")
        return 1
    if miss_kind != "miss":
        print("Self-check failed: an upward aim ray should miss.")
        return 1

    print(
        "Self-check passed: Stage 1 reaches the initial pose, the pan solver hits the target base y "
        "within the speed limit, and the aim ray hits the object before the table."
    )
    print("No cameras or servos were opened. Use run for the real arm.")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    if args.speed_scale <= 0 or args.speed_scale > 1:
        raise ValueError("speed scale must be in (0, 1]")
    if args.max_step_s <= 0:
        raise ValueError("--max-step-s must be positive")
    if args.init_speed_deg_s <= 0:
        raise ValueError("--init-speed-deg-s must be positive")
    if args.object_radius_mm <= 0:
        raise ValueError("--object-radius-mm must be positive")
    if args.video_fps <= 0:
        raise ValueError("--video-fps must be positive")

    show = _has_display() and not args.no_window
    if not show and args.frames is None:
        raise CameraError("no display found. Add --frames to save the video and exit after that many frames.")
    if not args.preview and not args.port:
        raise ValueError("driving the arm needs --port. Add --preview to only view and record.")

    model = load_urdf(args.urdf)
    tool = tcp_offset_m(model)
    init_pose = _init_pose_rad(model, args.init_pose_deg)
    init_speed = float(np.deg2rad(args.init_speed_deg_s))
    overhead_model, overhead_meta = load_intrinsics(args.overhead_intrinsics)
    world_from_overhead, _ = load_transform(args.world_extrinsics)
    base_from_world = _load_matrix(args.base_extrinsics, "T_base_world")
    gripper_from_wrist = _load_matrix(args.hand_eye, "T_gripper_wrist")
    wrist_model, _ = load_rgb_intrinsics(args.wrist_intrinsics)
    servo_map = _load_servo_map(args.servo_map) if args.port else None
    speed_limit = _speed_limit(model, args.speed_scale)
    table_z = _load_table_z(args.world_extrinsics)
    object_radius = args.object_radius_mm / 1000.0

    run_dir = args.output_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = run_dir / "log.jsonl"

    from soarm_aimbot.cameras.realsense import RealSenseRgbd
    from soarm_aimbot.cameras.rgb import RgbCamera, choose_device, list_v4l2_devices
    from soarm_aimbot.robots.sts3215 import Sts3215Bus

    overhead = RealSenseRgbd(
        width=overhead_model.width,
        height=overhead_model.height,
        fps=int(overhead_meta.get("fps", 30)),
        serial=args.serial or overhead_meta.get("serial"),
        name_substring=args.name_substring,
    )
    wrist_index = choose_device(list_v4l2_devices(), args.device)
    wrist = RgbCamera(wrist_index, wrist_model.width, wrist_model.height)

    bus = None
    torque_on = False
    overhead_writer = None
    wrist_writer = None

    # Tracking state (shared across stages)
    tracker = _tracker(args)
    locked_id = None

    # Stage state
    stage = 1                      # 1 = move to / hold initial pose,  2 = base pan trace
    locked_joints_1_4: np.ndarray | None = None   # joints 1-4 frozen in Stage 2

    last_reason = ""
    last_print = 0.0

    try:
        overhead.start()
        wrist.start()
        for _ in range(max(args.warmup, 0)):
            overhead.read()
            wrist.read()

        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"Writing videos and log to {run_dir}")

        q_cmd = _REACHABLE.copy()

        if args.port and servo_map is not None:
            bus = Sts3215Bus(args.port, args.baud)
            q_cmd = _read_arm_radians(bus, servo_map, model)
            if not args.preview:
                _enable_hold(bus, servo_map, q_cmd)
                torque_on = True
                print(
                    f"Torque on. Moving to the initial pose {np.round(args.init_pose_deg, 1).tolist()} deg "
                    f"at {args.init_speed_deg_s:.0f} deg/s."
                )
                print("Once there, place the object and press s for Stage 2; press s again to return to the initial pose.")
                print("Press q or Ctrl-C to stop, r to release torque, e to re-enable torque.")
            else:
                print("Preview mode: torque stays off and no goal positions are written.")
        elif args.preview:
            print("No --port given; the preview uses example joint angles instead of the real arm.")

        previous = time.monotonic()
        frames = 0

        with log_path.open("w", encoding="utf-8") as log:
            while args.frames is None or frames < args.frames:
                frame = overhead.read()
                wrist_bgr = wrist.read()
                now = time.monotonic()
                dt = min(max(now - previous, 0.0), args.max_step_s)
                previous = now

                # -- Detection ---------------------------------------------
                depth = spatial_median(frame.depth_m, args.median_ksize)
                pose_fk = forward_kinematics(model, dict(zip(ARM_JOINTS, q_cmd)))
                tip = pose_fk.translation + pose_fk.rotation @ tool
                boxes = arm_ignore_boxes(
                    model, q_cmd, base_from_world.inverse(), tip,
                    args.arm_radius_mm / 1000.0,
                )
                segmentation = _segment(args, depth, frame.color_bgr, overhead_model, world_from_overhead, boxes)
                positions = (
                    np.stack([item.position_m for item in segmentation.objects])
                    if segmentation.objects
                    else np.zeros((0, 3), dtype=np.float64)
                )
                timestamp = float(frame.timestamp_s) if frame.timestamp_s > 0 else now
                tracked = tracker.update(positions, timestamp)
                track, locked_id = resolve_track(tracked, args.track_id, locked_id)

                # -- Control -----------------------------------------------
                if stage == 1:
                    command = init_step(model, q_cmd, init_pose, init_speed, dt, track, base_from_world)
                    if command.mode == "init" and not args.preview and bus is not None and servo_map is not None:
                        _write_joints(bus, servo_map, command.joints_rad)
                        q_cmd = command.joints_rad.copy()
                        if _at_pose(q_cmd, init_pose):
                            print("Initial pose reached, torque holding. Place the object and press s to start Stage 2.")
                else:
                    command = stage2_step(
                        model, q_cmd, track, locked_joints_1_4,
                        base_from_world, speed_limit[0], dt,
                    )
                    if command.mode == "follow" and not args.preview and bus is not None and servo_map is not None:
                        _write_joints(bus, servo_map, command.joints_rad)
                        q_cmd = command.joints_rad.copy()

                # -- Visualisation -----------------------------------------
                overhead_view = _draw_overhead(
                    frame.color_bgr, overhead_model, world_from_overhead,
                    segmentation, tracked, track, command,
                    float(pose_fk.translation[2]) * 1000.0, args.preview,
                )
                if stage == 2:
                    overhead_view = _draw_stage2_overlay(overhead_view, locked_joints_1_4)

                wrist_view = draw_wrist_aimbot(
                    wrist_bgr, wrist_model, model, q_cmd,
                    gripper_from_wrist, command,
                    base_from_world, table_z, object_radius,
                )

                overhead_writer = _ensure_writer(overhead_writer, run_dir / "overhead.mp4", overhead_view, args.video_fps)
                wrist_writer = _ensure_writer(wrist_writer, run_dir / "wrist.mp4", wrist_view, args.video_fps)
                overhead_writer.write(overhead_view)
                wrist_writer.write(wrist_view)

                current_gripper_pose = forward_kinematics(model, dict(zip(ARM_JOINTS, q_cmd)))
                _write_log(log, now, command, track, q_cmd, current_gripper_pose)

                if command.reason != last_reason or now - last_print > 1.0:
                    _print_status(command, track, current_gripper_pose)
                    last_reason = command.reason
                    last_print = now

                frames += 1

                # -- Key handling ------------------------------------------
                if show:
                    cv2.imshow("overhead", overhead_view)
                    cv2.imshow("wrist aimbot", wrist_view)
                    key = int(cv2.waitKey(1)) & 0xFF

                    if key in (ord("q"), 27):
                        print("Stopping.")
                        break

                    elif key == ord("s"):
                        if stage == 1:
                            driving = not args.preview and bus is not None
                            if driving and not torque_on:
                                print("Torque is off. Press e to enable it, wait for the initial pose, then press s.")
                            elif driving and not _at_pose(q_cmd, init_pose):
                                print("Not at the initial pose yet. Wait for the arm to stop, then press s.")
                            else:
                                # Joints 1-4 are at the initial pose and held by torque
                                locked_joints_1_4 = q_cmd[1:].copy()
                                stage = 2
                                print(
                                    "Stage 2: base pan tracks the object. Locked joints: "
                                    f"{dict(zip(ARM_JOINTS[1:], np.round(np.rad2deg(locked_joints_1_4), 1).tolist()))} deg"
                                )
                        else:
                            # Back to Stage 1: the arm returns to the initial pose
                            locked_joints_1_4 = None
                            stage = 1
                            print("Back to Stage 1, returning to the initial pose.")

                    elif key == ord("r"):
                        if bus is not None and servo_map is not None:
                            _release_torque(bus, servo_map)
                            torque_on = False
                            print("Torque released. The arm can be moved by hand.")
                        else:
                            print("No servo bus connected; cannot change torque.")

                    elif key == ord("e"):
                        if bus is not None and servo_map is not None and not args.preview:
                            try:
                                current_joints = _read_arm_radians(bus, servo_map, model)
                            except ValueError as exc:
                                print(f"Could not read joint angles, torque stays off: {exc}")
                            else:
                                _enable_hold(bus, servo_map, current_joints)
                                q_cmd = current_joints.copy()
                                torque_on = True
                                print("Torque enabled at the current position.")
                        elif args.preview:
                            print("Preview mode never enables torque.")
                        else:
                            print("No servo bus connected; cannot change torque.")

    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        if overhead_writer is not None:
            overhead_writer.release()
            _reencode_h264(run_dir / "overhead.mp4")
        if wrist_writer is not None:
            wrist_writer.release()
            _reencode_h264(run_dir / "wrist.mp4")
        if show:
            cv2.destroyAllWindows()
        wrist.stop()
        overhead.stop()
        if bus is not None:
            if torque_on and servo_map is not None:
                _release_torque(bus, servo_map)
                print("Torque released.")
            bus.close()

    print(f"Videos and log are in {run_dir}")
    return 0


# ---------------------------------------------------------------------------
# Argument parsers
# ---------------------------------------------------------------------------

def _pose_deg(text: str) -> tuple[float, ...]:
    return _floats(text, 5, "initial pose must be pan,lift,elbow,wrist_flex,wrist_roll in degrees")


def _xy_bounds(text: str):
    values = _floats(text, 4, "table area must be xmin,ymin,xmax,ymax in metres")
    return (values[0], values[1]), (values[2], values[3])


def _hsv(text: str) -> tuple[int, int, int]:
    values = _floats(text, 3, "HSV must be H,S,V")
    triple = tuple(int(value) for value in values)
    if not (0 <= triple[0] <= 179 and 0 <= triple[1] <= 255 and 0 <= triple[2] <= 255):
        raise argparse.ArgumentTypeError("H must be 0-179, S and V must be 0-255")
    return triple  # type: ignore[return-value]


def _roi(text: str) -> tuple[int, int, int, int]:
    parts = text.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("ignore box must be x,y,w,h")
    return tuple(int(part) for part in parts)  # type: ignore[return-value]


def _floats(text: str, count: int, message: str) -> tuple[float, ...]:
    parts = text.split(",")
    if len(parts) != count:
        raise argparse.ArgumentTypeError(message)
    try:
        return tuple(float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(message) from exc


if __name__ == "__main__":
    raise SystemExit(main())
