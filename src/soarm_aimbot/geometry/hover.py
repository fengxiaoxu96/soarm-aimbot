"""Hover pose of the gripper: fingertips above an object, approach axis down.

This does not solve inverse kinematics and does not command the arm. It builds
T_base_gripper and the wrist-image pixel that pose predicts.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from soarm_aimbot.geometry.rgb_model import RgbCameraModel
from soarm_aimbot.geometry.transforms import RigidTransform
from soarm_aimbot.geometry.urdf_fk import UrdfModel

DOWN = np.array([0.0, 0.0, -1.0])
NAMED_AXES = {
    "+X": np.array([1.0, 0.0, 0.0]),
    "-X": np.array([-1.0, 0.0, 0.0]),
    "+Y": np.array([0.0, 1.0, 0.0]),
    "-Y": np.array([0.0, -1.0, 0.0]),
    "+Z": np.array([0.0, 0.0, 1.0]),
    "-Z": np.array([0.0, 0.0, -1.0]),
}


@dataclass(frozen=True)
class HoverPlan:
    base_from_gripper: RigidTransform
    yaw_deg: float
    height_m: float
    fingertip_m: np.ndarray
    object_m: np.ndarray
    pixel_uv: np.ndarray
    approach_axis: str
    approach_in_base: np.ndarray


def tcp_offset_m(model: UrdfModel) -> np.ndarray:
    """Fingertip point in gripper_link, from the fixed gripper_frame joint."""

    try:
        joint = model.joints["gripper_frame_joint"]
    except KeyError as exc:
        raise ValueError("The URDF has no gripper_frame_joint. Pass --tool-xyz-mm measured with a ruler.") from exc
    return joint.origin.translation.copy()


def nearest_axis(direction: np.ndarray) -> str:
    vector = np.asarray(direction, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        raise ValueError("The direction vector is zero, so the approach axis is undefined")
    unit = vector / norm
    name = max(NAMED_AXES, key=lambda item: float(np.dot(unit, NAMED_AXES[item])))
    return name


def nearest_axis_from_rotation(rotation: np.ndarray) -> str:
    """Which gripper axis, as a column of R_base_gripper, is closest to straight down."""

    best_name = "+Z"
    best_dot = -2.0
    for name, axis in NAMED_AXES.items():
        column = rotation @ axis
        score = float(np.dot(column, DOWN))
        if score > best_dot:
            best_dot = score
            best_name = name
    return best_name


def plan_hover(
    *,
    current_rotation: np.ndarray,
    approach_axis: str,
    tool_in_gripper_m: np.ndarray,
    object_in_base_m: np.ndarray,
    velocity_in_base_m_s: np.ndarray,
    reach_s: float,
    height_m: float,
    height_max_m: float,
    height_step_m: float,
    gripper_from_camera: RigidTransform,
    camera: RgbCameraModel,
    work_box_m: tuple[float, float, float, float, float],
    margin: float = 0.15,
    yaw_step_deg: float = 15.0,
    yaw_limit_deg: float = 90.0,
) -> HoverPlan:
    """Point the approach axis straight down and keep the object inside the wrist image.

    ``height_m`` is measured from the object point up to the fingertip. A moving
    object shifts only in x and y by ``velocity * reach_s``. Yaw is the extra
    spin about vertical after the smallest rotation that aims the approach axis
    downward from the current gripper orientation.
    """

    if approach_axis not in NAMED_AXES:
        known = ", ".join(NAMED_AXES)
        raise ValueError(f"The approach axis must be one of {known}, got {approach_axis}")
    if height_m <= 0 or height_max_m < height_m or height_step_m <= 0:
        raise ValueError("The hover height must be positive and the maximum must not be below the start height")
    if reach_s < 0:
        raise ValueError("The expected motion time must not be negative")
    if not 0 <= margin < 0.5:
        raise ValueError("The image margin fraction must be between 0 and 0.5")
    approach = NAMED_AXES[approach_axis]
    predicted = np.asarray(object_in_base_m, dtype=np.float64).reshape(3).copy()
    velocity = np.asarray(velocity_in_base_m_s, dtype=np.float64).reshape(3)
    predicted[0] += velocity[0] * reach_s
    predicted[1] += velocity[1] * reach_s
    tool = np.asarray(tool_in_gripper_m, dtype=np.float64).reshape(3)
    seed = _aim_down(np.asarray(current_rotation, dtype=np.float64), approach)
    heights = _heights(height_m, height_max_m, height_step_m)
    for height in heights:
        chosen = _best_yaw(
            seed,
            approach,
            approach_axis,
            tool,
            predicted,
            height,
            gripper_from_camera,
            camera,
            margin,
            yaw_step_deg,
            yaw_limit_deg,
        )
        if chosen is None:
            continue
        _require_inside_box(chosen, work_box_m)
        return chosen
    raise ValueError(
        "Up to the maximum hover height, the object still falls outside the wrist image. "
        "Increase --height-max-mm, or check the hand-eye calibration and the approach axis."
    )


def _aim_down(current_rotation: np.ndarray, approach: np.ndarray) -> np.ndarray:
    current = current_rotation @ approach
    return _rotation_between(current, DOWN) @ current_rotation


def _best_yaw(
    seed: np.ndarray,
    approach: np.ndarray,
    approach_axis: str,
    tool: np.ndarray,
    object_m: np.ndarray,
    height: float,
    gripper_from_camera: RigidTransform,
    camera: RgbCameraModel,
    margin: float,
    yaw_step_deg: float,
    yaw_limit_deg: float,
) -> HoverPlan | None:
    valid: list[tuple[float, float, HoverPlan]] = []
    for yaw in _yaw_candidates(yaw_step_deg, yaw_limit_deg):
        rotation = _yaw_matrix(yaw) @ seed
        aimed = rotation @ approach
        if float(np.dot(aimed, DOWN)) < 0.999:
            continue
        fingertip = object_m + np.array([0.0, 0.0, height])
        origin = fingertip - rotation @ tool
        pose = RigidTransform.from_rotation_translation(rotation, origin)
        pixel, depth = _project_object(pose, gripper_from_camera, camera, object_m)
        if pixel is None or depth <= 0.02:
            continue
        if not _inside(pixel, camera.width, camera.height, margin):
            continue
        clearance = _border_clearance(pixel, camera.width, camera.height)
        valid.append((abs(yaw), -clearance, _plan(pose, yaw, height, fingertip, object_m, pixel, approach_axis, aimed)))
    if not valid:
        return None
    valid.sort(key=lambda item: (item[0], item[1]))
    return valid[0][2]


def _plan(pose, yaw, height, fingertip, object_m, pixel, approach_axis, aimed) -> HoverPlan:
    return HoverPlan(
        base_from_gripper=pose,
        yaw_deg=float(yaw),
        height_m=float(height),
        fingertip_m=fingertip.copy(),
        object_m=object_m.copy(),
        pixel_uv=pixel.copy(),
        approach_axis=approach_axis,
        approach_in_base=aimed.copy(),
    )


def _project_object(base_from_gripper, gripper_from_camera, camera, object_m):
    base_from_camera = RigidTransform(base_from_gripper.matrix @ gripper_from_camera.matrix)
    point = base_from_camera.inverse().apply(object_m)
    if float(point[2]) <= 0:
        return None, float(point[2])
    pixel = camera.project(point.reshape(1, 3), np.zeros(3), np.zeros(3))[0]
    return pixel, float(point[2])


def _require_inside_box(plan: HoverPlan, work_box_m: tuple[float, float, float, float, float]) -> None:
    xmin, xmax, ymin, ymax, zmin = work_box_m
    points = (plan.fingertip_m, plan.base_from_gripper.translation)
    for point in points:
        if not (xmin <= point[0] <= xmax and ymin <= point[1] <= ymax and point[2] >= zmin):
            raise ValueError(
                "Target is outside the work box: "
                f"({point[0] * 1000:.1f}, {point[1] * 1000:.1f}, {point[2] * 1000:.1f}) mm. "
                "The work box is xmin,xmax,ymin,ymax,zmin in base_link."
            )


def _heights(start: float, limit: float, step: float) -> list[float]:
    values = []
    height = start
    while height <= limit + 1e-9:
        values.append(height)
        height += step
    return values


def _yaw_candidates(step: float, limit: float) -> list[float]:
    if step <= 0 or limit < 0:
        raise ValueError("The search step about the vertical axis must be positive")
    values = [0.0]
    yaw = step
    while yaw <= limit + 1e-9:
        values.append(yaw)
        values.append(-yaw)
        yaw += step
    return values


def _yaw_matrix(yaw_deg: float) -> np.ndarray:
    return cv2.Rodrigues(np.array([0.0, 0.0, np.deg2rad(yaw_deg)]))[0]


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
        axis = axis / np.linalg.norm(axis)
        return cv2.Rodrigues(axis * np.pi)[0]
    return cv2.Rodrigues(cross / sine * np.arctan2(sine, cosine))[0]


def _inside(pixel: np.ndarray, width: int, height: int, margin: float) -> bool:
    u, v = float(pixel[0]), float(pixel[1])
    return margin * width <= u <= (1.0 - margin) * width and margin * height <= v <= (1.0 - margin) * height


def _border_clearance(pixel: np.ndarray, width: int, height: int) -> float:
    u, v = float(pixel[0]), float(pixel[1])
    return min(u, v, width - 1 - u, height - 1 - v)
