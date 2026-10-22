"""Wrist-camera horizontal correction at the hover height.

A raw pixel becomes a ray. The ray meets the plane at the overhead object
height. The fingertip target moves to that point; the gripper rotation stays.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from soarm_aimbot.geometry.rgb_model import RgbCameraModel
from soarm_aimbot.geometry.transforms import RigidTransform


@dataclass(frozen=True)
class PlaneHit:
    point_m: np.ndarray
    distance_m: float


def ray_plane_hit(base_from_camera: RigidTransform, ray_in_camera: np.ndarray, plane_z_m: float) -> PlaneHit:
    """Intersect a camera ray with the horizontal plane z = plane_z_m in the base."""

    direction = base_from_camera.rotation @ np.asarray(ray_in_camera, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        raise ValueError("The pixel ray has zero length")
    direction = direction / norm
    if abs(float(direction[2])) < 1e-6:
        raise ValueError("The ray is parallel to the top plane and does not intersect it")
    distance = (plane_z_m - float(base_from_camera.translation[2])) / float(direction[2])
    if distance <= 0:
        raise ValueError("The ray intersection is behind the camera")
    return PlaneHit(base_from_camera.translation + distance * direction, distance)


def project_point(camera: RgbCameraModel, base_from_camera: RigidTransform, point_m: np.ndarray) -> np.ndarray | None:
    point = base_from_camera.inverse().apply(np.asarray(point_m, dtype=np.float64).reshape(3))
    if float(point[2]) <= 0.02:
        return None
    return camera.project(point.reshape(1, 3), np.zeros(3), np.zeros(3))[0]


def translate_target(
    rotation: np.ndarray,
    tool_m: np.ndarray,
    object_xy_m: np.ndarray,
    plane_z_m: float,
    height_m: float,
) -> tuple[RigidTransform, np.ndarray]:
    """Keep the gripper rotation and place the fingertip above a new horizontal position."""

    fingertip = np.array([float(object_xy_m[0]), float(object_xy_m[1]), plane_z_m + height_m], dtype=np.float64)
    origin = fingertip - np.asarray(rotation, dtype=np.float64) @ np.asarray(tool_m, dtype=np.float64).reshape(3)
    return RigidTransform.from_rotation_translation(rotation, origin), fingertip


def mask_centroid(mask: np.ndarray) -> np.ndarray | None:
    image = mask if mask.ndim == 2 else mask[:, :, 0]
    ys, xs = np.nonzero(image)
    if xs.size == 0:
        return None
    return np.array([float(np.mean(xs)), float(np.mean(ys))], dtype=np.float64)


def next_action(horizontal_m: float, previous_m: float | None, tolerance_m: float, corrections_done: int) -> str:
    """Return aligned, correct, or stopped.

    A result inside the planar gate is aligned. Another move is issued
    only while the residual is still outside that gate, has dropped to 70
    percent or less of the previous one, and fewer than two moves were sent.
    """

    if horizontal_m <= tolerance_m:
        return "aligned"
    if corrections_done >= 2:
        return "stopped"
    if previous_m is not None and horizontal_m > 0.7 * previous_m:
        return "stopped"
    return "correct"
