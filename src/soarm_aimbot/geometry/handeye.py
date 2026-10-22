"""Eye-to-hand calibration: a fixed overhead camera and a tag on the gripper.

The solved transform X = T_base_overhead satisfies A X = X B for relative
motions of the gripper and of the tag. OpenCV 5.0's Python package does not
export ``calibrateHandEye``, so Park's method and Andreff's linear method are
implemented here and compared with each other.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from soarm_aimbot.geometry.transforms import RigidTransform, rotation_angle_deg, translation_distance_m


@dataclass(frozen=True)
class HandEyeSolution:
    method: str
    base_from_camera: RigidTransform
    pair_count: int


def solve_eye_to_hand(
    base_from_gripper: list[RigidTransform],
    camera_from_tag: list[RigidTransform],
    *,
    method: str = "park",
    min_rotation_deg: float = 5.0,
) -> HandEyeSolution:
    """Return T_base_camera.

    ``base_from_gripper`` is the forward-kinematics pose. ``camera_from_tag``
    is the AprilTag pose in the overhead camera. Both lists are the same poses
    in the same order.
    """

    if len(base_from_gripper) != len(camera_from_tag):
        raise ValueError("The number of joint poses and marker poses differ")
    if len(base_from_gripper) < 4:
        raise ValueError("Eye-to-hand needs at least 4 poses; 15 to 20 are recommended")
    pairs = _relative_pairs(base_from_gripper, camera_from_tag, min_rotation_deg)
    if len(pairs) < 3:
        raise ValueError("Too few useful rotations. Consecutive poses need a clear gripper rotation, not just a translation.")
    solvers = {"park": _solve_park, "andreff": _solve_andreff}
    try:
        solver = solvers[method]
    except KeyError as exc:
        raise ValueError("The solver method must be park or andreff") from exc
    rotation, translation = solver(pairs)
    return HandEyeSolution(
        method=method,
        base_from_camera=RigidTransform.from_rotation_translation(rotation, translation),
        pair_count=len(pairs),
    )


@dataclass(frozen=True)
class EyeInHandSolution:
    method: str
    gripper_from_camera: RigidTransform
    pair_count: int


def solve_eye_in_hand(
    base_from_gripper: list[RigidTransform],
    camera_from_tag: list[RigidTransform],
    *,
    method: str = "park",
    min_rotation_deg: float = 5.0,
) -> EyeInHandSolution:
    """Return X = T_gripper_camera for a camera fixed on the gripper.

    The tag is fixed in the base. With G = T_base_gripper and C = T_camera_tag,

        inv(G_j) G_i X = X C_j inv(C_i)

    which is the same AX = XB equation as eye-to-hand, after the gripper poses
    are inverted. ``gripper_from_camera`` maps wrist-camera points into ``gripper_link``.
    """

    if len(base_from_gripper) != len(camera_from_tag):
        raise ValueError("The number of joint poses and marker poses differ")
    if len(base_from_gripper) < 4:
        raise ValueError("Eye-in-hand needs at least 4 poses; about 20 are recommended")
    inverted = [pose.inverse() for pose in base_from_gripper]
    solved = solve_eye_to_hand(
        inverted,
        camera_from_tag,
        method=method,
        min_rotation_deg=min_rotation_deg,
    )
    return EyeInHandSolution(
        method=solved.method,
        gripper_from_camera=solved.base_from_camera,
        pair_count=solved.pair_count,
    )


def solutions_agree(
    first: HandEyeSolution,
    second: HandEyeSolution,
    *,
    translation_m: float,
    rotation_deg: float,
) -> tuple[float, float, bool]:
    distance = translation_distance_m(first.base_from_camera, second.base_from_camera)
    angle = rotation_angle_deg(first.base_from_camera, second.base_from_camera)
    return distance, angle, distance <= translation_m and angle <= rotation_deg


def _relative_pairs(
    base_from_gripper: list[RigidTransform],
    camera_from_tag: list[RigidTransform],
    min_rotation_deg: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    pairs = []
    for index, gripper_i in enumerate(base_from_gripper):
        for later in range(index + 1, len(base_from_gripper)):
            motion_a = base_from_gripper[later].matrix @ np.linalg.inv(gripper_i.matrix)
            motion_b = camera_from_tag[later].matrix @ np.linalg.inv(camera_from_tag[index].matrix)
            angle = rotation_angle_deg(RigidTransform(motion_a), RigidTransform(np.eye(4)))
            if angle < min_rotation_deg:
                continue
            pairs.append((motion_a, motion_b))
    return pairs


def _solve_park(pairs: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    accumulator = np.zeros((3, 3), dtype=np.float64)
    for motion_a, motion_b in pairs:
        alpha = _rotation_vector(motion_a[:3, :3])
        beta = _rotation_vector(motion_b[:3, :3])
        accumulator += np.outer(alpha, beta)
    left, _, right = np.linalg.svd(accumulator)
    rotation = left @ right
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1.0
        rotation = left @ right
    return _orthonormal(rotation), _translation(pairs, rotation)


def _solve_andreff(pairs: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    blocks = []
    for motion_a, motion_b in pairs:
        rotation_a = motion_a[:3, :3]
        rotation_b = motion_b[:3, :3]
        blocks.append(np.kron(np.eye(3), rotation_a) - np.kron(rotation_b.T, np.eye(3)))
    _, _, vh = np.linalg.svd(np.vstack(blocks))
    rotation = vh[-1].reshape((3, 3), order="F")
    if np.linalg.det(rotation) < 0:
        rotation *= -1.0
    rotation = _orthonormal(rotation)
    return rotation, _translation(pairs, rotation)


def _translation(pairs: list[tuple[np.ndarray, np.ndarray]], rotation: np.ndarray) -> np.ndarray:
    rows = []
    values = []
    for motion_a, motion_b in pairs:
        rows.append(motion_a[:3, :3] - np.eye(3))
        values.append(rotation @ motion_b[:3, 3] - motion_a[:3, 3])
    solution, _, _, _ = np.linalg.lstsq(np.vstack(rows), np.concatenate(values), rcond=None)
    return solution


def _orthonormal(rotation: np.ndarray) -> np.ndarray:
    left, _, right = np.linalg.svd(rotation)
    result = left @ right
    if np.linalg.det(result) < 0:
        left[:, -1] *= -1.0
        result = left @ right
    return result


def _rotation_vector(rotation: np.ndarray) -> np.ndarray:
    vector, _ = cv2.Rodrigues(np.asarray(rotation, dtype=np.float64))
    return vector.reshape(3)
