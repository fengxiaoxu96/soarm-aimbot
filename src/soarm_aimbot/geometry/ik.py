"""Damped least-squares inverse kinematics on the project's URDF chain.

Five arm joints determine the gripper origin and the approach-axis direction.
The twist about that axis is the yaw chosen by the caller. Pinocchio and Placo are
not used: the solution is checked with the same forward kinematics as the
hover pose, so the two cannot disagree.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from soarm_aimbot.geometry.hover import NAMED_AXES
from soarm_aimbot.geometry.transforms import RigidTransform
from soarm_aimbot.geometry.urdf_fk import ARM_JOINTS, UrdfModel, forward_kinematics

_DIRECTION_WEIGHT = 0.08
_TWIST_WEIGHT = 0.03


@dataclass(frozen=True)
class IkSolution:
    converged: bool
    at_limit: bool
    joints_rad: np.ndarray | None
    position_error_m: float
    approach_angle_deg: float
    twist_deg: float
    iterations: int


def solve_ik(
    model: UrdfModel,
    seed_rad: np.ndarray,
    target: RigidTransform,
    approach_axis: str,
    *,
    position_tol_m: float = 0.002,
    approach_tol_deg: float = 1.0,
    twist_tol_deg: float = 2.0,
    max_iterations: int = 80,
) -> IkSolution:
    """Drive gripper origin and approach axis onto ``target``.

    ``joints_rad`` is set only when the iteration reaches the tolerances
    inside the URDF limits. A joint pressed against a stop without meeting
    those tolerances is reported as ``at_limit`` and carries no command.
    """

    if approach_axis not in NAMED_AXES:
        raise ValueError(f"Unknown approach axis {approach_axis}")
    approach = NAMED_AXES[approach_axis]
    reference = _perpendicular(approach)
    lower, upper = _limits(model)
    joints = np.clip(np.asarray(seed_rad, dtype=np.float64).reshape(5), lower, upper)
    damping = 1e-3
    last_position = float("inf")
    last_angle = 180.0
    last_twist = 180.0
    clamped = False
    for iteration in range(1, max_iterations + 1):
        residual, last_position, last_angle, last_twist = _residual(
            model, joints, target, approach, reference
        )
        if last_position <= position_tol_m and last_angle <= approach_tol_deg and abs(last_twist) <= twist_tol_deg:
            return IkSolution(True, _near_limit(joints, lower, upper), joints.copy(), last_position, last_angle, last_twist, iteration)
        jacobian = _jacobian(model, joints, target, approach, reference, lower, upper)
        step = _damped_step(jacobian, residual, damping)
        step = np.clip(step, -0.25, 0.25)
        updated = np.clip(joints + step, lower, upper)
        if np.any(updated != joints + step):
            clamped = True
        updated_residual, _, _, _ = _residual(model, updated, target, approach, reference)
        if float(np.linalg.norm(updated_residual)) > float(np.linalg.norm(residual)):
            damping = min(damping * 10.0, 1.0)
            continue
        damping = max(damping * 0.5, 1e-6)
        joints = updated
    at_limit = clamped or _near_limit(joints, lower, upper)
    return IkSolution(False, at_limit, None, last_position, last_angle, last_twist, max_iterations)


def _residual(model, joints, target, approach, reference):
    pose = _pose(model, joints)
    position = target.translation - pose.translation
    current_axis = pose.rotation @ approach
    desired_axis = target.rotation @ approach
    desired_axis = desired_axis / np.linalg.norm(desired_axis)
    direction = np.cross(current_axis, desired_axis)
    cosine = float(np.clip(np.dot(current_axis, desired_axis), -1.0, 1.0))
    if cosine < 0.0:
        direction = direction + _perpendicular(desired_axis) * (1.0 - cosine)
    twist = _twist(pose.rotation, target.rotation, reference, desired_axis)
    residual = np.concatenate([position, _DIRECTION_WEIGHT * direction, [_TWIST_WEIGHT * twist]])
    angle = float(np.degrees(np.arccos(cosine)))
    return residual, float(np.linalg.norm(position)), angle, float(np.degrees(twist))


def _jacobian(model, joints, target, approach, reference, lower, upper):
    residual, _, _, _ = _residual(model, joints, target, approach, reference)
    jacobian = np.zeros((residual.size, 5), dtype=np.float64)
    step = 1e-5
    for index in range(5):
        bumped = joints.copy()
        if bumped[index] + step <= upper[index]:
            bumped[index] += step
            scale = step
        else:
            bumped[index] -= step
            scale = -step
        bumped = np.clip(bumped, lower, upper)
        other, _, _, _ = _residual(model, bumped, target, approach, reference)
        jacobian[:, index] = (other - residual) / scale
    return jacobian


def _damped_step(jacobian: np.ndarray, residual: np.ndarray, damping: float) -> np.ndarray:
    normal = jacobian.T @ jacobian + damping * np.eye(jacobian.shape[1])
    return -np.linalg.solve(normal, jacobian.T @ residual)


def _twist(rotation, target_rotation, reference, desired_axis) -> float:
    current = rotation @ reference
    desired = target_rotation @ reference
    current = current - desired_axis * np.dot(current, desired_axis)
    desired = desired - desired_axis * np.dot(desired, desired_axis)
    if np.linalg.norm(current) < 1e-8 or np.linalg.norm(desired) < 1e-8:
        return 0.0
    current = current / np.linalg.norm(current)
    desired = desired / np.linalg.norm(desired)
    return float(np.arctan2(np.dot(desired_axis, np.cross(desired, current)), np.dot(desired, current)))


def _perpendicular(vector: np.ndarray) -> np.ndarray:
    helper = np.array([1.0, 0.0, 0.0]) if abs(float(vector[0])) < 0.9 else np.array([0.0, 1.0, 0.0])
    side = np.cross(vector, helper)
    return side / np.linalg.norm(side)


def _pose(model: UrdfModel, joints: np.ndarray) -> RigidTransform:
    return forward_kinematics(model, dict(zip(ARM_JOINTS, joints)))


def _limits(model: UrdfModel) -> tuple[np.ndarray, np.ndarray]:
    lower = []
    upper = []
    for name in ARM_JOINTS:
        joint = model.joints[name]
        if joint.lower is None or joint.upper is None:
            raise ValueError(f"{name} has no URDF limits")
        lower.append(joint.lower)
        upper.append(joint.upper)
    return np.array(lower), np.array(upper)


def _near_limit(joints: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> bool:
    return bool(np.any(joints <= lower + 1e-3) or np.any(joints >= upper - 1e-3))
