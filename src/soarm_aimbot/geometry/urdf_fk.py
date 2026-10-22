"""Forward kinematics from a URDF joint chain. No meshes and no robot SDK.

A revolute joint moves the child link by

    T_parent_child = T_origin * Rot(axis, q)

``T_origin`` is the URDF joint origin. ``q = 0`` is that origin, which for
``so101_new_calib.urdf`` is the calibrated zero, not a mechanical end stop.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from soarm_aimbot.geometry.transforms import RigidTransform

ARM_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)


@dataclass(frozen=True)
class UrdfJoint:
    name: str
    joint_type: str
    parent: str
    child: str
    origin: RigidTransform
    axis: np.ndarray
    lower: float | None
    upper: float | None
    velocity: float | None = None


@dataclass(frozen=True)
class UrdfModel:
    path: str
    joints: dict[str, UrdfJoint]

    def chain(self, root_link: str, tip_link: str) -> list[UrdfJoint]:
        by_parent: dict[str, list[UrdfJoint]] = {}
        for joint in self.joints.values():
            by_parent.setdefault(joint.parent, []).append(joint)
        stack: list[tuple[str, list[UrdfJoint]]] = [(root_link, [])]
        while stack:
            link, path = stack.pop()
            if link == tip_link:
                return path
            for joint in by_parent.get(link, []):
                stack.append((joint.child, path + [joint]))
        known = ", ".join(sorted({joint.child for joint in self.joints.values()} | {root_link}))
        raise ValueError(f"The URDF has no joint chain from {root_link} to {tip_link}. Available links: {known}")


def load_urdf(path: Path | str) -> UrdfModel:
    tree = ET.parse(Path(path))
    robot = tree.getroot()
    joints: dict[str, UrdfJoint] = {}
    for element in robot.findall("joint"):
        name = element.attrib["name"]
        origin = element.find("origin")
        axis = element.find("axis")
        limit = element.find("limit")
        xyz = _vector(origin.attrib.get("xyz", "0 0 0") if origin is not None else "0 0 0")
        rpy = _vector(origin.attrib.get("rpy", "0 0 0") if origin is not None else "0 0 0")
        joints[name] = UrdfJoint(
            name=name,
            joint_type=element.attrib["type"],
            parent=element.find("parent").attrib["link"],
            child=element.find("child").attrib["link"],
            origin=RigidTransform.from_rotation_translation(rpy_matrix(rpy), xyz),
            axis=_vector(axis.attrib.get("xyz", "0 0 1") if axis is not None else "0 0 1"),
            lower=float(limit.attrib["lower"]) if limit is not None and "lower" in limit.attrib else None,
            upper=float(limit.attrib["upper"]) if limit is not None and "upper" in limit.attrib else None,
            velocity=float(limit.attrib["velocity"]) if limit is not None and "velocity" in limit.attrib else None,
        )
    if not joints:
        raise ValueError(f"{path} has no joints")
    return UrdfModel(path=str(path), joints=joints)


def forward_kinematics(
    model: UrdfModel,
    joint_positions: dict[str, float],
    *,
    root_link: str = "base_link",
    tip_link: str = "gripper_link",
) -> RigidTransform:
    """Return T_root_tip, the transform that maps tip-link points into the root link."""

    pose = RigidTransform(np.eye(4))
    for joint in model.chain(root_link, tip_link):
        motion = _joint_motion(joint, joint_positions)
        pose = RigidTransform(pose.matrix @ joint.origin.matrix @ motion)
    return pose


def rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    """URDF fixed-axis roll, pitch, yaw: R = Rz(yaw) Ry(pitch) Rx(roll)."""

    roll, pitch, yaw = (float(value) for value in np.asarray(rpy, dtype=np.float64).reshape(3))
    cx, sx = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(yaw), np.sin(yaw)
    rotation_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    rotation_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rotation_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return rotation_z @ rotation_y @ rotation_x


def reading_to_urdf_radians(stop_min_deg: float, stop_max_deg: float, reading_deg: float) -> float:
    """Map a joint reading to radians whose zero is the midpoint of the two stops."""

    midpoint = 0.5 * (stop_min_deg + stop_max_deg)
    return float(np.deg2rad(reading_deg - midpoint))


def _joint_motion(joint: UrdfJoint, positions: dict[str, float]) -> np.ndarray:
    if joint.joint_type == "fixed":
        return np.eye(4)
    if joint.joint_type != "revolute":
        raise ValueError(f"Joint {joint.name} has type {joint.joint_type}, which is not supported in this chain")
    if joint.name not in positions:
        raise ValueError(f"Missing angle for joint {joint.name}")
    angle = float(positions[joint.name])
    if joint.lower is not None and angle < joint.lower - 0.1:
        raise ValueError(f"{joint.name}={angle:.4f} rad is below the limit {joint.lower:.4f}")
    if joint.upper is not None and angle > joint.upper + 0.1:
        raise ValueError(f"{joint.name}={angle:.4f} rad is above the limit {joint.upper:.4f}")
    axis = np.asarray(joint.axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        raise ValueError(f"Revolute joint {joint.name} has a zero axis")
    rotation, _ = cv2.Rodrigues((axis / norm) * angle)
    motion = np.eye(4)
    motion[:3, :3] = rotation
    return motion


def _vector(text: str) -> np.ndarray:
    values = [float(item) for item in text.split()]
    if len(values) != 3:
        raise ValueError(f"A URDF vector must have 3 numbers, got {text!r}")
    return np.array(values, dtype=np.float64)
