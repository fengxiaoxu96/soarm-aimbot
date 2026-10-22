"""Rigid transforms used to express overhead-camera points in the table frame."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml


class RigidTransform:
    """A 4×4 transform that maps points from the source frame into the destination frame."""

    def __init__(self, matrix: np.ndarray) -> None:
        array = np.asarray(matrix, dtype=np.float64)
        if array.shape != (4, 4):
            raise ValueError(f"A rigid transform must be 4x4, got {array.shape}")
        self.matrix = array

    @classmethod
    def from_rotation_translation(cls, rotation: np.ndarray, translation: np.ndarray) -> "RigidTransform":
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = np.asarray(rotation, dtype=np.float64)
        matrix[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
        return cls(matrix)

    def inverse(self) -> "RigidTransform":
        rotation = self.matrix[:3, :3]
        translation = self.matrix[:3, 3]
        return RigidTransform.from_rotation_translation(rotation.T, -(rotation.T @ translation))

    def apply(self, points: Any) -> np.ndarray:
        """Transform points of shape ``(3,)`` or ``(..., 3)``."""

        array = np.asarray(points, dtype=np.float64)
        flat = array.reshape(-1, 3)
        rotated = flat @ self.matrix[:3, :3].T + self.matrix[:3, 3]
        return rotated.reshape(array.shape)

    @property
    def rotation(self) -> np.ndarray:
        return self.matrix[:3, :3].copy()

    @property
    def translation(self) -> np.ndarray:
        return self.matrix[:3, 3].copy()


def average_transforms(transforms: list[RigidTransform]) -> RigidTransform:
    """Median translation and mean rotation. Quaternions are kept in one hemisphere."""

    if not transforms:
        raise ValueError("No transforms to average")
    translation = np.median(np.stack([item.translation for item in transforms]), axis=0)
    quaternions = [_rotation_to_quaternion(item.rotation) for item in transforms]
    reference = quaternions[0]
    aligned = [item if np.dot(item, reference) >= 0 else -item for item in quaternions]
    mean = np.mean(np.stack(aligned), axis=0)
    mean = mean / np.linalg.norm(mean)
    return RigidTransform.from_rotation_translation(_quaternion_to_rotation(mean), translation)


def rotation_angle_deg(first: RigidTransform, second: RigidTransform) -> float:
    relative = first.rotation.T @ second.rotation
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def translation_distance_m(first: RigidTransform, second: RigidTransform) -> float:
    return float(np.linalg.norm(first.translation - second.translation))


def save_transform(path: Path | str, transform: RigidTransform, metadata: dict[str, Any]) -> None:
    payload = {key: value for key, value in metadata.items() if key not in {"T_world_overhead", "translation_m"}}
    payload["T_world_overhead"] = transform.matrix.tolist()
    payload["translation_m"] = transform.translation.tolist()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def load_transform(path: Path | str) -> tuple[RigidTransform, dict[str, Any]]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "T_world_overhead" not in payload:
        raise ValueError(f"{path} is not a table extrinsics file")
    transform = RigidTransform(np.array(payload["T_world_overhead"], dtype=np.float64))
    metadata = {key: value for key, value in payload.items() if key != "T_world_overhead"}
    return transform, metadata


def _rotation_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    """Return ``(x, y, z, w)``."""

    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quaternion = np.array(
            [
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
                0.25 * scale,
            ]
        )
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        quaternion = np.array(
            [
                0.25 * scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
            ]
        )
    elif matrix[1, 1] > matrix[2, 2]:
        scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        quaternion = np.array(
            [
                (matrix[0, 1] + matrix[1, 0]) / scale,
                0.25 * scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
            ]
        )
    else:
        scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        quaternion = np.array(
            [
                (matrix[0, 2] + matrix[2, 0]) / scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
                0.25 * scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ]
        )
    return quaternion / np.linalg.norm(quaternion)


def _quaternion_to_rotation(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = (float(value) for value in quaternion)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
