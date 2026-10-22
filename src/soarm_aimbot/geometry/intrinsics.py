"""Pinhole intrinsics and the pixel-plus-depth backprojection from step 1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


@dataclass(frozen=True)
class PinholeIntrinsics:
    """Color-camera pinhole model. Depth passed to :meth:`backproject` is meters."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)
    depth_scale_m: float = 0.001

    def __post_init__(self) -> None:
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError(f"Focal lengths must be positive, got fx={self.fx}, fy={self.fy}")
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Invalid resolution: {self.width}x{self.height}")

    @property
    def camera_matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @property
    def distortion_coeffs(self) -> np.ndarray:
        if not self.distortion:
            return np.zeros((5, 1), dtype=np.float64)
        return np.array(self.distortion, dtype=np.float64).reshape(-1, 1)

    def backproject(self, u: Any, v: Any, z_m: Any) -> np.ndarray:
        """Map pixels and depth in meters to points in the camera frame.

        The camera frame follows OpenCV: X right, Y down, Z forward.
        Non-zero distortion is removed before the pinhole formula

            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy

        Scalar inputs return a point of shape ``(3,)``. Array inputs return
        an array of points with the same broadcast shape plus a last axis of 3.
        """

        u_arr = np.asarray(u, dtype=np.float64)
        v_arr = np.asarray(v, dtype=np.float64)
        z_arr = np.asarray(z_m, dtype=np.float64)
        uu, vv = self._undistort_pixels(u_arr, v_arr)
        x = (uu - self.cx) * z_arr / self.fx
        y = (vv - self.cy) * z_arr / self.fy
        points = np.stack([x, y, z_arr], axis=-1)
        if np.ndim(u) == 0 and np.ndim(v) == 0 and np.ndim(z_m) == 0:
            return points.reshape(3)
        return points

    def backproject_depth(self, depth_m: np.ndarray) -> np.ndarray:
        """Backproject an entire depth map. Invalid (non-positive) pixels stay zero."""

        if depth_m.shape != (self.height, self.width):
            raise ValueError(
                f"Depth map shape {depth_m.shape} does not match the intrinsics {self.width}x{self.height}"
            )
        rows, cols = np.indices(depth_m.shape)
        points = self.backproject(cols, rows, depth_m)
        points[depth_m <= 0] = 0.0
        return points

    def _undistort_pixels(
        self, u_arr: np.ndarray, v_arr: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if np.max(np.abs(self.distortion_coeffs)) < 1e-12:
            return u_arr, v_arr
        original_shape = np.broadcast(u_arr, v_arr).shape
        uu = np.broadcast_to(u_arr, original_shape).reshape(-1)
        vv = np.broadcast_to(v_arr, original_shape).reshape(-1)
        pixels = np.stack([uu, vv], axis=-1).reshape(-1, 1, 2)
        undistorted = cv2.undistortPoints(
            pixels,
            self.camera_matrix,
            self.distortion_coeffs,
            P=self.camera_matrix,
        )
        return (
            undistorted[:, 0, 0].reshape(original_shape),
            undistorted[:, 0, 1].reshape(original_shape),
        )


def save_intrinsics(path: Path | str, model: PinholeIntrinsics, metadata: dict[str, Any]) -> None:
    """Write the step-1 artifact ``calib/overhead_intrinsics.yaml``."""

    payload: dict[str, Any] = {
        "camera": "overhead",
        "frame": "overhead",
        "width": model.width,
        "height": model.height,
        "fx": float(model.fx),
        "fy": float(model.fy),
        "cx": float(model.cx),
        "cy": float(model.cy),
        "distortion": [float(v) for v in model.distortion],
        "depth_scale_m": float(model.depth_scale_m),
        "depth_unit": "meter",
    }
    payload.update(metadata)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def load_intrinsics(path: Path | str) -> tuple[PinholeIntrinsics, dict[str, Any]]:
    """Load a pinhole model and any extra fields saved next to it."""

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not an intrinsics file")
    model = PinholeIntrinsics(
        width=int(payload["width"]),
        height=int(payload["height"]),
        fx=float(payload["fx"]),
        fy=float(payload["fy"]),
        cx=float(payload["cx"]),
        cy=float(payload["cy"]),
        distortion=tuple(float(v) for v in payload.get("distortion", (0, 0, 0, 0, 0))),
        depth_scale_m=float(payload.get("depth_scale_m", 0.001)),
    )
    metadata = {
        key: value
        for key, value in payload.items()
        if key
        not in {
            "width",
            "height",
            "fx",
            "fy",
            "cx",
            "cy",
            "distortion",
            "depth_scale_m",
        }
    }
    return model, metadata
