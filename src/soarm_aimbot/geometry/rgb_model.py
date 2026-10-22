"""RGB pinhole or fisheye model: undistort an image, or turn a pixel into a ray.

This is the wrist-camera result of step 2. It does not know which device
produced the pictures. Depth is not part of the model; a ray is a direction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

BROWN = "brown"
FISHEYE = "fisheye"


@dataclass(frozen=True)
class RgbCameraModel:
    """Intrinsics of one RGB camera at one fixed resolution.

    ``pixel_to_ray`` consumes raw pixels and removes distortion internally.
    ``undistort_image`` returns a new image whose pinhole parameters are
    ``undistorted_*``. A later detector must use one of those two paths, and
    must not remove distortion a second time.
    """

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: tuple[float, ...]
    model: str = BROWN
    undistorted_fx: float | None = None
    undistorted_fy: float | None = None
    undistorted_cx: float | None = None
    undistorted_cy: float | None = None

    def __post_init__(self) -> None:
        if self.model not in (BROWN, FISHEYE):
            raise ValueError(f"Unknown camera model {self.model!r}; use brown or fisheye")
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError(f"Focal lengths must be positive, got fx={self.fx}, fy={self.fy}")
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Invalid resolution: {self.width}x{self.height}")
        if self.model == FISHEYE and len(self.distortion) != 4:
            raise ValueError("The fisheye model uses 4 distortion coefficients")

    @property
    def camera_matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @property
    def distortion_coeffs(self) -> np.ndarray:
        coeffs = np.array(self.distortion, dtype=np.float64).reshape(-1, 1)
        if self.model == BROWN and coeffs.size == 0:
            return np.zeros((5, 1), dtype=np.float64)
        return coeffs

    def pixel_to_ray(self, u: Any, v: Any) -> np.ndarray:
        """Unit direction in the camera frame for each raw pixel.

        The frame follows OpenCV: X right, Y down, Z forward. Scalar inputs
        return a vector of shape ``(3,)``.
        """

        u_arr = np.asarray(u, dtype=np.float64)
        v_arr = np.asarray(v, dtype=np.float64)
        original_shape = np.broadcast(u_arr, v_arr).shape
        pixels = np.stack(
            [
                np.broadcast_to(u_arr, original_shape).reshape(-1),
                np.broadcast_to(v_arr, original_shape).reshape(-1),
            ],
            axis=-1,
        ).reshape(-1, 1, 2)
        normalized = self._to_normalized(pixels)
        directions = np.concatenate(
            [normalized, np.ones((normalized.shape[0], 1), dtype=np.float64)],
            axis=1,
        )
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        rays = directions.reshape(original_shape + (3,))
        if np.ndim(u) == 0 and np.ndim(v) == 0:
            return rays.reshape(3)
        return rays

    def undistort_image(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the undistorted image and the 3x3 pinhole matrix that matches it."""

        if image.shape[1] != self.width or image.shape[0] != self.height:
            raise ValueError(
                f"Image size {image.shape[1]}x{image.shape[0]} does not match the calibrated resolution {self.width}x{self.height}"
            )
        map1, map2, new_k = self.undistort_maps()
        return cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR), new_k

    def undistort_maps(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        size = (self.width, self.height)
        if self.model == BROWN:
            new_k, _roi = cv2.getOptimalNewCameraMatrix(
                self.camera_matrix,
                self.distortion_coeffs,
                size,
                0,
                size,
            )
            map1, map2 = cv2.initUndistortRectifyMap(
                self.camera_matrix,
                self.distortion_coeffs,
                None,
                new_k,
                size,
                cv2.CV_16SC2,
            )
            return map1, map2, new_k
        new_k = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            self.camera_matrix,
            self.distortion_coeffs,
            size,
            np.eye(3),
            balance=0.0,
        )
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            self.camera_matrix,
            self.distortion_coeffs,
            np.eye(3),
            new_k,
            size,
            cv2.CV_16SC2,
        )
        return map1, map2, new_k

    def project(self, object_points: np.ndarray, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        """Project 3D points to raw pixels. Returns shape ``(N, 2)``."""

        if self.model == BROWN:
            projected, _ = cv2.projectPoints(
                np.asarray(object_points, dtype=np.float64),
                rvec,
                tvec,
                self.camera_matrix,
                self.distortion_coeffs,
            )
            return projected.reshape(-1, 2)
        projected, _ = cv2.fisheye.projectPoints(
            np.asarray(object_points, dtype=np.float64).reshape(1, -1, 3),
            np.asarray(rvec, dtype=np.float64).reshape(3, 1),
            np.asarray(tvec, dtype=np.float64).reshape(3, 1),
            self.camera_matrix,
            self.distortion_coeffs,
        )
        return projected.reshape(-1, 2)

    def with_undistorted_pinhole(self) -> "RgbCameraModel":
        _map1, _map2, new_k = self.undistort_maps()
        return RgbCameraModel(
            width=self.width,
            height=self.height,
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
            distortion=self.distortion,
            model=self.model,
            undistorted_fx=float(new_k[0, 0]),
            undistorted_fy=float(new_k[1, 1]),
            undistorted_cx=float(new_k[0, 2]),
            undistorted_cy=float(new_k[1, 2]),
        )

    def _to_normalized(self, pixels: np.ndarray) -> np.ndarray:
        if self.model == BROWN:
            normalized = cv2.undistortPoints(pixels, self.camera_matrix, self.distortion_coeffs)
        else:
            normalized = cv2.fisheye.undistortPoints(
                pixels,
                self.camera_matrix,
                self.distortion_coeffs,
            )
        return normalized.reshape(-1, 2)


def save_rgb_intrinsics(path: Path | str, model: RgbCameraModel, metadata: dict[str, Any]) -> None:
    payload: dict[str, Any] = {
        "camera": "wrist",
        "frame": "wrist",
        "model": model.model,
        "width": model.width,
        "height": model.height,
        "fx": float(model.fx),
        "fy": float(model.fy),
        "cx": float(model.cx),
        "cy": float(model.cy),
        "distortion": [float(value) for value in model.distortion],
        "undistorted_fx": None if model.undistorted_fx is None else float(model.undistorted_fx),
        "undistorted_fy": None if model.undistorted_fy is None else float(model.undistorted_fy),
        "undistorted_cx": None if model.undistorted_cx is None else float(model.undistorted_cx),
        "undistorted_cy": None if model.undistorted_cy is None else float(model.undistorted_cy),
    }
    payload.update(metadata)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def load_rgb_intrinsics(path: Path | str) -> tuple[RgbCameraModel, dict[str, Any]]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a wrist intrinsics file")
    model = RgbCameraModel(
        width=int(payload["width"]),
        height=int(payload["height"]),
        fx=float(payload["fx"]),
        fy=float(payload["fy"]),
        cx=float(payload["cx"]),
        cy=float(payload["cy"]),
        distortion=tuple(float(value) for value in payload.get("distortion", ())),
        model=str(payload.get("model", BROWN)),
        undistorted_fx=_optional_float(payload.get("undistorted_fx")),
        undistorted_fy=_optional_float(payload.get("undistorted_fy")),
        undistorted_cx=_optional_float(payload.get("undistorted_cx")),
        undistorted_cy=_optional_float(payload.get("undistorted_cy")),
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
            "model",
            "undistorted_fx",
            "undistorted_fy",
            "undistorted_cx",
            "undistorted_cy",
        }
    }
    return model, metadata


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
