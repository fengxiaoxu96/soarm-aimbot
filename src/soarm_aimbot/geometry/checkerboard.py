"""Checkerboard calibration for an RGB camera, with a held-out error check."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from soarm_aimbot.geometry.rgb_model import BROWN, FISHEYE, RgbCameraModel


@dataclass(frozen=True)
class ViewError:
    index: int
    role: str
    mean_error_px: float


@dataclass(frozen=True)
class CheckerboardCalibration:
    model: RgbCameraModel
    reprojection_rms_px: float
    calibration_mean_error_px: float
    holdout_mean_error_px: float | None
    view_errors: tuple[ViewError, ...]
    fisheye_check_cond: bool | None = None


def checkerboard_object(cols: int, rows: int, square_m: float) -> np.ndarray:
    if cols < 3 or rows < 3:
        raise ValueError("At least 3 inner corners per side are required")
    if square_m <= 0:
        raise ValueError("The square size must be positive")
    points = np.zeros((rows * cols, 3), np.float32)
    points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    points *= square_m
    return points


def find_corners(gray: np.ndarray, cols: int, rows: int) -> np.ndarray | None:
    """Return refined corners of shape ``(N, 1, 2)``, or None when the board is incomplete."""

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, (cols, rows), flags)
    if not found or corners is None:
        return None
    refined = cv2.cornerSubPix(
        gray,
        corners,
        (11, 11),
        (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
    )
    return refined


def split_holdout(count: int, holdout: int) -> tuple[list[int], list[int]]:
    """Pick held-out indexes evenly through the capture order.

    The calibration set keeps at least 8 views. Held-out views are spread
    across the sequence, so later poses are not the only ones left out.
    """

    if holdout < 0:
        raise ValueError("--holdout must not be negative")
    if holdout == 0:
        return list(range(count)), []
    if count - holdout < 8:
        raise ValueError(
            f"After holding out {holdout} images, only {count - holdout} remain for the solve; at least 8 are needed. "
            "Capture more images or lower --holdout."
        )
    chosen: list[int] = []
    used: set[int] = set()
    for i in range(holdout):
        pos = min(count - 1, int((i + 0.5) * count / holdout))
        while pos in used:
            pos += 1
            if pos >= count:
                raise ValueError("Cannot hold out these images evenly")
        used.add(pos)
        chosen.append(pos)
    calibration = [index for index in range(count) if index not in used]
    return calibration, chosen


def calibrate_checkerboard(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    image_size: tuple[int, int],
    model_name: str,
    holdout: int,
) -> CheckerboardCalibration:
    if len(object_points) != len(image_points):
        raise ValueError("Object points and image points have different image counts")
    if model_name not in (BROWN, FISHEYE):
        raise ValueError(f"Unknown camera model {model_name!r}")
    calibration_ids, holdout_ids = split_holdout(len(object_points), holdout)
    fit_obj = [object_points[index] for index in calibration_ids]
    fit_img = [image_points[index] for index in calibration_ids]
    if model_name == BROWN:
        model, rms, rvecs, tvecs, check_cond = _calibrate_brown(fit_obj, fit_img, image_size)
    else:
        model, rms, rvecs, tvecs, check_cond = _calibrate_fisheye(fit_obj, fit_img, image_size)
    model = model.with_undistorted_pinhole()

    errors: list[ViewError] = []
    for slot, index in enumerate(calibration_ids):
        mean_error = _mean_error(model, object_points[index], image_points[index], rvecs[slot], tvecs[slot])
        errors.append(ViewError(index=index, role="calibration", mean_error_px=mean_error))
    for index in holdout_ids:
        rvec, tvec = _solve_pose(model, object_points[index], image_points[index])
        mean_error = _mean_error(model, object_points[index], image_points[index], rvec, tvec)
        errors.append(ViewError(index=index, role="holdout", mean_error_px=mean_error))
    errors.sort(key=lambda item: item.index)

    calibration_mean = _mean_of([item.mean_error_px for item in errors if item.role == "calibration"])
    holdout_values = [item.mean_error_px for item in errors if item.role == "holdout"]
    holdout_mean = _mean_of(holdout_values) if holdout_values else None
    return CheckerboardCalibration(
        model=model,
        reprojection_rms_px=float(rms),
        calibration_mean_error_px=calibration_mean,
        holdout_mean_error_px=holdout_mean,
        view_errors=tuple(errors),
        fisheye_check_cond=check_cond,
    )


def coverage_warning(image_points: list[np.ndarray], width: int, height: int) -> str | None:
    """Return a hint when every board stayed in the middle of the image."""

    centers = [points.reshape(-1, 2).mean(axis=0) for points in image_points]
    xs = [float(center[0]) for center in centers]
    ys = [float(center[1]) for center in centers]
    notes: list[str] = []
    if min(xs) > width * 0.25 and max(xs) < width * 0.75:
        notes.append("All checkerboard centers are in the horizontal middle of the image. Add a few poses near the left and right edges.")
    if min(ys) > height * 0.25 and max(ys) < height * 0.75:
        notes.append("All checkerboard centers are in the vertical middle of the image. Add a few poses near the top and bottom edges.")
    if not notes:
        return None
    return "".join(notes)


def _calibrate_brown(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    image_size: tuple[int, int],
) -> tuple[RgbCameraModel, float, list[np.ndarray], list[np.ndarray], None]:
    rms, matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
        [_as_object(points) for points in object_points],
        [_as_image(points) for points in image_points],
        image_size,
        None,
        None,
    )
    model = RgbCameraModel(
        width=image_size[0],
        height=image_size[1],
        fx=float(matrix[0, 0]),
        fy=float(matrix[1, 1]),
        cx=float(matrix[0, 2]),
        cy=float(matrix[1, 2]),
        distortion=tuple(float(value) for value in np.asarray(distortion).reshape(-1)),
        model=BROWN,
    )
    return model, float(rms), list(rvecs), list(tvecs), None


def _calibrate_fisheye(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    image_size: tuple[int, int],
) -> tuple[RgbCameraModel, float, list[np.ndarray], list[np.ndarray], bool]:
    obj = [_as_object(points).reshape(1, -1, 3) for points in object_points]
    img = [_as_image(points).reshape(1, -1, 2) for points in image_points]
    matrix = np.zeros((3, 3), dtype=np.float64)
    distortion = np.zeros((4, 1), dtype=np.float64)
    # OpenCV 5 keeps these flags on cv2, not on cv2.fisheye.
    flags = cv2.CALIB_RECOMPUTE_EXTRINSIC + cv2.CALIB_FIX_SKEW
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    check_cond = True
    try:
        rms, matrix, distortion, rvecs, tvecs = cv2.fisheye.calibrate(
            obj,
            img,
            image_size,
            matrix,
            distortion,
            None,
            None,
            flags + cv2.CALIB_CHECK_COND,
            criteria,
        )
    except cv2.error:
        check_cond = False
        matrix = np.zeros((3, 3), dtype=np.float64)
        distortion = np.zeros((4, 1), dtype=np.float64)
        rms, matrix, distortion, rvecs, tvecs = cv2.fisheye.calibrate(
            obj,
            img,
            image_size,
            matrix,
            distortion,
            None,
            None,
            flags,
            criteria,
        )
    model = RgbCameraModel(
        width=image_size[0],
        height=image_size[1],
        fx=float(matrix[0, 0]),
        fy=float(matrix[1, 1]),
        cx=float(matrix[0, 2]),
        cy=float(matrix[1, 2]),
        distortion=tuple(float(value) for value in np.asarray(distortion).reshape(-1)),
        model=FISHEYE,
    )
    return model, float(rms), list(rvecs), list(tvecs), check_cond


def _solve_pose(model: RgbCameraModel, object_points: np.ndarray, image_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    obj = _as_object(object_points)
    img = _as_image(image_points)
    if model.model == BROWN:
        ok, rvec, tvec = cv2.solvePnP(
            obj,
            img,
            model.camera_matrix,
            model.distortion_coeffs,
        )
    else:
        ok, rvec, tvec = cv2.fisheye.solvePnP(
            obj.reshape(1, -1, 3),
            img.reshape(1, -1, 2),
            model.camera_matrix,
            model.distortion_coeffs,
        )
    if not ok:
        raise RuntimeError("Could not solve the checkerboard pose for one of the held-out images")
    return rvec, tvec


def _mean_error(
    model: RgbCameraModel,
    object_points: np.ndarray,
    image_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> float:
    projected = model.project(object_points, rvec, tvec)
    observed = image_points.reshape(-1, 2)
    return float(np.mean(np.linalg.norm(projected - observed, axis=1)))


def _as_object(points: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float32).reshape(-1, 3)


def _as_image(points: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)


def _mean_of(values: list[float]) -> float:
    if not values:
        raise ValueError("No reprojection errors to summarize")
    return float(np.mean(values))
