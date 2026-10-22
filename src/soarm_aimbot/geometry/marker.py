"""Planar AprilTag pose. The tag frame is the table frame when the tag lies face up."""

from __future__ import annotations

import cv2
import numpy as np

from soarm_aimbot.geometry.intrinsics import PinholeIntrinsics
from soarm_aimbot.geometry.rgb_model import FISHEYE, RgbCameraModel
from soarm_aimbot.geometry.transforms import RigidTransform

DICTIONARIES = {
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
}


def marker_object_points(size_m: float) -> np.ndarray:
    """Corners in the tag frame: origin at the center, Z out of the tag toward the camera.

    Order matches OpenCV detection: top-left, top-right, bottom-right, bottom-left of the
    printed bitmap. Tag +X points to the right of that bitmap, tag +Y toward its top.
    """

    if size_m <= 0:
        raise ValueError("The marker size must be positive")
    half = size_m / 2.0
    return np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float32,
    )


def detect_marker(image: np.ndarray, dictionary_name: str, marker_id: int) -> np.ndarray | None:
    """Return corners of shape ``(4, 2)`` for ``marker_id``, or None."""

    dictionary = _dictionary(dictionary_name)
    parameters = cv2.aruco.DetectorParameters()
    if "APRIL" in dictionary_name.upper():
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    else:
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _rejected = detector.detectMarkers(gray)
    if ids is None:
        return None
    for found_corners, found_id in zip(corners, ids.ravel()):
        if int(found_id) == marker_id:
            return found_corners.reshape(4, 2).astype(np.float32)
    return None


def pose_from_corners(
    corners: np.ndarray,
    size_m: float,
    intrinsics: PinholeIntrinsics,
) -> RigidTransform:
    """Return the transform that maps tag-frame points into the camera frame."""

    ok, rvec, tvec = cv2.solvePnP(
        marker_object_points(size_m),
        np.asarray(corners, dtype=np.float32).reshape(4, 2),
        intrinsics.camera_matrix,
        intrinsics.distortion_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not ok:
        raise RuntimeError("Could not solve a pose from the marker corners")
    rotation, _jacobian = cv2.Rodrigues(rvec)
    translation = np.asarray(tvec, dtype=np.float64).reshape(3)
    if translation[2] <= 0:
        raise RuntimeError("The marker pose is behind the camera")
    return RigidTransform.from_rotation_translation(rotation, translation)


def pose_from_corners_rgb(
    corners: np.ndarray,
    size_m: float,
    model: RgbCameraModel,
) -> RigidTransform:
    """Return T_camera_tag from raw pixels of a wrist RGB frame.

    Corners stay on the original image. A brown model is passed to solvePnP
    with its distortion. A fisheye model undistorts the four corners into the
    pinhole of the same K, then solves with zero distortion. The resulting
    pose is still in the camera frame, so ``model.project`` maps it back to
    raw pixels.
    """

    image_points = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    distortion = model.distortion_coeffs
    if model.model == FISHEYE:
        image_points = cv2.fisheye.undistortPoints(
            image_points.reshape(1, -1, 2),
            model.camera_matrix,
            distortion,
            P=model.camera_matrix,
        ).reshape(4, 2)
        distortion = np.zeros((5, 1), dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(
        marker_object_points(size_m),
        image_points,
        model.camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not ok:
        raise RuntimeError("Could not solve a pose from the marker corners")
    rotation, _jacobian = cv2.Rodrigues(rvec)
    translation = np.asarray(tvec, dtype=np.float64).reshape(3)
    if translation[2] <= 0:
        raise RuntimeError("The marker pose is behind the camera")
    return RigidTransform.from_rotation_translation(rotation, translation)


def reprojection_error_rgb(
    corners: np.ndarray,
    size_m: float,
    model: RgbCameraModel,
    camera_from_tag: RigidTransform,
) -> float:
    projected = model.project(
        marker_object_points(size_m),
        cv2.Rodrigues(camera_from_tag.rotation)[0],
        camera_from_tag.translation.reshape(3, 1),
    )
    observed = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    return float(np.mean(np.linalg.norm(projected - observed, axis=1)))


def reprojection_error_px(
    corners: np.ndarray,
    size_m: float,
    intrinsics: PinholeIntrinsics,
    camera_from_tag: RigidTransform,
) -> float:
    projected, _ = cv2.projectPoints(
        marker_object_points(size_m),
        cv2.Rodrigues(camera_from_tag.rotation)[0],
        camera_from_tag.translation.reshape(3, 1),
        intrinsics.camera_matrix,
        intrinsics.distortion_coeffs,
    )
    observed = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    return float(np.mean(np.linalg.norm(projected.reshape(4, 2) - observed, axis=1)))


def render_marker(dictionary_name: str, marker_id: int, side_px: int) -> np.ndarray:
    if side_px < 40:
        raise ValueError("The marker image must be at least 40 pixels")
    return cv2.aruco.generateImageMarker(_dictionary(dictionary_name), marker_id, side_px)


def _dictionary(name: str) -> cv2.aruco.Dictionary:
    try:
        code = DICTIONARIES[name]
    except KeyError as exc:
        known = ", ".join(DICTIONARIES)
        raise ValueError(f"Unknown marker dictionary {name!r}. Available: {known}") from exc
    return cv2.aruco.getPredefinedDictionary(code)
