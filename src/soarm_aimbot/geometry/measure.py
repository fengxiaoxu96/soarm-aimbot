"""Compare a physical block with backprojected depth. Used to check intrinsics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from soarm_aimbot.geometry.intrinsics import PinholeIntrinsics

Roi = tuple[int, int, int, int]


@dataclass(frozen=True)
class BlockMeasurement:
    """Sizes recovered from an overhead depth image.

    ``width_m`` is the 3D distance between the left and right edges of the
    top-face ROI, backprojected at the top face's median depth. It matches a
    caliper measurement when that face is about perpendicular to the optical
    axis, which is the overhead-camera case.

    ``height_m`` is the table depth minus the top-face depth along the optical
    axis. With the camera looking straight down, that difference is the block
    thickness.
    """

    top_depth_m: float
    table_depth_m: float
    width_m: float
    length_m: float
    height_m: float
    top_roi: Roi
    table_roi: Roi


def roi_median_depth(depth_m: np.ndarray, roi: Roi, min_valid: int = 30) -> float:
    x, y, w, h = roi
    _validate_roi(depth_m.shape, roi)
    patch = depth_m[y : y + h, x : x + w]
    valid = patch[patch > 0]
    if valid.size < min_valid:
        raise ValueError(f"ROI {roi} has only {valid.size} valid depth points; at least {min_valid} are needed")
    return float(np.median(valid))


def measure_block(
    depth_m: np.ndarray,
    intrinsics: PinholeIntrinsics,
    top_roi: Roi,
    table_roi: Roi,
) -> BlockMeasurement:
    z_top = roi_median_depth(depth_m, top_roi)
    z_table = roi_median_depth(depth_m, table_roi)
    width_m = _span(intrinsics, top_roi, z_top, axis="u")
    length_m = _span(intrinsics, top_roi, z_top, axis="v")
    return BlockMeasurement(
        top_depth_m=z_top,
        table_depth_m=z_table,
        width_m=width_m,
        length_m=length_m,
        height_m=z_table - z_top,
        top_roi=top_roi,
        table_roi=table_roi,
    )


def _span(intrinsics: PinholeIntrinsics, roi: Roi, z_m: float, axis: str) -> float:
    x, y, w, h = roi
    if axis == "u":
        v = y + (h - 1) / 2.0
        p0 = intrinsics.backproject(x, v, z_m)
        p1 = intrinsics.backproject(x + w - 1, v, z_m)
    elif axis == "v":
        u = x + (w - 1) / 2.0
        p0 = intrinsics.backproject(u, y, z_m)
        p1 = intrinsics.backproject(u, y + h - 1, z_m)
    else:
        raise ValueError(axis)
    return float(np.linalg.norm(p0 - p1))


def _validate_roi(shape: tuple[int, ...], roi: Roi) -> None:
    height, width = shape[:2]
    x, y, w, h = roi
    if w < 2 or h < 2:
        raise ValueError(f"The ROI must be at least 2x2 pixels, got {roi}")
    if x < 0 or y < 0 or x + w > width or y + h > height:
        raise ValueError(f"ROI {roi} is outside the {width}x{height} image")
