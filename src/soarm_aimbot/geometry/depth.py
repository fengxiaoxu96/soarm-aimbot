"""Depth filtering that does not depend on a particular camera SDK."""

from __future__ import annotations

import warnings

import cv2
import numpy as np


def spatial_median(depth_m: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Median-filter a metric depth map. Non-positive pixels stay invalid."""

    if ksize <= 1:
        return np.asarray(depth_m, dtype=np.float32)
    if ksize % 2 == 0 or ksize < 3:
        raise ValueError(f"Median kernel size must be an odd number of at least 3, got {ksize}")
    raw = np.asarray(depth_m, dtype=np.float32)
    filtered = cv2.medianBlur(raw, ksize)
    valid = raw > 0
    kernel = np.ones((ksize, ksize), dtype=np.uint8)
    # Only keep the filtered value where the whole window was already valid,
    # so missing depth does not pull nearby surfaces toward zero.
    window_valid = cv2.erode(valid.astype(np.uint8), kernel, iterations=1).astype(bool)
    output = raw.copy()
    output[window_valid] = filtered[window_valid]
    output[~valid] = 0.0
    return output


def temporal_median(depth_frames: list[np.ndarray]) -> np.ndarray:
    """Per-pixel median across frames. Pixels invalid in every frame stay zero."""

    if not depth_frames:
        raise ValueError("At least one depth frame is required")
    stack = np.stack([np.asarray(frame, dtype=np.float64) for frame in depth_frames], axis=0)
    stack[stack <= 0] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        fused = np.nanmedian(stack, axis=0)
    result = np.nan_to_num(fused, nan=0.0).astype(np.float32)
    return result


def valid_fraction(depth_m: np.ndarray) -> float:
    if depth_m.size == 0:
        return 0.0
    return float(np.count_nonzero(depth_m > 0) / depth_m.size)
