"""Frames shared by RGB-D adapters. No vendor SDK imports."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from soarm_aimbot.geometry.intrinsics import PinholeIntrinsics


class CameraError(RuntimeError):
    """The camera could not be opened or did not deliver an aligned frame."""


@dataclass
class RgbdFrame:
    """One color image and a depth map already aligned to that color image.

    ``depth_m`` is float32, in meters. Zero means the pixel has no depth.
    ``intrinsics`` are the color camera's intrinsics, which are the correct
    model for depth that has been aligned onto the color image.
    """

    color_bgr: np.ndarray
    depth_m: np.ndarray
    intrinsics: PinholeIntrinsics
    timestamp_s: float
    serial: str
    device_name: str

    def __post_init__(self) -> None:
        if self.color_bgr.shape[:2] != self.depth_m.shape[:2]:
            raise CameraError(
                "Color and depth image sizes differ:"
                f" color {self.color_bgr.shape[:2]}, depth {self.depth_m.shape[:2]}"
            )
