"""Intel RealSense RGB-D adapter. The overhead camera is a D455 / D455C.

The rest of the project talks to :class:`soarm_aimbot.cameras.base.RgbdFrame`.
Swapping the overhead camera means replacing this module, not the geometry.
"""

from __future__ import annotations

import numpy as np

from soarm_aimbot.cameras.base import CameraError, RgbdFrame
from soarm_aimbot.geometry.intrinsics import PinholeIntrinsics


class RealSenseRgbd:
    """Depth aligned onto the color image, with the color camera's intrinsics."""

    def __init__(
        self,
        width: int = 848,
        height: int = 480,
        fps: int = 30,
        serial: str | None = None,
        name_substring: str = "455",
        timeout_ms: int = 5000,
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.serial = serial
        self.name_substring = name_substring
        self.timeout_ms = timeout_ms
        self._pipeline = None
        self._align = None
        self._rs = None
        self.intrinsics: PinholeIntrinsics | None = None
        self.device_name = ""
        self.device_serial = ""
        self.firmware = ""

    def __enter__(self) -> "RealSenseRgbd":
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.stop()

    def start(self) -> PinholeIntrinsics:
        rs = _load_sdk()
        self._rs = rs
        pipeline = rs.pipeline()
        config = rs.config()
        if self.serial:
            config.enable_device(self.serial)
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        try:
            profile = pipeline.start(config)
        except RuntimeError as exc:
            devices = list_devices()
            found = ", ".join(f"{item['name']} ({item['serial']})" for item in devices) or "none"
            raise CameraError(
                "Could not open the RealSense color+depth streams at "
                f"{self.width}x{self.height}@{self.fps}. "
                f"Connected devices: {found}. Original error: {exc}"
            ) from exc

        device = profile.get_device()
        self.device_name = device.get_info(rs.camera_info.name)
        self.device_serial = device.get_info(rs.camera_info.serial_number)
        self.firmware = device.get_info(rs.camera_info.firmware_version)
        if self.name_substring and self.name_substring.lower() not in self.device_name.lower():
            pipeline.stop()
            raise CameraError(
                f"The opened camera is {self.device_name}, whose name does not contain {self.name_substring!r}. "
                "The overhead camera is expected to be a D455/D455C. If this is your overhead camera, pass --name-substring ''."
            )

        depth_sensor = device.first_depth_sensor()
        depth_scale = float(depth_sensor.get_depth_scale())
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.intrinsics = _intrinsics_from_rs(color_profile.get_intrinsics(), depth_scale)
        self._align = rs.align(rs.stream.color)
        self._pipeline = pipeline
        return self.intrinsics

    def read(self) -> RgbdFrame:
        if self._pipeline is None or self._align is None or self.intrinsics is None or self._rs is None:
            raise CameraError("Camera has not been started; call start() first")
        rs = self._rs
        depth_frame = None
        color_frame = None
        for _ in range(5):
            frames = self._pipeline.wait_for_frames(self.timeout_ms)
            aligned = self._align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if depth_frame and color_frame:
                break
        if depth_frame is None or color_frame is None:
            raise CameraError("Alignment did not produce both a color frame and a depth frame")

        color = np.asanyarray(color_frame.get_data())
        depth_raw = np.asanyarray(depth_frame.get_data())
        depth_m = depth_raw.astype(np.float32) * self.intrinsics.depth_scale_m
        timestamp_s = float(color_frame.get_timestamp()) / 1000.0
        return RgbdFrame(
            color_bgr=color,
            depth_m=depth_m,
            intrinsics=self.intrinsics,
            timestamp_s=timestamp_s,
            serial=self.device_serial,
            device_name=self.device_name,
        )

    def stop(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None
            self._align = None


def list_devices() -> list[dict[str, str]]:
    rs = _load_sdk()
    devices = []
    for device in rs.context().query_devices():
        devices.append(
            {
                "name": device.get_info(rs.camera_info.name),
                "serial": device.get_info(rs.camera_info.serial_number),
                "firmware": device.get_info(rs.camera_info.firmware_version),
            }
        )
    return devices


def _load_sdk():
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise CameraError(
            "pyrealsense2 is not installed. From the repository root run: uv sync"
        ) from exc
    return rs


def _intrinsics_from_rs(intrinsics, depth_scale_m: float) -> PinholeIntrinsics:
    coeffs = tuple(float(value) for value in intrinsics.coeffs)
    return PinholeIntrinsics(
        width=int(intrinsics.width),
        height=int(intrinsics.height),
        fx=float(intrinsics.fx),
        fy=float(intrinsics.fy),
        cx=float(intrinsics.ppx),
        cy=float(intrinsics.ppy),
        distortion=coeffs,
        depth_scale_m=depth_scale_m,
    )
