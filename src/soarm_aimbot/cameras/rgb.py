"""Generic USB RGB camera via OpenCV. The wrist camera uses this adapter.

RealSense also exposes ``/dev/video*`` nodes. Callers choose the index; this
module only opens the index it is given.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from soarm_aimbot.cameras.base import CameraError


class RgbCamera:
    """One RGB stream locked to the requested resolution."""

    def __init__(self, device: int, width: int, height: int) -> None:
        self.device = device
        self.width = width
        self.height = height
        self._cap: cv2.VideoCapture | None = None
        self.autofocus_disabled = False

    def __enter__(self) -> "RgbCamera":
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.stop()

    def start(self) -> None:
        capture = _open_capture(self.device, self.width, self.height)
        self.autofocus_disabled = bool(capture.set(cv2.CAP_PROP_AUTOFOCUS, 0))
        _request_mode(capture, self.width, self.height)
        frame = _read_settled(capture)
        actual_h, actual_w = frame.shape[:2]
        if (actual_w, actual_h) != (self.width, self.height):
            capture.release()
            raise CameraError(
                f"Requested {self.width}x{self.height}, but the camera delivered {actual_w}x{actual_h}. "
                "Use a resolution printed by aimbot-step02 probe. Do not change the resolution after calibration."
            )
        self._cap = capture

    def read(self) -> np.ndarray:
        if self._cap is None:
            raise CameraError("Camera has not been started; call start() first")
        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise CameraError(f"No image was read from /dev/video{self.device}")
        return frame

    def focus_value(self) -> float | None:
        if self._cap is None:
            return None
        value = float(self._cap.get(cv2.CAP_PROP_FOCUS))
        if value < 0:
            return None
        return value

    def set_focus(self, value: float) -> None:
        if self._cap is None:
            raise CameraError("Camera has not been started; call start() first")
        self._cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.autofocus_disabled = True
        self._cap.set(cv2.CAP_PROP_FOCUS, float(value))

    def stop(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


def list_v4l2_devices(sysfs_root: Path | None = None) -> list[dict[str, str | int]]:
    root = sysfs_root or Path("/sys/class/video4linux")
    if not root.exists():
        return []
    devices = []
    for entry in sorted(root.iterdir()):
        if not entry.name.startswith("video"):
            continue
        suffix = entry.name.removeprefix("video")
        if not suffix.isdigit():
            continue
        name_file = entry / "name"
        name = name_file.read_text(encoding="utf-8").strip() if name_file.exists() else ""
        devices.append({"index": int(suffix), "path": f"/dev/{entry.name}", "name": name})
    return devices


def choose_device(devices: list[dict[str, str | int]], requested: int | None) -> int:
    indexes = {int(device["index"]) for device in devices}
    if requested is not None:
        if devices and requested not in indexes:
            known = ", ".join(f"/dev/video{device['index']}" for device in devices)
            raise ValueError(f"/dev/video{requested} does not exist. Available nodes: {known}")
        return requested
    wrist = [device for device in devices if "realsense" not in str(device["name"]).lower()]
    if len(wrist) == 1:
        return int(wrist[0]["index"])
    if not devices:
        raise ValueError("No /dev/video* found. Check that the wrist camera is plugged in and that the current user is in the video group.")
    if not wrist:
        raise ValueError("Only RealSense video nodes were found. Plug in the wrist RGB camera and pass its index with --device.")
    lines = "\n".join(f"  /dev/video{device['index']}  {device['name']}" for device in wrist)
    raise ValueError(f"Several non-RealSense cameras found. Pass the wrist camera with --device:\n{lines}")


def probe_resolutions(device: int) -> list[tuple[int, int]]:
    """Return requested sizes that the camera actually delivers."""

    candidates = [(640, 480), (800, 600), (1280, 720), (1920, 1080)]
    accepted = []
    for width, height in candidates:
        capture = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not capture.isOpened():
            capture.release()
            raise CameraError(f"Could not open /dev/video{device}")
        _request_mode(capture, width, height)
        frame = _read_settled(capture)
        capture.release()
        if frame.shape[1] == width and frame.shape[0] == height:
            accepted.append((width, height))
    return accepted


def _open_capture(device: int, width: int, height: int) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not capture.isOpened():
        capture.release()
        raise CameraError(
            f"Could not open /dev/video{device}. Make sure the index comes from aimbot-step02 devices "
            "and that the current user is in the video group."
        )
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    _request_mode(capture, width, height)
    return capture


def _request_mode(capture: cv2.VideoCapture, width: int, height: int) -> None:
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


def _read_settled(capture: cv2.VideoCapture, attempts: int = 8) -> np.ndarray:
    frame = None
    for _ in range(attempts):
        ok, grabbed = capture.read()
        if ok and grabbed is not None:
            frame = grabbed
    if frame is None:
        raise CameraError("The camera opened, but no image was read")
    return frame
