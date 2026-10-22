"""Step 2: calibrate the wrist RGB camera intrinsics.

Run from the repository root with the virtual environment activated:

    aimbot-step02 devices
    aimbot-step02 --device 2 probe
    aimbot-step02 --device 2 --width 1280 --height 720 calibrate --cols 9 --rows 6 --square-mm 25

Camera options (--device, --width, --height) go before the subcommand.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from soarm_aimbot.cameras.base import CameraError
from soarm_aimbot.cameras.rgb import RgbCamera, choose_device, list_v4l2_devices, probe_resolutions
from soarm_aimbot.geometry.checkerboard import (
    calibrate_checkerboard,
    checkerboard_object,
    coverage_warning,
    find_corners,
)
from soarm_aimbot.geometry.rgb_model import BROWN, FISHEYE, RgbCameraModel, load_rgb_intrinsics, save_rgb_intrinsics

DEFAULT_INTRINSICS = Path("calib/wrist_intrinsics.yaml")
DEFAULT_OUTPUT = Path("output/step02")
HOLD_OUT_THRESHOLD_PX = 0.5


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except CameraError as exc:
        print(f"Camera error: {exc}")
        return 1
    except (ValueError, FileNotFoundError, KeyError, RuntimeError) as exc:
        print(f"Could not complete Step 2: {exc}")
        return 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aimbot-step02",
        description="Step 2: calibrate the intrinsics and distortion of the wrist RGB camera.",
    )
    parser.add_argument("--device", type=int, default=None, help="/dev/video index. Required when several cameras are connected")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    sub = parser.add_subparsers(dest="command", required=True)

    devices = sub.add_parser("devices", help="List /dev/video* nodes and their names")
    devices.set_defaults(func=_cmd_devices)

    probe = sub.add_parser("probe", help="Try several common resolutions and print the ones the camera actually accepts")
    probe.set_defaults(func=_cmd_probe)

    check = sub.add_parser("self-check", help="Check the principal-point ray without opening the camera")
    check.set_defaults(func=_cmd_self_check)

    show = sub.add_parser("show", help="Print the saved wrist intrinsics and an example ray")
    show.add_argument("--intrinsics", type=Path, default=DEFAULT_INTRINSICS)
    show.set_defaults(func=_cmd_show)

    preview = sub.add_parser("preview", help="Preview the wrist camera. Press [ ] to adjust focus, q to quit")
    preview.set_defaults(func=_cmd_preview)

    calibrate = sub.add_parser("calibrate", help="Capture checkerboard views and write the intrinsics")
    calibrate.add_argument("--output", type=Path, default=DEFAULT_INTRINSICS)
    calibrate.add_argument("--preview-dir", type=Path, default=DEFAULT_OUTPUT)
    calibrate.add_argument("--cols", type=int, required=True, help="Number of inner-corner columns on the checkerboard")
    calibrate.add_argument("--rows", type=int, required=True, help="Number of inner-corner rows on the checkerboard")
    calibrate.add_argument("--square-mm", type=float, required=True, help="Side length of one square measured on the printout, in millimeters")
    calibrate.add_argument("--views", type=int, default=20, help="Minimum number of views to capture")
    calibrate.add_argument("--holdout", type=int, default=4, help="Number of evenly spaced views held out of the solve")
    calibrate.add_argument("--model", choices=(BROWN, FISHEYE), default=BROWN)
    calibrate.add_argument("--focus", type=float, default=None, help="Focus value to lock, taken from the number shown in preview")
    calibrate.set_defaults(func=_cmd_calibrate)

    undistort = sub.add_parser("undistort", help="Show the raw and undistorted images side by side")
    undistort.add_argument("--intrinsics", type=Path, default=DEFAULT_INTRINSICS)
    undistort.set_defaults(func=_cmd_undistort)
    return parser


def _cmd_devices(args: argparse.Namespace) -> int:
    devices = list_v4l2_devices()
    if not devices:
        print("No /dev/video* found. Check that the wrist camera is plugged in and that the current user is in the video group.")
        return 1
    for device in devices:
        name = str(device["name"])
        kind = "RealSense node" if "realsense" in name.lower() else "RGB candidate"
        print(f"{kind}: /dev/video{device['index']}  {name}")
    try:
        chosen = choose_device(devices, args.device)
    except ValueError as exc:
        print(exc)
        return 1
    print(f"Without --device, /dev/video{chosen} will be used")
    return 0


def _cmd_probe(args: argparse.Namespace) -> int:
    device = choose_device(list_v4l2_devices(), args.device)
    print(f"Trying common resolutions on /dev/video{device}.")
    accepted = probe_resolutions(device)
    if not accepted:
        print("None of the common resolutions were accepted as requested. Try another mode, or check the resolutions listed in the camera's documentation.")
        return 1
    for width, height in accepted:
        print(f"Available: {width}x{height}")
    print("Use one of these resolutions for calibrate and for all later wrist images.")
    return 0


def _cmd_self_check(_args: argparse.Namespace) -> int:
    model = RgbCameraModel(
        width=640,
        height=480,
        fx=500.0,
        fy=500.0,
        cx=320.0,
        cy=240.0,
        distortion=(0.0, 0.0, 0.0, 0.0, 0.0),
    )
    center = model.pixel_to_ray(320, 240)
    offset = model.pixel_to_ray(420, 240)
    if not np.allclose(center, (0.0, 0.0, 1.0), atol=1e-6):
        print(f"The principal-point ray is not along the optical axis: {center}")
        return 1
    if not (offset[0] > 0 and abs(offset[2]) < 1):
        print(f"The ray for a pixel right of the principal point should point toward +X: {offset}")
        return 1
    print("Ray check passed: principal point -> (0, 0, 1), and a pixel right of the principal point has a positive X component. Axes are X right, Y down, Z forward.")
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    model, metadata = load_rgb_intrinsics(args.intrinsics)
    _print_model(model, metadata)
    return 0


def _cmd_preview(args: argparse.Namespace) -> int:
    _require_display()
    device = choose_device(list_v4l2_devices(), args.device)
    print(f"Previewing /dev/video{device} at {args.width}x{args.height}.")
    print("Once the image is sharp, note the focus number. Press [ or ] to change focus in steps of 5, q to quit.")
    with RgbCamera(device, args.width, args.height) as camera:
        while True:
            frame = camera.read()
            focus = camera.focus_value()
            label = "focus unavailable" if focus is None else f"focus {focus:.0f}"
            cv2.putText(frame, label, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("step02 wrist preview", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord("["), ord("]")) and focus is not None:
                step = -5 if key == ord("[") else 5
                camera.set_focus(focus + step)
    cv2.destroyAllWindows()
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    _require_display()
    if args.views < 1:
        raise ValueError("--views must be at least 1")
    device = choose_device(list_v4l2_devices(), args.device)
    square_m = args.square_mm / 1000.0
    board = checkerboard_object(args.cols, args.rows, square_m)
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    images: list[np.ndarray] = []
    print(
        f"Wrist camera /dev/video{device}, {args.width}x{args.height}, model {args.model}. "
        f"{args.cols}x{args.rows} inner corners, {args.square_mm} mm squares. "
        "Press SPACE to capture when the corners are detected, c to compute, q to abort."
    )
    with RgbCamera(device, args.width, args.height) as camera:
        if args.focus is not None:
            camera.set_focus(args.focus)
        focus_value = camera.focus_value()
        autofocus_disabled = camera.autofocus_disabled
        while True:
            frame = camera.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners = find_corners(gray, args.cols, args.rows)
            display = frame.copy()
            if corners is not None:
                cv2.drawChessboardCorners(display, (args.cols, args.rows), corners, True)
            cv2.putText(
                display,
                f"views {len(images)}/{args.views}",
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.imshow("step02 calibrate", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                cv2.destroyAllWindows()
                print("Aborted. No intrinsics file was written.")
                return 2
            if key == ord(" ") and corners is not None:
                object_points.append(board.copy())
                image_points.append(corners)
                images.append(frame.copy())
                print(f"Captured {len(images)} views")
            if key == ord(" ") and corners is None:
                print("This frame does not show the full checkerboard; not captured.")
            if key == ord("c"):
                if len(images) < args.views:
                    print(f"Need at least {args.views} views, have {len(images)}.")
                    continue
                result = calibrate_checkerboard(
                    object_points,
                    image_points,
                    (args.width, args.height),
                    args.model,
                    args.holdout,
                )
                break
    cv2.destroyAllWindows()
    warning = coverage_warning(image_points, args.width, args.height)
    if warning:
        print(warning)
    _write_views(args.preview_dir, images, image_points, args.cols, args.rows)
    sample, _new_k = result.model.undistort_image(images[-1])
    _write_sample(args.preview_dir, images[-1], sample)
    accepted = _accepted(result.holdout_mean_error_px, result.calibration_mean_error_px)
    ray = result.model.pixel_to_ray(result.model.cx, result.model.cy)
    save_rgb_intrinsics(
        args.output,
        result.model,
        {
            "alignment": "none",
            "source": "checkerboard",
            "device_index": device,
            "device_path": f"/dev/video{device}",
            "focus": None if focus_value is None else float(focus_value),
            "autofocus_disabled": autofocus_disabled,
            "checkerboard_cols": args.cols,
            "checkerboard_rows": args.rows,
            "checkerboard_square_m": square_m,
            "views": len(images),
            "holdout_views": args.holdout,
            "reprojection_rms_px": result.reprojection_rms_px,
            "calibration_mean_error_px": result.calibration_mean_error_px,
            "holdout_mean_error_px": result.holdout_mean_error_px,
            "threshold_px": HOLD_OUT_THRESHOLD_PX,
            "accepted": accepted,
            "fisheye_check_cond": result.fisheye_check_cond,
            "view_errors_px": [
                {"index": item.index, "role": item.role, "mean_error_px": item.mean_error_px}
                for item in result.view_errors
            ],
            "example_ray": {
                "pixel": [float(result.model.cx), float(result.model.cy)],
                "direction_unit": [float(value) for value in ray],
                "axes": "X right, Y down, Z forward",
            },
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
    _print_calibration(result, accepted)
    print(f"Wrote {args.output}")
    print(f"Captured images are in {args.preview_dir}")
    return 0 if accepted else 1


def _cmd_undistort(args: argparse.Namespace) -> int:
    _require_display()
    if not args.intrinsics.exists():
        raise FileNotFoundError(f"{args.intrinsics} does not exist yet. Run calibrate first.")
    model, metadata = load_rgb_intrinsics(args.intrinsics)
    device = choose_device(list_v4l2_devices(), args.device if args.device is not None else metadata.get("device_index"))
    if metadata.get("device_index") not in (None, device):
        print(f"The intrinsics file was recorded on /dev/video{metadata['device_index']}; opening /dev/video{device} this time.")
    print("Left: raw image. Right: undistorted image. Press q to quit.")
    with RgbCamera(device, model.width, model.height) as camera:
        focus = metadata.get("focus")
        if focus is not None:
            camera.set_focus(float(focus))
        while True:
            frame = camera.read()
            undistorted, _new_k = model.undistort_image(frame)
            view = np.hstack([frame, undistorted])
            cv2.imshow("step02 raw | undistorted", view)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break
    cv2.destroyAllWindows()
    return 0


def _accepted(holdout_mean: float | None, calibration_mean: float) -> bool:
    compared = calibration_mean if holdout_mean is None else holdout_mean
    return compared <= HOLD_OUT_THRESHOLD_PX


def _print_calibration(result, accepted: bool) -> None:
    model = result.model
    print(f"Model {model.model}  resolution {model.width}x{model.height}")
    print(f"fx={model.fx:.4f}  fy={model.fy:.4f}  cx={model.cx:.4f}  cy={model.cy:.4f}")
    print(f"Distortion: {tuple(round(value, 6) for value in model.distortion)}")
    print(f"Mean reprojection error of the solved views: {result.calibration_mean_error_px:.4f} px")
    if result.holdout_mean_error_px is None:
        print("No views were held out.")
        compared = result.calibration_mean_error_px
    else:
        print(f"Mean reprojection error of the held-out views: {result.holdout_mean_error_px:.4f} px")
        compared = result.holdout_mean_error_px
    if accepted:
        print(f"Held-out error {compared:.4f} px is within {HOLD_OUT_THRESHOLD_PX} px.")
    else:
        print(
            f"Held-out error {compared:.4f} px is above {HOLD_OUT_THRESHOLD_PX} px. "
            "Add more poses near the image edges and tilted poses, then calibrate again. The file was written with accepted: false."
        )
    if result.fisheye_check_cond is False:
        print("The fisheye solve relaxed its condition check. If straight lines still look curved in the undistorted image, add more tilted poses and calibrate again.")


def _print_model(model: RgbCameraModel, metadata: dict) -> None:
    print(f"Model {model.model}  resolution {model.width}x{model.height}")
    print(f"fx={model.fx:.4f}  fy={model.fy:.4f}  cx={model.cx:.4f}  cy={model.cy:.4f}")
    print(f"Distortion: {tuple(round(value, 6) for value in model.distortion)}")
    if model.undistorted_fx is not None:
        print(
            "Undistorted image intrinsics: "
            f"fx={model.undistorted_fx:.4f} fy={model.undistorted_fy:.4f} "
            f"cx={model.undistorted_cx:.4f} cy={model.undistorted_cy:.4f}"
        )
    holdout = metadata.get("holdout_mean_error_px")
    accepted = metadata.get("accepted")
    print(f"Held-out mean error: {holdout} px, accepted={accepted}")
    ray = model.pixel_to_ray(model.cx, model.cy)
    print(f"Principal-point ray: ({ray[0]:.4f}, {ray[1]:.4f}, {ray[2]:.4f})")


def _write_views(
    directory: Path,
    images: list[np.ndarray],
    corners: list[np.ndarray],
    cols: int,
    rows: int,
) -> None:
    view_dir = directory / "views"
    view_dir.mkdir(parents=True, exist_ok=True)
    for index, (image, found) in enumerate(zip(images, corners)):
        cv2.imwrite(str(view_dir / f"view_{index:02d}.png"), image)
        drawn = image.copy()
        cv2.drawChessboardCorners(drawn, (cols, rows), found, True)
        cv2.imwrite(str(view_dir / f"view_{index:02d}_corners.png"), drawn)


def _write_sample(directory: Path, raw: np.ndarray, undistorted: np.ndarray) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(directory / "raw_sample.png"), raw)
    cv2.imwrite(str(directory / "undistorted_sample.png"), undistorted)
    cv2.imwrite(str(directory / "raw_and_undistorted.png"), np.hstack([raw, undistorted]))


def _require_display() -> None:
    import os

    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return
    raise CameraError("No graphical display is available. devices, probe, self-check and show work without one.")


if __name__ == "__main__":
    raise SystemExit(main())
