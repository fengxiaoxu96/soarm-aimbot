"""Step 1: align RealSense color and depth, then check the color intrinsics.

Run from the repository root with the virtual environment activated:

    aimbot-step01 devices
    aimbot-step01 save
    aimbot-step01 verify --width-mm 40 --height-mm 30
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from soarm_aimbot.cameras.base import CameraError, RgbdFrame
from soarm_aimbot.geometry.depth import spatial_median, temporal_median, valid_fraction
from soarm_aimbot.geometry.intrinsics import PinholeIntrinsics, load_intrinsics, save_intrinsics
from soarm_aimbot.geometry.measure import BlockMeasurement, measure_block

DEFAULT_INTRINSICS = Path("calib/overhead_intrinsics.yaml")
DEFAULT_OUTPUT = Path("output/step01")


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except CameraError as exc:
        print(f"Camera error: {exc}")
        return 1
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"Could not complete Step 1: {exc}")
        return 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aimbot-step01",
        description="Step 1: align the overhead RealSense color and depth streams and confirm the intrinsics.",
    )
    parser.add_argument("--width", type=int, default=848)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--serial", default=None, help="Serial number to use when several RealSense cameras are connected")
    parser.add_argument(
        "--name-substring",
        default="455",
        help="Text the device name must contain. D455C contains 455. Pass an empty string to accept any RealSense.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    devices = sub.add_parser("devices", help="List the RealSense cameras connected to this computer")
    devices.set_defaults(func=_cmd_devices)

    check = sub.add_parser("self-check", help="Check the back-projection formula against known values without opening the camera")
    check.set_defaults(func=_cmd_self_check)

    save = sub.add_parser("save", help="Save the factory intrinsics and write an aligned color/depth preview")
    _add_capture_args(save)
    save.add_argument("--output", type=Path, default=DEFAULT_INTRINSICS)
    save.add_argument("--preview-dir", type=Path, default=DEFAULT_OUTPUT)
    save.set_defaults(func=_cmd_save)

    preview = sub.add_parser("preview", help="Show color and aligned depth side by side. Press s to save, q to quit")
    _add_capture_args(preview)
    preview.add_argument("--output", type=Path, default=DEFAULT_INTRINSICS)
    preview.add_argument("--preview-dir", type=Path, default=DEFAULT_OUTPUT)
    preview.set_defaults(func=_cmd_preview)

    calibrate = sub.add_parser("calibrate", help="Recalibrate the color intrinsics with a checkerboard, replacing the focal length and principal point in the yaml")
    _add_capture_args(calibrate)
    calibrate.add_argument("--output", type=Path, default=DEFAULT_INTRINSICS)
    calibrate.add_argument("--cols", type=int, required=True, help="Number of inner-corner columns on the checkerboard")
    calibrate.add_argument("--rows", type=int, required=True, help="Number of inner-corner rows on the checkerboard")
    calibrate.add_argument("--square-mm", type=float, required=True, help="Side length of one square, in millimeters")
    calibrate.add_argument("--views", type=int, default=12, help="Minimum number of views to capture")
    calibrate.set_defaults(func=_cmd_calibrate)

    verify = sub.add_parser("verify", help="Check back-projected width and thickness against a block of known size")
    _add_capture_args(verify)
    verify.add_argument("--intrinsics", type=Path, default=DEFAULT_INTRINSICS)
    verify.add_argument("--preview-dir", type=Path, default=DEFAULT_OUTPUT)
    verify.add_argument("--width-mm", type=float, required=True, help="Measured size of the block's top face along the image width, in millimeters")
    verify.add_argument("--height-mm", type=float, required=True, help="Measured thickness of the block, in millimeters")
    verify.add_argument("--length-mm", type=float, default=None, help="Measured size of the top face along the image height, in millimeters (optional)")
    verify.add_argument("--top-roi", type=_roi, default=None, help="Top-face ROI, as x,y,w,h")
    verify.add_argument("--table-roi", type=_roi, default=None, help="Table ROI, as x,y,w,h")
    verify.set_defaults(func=_cmd_verify)
    return parser


def _add_capture_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--warmup", type=int, default=30, help="Number of warm-up frames to discard")
    parser.add_argument("--frames", type=int, default=15, help="Number of frames in the temporal median")
    parser.add_argument("--median-ksize", type=int, default=5, help="Spatial median kernel size; 1 disables it")


def _cmd_devices(_args: argparse.Namespace) -> int:
    from soarm_aimbot.cameras.realsense import list_devices

    devices = list_devices()
    if not devices:
        print("No RealSense found. Check the USB connection. If the device is present but cannot be opened, install the udev rules and replug it.")
        return 1
    for device in devices:
        mark = "overhead candidate" if "455" in device["name"] else "other"
        print(f"{mark}: {device['name']}  serial={device['serial']}  firmware={device['firmware']}")
    return 0


def _cmd_self_check(_args: argparse.Namespace) -> int:
    model = PinholeIntrinsics(width=640, height=480, fx=600.0, fy=600.0, cx=320.0, cy=240.0)
    origin = model.backproject(320, 240, 1.0)
    offset = model.backproject(620, 240, 1.0)
    if not np.allclose(origin, (0.0, 0.0, 1.0)) or not np.allclose(offset, (0.5, 0.0, 1.0)):
        print(f"Back-projection does not match the formula: center {origin}, 300 px to the right {offset}")
        return 1
    print("Back-projection check passed: principal point at depth 1 m -> (0, 0, 1); 300 px right of the principal point with fx=600 -> X=0.5 m.")
    return 0


def _cmd_save(args: argparse.Namespace) -> int:
    frame = _capture(args)
    example = _center_example(frame)
    _write_intrinsics(args.output, frame, source="realsense_color_factory", example=example, fps=args.fps)
    _write_preview(args.preview_dir, frame, example)
    _print_intrinsics(frame, example)
    fraction = valid_fraction(frame.depth_m)
    print(f"Valid depth pixel fraction: {fraction:.1%}")
    if fraction < 0.2:
        print("Too little valid depth. Check for lens obstruction and working distance, and whether color edges line up with depth edges.")
    print(f"Wrote {args.output}")
    return 0


def _cmd_preview(args: argparse.Namespace) -> int:
    from soarm_aimbot.cameras.realsense import RealSenseRgbd

    _require_display()
    camera = _camera_from_args(args, RealSenseRgbd)
    print("Previewing. Press s to save the intrinsics and the current aligned frame, q to quit.")
    with camera:
        while True:
            frame = camera.read()
            view = _side_by_side(frame.color_bgr, frame.depth_m)
            cv2.imshow("step01 overhead: color | aligned depth", view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                stable = _fuse(camera, args.warmup, args.frames, args.median_ksize)
                example = _center_example(stable)
                _write_intrinsics(
                    args.output, stable, source="realsense_color_factory", example=example, fps=args.fps
                )
                _write_preview(args.preview_dir, stable, example)
                print(f"Wrote {args.output}")
    cv2.destroyAllWindows()
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    from soarm_aimbot.cameras.realsense import RealSenseRgbd

    _require_display()
    square_m = args.square_mm / 1000.0
    pattern = (args.cols, args.rows)
    object_point = np.zeros((args.rows * args.cols, 3), np.float32)
    object_point[:, :2] = np.mgrid[0 : args.cols, 0 : args.rows].T.reshape(-1, 2)
    object_point *= square_m
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    camera = _camera_from_args(args, RealSenseRgbd)
    print(
        f"Place the checkerboard in the overhead view: {args.cols}x{args.rows} inner corners, {args.square_mm} mm squares. "
        "Press SPACE to capture when the corners are detected, c to compute, q to abort."
    )
    model: PinholeIntrinsics | None = None
    with camera:
        while True:
            frame = camera.read()
            gray = cv2.cvtColor(frame.color_bgr, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(gray, pattern)
            display = frame.color_bgr.copy()
            if found:
                cv2.drawChessboardCorners(display, pattern, corners, found)
            cv2.putText(
                display,
                f"views {len(object_points)}/{args.views}",
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.imshow("step01 calibrate", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                cv2.destroyAllWindows()
                print("Aborted. The intrinsics file was not changed.")
                return 2
            if key == ord(" ") and found:
                refined = cv2.cornerSubPix(
                    gray,
                    corners,
                    (11, 11),
                    (-1, -1),
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
                )
                object_points.append(object_point.copy())
                image_points.append(refined)
                print(f"Captured {len(object_points)} views")
            if key == ord("c"):
                if len(object_points) < args.views:
                    print(f"Need at least {args.views} views, have {len(object_points)}.")
                    continue
                model = _solve_checkerboard(frame, object_points, image_points)
                break
    cv2.destroyAllWindows()
    if model is None:
        return 2
    _write_intrinsics(
        args.output,
        frame,
        source="checkerboard",
        example=None,
        fps=args.fps,
        override=model,
        extra={
            "checkerboard_cols": args.cols,
            "checkerboard_rows": args.rows,
            "checkerboard_square_m": square_m,
            "views": len(object_points),
        },
    )
    print(f"Overwrote {args.output} with the checkerboard intrinsics")
    print("Depth-to-color alignment still uses the camera's internal extrinsics. Next, run verify to measure the block again.")
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    if not args.intrinsics.exists():
        raise FileNotFoundError(f"{args.intrinsics} does not exist yet. Run aimbot-step01 save first.")
    saved, metadata = load_intrinsics(args.intrinsics)
    from soarm_aimbot.cameras.realsense import RealSenseRgbd

    camera = RealSenseRgbd(
        width=saved.width,
        height=saved.height,
        fps=int(metadata.get("fps", args.fps)),
        serial=args.serial or metadata.get("serial"),
        name_substring=args.name_substring,
    )
    with camera:
        _warn_if_intrinsics_differ(saved, camera.intrinsics)
        frame = _fuse(camera, args.warmup, args.frames, args.median_ksize)
        captured = RgbdFrame(
            color_bgr=frame.color_bgr,
            depth_m=frame.depth_m,
            intrinsics=saved,
            timestamp_s=frame.timestamp_s,
            serial=frame.serial,
            device_name=frame.device_name,
        )
    top_roi, table_roi = _resolve_rois(args.top_roi, args.table_roi, captured.color_bgr)
    measurement = measure_block(captured.depth_m, saved, top_roi, table_roi)
    report_path = _write_verification(args, measurement, saved)
    _print_verification(args, measurement)
    print(f"Verification record written to {report_path}")
    return 0


def _capture(args: argparse.Namespace) -> RgbdFrame:
    from soarm_aimbot.cameras.realsense import RealSenseRgbd

    camera = _camera_from_args(args, RealSenseRgbd)
    with camera:
        return _fuse(camera, args.warmup, args.frames, args.median_ksize)


def _camera_from_args(args: argparse.Namespace, camera_type):
    return camera_type(
        width=args.width,
        height=args.height,
        fps=args.fps,
        serial=args.serial,
        name_substring=args.name_substring,
    )


def _fuse(camera, warmup: int, count: int, median_ksize: int) -> RgbdFrame:
    if count < 1:
        raise ValueError("--frames must be at least 1")
    for _ in range(max(warmup, 0)):
        camera.read()
    frames = [camera.read() for _ in range(count)]
    depth = temporal_median([frame.depth_m for frame in frames])
    depth = spatial_median(depth, median_ksize)
    middle = frames[len(frames) // 2]
    return RgbdFrame(
        color_bgr=middle.color_bgr,
        depth_m=depth,
        intrinsics=middle.intrinsics,
        timestamp_s=middle.timestamp_s,
        serial=middle.serial,
        device_name=middle.device_name,
    )


def _solve_checkerboard(frame: RgbdFrame, object_points, image_points) -> PinholeIntrinsics:
    image_size = (frame.color_bgr.shape[1], frame.color_bgr.shape[0])
    rms, matrix, distortion, _rvecs, _tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )
    print(f"Checkerboard reprojection error: {rms:.4f} px")
    if rms > 0.5:
        print("Reprojection error is above 0.5 px. Add more tilted poses and compute again, and confirm --square-mm is the real square size.")
    previous = frame.intrinsics
    return PinholeIntrinsics(
        width=previous.width,
        height=previous.height,
        fx=float(matrix[0, 0]),
        fy=float(matrix[1, 1]),
        cx=float(matrix[0, 2]),
        cy=float(matrix[1, 2]),
        distortion=tuple(float(value) for value in distortion.reshape(-1)),
        depth_scale_m=previous.depth_scale_m,
    )


def _write_intrinsics(
    path: Path,
    frame: RgbdFrame,
    source: str,
    example: dict | None,
    fps: int,
    override: PinholeIntrinsics | None = None,
    extra: dict | None = None,
) -> None:
    model = override or frame.intrinsics
    metadata = {
        "alignment": "depth_to_color",
        "source": source,
        "device_name": frame.device_name,
        "serial": frame.serial,
        "fps": fps,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "valid_depth_fraction": round(valid_fraction(frame.depth_m), 4),
    }
    if example is not None:
        metadata["example_backproject"] = example
    if extra:
        metadata.update(extra)
    save_intrinsics(path, model, metadata)


def _write_preview(directory: Path, frame: RgbdFrame, example: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    view = _side_by_side(frame.color_bgr, frame.depth_m)
    point = example["point_m"]
    label = f"center Z={example['depth_m']:.3f}m  XYZ=({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f})"
    cv2.putText(view, label, (16, view.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imwrite(str(directory / "aligned_preview.png"), view)
    cv2.imwrite(str(directory / "color.png"), frame.color_bgr)
    np.save(directory / "depth_m.npy", frame.depth_m)


def _write_verification(args: argparse.Namespace, measurement: BlockMeasurement, model: PinholeIntrinsics) -> Path:
    args.preview_dir.mkdir(parents=True, exist_ok=True)
    width_m = args.width_mm / 1000.0
    height_m = args.height_mm / 1000.0
    payload = {
        "intrinsics": str(args.intrinsics),
        "fx": model.fx,
        "fy": model.fy,
        "measured_width_m": width_m,
        "measured_height_m": height_m,
        "estimated_width_m": measurement.width_m,
        "estimated_length_m": measurement.length_m,
        "estimated_height_m": measurement.height_m,
        "width_error_m": measurement.width_m - width_m,
        "height_error_m": measurement.height_m - height_m,
        "top_depth_m": measurement.top_depth_m,
        "table_depth_m": measurement.table_depth_m,
        "top_roi": list(measurement.top_roi),
        "table_roi": list(measurement.table_roi),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    if args.length_mm is not None:
        length_m = args.length_mm / 1000.0
        payload["measured_length_m"] = length_m
        payload["length_error_m"] = measurement.length_m - length_m
    path = args.preview_dir / "verify.yaml"
    import yaml

    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def _print_intrinsics(frame: RgbdFrame, example: dict) -> None:
    model = frame.intrinsics
    point = example["point_m"]
    print(f"Device: {frame.device_name}  serial={frame.serial}")
    print(f"Resolution: {model.width}x{model.height}  alignment: depth_to_color")
    print(f"fx={model.fx:.4f}  fy={model.fy:.4f}  cx={model.cx:.4f}  cy={model.cy:.4f}")
    print(f"Distortion: {tuple(round(v, 6) for v in model.distortion)}")
    print(
        "Image center back-projection: "
        f"pixel {example['pixel']}, depth {example['depth_m']:.4f} m -> "
        f"({point[0]:.4f}, {point[1]:.4f}, {point[2]:.4f}) m"
    )


def _print_verification(args: argparse.Namespace, measurement: BlockMeasurement) -> None:
    width_mm = measurement.width_m * 1000.0
    height_mm = measurement.height_m * 1000.0
    print(f"Top-face depth {measurement.top_depth_m * 1000:.1f} mm, table depth {measurement.table_depth_m * 1000:.1f} mm")
    print(
        f"Back-projected width {width_mm:.1f} mm, measured {args.width_mm:.1f} mm, "
        f"difference {width_mm - args.width_mm:.1f} mm"
    )
    print(
        f"Thickness along the optical axis {height_mm:.1f} mm, measured {args.height_mm:.1f} mm, "
        f"difference {height_mm - args.height_mm:.1f} mm"
    )
    print(f"Back-projected length of the other top-face side {measurement.length_m * 1000:.1f} mm")
    print("When the camera points roughly straight down, the thickness along the optical axis is the block height. The width is between the left and right edges of the top face you selected.")
    if args.width_mm > 0 and abs(width_mm - args.width_mm) / args.width_mm > 0.05:
        print("Width error is above 5%. Make sure the ROI hugs the top-face edges; if the edges are right, run calibrate to recalibrate the color intrinsics.")
    if abs(height_mm - args.height_mm) > 10:
        print("Thickness error is above 10 mm. Make sure the table ROI is on the table next to the block and that depth is aligned to color.")


def _center_example(frame: RgbdFrame) -> dict:
    depth = frame.depth_m
    height, width = depth.shape
    cx = width // 2
    cy = height // 2
    y0, y1 = max(0, cy - 40), min(height, cy + 40)
    x0, x1 = max(0, cx - 40), min(width, cx + 40)
    window = depth[y0:y1, x0:x1]
    valid = window[window > 0]
    if valid.size == 0:
        raise CameraError("No valid depth at the image center, so no back-projection example can be given.")
    z_m = float(np.median(valid))
    point = frame.intrinsics.backproject(cx, cy, z_m)
    return {
        "pixel": [int(cx), int(cy)],
        "depth_m": z_m,
        "point_m": [float(value) for value in point],
    }


def _side_by_side(color_bgr: np.ndarray, depth_m: np.ndarray) -> np.ndarray:
    depth_bgr = _colorize_depth(depth_m)
    view = np.hstack([color_bgr, depth_bgr])
    cv2.putText(view, "color", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(
        view,
        "depth aligned to color",
        (color_bgr.shape[1] + 16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    return view


def _colorize_depth(depth_m: np.ndarray) -> np.ndarray:
    valid = depth_m > 0
    canvas = np.zeros((*depth_m.shape, 3), dtype=np.uint8)
    if not np.any(valid):
        return canvas
    lo, hi = np.percentile(depth_m[valid], [5, 95])
    if hi <= lo:
        hi = lo + 1e-3
    normalized = np.clip((depth_m - lo) / (hi - lo), 0.0, 1.0)
    gray = (normalized * 255).astype(np.uint8)
    colored = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored


def _resolve_rois(top_roi, table_roi, color_bgr: np.ndarray):
    if top_roi is None or table_roi is None:
        _require_display()
    if top_roi is None:
        print("Draw a box around the block's top face and press Enter. The top face should fill the box without including any table.")
        top_roi = _select_one_roi("step01: top face, then ENTER", color_bgr)
    if table_roi is None:
        print("Draw a box on the table next to the block and press Enter. Do not include the block.")
        table_roi = _select_one_roi("step01: table patch, then ENTER", color_bgr)
    return top_roi, table_roi


def _select_one_roi(title: str, image: np.ndarray) -> tuple[int, int, int, int]:
    roi = cv2.selectROI(title, image, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(title)
    selected = tuple(int(value) for value in roi)
    if selected[2] < 2 or selected[3] < 2:
        raise ValueError("No ROI was selected")
    return selected  # type: ignore[return-value]


def _warn_if_intrinsics_differ(saved: PinholeIntrinsics, live: PinholeIntrinsics | None) -> None:
    if live is None:
        return
    if saved.width != live.width or saved.height != live.height:
        raise CameraError("The saved resolution does not match the current camera stream. Run save again at the same resolution.")
    delta = max(abs(saved.fx - live.fx), abs(saved.fy - live.fy))
    if delta > 1.0:
        print(
            f"The fx/fy in the file differ from the camera's current factory values by {delta:.2f} px. "
            "This check uses the intrinsics from the file."
        )


def _require_display() -> None:
    import os

    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return
    raise CameraError("No graphical display is available. save works without one; for verify, pass --top-roi and --table-roi.")


def _roi(text: str) -> tuple[int, int, int, int]:
    parts = text.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("ROI format is x,y,w,h")
    values = tuple(int(part) for part in parts)
    return values  # type: ignore[return-value]


if __name__ == "__main__":
    raise SystemExit(main())
