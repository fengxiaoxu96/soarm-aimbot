"""Step 3: fix the overhead camera to a table frame defined by one AprilTag.

Run from the repository root:

    aimbot-step03 print-markers --size-mm 120
    aimbot-step03 calibrate --size-mm 120
    aimbot-step03 check
    aimbot-step03 verify --x-mm 200 --y-mm 50
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from soarm_aimbot.cameras.base import CameraError
from soarm_aimbot.geometry.intrinsics import PinholeIntrinsics, load_intrinsics
from soarm_aimbot.geometry.marker import (
    DICTIONARIES,
    detect_marker,
    marker_object_points,
    pose_from_corners,
    render_marker,
    reprojection_error_px,
)
from soarm_aimbot.geometry.transforms import (
    RigidTransform,
    average_transforms,
    load_transform,
    rotation_angle_deg,
    save_transform,
    translation_distance_m,
)

DEFAULT_INTRINSICS = Path("calib/overhead_intrinsics.yaml")
DEFAULT_EXTRINSIC = Path("calib/T_world_overhead.yaml")
DEFAULT_OUTPUT = Path("output/step03")
DEFAULT_DICTIONARY = "DICT_APRILTAG_36h11"


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except CameraError as exc:
        print(f"Camera error: {exc}")
        return 1
    except (ValueError, FileNotFoundError, KeyError, RuntimeError) as exc:
        print(f"Could not complete Step 3: {exc}")
        return 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aimbot-step03",
        description="Step 3: calibrate the overhead camera to the table frame using a marker fixed to the table.",
    )
    parser.add_argument("--intrinsics", type=Path, default=DEFAULT_INTRINSICS)
    parser.add_argument("--dictionary", default=DEFAULT_DICTIONARY, choices=sorted(DICTIONARIES))
    parser.add_argument("--marker-id", type=int, default=0, help="id of the marker that defines the table origin")
    parser.add_argument("--serial", default=None)
    parser.add_argument("--name-substring", default="455")
    sub = parser.add_subparsers(dest="command", required=True)

    check_math = sub.add_parser("self-check", help="check the marker origin and axes without opening the camera")
    check_math.set_defaults(func=_cmd_self_check)

    sheets = sub.add_parser("print-markers", help="generate printable world and verification markers")
    sheets.add_argument("--size-mm", type=float, required=True, help="design side length of the black square, outer edge, mm")
    sheets.add_argument("--verify-id", type=int, default=1)
    sheets.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    sheets.set_defaults(func=_cmd_print)

    calibrate = sub.add_parser("calibrate", help="average many frames and write T_world_overhead")
    calibrate.add_argument("--size-mm", type=float, required=True, help="measured outer edge of the printed black square, mm")
    calibrate.add_argument("--output", type=Path, default=DEFAULT_EXTRINSIC)
    calibrate.add_argument("--preview-dir", type=Path, default=DEFAULT_OUTPUT)
    calibrate.add_argument("--samples", type=int, default=30)
    calibrate.add_argument("--warmup", type=int, default=15)
    calibrate.set_defaults(func=_cmd_calibrate)

    check = sub.add_parser("check", help="start-up re-check. Exits with code 1 if the drift exceeds the thresholds")
    check.add_argument("--extrinsic", type=Path, default=DEFAULT_EXTRINSIC)
    check.add_argument("--samples", type=int, default=20)
    check.add_argument("--warmup", type=int, default=15)
    check.add_argument("--trans-mm", type=float, default=5.0)
    check.add_argument("--rot-deg", type=float, default=1.0)
    check.add_argument("--update", action="store_true", help="overwrite the extrinsic file with this result when the thresholds are exceeded")
    check.set_defaults(func=_cmd_check)

    verify = sub.add_parser("verify", help="check table coordinates with a second marker")
    verify.add_argument("--extrinsic", type=Path, default=DEFAULT_EXTRINSIC)
    verify.add_argument("--verify-id", type=int, default=1)
    verify.add_argument("--x-mm", type=float, required=True, help="measured distance of the verification marker center along world +X, mm")
    verify.add_argument("--y-mm", type=float, required=True, help="measured distance of the verification marker center along world +Y, mm")
    verify.add_argument("--z-mm", type=float, default=0.0, help="height of the verification marker center above the world origin, mm")
    verify.add_argument("--tolerance-mm", type=float, default=10.0)
    verify.add_argument("--samples", type=int, default=20)
    verify.add_argument("--warmup", type=int, default=15)
    verify.set_defaults(func=_cmd_verify)
    return parser


def _cmd_self_check(_args: argparse.Namespace) -> int:
    intrinsics = PinholeIntrinsics(640, 480, 500, 500, 320, 240)
    size_m = 0.08
    rotation = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    translation = np.array([0.01, -0.02, 0.75])
    camera_from_tag = RigidTransform.from_rotation_translation(rotation, translation)
    image, _ = cv2.projectPoints(
        marker_object_points(size_m),
        cv2.Rodrigues(rotation)[0],
        translation.reshape(3, 1),
        intrinsics.camera_matrix,
        intrinsics.distortion_coeffs,
    )
    recovered = pose_from_corners(image.reshape(4, 2), size_m, intrinsics)
    world_from_camera = recovered.inverse()
    origin = world_from_camera.apply(recovered.translation)
    x_axis = world_from_camera.apply(recovered.apply(np.array([size_m / 2.0, 0.0, 0.0])))
    if np.linalg.norm(origin) > 1e-4 or np.linalg.norm(x_axis - np.array([size_m / 2.0, 0.0, 0.0])) > 1e-4:
        print(f"Table-frame round trip is inconsistent: origin {origin}, +X point {x_axis}")
        return 1
    print("Extrinsic check passed: the marker center maps to the table origin and a point along marker +X stays on world +X. Z points out of the marker face, that is, up toward the camera.")
    return 0


def _cmd_print(args: argparse.Namespace) -> int:
    if args.verify_id == args.marker_id:
        raise ValueError("The verification marker id must differ from the world marker id")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    world_path = args.output_dir / "world_marker.png"
    verify_path = args.output_dir / "verify_marker.png"
    _write_sheet(
        world_path,
        args.dictionary,
        args.marker_id,
        args.size_mm,
        title=f"world id {args.marker_id}",
        draw_axes=True,
    )
    _write_sheet(
        verify_path,
        args.dictionary,
        args.verify_id,
        args.size_mm,
        title=f"verify id {args.verify_id}",
        draw_axes=False,
    )
    print(f"Wrote {world_path}")
    print(f"Wrote {verify_path}")
    print("Print at 100% actual size. Measure the outer edge of the black square with calipers and pass that value to calibrate --size-mm.")
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    intrinsics, metadata = load_intrinsics(args.intrinsics)
    size_m = args.size_mm / 1000.0
    collected = _collect_marker_poses(
        intrinsics,
        metadata,
        args.dictionary,
        args.marker_id,
        size_m,
        args.samples,
        args.warmup,
        args.serial,
        args.name_substring,
    )
    averaged = average_transforms(collected.world_from_camera)
    angles = [rotation_angle_deg(averaged, item) for item in collected.world_from_camera]
    translations = np.stack([item.translation for item in collected.world_from_camera])
    table_z = _table_z_m(collected.frame, collected.corners, intrinsics, averaged)
    _save_detection(args.preview_dir, collected, size_m, intrinsics)
    save_transform(
        args.output,
        averaged,
        {
            "source_frame": "overhead",
            "destination_frame": "world",
            "world_origin": "marker center",
            "world_x": "printed +X arrow, on the table",
            "world_y": "printed +Y arrow, on the table",
            "world_z": "up, out of the marker toward the camera",
            "dictionary": args.dictionary,
            "marker_id": args.marker_id,
            "marker_size_m": size_m,
            "intrinsics": str(args.intrinsics),
            "samples": len(collected.world_from_camera),
            "reprojection_error_px": float(np.median(collected.errors_px)),
            "translation_std_m": np.std(translations, axis=0).tolist(),
            "rotation_jitter_deg": float(np.median(angles)),
            "table_z_m": table_z,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
    _print_pose(averaged, collected)
    if table_z is None:
        print("Not enough valid depth around the marker; skipped the table height check.")
    elif abs(table_z) > 0.015:
        print(f"Median world height of table points around the marker: {table_z * 1000:.1f} mm. With the camera looking roughly down, this should be close to 0.")
    else:
        print(f"Median world height of table points around the marker: {table_z * 1000:.1f} mm.")
    print(f"Wrote {args.output}")
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    saved, metadata = load_transform(args.extrinsic)
    intrinsics, camera_meta = load_intrinsics(args.intrinsics)
    collected = _collect_marker_poses(
        intrinsics,
        camera_meta,
        str(metadata["dictionary"]),
        int(metadata["marker_id"]),
        float(metadata["marker_size_m"]),
        args.samples,
        args.warmup,
        args.serial,
        args.name_substring,
    )
    live = average_transforms(collected.world_from_camera)
    distance_m = translation_distance_m(saved, live)
    angle_deg = rotation_angle_deg(saved, live)
    print(f"Compared with the saved extrinsic: translation {distance_m * 1000:.2f} mm, rotation {angle_deg:.3f}°")
    within = distance_m * 1000.0 <= args.trans_mm and angle_deg <= args.rot_deg
    if within:
        print(f"Within the thresholds of {args.trans_mm:.1f} mm and {args.rot_deg:.1f}°.")
        return 0
    print(f"Exceeds the thresholds of {args.trans_mm:.1f} mm and {args.rot_deg:.1f}°.")
    if not args.update:
        print("If the marker or camera was bumped, put it back and run check again. Once you are sure the new position is valid, add --update to overwrite the extrinsic.")
        return 1
    metadata = dict(metadata)
    metadata["saved_at"] = datetime.now().isoformat(timespec="seconds")
    metadata["updated_from_check"] = True
    save_transform(args.extrinsic, live, metadata)
    print(f"Overwrote {args.extrinsic} with this result")
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    saved, metadata = load_transform(args.extrinsic)
    intrinsics, camera_meta = load_intrinsics(args.intrinsics)
    dictionary = str(metadata["dictionary"])
    world_id = int(metadata["marker_id"])
    size_m = float(metadata["marker_size_m"])
    if args.verify_id == world_id:
        raise ValueError("The verification marker id must differ from the world marker id")
    measured = np.array([args.x_mm, args.y_mm, args.z_mm], dtype=np.float64) / 1000.0
    samples = []
    camera = _open_camera(intrinsics, camera_meta, args.serial, args.name_substring)
    with camera:
        _warmup(camera, args.warmup)
        attempts = 0
        last_frame = None
        while len(samples) < args.samples:
            attempts += 1
            if attempts > args.samples * 15:
                raise RuntimeError("Did not see the world marker and the verification marker together. Both markers must be fully visible in the overhead image.")
            frame = camera.read()
            last_frame = frame
            world_corners = detect_marker(frame.color_bgr, dictionary, world_id)
            verify_corners = detect_marker(frame.color_bgr, dictionary, args.verify_id)
            if world_corners is None or verify_corners is None:
                continue
            verify_pose = pose_from_corners(verify_corners, size_m, intrinsics)
            samples.append(saved.apply(verify_pose.translation))
    estimate = np.median(np.stack(samples), axis=0)
    error_m = float(np.linalg.norm(estimate - measured))
    _write_verify_preview(last_frame, dictionary, world_id, args.verify_id)
    print(
        "Verification marker center in world: "
        f"({estimate[0] * 1000:.1f}, {estimate[1] * 1000:.1f}, {estimate[2] * 1000:.1f}) mm"
    )
    print(
        "Measured: "
        f"({measured[0] * 1000:.1f}, {measured[1] * 1000:.1f}, {measured[2] * 1000:.1f}) mm, "
        f"error {error_m * 1000:.1f} mm"
    )
    if error_m * 1000.0 <= args.tolerance_mm:
        print(f"Error is within {args.tolerance_mm:.1f} mm.")
        return 0
    print(f"Error exceeds {args.tolerance_mm:.1f} mm. Check --size-mm, the measured center offset between the two markers, and that +X/+Y follow the printed arrows.")
    return 1


class _Collection:
    def __init__(self) -> None:
        self.world_from_camera: list[RigidTransform] = []
        self.errors_px: list[float] = []
        self.frame = None
        self.corners: np.ndarray | None = None
        self.camera_from_tag: RigidTransform | None = None


def _collect_marker_poses(
    intrinsics: PinholeIntrinsics,
    metadata: dict,
    dictionary: str,
    marker_id: int,
    size_m: float,
    samples: int,
    warmup: int,
    serial: str | None,
    name_substring: str,
) -> _Collection:
    if samples < 1:
        raise ValueError("--samples must be at least 1")
    collected = _Collection()
    camera = _open_camera(intrinsics, metadata, serial, name_substring)
    limit = samples * 15
    attempts = 0
    with camera:
        _warmup(camera, warmup)
        while len(collected.world_from_camera) < samples:
            attempts += 1
            if attempts > limit:
                if collected.frame is not None:
                    DEFAULT_OUTPUT.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(DEFAULT_OUTPUT / "detection_failed.png"), collected.frame.color_bgr)
                raise RuntimeError(
                    f"Saw marker {marker_id} only {len(collected.world_from_camera)} times in {limit} frames. "
                    "Make sure the marker lies flat, faces up, and the whole black square is visible in the overhead image."
                )
            frame = camera.read()
            collected.frame = frame
            corners = detect_marker(frame.color_bgr, dictionary, marker_id)
            if corners is None:
                _show(frame.color_bgr, None, None, size_m, intrinsics)
                continue
            camera_from_tag = pose_from_corners(corners, size_m, intrinsics)
            error_px = reprojection_error_px(corners, size_m, intrinsics, camera_from_tag)
            if error_px > 2.0:
                continue
            collected.world_from_camera.append(camera_from_tag.inverse())
            collected.errors_px.append(error_px)
            collected.corners = corners
            collected.camera_from_tag = camera_from_tag
            _show(frame.color_bgr, corners, camera_from_tag, size_m, intrinsics)
    _close_window()
    return collected


def _table_z_m(frame, corners: np.ndarray | None, intrinsics: PinholeIntrinsics, world_from_camera: RigidTransform):
    if frame is None or corners is None:
        return None
    mask = np.zeros(frame.depth_m.shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(corners).astype(np.int32), 1)
    ring = cv2.dilate(mask, np.ones((31, 31), np.uint8))
    usable = (ring > 0) & (mask == 0) & (frame.depth_m > 0)
    if int(np.count_nonzero(usable)) < 30:
        return None
    rows, cols = np.nonzero(usable)
    if rows.size > 4000:
        choice = np.linspace(0, rows.size - 1, 4000).astype(int)
        rows, cols = rows[choice], cols[choice]
    points = intrinsics.backproject(cols, rows, frame.depth_m[rows, cols])
    world = world_from_camera.apply(points)
    return float(np.median(world[:, 2]))


def _print_pose(averaged: RigidTransform, collected: _Collection) -> None:
    camera_in_world = averaged.translation
    optical_axis = averaged.rotation[:, 2]
    print(
        "Camera center in the table frame: "
        f"({camera_in_world[0]:.3f}, {camera_in_world[1]:.3f}, {camera_in_world[2]:.3f}) m"
    )
    print(
        "Camera optical axis in the table frame: "
        f"({optical_axis[0]:.3f}, {optical_axis[1]:.3f}, {optical_axis[2]:.3f})"
        ". With the camera looking down, this is close to (0, 0, -1)."
    )
    print(f"Median reprojection error: {float(np.median(collected.errors_px)):.3f} px")
    translations = np.stack([item.translation for item in collected.world_from_camera])
    print(f"Translation jitter std: {np.std(translations, axis=0) * 1000} mm")


def _save_detection(directory: Path, collected: _Collection, size_m: float, intrinsics: PinholeIntrinsics) -> None:
    if collected.frame is None or collected.camera_from_tag is None:
        return
    directory.mkdir(parents=True, exist_ok=True)
    image = _draw_axes(collected.frame.color_bgr, collected.camera_from_tag, size_m, intrinsics)
    cv2.imwrite(str(directory / "detection.png"), image)


def _write_verify_preview(frame, dictionary: str, world_id: int, verify_id: int) -> None:
    if frame is None:
        return
    image = frame.color_bgr.copy()
    for marker_id, color in ((world_id, (0, 255, 0)), (verify_id, (0, 255, 255))):
        corners = detect_marker(image, dictionary, marker_id)
        if corners is None:
            continue
        cv2.polylines(image, [np.round(corners).astype(np.int32)], True, color, 2)
    DEFAULT_OUTPUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(DEFAULT_OUTPUT / "verify.png"), image)


def _write_sheet(path: Path, dictionary: str, marker_id: int, size_mm: float, title: str, draw_axes: bool) -> None:
    side = max(int(round(size_mm / 25.4 * 300)), 200)
    marker = render_marker(dictionary, marker_id, side)
    margin = max(side // 5, 80)
    canvas = np.full((side + margin * 2 + 80, side + margin * 2), 255, np.uint8)
    x0, y0 = margin, margin + 60
    canvas[y0 : y0 + side, x0 : x0 + side] = marker
    color = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    cv2.putText(color, title, (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    caption = f"design {size_mm:.1f} mm at 300 DPI, 100 percent. Measure the black square."
    cv2.putText(color, caption, (40, color.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    if draw_axes:
        center_x = x0 + side // 2
        center_y = y0 + side // 2
        cv2.arrowedLine(color, (x0 + side + 10, center_y), (color.shape[1] - 30, center_y), (0, 0, 255), 4)
        cv2.putText(color, "+X", (color.shape[1] - 90, center_y - 12), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.arrowedLine(color, (center_x, y0 - 10), (center_x, 70), (0, 180, 0), 4)
        cv2.putText(color, "+Y", (center_x + 12, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 180, 0), 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), color)


def _draw_axes(image: np.ndarray, camera_from_tag: RigidTransform, size_m: float, intrinsics: PinholeIntrinsics):
    drawn = image.copy()
    cv2.drawFrameAxes(
        drawn,
        intrinsics.camera_matrix,
        intrinsics.distortion_coeffs,
        cv2.Rodrigues(camera_from_tag.rotation)[0],
        camera_from_tag.translation.reshape(3, 1),
        size_m * 0.6,
        3,
    )
    return drawn


def _show(image, corners, camera_from_tag, size_m: float, intrinsics: PinholeIntrinsics) -> None:
    import os

    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return
    drawn = image.copy()
    if corners is not None and camera_from_tag is not None:
        drawn = _draw_axes(drawn, camera_from_tag, size_m, intrinsics)
    cv2.imshow("step03 overhead marker", drawn)
    if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
        _close_window()
        raise RuntimeError("Aborted; the extrinsic file was not written.")


def _close_window() -> None:
    import os

    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        cv2.destroyAllWindows()


def _open_camera(intrinsics: PinholeIntrinsics, metadata: dict, serial: str | None, name_substring: str):
    from soarm_aimbot.cameras.realsense import RealSenseRgbd

    return RealSenseRgbd(
        width=intrinsics.width,
        height=intrinsics.height,
        fps=int(metadata.get("fps", 30)),
        serial=serial or metadata.get("serial"),
        name_substring=name_substring,
    )


def _warmup(camera, count: int) -> None:
    for _ in range(max(count, 0)):
        camera.read()


if __name__ == "__main__":
    raise SystemExit(main())
