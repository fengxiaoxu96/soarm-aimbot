"""Step 5: eye-in-hand calibration of the wrist RGB camera to gripper_link.

Run from the repository root with the virtual environment activated:

    aimbot-step05 self-check
    aimbot-step05 joints
    aimbot-step05 print-marker --size-mm 40
    aimbot-step05 capture --check
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

from soarm_aimbot.cameras.base import CameraError
from soarm_aimbot.cameras.rgb import RgbCamera, choose_device, list_v4l2_devices
from soarm_aimbot.geometry.handeye import solve_eye_in_hand
from soarm_aimbot.geometry.marker import (
    detect_marker,
    pose_from_corners_rgb,
    render_marker,
    reprojection_error_rgb,
)
from soarm_aimbot.geometry.rgb_model import load_rgb_intrinsics
from soarm_aimbot.geometry.transforms import (
    RigidTransform,
    average_transforms,
    rotation_angle_deg,
    translation_distance_m,
)
from soarm_aimbot.geometry.urdf_fk import (
    ARM_JOINTS,
    UrdfModel,
    forward_kinematics,
    load_urdf,
    reading_to_urdf_radians,
)
from soarm_aimbot.robots.sts3215 import tick_to_radians

DEFAULT_URDF = Path("assets/so101/so101_new_calib.urdf")
DEFAULT_INTRINSICS = Path("calib/wrist_intrinsics.yaml")
DEFAULT_OUTPUT = Path("calib/T_gripper_wrist.yaml")
DEFAULT_SAMPLES = Path("output/step05/samples.yaml")
DEFAULT_MARKER_DIR = Path("output/step05")
DEFAULT_SERVO_MAP = Path("calib/servo_map.yaml")
RESERVED_MARKER_IDS = {0: "the table world origin", 2: "the temporary gripper tag of Step 4"}
TAG_READY_PX = 1.5


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except CameraError as exc:
        print(f"Camera error: {exc}")
        return 1
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"Could not complete Step 5: {exc}")
        return 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aimbot-step05",
        description="Step 5: calibrate the wrist camera pose relative to the gripper from a table AprilTag and URDF forward kinematics.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("self-check", help="check the forward-kinematics zero and the eye-in-hand solver without opening a camera")
    check.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    check.set_defaults(func=_cmd_self_check)

    joints = sub.add_parser("joints", help="print the joint chain, the limits, and the gripper position at all-zero joints")
    joints.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    joints.add_argument("--root-link", default="base_link")
    joints.add_argument("--tip-link", default="gripper_link")
    joints.set_defaults(func=_cmd_joints)

    fk = sub.add_parser("fk", help="run forward kinematics for five joint angles")
    _add_model_args(fk)
    fk.set_defaults(func=_cmd_fk)

    zero = sub.add_parser("zero", help="convert readings between two mechanical stops into URDF radians")
    zero.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    zero.add_argument("--stops", type=Path, required=True)
    zero.set_defaults(func=_cmd_zero)

    printed = sub.add_parser("print-marker", help="render the AprilTag to fix on the table; its id must differ from the origin and gripper tags")
    printed.add_argument("--dictionary", default="DICT_APRILTAG_36h11")
    printed.add_argument("--marker-id", type=int, default=3)
    printed.add_argument("--size-mm", type=float, default=40.0)
    printed.add_argument("--output-dir", type=Path, default=DEFAULT_MARKER_DIR)
    printed.set_defaults(func=_cmd_print)

    capture = sub.add_parser("capture", help="move the arm by hand, preview the tag in the wrist camera, save poses with a key, then solve")
    _add_model_args(capture)
    _add_tag_args(capture, require_size=False)
    capture.add_argument("--samples-file", type=Path, default=DEFAULT_SAMPLES)
    capture.add_argument("--frames", type=int, default=15)
    capture.add_argument("--device", type=int, default=None)
    capture.add_argument("--focus", type=float, default=None)
    capture.add_argument("--reset", action="store_true")
    capture.add_argument("--servo-map", type=Path, default=DEFAULT_SERVO_MAP)
    capture.add_argument("--port", default=None)
    capture.add_argument("--baud", type=int, default=1000000)
    capture.add_argument("--move", action="store_true", help="open the serial port and the wrist camera. Torque stays off; you move the arm by hand")
    capture.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    capture.add_argument("--holdout", type=int, default=4)
    capture.add_argument("--agree-mm", type=float, default=10.0)
    capture.add_argument("--agree-deg", type=float, default=3.0)
    capture.add_argument("--verify-mm", type=float, default=5.0)
    capture.add_argument("--verify-px", type=float, default=1.0)
    capture.add_argument("--check", action="store_true", help="check that the preview keeps running on failures, without opening the serial port or camera")
    capture.set_defaults(func=_cmd_capture)

    solve = sub.add_parser("solve", help="solve T_gripper_wrist")
    solve.add_argument("--samples-file", type=Path, default=DEFAULT_SAMPLES)
    solve.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    solve.add_argument("--holdout", type=int, default=4)
    solve.add_argument("--agree-mm", type=float, default=10.0)
    solve.add_argument("--agree-deg", type=float, default=3.0)
    solve.add_argument("--verify-mm", type=float, default=5.0)
    solve.add_argument("--verify-px", type=float, default=1.0)
    solve.set_defaults(func=_cmd_solve)

    verify = sub.add_parser("verify", help="move to one more pose and compare the seen tag with the forward-kinematics prediction")
    _add_model_args(verify)
    _add_tag_args(verify)
    verify.add_argument("--calibration", type=Path, default=DEFAULT_OUTPUT)
    verify.add_argument("--samples-file", type=Path, default=DEFAULT_SAMPLES)
    verify.add_argument("--device", type=int, default=None)
    verify.add_argument("--focus", type=float, default=None)
    verify.add_argument("--frames", type=int, default=15)
    verify.add_argument("--tolerance-mm", type=float, default=5.0)
    verify.add_argument("--tolerance-px", type=float, default=1.0)
    verify.set_defaults(func=_cmd_verify)

    pose = sub.add_parser("pose", help="compute the wrist camera pose in base_link from joint angles")
    _add_model_args(pose)
    pose.add_argument("--calibration", type=Path, default=DEFAULT_OUTPUT)
    pose.set_defaults(func=_cmd_pose)
    return parser


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--root-link", default="base_link")
    parser.add_argument("--tip-link", default="gripper_link")
    parser.add_argument("--joints", default=None, help="five joint angles in radians")
    parser.add_argument("--joints-file", type=Path, default=None)


def _add_tag_args(parser: argparse.ArgumentParser, *, require_size: bool = True) -> None:
    parser.add_argument("--intrinsics", type=Path, default=DEFAULT_INTRINSICS)
    parser.add_argument("--dictionary", default="DICT_APRILTAG_36h11")
    parser.add_argument("--marker-id", type=int, default=3)
    parser.add_argument("--size-mm", type=float, required=require_size)
    parser.add_argument("--warmup", type=int, default=10)


def _cmd_self_check(args: argparse.Namespace) -> int:
    model = load_urdf(args.urdf)
    pose = forward_kinematics(model, {name: 0.0 for name in ARM_JOINTS})
    if abs(pose.translation[0] - 0.293234) > 1e-3 or abs(pose.translation[2] - 0.234370) > 1e-3:
        print(f"Self-check failed: the gripper origin at all-zero joints is {pose.translation}, which does not match so101_new_calib.")
        return 1
    generator = np.random.default_rng(1)
    gripper_from_camera = RigidTransform.from_rotation_translation(
        cv2.Rodrigues(np.array([0.2, -0.4, 0.1]))[0],
        [0.02, -0.03, 0.05],
    )
    board = RigidTransform.from_rotation_translation(np.eye(3), [0.25, 0.05, 0.0])
    gripper_poses = []
    tag_poses = []
    for _ in range(8):
        axis = generator.normal(size=3)
        axis = axis / np.linalg.norm(axis)
        rotation = cv2.Rodrigues(axis * float(generator.uniform(0.4, 1.2)))[0]
        gripper = RigidTransform.from_rotation_translation(rotation, generator.normal(size=3) * 0.2 + np.array([0.3, 0.0, 0.2]))
        camera_from_tag = RigidTransform(
            np.linalg.inv(gripper_from_camera.matrix) @ np.linalg.inv(gripper.matrix) @ board.matrix
        )
        gripper_poses.append(gripper)
        tag_poses.append(camera_from_tag)
    park = solve_eye_in_hand(gripper_poses, tag_poses, method="park")
    andreff = solve_eye_in_hand(gripper_poses, tag_poses, method="andreff")
    distance = translation_distance_m(park.gripper_from_camera, andreff.gripper_from_camera)
    angle = rotation_angle_deg(park.gripper_from_camera, andreff.gripper_from_camera)
    error = translation_distance_m(park.gripper_from_camera, gripper_from_camera)
    if distance > 1e-6 or angle > 1e-3 or error > 1e-6:
        print(f"Self-check failed: the two solvers differ by {distance * 1000:.3f} mm and {angle:.3f}°, translation error {error:.3e} m.")
        return 1
    print(
        "Self-check passed: at all-zero joints gripper_link is at about "
        f"({pose.translation[0] * 1000:.1f}, {pose.translation[1] * 1000:.1f}, {pose.translation[2] * 1000:.1f}) mm in base_link. "
        "The eye-in-hand solver recovers a known camera mount."
    )
    print("The URDF has no wrist-camera link. T_gripper_wrist is the calibrated fixed mount and does not change with the joint angles.")
    print("q=0 is the zero of so101_new_calib.urdf. The table board uses its own id, 3 by default.")
    return 0


def _cmd_joints(args: argparse.Namespace) -> int:
    model = load_urdf(args.urdf)
    chain = model.chain(args.root_link, args.tip_link)
    print(f"Chain: {args.root_link} -> {args.tip_link}")
    for joint in chain:
        if joint.joint_type == "fixed":
            print(f"  {joint.name}  fixed  {joint.parent} -> {joint.child}")
            continue
        lower = joint.lower if joint.lower is not None else float("nan")
        upper = joint.upper if joint.upper is not None else float("nan")
        print(
            f"  {joint.name}  {joint.parent} -> {joint.child}  "
            f"limits [{lower:.5f}, {upper:.5f}] rad "
            f"({np.rad2deg(lower):.1f}°, {np.rad2deg(upper):.1f}°)"
        )
    pose = forward_kinematics(model, {name: 0.0 for name in ARM_JOINTS}, root_link=args.root_link, tip_link=args.tip_link)
    _print_pose("All-zero gripper_link", pose, "the root link")
    print("The wrist camera is not on this chain. After calibrating T_gripper_wrist, the pose command appends it to forward kinematics.")
    return 0


def _cmd_fk(args: argparse.Namespace) -> int:
    model = load_urdf(args.urdf)
    pose = forward_kinematics(model, _joint_map(args, model), root_link=args.root_link, tip_link=args.tip_link)
    _print_pose(args.tip_link, pose, "the root link")
    return 0


def _cmd_zero(args: argparse.Namespace) -> int:
    model = load_urdf(args.urdf)
    payload = yaml.safe_load(Path(args.stops).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("The stops file must map joint names to stop_min_deg, stop_max_deg, and reading_deg")
    print("These radians treat the midpoint of the two stops as 0. URDF q=0 matches only if the stops are symmetric about the software zero.")
    for name in ARM_JOINTS:
        item = payload[name]
        angle = reading_to_urdf_radians(
            float(item["stop_min_deg"]), float(item["stop_max_deg"]), float(item["reading_deg"])
        )
        joint = model.joints[name]
        midpoint = 0.5 * (joint.lower + joint.upper)
        print(
            f"{name}: {angle:.5f} rad ({np.rad2deg(angle):.2f}°), "
            f"URDF limit midpoint {midpoint:.5f} rad ({np.rad2deg(midpoint):.2f}°)"
        )
        if abs(midpoint) > 0.02:
            print(f"  The URDF limit midpoint of {name} is not 0. Do not use the mechanical stop midpoint as q=0 for this joint.")
        if angle < joint.lower - 1e-3 or angle > joint.upper + 1e-3:
            print(f"  The converted value is outside the URDF limits of {name}.")
    return 0


def _cmd_print(args: argparse.Namespace) -> int:
    _reject_marker_id(args.marker_id)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    side = max(int(round(args.size_mm / 25.4 * 300)), 80)
    marker = render_marker(args.dictionary, args.marker_id, side)
    margin = side // 3
    canvas = np.full((side + 2 * margin, side + 2 * margin), 255, dtype=np.uint8)
    canvas[margin : margin + side, margin : margin + side] = marker
    color = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    axis_origin = (margin + side + 12, margin + side - 12)
    cv2.arrowedLine(color, axis_origin, (axis_origin[0] + margin // 2, axis_origin[1]), (0, 0, 180), 2)
    cv2.arrowedLine(color, axis_origin, (axis_origin[0], axis_origin[1] - margin // 2), (0, 140, 0), 2)
    path = args.output_dir / "wrist_marker.png"
    cv2.imwrite(str(path), color)
    print(f"Wrote {path}")
    print("Print at 100% scale. Measure the outer edge of the black square with calipers and pass it as --size-mm. The red arrow is the tag +X, the green one is +Y.")
    print("Fix this tag on the table where the wrist camera can see the whole square from many directions. Do not move the board during calibration.")
    print("It does not define the world frame; the world origin is still id 0 on the table. Remove the Step 4 gripper tag before starting this step.")
    return 0


def _cmd_capture(args: argparse.Namespace) -> int:
    if args.check:
        return _capture_check()
    _reject_marker_id(args.marker_id)
    if args.size_mm is None:
        raise ValueError("Capturing the tag needs --size-mm.")
    servo_map = _load_servo_map(args.servo_map)
    if not args.move:
        print("Step 5 capture is hand-guided. Add --move --port to open the serial port and the wrist camera.")
        print("Torque stays off. The window and terminal show the current joint angles and the tag. Press s to save, q to finish and solve.")
        return 0
    if not args.port:
        raise ValueError("Hand-guided capture needs --port.")
    model = load_urdf(args.urdf)
    camera_model, metadata = load_rgb_intrinsics(args.intrinsics)
    device = choose_device(
        list_v4l2_devices(), args.device if args.device is not None else metadata.get("device_index")
    )
    from soarm_aimbot.robots.sts3215 import Sts3215Bus

    bus = Sts3215Bus(args.port, args.baud)
    saved = 0
    try:
        _set_arm_torque(bus, servo_map, False)
        print("Torque is off on the five arm joints. Move the arm by hand to a pose where the whole table tag is visible and the reprojection is at most 1.5 px.")
        print("Press s to save the current pose, q to stop capturing and solve with the saved samples.")
        with RgbCamera(device, camera_model.width, camera_model.height) as camera:
            focus = args.focus if args.focus is not None else metadata.get("focus")
            if focus is not None:
                camera.set_focus(float(focus))
            for _ in range(max(args.warmup, 0)):
                camera.read()
            try:
                saved = _run_manual_capture(args, bus, camera, servo_map, model, camera_model)
            except KeyboardInterrupt:
                print()
                print("Capture interrupted. Saved samples are kept.")
    finally:
        cv2.destroyAllWindows()
        try:
            _set_arm_torque(bus, servo_map, False)
        finally:
            bus.close()
    print(f"Capture finished with {saved} poses. Torque is off.")
    return _solve_after_capture(args)


def _cmd_solve(args: argparse.Namespace) -> int:
    store = _load_samples(args.samples_file, reset=False)
    samples = store.get("samples", [])
    if args.holdout < 0 or args.holdout >= len(samples):
        raise ValueError("The number of held-out poses must be smaller than the number captured")
    used = samples if args.holdout == 0 else samples[: -args.holdout]
    held = [] if args.holdout == 0 else samples[-args.holdout :]
    if len(used) < 4:
        raise ValueError(f"Only {len(used)} poses take part in the solve. At least 4 are needed; capture about 20 in total and hold out 4.")
    urdf = load_urdf(store["urdf"])
    gripper_poses, tag_poses = _poses_from_samples(urdf, store, used)
    park = solve_eye_in_hand(gripper_poses, tag_poses, method="park")
    andreff = solve_eye_in_hand(gripper_poses, tag_poses, method="andreff")
    distance = translation_distance_m(park.gripper_from_camera, andreff.gripper_from_camera)
    angle = rotation_angle_deg(park.gripper_from_camera, andreff.gripper_from_camera)
    print(f"Park and Andreff differ by {distance * 1000:.1f} mm and {angle:.2f}°, using {park.pair_count} relative motion pairs.")
    if distance > args.agree_mm / 1000.0 or angle > args.agree_deg:
        print("The two solvers disagree too much. Add poses with larger rotation differences and solve again. Nothing was written.")
        return 1
    camera_model = _load_optional_model(store.get("intrinsics"))
    park_mm, park_px = _holdout_metrics(urdf, store, used, held, park.gripper_from_camera, camera_model)
    andreff_mm, andreff_px = _holdout_metrics(urdf, store, used, held, andreff.gripper_from_camera, camera_model)
    chosen, holdout_mm, holdout_px = _choose_solution(
        park, andreff, park_mm, park_px, andreff_mm, andreff_px
    )
    print(f"Using {chosen.method}.")
    payload = _calibration_payload(store, chosen, distance, angle, len(used), holdout_mm, holdout_px)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    _print_mount(chosen.gripper_from_camera)
    print(f"Wrote {destination}")
    failed = False
    if holdout_mm and max(holdout_mm) * 1000.0 > args.verify_mm:
        print(f"Held-out translation error exceeds {args.verify_mm:.0f} mm. The file was written, but this check did not pass.")
        failed = True
    if holdout_px and max(holdout_px) > args.verify_px:
        print(f"Held-out reprojection exceeds {args.verify_px:.1f} px. The file was written, but this check did not pass.")
        failed = True
    return 1 if failed else 0


def _cmd_verify(args: argparse.Namespace) -> int:
    calibration = yaml.safe_load(Path(args.calibration).read_text(encoding="utf-8"))
    gripper_from_camera = RigidTransform(np.array(calibration["T_gripper_wrist"], dtype=np.float64))
    store = _load_samples(args.samples_file, reset=False)
    urdf = load_urdf(args.urdf)
    joints = _joint_map(args, urdf)
    camera_model, metadata = load_rgb_intrinsics(args.intrinsics)
    observation = _average_tag_pose(args, camera_model, metadata)
    stored = store.get("samples", [])
    used_count = int(calibration.get("sample_count", len(stored)))
    used = stored[:used_count]
    if len(used) < 4:
        raise ValueError("Verification needs the solved poses in the samples file to estimate where the board is in the base frame.")
    board = _board_in_base(urdf, store, used, gripper_from_camera)
    gripper = forward_kinematics(urdf, joints, root_link=args.root_link, tip_link=args.tip_link)
    predicted = RigidTransform(np.linalg.inv(gripper_from_camera.matrix) @ np.linalg.inv(gripper.matrix) @ board.matrix)
    distance = translation_distance_m(predicted, observation.transform)
    pixel_error = reprojection_error_rgb(
        observation.corners, args.size_mm / 1000.0, camera_model, predicted
    )
    angle = rotation_angle_deg(predicted, observation.transform)
    print(
        f"Tag origin: forward kinematics and vision differ by {distance * 1000:.1f} mm and {angle:.2f}°, "
        f"corner reprojection {pixel_error:.3f} px."
    )
    if distance > args.tolerance_mm / 1000.0 or pixel_error > args.tolerance_px:
        print("The error is above the threshold. Check that the board was not bumped and the joint zeros still match the URDF, then recalibrate with larger rotation differences.")
        return 1
    print("Translation error and reprojection for this pose are both within the thresholds.")
    return 0


def _cmd_pose(args: argparse.Namespace) -> int:
    calibration = yaml.safe_load(Path(args.calibration).read_text(encoding="utf-8"))
    gripper_from_camera = RigidTransform(np.array(calibration["T_gripper_wrist"], dtype=np.float64))
    model = load_urdf(args.urdf)
    joints = _joint_map(args, model)
    gripper = forward_kinematics(model, joints, root_link=args.root_link, tip_link=args.tip_link)
    camera = RigidTransform(gripper.matrix @ gripper_from_camera.matrix)
    _print_pose("Wrist camera", camera, "base_link")
    print(
        "Camera origin in gripper_link: "
        f"({gripper_from_camera.translation[0] * 1000:.1f}, "
        f"{gripper_from_camera.translation[1] * 1000:.1f}, "
        f"{gripper_from_camera.translation[2] * 1000:.1f}) mm."
    )
    return 0


def _capture_check() -> int:
    from soarm_aimbot.geometry.rgb_model import RgbCameraModel

    blank = np.zeros((48, 64, 3), dtype=np.uint8)
    camera_model = RgbCameraModel(64, 48, 50.0, 50.0, 32.0, 24.0, (0.0, 0.0, 0.0, 0.0, 0.0))
    view_args = argparse.Namespace(dictionary="DICT_APRILTAG_36h11", marker_id=3, size_mm=40.0, frames=4)
    if _inspect_tag(blank, view_args, camera_model) is not None:
        print("Self-check failed: a blank image should not yield a tag.")
        return 1

    class Camera:
        def read(self):
            return blank

    if _average_tag_pose_from_camera(Camera(), view_args, camera_model) is not None:
        print("Self-check failed: a blank sequence should not yield an average pose.")
        return 1
    print("Self-check passed: when the tag is not visible, the preview and the multi-frame average keep going without stopping the capture. Press s to save, q to finish and solve.")
    return 0


def _run_manual_capture(args, bus, camera, servo_map, model, camera_model, wait_key=None, show=None) -> int:
    wait_key = cv2.waitKey if wait_key is None else wait_key
    show = cv2.imshow if show is None else show
    saved = _sample_count(args.samples_file) if not args.reset else 0
    while True:
        frame = camera.read()
        measured = _try_read_arm_radians(bus, servo_map, model)
        joints_text = (
            ",".join(f"{value:.3f}" for value in measured) if measured is not None else "out of limits or read failed"
        )
        view = _inspect_tag(frame, args, camera_model)
        show("step05 capture", _draw_capture_preview(frame, view, joints_text, saved))
        key = int(wait_key(1)) & 0xFF
        tag_text = "no tag" if view is None else f"reprojection {view.error:.2f} px"
        if view is not None and view.ready:
            tag_text += ", ready to save"
        print(f"\rjoints {joints_text}  {tag_text}  saved {saved}     ", end="", flush=True)
        if key in (ord("q"), 27):
            print()
            return saved
        if key not in (ord("s"), ord(" ")):
            continue
        print()
        if measured is None:
            print("The current joint angles are not usable; not saved. Keep moving the arm.")
            continue
        if view is None or not view.ready:
            print(f"The current view is not yet at reprojection <= {TAG_READY_PX:.1f} px; not saved. Keep moving the arm.")
            continue
        _hold_ticks(bus, servo_map, _read_ticks(bus, servo_map))
        observation = _average_tag_pose_from_camera(camera, args, camera_model)
        _set_arm_torque(bus, servo_map, False)
        if observation is None:
            print("Not enough valid frames this time; nothing was written. Move to the next pose.")
            continue
        held = _try_read_arm_radians(bus, servo_map, model)
        values = held if held is not None else measured
        joints = {name: float(value) for name, value in zip(ARM_JOINTS, values)}
        _store_sample(args, joints, observation)
        args.reset = False
        saved = _sample_count(args.samples_file)
        print(f"Poses collected: {saved}. Torque released; keep moving the arm.")


def _solve_after_capture(args: argparse.Namespace) -> int:
    samples = _load_samples(args.samples_file, reset=False).get("samples", [])
    if len(samples) < 4:
        print(f"{len(samples)} poses saved. Solving needs at least 4, so no solve this time.")
        return 0
    if args.holdout >= len(samples) or len(samples) - args.holdout < 4:
        print(f"Holding out {args.holdout} would leave fewer than 4 poses for the solve; using all poses instead.")
        args.holdout = 0
    print("Solving with the saved samples.")
    return _cmd_solve(args)


class _TagView:
    def __init__(self, corners: np.ndarray, pose: RigidTransform, error: float) -> None:
        self.corners = corners
        self.pose = pose
        self.error = float(error)
        self.ready = self.error <= TAG_READY_PX


def _inspect_tag(color_bgr, args, camera_model) -> _TagView | None:
    corners = detect_marker(color_bgr, args.dictionary, args.marker_id)
    if corners is None:
        return None
    try:
        pose = pose_from_corners_rgb(corners, args.size_mm / 1000.0, camera_model)
    except RuntimeError:
        return None
    error = reprojection_error_rgb(corners, args.size_mm / 1000.0, camera_model, pose)
    return _TagView(corners, pose, error)


def _draw_capture_preview(color_bgr, view, joints_text: str, saved: int) -> np.ndarray:
    image = np.ascontiguousarray(color_bgr.copy())
    if view is not None:
        points = np.round(view.corners).astype(np.int32).reshape(-1, 1, 2)
        color = (0, 200, 0) if view.ready else (0, 180, 220)
        cv2.polylines(image, [points], True, color, 2)
    lines = [
        f"joints {joints_text}",
        "tag ready" if view is not None and view.ready else "tag wait",
        f"samples {saved}   s=save  q=solve",
    ]
    if view is not None:
        lines.insert(1, f"reproj {view.error:.2f} px")
    for index, line in enumerate(lines):
        cv2.putText(image, line, (12, 28 + 24 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return image


def _store_sample(args, joints: dict[str, float], observation) -> None:
    store = _load_samples(args.samples_file, reset=args.reset)
    store.update(
        {
            "urdf": str(args.urdf),
            "root_link": args.root_link,
            "tip_link": args.tip_link,
            "intrinsics": str(args.intrinsics),
            "dictionary": args.dictionary,
            "marker_id": args.marker_id,
            "marker_size_m": args.size_mm / 1000.0,
        }
    )
    store.setdefault("samples", []).append(
        {
            "joints_rad": [joints[name] for name in ARM_JOINTS],
            "T_wrist_tag": observation.transform.matrix.tolist(),
            "corners_px": observation.corners.tolist(),
            "reprojection_px": float(observation.reprojection_px),
        }
    )
    path = Path(args.samples_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(store, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"Recorded pose {len(store['samples'])}, reprojection {observation.reprojection_px:.3f} px, wrote {path}")


def _load_servo_map(path: Path) -> dict:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "joints" not in payload:
        raise ValueError(f"{path} has no joints. Finish servo-zero in Step 4.1 first.")
    for name in ARM_JOINTS:
        item = payload["joints"][name]
        if int(item["sign"]) not in (-1, 1):
            raise ValueError(f"The sign of {name} must be 1 or -1")
    return payload


def _radians_from_ticks(servo_map: dict, ticks: dict[str, int]) -> np.ndarray:
    per_rev = int(servo_map.get("ticks_per_revolution", 4096))
    return np.array(
        [
            tick_to_radians(
                ticks[name],
                int(servo_map["joints"][name]["tick_at_urdf_zero"]),
                int(servo_map["joints"][name]["sign"]),
                per_rev,
            )
            for name in ARM_JOINTS
        ]
    )


def _read_ticks(bus, servo_map: dict) -> dict[str, int]:
    return {name: int(bus.read_position(int(servo_map["joints"][name]["id"]))) for name in ARM_JOINTS}


def _read_arm_radians(bus, servo_map: dict, model: UrdfModel) -> np.ndarray:
    joints = _radians_from_ticks(servo_map, _read_ticks(bus, servo_map))
    _require_joints(model, joints)
    return joints


def _try_read_arm_radians(bus, servo_map: dict, model: UrdfModel) -> np.ndarray | None:
    try:
        return _read_arm_radians(bus, servo_map, model)
    except ValueError:
        return None


def _require_joints(model: UrdfModel, joints: np.ndarray) -> None:
    values = np.asarray(joints, dtype=np.float64).reshape(-1)
    if values.size != 5:
        raise ValueError("Each pose needs five joint angles")
    for name, angle in zip(ARM_JOINTS, values):
        joint = model.joints[name]
        if joint.lower is not None and not joint.lower - 1e-3 <= float(angle) <= joint.upper + 1e-3:
            raise ValueError(f"{name}={float(angle):.4f} is outside the URDF limits")


def _hold_ticks(bus, servo_map: dict, ticks: dict[str, int]) -> None:
    for name in ARM_JOINTS:
        bus.write_goal(int(servo_map["joints"][name]["id"]), int(ticks[name]))
    for name in ARM_JOINTS:
        bus.set_torque(int(servo_map["joints"][name]["id"]), True)


def _set_arm_torque(bus, servo_map: dict, enabled: bool) -> None:
    for item in servo_map["joints"].values():
        bus.set_torque(int(item["id"]), enabled)


def _sample_count(path: Path) -> int:
    return len(_load_samples(path, reset=False).get("samples", []))


def _average_tag_pose(args, camera_model, metadata: dict):
    device = choose_device(list_v4l2_devices(), args.device if args.device is not None else metadata.get("device_index"))
    with RgbCamera(device, camera_model.width, camera_model.height) as camera:
        focus = args.focus if args.focus is not None else metadata.get("focus")
        if focus is not None:
            camera.set_focus(float(focus))
        for _ in range(max(args.warmup, 0)):
            camera.read()
        observation = _average_tag_pose_from_camera(camera, args, camera_model)
    if observation is None:
        raise ValueError("The table tag was detected in too few frames. Check the id and the size, and that the wrist camera sees the whole square.")
    return observation


def _average_tag_pose_from_camera(camera, args, camera_model):
    found: list[tuple[RigidTransform, np.ndarray, float]] = []
    attempts = 0
    while len(found) < args.frames and attempts < args.frames * 15:
        attempts += 1
        frame = camera.read()
        view = _inspect_tag(frame, args, camera_model)
        if view is None or not view.ready:
            continue
        found.append((view.pose, view.corners, view.error))
    if len(found) < max(5, args.frames // 2):
        return None
    corners = np.median(np.stack([item[1] for item in found]), axis=0)
    return _Observation(
        average_transforms([item[0] for item in found]),
        corners.astype(np.float64),
        float(np.mean([item[2] for item in found])),
    )


class _Observation:
    def __init__(self, transform: RigidTransform, corners: np.ndarray, reprojection_px: float) -> None:
        self.transform = transform
        self.corners = corners
        self.reprojection_px = reprojection_px


def _poses_from_samples(model: UrdfModel, store: dict, samples: list[dict]):
    gripper_poses = []
    tag_poses = []
    for item in samples:
        joints = dict(zip(ARM_JOINTS, item["joints_rad"]))
        gripper_poses.append(
            forward_kinematics(model, joints, root_link=store["root_link"], tip_link=store["tip_link"])
        )
        tag_poses.append(RigidTransform(np.array(item["T_wrist_tag"], dtype=np.float64)))
    return gripper_poses, tag_poses


def _board_in_base(model, store, samples, gripper_from_camera: RigidTransform) -> RigidTransform:
    poses = []
    gripper_poses, tag_poses = _poses_from_samples(model, store, samples)
    for gripper, tag in zip(gripper_poses, tag_poses):
        poses.append(RigidTransform(gripper.matrix @ gripper_from_camera.matrix @ tag.matrix))
    return average_transforms(poses)


def _holdout_metrics(model, store, used, held, gripper_from_camera, camera_model):
    if not held:
        return [], []
    board = _board_in_base(model, store, used, gripper_from_camera)
    size_m = float(store["marker_size_m"])
    distances = []
    pixels = []
    for item in held:
        joints = dict(zip(ARM_JOINTS, item["joints_rad"]))
        gripper = forward_kinematics(model, joints, root_link=store["root_link"], tip_link=store["tip_link"])
        predicted = RigidTransform(
            np.linalg.inv(gripper_from_camera.matrix) @ np.linalg.inv(gripper.matrix) @ board.matrix
        )
        observed = RigidTransform(np.array(item["T_wrist_tag"], dtype=np.float64))
        distances.append(translation_distance_m(predicted, observed))
        if camera_model is not None and "corners_px" in item:
            pixels.append(
                reprojection_error_rgb(
                    np.array(item["corners_px"], dtype=np.float64),
                    size_m,
                    camera_model,
                    predicted,
                )
            )
    return distances, pixels


def _choose_solution(park, andreff, park_mm, park_px, andreff_mm, andreff_px):
    park_score = float(np.mean(park_px)) if park_px else float(np.mean(park_mm) if park_mm else 0.0)
    andreff_score = float(np.mean(andreff_px)) if andreff_px else float(np.mean(andreff_mm) if andreff_mm else 0.0)
    if andreff_px and park_px and andreff_score + 0.01 < park_score:
        return andreff, andreff_mm, andreff_px
    if not park_px and not andreff_px and andreff_mm and park_mm and andreff_score + 1e-6 < park_score:
        return andreff, andreff_mm, andreff_px
    return park, park_mm, park_px


def _calibration_payload(store, solution, distance, angle, sample_count, holdout_mm, holdout_px) -> dict:
    mount = solution.gripper_from_camera
    return {
        "root_link": store["root_link"],
        "tip_link": store["tip_link"],
        "urdf": store["urdf"],
        "intrinsics": store.get("intrinsics"),
        "dictionary": store.get("dictionary"),
        "marker_id": store.get("marker_id"),
        "marker_size_m": store.get("marker_size_m"),
        "method": solution.method,
        "checked_against": "andreff" if solution.method == "park" else "park",
        "solver_translation_disagreement_m": distance,
        "solver_rotation_disagreement_deg": angle,
        "pair_count": solution.pair_count,
        "sample_count": sample_count,
        "holdout_count": len(holdout_mm),
        "holdout_translation_error_m": holdout_mm,
        "holdout_reprojection_px": holdout_px,
        "T_gripper_wrist": mount.matrix.tolist(),
        "camera_origin_in_gripper_m": mount.translation.tolist(),
        "optical_axis_in_gripper": mount.rotation[:, 2].tolist(),
        "image_right_in_gripper": mount.rotation[:, 0].tolist(),
        "image_down_in_gripper": mount.rotation[:, 1].tolist(),
        "optical_axis_nearest": _nearest_gripper_axis(mount.rotation[:, 2]),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }


def _nearest_gripper_axis(direction: np.ndarray) -> str:
    labels = ("gripper +X", "gripper +Y", "gripper +Z")
    axis = int(np.argmax(np.abs(direction)))
    sign = "+" if direction[axis] >= 0 else "-"
    return f"gripper {sign}{labels[axis][-1]}"


def _print_mount(mount: RigidTransform) -> None:
    origin = mount.translation
    print(
        "Camera origin in gripper_link: "
        f"({origin[0] * 1000:.1f}, {origin[1] * 1000:.1f}, {origin[2] * 1000:.1f}) mm."
    )
    print(f"The optical axis (camera +Z) is closest to {_nearest_gripper_axis(mount.rotation[:, 2])}.")
    print(
        "Image right (camera +X) in the gripper frame: "
        f"({mount.rotation[0, 0]:.3f}, {mount.rotation[1, 0]:.3f}, {mount.rotation[2, 0]:.3f})."
    )


def _load_optional_model(path):
    if not path:
        return None
    model, _metadata = load_rgb_intrinsics(path)
    return model


def _reject_marker_id(marker_id: int) -> None:
    if marker_id in RESERVED_MARKER_IDS:
        raise ValueError(f"id {marker_id} is already used for {RESERVED_MARKER_IDS[marker_id]}. Use another id for the table board; the default is 3.")


def _joint_map(args, model: UrdfModel) -> dict[str, float]:
    if args.joints_file is not None:
        payload = yaml.safe_load(Path(args.joints_file).read_text(encoding="utf-8"))
        return {name: float(payload[name]) for name in ARM_JOINTS}
    if not args.joints:
        raise ValueError("Give five joint angles in radians with --joints or --joints-file.")
    values = [float(item) for item in args.joints.split(",")]
    if len(values) != len(ARM_JOINTS):
        raise ValueError("Five joint angles are needed: shoulder_pan,shoulder_lift,elbow_flex,wrist_flex,wrist_roll")
    mapping = dict(zip(ARM_JOINTS, values))
    for name, angle in mapping.items():
        joint = model.joints[name]
        if joint.lower is not None and not joint.lower - 1e-4 <= angle <= joint.upper + 1e-4:
            raise ValueError(f"{name}={angle:.4f} is outside the URDF limits [{joint.lower:.4f}, {joint.upper:.4f}]")
    return mapping


def _load_samples(path: Path, reset: bool) -> dict:
    if reset or not Path(path).is_file():
        return {"samples": []}
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {"samples": []}
    payload.setdefault("samples", [])
    return payload


def _print_pose(title: str, pose: RigidTransform, root_name: str) -> None:
    origin = pose.translation
    print(f"{title} origin in {root_name}: ({origin[0] * 1000:.1f}, {origin[1] * 1000:.1f}, {origin[2] * 1000:.1f}) mm")
    print(f"  +X: ({pose.rotation[0, 0]:.3f}, {pose.rotation[1, 0]:.3f}, {pose.rotation[2, 0]:.3f})")
    print(f"  +Y: ({pose.rotation[0, 1]:.3f}, {pose.rotation[1, 1]:.3f}, {pose.rotation[2, 1]:.3f})")
    print(f"  +Z: ({pose.rotation[0, 2]:.3f}, {pose.rotation[1, 2]:.3f}, {pose.rotation[2, 2]:.3f})")


if __name__ == "__main__":
    raise SystemExit(main())
