"""Step 4: eye-to-hand calibration from the overhead camera to the arm base.

Run from the repository root:

    aimbot-step04 self-check
    aimbot-step04 joints
    aimbot-step04 fk --joints 0,0,0,0,0
    aimbot-step04 print-marker --size-mm 40
    aimbot-step04 servo-zero --check
    aimbot-step04 move-zero --servo-map calib/servo_map.yaml
    aimbot-step04 servo-capture --check
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

from soarm_aimbot.cameras.base import CameraError
from soarm_aimbot.geometry.handeye import solve_eye_to_hand, solutions_agree
from soarm_aimbot.geometry.intrinsics import PinholeIntrinsics, load_intrinsics
from soarm_aimbot.geometry.marker import detect_marker, pose_from_corners, render_marker, reprojection_error_px
from soarm_aimbot.geometry.transforms import (
    RigidTransform,
    average_transforms,
    load_transform,
    rotation_angle_deg,
    translation_distance_m,
)
from soarm_aimbot.geometry.trajectory import interpolate, play, segment_speeds
from soarm_aimbot.geometry.urdf_fk import (
    ARM_JOINTS,
    UrdfModel,
    forward_kinematics,
    load_urdf,
    reading_to_urdf_radians,
)
from soarm_aimbot.robots.sts3215 import (
    TICKS_PER_REVOLUTION,
    parse_status,
    radians_to_tick,
    read_position_packet,
    sign_from_ticks,
    tick_to_radians,
    torque_packet,
)

DEFAULT_URDF = Path("assets/so101/so101_new_calib.urdf")
DEFAULT_INTRINSICS = Path("calib/overhead_intrinsics.yaml")
DEFAULT_EXTRINSICS = Path("calib/T_world_overhead.yaml")
DEFAULT_OUTPUT = Path("calib/T_base_world.yaml")
DEFAULT_SAMPLES = Path("output/step04/samples.yaml")
DEFAULT_MARKER_DIR = Path("output/step04")
DEFAULT_SERVO_MAP = Path("calib/servo_map.yaml")


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except CameraError as exc:
        print(f"Camera error: {exc}")
        return 1
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"Could not complete Step 4: {exc}")
        return 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aimbot-step04",
        description="Step 4: calibrate the overhead camera to the arm base using an AprilTag on the gripper and URDF forward kinematics.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("self-check", help="check the forward-kinematics zero pose and the eye-to-hand solver without opening the camera")
    check.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    check.set_defaults(func=_cmd_self_check)

    joints = sub.add_parser("joints", help="print the joint chain, limits, and the gripper position at the all-zero pose")
    joints.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    joints.add_argument("--root-link", default="base_link")
    joints.add_argument("--tip-link", default="gripper_link")
    joints.set_defaults(func=_cmd_joints)

    fk = sub.add_parser("fk", help="run forward kinematics for five joint angles")
    _add_model_args(fk)
    fk.set_defaults(func=_cmd_fk)

    zero = sub.add_parser("zero", help="convert readings between the two mechanical stops to URDF radians")
    zero.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    zero.add_argument("--stops", type=Path, required=True, help="yaml with stop_min_deg, stop_max_deg, reading_deg for each joint")
    zero.set_defaults(func=_cmd_zero)

    servo_zero = sub.add_parser("servo-zero", help="Step 4.1: with torque off, record each joint's zero tick and positive direction, then write the servo map")
    servo_zero.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    servo_zero.add_argument("--output", type=Path, default=DEFAULT_SERVO_MAP)
    servo_zero.add_argument("--ids", default="1,2,3,4,5", help="servo ids of the five arm joints, excluding the gripper")
    servo_zero.add_argument("--port", default=None)
    servo_zero.add_argument("--baud", type=int, default=1000000)
    servo_zero.add_argument("--move", action="store_true", help="open the serial port. You move the current joint by hand; joints already recorded hold at zero")
    servo_zero.add_argument("--check", action="store_true", help="check the tick conversion without opening the serial port")
    servo_zero.set_defaults(func=_cmd_servo_zero)

    move_zero = sub.add_parser("move-zero", help="drive the five arm joints to q=0 using the zero ticks recorded by servo-zero")
    move_zero.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    move_zero.add_argument("--servo-map", type=Path, default=DEFAULT_SERVO_MAP)
    move_zero.add_argument("--port", default=None)
    move_zero.add_argument("--baud", type=int, default=1000000)
    move_zero.add_argument("--speed-scale", type=float, default=0.05)
    move_zero.add_argument("--period-s", type=float, default=0.05)
    move_zero.add_argument("--arrive-rad", type=float, default=0.08, help="stop and report if a read-back angle differs from 0 by more than this")
    move_zero.add_argument("--move", action="store_true", help="enable torque and move to zero. Without it, only print the zero ticks")
    move_zero.add_argument("--release", action="store_true", help="disable torque after arriving")
    move_zero.set_defaults(func=_cmd_move_zero)

    printed = sub.add_parser("print-marker", help="generate the AprilTag for the gripper. Its id must differ from the table origin marker")
    printed.add_argument("--dictionary", default="DICT_APRILTAG_36h11")
    printed.add_argument("--marker-id", type=int, default=2)
    printed.add_argument("--size-mm", type=float, default=40.0, help="design side length. After printing, use the caliper measurement instead")
    printed.add_argument("--output-dir", type=Path, default=DEFAULT_MARKER_DIR)
    printed.set_defaults(func=_cmd_print)

    capture = sub.add_parser("capture", help="record the current five joint angles and the gripper marker pose seen by the overhead camera")
    _add_model_args(capture)
    _add_tag_args(capture)
    capture.add_argument("--samples-file", type=Path, default=DEFAULT_SAMPLES)
    capture.add_argument("--frames", type=int, default=20)
    capture.add_argument("--reset", action="store_true")
    capture.set_defaults(func=_cmd_capture)

    servo_capture = sub.add_parser("servo-capture", help="Step 4.2: move the arm by hand, preview the marker, save poses with a key press, then solve")
    _add_model_args(servo_capture)
    _add_tag_args(servo_capture, require_size=False)
    servo_capture.add_argument("--samples-file", type=Path, default=DEFAULT_SAMPLES)
    servo_capture.add_argument("--frames", type=int, default=20)
    servo_capture.add_argument("--reset", action="store_true")
    servo_capture.add_argument("--servo-map", type=Path, default=DEFAULT_SERVO_MAP)
    servo_capture.add_argument("--port", default=None)
    servo_capture.add_argument("--baud", type=int, default=1000000)
    servo_capture.add_argument("--move", action="store_true", help="open the serial port and the overhead camera. Torque stays off and you move the arm by hand")
    servo_capture.add_argument("--extrinsics", type=Path, default=DEFAULT_EXTRINSICS)
    servo_capture.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    servo_capture.add_argument("--holdout", type=int, default=3)
    servo_capture.add_argument("--agree-mm", type=float, default=20.0)
    servo_capture.add_argument("--agree-deg", type=float, default=5.0)
    servo_capture.add_argument("--verify-mm", type=float, default=15.0)
    servo_capture.add_argument("--check", action="store_true", help="check that the preview and failed detections do not abort capture, without opening the serial port or camera")
    servo_capture.set_defaults(func=_cmd_servo_capture)

    solve = sub.add_parser("solve", help="solve T_base_overhead and T_base_world from the captured poses")
    solve.add_argument("--samples-file", type=Path, default=DEFAULT_SAMPLES)
    solve.add_argument("--extrinsics", type=Path, default=DEFAULT_EXTRINSICS)
    solve.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    solve.add_argument("--holdout", type=int, default=3, help="number of final poses left out of the solve and used for verification")
    solve.add_argument("--agree-mm", type=float, default=20.0)
    solve.add_argument("--agree-deg", type=float, default=5.0)
    solve.add_argument("--verify-mm", type=float, default=15.0)
    solve.set_defaults(func=_cmd_solve)

    verify = sub.add_parser("verify", help="at a new pose, compare the marker position from vision with the forward-kinematics position")
    _add_model_args(verify)
    _add_tag_args(verify)
    verify.add_argument("--calibration", type=Path, default=DEFAULT_OUTPUT)
    verify.add_argument("--tolerance-mm", type=float, default=15.0)
    verify.add_argument("--frames", type=int, default=20)
    verify.set_defaults(func=_cmd_verify)

    mapped = sub.add_parser("map", help="map one table (world) coordinate into base_link")
    mapped.add_argument("--calibration", type=Path, default=DEFAULT_OUTPUT)
    mapped.add_argument("--xyz-mm", required=True, help="world coordinates x,y,z in mm")
    mapped.set_defaults(func=_cmd_map)
    return parser


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--root-link", default="base_link")
    parser.add_argument("--tip-link", default="gripper_link")
    parser.add_argument("--joints", default=None, help="five joint angles in radians, in the order printed by the joints command")
    parser.add_argument("--joints-file", type=Path, default=None, help="yaml mapping the five joint names to radians")


def _add_tag_args(parser: argparse.ArgumentParser, *, require_size: bool = True) -> None:
    parser.add_argument("--intrinsics", type=Path, default=DEFAULT_INTRINSICS)
    parser.add_argument("--dictionary", default="DICT_APRILTAG_36h11")
    parser.add_argument("--marker-id", type=int, default=2)
    parser.add_argument("--size-mm", type=float, required=require_size, help="caliper-measured outer edge of the gripper marker's black square, mm")
    parser.add_argument("--tag-xyz-mm", default="0,0,0", help="marker center position in the tip link, mm")
    parser.add_argument("--tag-rpy-deg", default="0,0,0", help="roll,pitch,yaw of the marker frame relative to the tip link, degrees")
    parser.add_argument("--serial", default=None)
    parser.add_argument("--name-substring", default="455")
    parser.add_argument("--warmup", type=int, default=15)


def _cmd_self_check(args: argparse.Namespace) -> int:
    model = load_urdf(args.urdf)
    pose = forward_kinematics(model, {name: 0.0 for name in ARM_JOINTS})
    if abs(pose.translation[0] - 0.293234) > 1e-3 or abs(pose.translation[2] - 0.234370) > 1e-3:
        print(f"Self-check failed: the gripper origin at the all-zero pose is {pose.translation}, which does not match so101_new_calib.")
        return 1
    base_from_camera = RigidTransform.from_rotation_translation(np.eye(3), [0.2, -0.1, 0.8])
    gripper_from_tag = RigidTransform.from_rotation_translation(np.eye(3), [0.0, 0.0, 0.03])
    generator = np.random.default_rng(0)
    gripper_poses = []
    tag_poses = []
    for _ in range(8):
        axis = generator.normal(size=3)
        axis = axis / np.linalg.norm(axis)
        rotation, _ = cv2.Rodrigues(axis * float(generator.uniform(0.4, 1.3)))
        gripper = RigidTransform.from_rotation_translation(rotation, generator.normal(size=3) * 0.25)
        tag = RigidTransform(np.linalg.inv(base_from_camera.matrix) @ gripper.matrix @ gripper_from_tag.matrix)
        gripper_poses.append(gripper)
        tag_poses.append(tag)
    recovered = solve_eye_to_hand(gripper_poses, tag_poses, method="park")
    other = solve_eye_to_hand(gripper_poses, tag_poses, method="andreff")
    distance, angle, agreed = solutions_agree(recovered, other, translation_m=1e-6, rotation_deg=1e-3)
    error = np.linalg.norm(recovered.base_from_camera.translation - base_from_camera.translation)
    if not agreed or error > 1e-6:
        print(f"Self-check failed: the two solvers differ by {distance * 1000:.3f} mm and {angle:.3f}°; translation error {error:.3e} m.")
        return 1
    print(
        "Self-check passed: at the all-zero pose, gripper_link is at about "
        f"({pose.translation[0] * 1000:.1f}, {pose.translation[1] * 1000:.1f}, {pose.translation[2] * 1000:.1f}) mm in base_link. "
        "The eye-to-hand solver recovers a known camera pose."
    )
    print("q=0 is the zero pose of so101_new_calib.urdf, not a mechanical stop. Do not use the table origin marker as the gripper marker.")
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
    _print_pose("All-zero pose", pose)
    frame = forward_kinematics(
        model, {name: 0.0 for name in ARM_JOINTS}, root_link=args.root_link, tip_link="gripper_frame_link"
    )
    print(
        "The fingertip reference gripper_frame_link at the all-zero pose is at "
        f"({frame.translation[0] * 1000:.1f}, {frame.translation[1] * 1000:.1f}, {frame.translation[2] * 1000:.1f}) mm. "
        "It sits closer to the fingers than gripper_link, is attached by a fixed joint, and does not move when the gripper opens or closes."
    )
    return 0


def _cmd_fk(args: argparse.Namespace) -> int:
    model = load_urdf(args.urdf)
    pose = forward_kinematics(
        model, _joint_map(args, model), root_link=args.root_link, tip_link=args.tip_link
    )
    _print_pose(args.tip_link, pose)
    return 0


def _cmd_zero(args: argparse.Namespace) -> int:
    model = load_urdf(args.urdf)
    payload = yaml.safe_load(Path(args.stops).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("The stops file must map joint names to stop_min_deg, stop_max_deg, reading_deg")
    print("The radians below treat the midpoint of the two stops as 0. This matches URDF q=0 only if the stops are symmetric about the software zero.")
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
            print(f"  The URDF limit midpoint of {name} is not 0. Do not treat the mechanical stop midpoint as q=0 for this joint.")
        if angle < joint.lower - 1e-3 or angle > joint.upper + 1e-3:
            print(f"  The converted angle is outside the URDF limits of {name}.")
    return 0


def _cmd_print(args: argparse.Namespace) -> int:
    if args.marker_id == 0:
        raise ValueError("id 0 is the table origin. Use a different id for the gripper marker (default 2).")
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
    path = args.output_dir / "gripper_marker.png"
    cv2.imwrite(str(path), color)
    print(f"Wrote {path}")
    print("Print at 100%. Measure the outer edge of the black square with calipers and use it as --size-mm. The red arrow is marker +X, the green one is +Y.")
    print("Attach this marker to the gripper, not to the table, and do not cover the id 0 world origin. Remove it after calibration, or make sure it does not block the wrist camera.")
    return 0


def _cmd_servo_zero(args: argparse.Namespace) -> int:
    if args.check:
        return _servo_zero_check()
    if not args.move or not args.port:
        raise ValueError("Step 4.1 needs both --move and --port. Torque stays off and you move each joint to zero by hand.")
    model = load_urdf(args.urdf)
    ids = _servo_ids(args.ids)
    from soarm_aimbot.robots.sts3215 import Sts3215Bus

    bus = Sts3215Bus(args.port, args.baud)
    held: list[int] = []
    try:
        for servo_id in ids.values():
            bus.set_torque(servo_id, False)
        print("Torque is off on the five arm joints. Support the arm with your hand while moving it.")
        mapping = {"ticks_per_revolution": 4096, "joints": {}}
        for name in ARM_JOINTS:
            _print_positive_hint(model, name)
            _print_zero_target(name)
            _live_tick_until_enter(bus, ids[name], name)
            tick_zero = _median_tick(bus, ids[name])
            print(f"{name} zero tick confirmed as {tick_zero}.")
            input(f"Turn only {name} about 10° in the positive direction printed above, then press Enter to read it.")
            tick_positive = _median_tick(bus, ids[name])
            sign = sign_from_ticks(tick_zero, tick_positive)
            mapping["joints"][name] = {"id": ids[name], "tick_at_urdf_zero": tick_zero, "sign": sign}
            print(f"{name}: zero tick {tick_zero}, sign {sign:+d}. Returning to zero and holding; support the arm.")
            held.append(ids[name])
            _return_and_hold(bus, ids[name], tick_zero)
            print(f"{name} is holding at zero tick {tick_zero}. The remaining joints can still be moved by hand.")
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(yaml.safe_dump(mapping, sort_keys=False, allow_unicode=True), encoding="utf-8")
        print(f"Wrote {destination}.")
    finally:
        try:
            for servo_id in held:
                bus.set_torque(servo_id, False)
        finally:
            bus.close()
    print("Torque is off on all five joints.")
    return 0


def _cmd_move_zero(args: argparse.Namespace) -> int:
    model = load_urdf(args.urdf)
    servo_map = _load_servo_map(args.servo_map)
    goal = np.zeros(5)
    zero_ticks = _ticks_for_joints(servo_map, goal)
    named = dict(zip(ARM_JOINTS, zero_ticks))
    print("Target ticks for q=0: " + ", ".join(f"{name} {tick}" for name, tick in named.items()) + ".")
    if not args.move:
        print("No --move given; not opening the serial port.")
        return 0
    if not args.port:
        raise ValueError("Moving to zero needs --port.")
    from soarm_aimbot.robots.sts3215 import Sts3215Bus

    bus = Sts3215Bus(args.port, args.baud)
    try:
        start_ticks = _read_ticks(bus, servo_map)
        start = _radians_from_ticks(servo_map, start_ticks)
        _require_joints(model, start)
        for row in interpolate(start, goal, _speed_limit(model, args.speed_scale), args.period_s).joints_rad:
            _ticks_for_joints(servo_map, row)
        _hold_ticks(bus, servo_map, start_ticks)
        _move_arm(bus, servo_map, model, start, goal, args.speed_scale, args.period_s)
        measured = _read_arm_radians(bus, servo_map, model)
        error = float(np.max(np.abs(measured)))
        if error > args.arrive_rad:
            raise ValueError(f"Read-back joint angles differ from 0 by {error:.3f} rad, more than {args.arrive_rad:.3f} rad. Torque is still on.")
        print("All five joints reached q=0, max deviation {:.4f} rad.".format(error))
        if args.release:
            for item in servo_map["joints"].values():
                bus.set_torque(int(item["id"]), False)
            print("Torque is off.")
        else:
            print("Torque is still on; the arm is holding at zero.")
    finally:
        bus.close()
    return 0


def _cmd_servo_capture(args: argparse.Namespace) -> int:
    if args.check:
        return _servo_capture_check()
    if args.size_mm is None:
        raise ValueError("Capturing the marker needs --size-mm.")
    servo_map = _load_servo_map(args.servo_map)
    if not args.move:
        print("Step 4.2 is a hand-guided capture. Add --move --port to open the serial port and the overhead camera.")
        print("Torque stays off. The window and terminal show the current joint angles and marker. s saves, q finishes and solves.")
        return 0
    if not args.port:
        raise ValueError("Hand-guided capture needs --port.")
    model = load_urdf(args.urdf)
    intrinsics, metadata = load_intrinsics(args.intrinsics)
    from soarm_aimbot.cameras.realsense import RealSenseRgbd
    from soarm_aimbot.robots.sts3215 import Sts3215Bus

    camera = RealSenseRgbd(
        width=intrinsics.width,
        height=intrinsics.height,
        fps=int(metadata.get("fps", 30)),
        serial=args.serial or metadata.get("serial"),
        name_substring=args.name_substring,
    )
    bus = Sts3215Bus(args.port, args.baud)
    saved = 0
    try:
        _set_arm_torque(bus, servo_map, False)
        print("Torque is off on the five arm joints. Move the arm by hand to a pose where the marker is fully visible and the reprojection error is at most 2 px.")
        print("Press s to save the current pose, or q to stop capturing and solve with the saved samples.")
        with camera:
            for _ in range(max(args.warmup, 0)):
                camera.read()
            try:
                saved = _run_manual_capture(args, bus, camera, servo_map, model, intrinsics)
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


def _cmd_capture(args: argparse.Namespace) -> int:
    model = load_urdf(args.urdf)
    joints = _joint_map(args, model)
    pose = _average_tag_pose(args)
    _store_sample(args, joints, pose)
    return 0


def _store_sample(args, joints: dict[str, float], pose) -> None:
    store = _load_samples(args.samples_file, reset=args.reset)
    store.update(
        {
            "urdf": str(args.urdf),
            "root_link": args.root_link,
            "tip_link": args.tip_link,
            "dictionary": args.dictionary,
            "marker_id": args.marker_id,
            "marker_size_m": args.size_mm / 1000.0,
            "tag_xyz_m": [value / 1000.0 for value in _triplet(args.tag_xyz_mm)],
            "tag_rpy_deg": list(_triplet(args.tag_rpy_deg)),
        }
    )
    store.setdefault("samples", []).append(
        {
            "joints_rad": [joints[name] for name in ARM_JOINTS],
            "T_overhead_tag": pose.transform.matrix.tolist(),
            "reprojection_px": pose.reprojection_px,
        }
    )
    path = Path(args.samples_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(store, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"Recorded pose {len(store['samples'])}, reprojection {pose.reprojection_px:.3f} px, written to {path}")
    return 0


def _cmd_solve(args: argparse.Namespace) -> int:
    store = _load_samples(args.samples_file, reset=False)
    samples = store.get("samples", [])
    if args.holdout < 0 or args.holdout >= len(samples):
        raise ValueError("--holdout must be smaller than the number of captured poses")
    used = samples if args.holdout == 0 else samples[: -args.holdout]
    held = [] if args.holdout == 0 else samples[-args.holdout :]
    model = load_urdf(store["urdf"])
    gripper_poses, tag_poses = _poses_from_samples(model, store, used)
    park = solve_eye_to_hand(gripper_poses, tag_poses, method="park")
    andreff = solve_eye_to_hand(gripper_poses, tag_poses, method="andreff")
    distance, angle, agreed = solutions_agree(
        park, andreff, translation_m=args.agree_mm / 1000.0, rotation_deg=args.agree_deg
    )
    print(
        f"Park and Andreff differ by {distance * 1000:.1f} mm and {angle:.2f}°, "
        f"using {park.pair_count} relative-motion pairs."
    )
    if not agreed:
        print("The two solvers disagree too much. Add poses with more rotation between them and solve again. Nothing was written.")
        return 1
    world_from_overhead, _ = load_transform(args.extrinsics)
    base_from_world = RigidTransform(park.base_from_camera.matrix @ world_from_overhead.inverse().matrix)
    holdout_errors = [
        _tag_error_m(model, store, item, park.base_from_camera) for item in held
    ]
    payload = {
        "root_link": store["root_link"],
        "tip_link": store["tip_link"],
        "urdf": store["urdf"],
        "method": "park",
        "checked_against": "andreff",
        "solver_translation_disagreement_m": distance,
        "solver_rotation_disagreement_deg": angle,
        "pair_count": park.pair_count,
        "sample_count": len(used),
        "holdout_count": len(held),
        "holdout_translation_error_m": holdout_errors,
        "T_base_overhead": park.base_from_camera.matrix.tolist(),
        "T_base_world": base_from_world.matrix.tolist(),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"Wrote {destination}")
    print(
        "T_base_world maps table coordinates into base_link: "
        f"world origin in base at ({base_from_world.translation[0]:.3f}, "
        f"{base_from_world.translation[1]:.3f}, {base_from_world.translation[2]:.3f}) m."
    )
    if holdout_errors:
        worst = max(holdout_errors) * 1000.0
        print(f"Largest marker translation error on held-out poses: {worst:.1f} mm.")
        if worst > args.verify_mm:
            print(f"Held-out poses exceed {args.verify_mm:.0f} mm. The file was written, but this verification did not pass.")
            return 1
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    calibration = yaml.safe_load(Path(args.calibration).read_text(encoding="utf-8"))
    base_from_camera = RigidTransform(np.array(calibration["T_base_overhead"], dtype=np.float64))
    model = load_urdf(args.urdf)
    joints = _joint_map(args, model)
    observed = _average_tag_pose(args)
    mount = _tag_mount(args.tag_xyz_mm, args.tag_rpy_deg)
    predicted = RigidTransform(
        forward_kinematics(model, joints, root_link=args.root_link, tip_link=args.tip_link).matrix @ mount.matrix
    )
    seen = RigidTransform(base_from_camera.matrix @ observed.transform.matrix)
    distance = translation_distance_m(predicted, seen)
    angle = rotation_angle_deg(predicted, seen)
    print(f"Marker origin: forward kinematics and vision differ by {distance * 1000:.1f} mm, orientation by {angle:.2f}°.")
    if distance > args.tolerance_mm / 1000.0:
        print("The difference exceeds the tolerance. Check the gripper marker mounting pose and the joint zeros before recapturing poses.")
        return 1
    print("The translation error at this pose is within the tolerance.")
    return 0


def _cmd_map(args: argparse.Namespace) -> int:
    calibration = yaml.safe_load(Path(args.calibration).read_text(encoding="utf-8"))
    base_from_world = RigidTransform(np.array(calibration["T_base_world"], dtype=np.float64))
    point = np.array(_triplet(args.xyz_mm), dtype=np.float64) / 1000.0
    mapped = base_from_world.apply(point)
    print(
        "base_link coordinates "
        f"({mapped[0] * 1000:.1f}, {mapped[1] * 1000:.1f}, {mapped[2] * 1000:.1f}) mm"
    )
    return 0


def _servo_zero_check() -> int:
    packet = read_position_packet(1)
    if packet[4] != 0x02 or packet[5] != 56 or (~sum(packet[2:-1]) & 0xFF) != packet[-1]:
        print("Self-check failed: the read-position packet is wrong.")
        return 1
    enabled = torque_packet(2, True)
    if enabled[5] != 40 or enabled[6] != 1:
        print("Self-check failed: the torque packet is wrong.")
        return 1
    status = bytes([0xFF, 0xFF, 1, 4, 0, 0x00, 0x08, 0])
    status = status[:-1] + bytes([(~sum(status[2:-1]) & 0xFF)])
    parsed = parse_status(status, 1)
    if parsed != b"\x00\x08":
        print("Self-check failed: the position reply did not decode to tick bytes.")
        return 1
    if sign_from_ticks(2000, 2100) != 1 or sign_from_ticks(2000, 1900) != -1:
        print("Self-check failed: the tick change in the positive direction does not match sign.")
        return 1
    angle = 0.2
    recovered = tick_to_radians(radians_to_tick(angle, 2000, -1), 2000, -1)
    if abs(recovered - angle) > np.pi / TICKS_PER_REVOLUTION:
        print("Self-check failed: tick and radian conversions are not inverses.")
        return 1
    print("Self-check passed: read-position, torque on/off and tick conversion match the packet format. sign is +1 when ticks increase in the positive direction.")
    print("Step 4.1 first turns torque off. After a joint's zero and positive direction are recorded, it returns to zero and holds. Torque is turned off once all five joints are done.")
    return 0


def _servo_capture_check() -> int:
    from soarm_aimbot.geometry.intrinsics import PinholeIntrinsics

    blank = np.zeros((48, 64, 3), dtype=np.uint8)
    intrinsics = PinholeIntrinsics(64, 48, 50.0, 50.0, 32.0, 24.0)
    view_args = argparse.Namespace(dictionary="DICT_APRILTAG_36h11", marker_id=2, size_mm=40.0, frames=4)
    if _inspect_tag(blank, view_args, intrinsics) is not None:
        print("Self-check failed: a blank image should not yield a marker.")
        return 1

    class Camera:
        def read(self):
            from soarm_aimbot.cameras.base import RgbdFrame

            depth = np.zeros((48, 64), dtype=np.float32)
            return RgbdFrame(blank, depth, intrinsics, 0.0, "", "")

    if _average_tag_pose_from_camera(Camera(), view_args, intrinsics) is not None:
        print("Self-check failed: a blank sequence should not yield an average pose.")
        return 1
    print("Self-check passed: when the marker is not visible, the preview and multi-frame averaging carry on without aborting capture. Press s to save, q to finish and solve.")
    return 0


def _servo_ids(text: str) -> dict[str, int]:
    values = [int(item) for item in text.split(",")]
    if len(values) != 5 or len(set(values)) != 5:
        raise ValueError("Need five distinct servo ids, in the order shoulder_pan,shoulder_lift,elbow_flex,wrist_flex,wrist_roll")
    if any(not 0 <= value <= 253 for value in values):
        raise ValueError("Servo ids must be between 0 and 253")
    return dict(zip(ARM_JOINTS, values))


def _print_zero_target(name: str) -> None:
    if name == "wrist_roll":
        print(
            "wrist_roll: first note the ticks at both stops. q=0 is 49.1% of the way from the negative stop to the positive stop, "
            "about 2.8° from the midpoint toward the negative side. The positive direction is the fingertip motion printed above."
        )
    else:
        print(f"{name}: push it against both stops and note the ticks, then set it to the average of the two.")
    print("The current tick refreshes on one line. Press Enter when it is in position.")


def _return_and_hold(bus, servo_id: int, tick_zero: int, period_s: float = 0.05) -> None:
    """Enable torque at the present tick, then step the goal back to ``tick_zero``."""

    present = int(bus.read_position(servo_id))
    bus.write_goal(servo_id, present)
    bus.set_torque(servo_id, True)
    span = int(tick_zero) - present
    steps = max(int(abs(span) / 16), 1)
    for index in range(1, steps + 1):
        tick = int(round(present + span * index / steps))
        bus.write_goal(servo_id, tick)
        _sleep(period_s)


def _live_tick_until_enter(bus, servo_id: int, name: str) -> None:
    """Refresh one joint's present tick until Enter. Torque stays off."""

    import select
    import sys

    try:
        while True:
            tick = int(bus.read_position(servo_id))
            sys.stdout.write(f"\r{name:<16} tick {tick:4d}   ")
            sys.stdout.flush()
            ready, _, _ = select.select([sys.stdin], [], [], 0.05)
            if not ready:
                continue
            sys.stdin.readline()
            return
    finally:
        sys.stdout.write("\n")
        sys.stdout.flush()


def _print_positive_hint(model: UrdfModel, name: str) -> None:
    zeros = {joint: 0.0 for joint in ARM_JOINTS}
    moved = dict(zeros)
    moved[name] = 0.2
    origin = forward_kinematics(model, zeros).translation
    shifted = forward_kinematics(model, moved).translation
    delta = (shifted - origin) * 1000.0
    print(
        f"Turning {name} about 11° in the URDF positive direction moves the fingertip by about "
        f"({delta[0]:.0f}, {delta[1]:.0f}, {delta[2]:.0f}) mm in base_link. "
        "To see what q=0 looks like, compare with aimbot-step04 fk --joints 0,0,0,0,0."
    )


def _median_tick(bus, servo_id: int) -> int:
    ticks = [int(bus.read_position(servo_id)) for _ in range(5)]
    return int(np.median(ticks))


def _capture_goals(args) -> list[np.ndarray]:
    if args.joints and args.poses_file is not None:
        raise ValueError("Give only one of --joints and --poses-file")
    if args.poses_file is not None:
        payload = yaml.safe_load(Path(args.poses_file).read_text(encoding="utf-8"))
        rows = payload["poses"] if isinstance(payload, dict) else payload
        if not isinstance(rows, list) or not rows:
            raise ValueError("The poses file must list poses of five angles in radians")
        goals = []
        for row in rows:
            values = row["joints_rad"] if isinstance(row, dict) else row
            goals.append(np.asarray(values, dtype=np.float64))
        return goals
    if not args.joints:
        raise ValueError("Give the target joint angles with --joints or --poses-file.")
    return [np.asarray([float(item) for item in args.joints.split(",")], dtype=np.float64)]


def _require_joints(model: UrdfModel, joints: np.ndarray) -> None:
    values = np.asarray(joints, dtype=np.float64).reshape(-1)
    if values.size != 5:
        raise ValueError("Each pose needs five joint angles")
    for name, angle in zip(ARM_JOINTS, values):
        joint = model.joints[name]
        if joint.lower is not None and not joint.lower - 1e-3 <= float(angle) <= joint.upper + 1e-3:
            raise ValueError(f"{name}={float(angle):.4f} is outside the URDF limits")


def _load_servo_map(path: Path) -> dict:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "joints" not in payload:
        raise ValueError(f"{path} has no joints. Run Step 4.1 servo-zero first.")
    for name in ARM_JOINTS:
        item = payload["joints"][name]
        if int(item["sign"]) not in (-1, 1):
            raise ValueError(f"sign of {name} must be 1 or -1")
    return payload


def _ticks_for_joints(servo_map: dict, joints: np.ndarray) -> list[int]:
    per_rev = int(servo_map.get("ticks_per_revolution", 4096))
    ticks = []
    for name, angle in zip(ARM_JOINTS, np.asarray(joints, dtype=np.float64).reshape(5)):
        item = servo_map["joints"][name]
        ticks.append(radians_to_tick(float(angle), int(item["tick_at_urdf_zero"]), int(item["sign"]), per_rev))
    return ticks


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


def _hold_ticks(bus, servo_map: dict, ticks: dict[str, int]) -> None:
    for name in ARM_JOINTS:
        bus.write_goal(int(servo_map["joints"][name]["id"]), int(ticks[name]))
    for name in ARM_JOINTS:
        bus.set_torque(int(servo_map["joints"][name]["id"]), True)


def _move_arm(bus, servo_map, model, start, goal, speed_scale: float, period_s: float) -> None:
    limit = _speed_limit(model, speed_scale)
    trajectory = interpolate(start, goal, limit, period_s)
    for row in trajectory.joints_rad:
        _ticks_for_joints(servo_map, row)

    def send(point: np.ndarray) -> None:
        for name, tick in zip(ARM_JOINTS, _ticks_for_joints(servo_map, point)):
            bus.write_goal(int(servo_map["joints"][name]["id"]), tick)

    result = play(trajectory.joints_rad, period_s, send, _sleep, lambda: False)
    if result.status == "estop":
        raise ValueError("Emergency stop. The arm holds at the last commanded position with torque still on.")


def _speed_limit(model: UrdfModel, scale: float) -> np.ndarray:
    if not 0 < scale <= 1:
        raise ValueError("The speed scale must be between 0 and 1")
    return np.array([float(model.joints[name].velocity) * scale for name in ARM_JOINTS])


def _sleep(period_s: float) -> None:
    import time

    time.sleep(period_s)


def _poses_from_samples(model: UrdfModel, store: dict, samples: list[dict]):
    gripper_poses = []
    tag_poses = []
    for item in samples:
        joints = dict(zip(ARM_JOINTS, item["joints_rad"]))
        gripper_poses.append(
            forward_kinematics(model, joints, root_link=store["root_link"], tip_link=store["tip_link"])
        )
        tag_poses.append(RigidTransform(np.array(item["T_overhead_tag"], dtype=np.float64)))
    return gripper_poses, tag_poses


def _tag_error_m(model, store, item, base_from_camera: RigidTransform) -> float:
    joints = dict(zip(ARM_JOINTS, item["joints_rad"]))
    gripper = forward_kinematics(model, joints, root_link=store["root_link"], tip_link=store["tip_link"])
    mount = RigidTransform.from_rotation_translation(
        _rpy_deg(store.get("tag_rpy_deg", [0, 0, 0])),
        np.array(store.get("tag_xyz_m", [0, 0, 0]), dtype=np.float64),
    )
    predicted = gripper.matrix @ mount.matrix
    seen = base_from_camera.matrix @ np.array(item["T_overhead_tag"], dtype=np.float64)
    return float(np.linalg.norm(predicted[:3, 3] - seen[:3, 3]))


class _TagView:
    def __init__(self, corners: np.ndarray, pose: RigidTransform, error: float) -> None:
        self.corners = corners
        self.pose = pose
        self.error = float(error)
        self.ready = self.error <= 2.0


def _run_manual_capture(args, bus, camera, servo_map, model, intrinsics, wait_key=None, show=None) -> int:
    wait_key = cv2.waitKey if wait_key is None else wait_key
    show = cv2.imshow if show is None else show
    saved = _sample_count(args.samples_file) if not args.reset else 0
    while True:
        frame = camera.read()
        measured = _try_read_arm_radians(bus, servo_map, model)
        joints_text = (
            ",".join(f"{value:.3f}" for value in measured) if measured is not None else "out of limits or read failed"
        )
        view = _inspect_tag(frame.color_bgr, args, intrinsics)
        show("step04 servo-capture", _draw_capture_preview(frame.color_bgr, view, joints_text, saved))
        key = int(wait_key(1)) & 0xFF
        tag_text = "no marker" if view is None else f"reproj {view.error:.2f} px"
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
            print("The current joint angles are not usable; not saved. Keep posing.")
            continue
        if view is None or not view.ready:
            print("The current view does not meet reprojection ≤ 2 px yet; not saved. Keep posing.")
            continue
        _hold_ticks(bus, servo_map, _read_ticks(bus, servo_map))
        pose = _average_tag_pose_from_camera(camera, args, intrinsics)
        _set_arm_torque(bus, servo_map, False)
        if pose is None:
            print("Not enough valid frames this time; nothing was written. Move on to the next pose.")
            continue
        held = _try_read_arm_radians(bus, servo_map, model)
        values = held if held is not None else measured
        joints = {name: float(value) for name, value in zip(ARM_JOINTS, values)}
        _store_sample(args, joints, pose)
        args.reset = False
        saved = _sample_count(args.samples_file)
        print(f"Total poses collected: {saved}. Torque is released; you can keep posing.")


def _solve_after_capture(args: argparse.Namespace) -> int:
    samples = _load_samples(args.samples_file, reset=False).get("samples", [])
    if len(samples) < 4:
        print(f"Saved {len(samples)} poses. Solving needs at least 4, so skipping the solve.")
        return 0
    if args.holdout >= len(samples) or len(samples) - args.holdout < 4:
        print(f"Holding out {args.holdout} poses would leave fewer than 4 for the solve; using all poses instead.")
        args.holdout = 0
    print("Solving with the saved samples.")
    return _cmd_solve(args)


def _inspect_tag(color_bgr, args, intrinsics) -> _TagView | None:
    corners = detect_marker(color_bgr, args.dictionary, args.marker_id)
    if corners is None:
        return None
    try:
        pose = pose_from_corners(corners, args.size_mm / 1000.0, intrinsics)
    except RuntimeError:
        return None
    error = reprojection_error_px(corners, args.size_mm / 1000.0, intrinsics, pose)
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


def _set_arm_torque(bus, servo_map: dict, enabled: bool) -> None:
    for item in servo_map["joints"].values():
        bus.set_torque(int(item["id"]), enabled)


def _try_read_arm_radians(bus, servo_map: dict, model: UrdfModel) -> np.ndarray | None:
    try:
        return _read_arm_radians(bus, servo_map, model)
    except ValueError:
        return None


def _sample_count(path: Path) -> int:
    return len(_load_samples(path, reset=False).get("samples", []))


def _average_tag_pose(args):
    intrinsics, metadata = load_intrinsics(args.intrinsics)
    from soarm_aimbot.cameras.realsense import RealSenseRgbd

    camera = RealSenseRgbd(
        width=intrinsics.width,
        height=intrinsics.height,
        fps=int(metadata.get("fps", 30)),
        serial=args.serial or metadata.get("serial"),
        name_substring=args.name_substring,
    )
    with camera:
        for _ in range(max(args.warmup, 0)):
            camera.read()
        pose = _average_tag_pose_from_camera(camera, args, intrinsics)
    if pose is None:
        raise ValueError("The gripper marker was detected in too few frames. Check the id and size, and that the marker faces the overhead camera.")
    return pose


def _average_tag_pose_from_camera(camera, args, intrinsics):
    size_m = args.size_mm / 1000.0
    found: list[tuple[RigidTransform, float]] = []
    attempts = 0
    while len(found) < args.frames and attempts < args.frames * 15:
        attempts += 1
        frame = camera.read()
        view = _inspect_tag(frame.color_bgr, args, intrinsics)
        if view is None or not view.ready:
            continue
        found.append((view.pose, view.error))
    if len(found) < max(5, args.frames // 2):
        return None
    mean_error = float(np.mean([item[1] for item in found]))
    return _Pose(average_transforms([item[0] for item in found]), mean_error)


class _Pose:
    def __init__(self, transform: RigidTransform, reprojection_px: float) -> None:
        self.transform = transform
        self.reprojection_px = reprojection_px


def _joint_map(args, model: UrdfModel) -> dict[str, float]:
    if args.joints_file is not None:
        payload = yaml.safe_load(Path(args.joints_file).read_text(encoding="utf-8"))
        return {name: float(payload[name]) for name in ARM_JOINTS}
    if not args.joints:
        raise ValueError("Give five joint angles in radians with --joints or --joints-file.")
    values = [float(item) for item in args.joints.split(",")]
    if len(values) != len(ARM_JOINTS):
        raise ValueError("Need five joint angles: shoulder_pan,shoulder_lift,elbow_flex,wrist_flex,wrist_roll")
    mapping = dict(zip(ARM_JOINTS, values))
    for name, angle in mapping.items():
        joint = model.joints[name]
        if joint.lower is not None and not joint.lower - 1e-4 <= angle <= joint.upper + 1e-4:
            raise ValueError(f"{name}={angle:.4f} is outside the URDF limits [{joint.lower:.4f}, {joint.upper:.4f}]")
    return mapping


def _tag_mount(xyz_mm: str, rpy_deg: str) -> RigidTransform:
    return RigidTransform.from_rotation_translation(
        _rpy_deg(_triplet(rpy_deg)),
        np.array(_triplet(xyz_mm), dtype=np.float64) / 1000.0,
    )


def _rpy_deg(values) -> np.ndarray:
    from soarm_aimbot.geometry.urdf_fk import rpy_matrix

    return rpy_matrix(np.deg2rad(np.asarray(values, dtype=np.float64)))


def _triplet(text: str) -> tuple[float, float, float]:
    parts = text.split(",")
    if len(parts) != 3:
        raise ValueError("Need three comma-separated numbers")
    return tuple(float(part) for part in parts)  # type: ignore[return-value]


def _load_samples(path: Path, reset: bool) -> dict:
    if reset or not Path(path).is_file():
        return {"samples": []}
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {"samples": []}
    payload.setdefault("samples", [])
    return payload


def _print_pose(title: str, pose: RigidTransform) -> None:
    origin = pose.translation
    print(
        f"{title} origin in the root link: ({origin[0] * 1000:.1f}, {origin[1] * 1000:.1f}, {origin[2] * 1000:.1f}) mm"
    )
    print(f"  link +X in the root link: ({pose.rotation[0, 0]:.3f}, {pose.rotation[1, 0]:.3f}, {pose.rotation[2, 0]:.3f})")
    print(f"  link +Y in the root link: ({pose.rotation[0, 1]:.3f}, {pose.rotation[1, 1]:.3f}, {pose.rotation[2, 1]:.3f})")
    print(f"  link +Z in the root link: ({pose.rotation[0, 2]:.3f}, {pose.rotation[1, 2]:.3f}, {pose.rotation[2, 2]:.3f})")


if __name__ == "__main__":
    raise SystemExit(main())
