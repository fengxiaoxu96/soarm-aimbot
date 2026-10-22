import cv2
import numpy as np
import yaml

from soarm_aimbot.geometry.marker import marker_object_points, pose_from_corners_rgb, reprojection_error_rgb
from soarm_aimbot.geometry.rgb_model import RgbCameraModel, save_rgb_intrinsics
from soarm_aimbot.geometry.transforms import RigidTransform
from soarm_aimbot.geometry.urdf_fk import ARM_JOINTS, forward_kinematics, load_urdf
from soarm_aimbot.steps import step05_wrist_handeye
from soarm_aimbot.steps.step05_wrist_handeye import main


def test_capture_check_and_dry_run(tmp_path, capsys):
    assert main(["capture", "--check"]) == 0
    servo_map = {
        "ticks_per_revolution": 4096,
        "joints": {name: {"id": index + 1, "tick_at_urdf_zero": 2048, "sign": 1} for index, name in enumerate(ARM_JOINTS)},
    }
    path = tmp_path / "servo_map.yaml"
    path.write_text(yaml.safe_dump(servo_map), encoding="utf-8")
    assert main(["capture", "--servo-map", str(path), "--size-mm", "40"]) == 0
    shown = capsys.readouterr().out
    assert "hand-guided" in shown
    assert "Press s to save" in shown


def test_manual_capture_saves_on_s_and_keeps_going_when_average_fails(tmp_path, monkeypatch, capsys):
    servo_map = {
        "ticks_per_revolution": 4096,
        "joints": {name: {"id": index + 1, "tick_at_urdf_zero": 2048, "sign": 1} for index, name in enumerate(ARM_JOINTS)},
    }
    samples = tmp_path / "samples.yaml"
    args = step05_wrist_handeye._parser().parse_args(
        [
            "capture",
            "--servo-map",
            str(tmp_path / "unused.yaml"),
            "--samples-file",
            str(samples),
            "--size-mm",
            "40",
            "--reset",
        ]
    )
    model = load_urdf(args.urdf)
    camera_model = RgbCameraModel(64, 48, 50.0, 50.0, 32.0, 24.0, (0.0, 0.0, 0.0, 0.0, 0.0))
    zeros = np.zeros(5)
    keys = iter([ord("s"), ord("s"), ord("q")])

    class Bus:
        def read_position(self, servo_id):
            return 2048

        def write_goal(self, servo_id, tick):
            return None

        def set_torque(self, servo_id, enabled):
            return None

    class Camera:
        def read(self):
            return np.zeros((48, 64, 3), dtype=np.uint8)

    ready = step05_wrist_handeye._TagView(
        np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32),
        RigidTransform(np.eye(4)),
        0.4,
    )
    observations = iter(
        [
            None,
            step05_wrist_handeye._Observation(
                RigidTransform(np.eye(4)),
                np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64),
                0.4,
            ),
        ]
    )
    monkeypatch.setattr(step05_wrist_handeye, "_inspect_tag", lambda *_args, **_kwargs: ready)
    monkeypatch.setattr(step05_wrist_handeye, "_try_read_arm_radians", lambda *_args, **_kwargs: zeros)
    monkeypatch.setattr(step05_wrist_handeye, "_read_ticks", lambda *_args, **_kwargs: {name: 2048 for name in ARM_JOINTS})
    monkeypatch.setattr(step05_wrist_handeye, "_average_tag_pose_from_camera", lambda *_args, **_kwargs: next(observations))
    saved = step05_wrist_handeye._run_manual_capture(
        args,
        Bus(),
        Camera(),
        servo_map,
        model,
        camera_model,
        wait_key=lambda _delay: next(keys),
        show=lambda *_args, **_kwargs: None,
    )
    shown = capsys.readouterr().out
    assert saved == 1
    assert "Not enough valid frames" in shown
    assert "Poses collected: 1." in shown
    payload = yaml.safe_load(samples.read_text(encoding="utf-8"))
    assert len(payload["samples"]) == 1


def test_self_check_joints_and_zero(tmp_path):
    assert main(["self-check"]) == 0
    assert main(["joints"]) == 0
    stops = {
        name: {"stop_min_deg": -90, "stop_max_deg": 90, "reading_deg": 0}
        for name in ARM_JOINTS
    }
    stops["wrist_roll"] = {"stop_min_deg": -157.2, "stop_max_deg": 162.8, "reading_deg": 2.8}
    path = tmp_path / "stops.yaml"
    path.write_text(yaml.safe_dump(stops), encoding="utf-8")
    assert main(["zero", "--stops", str(path)]) == 0
    assert main(["print-marker", "--marker-id", "0"]) == 1


def test_brown_pose_roundtrip():
    model = RgbCameraModel(640, 480, 500.0, 510.0, 320.0, 240.0, (0.1, -0.05, 0.0, 0.0, 0.0))
    expected = RigidTransform.from_rotation_translation(np.eye(3), [0.01, -0.02, 0.25])
    corners = model.project(
        marker_object_points(0.04),
        np.zeros((3, 1)),
        expected.translation.reshape(3, 1),
    )
    recovered = pose_from_corners_rgb(corners, 0.04, model)
    assert np.linalg.norm(recovered.translation - expected.translation) < 1e-4
    assert reprojection_error_rgb(corners, 0.04, model, recovered) < 0.05


def test_solve_recovers_camera_mount(tmp_path):
    urdf = load_urdf("assets/so101/so101_new_calib.urdf")
    rotation, _ = cv2.Rodrigues(np.array([0.3, -0.2, 0.4]))
    gripper_from_camera = RigidTransform.from_rotation_translation(rotation, [0.02, -0.04, 0.03])
    board = RigidTransform.from_rotation_translation(np.eye(3), [0.28, 0.04, 0.01])
    camera = RgbCameraModel(640, 480, 520.0, 520.0, 320.0, 240.0, (0.0, 0.0, 0.0, 0.0, 0.0))
    intrinsics = tmp_path / "wrist.yaml"
    save_rgb_intrinsics(intrinsics, camera, {"camera": "wrist"})
    samples = []
    for index in range(8):
        joints = [
            -0.9 + 0.22 * index,
            -0.5 + 0.12 * index,
            -0.4 + 0.08 * index,
            0.6 - 0.15 * index,
            -1.0 + 0.25 * index,
        ]
        gripper = forward_kinematics(urdf, dict(zip(ARM_JOINTS, joints)))
        tag = RigidTransform(
            np.linalg.inv(gripper_from_camera.matrix) @ np.linalg.inv(gripper.matrix) @ board.matrix
        )
        corners = camera.project(
            marker_object_points(0.04),
            cv2.Rodrigues(tag.rotation)[0],
            tag.translation.reshape(3, 1),
        )
        samples.append(
            {
                "joints_rad": joints,
                "T_wrist_tag": tag.matrix.tolist(),
                "corners_px": corners.tolist(),
                "reprojection_px": 0.0,
            }
        )
    samples_path = tmp_path / "samples.yaml"
    samples_path.write_text(
        yaml.safe_dump(
            {
                "urdf": "assets/so101/so101_new_calib.urdf",
                "root_link": "base_link",
                "tip_link": "gripper_link",
                "intrinsics": str(intrinsics),
                "marker_size_m": 0.04,
                "samples": samples,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    output = tmp_path / "T_gripper_wrist.yaml"
    assert (
        main(
            [
                "solve",
                "--samples-file",
                str(samples_path),
                "--output",
                str(output),
                "--holdout",
                "2",
            ]
        )
        == 0
    )
    payload = yaml.safe_load(output.read_text(encoding="utf-8"))
    recovered = np.array(payload["T_gripper_wrist"], dtype=np.float64)
    assert np.linalg.norm(recovered[:3, 3] - gripper_from_camera.translation) < 1e-4
    assert max(payload["holdout_translation_error_m"]) < 1e-4
    assert max(payload["holdout_reprojection_px"]) < 0.05
    assert (
        main(
            [
                "pose",
                "--calibration",
                str(output),
                "--joints",
                "0,0,0,0,0",
            ]
        )
        == 0
    )
