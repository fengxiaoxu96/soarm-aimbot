import io
import sys

import numpy as np
import yaml

from soarm_aimbot.geometry.transforms import RigidTransform, save_transform
from soarm_aimbot.geometry.urdf_fk import ARM_JOINTS, forward_kinematics, load_urdf
from soarm_aimbot.steps import step04_base_frame
from soarm_aimbot.steps.step04_base_frame import main


def test_return_and_hold_starts_at_the_present_tick(monkeypatch):
    class Bus:
        def __init__(self):
            self.calls = []

        def read_position(self, servo_id):
            return 2100

        def write_goal(self, servo_id, tick):
            self.calls.append(("goal", servo_id, tick))

        def set_torque(self, servo_id, enabled):
            self.calls.append(("torque", servo_id, enabled))

    monkeypatch.setattr(step04_base_frame, "_sleep", lambda _period: None)
    bus = Bus()
    step04_base_frame._return_and_hold(bus, 2, 2000)
    assert bus.calls[0] == ("goal", 2, 2100)
    assert bus.calls[1] == ("torque", 2, True)
    assert bus.calls[-1] == ("goal", 2, 2000)
    goals = [tick for kind, _servo_id, tick in bus.calls if kind == "goal"]
    assert goals == sorted(goals, reverse=True)


def test_move_zero_prints_recorded_ticks_without_opening_a_port(tmp_path, capsys):
    servo_map = {
        "ticks_per_revolution": 4096,
        "joints": {
            name: {"id": index + 1, "tick_at_urdf_zero": 1500 + index, "sign": 1}
            for index, name in enumerate(ARM_JOINTS)
        },
    }
    path = tmp_path / "servo_map.yaml"
    path.write_text(yaml.safe_dump(servo_map), encoding="utf-8")
    assert main(["move-zero", "--servo-map", str(path)]) == 0
    shown = capsys.readouterr().out
    assert "shoulder_pan 1500" in shown
    assert "wrist_roll 1504" in shown
    assert "No --move" in shown


def test_live_tick_refreshes_until_enter(monkeypatch, capsys):
    class Bus:
        def read_position(self, servo_id):
            return 1800 + servo_id

    monkeypatch.setattr("select.select", lambda *_args, **_kwargs: ([sys.stdin], [], []))
    monkeypatch.setattr(sys, "stdin", io.StringIO("\n"))
    step04_base_frame._live_tick_until_enter(Bus(), 3, "elbow_flex")
    shown = capsys.readouterr().out
    assert "elbow_flex" in shown
    assert "1803" in shown


def test_servo_checks_and_dry_run(tmp_path, capsys):
    assert main(["servo-zero", "--check"]) == 0
    assert main(["servo-capture", "--check"]) == 0
    servo_map = {
        "ticks_per_revolution": 4096,
        "joints": {name: {"id": index + 1, "tick_at_urdf_zero": 2048, "sign": 1} for index, name in enumerate(ARM_JOINTS)},
    }
    path = tmp_path / "servo_map.yaml"
    path.write_text(yaml.safe_dump(servo_map), encoding="utf-8")
    assert main(["servo-capture", "--servo-map", str(path), "--size-mm", "40"]) == 0
    shown = capsys.readouterr().out
    assert "hand-guided capture" in shown
    assert "s saves" in shown


def test_manual_capture_saves_on_s_and_keeps_going_when_average_fails(tmp_path, monkeypatch, capsys):
    from soarm_aimbot.geometry.intrinsics import PinholeIntrinsics

    servo_map = {
        "ticks_per_revolution": 4096,
        "joints": {name: {"id": index + 1, "tick_at_urdf_zero": 2048, "sign": 1} for index, name in enumerate(ARM_JOINTS)},
    }
    samples = tmp_path / "samples.yaml"
    args = step04_base_frame._parser().parse_args(
        [
            "servo-capture",
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
    intrinsics = PinholeIntrinsics(64, 48, 50.0, 50.0, 32.0, 24.0)
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
            return type("Frame", (), {"color_bgr": np.zeros((48, 64, 3), dtype=np.uint8)})()

    ready = step04_base_frame._TagView(np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32), RigidTransform(np.eye(4)), 0.4)
    poses = iter([None, step04_base_frame._Pose(RigidTransform(np.eye(4)), 0.4)])
    monkeypatch.setattr(step04_base_frame, "_inspect_tag", lambda *_args, **_kwargs: ready)
    monkeypatch.setattr(step04_base_frame, "_try_read_arm_radians", lambda *_args, **_kwargs: zeros)
    monkeypatch.setattr(step04_base_frame, "_read_ticks", lambda *_args, **_kwargs: {name: 2048 for name in ARM_JOINTS})
    monkeypatch.setattr(step04_base_frame, "_average_tag_pose_from_camera", lambda *_args, **_kwargs: next(poses))
    saved = step04_base_frame._run_manual_capture(
        args,
        Bus(),
        Camera(),
        servo_map,
        model,
        intrinsics,
        wait_key=lambda _delay: next(keys),
        show=lambda *_args, **_kwargs: None,
    )
    shown = capsys.readouterr().out
    assert saved == 1
    assert "Not enough valid frames" in shown
    assert "Total poses collected: 1" in shown
    payload = yaml.safe_load(samples.read_text(encoding="utf-8"))
    assert len(payload["samples"]) == 1


def test_self_check_joints_and_zero(tmp_path):
    assert main(["self-check"]) == 0
    assert main(["joints"]) == 0
    stops = {
        name: {"stop_min_deg": -90, "stop_max_deg": 90, "reading_deg": 0}
        for name in ARM_JOINTS
    }
    stops["wrist_roll"] = {"stop_min_deg": -157.2, "stop_max_deg": 162.8, "reading_deg": 0}
    path = tmp_path / "stops.yaml"
    path.write_text(yaml.safe_dump(stops), encoding="utf-8")
    assert main(["zero", "--stops", str(path)]) == 0


def test_solve_recovers_camera_pose_from_fk_samples(tmp_path):
    model = load_urdf("assets/so101/so101_new_calib.urdf")
    base_from_camera = RigidTransform.from_rotation_translation(np.eye(3), [0.15, -0.2, 0.75])
    mount = RigidTransform.from_rotation_translation(np.eye(3), [0.0, 0.0, 0.03])
    samples = []
    for index in range(8):
        joints = [
            -0.9 + 0.22 * index,
            -0.5 + 0.12 * index,
            -0.4 + 0.08 * index,
            0.6 - 0.15 * index,
            -1.0 + 0.25 * index,
        ]
        gripper = forward_kinematics(model, dict(zip(ARM_JOINTS, joints)))
        tag = RigidTransform(np.linalg.inv(base_from_camera.matrix) @ gripper.matrix @ mount.matrix)
        samples.append({"joints_rad": joints, "T_overhead_tag": tag.matrix.tolist(), "reprojection_px": 0.1})
    samples_path = tmp_path / "samples.yaml"
    samples_path.write_text(
        yaml.safe_dump(
            {
                "urdf": "assets/so101/so101_new_calib.urdf",
                "root_link": "base_link",
                "tip_link": "gripper_link",
                "tag_xyz_m": [0.0, 0.0, 0.03],
                "tag_rpy_deg": [0, 0, 0],
                "samples": samples,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    extrinsics = tmp_path / "world.yaml"
    save_transform(
        extrinsics,
        RigidTransform(np.eye(4)),
        {"source_frame": "overhead", "destination_frame": "world"},
    )
    output = tmp_path / "T_base_world.yaml"
    assert (
        main(
            [
                "solve",
                "--samples-file",
                str(samples_path),
                "--extrinsics",
                str(extrinsics),
                "--output",
                str(output),
                "--holdout",
                "2",
            ]
        )
        == 0
    )
    payload = yaml.safe_load(output.read_text(encoding="utf-8"))
    recovered = np.array(payload["T_base_overhead"], dtype=np.float64)
    assert np.linalg.norm(recovered[:3, 3] - base_from_camera.translation) < 1e-4
    assert max(payload["holdout_translation_error_m"]) < 1e-4
