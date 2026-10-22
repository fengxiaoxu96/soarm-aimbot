import numpy as np

from soarm_aimbot.geometry.tracks import TrackFrame, TrackState
from soarm_aimbot.geometry.transforms import RigidTransform
from soarm_aimbot.geometry.urdf_fk import ARM_JOINTS, forward_kinematics, load_urdf
from soarm_aimbot.steps.aimbot_wrist import (
    INIT_POSE_DEG,
    aim_line_endpoint,
    init_step,
    limit_joint_step,
    main,
    resolve_track,
    solve_base_for_y,
    stage2_step,
)

URDF = "assets/so101/so101_new_calib.urdf"


def test_self_check():
    assert main(["self-check"]) == 0


def test_run_refuses_to_start_without_a_servo_port():
    assert main(["run", "--frames", "1", "--no-window"]) == 1


def test_joint_step_respects_the_speed_limit():
    stepped = limit_joint_step(np.zeros(5), np.ones(5), np.full(5, 2.0), 0.05)
    assert np.allclose(stepped, np.full(5, 0.1))


def test_resolve_track_keeps_the_id_while_it_is_missing():
    ready = _state(1, np.zeros(3), missed=0)
    other = _state(2, np.array([0.2, 0.0, 0.0]), missed=0)
    chosen, locked = resolve_track(TrackFrame(0.0, 0.05, [other, ready]), None, None)
    assert locked == 1
    assert chosen is not None and chosen.track_id == 1
    missed = _state(1, np.zeros(3), missed=2)
    chosen, locked = resolve_track(TrackFrame(0.1, 0.05, [missed, other]), None, locked)
    assert chosen is None and locked == 1


def test_aim_ray_hits_the_object_before_the_table():
    identity = RigidTransform(np.eye(4))
    ee = np.array([0.2, 0.0, 0.2])
    down = np.array([0.0, 0.0, -1.0])
    point, kind = aim_line_endpoint(ee, down, identity, 0.0, np.array([0.2, 0.0, 0.05]), 0.04)
    assert kind == "object"
    assert np.allclose(point, [0.2, 0.0, 0.09])
    point, kind = aim_line_endpoint(ee, down, identity, 0.0, np.array([0.5, 0.0, 0.05]), 0.04)
    assert kind == "table"
    assert np.allclose(point, [0.2, 0.0, 0.0])


def test_aim_ray_uses_the_world_table_height():
    base_from_world = RigidTransform.from_rotation_translation(np.eye(3), np.array([0.0, 0.0, -0.02]))
    point, kind = aim_line_endpoint(np.array([0.1, 0.1, 0.3]), np.array([0.0, 0.0, -1.0]), base_from_world, 0.01, None)
    assert kind == "table"
    assert np.allclose(point, [0.1, 0.1, -0.01])


def test_pan_solver_reaches_the_target_base_y():
    model = load_urdf(URDF)
    locked = np.deg2rad(np.asarray(INIT_POSE_DEG[1:]))
    target = _fk_y(model, np.concatenate([[-0.4], locked]))
    pan, _ = solve_base_for_y(model, locked, 0.0, target, 10.0, 1.0)
    assert pan is not None
    assert abs(_fk_y(model, np.concatenate([[pan], locked])) - target) < 1e-6
    unreachable, _ = solve_base_for_y(model, locked, 0.0, 5.0, 10.0, 1.0)
    assert unreachable is None


def test_stage2_only_moves_the_pan_joint():
    model = load_urdf(URDF)
    init_pose = np.deg2rad(np.asarray(INIT_POSE_DEG))
    locked = init_pose[1:]
    goal = forward_kinematics(model, dict(zip(ARM_JOINTS, np.concatenate([[0.3], locked])))).translation
    command = stage2_step(model, init_pose, _state(1, goal, missed=0), locked, RigidTransform(np.eye(4)), 1.0, 0.05)
    assert command.mode == "follow"
    assert np.allclose(command.joints_rad[1:], locked)
    assert abs(command.joints_rad[0] - 0.05) < 1e-9
    held = stage2_step(model, init_pose, None, locked, RigidTransform(np.eye(4)), 1.0, 0.05)
    assert held.mode == "hold" and np.allclose(held.joints_rad, init_pose)


def test_stage1_holds_once_the_initial_pose_is_reached():
    model = load_urdf(URDF)
    init_pose = np.deg2rad(np.asarray(INIT_POSE_DEG))
    command = init_step(model, init_pose, init_pose, 0.5, 0.05, None, RigidTransform(np.eye(4)))
    assert command.mode == "hold"
    moving = init_step(model, np.zeros(5), init_pose, 0.5, 0.05, None, RigidTransform(np.eye(4)))
    assert moving.mode == "init"
    assert np.all(np.abs(moving.joints_rad) <= 0.025 + 1e-12)


def _fk_y(model, joints):
    return float(forward_kinematics(model, dict(zip(ARM_JOINTS, joints))).translation[1])


def _state(track_id: int, position: np.ndarray, missed: int) -> TrackState:
    point = np.asarray(position, dtype=np.float64)
    measured = None if missed else point.copy()
    return TrackState(track_id, point, np.zeros(3), measured, True, 3, missed, 0.0)
