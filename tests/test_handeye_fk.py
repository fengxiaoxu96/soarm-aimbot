import numpy as np

from soarm_aimbot.geometry.handeye import solve_eye_to_hand, solutions_agree
from soarm_aimbot.geometry.transforms import RigidTransform
from soarm_aimbot.geometry.urdf_fk import (
    ARM_JOINTS,
    forward_kinematics,
    load_urdf,
    reading_to_urdf_radians,
)


URDF = "assets/so101/so101_new_calib.urdf"


def _random_pose(rng, translation_scale=0.3):
    axis = rng.normal(size=3)
    axis = axis / np.linalg.norm(axis)
    angle = rng.uniform(0.4, 1.4)
    rotation = _rodrigues(axis * angle)
    translation = rng.normal(size=3) * translation_scale
    return RigidTransform.from_rotation_translation(rotation, translation)


def _rodrigues(vector):
    import cv2

    return cv2.Rodrigues(np.asarray(vector, dtype=np.float64))[0]


def test_zero_pose_places_the_gripper_at_the_urdf_origin_chain():
    model = load_urdf(URDF)
    pose = forward_kinematics(model, {name: 0.0 for name in ARM_JOINTS})
    np.testing.assert_allclose(
        pose.translation,
        [0.29323402, -0.000176767, 0.234370327],
        atol=1e-6,
    )
    for name in ARM_JOINTS:
        joint = model.joints[name]
        assert joint.lower < 0 < joint.upper


def test_positive_shoulder_pan_yaws_the_arm():
    model = load_urdf(URDF)
    home = forward_kinematics(model, {name: 0.0 for name in ARM_JOINTS})
    turned = forward_kinematics(model, {**{name: 0.0 for name in ARM_JOINTS}, "shoulder_pan": 0.4})
    assert abs(turned.translation[0] - home.translation[0]) > 0.01 or abs(
        turned.translation[1] - home.translation[1]
    ) > 0.01
    assert abs(turned.translation[2] - home.translation[2]) < 1e-6


def test_midpoint_of_symmetric_stops_is_urdf_zero():
    assert reading_to_urdf_radians(-110, 110, 0) == 0
    assert abs(reading_to_urdf_radians(-110, 110, 10) - np.deg2rad(10)) < 1e-12


def test_eye_to_hand_recovers_a_known_camera_pose():
    rng = np.random.default_rng(2)
    base_from_camera = _random_pose(rng, 0.8)
    gripper_from_tag = _random_pose(rng, 0.05)
    base_from_gripper = [_random_pose(rng) for _ in range(8)]
    camera_from_tag = [
        RigidTransform(
            np.linalg.inv(base_from_camera.matrix) @ pose.matrix @ gripper_from_tag.matrix
        )
        for pose in base_from_gripper
    ]
    park = solve_eye_to_hand(base_from_gripper, camera_from_tag, method="park")
    andreff = solve_eye_to_hand(base_from_gripper, camera_from_tag, method="andreff")
    distance, angle, agreed = solutions_agree(park, andreff, translation_m=1e-6, rotation_deg=1e-3)
    assert agreed, (distance, angle)
    assert np.linalg.norm(park.base_from_camera.translation - base_from_camera.translation) < 1e-6
    assert np.linalg.norm(park.base_from_camera.rotation - base_from_camera.rotation) < 1e-6
