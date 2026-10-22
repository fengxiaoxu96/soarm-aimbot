import numpy as np

from soarm_aimbot.geometry.transforms import (
    RigidTransform,
    average_transforms,
    load_transform,
    rotation_angle_deg,
    save_transform,
)


def test_inverse_puts_the_translated_origin_at_zero():
    rotation = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    translation = np.array([0.02, -0.03, 0.8])
    camera_from_tag = RigidTransform.from_rotation_translation(rotation, translation)
    origin = camera_from_tag.inverse().apply(translation)
    assert np.allclose(origin, 0.0, atol=1e-9)


def test_average_ignores_quaternion_sign_and_one_outlier_translation():
    rotation = np.eye(4)
    rotation[:3, :3] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    first = RigidTransform(rotation.copy())
    first.matrix[:3, 3] = [0.10, 0.0, 0.0]
    second = RigidTransform(rotation.copy())
    second.matrix[:3, 3] = [0.12, 0.0, 0.0]
    third = RigidTransform(rotation.copy())
    third.matrix[:3, 3] = [0.50, 0.0, 0.0]
    averaged = average_transforms([first, second, third])
    assert np.allclose(averaged.translation, [0.12, 0.0, 0.0])
    assert rotation_angle_deg(first, averaged) < 1e-6


def test_yaml_roundtrip_keeps_the_new_translation(tmp_path):
    transform = RigidTransform(np.eye(4))
    transform.matrix[:3, 3] = [0.1, 0.2, 0.8]
    path = tmp_path / "T_world_overhead.yaml"
    save_transform(path, transform, {"marker_id": 0, "translation_m": [9, 9, 9]})
    loaded, metadata = load_transform(path)
    assert np.allclose(loaded.matrix, transform.matrix)
    assert metadata["translation_m"] == [0.1, 0.2, 0.8]
    assert metadata["marker_id"] == 0
