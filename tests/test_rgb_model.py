import numpy as np

from soarm_aimbot.geometry.rgb_model import RgbCameraModel, load_rgb_intrinsics, save_rgb_intrinsics


def test_principal_ray_is_the_optical_axis():
    model = RgbCameraModel(640, 480, 500, 480, 320, 240, (0, 0, 0, 0, 0))
    ray = model.pixel_to_ray(320, 240)
    assert np.allclose(ray, (0, 0, 1), atol=1e-6)


def test_pixel_to_the_right_has_positive_x():
    model = RgbCameraModel(640, 480, 500, 500, 320, 240, (0.1, -0.05, 0.0, 0.0, 0.0))
    ray = model.pixel_to_ray(420, 240)
    assert ray[0] > 0
    assert abs(np.linalg.norm(ray) - 1) < 1e-6


def test_undistort_keeps_image_size():
    model = RgbCameraModel(32, 24, 40, 40, 16, 12, (0, 0, 0, 0, 0)).with_undistorted_pinhole()
    image = np.zeros((24, 32, 3), np.uint8)
    undistorted, new_k = model.undistort_image(image)
    assert undistorted.shape == image.shape
    assert new_k.shape == (3, 3)


def test_yaml_roundtrip(tmp_path):
    model = RgbCameraModel(
        1280,
        720,
        900.5,
        901.25,
        640.0,
        360.0,
        (0.1, -0.2, 0.0, 0.001, 0.0),
        undistorted_fx=890.0,
        undistorted_fy=891.0,
        undistorted_cx=640.0,
        undistorted_cy=360.0,
    )
    path = tmp_path / "wrist_intrinsics.yaml"
    save_rgb_intrinsics(path, model, {"accepted": True, "holdout_mean_error_px": 0.2})
    loaded, metadata = load_rgb_intrinsics(path)
    assert loaded == model
    assert metadata["accepted"] is True
    assert metadata["camera"] == "wrist"
