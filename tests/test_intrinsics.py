import numpy as np

from soarm_aimbot.geometry.intrinsics import PinholeIntrinsics, load_intrinsics, save_intrinsics


def test_principal_point_lands_on_optical_axis():
    model = PinholeIntrinsics(width=640, height=480, fx=600, fy=500, cx=320, cy=240)
    point = model.backproject(320, 240, 1.25)
    assert np.allclose(point, (0.0, 0.0, 1.25))


def test_backproject_matches_pinhole_formula():
    model = PinholeIntrinsics(width=640, height=480, fx=400, fy=200, cx=100, cy=50)
    point = model.backproject(300, 150, 0.8)
    assert np.allclose(point, ((300 - 100) * 0.8 / 400, (150 - 50) * 0.8 / 200, 0.8))


def test_invalid_depth_stays_zero_in_image_backproject():
    model = PinholeIntrinsics(width=2, height=2, fx=10, fy=10, cx=0.5, cy=0.5)
    depth = np.array([[0.0, 2.0], [0.0, 0.0]], dtype=np.float32)
    points = model.backproject_depth(depth)
    assert np.all(points[0, 0] == 0)
    assert points[0, 1, 2] == 2.0


def test_yaml_roundtrip(tmp_path):
    model = PinholeIntrinsics(
        width=848,
        height=480,
        fx=430.5,
        fy=431.25,
        cx=424.0,
        cy=240.5,
        distortion=(0.1, -0.2, 0.0, 0.0, 0.01),
        depth_scale_m=0.001,
    )
    path = tmp_path / "overhead_intrinsics.yaml"
    save_intrinsics(path, model, {"alignment": "depth_to_color", "serial": "0123"})
    loaded, metadata = load_intrinsics(path)
    assert loaded == model
    assert metadata["alignment"] == "depth_to_color"
    assert metadata["serial"] == "0123"
