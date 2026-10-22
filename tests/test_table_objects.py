import numpy as np

from soarm_aimbot.geometry.intrinsics import PinholeIntrinsics
from soarm_aimbot.geometry.table_objects import fit_table_plane, segment_depth, segment_world_points
from soarm_aimbot.geometry.transforms import RigidTransform


def _grid(x0, x1, y0, y1, z, count):
    xs = np.linspace(x0, x1, count)
    ys = np.linspace(y0, y1, count)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    return np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, z)])


def _scene():
    table = _grid(-0.25, 0.25, -0.25, 0.25, 0.0, 26)
    left = _grid(-0.16, -0.10, 0.04, 0.10, 0.04, 11)
    right = _grid(0.10, 0.16, -0.12, -0.06, 0.06, 11)
    return np.vstack([table, left, right])


def test_plane_normal_points_up():
    plane = fit_table_plane(_scene(), distance_m=0.008, seed=1)
    assert plane.normal[2] > 0.99
    assert abs(plane.offset_m) < 0.002


def test_two_blocks_keep_separate_temporary_ids():
    result = segment_world_points(_scene(), min_points=30, cluster_m=0.015)
    assert len(result.objects) == 2
    assert [item.temporary_id for item in result.objects] == [0, 1]
    assert result.objects[0].position_m[0] < 0
    assert result.objects[1].position_m[0] > 0
    np.testing.assert_allclose(result.objects[0].position_m, [-0.13, 0.07, 0.04], atol=0.003)
    np.testing.assert_allclose(result.objects[1].height_above_table_m, 0.06, atol=0.003)


def test_ignore_box_removes_one_block():
    result = segment_world_points(
        _scene(),
        min_points=30,
        cluster_m=0.015,
        ignore_boxes=[(-0.20, 0.00, 0.01, -0.05, 0.15, 0.10)],
    )
    assert len(result.objects) == 1
    assert result.objects[0].position_m[0] > 0


def test_height_band_drops_the_table_and_a_tall_outlier():
    table = _grid(-0.2, 0.2, -0.2, 0.2, 0.0, 20)
    block = _grid(0.0, 0.04, 0.0, 0.04, 0.03, 10)
    ceiling = _grid(0.1, 0.14, 0.1, 0.14, 0.5, 8)
    result = segment_world_points(
        np.vstack([table, block, ceiling]),
        min_points=20,
        max_height_m=0.3,
    )
    assert len(result.objects) == 1
    assert result.objects[0].height_above_table_m < 0.05


def test_xy_window_drops_points_outside_the_table():
    table = _grid(-0.2, 0.2, -0.2, 0.2, 0.0, 20)
    inside = _grid(0.02, 0.06, 0.02, 0.06, 0.04, 8)
    outside = _grid(0.6, 0.66, 0.6, 0.66, 0.04, 8)
    result = segment_world_points(
        np.vstack([table, inside, outside]),
        min_points=20,
        xy_min=(-0.25, -0.25),
        xy_max=(0.25, 0.25),
    )
    assert len(result.objects) == 1
    assert result.objects[0].position_m[0] < 0.2


def test_one_outlier_does_not_stretch_the_percentile_box():
    table = _grid(-0.2, 0.2, -0.2, 0.2, 0.0, 20)
    block = _grid(0.0, 0.04, 0.0, 0.04, 0.04, 15)
    outlier = np.array([[0.055, 0.02, 0.04]])
    result = segment_world_points(
        np.vstack([table, block, outlier]),
        min_points=20,
        percentile=2.0,
        cluster_m=0.02,
    )
    assert len(result.objects) == 1
    assert result.objects[0].extent_m[0] < 0.05


def test_depth_image_roundtrip_looking_down():
    intrinsics = PinholeIntrinsics(width=160, height=120, fx=200, fy=200, cx=80, cy=60)
    rotation = np.diag([1.0, -1.0, -1.0])
    world_from_camera = RigidTransform.from_rotation_translation(rotation, [0.0, 0.0, 0.8])
    camera_from_world = world_from_camera.inverse()
    table = _grid(-0.15, 0.15, -0.12, 0.12, 0.0, 31)
    block = _grid(0.04, 0.08, -0.02, 0.02, 0.05, 13)
    depth = np.zeros((120, 160), dtype=np.float32)
    camera_points = camera_from_world.apply(np.vstack([table, block]))
    u = np.rint(intrinsics.cx + intrinsics.fx * camera_points[:, 0] / camera_points[:, 2]).astype(int)
    v = np.rint(intrinsics.cy + intrinsics.fy * camera_points[:, 1] / camera_points[:, 2]).astype(int)
    inside = (u >= 0) & (u < 160) & (v >= 0) & (v < 120)
    for x, y, z in zip(u[inside], v[inside], camera_points[inside, 2]):
        if depth[y, x] == 0 or z < depth[y, x]:
            depth[y, x] = z
    result = segment_depth(
        depth,
        intrinsics,
        world_from_camera,
        pixel_keep=None,
        min_points=20,
        cluster_m=0.012,
    )
    assert len(result.objects) == 1
    found = result.objects[0]
    assert abs(found.height_above_table_m - 0.05) < 0.008
    assert abs(found.position_m[0] - 0.06) < 0.01
    assert abs(found.position_m[1]) < 0.01
    masked = segment_depth(
        depth,
        intrinsics,
        world_from_camera,
        pixel_keep=np.zeros(depth.shape, dtype=bool),
        min_points=20,
    )
    assert masked.objects == []
