import cv2
import numpy as np

from soarm_aimbot.geometry.intrinsics import PinholeIntrinsics
from soarm_aimbot.geometry.marker import detect_marker, marker_object_points, pose_from_corners, render_marker


def test_upright_marker_has_x_right_and_y_toward_the_top_of_the_image():
    marker = render_marker("DICT_APRILTAG_36h11", 0, 180)
    canvas = np.full((420, 420), 255, np.uint8)
    canvas[120:300, 120:300] = marker
    corners = detect_marker(canvas, "DICT_APRILTAG_36h11", 0)
    assert corners is not None
    intrinsics = PinholeIntrinsics(420, 420, 700, 700, 210, 210)
    pose = pose_from_corners(corners, 0.08, intrinsics)
    tag_x = pose.rotation @ np.array([1.0, 0.0, 0.0])
    tag_y = pose.rotation @ np.array([0.0, 1.0, 0.0])
    tag_z = pose.rotation @ np.array([0.0, 0.0, 1.0])
    assert tag_x[0] > 0.9
    assert tag_y[1] < -0.9
    assert tag_z[2] < -0.9


def test_tag_points_roundtrip_into_the_table_frame():
    intrinsics = PinholeIntrinsics(640, 480, 500, 500, 320, 240)
    size_m = 0.1
    rotation = cv2.Rodrigues(np.array([0.2, -0.1, 0.4]))[0]
    translation = np.array([0.03, -0.04, 0.7])
    image, _ = cv2.projectPoints(
        marker_object_points(size_m),
        cv2.Rodrigues(rotation)[0],
        translation.reshape(3, 1),
        intrinsics.camera_matrix,
        intrinsics.distortion_coeffs,
    )
    pose = pose_from_corners(image.reshape(4, 2), size_m, intrinsics)
    recovered = pose.inverse().apply(pose.apply(marker_object_points(size_m)))
    assert np.allclose(recovered, marker_object_points(size_m), atol=1e-4)
