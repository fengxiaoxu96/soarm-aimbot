import cv2
import numpy as np

from soarm_aimbot.geometry.checkerboard import calibrate_checkerboard, checkerboard_object, split_holdout
from soarm_aimbot.geometry.rgb_model import BROWN, FISHEYE


def test_holdout_is_spread_across_the_sequence():
    calibration, holdout = split_holdout(20, 4)
    assert holdout == [2, 7, 12, 17]
    assert len(calibration) == 16
    assert set(calibration).isdisjoint(holdout)


def test_brown_calibration_recovers_a_known_camera():
    cols, rows = 9, 6
    board = checkerboard_object(cols, rows, 0.02)
    matrix = np.array([[600.0, 0.0, 320.0], [0.0, 590.0, 240.0], [0.0, 0.0, 1.0]])
    distortion = np.zeros((5, 1))
    object_points = []
    image_points = []
    for index in range(10):
        rotation = np.array([[0.15], [0.2 - 0.04 * index], [0.05]], dtype=np.float64)
        translation = np.array([[-0.02 + 0.005 * index], [0.0], [0.45]], dtype=np.float64)
        image, _ = cv2.projectPoints(board, rotation, translation, matrix, distortion)
        assert image[:, 0, 0].min() > 0 and image[:, 0, 0].max() < 640
        assert image[:, 0, 1].min() > 0 and image[:, 0, 1].max() < 480
        object_points.append(board)
        image_points.append(image)
    result = calibrate_checkerboard(object_points, image_points, (640, 480), BROWN, holdout=0)
    assert abs(result.model.fx - 600) < 1.0
    assert abs(result.model.fy - 590) < 1.0
    assert result.calibration_mean_error_px < 0.05


def test_fisheye_calibration_recovers_a_known_camera():
    board = checkerboard_object(9, 6, 0.02)
    matrix = np.array([[400.0, 0.0, 320.0], [0.0, 400.0, 240.0], [0.0, 0.0, 1.0]])
    distortion = np.array([[0.02], [0.0], [0.0], [0.0]])
    object_points = []
    image_points = []
    for index in range(10):
        rotation = np.array([[0.25], [0.2 - 0.04 * index], [0.0]], dtype=np.float64)
        translation = np.array([[0.0], [0.0], [0.40]], dtype=np.float64)
        image, _ = cv2.fisheye.projectPoints(board.reshape(1, -1, 3), rotation, translation, matrix, distortion)
        pixels = image.reshape(-1, 2)
        assert pixels[:, 0].min() > 0 and pixels[:, 0].max() < 640
        assert pixels[:, 1].min() > 0 and pixels[:, 1].max() < 480
        object_points.append(board)
        image_points.append(image)
    result = calibrate_checkerboard(object_points, image_points, (640, 480), FISHEYE, holdout=0)
    assert abs(result.model.fx - 400) < 1.0
    assert result.calibration_mean_error_px < 0.05
    ray = result.model.pixel_to_ray(result.model.cx, result.model.cy)
    assert np.allclose(ray, (0, 0, 1), atol=1e-3)
