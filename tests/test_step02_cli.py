import pytest

from soarm_aimbot.cameras.rgb import choose_device
from soarm_aimbot.steps.step02_wrist_intrinsics import main


def test_self_check_does_not_need_a_camera():
    assert main(["self-check"]) == 0


def test_one_non_realsense_camera_is_selected():
    devices = [
        {"index": 0, "path": "/dev/video0", "name": "RealSense Depth Camera"},
        {"index": 4, "path": "/dev/video4", "name": "USB Camera"},
    ]
    assert choose_device(devices, None) == 4


def test_several_rgb_cameras_need_an_explicit_index():
    devices = [
        {"index": 2, "path": "/dev/video2", "name": "USB Camera"},
        {"index": 4, "path": "/dev/video4", "name": "HD Camera"},
    ]
    with pytest.raises(ValueError, match="--device"):
        choose_device(devices, None)


def test_explicit_device_must_exist():
    devices = [{"index": 2, "path": "/dev/video2", "name": "USB Camera"}]
    with pytest.raises(ValueError, match="video5"):
        choose_device(devices, 5)
