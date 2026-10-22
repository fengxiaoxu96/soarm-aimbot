import numpy as np

from soarm_aimbot.geometry.depth import spatial_median, temporal_median, valid_fraction


def test_spatial_median_removes_spike_and_keeps_holes():
    depth = np.ones((5, 5), dtype=np.float32)
    depth[2, 2] = 9.0
    depth[0, 0] = 0.0
    filtered = spatial_median(depth, 3)
    assert filtered[0, 0] == 0.0
    assert filtered[2, 2] == 1.0


def test_temporal_median_ignores_dropouts():
    first = np.array([[1.0, 0.0]], dtype=np.float32)
    second = np.array([[3.0, 4.0]], dtype=np.float32)
    third = np.array([[2.0, 0.0]], dtype=np.float32)
    fused = temporal_median([first, second, third])
    assert fused[0, 0] == 2.0
    assert fused[0, 1] == 4.0
    assert valid_fraction(fused) == 1.0
