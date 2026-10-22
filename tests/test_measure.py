import numpy as np
import pytest

from soarm_aimbot.geometry.intrinsics import PinholeIntrinsics
from soarm_aimbot.geometry.measure import measure_block


def test_block_width_and_height_from_overhead_depth():
    model = PinholeIntrinsics(width=200, height=200, fx=400, fy=400, cx=100, cy=100)
    depth = np.full((200, 200), 0.85, dtype=np.float32)
    depth[80:120, 60:140] = 0.80
    measurement = measure_block(depth, model, top_roi=(60, 80, 80, 40), table_roi=(10, 10, 20, 20))
    # Left/right edges u=60 and u=139 at Z=0.80, fx=400, cx=100.
    expected_width = abs((139 - 100) * 0.80 / 400 - (60 - 100) * 0.80 / 400)
    assert measurement.height_m == pytest.approx(0.05, abs=1e-5)
    assert measurement.width_m == pytest.approx(expected_width, abs=1e-5)
    assert measurement.top_depth_m == pytest.approx(0.80, abs=1e-5)
    assert measurement.table_depth_m == pytest.approx(0.85, abs=1e-5)
