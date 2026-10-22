from soarm_aimbot.steps.step01_overhead_align import main


def test_self_check_does_not_need_a_camera():
    assert main(["self-check"]) == 0
