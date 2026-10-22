from soarm_aimbot.steps.step03_table_frame import main


def test_self_check_does_not_need_a_camera():
    assert main(["self-check"]) == 0
