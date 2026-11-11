import numpy as np
import pytest

from pose_mimic_arm.filters import EMA, MotionSmoother, SmootherConfig, deadband, limit_step
from pose_mimic_arm.mapping import (FOLLOWING, LOST, WAITING, ArmMapper, ArmPose,
                                    MapperConfig, Workspace, parse_axis_map)

DT = 1 / 30
SHOULDER = np.array([-0.2, -0.4, 2.0])
REST = SHOULDER + np.array([0.0, 0.22, -0.30])


def pose(hand_offset=(0, 0, 0), shoulder_shift=(0, 0, 0)):
    s = SHOULDER + np.array(shoulder_shift)
    w = REST + np.array(hand_offset) + np.array(shoulder_shift)
    return ArmPose(s, (s + w) / 2, w)


def run(mapper, poses, t0=0.0):
    out = None
    for i, p in enumerate(poses):
        out = mapper.update(p, t0 + i * DT)
    return out


def follow(mapper, frames=70):
    """Hold still long enough to zero (2 s at 30 fps)."""
    return run(mapper, [pose()] * frames)


def test_parse_axis_map():
    M = parse_axis_map("-z,-x,-y")
    assert M @ np.array([1, 2, 3]) == pytest.approx([-3, -1, -2])
    with pytest.raises(ValueError):
        parse_axis_map("x,x,y")
    with pytest.raises(ValueError):
        parse_axis_map("x,y")


def test_waits_for_stillness_then_follows():
    m = ArmMapper()
    out = run(m, [pose()] * 30)
    assert out.state == WAITING and 0.4 < out.still_progress < 0.6
    out = follow(m)
    assert out.state == FOLLOWING
    assert out.target == pytest.approx(m.cfg.home)  # no jump at the start


def test_moving_restarts_the_still_timer():
    m = ArmMapper()
    frames = [pose((0.1 * np.sin(i / 5), 0, 0)) for i in range(120)]
    assert run(m, frames).state == WAITING


def test_mirror_mapping_directions():
    m = ArmMapper()
    follow(m)
    home = m.cfg.home
    # toward the camera -> robot reaches toward the person (+x)
    out = run(m, [pose((0, 0, -0.10))] * 60, t0=3.0)
    assert out.target[0] - home[0] == pytest.approx(0.6 * 0.10 - 0.6 * 0.007, abs=2e-3)
    # hand up (image -y) -> robot up (+z)
    m2 = ArmMapper()
    follow(m2)
    out = run(m2, [pose((0, -0.10, 0))] * 60, t0=3.0)
    assert out.target[2] > home[2] + 0.05
    # image right (+x) -> robot -y
    m3 = ArmMapper()
    follow(m3)
    out = run(m3, [pose((0.10, 0, 0))] * 60, t0=3.0)
    assert out.target[1] < home[1] - 0.05


def test_body_sway_does_not_move_robot():
    m = ArmMapper()
    follow(m)
    out = run(m, [pose(shoulder_shift=(0.15, 0.02, 0.1))] * 30, t0=3.0)
    assert out.target == pytest.approx(m.cfg.home, abs=1e-9)


def test_small_jitter_is_ignored():
    m = ArmMapper()
    follow(m)
    rng = np.random.default_rng(0)
    out = run(m, [pose(rng.normal(0, 0.002, 3)) for _ in range(60)], t0=3.0)
    assert np.linalg.norm(out.target - m.cfg.home) < 2e-3


def test_target_stays_in_workspace():
    m = ArmMapper()
    follow(m)
    for off in ([0, 0, -2.0], [2.0, -2.0, 0], [-2.0, 2.0, 2.0]):
        out = run(m, [pose(off)] * 90, t0=3.0)
        assert m.cfg.workspace.contains(out.target, tol=1e-3)


def test_lost_holds_then_rezeroes():
    m = ArmMapper()
    follow(m)
    moved = run(m, [pose((0, 0, -0.1))] * 40, t0=3.0).target
    out = run(m, [None] * 30, t0=5.0)       # 1 s gone
    assert out.state == LOST and out.target == pytest.approx(moved)
    out = run(m, [None] * 100, t0=6.0)      # > 3 s gone
    assert out.state == WAITING and out.target == pytest.approx(moved)


def test_comeback_after_short_loss_is_rate_limited():
    m = ArmMapper(MapperConfig(ema_alpha=1.0))
    follow(m)
    run(m, [None] * 10, t0=3.0)
    out = m.update(pose((0, 0, -0.3)), 3.5)  # hand moved 30 cm while unseen
    assert out.state == FOLLOWING
    assert out.target[0] - m.cfg.home[0] <= 0.6 * 0.05 + 1e-9  # one max_jump step


def test_workspace_clamp_reaches_intersection():
    ws = Workspace()
    for p in ([5, 5, 5], [-5, 0, 0], [0.6, 0.28, 0.78], [0.45, 0, 0.55]):
        assert ws.contains(ws.clamp(np.array(p, float)), tol=1e-3)


def test_filters():
    assert deadband(np.array([0.003, 0, 0]), 0.005) == pytest.approx([0, 0, 0])
    assert deadband(np.array([0.010, 0, 0]), 0.005) == pytest.approx([0.005, 0, 0])
    assert limit_step(np.zeros(3), np.array([1.0, 0, 0]), 0.1) == pytest.approx([0.1, 0, 0])
    ema = EMA(0.5)
    ema.update(np.zeros(3))
    assert ema.update(np.ones(3)) == pytest.approx([0.5] * 3)


def test_smoother_respects_speed_limit():
    cfg = SmootherConfig(window=1, max_speed=0.5, max_accel=100.0)
    s = MotionSmoother(DT, cfg)
    s.reset(np.zeros(3))
    prev = np.zeros(3)
    for _ in range(30):
        out = s.update(np.array([1.0, 0, 0]))
        assert np.linalg.norm(out - prev) <= 0.5 * DT + 1e-12
        prev = out


def test_smoother_respects_accel_limit():
    cfg = SmootherConfig(window=1, max_speed=10.0, max_accel=2.0)
    s = MotionSmoother(DT, cfg)
    s.reset(np.zeros(3))
    v_prev = np.zeros(3)
    prev = np.zeros(3)
    for _ in range(30):
        out = s.update(np.array([1.0, 0, 0]))
        v = (out - prev) / DT
        assert np.linalg.norm(v - v_prev) <= 2.0 * DT + 1e-9
        v_prev, prev = v, out
