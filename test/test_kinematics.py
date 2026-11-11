import os

import numpy as np
import pytest

from pose_mimic_arm.kinematics import Chain, rot_axis, rotation_error

URDF = os.path.join(os.path.dirname(__file__), "..", "urdf", "mimic_arm.urdf")
HOME = np.array([0.0, -0.1737, 1.8133, -0.069, 0.0, 0.0])
TOOL_FORWARD = rot_axis(np.array([0.0, 1.0, 0.0]), np.pi / 2)


@pytest.fixture(scope="module")
def chain():
    return Chain.from_urdf(URDF)


def test_chain_reads_six_joints_with_limits(chain):
    assert chain.names == [f"joint{i}" for i in range(1, 7)]
    assert chain.lower[2] == pytest.approx(-0.1) and chain.upper[2] == pytest.approx(2.6)


def test_zero_pose_points_straight_up(chain):
    # 0.10 + 0.08 + 0.40 + 0.35 + 0.06 + 0.06 + 0.05
    assert chain.fk(np.zeros(6))[:3, 3] == pytest.approx([0, 0, 1.10], abs=1e-9)


def test_home_pose_reaches_toward_the_person(chain):
    T = chain.fk(HOME)
    assert T[:3, 3] == pytest.approx([0.45, 0.0, 0.55], abs=1e-3)
    assert T[:3, 2] == pytest.approx([1.0, 0.0, 0.0], abs=1e-2)  # tool z -> +x


def test_jacobian_matches_finite_differences(chain):
    q = HOME + np.array([0.3, -0.2, 0.1, 0.4, -0.5, 0.2])
    J = chain.jacobian(q)
    eps = 1e-6
    for i in range(chain.dof):
        dq = np.zeros(6)
        dq[i] = eps
        dp = (chain.fk(q + dq)[:3, 3] - chain.fk(q - dq)[:3, 3]) / (2 * eps)
        assert J[:3, i] == pytest.approx(dp, abs=1e-6)


@pytest.mark.parametrize("angle", [0.0, 0.3, 1.5, 3.0, np.pi])
def test_rotation_error_is_axis_angle(angle):
    axis = np.array([1.0, 2.0, -0.5])
    axis /= np.linalg.norm(axis)
    e = rotation_error(rot_axis(axis, angle), np.eye(3))
    assert np.linalg.norm(e) == pytest.approx(angle, abs=1e-6)
    if angle > 0:
        assert abs(np.dot(e / np.linalg.norm(e), axis)) == pytest.approx(1.0, abs=1e-6)


def test_ik_round_trip(chain):
    rng = np.random.default_rng(42)
    for _ in range(100):
        q = chain.clamp(HOME + rng.uniform(-0.5, 0.5, 6))
        T = chain.fk(q)
        res = chain.ik(T[:3, 3], T[:3, :3], seed=HOME)
        assert res.converged
        assert res.position_error < 1e-3
        assert np.all(res.q >= chain.lower - 1e-9) and np.all(res.q <= chain.upper + 1e-9)


def test_ik_axis_mode_keeps_tool_pointing(chain):
    res = chain.ik([0.5, 0.2, 0.6], TOOL_FORWARD, seed=HOME, orientation="axis")
    assert res.converged
    assert chain.fk(res.q)[:3, 2] == pytest.approx([1, 0, 0], abs=1e-3)


def test_ik_unreachable_returns_best_effort(chain):
    res = chain.ik([2.0, 0.0, 0.5], seed=HOME)
    assert not res.converged
    reach = np.linalg.norm(chain.fk(res.q)[:3, 3] - [0, 0, 0.18])
    assert reach == pytest.approx(0.92, abs=0.02)  # fully stretched toward it


def test_restart_recovers_from_a_bad_seed(chain):
    target = chain.fk(HOME)
    bad = chain.clamp(np.array([3.0, 2.0, -0.1, 2.2, 2.2, 3.0]))
    plain = chain.ik(target[:3, 3], TOOL_FORWARD, seed=bad, orientation="axis", max_iterations=30)
    fixed = chain.ik_with_restart(target[:3, 3], TOOL_FORWARD, seed=bad, restart_seed=HOME,
                                  orientation="axis", max_iterations=30)
    assert fixed.converged
    assert fixed.position_error <= plain.position_error
