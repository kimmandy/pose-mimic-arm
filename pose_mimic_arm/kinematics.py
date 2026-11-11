"""Forward and inverse kinematics for a serial arm described by a URDF.

The chain is read straight from the URDF, so the solver and RViz always agree
on the geometry. IK uses damped least squares (Levenberg-Marquardt style):

    dq = J^T (J J^T + lambda^2 I)^-1 e

where e stacks the position error and a weighted orientation error. Damping
keeps the step bounded near singularities, for example with the arm fully
stretched, where a plain pseudo-inverse would jump.

Only numpy is needed, so the module is usable and testable without ROS.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np


def rot_rpy(r: float, p: float, y: float) -> np.ndarray:
    """URDF roll-pitch-yaw (fixed axes X, then Y, then Z) to a rotation matrix."""
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def rot_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues' formula: rotation by `angle` about the unit vector `axis`."""
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    C = 1.0 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def transform(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def quat_from_matrix(R: np.ndarray) -> np.ndarray:
    """Unit quaternion (w, x, y, z) of a rotation matrix, w >= 0."""
    t = np.trace(R)
    if t > 0:
        s = 2.0 * np.sqrt(t + 1.0)
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s,
                      (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
        q = np.empty(4)
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def rotation_error(R_target: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Axis-angle vector (world frame) that rotates R onto R_target.

    Computed through a quaternion so it stays well defined all the way to 180
    degrees, where the matrix-log formula divides by sin(angle) = 0.
    """
    q = quat_from_matrix(R_target @ R.T)
    v = q[1:]
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.zeros(3)
    return 2.0 * np.arctan2(n, q[0]) * v / n


@dataclass
class Joint:
    name: str
    kind: str              # "revolute" or "fixed"
    origin: np.ndarray     # 4x4 transform from parent link to joint frame
    axis: np.ndarray       # unit axis in the joint frame
    lower: float = -np.pi
    upper: float = np.pi


@dataclass
class IKResult:
    q: np.ndarray
    position_error: float      # metres
    orientation_error: float   # radians
    iterations: int
    converged: bool


class Chain:
    """The serial chain from `base` to `tip` in a URDF."""

    def __init__(self, joints: List[Joint]):
        self.joints = joints
        self.active = [j for j in joints if j.kind != "fixed"]
        self.names = [j.name for j in self.active]
        self.lower = np.array([j.lower for j in self.active])
        self.upper = np.array([j.upper for j in self.active])

    @property
    def dof(self) -> int:
        return len(self.active)

    @classmethod
    def from_urdf(cls, urdf: str, base: str = "base_link", tip: str = "tool0") -> "Chain":
        """Build a chain from a URDF file path or XML string."""
        root = ET.fromstring(urdf) if urdf.lstrip().startswith("<") else ET.parse(urdf).getroot()
        by_child = {}
        for j in root.findall("joint"):
            by_child[j.find("child").get("link")] = j
        chain, link = [], tip
        while link != base:
            if link not in by_child:
                raise ValueError(f"no joint chain from '{base}' to '{tip}'")
            j = by_child[link]
            chain.append(cls._parse_joint(j))
            link = j.find("parent").get("link")
        return cls(chain[::-1])

    @staticmethod
    def _parse_joint(j: ET.Element) -> Joint:
        kind = j.get("type")
        if kind not in ("revolute", "fixed", "continuous"):
            raise ValueError(f"joint '{j.get('name')}': type '{kind}' is not supported")
        o = j.find("origin")
        xyz = np.array([float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()])
        rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
        a = j.find("axis")
        axis = np.array([float(v) for v in (a.get("xyz") if a is not None else "1 0 0").split()])
        axis = axis / np.linalg.norm(axis)
        lim = j.find("limit")
        lower, upper = -np.pi, np.pi
        if kind == "revolute" and lim is not None:
            lower, upper = float(lim.get("lower", -np.pi)), float(lim.get("upper", np.pi))
        return Joint(j.get("name"), "fixed" if kind == "fixed" else "revolute",
                     transform(rot_rpy(*rpy), xyz), axis, lower, upper)

    # ------------------------------------------------------------------
    def frames(self, q: Sequence[float]):
        """World transform of every active joint frame, plus the tip transform."""
        q = np.asarray(q, dtype=float)
        T = np.eye(4)
        joint_frames = []
        i = 0
        for j in self.joints:
            T = T @ j.origin
            if j.kind != "fixed":
                joint_frames.append(T.copy())
                T = T @ transform(rot_axis(j.axis, q[i]), np.zeros(3))
                i += 1
        return joint_frames, T

    def fk(self, q: Sequence[float]) -> np.ndarray:
        """4x4 transform of the tip for joint angles q."""
        return self.frames(q)[1]

    def link_points(self, q: Sequence[float]) -> np.ndarray:
        """Positions of the base, every joint and the tip (for drawing)."""
        pts = [np.zeros(3)]
        T = np.eye(4)
        i = 0
        q = np.asarray(q, dtype=float)
        for j in self.joints:
            T = T @ j.origin
            pts.append(T[:3, 3].copy())
            if j.kind != "fixed":
                T = T @ transform(rot_axis(j.axis, q[i]), np.zeros(3))
                i += 1
        return np.array(pts)

    def jacobian(self, q: Sequence[float]) -> np.ndarray:
        """6xN geometric Jacobian: rows 0-2 linear, rows 3-5 angular velocity."""
        frames, T_tip = self.frames(q)
        p_tip = T_tip[:3, 3]
        J = np.zeros((6, self.dof))
        for i, (Tj, j) in enumerate(zip(frames, self.active)):
            z = Tj[:3, :3] @ j.axis
            J[:3, i] = np.cross(z, p_tip - Tj[:3, 3])
            J[3:, i] = z
        return J

    def clamp(self, q: np.ndarray) -> np.ndarray:
        return np.clip(q, self.lower, self.upper)

    def ik(self, position: Sequence[float], rotation: Optional[np.ndarray] = None,
           seed: Optional[Sequence[float]] = None, *, orientation: str = "full",
           orientation_weight: float = 0.3, damping: float = 0.05,
           max_iterations: int = 100, tolerance: float = 1e-4,
           max_step: float = 0.2) -> IKResult:
        """Joint angles that put the tip at `position` (and `rotation`, if given).

        orientation="full" matches the whole rotation. orientation="axis" only
        makes the tool's z axis point the same way as rotation's z axis and
        leaves the roll about it free, which gives the arm a larger workspace.

        `seed` should be the previous solution: starting there keeps the arm in
        the same configuration from frame to frame instead of flipping elbows.
        Joint limits are enforced by clamping after every step.
        """
        if orientation not in ("full", "axis"):
            raise ValueError("orientation must be 'full' or 'axis'")
        target_p = np.asarray(position, dtype=float)
        q = self.clamp(np.zeros(self.dof) if seed is None else np.array(seed, dtype=float))
        w = orientation_weight if rotation is not None else 0.0

        def orient_err(R):
            if rotation is None:
                return np.zeros(3)
            if orientation == "axis":
                return np.cross(R[:, 2], rotation[:, 2])
            return rotation_error(rotation, R)

        it = 0
        for it in range(1, max_iterations + 1):
            T = self.fk(q)
            e_p = target_p - T[:3, 3]
            e_r = orient_err(T[:3, :3])
            if np.linalg.norm(e_p) < tolerance and np.linalg.norm(e_r) < 10 * tolerance:
                return IKResult(q, float(np.linalg.norm(e_p)), float(np.linalg.norm(e_r)), it, True)
            J = self.jacobian(q)
            if w > 0:
                e = np.concatenate([e_p, w * e_r])
                J = np.vstack([J[:3], w * J[3:]])
            else:
                e, J = e_p, J[:3]
            # Damp hard while far away (big, safe steps), lightly when close,
            # so the last millimetre converges in a few steps instead of crawling.
            lam = damping * min(1.0, max(float(np.linalg.norm(e)) / 0.05, 0.02))
            dq = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(len(e)), e)
            n = np.linalg.norm(dq)
            if n > max_step:
                dq *= max_step / n
            q = self.clamp(q + dq)
        T = self.fk(q)
        e_p = float(np.linalg.norm(target_p - T[:3, 3]))
        e_r = float(np.linalg.norm(orient_err(T[:3, :3])))
        return IKResult(q, e_p, e_r, it, False)

    def ik_with_restart(self, position, rotation=None, seed=None, restart_seed=None,
                        **kwargs) -> IKResult:
        """ik() from `seed`, and if that fails, once more from `restart_seed`.

        A target outside the workspace can leave the previous solution pressed
        against a joint limit, from where the solver no longer finds its way
        back. Restarting from a known good pose (the home pose) recovers.
        """
        first = self.ik(position, rotation, seed, **kwargs)
        if first.converged or restart_seed is None:
            return first
        second = self.ik(position, rotation, restart_seed, **kwargs)
        return second if second.position_error < first.position_error else first
