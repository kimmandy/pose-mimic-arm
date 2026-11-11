"""Turn a tracked human arm into a target position for the robot's tool.

The mapper is a small state machine:

    WAITING    the person holds the arm still for `still_seconds`; that pose
               becomes the zero point, so following starts without a jump
    FOLLOWING  target = home + gain * M (hand - hand_at_zero), filtered
    LOST       no arm in view: hold the last target, and go back to WAITING
               after `lost_reset_seconds` so a person who steps back in
               somewhere else does not yank the robot across the room

Hand position is taken relative to the shoulder, so the person can shift on
their feet without moving the robot. Everything here is numpy only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from .filters import EMA, deadband, limit_step

WAITING, FOLLOWING, LOST = "WAITING", "FOLLOWING", "LOST"

_AXES = {"x": 0, "y": 1, "z": 2}


def parse_axis_map(spec: str) -> np.ndarray:
    """'-z,-x,-y' -> 3x3 matrix M with robot = M @ camera.

    Each entry names the camera axis (and sign) that drives robot x, y, z.
    """
    parts = [p.strip().lower() for p in spec.split(",")]
    if len(parts) != 3:
        raise ValueError(f"axis map needs three entries, got '{spec}'")
    M = np.zeros((3, 3))
    used = set()
    for row, p in enumerate(parts):
        sign = -1.0 if p.startswith("-") else 1.0
        axis = p.lstrip("+-")
        if axis not in _AXES or axis in used:
            raise ValueError(f"bad axis map '{spec}': use each of x, y, z once")
        used.add(axis)
        M[row, _AXES[axis]] = sign
    return M


@dataclass
class Workspace:
    """Box intersected with a sphere (the arm's reach)."""
    lower: np.ndarray = field(default_factory=lambda: np.array([0.30, -0.28, 0.35]))
    upper: np.ndarray = field(default_factory=lambda: np.array([0.60, 0.28, 0.78]))
    center: np.ndarray = field(default_factory=lambda: np.array([0.17, 0.0, 0.18]))
    radius: float = 0.72

    def clamp(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        for _ in range(4):  # alternate projections converge onto the intersection
            p = np.clip(p, self.lower, self.upper)
            d = p - self.center
            n = float(np.linalg.norm(d))
            if n > self.radius:
                p = self.center + d * (self.radius / n)
        return np.clip(p, self.lower, self.upper)

    def contains(self, p: np.ndarray, tol: float = 1e-6) -> bool:
        p = np.asarray(p, dtype=float)
        return bool(np.all(p >= self.lower - tol) and np.all(p <= self.upper + tol)
                    and np.linalg.norm(p - self.center) <= self.radius + tol)


@dataclass
class MapperConfig:
    home: np.ndarray = field(default_factory=lambda: np.array([0.45, 0.0, 0.55]))
    gain: float = 0.6                 # robot metres per human metre
    axis_map: str = "-z,-x,-y"        # mirror: see README
    still_seconds: float = 2.0
    still_tolerance: float = 0.03     # m the hand may wander while "still"
    deadband: float = 0.007           # m around the zero pose that is ignored
    max_jump: float = 0.05            # m the hand may move between two frames
    ema_alpha: float = 0.4
    lost_reset_seconds: float = 3.0
    workspace: Workspace = field(default_factory=Workspace)


@dataclass
class ArmPose:
    """Shoulder, elbow and wrist in camera coordinates (metres)."""
    shoulder: np.ndarray
    elbow: np.ndarray
    wrist: np.ndarray

    @property
    def hand(self) -> np.ndarray:
        return self.wrist - self.shoulder


@dataclass
class MapperOutput:
    state: str
    target: np.ndarray                 # robot tool target (always valid)
    still_progress: float = 0.0        # 0..1 while WAITING
    arm_in_robot: Optional[np.ndarray] = None  # 3x3: shoulder, elbow, wrist


class ArmMapper:
    def __init__(self, config: MapperConfig = MapperConfig()):
        self.cfg = config
        self.M = parse_axis_map(config.axis_map)
        self.ema = EMA(config.ema_alpha)
        self.reset()

    def reset(self):
        self.state = WAITING
        self.zero: Optional[np.ndarray] = None
        self.still_anchor: Optional[np.ndarray] = None
        self.still_since: Optional[float] = None
        self.still_samples = []
        self.prev_disp: Optional[np.ndarray] = None
        self.last_seen: Optional[float] = None
        self.target = self.cfg.workspace.clamp(self.cfg.home)
        self.ema.reset()

    def _to_robot(self, v_cam: np.ndarray) -> np.ndarray:
        return self.cfg.gain * (self.M @ v_cam)

    def _arm_drawing(self, pose: ArmPose, anchor: np.ndarray) -> np.ndarray:
        """The human arm drawn in robot space with its wrist at `anchor`."""
        pts = [pose.shoulder, pose.elbow, pose.wrist]
        return np.array([anchor + self._to_robot(p - pose.wrist) for p in pts])

    def update(self, pose: Optional[ArmPose], t: float) -> MapperOutput:
        """Feed one frame. `pose` is None when no arm was detected."""
        if pose is None:
            return self._on_lost(t)
        self.last_seen = t
        if self.state == LOST:
            self.state = FOLLOWING if self.zero is not None else WAITING
        if self.state == WAITING:
            return self._on_waiting(pose, t)
        return self._on_following(pose)

    def _on_lost(self, t: float) -> MapperOutput:
        if self.state == FOLLOWING:
            self.state = LOST
        if (self.state == LOST and self.last_seen is not None
                and t - self.last_seen > self.cfg.lost_reset_seconds):
            target = self.target
            self.reset()
            self.target = target   # stay where we are until the next zero
            return MapperOutput(WAITING, self.target.copy())
        if self.state == WAITING:
            self.still_since = None
        return MapperOutput(self.state, self.target.copy())

    def _on_waiting(self, pose: ArmPose, t: float) -> MapperOutput:
        hand = pose.hand
        if (self.still_anchor is None
                or np.linalg.norm(hand - self.still_anchor) > self.cfg.still_tolerance
                or self.still_since is None):
            self.still_anchor = hand.copy()
            self.still_since = t
            self.still_samples = []
        self.still_samples.append(hand)
        held = t - self.still_since
        progress = min(1.0, held / self.cfg.still_seconds)
        if held >= self.cfg.still_seconds:
            self.zero = np.mean(self.still_samples, axis=0)
            self.state = FOLLOWING
            self.prev_disp = np.zeros(3)
            self.ema.reset(self.target)
        return MapperOutput(WAITING if self.state == WAITING else FOLLOWING,
                            self.target.copy(), progress,
                            self._arm_drawing(pose, self.target))

    def _on_following(self, pose: ArmPose) -> MapperOutput:
        disp = deadband(pose.hand - self.zero, self.cfg.deadband)
        disp = limit_step(self.prev_disp, disp, self.cfg.max_jump)
        self.prev_disp = disp
        raw = self.cfg.home + self._to_robot(disp)
        self.target = self.cfg.workspace.clamp(self.ema.update(raw))
        return MapperOutput(FOLLOWING, self.target.copy(), 1.0,
                            self._arm_drawing(pose, self.target))
