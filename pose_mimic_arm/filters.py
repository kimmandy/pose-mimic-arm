"""Small signal filters for 3-D points. numpy only, no ROS."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np


class EMA:
    """Exponential moving average: y += alpha * (x - y).

    alpha near 1 follows quickly, alpha near 0 smooths heavily.
    """

    def __init__(self, alpha: float):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self.value: Optional[np.ndarray] = None

    def reset(self, value: Optional[np.ndarray] = None):
        self.value = None if value is None else np.array(value, dtype=float)

    def update(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self.value is None:
            self.value = x.copy()
        else:
            self.value = self.value + self.alpha * (x - self.value)
        return self.value.copy()


def deadband(v: np.ndarray, band: float) -> np.ndarray:
    """Zero inside `band`, then grow from zero, so there is no step at the edge."""
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    if n <= band:
        return np.zeros_like(v)
    return v * (n - band) / n


def limit_step(prev: np.ndarray, new: np.ndarray, max_step: float) -> np.ndarray:
    """Move from prev toward new by at most max_step (Euclidean)."""
    d = np.asarray(new, dtype=float) - prev
    n = float(np.linalg.norm(d))
    if n <= max_step or n == 0.0:
        return np.asarray(new, dtype=float).copy()
    return prev + d * (max_step / n)


@dataclass
class SmootherConfig:
    window: int = 4            # moving-average length, in control ticks
    max_speed: float = 0.8     # m/s
    max_accel: float = 4.0     # m/s^2


class MotionSmoother:
    """Turns a jumpy target stream into a motion a robot can follow.

    Each tick: average the last `window` targets, then limit how far the
    output may move (speed) and how much that move may differ from the
    previous one (acceleration). The limits are what keep a mis-detected
    frame from becoming a lunge.
    """

    def __init__(self, dt: float, config: SmootherConfig = SmootherConfig()):
        self.dt = dt
        self.cfg = config
        self.buf: deque = deque(maxlen=max(1, config.window))
        self.out: Optional[np.ndarray] = None
        self.vel = np.zeros(3)

    def reset(self, position: Optional[np.ndarray] = None):
        self.buf.clear()
        self.out = None if position is None else np.array(position, dtype=float)
        self.vel = np.zeros(3)

    def update(self, target: np.ndarray) -> np.ndarray:
        self.buf.append(np.asarray(target, dtype=float))
        goal = np.mean(self.buf, axis=0)
        if self.out is None:
            self.out = goal.copy()
            return self.out.copy()
        step = goal - self.out
        max_step = self.cfg.max_speed * self.dt
        n = float(np.linalg.norm(step))
        if n > max_step:
            step *= max_step / n
        max_dv = self.cfg.max_accel * self.dt * self.dt
        dv = step - self.vel * self.dt
        n = float(np.linalg.norm(dv))
        if n > max_dv:
            step = self.vel * self.dt + dv * (max_dv / n)
        self.vel = step / self.dt
        self.out = self.out + step
        return self.out.copy()
