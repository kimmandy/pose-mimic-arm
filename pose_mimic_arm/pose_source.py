"""Where arm poses come from: a webcam or video file via MediaPipe, or a
synthetic person for trying the pipeline without a camera.

Every source returns Frame objects with the arm in *camera coordinates*,
in metres: x to the right of the image, y down, z away from the camera.
MediaPipe's world landmarks already use these axes, so no depth camera is
needed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .mapping import ArmPose

# MediaPipe Pose landmark indices
_LANDMARKS = {
    "right": {"shoulder": 12, "elbow": 14, "wrist": 16},
    "left": {"shoulder": 11, "elbow": 13, "wrist": 15},
}


@dataclass
class Frame:
    t: float                                   # seconds
    pose: Optional[ArmPose]                    # None: no arm found
    image: Optional[np.ndarray] = None         # BGR, with the arm drawn on it
    pixels: Optional[Dict[str, Tuple[int, int]]] = None


def draw_arm(image: np.ndarray, pixels: Dict[str, Tuple[int, int]]):
    import cv2
    s, e, w = pixels["shoulder"], pixels["elbow"], pixels["wrist"]
    cv2.line(image, s, e, (40, 160, 255), 6, cv2.LINE_AA)
    cv2.line(image, e, w, (40, 160, 255), 6, cv2.LINE_AA)
    for p, r in ((s, 9), (e, 8), (w, 11)):
        cv2.circle(image, p, r, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(image, p, r, (40, 160, 255), 2, cv2.LINE_AA)


class MediaPipeSource:
    """Webcam index or video path -> arm poses from MediaPipe Pose."""

    def __init__(self, device, arm: str = "right", loop: bool = False,
                 model_complexity: int = 1, min_visibility: float = 0.5,
                 width: int = 1280, height: int = 720):
        import cv2
        import mediapipe as mp

        if arm not in _LANDMARKS:
            raise ValueError("arm must be 'right' or 'left'")
        self.cv2 = cv2
        self.ids = _LANDMARKS[arm]
        self.loop = loop
        self.min_visibility = min_visibility
        self.is_file = isinstance(device, str) and not device.isdigit()
        self.cap = cv2.VideoCapture(device if self.is_file else int(device))
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open camera or video: {device}")
        if not self.is_file:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and fps > 1 else 30.0
        self.frame_no = 0
        self.t0 = time.monotonic()
        self.pose = mp.solutions.pose.Pose(
            model_complexity=model_complexity, smooth_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def read(self) -> Optional[Frame]:
        ok, img = self.cap.read()
        if not ok and self.is_file and self.loop:
            self.cap.set(self.cv2.CAP_PROP_POS_FRAMES, 0)
            ok, img = self.cap.read()
        if not ok:
            return None
        # A video file plays on its own clock so results do not depend on CPU speed.
        t = self.frame_no / self.fps if self.is_file else time.monotonic() - self.t0
        self.frame_no += 1

        res = self.pose.process(self.cv2.cvtColor(img, self.cv2.COLOR_BGR2RGB))
        if not res.pose_world_landmarks or not res.pose_landmarks:
            return Frame(t, None, img)
        world = res.pose_world_landmarks.landmark
        norm = res.pose_landmarks.landmark
        if min(world[i].visibility for i in self.ids.values()) < self.min_visibility:
            return Frame(t, None, img)
        h, w = img.shape[:2]
        pts = {k: np.array([world[i].x, world[i].y, world[i].z]) for k, i in self.ids.items()}
        pixels = {k: (int(norm[i].x * w), int(norm[i].y * h)) for k, i in self.ids.items()}
        draw_arm(img, pixels)
        return Frame(t, ArmPose(pts["shoulder"], pts["elbow"], pts["wrist"]), img, pixels)

    def close(self):
        self.cap.release()
        self.pose.close()


def two_link_elbow(shoulder, wrist, upper, fore, pole):
    """Elbow position for a 2-link arm, bending toward `pole`."""
    d_vec = wrist - shoulder
    d = float(np.linalg.norm(d_vec))
    d = min(max(d, abs(upper - fore) + 1e-6), upper + fore - 1e-6)
    u = d_vec / (np.linalg.norm(d_vec) + 1e-12)
    along = (upper ** 2 - fore ** 2 + d ** 2) / (2 * d)
    h = np.sqrt(max(upper ** 2 - along ** 2, 0.0))
    p = pole - np.dot(pole, u) * u
    p /= np.linalg.norm(p) + 1e-12
    return shoulder + along * u + h * p


class SyntheticSource:
    """A scripted person, so the whole pipeline runs without a camera.

    Timeline (seconds): 0-2.5 hold still (the mapper zeroes here), then a slow
    circle in front of the body, a reach toward the camera and back, a quick
    wave, and from `dropout` the arm leaves the view for one second.
    """

    UPPER, FORE = 0.29, 0.27

    def __init__(self, fps: float = 30.0, noise: float = 0.003,
                 dropout: Optional[Tuple[float, float]] = (17.0, 18.0),
                 width: int = 640, height: int = 480, realtime: bool = True,
                 seed: int = 0):
        self.fps = fps
        self.noise = noise
        self.dropout = dropout
        self.w, self.h = width, height
        self.realtime = realtime
        self.rng = np.random.default_rng(seed)
        self.frame_no = 0
        self.t0 = time.monotonic()
        # The person's right shoulder appears on the image's left (x < 0).
        self.shoulder = np.array([-0.18, -0.45, 2.0])
        self.rest = self.shoulder + np.array([0.0, 0.22, -0.30])

    PERIOD = 24.0

    def hand_at(self, t: float) -> np.ndarray:
        t = t % self.PERIOD
        if t < 3.0:
            return self.rest.copy()
        s = t - 3.0
        if s < 6.0:                      # circle in the image plane
            a = 2 * np.pi * s / 6.0
            return self.rest + np.array([0.15 * np.sin(a), -0.15 * (1 - np.cos(a)), 0.0])
        s -= 6.0
        if s < 4.0:                      # reach toward the camera and back
            return self.rest + np.array([0.0, -0.05, -0.18 * np.sin(np.pi * s / 4.0)])
        s -= 4.0
        if s < 3.0:                      # raise and wave
            up = min(1.0, s / 0.8)
            return self.rest + np.array([0.12 * np.sin(2 * np.pi * 1.2 * s) * up,
                                         -0.32 * up, 0.05 * up])
        s -= 3.0
        return self.rest + np.array([0.0, -0.32 * max(0.0, 1 - s / 1.5), 0.05])

    def read(self) -> Optional[Frame]:
        if self.realtime:
            due = self.t0 + self.frame_no / self.fps
            delay = due - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        t = self.frame_no / self.fps
        self.frame_no += 1
        hand = self.hand_at(t)
        elbow = two_link_elbow(self.shoulder, hand, self.UPPER, self.FORE,
                               pole=np.array([-0.4, 1.0, 0.3]))  # elbow out and down
        jitter = lambda: self.rng.normal(0.0, self.noise, 3)  # noqa: E731
        pose = ArmPose(self.shoulder + jitter(), elbow + jitter(), hand + jitter())
        if self.dropout and self.dropout[0] <= (t % self.PERIOD) < self.dropout[1]:
            pose = None
        image, pixels = self._render(pose)
        return Frame(t, pose, image, pixels)

    def _project(self, p) -> Tuple[int, int]:
        f = 0.9 * self.w
        return (int(self.w / 2 + f * p[0] / p[2]), int(self.h / 2 + f * (p[1] + 0.45) / p[2]))

    def _render(self, pose: Optional[ArmPose]):
        try:
            import cv2
        except ImportError:
            return None, None
        img = np.full((self.h, self.w, 3), (46, 40, 36), np.uint8)
        P = self._project
        neck = np.array([0.0, -0.50, 2.0])
        hip = np.array([0.0, 0.05, 2.0])
        l_sh = np.array([0.18, -0.45, 2.0])
        body = (120, 120, 120)
        cv2.circle(img, P(neck + [0, -0.14, 0]), int(0.09 * 0.9 * self.w / 2.0), body, 3, cv2.LINE_AA)
        cv2.line(img, P(neck), P(hip), body, 4, cv2.LINE_AA)
        cv2.line(img, P(self.shoulder), P(l_sh), body, 4, cv2.LINE_AA)
        cv2.line(img, P(l_sh), P(l_sh + [0.02, 0.30, 0]), body, 4, cv2.LINE_AA)
        cv2.line(img, P(l_sh + [0.02, 0.30, 0]), P(l_sh + [0.03, 0.56, 0]), body, 4, cv2.LINE_AA)
        if pose is None:
            return img, None
        pixels = {"shoulder": P(pose.shoulder), "elbow": P(pose.elbow), "wrist": P(pose.wrist)}
        draw_arm(img, pixels)
        return img, pixels

    def close(self):
        pass


def open_source(kind: str, *, camera_index: int = 0, video_path: str = "",
                arm: str = "right", loop: bool = True, fps: float = 30.0):
    if kind == "webcam":
        return MediaPipeSource(str(camera_index), arm=arm)
    if kind == "video":
        if not video_path:
            raise ValueError("source 'video' needs video_path")
        return MediaPipeSource(video_path, arm=arm, loop=loop)
    if kind == "synthetic":
        return SyntheticSource(fps=fps)
    raise ValueError(f"unknown source '{kind}' (webcam, video or synthetic)")
