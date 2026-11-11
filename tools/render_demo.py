#!/usr/bin/env python3
"""Render docs/demo_sim.gif: the synthetic person next to the simulated arm.

Runs the same pipeline as the ROS nodes (mapper -> smoother -> IK) offline,
without ROS or a display, and draws every few frames with matplotlib.

    python3 tools/render_demo.py                 # writes docs/demo_sim.gif
    python3 tools/render_demo.py --seconds 10 --out /tmp/demo.gif
"""

import argparse
import io
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from pose_mimic_arm.filters import MotionSmoother  # noqa: E402
from pose_mimic_arm.kinematics import Chain, rot_axis  # noqa: E402
from pose_mimic_arm.mapping import FOLLOWING, LOST, WAITING, ArmMapper  # noqa: E402
from pose_mimic_arm.pose_source import SyntheticSource  # noqa: E402

HOME = np.array([0.0, -0.1737, 1.8133, -0.069, 0.0, 0.0])
STATE_RGB = {WAITING: (1.0, 0.78, 0.24), FOLLOWING: (0.35, 0.86, 0.35), LOST: (1.0, 0.31, 0.31)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=19.0)
    ap.add_argument("--fps", type=float, default=10.0, help="GIF frame rate")
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "demo_sim.gif"))
    args = ap.parse_args()

    chain = Chain.from_urdf(os.path.join(ROOT, "urdf", "mimic_arm.urdf"))
    tool_rot = rot_axis(np.array([0.0, 1.0, 0.0]), np.pi / 2)
    src = SyntheticSource(realtime=False)
    mapper = ArmMapper()
    sim_hz = 30.0
    smoother = MotionSmoother(1.0 / sim_hz)
    smoother.reset(chain.fk(HOME)[:3, 3])
    q = HOME.copy()
    every = max(1, int(round(sim_hz / args.fps)))
    ws = mapper.cfg.workspace

    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(9.6, 4.2), dpi=90, facecolor="#1f2226")
    ax_img = fig.add_axes([0.01, 0.06, 0.47, 0.84])
    ax3d = fig.add_axes([0.44, -0.08, 0.6, 1.12], projection="3d")
    frames = []

    for i in range(int(args.seconds * sim_hz)):
        f = src.read()
        out = mapper.update(f.pose, f.t)
        cmd = smoother.update(out.target)
        q = chain.ik_with_restart(cmd, tool_rot, seed=q, restart_seed=HOME, orientation="axis").q
        if i % every:
            continue

        ax_img.clear()
        ax_img.imshow(f.image[:, ::-1, ::-1])  # mirrored, like a selfie preview
        ax_img.set_axis_off()
        label = out.state if out.state != WAITING else f"WAITING (hold still) {out.still_progress:.0%}"
        ax_img.set_title(f"camera   t = {f.t:4.1f} s", color="#c8ccd2", loc="left", pad=4)
        ax_img.text(10, 28, label, color=STATE_RGB[out.state], fontsize=12, weight="bold")

        ax3d.clear()
        ax3d.set_facecolor("#1f2226")
        pts = chain.link_points(q)
        ax3d.plot(pts[:, 0], pts[:, 1], pts[:, 2], "-", color="#e3e5e8", lw=7, solid_capstyle="round")
        ax3d.scatter(pts[1:-1, 0], pts[1:-1, 1], pts[1:-1, 2], s=40, color="#2e6bc6", depthshade=False)
        ax3d.scatter(*pts[-1], s=60, color="#f08c26", depthshade=False)
        ax3d.plot([0, 0], [0, 0], [0, 0.1], color="#555a61", lw=16)
        for g in np.linspace(-0.3, 0.3, 7):   # floor grid for depth
            ax3d.plot([-0.05, 0.7], [g, g], [0, 0], color="#3a3f46", lw=0.8)
        for g in np.linspace(-0.05, 0.7, 6):
            ax3d.plot([g, g], [-0.3, 0.3], [0, 0], color="#3a3f46", lw=0.8)
        ax3d.scatter(*out.target, s=50, color="#f04040", depthshade=False)
        lo, hi = ws.lower, ws.upper
        for x in (lo[0], hi[0]):
            for y in (lo[1], hi[1]):
                ax3d.plot([x, x], [y, y], [lo[2], hi[2]], color="#6b7a99", lw=0.6, alpha=0.6)
        for z in (lo[2], hi[2]):
            ax3d.plot([lo[0], hi[0], hi[0], lo[0], lo[0]], [lo[1], lo[1], hi[1], hi[1], lo[1]],
                      [z] * 5, color="#6b7a99", lw=0.6, alpha=0.6)
        ax3d.set_xlim(-0.05, 0.7)
        ax3d.set_ylim(-0.375, 0.375)
        ax3d.set_zlim(0.0, 0.75)
        ax3d.set_box_aspect((1, 1, 1))
        ax3d.view_init(elev=12, azim=20)  # roughly from where the person stands
        ax3d.set_axis_off()
        ax3d.text2D(0.08, 0.88, "simulated arm (facing you)   red dot: target",
                    transform=ax3d.transAxes, color="#c8ccd2")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
        buf.seek(0)
        frames.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE, colors=96))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    frames[0].save(args.out, save_all=True, append_images=frames[1:],
                   duration=int(1000 / args.fps), loop=0, optimize=True)
    print(f"wrote {args.out}: {len(frames)} frames, {os.path.getsize(args.out) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
