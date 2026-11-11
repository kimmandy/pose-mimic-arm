"""ROS 2 node: robot target -> smoothed motion -> IK -> /joint_states.

Subscribes /mimic/target (geometry_msgs/PointStamped, base_link, metres).
Publishes
  /joint_states   sensor_msgs/JointState   drives robot_state_publisher / RViz
  /mimic/tcp      geometry_msgs/PointStamped  where the tool actually is
  /mimic/markers  visualization_msgs/MarkerArray  target (red) and tool (green)

This node stands in for a real robot controller: the same /mimic/target
stream could feed a servo interface instead.
"""

from __future__ import annotations

import os
import time

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Point, PointStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from .filters import MotionSmoother, SmootherConfig
from .kinematics import Chain, rot_axis


class IKNode(Node):
    def __init__(self):
        super().__init__("ik_node")
        p = self.declare_parameter
        default_urdf = os.path.join(get_package_share_directory("pose_mimic_arm"),
                                    "urdf", "mimic_arm.urdf")
        urdf = p("urdf_path", default_urdf).value
        rate = p("rate", 30.0).value
        self.frame_id = p("frame_id", "base_link").value
        self.orientation = p("orientation", "axis").value
        self.home = np.array(p("home_joints", [0.0, -0.1737, 1.8133, -0.069, 0.0, 0.0]).value)
        self.timeout = p("target_timeout", 1.0).value
        smoother = SmootherConfig(
            window=p("smooth_window", 4).value,
            max_speed=p("max_speed", 0.8).value,
            max_accel=p("max_accel", 4.0).value,
        )

        self.chain = Chain.from_urdf(urdf)
        if len(self.home) != self.chain.dof:
            raise ValueError(f"home_joints needs {self.chain.dof} values")
        # The tool points at the person (+x); with orientation 'axis' its roll is free.
        self.tool_rotation = rot_axis(np.array([0.0, 1.0, 0.0]), np.pi / 2)
        self.q = self.home.copy()
        self.smoother = MotionSmoother(1.0 / rate, smoother)
        self.smoother.reset(self.chain.fk(self.q)[:3, 3])

        self.target = None
        self.target_time = 0.0
        self.stats = {"n": 0, "err": 0.0, "fail": 0, "ms": 0.0}

        self.create_subscription(PointStamped, "/mimic/target", self.on_target, 10)
        self.pub_joints = self.create_publisher(JointState, "/joint_states", 10)
        self.pub_tcp = self.create_publisher(PointStamped, "/mimic/tcp", 10)
        self.pub_markers = self.create_publisher(MarkerArray, "/mimic/markers", 10)
        self.create_timer(1.0 / rate, self.tick)
        self.create_timer(5.0, self.report)
        self.get_logger().info(f"{self.chain.dof}-axis arm from {urdf}, {rate:.0f} Hz")

    def on_target(self, msg: PointStamped):
        self.target = np.array([msg.point.x, msg.point.y, msg.point.z])
        self.target_time = time.monotonic()

    def tick(self):
        stamp = self.get_clock().now().to_msg()
        fresh = self.target is not None and time.monotonic() - self.target_time < self.timeout
        if fresh:
            cmd = self.smoother.update(self.target)
            t0 = time.perf_counter()
            res = self.chain.ik_with_restart(cmd, self.tool_rotation, seed=self.q,
                                             restart_seed=self.home,
                                             orientation=self.orientation)
            self.stats["ms"] += (time.perf_counter() - t0) * 1000
            self.stats["fail"] += not res.converged
            self.q = res.q
        tcp = self.chain.fk(self.q)[:3, 3]
        if fresh:
            self.stats["n"] += 1
            self.stats["err"] += float(np.linalg.norm(tcp - self.target))

        js = JointState()
        js.header.stamp = stamp
        js.name = self.chain.names
        js.position = [float(v) for v in self.q]
        self.pub_joints.publish(js)

        msg = PointStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        msg.point = Point(x=float(tcp[0]), y=float(tcp[1]), z=float(tcp[2]))
        self.pub_tcp.publish(msg)
        self.publish_markers(tcp, stamp)

    def publish_markers(self, tcp, stamp):
        markers = []
        for i, (pos, rgb) in enumerate(((self.target, (0.95, 0.25, 0.25)),
                                        (tcp, (0.25, 0.9, 0.35)))):
            if pos is None:
                continue
            m = Marker()
            m.header.frame_id = self.frame_id
            m.header.stamp = stamp
            m.ns, m.id = "ik", i
            m.type = Marker.SPHERE
            m.pose.position = Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.04 if i == 0 else 0.03
            m.color = ColorRGBA(r=rgb[0], g=rgb[1], b=rgb[2], a=0.85)
            markers.append(m)
        self.pub_markers.publish(MarkerArray(markers=markers))

    def report(self):
        s = self.stats
        if s["n"]:
            self.get_logger().info(
                f"tracking error {1000 * s['err'] / s['n']:.1f} mm avg, IK {s['ms'] / s['n']:.2f} ms, "
                f"{s['fail']} unreached of {s['n']}")
        self.stats = {"n": 0, "err": 0.0, "fail": 0, "ms": 0.0}


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = IKNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException, SystemExit):
        pass
    finally:
        # Ctrl+C in a terminal reaches both `ros2 launch` and the node, so a
        # second SIGINT can land during cleanup. Finish quietly either way.
        try:
            if node is not None:
                node.destroy_node()
            rclpy.try_shutdown()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
