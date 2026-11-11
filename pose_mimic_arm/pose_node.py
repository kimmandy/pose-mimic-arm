"""ROS 2 node: camera (or synthetic person) -> arm tracking -> robot target.

Publishes
  /mimic/target   geometry_msgs/PointStamped  tool target in base_link (m)
  /mimic/state    std_msgs/String             WAITING 40% / FOLLOWING / LOST
  /mimic/markers  visualization_msgs/MarkerArray  the person's arm, workspace
  /mimic/image    sensor_msgs/Image           camera view with the arm drawn
"""

from __future__ import annotations

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from geometry_msgs.msg import Point, PointStamped
from rclpy.node import Node
from std_msgs.msg import ColorRGBA, String
from visualization_msgs.msg import Marker, MarkerArray

from .mapping import FOLLOWING, LOST, WAITING, ArmMapper, MapperConfig, Workspace
from .pose_source import open_source

STATE_COLORS = {  # BGR for the image overlay
    WAITING: (60, 200, 255),
    FOLLOWING: (90, 220, 90),
    LOST: (80, 80, 255),
}


class PoseNode(Node):
    def __init__(self):
        super().__init__("pose_node")
        p = self.declare_parameter
        self.source_kind = p("source", "synthetic").value
        camera_index = p("camera_index", 0).value
        video_path = p("video_path", "").value
        arm = p("arm", "right").value
        fps = p("fps", 30.0).value
        self.frame_id = p("frame_id", "base_link").value
        self.show_window = p("show_window", False).value
        self.publish_image = p("publish_image", True).value
        # Show the camera like a mirror, as webcam apps do, so the person and
        # the robot (which faces them) move the same way on screen.
        self.mirror_view = p("mirror_view", True).value

        ws = Workspace(
            lower=np.array(p("workspace_lower", [0.30, -0.28, 0.35]).value),
            upper=np.array(p("workspace_upper", [0.60, 0.28, 0.78]).value),
            center=np.array(p("workspace_center", [0.17, 0.0, 0.18]).value),
            radius=p("workspace_radius", 0.72).value,
        )
        cfg = MapperConfig(
            home=np.array(p("home", [0.45, 0.0, 0.55]).value),
            gain=p("gain", 0.6).value,
            axis_map=p("axis_map", "-z,-x,-y").value,
            still_seconds=p("still_seconds", 2.0).value,
            still_tolerance=p("still_tolerance", 0.03).value,
            deadband=p("deadband", 0.007).value,
            max_jump=p("max_jump", 0.05).value,
            ema_alpha=p("ema_alpha", 0.4).value,
            lost_reset_seconds=p("lost_reset_seconds", 3.0).value,
            workspace=ws,
        )
        self.mapper = ArmMapper(cfg)
        self.source = open_source(self.source_kind, camera_index=camera_index,
                                  video_path=video_path, arm=arm, fps=fps)

        self.pub_target = self.create_publisher(PointStamped, "/mimic/target", 10)
        self.pub_state = self.create_publisher(String, "/mimic/state", 10)
        self.pub_markers = self.create_publisher(MarkerArray, "/mimic/markers", 10)
        self.pub_image = None
        if self.publish_image:
            from cv_bridge import CvBridge
            from sensor_msgs.msg import Image
            self.bridge = CvBridge()
            self.pub_image = self.create_publisher(Image, "/mimic/image", 2)

        self.last_state = None
        self.create_timer(1.0 / fps, self.tick)
        self.create_timer(1.0, self.publish_workspace)
        self.get_logger().info(
            f"source={self.source_kind} arm={arm}. Hold your arm still for "
            f"{cfg.still_seconds:.0f} s to start; press r in the window to re-zero.")

    def tick(self):
        frame = self.source.read()
        if frame is None:
            self.get_logger().info("end of video")
            raise SystemExit
        out = self.mapper.update(frame.pose, frame.t)
        stamp = self.get_clock().now().to_msg()

        msg = PointStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        msg.point = Point(x=float(out.target[0]), y=float(out.target[1]), z=float(out.target[2]))
        self.pub_target.publish(msg)

        label = out.state if out.state != WAITING else f"WAITING {out.still_progress:.0%}"
        self.pub_state.publish(String(data=label))
        if out.state != self.last_state:
            self.get_logger().info(f"state: {out.state}")
            self.last_state = out.state

        self.publish_arm(out.arm_in_robot, stamp)
        if frame.image is not None and (self.pub_image or self.show_window):
            self.show(frame.image, out, label, stamp)

    def show(self, img, out, label, stamp):
        import cv2
        if self.mirror_view:
            img = cv2.flip(img, 1)
        color = STATE_COLORS[out.state]
        cv2.rectangle(img, (0, 0), (img.shape[1], 44), (30, 30, 30), -1)
        cv2.putText(img, label, (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
        if out.state == WAITING:
            w = int(img.shape[1] * out.still_progress)
            cv2.rectangle(img, (0, 40), (w, 44), color, -1)
        t = out.target
        cv2.putText(img, f"target x {t[0]:.2f}  y {t[1]:+.2f}  z {t[2]:.2f} m",
                    (12, img.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (230, 230, 230), 1, cv2.LINE_AA)
        if self.pub_image:
            m = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
            m.header.stamp = stamp
            m.header.frame_id = "camera"
            self.pub_image.publish(m)
        if self.show_window:
            cv2.imshow("pose_mimic_arm", img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("r"):
                self.mapper.reset()
                self.get_logger().info("re-zero: hold still")
            elif key in (ord("q"), 27):
                raise SystemExit

    def publish_arm(self, arm, stamp):
        mk = Marker()
        mk.header.frame_id = self.frame_id
        mk.header.stamp = stamp
        mk.ns, mk.id = "human_arm", 0
        if arm is None:
            gone = []
            for i in (0, 1):
                d = Marker()
                d.header = mk.header
                d.ns, d.id, d.action = "human_arm", i, Marker.DELETE
                gone.append(d)
            self.pub_markers.publish(MarkerArray(markers=gone))
            return
        mk.type = Marker.LINE_STRIP
        mk.scale.x = 0.025
        mk.color = ColorRGBA(r=1.0, g=0.62, b=0.16, a=0.9)
        mk.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in arm]
        joints = Marker()
        joints.header = mk.header
        joints.ns, joints.id = "human_arm", 1
        joints.type = Marker.SPHERE_LIST
        joints.scale.x = joints.scale.y = joints.scale.z = 0.045
        joints.color = ColorRGBA(r=1.0, g=0.85, b=0.6, a=0.95)
        joints.points = mk.points
        self.pub_markers.publish(MarkerArray(markers=[mk, joints]))

    def publish_workspace(self):
        ws = self.mapper.cfg.workspace
        lo, hi = ws.lower, ws.upper
        c = [np.array([x, y, z]) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
        edges = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7),
                 (0, 4), (1, 5), (2, 6), (3, 7)]
        mk = Marker()
        mk.header.frame_id = self.frame_id
        mk.header.stamp = self.get_clock().now().to_msg()
        mk.ns, mk.id = "workspace", 0
        mk.type = Marker.LINE_LIST
        mk.scale.x = 0.004
        mk.color = ColorRGBA(r=0.6, g=0.7, b=0.9, a=0.5)
        mk.points = [Point(x=float(c[i][0]), y=float(c[i][1]), z=float(c[i][2]))
                     for e in edges for i in e]
        self.pub_markers.publish(MarkerArray(markers=[mk]))

    def destroy_node(self):
        self.source.close()
        if self.show_window:
            import cv2
            cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = PoseNode()
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
