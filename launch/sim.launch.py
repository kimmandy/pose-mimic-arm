"""Camera (or synthetic person) -> pose_node -> ik_node -> RViz2.

    ros2 launch pose_mimic_arm sim.launch.py                    # no camera needed
    ros2 launch pose_mimic_arm sim.launch.py source:=webcam
    ros2 launch pose_mimic_arm sim.launch.py source:=video video_path:=/path/clip.mp4
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = get_package_share_directory("pose_mimic_arm")
    urdf_path = os.path.join(share, "urdf", "mimic_arm.urdf")
    with open(urdf_path, encoding="utf-8") as f:
        robot_description = f.read()

    def arg(name, default, doc):
        return DeclareLaunchArgument(name, default_value=default, description=doc)

    def val(name, kind):
        return ParameterValue(LaunchConfiguration(name), value_type=kind)

    return LaunchDescription([
        arg("source", "synthetic", "webcam, video or synthetic"),
        arg("camera_index", "0", "webcam index (/dev/videoN)"),
        arg("video_path", "", "video file for source:=video"),
        arg("arm", "right", "which of the person's arms to follow: right or left"),
        arg("gain", "0.6", "robot metres per metre of hand motion"),
        arg("show_window", "false", "also open an OpenCV window (press r to re-zero)"),
        arg("rviz", "true", "start RViz2"),

        Node(package="robot_state_publisher", executable="robot_state_publisher",
             parameters=[{"robot_description": robot_description}]),
        Node(package="pose_mimic_arm", executable="ik_node", output="screen",
             parameters=[{"urdf_path": urdf_path}]),
        Node(package="pose_mimic_arm", executable="pose_node", output="screen",
             parameters=[{
                 "source": val("source", str),
                 "camera_index": val("camera_index", int),
                 "video_path": val("video_path", str),
                 "arm": val("arm", str),
                 "gain": val("gain", float),
                 "show_window": val("show_window", bool),
             }]),
        Node(package="rviz2", executable="rviz2",
             arguments=["-d", os.path.join(share, "rviz", "sim.rviz")],
             condition=IfCondition(LaunchConfiguration("rviz"))),
    ])
