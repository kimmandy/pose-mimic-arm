from glob import glob

from setuptools import setup

package_name = "pose_mimic_arm"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/urdf", glob("urdf/*.urdf")),
        ("share/" + package_name + "/rviz", glob("rviz/*.rviz")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="pose_mimic_arm maintainers",
    maintainer_email="maintainer@example.com",
    description="A simulated arm that follows a person's arm seen by a webcam.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "pose_node = pose_mimic_arm.pose_node:main",
            "ik_node = pose_mimic_arm.ik_node:main",
        ],
    },
)
