#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory("wireandpipe_detection_cpp")

    # 参数文件路径
    default_params_file = os.path.join(pkg_share, "config", "params.yaml")

    params_file = LaunchConfiguration("params_file", default=default_params_file)

    declare_params_file = DeclareLaunchArgument(
        "params_file",
        default_value=default_params_file,
        description="Full path to the ROS parameters file for wire/pipe detection",
    )

    wireandpipe_detection_node = Node(
        package="wireandpipe_detection_cpp",
        executable="wireandpipe_detection_node",
        name="wireandpipe_detection_node",
        output="screen",
        parameters=[params_file],
        respawn=True,
        respawn_delay=10.0,
    )

    return LaunchDescription([declare_params_file, wireandpipe_detection_node])
