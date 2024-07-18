#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # Get the path to the ArUco Pose package
    aruco_pose_package_dir = get_package_share_directory('aruco_pose')

    # Get the path to the RViz configuration file
    rviz_config_file = os.path.join(aruco_pose_package_dir, 'rviz', 'view.rviz')

    ld = LaunchDescription()

    pose_publisher = Node(
            package='aruco_pose',
            executable='pose_publisher',
            name='pose_publisher',
            output='screen'
        )
    
    rviz_node = Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config_file],
            output='screen'
        )

    ld.add_action(pose_publisher)
    ld.add_action(rviz_node)

    return ld