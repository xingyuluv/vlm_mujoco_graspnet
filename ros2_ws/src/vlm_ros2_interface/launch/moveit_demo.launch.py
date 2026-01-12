import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # 1. 获取官方 UR MoveIt 配置路径
    ur_moveit_pkg = get_package_share_directory('ur_moveit_config')
    
    # 2. 加载 MoveIt (使用官方提供的 ur_moveit.launch.py)
    # 我们使用 UR5e 的参数，并设置为仿真模式(use_fake_hardware=true)
    # 这样 MoveIt 不会去寻找真实的 UR 驱动，而是单纯发布轨迹
    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(ur_moveit_pkg, 'launch', 'ur_moveit.launch.py')),
        launch_arguments={
            'ur_type': 'ur5e',
            'launch_rviz': 'true',
            'use_fake_hardware': 'true',  # 使用虚拟硬件，避免连接真实手臂报错
        }.items()
    )

    return LaunchDescription([
        moveit_launch
    ])