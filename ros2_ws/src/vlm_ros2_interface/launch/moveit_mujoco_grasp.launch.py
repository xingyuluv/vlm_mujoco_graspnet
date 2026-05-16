import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    """
    启动 MoveIt2 + MuJoCo 集成的抓取系统
    使用 RRT* 算法进行路径规划
    """

    # 1. 启动 MoveIt2 (仅MoveIt2核心，不包含Gazebo仿真)
    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('ur5e_gripper_moveit_config'),
                'launch/ur5e_gripper_moveit.launch.py'
            )
        ),
        launch_arguments={
            'use_sim_time': 'false',  # 与MuJoCo同步
        }.items()
    )

    # 2. 启动 MuJoCo 核心 (集成 MoveIt2 路径规划)
    mujoco_core_node = Node(
        package='vlm_ros2_interface',
        executable='ros2_mujoco_core',
        name='mujoco_core',
        output='screen',
        parameters=[{'use_sim_time': False}]
    )

    # 3. 启动 VLM 管理器 (视觉语言模型)
    vlm_manager_node = Node(
        package='vlm_ros2_interface',
        executable='ros2_vlm_manager',
        name='vlm_manager',
        output='screen',
        parameters=[{'use_sim_time': False}]
    )

    return LaunchDescription([
        # MoveIt2 路径规划
        moveit_launch,

        # MuJoCo 仿真执行
        mujoco_core_node,

        # VLM 视觉处理
        vlm_manager_node,
    ])

if __name__ == '__main__':
    generate_launch_description()