import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory
import xacro
import yaml

def load_file(package_name, file_path):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)
    try:
        with open(absolute_file_path, 'r') as file:
            return file.read()
    except EnvironmentError:
        return None

def load_yaml(package_name, file_path):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)
    try:
        with open(absolute_file_path, 'r') as file:
            return yaml.safe_load(file)
    except EnvironmentError:
        return None

def generate_launch_description():
    pkg_name = 'vlm_ros2_interface'

    # 1. 机器人描述 (URDF)
    xacro_file = os.path.join(get_package_share_directory(pkg_name), 'urdf', 'ur5e_robotiq.urdf.xacro')
    doc = xacro.process_file(xacro_file)
    robot_description_config = doc.toxml()
    robot_description = {'robot_description': robot_description_config}

    # 2. 语义描述 (SRDF)
    srdf_content = load_file(pkg_name, 'config/ur5e.srdf')
    robot_description_semantic = {'robot_description_semantic': srdf_content}

    # 3. 运动学求解器配置
    kinematics_yaml = {
        'manipulator': {
            'kinematics_solver': 'kdl_kinematics_plugin/KDLKinematicsPlugin',
            'kinematics_solver_search_resolution': 0.005,
            'kinematics_solver_timeout': 0.005,
        }
    }

    # 4. 控制器配置
    controllers_yaml = load_yaml(pkg_name, 'config/moveit_controllers.yaml')
    moveit_controllers = {
        'moveit_simple_controller_manager': controllers_yaml['moveit_simple_controller_manager'],
        'moveit_controller_manager': 'moveit_simple_controller_manager/MoveItSimpleControllerManager',
    }

    # 5. Planning Scene Monitor
    planning_scene_monitor_parameters = {
        'publish_planning_scene': True,
        'publish_geometry_updates': True,
        'publish_state_updates': True,
        'publish_transforms_updates': True,
    }

    # === 节点定义 ===

    # 【关键修复】Robot State Publisher (发布 /robot_description 和 TF)
    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[robot_description],
    )

    # Move Group Node (核心规划)
    run_move_group_node = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        output='screen',
        parameters=[
            robot_description,
            robot_description_semantic,
            kinematics_yaml,
            moveit_controllers,
            planning_scene_monitor_parameters,
            {'use_sim_time': False},
        ],
    )

    # RViz Node
    rviz_config_file = os.path.join(get_package_share_directory('ur_moveit_config'), 'config', 'moveit.rviz')
    
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='log',
        arguments=['-d', rviz_config_file],
        parameters=[
            robot_description,
            robot_description_semantic,
            kinematics_yaml,
        ],
    )

    # Static TF (World -> Base Link)
    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'world', 'base_link'],
    )

    return LaunchDescription([
        static_tf,
        node_robot_state_publisher, # <--- 必须包含这个
        run_move_group_node,
        rviz_node
    ])