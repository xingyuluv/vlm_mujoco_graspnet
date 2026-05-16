#!/usr/bin/env python3
"""
简单测试 MoveIt2 路径规划功能
"""

import sys
import os

# 添加路径
PROJECT_ROOT = '/home/xingyu/projects/VLM_Grasp_Interactive'
vlm_path = os.path.join(PROJECT_ROOT, 'ros2_ws/src/vlm_ros2_interface')
sys.path.insert(0, vlm_path)
sys.path.insert(0, os.path.join(vlm_path, 'vlm_ros2_interface'))

import rclpy
from vlm_ros2_interface.moveit_planner_client import MoveItPlannerClient
import numpy as np

def test_moveit_basic():
    """基本 MoveIt2 连接和规划测试"""
    print("🚀 测试 MoveIt2 基本连接和规划...")

    rclpy.init()

    try:
        planner = MoveItPlannerClient()

        # 等待服务就绪
        print("等待 MoveIt2 服务...")
        import time
        time.sleep(2)

        print("✅ MoveIt2 客户端初始化成功")

        # 测试关节空间规划
        print("\n📐 测试关节空间规划 (RRT*)...")
        start_joints = np.array([0.0, -np.pi/2, np.pi/2, 0.0, -np.pi/2, 0.0])
        goal_joints = np.array([0.3, -np.pi/2, np.pi/2, 0.0, -np.pi/2, 0.0])

        print(f"起始关节: {start_joints}")
        print(f"目标关节: {goal_joints}")

        trajectory, success = planner.plan_joint_trajectory(start_joints, goal_joints, planning_time=5.0)

        if success:
            print("✅ 关节空间规划成功!")
            print(f"轨迹点数: {len(trajectory)}")
            print(f"轨迹形状: {trajectory.shape}")
            if len(trajectory) > 0:
                print(f"起始点: {trajectory[0]}")
                print(f"结束点: {trajectory[-1]}")
        else:
            print("❌ 关节空间规划失败")

        print("\n🎉 MoveIt2 RRT* 路径规划测试完成!")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if 'planner' in locals():
            planner.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    test_moveit_basic()