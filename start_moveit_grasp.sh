#!/bin/bash

# MoveIt2集成抓取系统启动脚本
# 使用RRT*算法进行路径规划，只使用MuJoCo仿真（不启动Gazebo）

echo "🚀 启动 MoveIt2 + MuJoCo 集成抓取系统"
echo "======================================"
echo "注意: 只启动MoveIt2路径规划和MuJoCo仿真，不包含Gazebo"
echo ""

# 检查是否在正确的目录
if [ ! -d "ros2_ws" ]; then
    echo "❌ 错误: 请在项目根目录下运行此脚本"
    exit 1
fi

cd ros2_ws

# 构建项目 (如果需要)
echo "🔨 检查项目构建状态..."
if [ ! -d "install" ]; then
    echo "构建项目..."
    colcon build
    if [ $? -ne 0 ]; then
        echo "❌ 构建失败"
        exit 1
    fi
fi

# 启动集成系统
echo "🎯 启动 MoveIt2 + MuJoCo 抓取系统..."
source install/setup.bash
ros2 launch vlm_ros2_interface moveit_mujoco_grasp.launch.py

echo "✅ 系统已启动!"
echo ""
echo "📝 使用说明:"
echo "1. 在RViz中可以看到机器人模型"
echo "2. 在另一个终端运行: python3 ../demo_moveit_integration.py"
echo "3. 或者手动发布抓取目标: ros2 topic pub /robot/target_pose std_msgs/msg/Float64MultiArray '{data: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.4, 0.0, 0.1, 0.0]}'"
echo ""
echo "🛑 停止系统: Ctrl+C"