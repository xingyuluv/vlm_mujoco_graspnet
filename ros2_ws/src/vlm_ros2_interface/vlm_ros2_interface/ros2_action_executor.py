import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import numpy as np
import time
import os
import sys
import spatialmath as sm

# ================= 路径配置 =================
PROJECT_ROOT = '/home/xingyu/projects/VLM_Grasp_Interactive'
MANIPULATOR_PATH = os.path.join(PROJECT_ROOT, 'manipulator_grasp')

# 确保能导入自定义库
if MANIPULATOR_PATH not in sys.path:
    sys.path.insert(0, MANIPULATOR_PATH)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(1, PROJECT_ROOT)

try:
    from manipulator_grasp.arm.robot import UR5e
    from manipulator_grasp.arm.motion_planning import (
        JointParameter, QuinticVelocityParameter, TrajectoryParameter,
        TrajectoryPlanner, LinePositionParameter, OneAttitudeParameter,
        CartesianParameter
    )
    print("✅ 成功加载 manipulator_grasp 运动规划模块")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)
# ================================================================

class ActionExecutorNode(Node):
    def __init__(self):
        super().__init__('action_executor')
        self.get_logger().info('正在初始化执行器 (基于 manipulator_grasp 修复版)...')

        self.pose_sub = self.create_subscription(Float64MultiArray, '/robot/target_pose', self.grasp_callback, 10)
        self.cmd_pub = self.create_publisher(Float64MultiArray, '/robot/joint_cmd', 10)

        # 1. 初始化机器人模型
        self.robot = UR5e()
        # 基座位置必须与 scene0.xml 和 ur5_grasp_old_env.py 一致
        self.robot.set_base(sm.SE3(0.8, 0.6, 0.74).t)
        # 工具偏移 (2f85)
        robot_tool = sm.SE3.Trans(0.0, 0.0, 0.13) * sm.SE3.RPY(-np.pi / 2, -np.pi / 2, 0.0)
        self.robot.set_tool(robot_tool)

        # 初始姿态 (Home)
        self.home_q = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0]) 
        # 更新机器人内部状态
        self.robot.set_joint(self.home_q)

        # 记录当前状态
        self.current_q = self.home_q.copy()
        self.current_gripper = 0.0  # 0.0 = Open, 255.0 = Close

        self.get_logger().info('✅ 机器人模型初始化完成，等待指令...')

    def grasp_callback(self, msg):
        self.get_logger().info('收到抓取指令，开始规划...')
        data = np.array(msg.data)
        
        # 解析数据 (Rot 3x3 + Trans 3)
        R_raw = data[:9].reshape((3, 3))
        t_co = data[9:12]

        # 1. 坐标系转换 (使用 Rt 构造避免报错)
        # 相机参数 (需与 scene.xml 和 grasp_process_old.py 保持一致)
        n_wc = np.array([0.0, -1.0, 0.0])
        o_wc = np.array([-1.0, 0.0, -0.5])
        t_wc = np.array([0.85, 0.8, 1.6]) 

        # 计算相机旋转矩阵
        R_wc = sm.SO3.TwoVectors(x=n_wc, y=o_wc).R
        # 组合世界到相机的变换
        T_wc = sm.SE3.Rt(R_wc, t_wc)
        
        # 构造 T_co (Camera -> Object)
        # check=False 防止因为 GraspNet 输出的矩阵存在微小数值误差而报错
        T_co = sm.SE3.Rt(R_raw, t_co, check=False)
        
        # 计算 T_wo (World -> Object)
        T_wo = T_wc * T_co

        self.get_logger().info(f"目标世界坐标: {T_wo.t}")
        
        # 执行完整的动作序列
        self.execute_sequence_old_style(T_wo)

    def execute_sequence_old_style(self, T_wo):
        """
        完全复刻 grasp_process_old.py 中的 execute_grasp 流程
        """
        
        # 强制同步当前机器人状态到内部模型 (防止漂移)
        self.robot.set_joint(self.current_q)

        # === 1. 移动到预抓取位姿 (Joint Space) ===
        # 使用 Home 姿态作为过渡，防止直接插值碰到奇怪的角度
        q1 = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])
        self.plan_and_execute_joint(self.current_q, q1, time_duration=2.0)

        # === 2. 接近抓取位姿 (Cartesian) ===
        # T2 = T_wo 沿自身坐标系后退 15cm
        self.robot.set_joint(self.current_q) # 更新 FK
        T1 = self.robot.get_cartesian()
        T2 = T_wo * sm.SE3(-0.15, 0.0, 0.0) 
        
        self.get_logger().info("-> 接近物体...")
        self.plan_and_execute_cartesian(T1, T2, time_duration=2.0)

        # === 3. 执行抓取插入 (Cartesian) ===
        # 从 T2 直线走到 T3 (T_wo)
        T3 = T_wo
        self.get_logger().info("-> 插入...")
        self.plan_and_execute_cartesian(T2, T3, time_duration=1.5)

        # === 4. 闭合夹爪 ===
        self.get_logger().info("-> 抓取闭合...")
        self.operate_gripper(target_value=255.0, steps=50) # 255 = Close

        # === 5. 提起物体 (Cartesian) ===
        # T4 = T3 向上 0.25m
        T4 = sm.SE3.Trans(0.0, 0.0, 0.25) * T3
        self.get_logger().info("-> 提起...")
        self.plan_and_execute_cartesian(T3, T4, time_duration=2.0)

        # === 6. 水平移动 (Cartesian) ===
        # T5 = 固定点 [1.4, 0.3] + 保持 T4 的高度和姿态
        target_pos = np.array([1.4, 0.3, T4.t[2]])
        # check=False 避免构造报错
        T5 = sm.SE3.Rt(T4.R, target_pos, check=False)
        
        self.get_logger().info("-> 移动到放置区...")
        self.plan_and_execute_cartesian(T4, T5, time_duration=2.0)

        # === 7. 放置下降 (Cartesian) ===
        T6 = sm.SE3.Trans(0.0, 0.0, -0.1) * T5
        self.get_logger().info("-> 下降...")
        self.plan_and_execute_cartesian(T5, T6, time_duration=1.5)

        # === 8. 打开夹爪 ===
        self.get_logger().info("-> 释放...")
        self.operate_gripper(target_value=0.0, steps=50)

        # === 9. 抬起并复位 ===
        T7 = sm.SE3.Trans(0.0, 0.0, 0.1) * T6
        self.plan_and_execute_cartesian(T6, T7, time_duration=1.0)
        
        self.get_logger().info("-> 复位...")
        self.plan_and_execute_joint(self.current_q, self.home_q, time_duration=2.0)

        self.get_logger().info("✅ 任务完成")

    # ================= 核心工具函数 =================

    def plan_and_execute_joint(self, q_start, q_end, time_duration=1.0):
        """ 关节空间规划 """
        parameter = JointParameter(q_start, q_end)
        velocity_parameter = QuinticVelocityParameter(time_duration)
        trajectory_parameter = TrajectoryParameter(parameter, velocity_parameter)
        planner = TrajectoryPlanner(trajectory_parameter)
        self.run_planner(planner, time_duration)

    def plan_and_execute_cartesian(self, T_start, T_end, time_duration=1.0):
        """ 笛卡尔空间直线规划 """
        position_parameter = LinePositionParameter(T_start.t, T_end.t)
        
        # check=False 避免严格正交检查报错
        attitude_parameter = OneAttitudeParameter(sm.SO3(T_start.R, check=False), sm.SO3(T_end.R, check=False))
        
        cartesian_parameter = CartesianParameter(position_parameter, attitude_parameter)
        velocity_parameter = QuinticVelocityParameter(time_duration)
        trajectory_parameter = TrajectoryParameter(cartesian_parameter, velocity_parameter)
        planner = TrajectoryPlanner(trajectory_parameter)
        self.run_planner(planner, time_duration)

    def run_planner(self, planner, total_time):
        """
        替代原本 env loop 的核心函数
        """
        # 仿真步长 0.002s，与 Mujoco 保持一致以获得平滑轨迹
        dt = 0.002 
        steps = int(total_time / dt)
        
        for i in range(steps):
            t = i * dt
            planner_interpolate = planner.interpolate(t)
            
            if isinstance(planner_interpolate, np.ndarray):
                # 关节空间结果
                joint_cmd = planner_interpolate
                # 关键：手动更新 robot 对象的状态，以便下一帧计算正确
                self.robot.move_joint(joint_cmd) 
            else:
                # 笛卡尔空间结果 -> 调用 robot 的 IK
                # robot.move_cartesian 内部会调用 IK 并更新 self.robot.q
                self.robot.move_cartesian(planner_interpolate)
                joint_cmd = self.robot.get_joint()
            
            # 发布 ROS 指令
            self.publish_action(joint_cmd, self.current_gripper)
            
            # 控制发送频率，避免 Python 占用过高
            time.sleep(dt)
        
        # 更新最终状态
        self.current_q = self.robot.get_joint()

    def operate_gripper(self, target_value, steps=50):
        """ 逐步改变夹爪数值 """
        start_val = self.current_gripper
        for i in range(steps):
            val = start_val + (target_value - start_val) * (i / steps)
            self.publish_action(self.current_q, val)
            time.sleep(0.01)
        self.current_gripper = float(target_value)
        self.publish_action(self.current_q, self.current_gripper)

    def publish_action(self, joint, gripper):
        msg = Float64MultiArray()
        # 确保数据扁平化且为 float 类型
        msg.data = joint.tolist() + [float(gripper)]
        self.cmd_pub.publish(msg)

def main():
    rclpy.init()
    node = ActionExecutorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()