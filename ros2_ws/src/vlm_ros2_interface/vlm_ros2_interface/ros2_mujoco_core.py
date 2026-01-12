import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray
from cv_bridge import CvBridge
import numpy as np
import os
import sys
import cv2
import threading
import time
import spatialmath as sm

# ================= 路径配置 =================
PROJECT_ROOT = '/home/xingyu/projects/VLM_Grasp_Interactive'
MANIPULATOR_PATH = os.path.join(PROJECT_ROOT, 'manipulator_grasp')
ARM_PARENT_PATH = os.path.join(MANIPULATOR_PATH, 'arm')

for p in [PROJECT_ROOT, MANIPULATOR_PATH, ARM_PARENT_PATH]:
    if p not in sys.path:
        sys.path.insert(0, p)
sys.path.append(os.path.join(MANIPULATOR_PATH))

try:
    from manipulator_grasp.env.ur5_grasp_old_env import UR5GraspEnv
    from manipulator_grasp.arm.motion_planning import (
        JointParameter, QuinticVelocityParameter, TrajectoryParameter,
        TrajectoryPlanner, LinePositionParameter, OneAttitudeParameter,
        CartesianParameter
    )
    print("✅ 成功加载环境与运动规划模块")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

class MujocoIntegratedNode(Node):
    def __init__(self):
        super().__init__('mujoco_ros_core')
        self.get_logger().info('正在启动集成版 MuJoCo 核心 (优化版) ..')
        
        # 1. 初始化环境
        self.env = UR5GraspEnv()
        self.env.reset()
        self.bridge = CvBridge()
        self.env_lock = threading.Lock()
        
        self.is_grasping = False
        self.current_action = np.zeros(7)
        self.current_action[:6] = self.env.robot_q 
        
        # 2. ROS 接口
        from rclpy.qos import qos_profile_sensor_data
        self.image_pub = self.create_publisher(Image, '/camera/color', qos_profile_sensor_data)
        self.depth_pub = self.create_publisher(Image, '/camera/depth', qos_profile_sensor_data)
        self.pose_sub = self.create_subscription(Float64MultiArray, '/robot/target_pose', self.grasp_callback, 10)
        
        # 3. 定时器
        # 【优化】将空闲物理步进频率从 500Hz (0.002) 降为 125Hz (0.008)
        # 这能大幅降低待机时的 CPU 占用和发热，且不影响视觉显示的流畅度
        self.timer = self.create_timer(0.008, self.idle_step_logic)
        self.img_timer = self.create_timer(0.033, self.publish_images)
        self.robot_ik = self.env.robot 

    def idle_step_logic(self):
        if not self.is_grasping:
            with self.env_lock:
                self.env.step(self.current_action)

    def publish_images(self):
        with self.env_lock:
            obs = self.env.render()
        if obs['img'] is not None:
            bgr = cv2.cvtColor(obs['img'], cv2.COLOR_RGB2BGR)
            img_msg = self.bridge.cv2_to_imgmsg(bgr, encoding="bgr8")
            self.image_pub.publish(img_msg)
            if obs['depth'] is not None:
                depth_msg = self.bridge.cv2_to_imgmsg(obs['depth'], encoding="32FC1")
                self.depth_pub.publish(depth_msg)

    def grasp_callback(self, msg):
        if self.is_grasping:
            self.get_logger().warn("正在执行抓取，忽略新指令")
            return

        self.get_logger().info("🚀 收到目标，启动抓取序列...")
        
        data = np.array(msg.data)
        # 解析基础位姿数据 (前12位)
        R_raw = data[:9].reshape((3, 3))
        t_co = data[9:12]
        
        # =================【新增】解析物体类型标志位 =================
        # 约定: data[12] == 1.0 表示是"薄平物体" (锤子/鼠标)
        # 如果发送端没有发这一位，默认为 False
        is_thin_object = False
        if len(data) >= 13:
            is_thin_object = (data[12] > 0.5)
            
        if is_thin_object:
            self.get_logger().info("🛠️ 检测到薄平物体(锤子/鼠标)，启用特殊抬升策略")
        else:
            self.get_logger().info("📦 检测到普通物体，启用标准策略")
        # ==========================================================
        
        # 1. 计算相机位姿
        n_wc = np.array([0.0, -1.0, 0.0])
        o_wc = np.array([-1.0, 0.0, -0.5])
        t_wc = np.array([0.85, 0.8, 1.6]) 
        T_wc = sm.SE3.Trans(t_wc) * sm.SE3(sm.SO3.TwoVectors(x=n_wc, y=o_wc))
        
        # 2. 计算物体位姿
        R_co = sm.SO3.TwoVectors(x=R_raw[:, 0], y=R_raw[:, 1])
        T_co = sm.SE3.Trans(t_co) * sm.SE3(R_co)
        
        # 3. 得到世界坐标系下的抓取位姿
        T_wo = T_wc * T_co
        
        # =================【关键修改】条件式高度修正 =================
        if is_thin_object:
            # 针对锤子/鼠标：向上抬高 2cm，防止夹爪铲地
            Z_HEIGHT_OFFSET = 0.020
            T_wo = sm.SE3.Trans(0, 0, Z_HEIGHT_OFFSET) * T_wo
            
            # 强力防撞限位 (2.5cm)
            TABLE_HEIGHT_LIMIT = 0.74 
            SAFE_MARGIN = 0.030 
            MIN_Z = TABLE_HEIGHT_LIMIT + SAFE_MARGIN

            current_z = T_wo.t[2]
            if current_z < MIN_Z:
                self.get_logger().warn(f"⚠️ 薄物体抓取点过低 ({current_z:.3f}m)，强制修正至安全高度 ({MIN_Z:.3f}m)")
                T_wo.t[2] = MIN_Z
        else:
            # 针对普通物体：不进行额外抬升，以免抓空
            # 但保留最基础的物理防穿模限位 (例如 0.5cm)
            TABLE_HEIGHT_LIMIT = 0.74
            BASIC_MARGIN = 0.005 # 5mm
            MIN_Z_BASIC = TABLE_HEIGHT_LIMIT + BASIC_MARGIN
            
            if T_wo.t[2] < MIN_Z_BASIC:
                 self.get_logger().info(f"🛡️ 触底保护: {T_wo.t[2]:.3f} -> {MIN_Z_BASIC:.3f}")
                 T_wo.t[2] = MIN_Z_BASIC
        # ==========================================================
        
        self.get_logger().info(f"目标世界坐标(最终): {T_wo.t}")
        
        threading.Thread(target=self.execute_grasp_logic, args=(T_wo,), daemon=True).start()

    def execute_grasp_logic(self, T_wo):
        self.is_grasping = True 
        time.sleep(0.02)
        
        try:
            self.current_action[:6] = self.env.robot.get_joint()
            current_gripper = self.current_action[6]

            # 初始全零姿态
            q_init = np.zeros(6) 
            q_pre_joint = np.array([0.0, 0.0, np.pi / 2, 0.0, -np.pi / 2, 0.0])

            # === 1. 移动到预抓取 (Joint Space) ===
            self.run_trajectory(self.plan_joint(self.current_action[:6], q_pre_joint, 0.7), 0.7)

            # === 2. 接近 (Cartesian) ===
            T_current = self.env.robot.get_cartesian()
            T_pre = T_wo * sm.SE3(-0.15, 0.0, 0.0)
            self.run_trajectory(self.plan_cartesian(T_current, T_pre, 0.6), 0.6)

            # === 3. 插入 (加深版 + 慢速) ===
            # 深度偏移: +1.5cm (沿抓取方向前进，让物体进得更深)
            T_deep = T_wo * sm.SE3(0.0, 0.0, 0.0) 
            self.get_logger().info(f"-> 慢速接近目标...")
            # 时间 2.5s，慢速插入
            self.run_trajectory(self.plan_cartesian(T_pre, T_deep, 1.0), 1.0)

            # === 等待稳定 ===
            self.get_logger().info(f"-> 等待稳定 (1.0s)...")
            self.wait_static(0.5)

            # === 4. 闭合 (慢速) ===
            self.get_logger().info(f"-> 慢慢闭合...")
            self.operate_gripper(255.0, 100)
            current_gripper = 255.0

            # === 5. 提起 ===
            self.get_logger().info(f"-> 提起...")
            T_lift = sm.SE3.Trans(0.0, 0.0, 0.25) * T_wo
            self.run_trajectory(self.plan_cartesian(T_deep, T_lift, 0.6), 0.6)

            # === 6. 移动到放置点 ===
            target_pos = np.array([1.4, 0.3, T_lift.t[2]])
            # check=False 防止矩阵正交误差报错
            T_drop = sm.SE3.Rt(T_lift.R, target_pos, check=False) 
            self.run_trajectory(self.plan_cartesian(T_lift, T_drop, 0.8), 0.8)

            # === 7. 下降 ===
            T_down = sm.SE3.Trans(0.0, 0.0, -0.1) * T_drop
            self.run_trajectory(self.plan_cartesian(T_drop, T_down, 0.6), 0.6)
            self.wait_static(0.5)

            # === 8. 释放 ===
            self.operate_gripper(0.0, 30)
            current_gripper = 0.0

            # === 9. 复位 ===
            T_up = sm.SE3.Trans(0.0, 0.0, 0.1) * T_down
            self.run_trajectory(self.plan_cartesian(T_down, T_up, 0.5), 0.5)
            
            # 回到全零状态
            q_now = self.env.robot.get_joint()
            self.run_trajectory(self.plan_joint(q_now, q_init, 0.8), 0.8)
            
            self.get_logger().info("✅ 抓取任务完成，已回零")

        except Exception as e:
            self.get_logger().error(f"抓取过程出错: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.current_action[:6] = self.env.robot.get_joint()
            self.current_action[6] = current_gripper
            self.is_grasping = False

    # ==================== 辅助函数 ====================
    
    def wait_static(self, duration):
        dt = 0.002
        steps = int(duration / dt)
        for _ in range(steps):
            with self.env_lock:
                self.env.step(self.current_action)
            # 这里也建议加上 sleep，虽然不加影响不大因为时间短
            time.sleep(dt)

    def run_trajectory(self, planner, total_time):
        dt = 0.002
        steps = int(total_time / dt)
        
        # 记录开始时间，用于更精确的同步（可选）
        # start_time = time.time()

        for i in range(steps):
            t = i * dt
            planner_interpolate = planner.interpolate(t)
            
            action = self.current_action.copy()
            if isinstance(planner_interpolate, np.ndarray):
                action[:6] = planner_interpolate
                self.env.robot.move_joint(action[:6]) 
            else:
                self.env.robot.move_cartesian(planner_interpolate)
                action[:6] = self.env.robot.get_joint()
            
            with self.env_lock:
                self.env.step(action)
            self.current_action = action
            
            # =================【关键修复】=================
            # 必须添加 sleep，否则 CPU 会单核 100% 满载空转
            # 这不仅能降温，还能让仿真速度与真实时间同步
            time.sleep(dt) 
            # ============================================

    def operate_gripper(self, target, steps):
        start = self.current_action[6]
        for i in range(steps):
            val = start + (target - start) * (i / steps)
            action = self.current_action.copy()
            action[6] = val
            with self.env_lock:
                self.env.step(action)
            self.current_action = action
            time.sleep(0.01)

    def plan_joint(self, q_start, q_end, duration):
        param = JointParameter(q_start, q_end)
        vel = QuinticVelocityParameter(duration)
        traj = TrajectoryParameter(param, vel)
        return TrajectoryPlanner(traj)

    def plan_cartesian(self, T_start, T_end, duration):
        pos = LinePositionParameter(T_start.t, T_end.t)
        # check=False 避免矩阵数值误差导致的报错
        att = OneAttitudeParameter(sm.SO3(T_start.R, check=False), sm.SO3(T_end.R, check=False))
        cart = CartesianParameter(pos, att)
        vel = QuinticVelocityParameter(duration)
        traj = TrajectoryParameter(cart, vel)
        return TrajectoryPlanner(traj)

def main():
    rclpy.init()
    node = MujocoIntegratedNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.env.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()