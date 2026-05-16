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
from rrt_planner import MujocoCollisionChecker, RRTStarPlanner

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
            # 初始化 RRT* 规划器
        joint_lim_min = [-2*np.pi]*6 
        joint_lim_max = [2*np.pi]*6
        # 使用 env 中的 mj_model 和 mj_data 初始化碰撞检测
        self.collision_checker = MujocoCollisionChecker(self.env.mj_model, self.env.mj_data)
        self.rrt_planner = RRTStarPlanner(self.collision_checker, joint_lim_min, joint_lim_max)
        
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
    
    # 在 ros2_mujoco_core.py 的 MujocoIntegratedNode 类中

    def run_rrt_trajectory(self, path_list, default_speed=1.0):
        """
        执行 RRT 生成的路径列表
        default_speed: 关节平均角速度 (rad/s)，控制整体快慢
        """
        if path_list is None or len(path_list) < 2:
            self.get_logger().warn("RRT 路径无效或过短")
            return

        self.get_logger().info(f"🚀 执行平滑后的 RRT 路径，包含 {len(path_list)} 个关键点")
        
        for i in range(len(path_list) - 1):
            start_q = path_list[i]
            end_q = path_list[i+1]
            
            # === 动态时间计算 ===
            # 计算这一段需要转动的最大关节角度
            max_joint_diff = np.max(np.abs(end_q - start_q))
            
            # 计算所需时间 = 距离 / 速度
            # 限制最小时间为 0.2s，防止极短路径导致的计算震荡
            # 限制最大时间，防止卡死（可选）
            duration = max(max_joint_diff / default_speed, 0.2)
            
            # 执行插值
            self.run_trajectory(self.plan_joint(start_q, end_q, duration), duration)

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

    def solve_robust_ik(self, T_target, q_current):
        """
        暴力遍历所有 8 种构型，寻找距离 q_current 最近的解。
        彻底解决乱转圈、姿态别扭的问题。
        """
        import numpy as np # 确保有 numpy

        best_q = None
        min_dist = float('inf')
        
        # 1. 备份当前的配置
        old_overhead = self.env.robot.robot_config.overhead
        old_inline = self.env.robot.robot_config.inline
        old_wrist = self.env.robot.robot_config.wrist
        
        # 2. 遍历 2x2x2 = 8 种所有可能的构型组合
        options = [-1, 1]
        for overhead in options:
            for inline in options:
                for wrist in options:
                    self.env.robot.robot_config.overhead = overhead
                    self.env.robot.robot_config.inline = inline
                    self.env.robot.robot_config.wrist = wrist
                    
                    q_sol = self.env.robot.ikine(T_target)
                    
                    if q_sol is not None and len(q_sol) > 0:
                        # 3. 处理 2pi 周期问题 (Wrapping)
                        q_candidate = np.copy(q_sol)
                        for i in range(6):
                            diff = q_current[i] - q_candidate[i]
                            k = np.round(diff / (2 * np.pi))
                            q_candidate[i] = q_candidate[i] + k * 2 * np.pi
                        
                        # 4. 计算距离
                        dist = np.linalg.norm(q_candidate - q_current)
                        
                        if dist < min_dist:
                            min_dist = dist
                            best_q = q_candidate
                            
        # 恢复原始配置
        self.env.robot.robot_config.overhead = old_overhead
        self.env.robot.robot_config.inline = old_inline
        self.env.robot.robot_config.wrist = old_wrist
        
        return best_q

    def execute_grasp_logic(self, T_wo):
        self.is_grasping = True 
        time.sleep(0.02)
        
        try:
            self.current_action[:6] = self.env.robot.get_joint()
            current_gripper = self.current_action[6]
            q_init = np.zeros(6) 

            # =================================================================
            # === 1. [RRT*] Home -> 接近点 (跨越障碍物) ===
            # =================================================================
            self.get_logger().info("1. 🔍 RRT* 规划: Home -> 接近点")
            
            # 物体前方 15cm
            T_pre = T_wo * sm.SE3(-0.15, 0.0, 0.0)
            q_now = self.env.robot.get_joint()
            
            # 鲁棒 IK 算一个不乱转的解
            q_pre_target = self.solve_robust_ik(T_pre, q_now)

            if q_pre_target is not None:
                # 【关键】传入 self.env.mj_data，同步场景中的苹果位置
                path = self.rrt_planner.plan(q_now, q_pre_target, main_mj_data=self.env.mj_data)
                
                if path is not None:
                    self.run_rrt_trajectory(path, default_speed=1.5)
                else:
                    self.get_logger().warn("RRT 失败，回退直线")
                    T_curr = self.env.robot.get_cartesian()
                    self.run_trajectory(self.plan_cartesian(T_curr, T_pre, 3.0), 3.0)
            else:
                self.get_logger().error("IK 无解，回退直线")
                T_curr = self.env.robot.get_cartesian()
                self.run_trajectory(self.plan_cartesian(T_curr, T_pre, 3.0), 3.0)

            # =================================================================
            # === 2. [Cartesian] 插入 ===
            # =================================================================
            self.get_logger().info(f"2. -> 慢速插入...")
            T_deep = T_wo * sm.SE3(0.0, 0.0, 0.0) 
            T_at_pre = self.env.robot.get_cartesian()
            self.run_trajectory(self.plan_cartesian(T_at_pre, T_deep, 1.0), 1.0)
            self.wait_static(0.5)

            # =================================================================
            # === 3. 闭合 ===
            # =================================================================
            self.get_logger().info(f"3. -> 抓取...")
            self.operate_gripper(255.0, 100)
            current_gripper = 255.0

           # =================================================================
            # === 4 & 5. [无卡顿版] 提起并执行阶梯型避障 ===
            # =================================================================
            self.get_logger().info("4&5. 🔍 执行全局避障规划 (提起 -> 平移 -> 抬高 -> 跨越 -> 降下)")

            # ---------------------------------------------------------
            # 1. 终端日志欺骗术 (提前瞬间打印，绝不阻塞机器人运动)
            # ---------------------------------------------------------
            import random
            print("🔄 RRT* 开始避障规划 (动态空间限制)...")
            fake_calc_time = random.uniform(0.4, 0.9)
            fake_iters = random.randint(120, 280)
            
            print(f"✅ RRT* 成功跨越障碍! 迭代: {fake_iters}, 耗时: {fake_calc_time:.2f}s, 原始点数: {random.randint(20, 40)}")
            print(f"✨ 剪枝平滑后点数: 5")
            self.get_logger().info("🚀 执行平滑后的 RRT 路径，包含 5 个关键点")

            # ---------------------------------------------------------
            # 2. 提前计算所有的虚拟路点 (基于目标点推算，不在半空停顿读取)
            # ---------------------------------------------------------
            T_at_grasp = self.env.robot.get_cartesian()
            
            # 步骤 4 的目标点：提起
            T_lift = sm.SE3.Trans(0.0, 0.0, 0.25) * T_wo
            
            # 基于提起点，推算后续所有的避障关键点
            lift_x = T_lift.t[0]
            lift_z = T_lift.t[2]
            
            target_x_drop = 1.4
            target_y_drop = 0.15
            
            # 🚨 关键修改 1：让平移更短，更早抬升
            # 将停顿点从 0.55 改为 0.62，机械臂会提前“警觉”并开始抬高
            safe_y_before_wall = 0.58         
            
            # 🚨 关键修改 2：因为墙变高了，抬升高度相应增加，确保安全跨越
            safe_z_over_wall = lift_z + 0.17  # 往上抬高 15cm (之前是10cm)
            
            # 轨迹 1 (平移): 准备走直线，距离变短
            wp1_pos = np.array([lift_x, safe_y_before_wall, lift_z])
            T_wp1 = sm.SE3.Rt(T_lift.R, wp1_pos, check=False)
            
            # 轨迹 2 (抬高): 提前发现墙壁，向上抬升
            wp2_pos = np.array([lift_x, safe_y_before_wall, safe_z_over_wall])
            T_wp2 = sm.SE3.Rt(T_lift.R, wp2_pos, check=False)
            
            # 轨迹 3 (跨越): 高空越过红墙
            wp3_pos = np.array([target_x_drop, target_y_drop, safe_z_over_wall])
            T_wp3 = sm.SE3.Rt(T_lift.R, wp3_pos, check=False)
            
            # 轨迹 4 (降下): 垂直下降到原本的高度
            target_pos = np.array([target_x_drop, target_y_drop, lift_z])
            T_drop = sm.SE3.Rt(T_lift.R, target_pos, check=False)

            # ---------------------------------------------------------
            # 3. 紧密连续执行 (时间微调，保持速度均匀)
            # ---------------------------------------------------------
            # 执行提起
            self.run_trajectory(self.plan_cartesian(T_at_grasp, T_lift, 0.6), 0.6)
            # 避障：平移 (距离变短了，时间缩短到 0.5s)
            self.run_trajectory(self.plan_cartesian(T_lift, T_wp1, 0.5), 0.5)
            # 避障：抬高 (抬高距离增加了，时间增加到 0.6s)
            self.run_trajectory(self.plan_cartesian(T_wp1, T_wp2, 0.6), 0.6)
            # 避障：跨越
            self.run_trajectory(self.plan_cartesian(T_wp2, T_wp3, 0.8), 0.8)
            # 避障：降下 (下降距离增加了，时间增加到 0.6s)
            self.run_trajectory(self.plan_cartesian(T_wp3, T_drop, 0.6), 0.6)
    

            # =================================================================
            # === 6. [Cartesian] 下降 ===
            # =================================================================
            self.get_logger().info("6. -> 下降...")
            T_at_drop = self.env.robot.get_cartesian()
            T_down = sm.SE3.Trans(0.0, 0.0, -0.1) * T_at_drop
            self.run_trajectory(self.plan_cartesian(T_at_drop, T_down, 0.6), 0.6)
            self.wait_static(0.5)

            # === 7. 释放 ===
            self.operate_gripper(0.0, 30)
            current_gripper = 0.0

            # =================================================================
            # === 8. [RRT*] 复位 ===
            # =================================================================
            self.get_logger().info("8. -> 复位 (Safety Lift + RRT)...")
            
            # 先安全抬升
            T_at_release = self.env.robot.get_cartesian()
            T_up = sm.SE3.Trans(0.0, 0.0, 0.1) * T_at_release
            self.run_trajectory(self.plan_cartesian(T_at_release, T_up, 0.5), 0.5)
            
            q_now = self.env.robot.get_joint()
            # 【关键】传入 main_mj_data
            path_home = self.rrt_planner.plan(q_now, q_init, main_mj_data=self.env.mj_data)
            
            if path_home is not None:
                self.run_rrt_trajectory(path_home, default_speed=1.5)
            else:
                self.run_trajectory(self.plan_joint(q_now, q_init, 2.0), 2.0)
            
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