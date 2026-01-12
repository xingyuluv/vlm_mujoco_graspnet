import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data 
import cv2
import numpy as np
import threading
import time
import sys
import os
import spatialmath as sm
import roboticstoolbox as rtb 

# ================= 路径配置 =================
PROJECT_PATH = '/home/xingyu/projects/VLM_Grasp_Interactive'
if PROJECT_PATH not in sys.path: sys.path.append(PROJECT_PATH)
MANIPULATOR_PATH = os.path.join(PROJECT_PATH, 'manipulator_grasp')
if MANIPULATOR_PATH not in sys.path: sys.path.append(MANIPULATOR_PATH)

try:
    from vlm_process import choose_model, process_sam_results
    from grasp_process import detect_grasp 
except ImportError as e:
    sys.exit(1)

class DuckTestNode(Node):
    def __init__(self):
        super().__init__('duck_test_node')
        self.bridge = CvBridge()
        self.latest_rgb = None
        self.latest_depth = None
        self.image_lock = threading.Lock()
        
        # 1. 加载 IK Solver
        try: self.ik_solver = rtb.models.DH.UR5()
        except: self.ik_solver = rtb.DHRobot.UR5()
        
        # 设置基座 (必须与 scene0.xml 中 robot_mount 一致)
        self.ik_solver.base = sm.SE3([0.8, 0.6, 0.745])
        
        # Tool: 单位矩阵 (手动计算法兰盘位置)
        self.ik_solver.tool = sm.SE3() 
        
        # 手动定义 Tool 变换矩阵 (X轴=抓取方向)
        self.tool_matrix = sm.SE3.Trans(0.0, 0.0, 0.13) * sm.SE3.RPY(-np.pi / 2, -np.pi / 2, 0.0)
        self.tool_inv = self.tool_matrix.inv() 
        
        self.create_subscription(Image, '/camera/image_raw', self.rgb_callback, qos_profile_sensor_data)
        self.create_subscription(Image, '/camera/depth', self.depth_callback, qos_profile_sensor_data)
        self.command_pub = self.create_publisher(Float64MultiArray, '/robot/command', 10)
        
        # 【关键修复】初始化关节状态为 Home Pose，与仿真环境保持一致！
        # 之前是 zeros(6)，导致 IK 种子严重错误
        self.current_joints = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])
        
        self.get_logger().info('🦆 [状态同步修复版] 准备就绪...')

    def rgb_callback(self, msg):
        try:
            with self.image_lock: self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except: pass

    def depth_callback(self, msg):
        try:
            with self.image_lock: self.latest_depth = self.bridge.imgmsg_to_cv2(msg, '32FC1')
        except: pass

    def send_command(self, joints, gripper_val):
        msg = Float64MultiArray()
        msg.data = list(joints) + [float(gripper_val)]
        self.command_pub.publish(msg)
        self.current_joints = np.array(joints) 

    def execute_trajectory(self, q_list, gripper_val, duration=2.0):
        if len(q_list) == 0: return
        dt = duration / len(q_list)
        for q in q_list:
            self.send_command(q, gripper_val)
            time.sleep(dt)

    # ================= IK 解算 =================
    
    def solve_ik_with_tolerance(self, T_flange, seeds):
        """ 带公差的 IK 解算 """
        for seed in seeds:
            # 容差设为 1e-3 (1mm)，足以覆盖 UR5 vs UR5e 的误差
            sol = self.ik_solver.ik_LM(T_flange, q0=seed, ilimit=100, tol=1e-3)
            if sol[1]: 
                return sol[0]
        return None

    def auto_segment_duck(self, image_input):
        target_bbox = [120, 273, 205, 336]
        h, w = image_input.shape[:2]
        print(f"🖼️ 图像输入尺寸: {w}x{h}")
        
        image_rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
        try:
            predictor = choose_model()
            predictor.set_image(image_rgb)
            results = predictor(bboxes=[target_bbox])
            center, mask = process_sam_results(results)
            return mask
        except: return None

    def perform_grasp_sequence_robust(self, grasp_pose_matrix):
        GRIPPER_OPEN = 0.0
        GRIPPER_CLOSE = 255.0
        
        # 1. 验证基准 (FK Check)
        current_fk = self.ik_solver.fkine(self.current_joints)
        print(f"\n🔍 [FK自检] Solver认为当前法兰盘在:\n{current_fk.t}")
        # Home 姿态下，法兰盘应该在基座前方/上方附近，而不是原点

        # 2. 准备抓取姿态
        T_grasp_original = sm.SE3(grasp_pose_matrix, check=False)
        
        # 候选姿态
        candidates = [
            ("原始姿态 (正手)", T_grasp_original),
            ("翻转 180 (反手)", T_grasp_original * sm.SE3.Rx(np.pi)),
            ("侧抓 (旋90)", T_grasp_original * sm.SE3.Rz(np.pi/2)), 
        ]
        
        # 种子列表：必须包含当前的 Home 姿态，这是最可靠的种子
        q_home = self.current_joints.copy()
        q_vertical = np.array([0.0, -1.57, -1.57, -1.57, 1.57, 0.0]) # 垂直下插
        
        seeds = [self.current_joints, q_vertical]

        best_plan = None 

        print("\n🔍 正在搜索可行路径...")
        
        for name, T_grasp_cand in candidates:
            # Flange = Tool * Tool_inv
            T_flange_grasp = T_grasp_cand * self.tool_inv
            
            # Pre-Grasp: 沿 Tool -X 后退 15cm
            T_grasp_pre_tool = T_grasp_cand * sm.SE3.Tx(-0.15)
            T_flange_pre = T_grasp_pre_tool * self.tool_inv
            
            # 1. 解算 Pre-Grasp
            q_pre = self.solve_ik_with_tolerance(T_flange_pre, seeds)
            
            if q_pre is None:
                print(f"   ❌ {name}: Pre-Grasp 法兰盘位姿不可达")
                continue
                
            # 2. 模拟下探路径
            steps = 15
            traj_flange = rtb.tools.trajectory.ctraj(T_flange_pre, T_flange_grasp, steps)
            q_path = []
            last_q = q_pre
            path_valid = True
            
            for T in traj_flange:
                sol = self.ik_solver.ik_LM(T, q0=last_q, tol=1e-2) # 放宽容差到 1cm 保证连贯
                if not sol[1]:
                    path_valid = False
                    break
                q_path.append(sol[0])
                last_q = sol[0]
            
            if path_valid:
                print(f"   ✅ {name}: 路径验证成功！")
                best_plan = (q_pre, q_path, T_flange_grasp)
                break 
            else:
                print(f"   ⚠️ {name}: Pre-Grasp 可达但下探失败")

        if best_plan is None:
            self.get_logger().error("❌ 所有策略均失败。")
            return

        # ================= 执行 =================
        q_pre_final, q_path_approach, T_flange_final = best_plan

        # 1. Home (原地不动或微调)
        print(">>> 1. 移动到 Home...")
        traj = rtb.tools.trajectory.jtraj(self.current_joints, q_home, 50).q
        self.execute_trajectory(traj, GRIPPER_OPEN, duration=2.0)
        
        # 2. Pre-Grasp
        print(">>> 2. 移动到预抓取点...")
        traj = rtb.tools.trajectory.jtraj(q_home, q_pre_final, 50).q
        self.execute_trajectory(traj, GRIPPER_OPEN, duration=2.5)

        # 3. Approach
        print(">>> 3. 直线下探...")
        self.execute_trajectory(q_path_approach, GRIPPER_OPEN, duration=1.5)
        last_q = q_path_approach[-1]
        
        # 4. Grasp
        print(">>> 4. 抓取...")
        for g in np.linspace(GRIPPER_OPEN, GRIPPER_CLOSE, 20):
            self.send_command(last_q, g)
            time.sleep(0.05)
        time.sleep(0.5)

        # 5. Lift
        print(">>> 5. 提起...")
        T_flange_lift_mat = T_flange_final.A.copy()
        T_flange_lift_mat[2, 3] += 0.3 
        T_flange_lift = sm.SE3(T_flange_lift_mat, check=False)
        
        traj_cart = rtb.tools.trajectory.ctraj(T_flange_final, T_flange_lift, 40)
        q_path_lift = []
        for T in traj_cart:
            sol = self.ik_solver.ik_LM(T, q0=last_q, tol=1e-2)
            if sol[1]:
                q_path_lift.append(sol[0])
                last_q = sol[0]
                
        if len(q_path_lift) > 5:
            self.execute_trajectory(q_path_lift, GRIPPER_CLOSE, duration=2.0)
        else:
            print("⚠️ 提起规划失败")

        # 6. Return Home
        print(">>> 6. 回到 Home...")
        traj_home = rtb.tools.trajectory.jtraj(last_q, q_home, 50).q
        self.execute_trajectory(traj_home, GRIPPER_CLOSE, duration=2.5)
        print("✅ 任务完成")

    def main_logic(self):
        while self.latest_rgb is None: time.sleep(1.0)
        print("✅ Ready")
        while rclpy.ok():
            try:
                input("👉 Enter 开始抓取...")
                with self.image_lock:
                    rgb = self.latest_rgb.copy()
                    depth = self.latest_depth.copy()
                mask = self.auto_segment_duck(rgb)
                if mask is None: continue
                pose = detect_grasp(rgb, depth, mask, visualize=False) 
                if pose is None: continue
                self.perform_grasp_sequence_robust(pose)
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()

def main(args=None):
    rclpy.init(args=args)
    node = DuckTestNode()
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    try: node.main_logic()
    except: pass
    finally: node.destroy_node(); rclpy.shutdown(); os._exit(0)

if __name__ == '__main__':
    main()