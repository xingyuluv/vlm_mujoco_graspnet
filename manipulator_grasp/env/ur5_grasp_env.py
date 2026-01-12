import os.path
import sys

# 动态添加路径，确保在 ROS 节点中能找到 custom modules
current_dir = os.path.dirname(os.path.abspath(__file__))
manipulator_grasp_path = os.path.join(current_dir, '../../manipulator_grasp')
if os.path.exists(manipulator_grasp_path):
    sys.path.append(manipulator_grasp_path)
else:
    # 回退方案：根据你之前的路径设置
    sys.path.append('/home/xingyu/projects/VLM_Grasp_Interactive/manipulator_grasp')

import time
import numpy as np
import spatialmath as sm
import mujoco
import mujoco.viewer
import cv2

from manipulator_grasp.arm.robot import Robot, UR5e
from manipulator_grasp.utils import mj

class UR5GraspEnv:

    def __init__(self, render_mode='window'):
        self.sim_hz = 500
        self.render_mode = render_mode

        self.mj_model: mujoco.MjModel = None
        self.mj_data: mujoco.MjData = None
        self.robot: Robot = None
        self.joint_names = []
        
        # 定义初始姿态 (Home Pose)
        # 对应关节: shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3
        self.init_q = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0]) 
        self.robot_q = self.init_q.copy()
        
        # --- ROS 控制核心变量 ---
        # 存储期望的目标关节角度，step() 函数会不断尝试去达到这个角度
        self.target_q = self.init_q.copy()
        
        self.robot_T = sm.SE3()
        self.T0 = sm.SE3()

        self.mj_renderer: mujoco.Renderer = None
        self.mj_depth_renderer: mujoco.Renderer = None
        self.mj_viewer = None

        # [关键修复] 将分辨率改回 640x640 以匹配 Old 版本
        # 这确保了焦距计算 (Intrinsic) 与 Old 版本一致，从而保证点云坐标正确
        self.height = 640 
        self.width = 640 
        
        # 对应 scene.xml 里的 <camera name="cam" ... />
        self.camera_name = "cam"

    def reset(self):
        # 加载 XML 模型
        # 注意：这里我们尝试加载 scene0.xml (你的测试场景)，如果不存在则回退到 scene.xml
        scene_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'scenes', 'scene0.xml')
        if not os.path.exists(scene_path):
            print(f"⚠️ 未找到 scene0.xml，回退到 scene.xml")
            scene_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'scenes', 'scene.xml')
        
        if not os.path.exists(scene_path):
            raise FileNotFoundError(f"❌ 找不到场景文件: {scene_path}")

        self.mj_model = mujoco.MjModel.from_xml_path(scene_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        mujoco.mj_forward(self.mj_model, self.mj_data)

        # 初始化机器人对象
        self.robot = UR5e()
        try:
            base_pose = mj.get_body_pose(self.mj_model, self.mj_data, "ur5e_base").t
            self.robot.set_base(base_pose)
        except Exception as e:
            print(f"⚠️ 警告: 设置基座位置失败 (可能是XML命名不匹配): {e}")

        # 设置初始关节角度
        self.robot.set_joint(self.init_q)
        self.joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint",
                            "wrist_2_joint", "wrist_3_joint"]
        
        # 将初始角度写入物理引擎
        for i, jn in enumerate(self.joint_names):
            mj.set_joint_q(self.mj_model, self.mj_data, jn, self.init_q[i])
        
        # 重置目标为初始位置
        self.target_q = self.init_q.copy()
        
        mujoco.mj_forward(self.mj_model, self.mj_data)
        
        # 尝试附加夹爪约束 (适配 scene.xml 中的 weld constraint)
        try:
            mj.attach(self.mj_model, self.mj_data, "attach", "2f85", self.robot.fkine(self.init_q))
        except:
            pass

        # 设置工具坐标系 (与 Old 版本保持一致)
        # Tool X轴 = 接近方向
        robot_tool = sm.SE3.Trans(0.0, 0.0, 0.13) * sm.SE3.RPY(-np.pi / 2, -np.pi / 2, 0.0)
        self.robot.set_tool(robot_tool)
        self.robot_T = self.robot.fkine(self.init_q)
        self.T0 = self.robot_T.copy()

        # 初始化渲染器 (用于 ROS 发布图像)
        self.mj_renderer = mujoco.renderer.Renderer(self.mj_model, height=self.height, width=self.width)
        self.mj_depth_renderer = mujoco.renderer.Renderer(self.mj_model, height=self.height, width=self.width)
        self.mj_renderer.update_scene(self.mj_data, camera=self.camera_name)
        self.mj_depth_renderer.update_scene(self.mj_data, camera=self.camera_name)
        self.mj_depth_renderer.enable_depth_rendering()
        
        # 初始化可视化窗口 (仅当 render_mode='window' 时)
        if self.render_mode == 'window':
            self.mj_viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)
            # 设置默认观察视角
            self.mj_viewer.cam.lookat[:] = [1.3, 0.7, 0.8] 
            self.mj_viewer.cam.azimuth = 210
            self.mj_viewer.cam.elevation = -35
            self.mj_viewer.cam.distance = 1.2
            self.mj_viewer.sync()

    def close(self):
        if self.mj_viewer is not None:
            self.mj_viewer.close()
        if self.mj_renderer is not None:
            self.mj_renderer.close()
        if self.mj_depth_renderer is not None:
            self.mj_depth_renderer.close()
        cv2.destroyAllWindows()

    # --- ROS 专用接口 ---

    def goto_joints(self, target_joints):
        """
        [ROS 接口] 更新期望的关节角度。
        target_joints: list 或 np.array, 包含 6 个关节角
        """
        if len(target_joints) >= 6:
            self.target_q = np.array(target_joints[:6])
        else:
            print(f"❌ 错误: 目标关节数量不正确 {len(target_joints)}，需要 6 个")

    def get_observation(self):
        """
        [ROS 接口] 获取当前相机的 RGB 和 Depth 图像
        """
        try:
            # 1. 更新 RGB
            self.mj_renderer.update_scene(self.mj_data, camera=self.camera_name)
            rgb = self.mj_renderer.render()
            
            # 2. 更新 Depth
            self.mj_depth_renderer.update_scene(self.mj_data, camera=self.camera_name)
            depth = self.mj_depth_renderer.render()
            
            return rgb, depth
        except Exception as e:
            print(f"❌ 获取图像失败: {e}")
            return None, None

    def step(self, action=None):
        """
        [物理步进] 包含位置控制逻辑
        """
        # 1. 应用控制信号
        if action is not None:
            n_ctrl = len(self.mj_data.ctrl)
            n_action = len(action)
            self.mj_data.ctrl[:min(n_ctrl, n_action)] = action[:min(n_ctrl, n_action)]
        else:
            # [ROS 模式] 使用内部的 target_q 驱动电机
            if len(self.mj_data.ctrl) >= 6:
                self.mj_data.ctrl[:6] = self.target_q

        # 2. 物理引擎计算
        mujoco.mj_step(self.mj_model, self.mj_data)

        # 3. 同步显示窗口
        if self.mj_viewer is not None:
            self.mj_viewer.sync()

    def render(self):
        """兼容旧接口"""
        return {
            'img': self.get_image(),
            'depth': None 
        }

if __name__ == '__main__':
    env = UR5GraspEnv(render_mode='window')
    env.reset()
    print("✅ 环境启动成功 (640x640)！")
    
    target = env.init_q.copy()
    target[2] += 0.5 
    env.goto_joints(target)
    
    for i in range(1000):
        env.step()
        time.sleep(0.002) 
        
    env.close()