import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray
from cv_bridge import CvBridge
import cv2
import numpy as np
import sys
import os

# 将项目根目录添加到路径，以便导入 manipulator_grasp
sys.path.append('/home/xingyu/projects/VLM_Grasp_Interactive') 
# ⚠️ 注意：请根据你的实际路径修改上面这一行，确保能找到 manipulator_grasp

from manipulator_grasp.env.ur5_grasp_env import UR5GraspEnv

class MujocoBridgeNode(Node):
    def __init__(self):
        super().__init__('mujoco_bridge_node')
        
        # 1. 初始化 MuJoCo 环境
        self.get_logger().info('正在初始化 MuJoCo 环境...')
        self.env = UR5GraspEnv(render_mode='window') # 或者 'offscreen'
        self.env.reset()
        
        # 2. 初始化 ROS 通信工具
        self.bridge = CvBridge()
        
        # [发布者] 发布相机图像
        self.image_pub = self.create_publisher(Image, '/camera/image_raw', 10)
        
        # [订阅者] 接收关节控制指令 (例如: [j1, j2, j3, j4, j5, j6])
        self.subscription = self.create_subscription(
            Float64MultiArray,
            '/robot/command',
            self.command_callback,
            10
        )

        # 3. 设置定时器 (例如 30Hz)，用于步进仿真和发布图像
        self.timer = self.create_timer(0.033, self.timer_callback)
        self.get_logger().info('MuJoCo Bridge 节点已启动!')

    def timer_callback(self):
        """周期性运行：仿真步进 + 图像发布"""
        # 1. 获取相机图像 (这里假设 env 有 get_camera_data 方法，或者我们手动获取)
        # 根据你之前的代码，我们可以模拟获取图像
        # 如果 env.render() 返回图像数组，可以直接用
        
        # 暂时用 env 的 sim 获取图像演示 (需要根据你的 env 具体实现调整)
        # 这里假设 env.cur_image 存储了当前的 RGB 图像
        rgb_image = self.env.get_image() # 假设你在 env 里封装了这个方法
        
        if rgb_image is not None:
            # 转换 OpenCV/Numpy 图像为 ROS 消息
            try:
                # 确保是 uint8 格式
                if rgb_image.dtype != np.uint8:
                    rgb_image = (rgb_image * 255).astype(np.uint8)
                
                # 如果是 BGR (OpenCV默认)，ROS通常用 RGB，根据需要转换
                # msg = self.bridge.cv2_to_imgmsg(rgb_image, encoding="bgr8")
                msg = self.bridge.cv2_to_imgmsg(rgb_image, encoding="rgb8")
                self.image_pub.publish(msg)
            except Exception as e:
                self.get_logger().error(f'图像转换失败: {e}')

        # 2. 物理步进 (Step Simulation)
        # self.env.step(action) # 如果有持续的动作需要在这里调用

    def command_callback(self, msg):
        """接收到控制指令的回调函数"""
        target_joints = msg.data
        self.get_logger().info(f'收到指令: {target_joints}')
        
        # 调用 MuJoCo 的控制接口
        # 假设你的 env 有一个 step 或者 control 方法
        # self.env.step(np.array(target_joints)) 
        # 或者直接控制关节:
        try:
            # 这里需要适配你的 grasp_process.py 里的控制逻辑
            # 简单示例：直接设置目标位置
            self.env.goto_joints(np.array(target_joints)) 
        except Exception as e:
            self.get_logger().error(f'执行指令失败: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = MujocoBridgeNode()
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