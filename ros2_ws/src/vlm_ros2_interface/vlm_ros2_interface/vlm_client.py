import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
from cv_bridge import CvBridge
import cv2
import numpy as np
import sys
import os

# 引入你的 VLM 和 Grasp 处理模块
# 同样需要添加路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))

# 假设你有这几个模块 (根据你的 main_vlm.py 调整)
# from vlm_process import VLMProcessor
# from grasp_process import GraspProcessor

class VLMClient(Node):
    def __init__(self):
        super().__init__('vlm_brain_node')
        
        self.cv_bridge = CvBridge()
        self.latest_rgb = None
        self.latest_depth = None
        
        # 1. 订阅仿真画面
        self.create_subscription(Image, '/camera/rgb', self.rgb_callback, 10)
        self.create_subscription(Image, '/camera/depth', self.depth_callback, 10)
        
        # 2. 发布控制指令
        self.cmd_pub = self.create_publisher(Pose, '/robot/execute_grasp', 10)
        
        # 3. 逻辑控制定时器 (例如每 5 秒做一次决策，或者等待用户输入)
        self.process_timer = self.create_timer(5.0, self.decision_loop)
        
        self.get_logger().info('VLM Brain Node Ready. Waiting for images...')

    def rgb_callback(self, msg):
        try:
            self.latest_rgb = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        except Exception as e:
            self.get_logger().error(f'RGB decode error: {e}')

    def depth_callback(self, msg):
        try:
            self.latest_depth = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().error(f'Depth decode error: {e}')

    def decision_loop(self):
        """
        这里放你 main_vlm.py 的核心逻辑
        """
        if self.latest_rgb is None:
            self.get_logger().info('Waiting for camera data...')
            return
            
        self.get_logger().info('Processing VLM logic...')
        
        # --- 伪代码：集成你的 VLM 逻辑 ---
        # 1. 保存图片
        # cv2.imwrite("temp_vlm.jpg", self.latest_rgb)
        
        # 2. 调用 VLM (Gemini/GPT4v)
        # text_instruction = "Grasp the red apple"
        # grasp_target = vlm_processor.process(text_instruction, "temp_vlm.jpg")
        
        # 3. 调用 GraspNet
        # grasp_pose = graspnet.detect(self.latest_rgb, self.latest_depth, grasp_target)
        
        # 4. 如果找到抓取位姿，发送给 MuJoCo Bridge
        # 假设 grasp_pose 是 [x, y, z]
        found_grasp = True # 模拟
        target_pos = [0.5, 0.0, 0.3] # 模拟数据
        
        if found_grasp:
            msg = Pose()
            msg.position.x = float(target_pos[0])
            msg.position.y = float(target_pos[1])
            msg.position.z = float(target_pos[2])
            # msg.orientation...
            
            self.cmd_pub.publish(msg)
            self.get_logger().info(f'Sent grasp command: {target_pos}')
        else:
            self.get_logger().info('No grasp found.')

def main(args=None):
    rclpy.init(args=args)
    node = VLMClient()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()