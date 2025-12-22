import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray
from cv_bridge import CvBridge
import cv2
import numpy as np
import threading
import time

# 导入你原来的 VLM 逻辑
import sys
sys.path.append('/home/xingyu/projects/VLM_Grasp_Interactive')
from vlm_process import generate_robot_actions, segment_image
# 注意：你需要把 main_vlm.py 里的逻辑拆解一下，或者在这里重写

class VLMClientNode(Node):
    def __init__(self):
        super().__init__('vlm_client_node')
        
        self.bridge = CvBridge()
        self.latest_image = None
        self.image_lock = threading.Lock()
        
        # [订阅者] 接收图像
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        
        # [发布者] 发送控制指令
        self.command_pub = self.create_publisher(Float64MultiArray, '/robot/command', 10)
        
        self.get_logger().info('VLM Client 节点已启动，等待图像...')
        
        # 在单独线程中运行主逻辑，避免阻塞 ROS 回调
        self.main_thread = threading.Thread(target=self.main_logic)
        self.main_thread.start()

    def image_callback(self, msg):
        """接收图像并保存"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            with self.image_lock:
                self.latest_image = cv_image
        except Exception as e:
            self.get_logger().error(f'图像接收错误: {e}')

    def publish_joints(self, joints):
        """发送关节角度"""
        msg = Float64MultiArray()
        msg.data = joints.tolist() if isinstance(joints, np.ndarray) else joints
        self.command_pub.publish(msg)
        self.get_logger().info(f'已发送指令: {joints}')

    def main_logic(self):
        """主控制循环 (相当于原来的 main_vlm.py)"""
        
        # 等待第一帧图像
        while self.latest_image is None:
            time.sleep(0.1)
        self.get_logger().info('收到第一帧图像，开始交互流程...')

        while rclpy.ok():
            # 1. 获取当前图像快照
            with self.image_lock:
                current_img = self.latest_image.copy()
            
            # 2. (可选) 获取用户指令
            # command_text = input("请输入抓取指令: ")
            command_text = "抓取红色的方块" # 模拟输入
            
            self.get_logger().info(f'处理指令: {command_text}')

            # 3. 调用你的 VLM/GraspNet 算法 (复用你现有的函数)
            # 注意：这里需要根据你 vlm_process.py 的具体返回值修改
            try:
                # 示例逻辑：
                # mask = segment_image(current_img, command_text)
                # grasp_pose = calculate_grasp(mask, ...)
                # joint_path = ik_solver(grasp_pose)
                
                # 假设算出了一组目标关节角度
                dummy_joints = [0.0, -1.57, 1.57, 0.0, 0.0, 0.0] 
                
                # 4. 发送给 MuJoCo Bridge 执行
                self.publish_joints(dummy_joints)
                
            except Exception as e:
                self.get_logger().error(f'算法处理出错: {e}')
            
            # 暂停一下，模拟交互间隔
            time.sleep(5)

def main(args=None):
    rclpy.init(args=args)
    node = VLMClientNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()