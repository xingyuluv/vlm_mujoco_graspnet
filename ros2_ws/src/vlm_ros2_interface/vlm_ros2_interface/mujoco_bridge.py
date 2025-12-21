import sys
import os
import rclpy
from rclpy.node import Node
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
from std_srvs.srv import Trigger  # 用于简单的触发服务
# 这里我们假设使用一个简单的自定义服务或者复用现有类型，为了简单起见，暂时用 Pose 作为单纯的各种指令的载体
# 在实际 ROS2 工程中，通常会定义专门的 .srv 文件，例如 ExecuteGrasp.srv

# --- 关键：将项目根目录加入路径，以便导入原有的 manipulator_grasp ---
# 假设你在 vlm_mujoco_graspnet 根目录下运行 colcon build 或 python 脚本
# 你需要根据实际运行路径调整这里
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))

try:
    from manipulator_grasp.env.ur5_grasp_env import Ur5GraspEnv
except ImportError as e:
    print("Error importing Ur5GraspEnv. Make sure project root is in PYTHONPATH.")
    raise e

class MujocoBridge(Node):
    def __init__(self):
        super().__init__('mujoco_bridge_node')
        
        # 1. 初始化仿真环境
        self.get_logger().info('Initializing MuJoCo Environment...')
        self.env = Ur5GraspEnv(render_mode='human')
        self.obs = self.env.reset()
        
        # 2. 初始化 ROS 工具
        self.cv_bridge = CvBridge()
        
        # 3. 创建发布者 (Publishers) - 发布传感器数据
        # 发布 RGB 图像
        self.rgb_pub = self.create_publisher(Image, '/camera/rgb', 10)
        # 发布深度图
        self.depth_pub = self.create_publisher(Image, '/camera/depth', 10)
        
        # 4. 创建定时器 (Timer) - 控制仿真步进和数据发布频率
        # 假设 30Hz
        self.timer = self.create_timer(1.0/30.0, self.timer_callback)
        
        # 5. 创建服务 (Services) - 接收来自大脑的指令
        # 这里为了演示，我们创建一个服务来接收抓取指令
        # 在实际复杂项目中，可能会用 Action Server
        # 这里我们简单化：订阅一个 Pose 话题来执行抓取，或者用 Timer 里的逻辑
        self.grasp_sub = self.create_subscription(
            Pose,
            '/robot/execute_grasp',
            self.grasp_callback,
            10
        )
        
        self.get_logger().info('MuJoCo Bridge is Ready!')

    def timer_callback(self):
        """
        这个函数会不断被调用，用于渲染仿真并发布图像
        """
        # 这一步主要是为了保持查看器活跃
        # 注意：如果你的 env.step() 包含渲染，这里可能不需要额外渲染
        # self.env.render() 
        
        # 获取当前图像 (假设 env 里面有获取图像的方法，根据你的代码逻辑调整)
        # 你的 Ur5GraspEnv 似乎在 step 或 reset 时返回 obs
        # 我们假设可以通过某种方式获取当前帧，例如:
        rgb_img = self.env.get_rgb_image() # 需要你在 Ur5GraspEnv 中确认有这个接口或者自行添加
        depth_img = self.env.get_depth_image()

        if rgb_img is not None:
            # 转换 OpenCV/Numpy 图片到 ROS2 消息
            try:
                # 确保图片是 uint8
                if rgb_img.dtype != np.uint8:
                    rgb_img = rgb_img.astype(np.uint8)
                
                msg = self.cv_bridge.cv2_to_imgmsg(rgb_img, encoding="rgb8")
                self.rgb_pub.publish(msg)
            except Exception as e:
                self.get_logger().error(f'Failed to publish RGB: {e}')

        if depth_img is not None:
            try:
                # 深度图通常是 float32 或 uint16
                msg = self.cv_bridge.cv2_to_imgmsg(depth_img, encoding="passthrough")
                self.depth_pub.publish(msg)
            except Exception as e:
                self.get_logger().error(f'Failed to publish Depth: {e}')

    def grasp_callback(self, msg: Pose):
        """
        当接收到 '/robot/execute_grasp' 消息时调用
        msg 包含位置 (x,y,z) 和 姿态 (orientation)
        """
        self.get_logger().info(f'Received Grasp Command: {msg.position}')
        
        # 将 ROS 消息转换为你原本项目需要的格式
        position = [msg.position.x, msg.position.y, msg.position.z]
        # orientation = [msg.orientation.x, ... ] 
        
        # 调用原来的环境控制接口
        # 比如: self.env.step_to_position(position)
        # 注意：这里需要阻塞主线程还是异步运行取决于你的 Env 实现
        
        # 模拟执行
        # action = ... calculate inverse kinematics or low level control
        # self.env.step(action)
        self.get_logger().info('Executing grasp logic in simulation...')

def main(args=None):
    rclpy.init(args=args)
    node = MujocoBridge()
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