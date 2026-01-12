import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray
from cv_bridge import CvBridge
import threading
import numpy as np
import os
import sys
import cv2
import time

# ================= 核心：路径修复与环境隔离 =================
# 1. 物理路径定义
PROJECT_ROOT = '/home/xingyu/projects/VLM_Grasp_Interactive'
MANIPULATOR_PATH = os.path.join(PROJECT_ROOT, 'manipulator_grasp')

# 2. 代理清理 (防止干扰本地/云端 API 调用)
proxy_vars = [
    "http_proxy", "https_proxy", "all_proxy", 
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
    "no_proxy", "NO_PROXY"
]
for var in proxy_vars:
    if var in os.environ:
        os.environ.pop(var)

# 3. 路径注入 (确保 manipulator_grasp 在最前)
if MANIPULATOR_PATH not in sys.path:
    sys.path.insert(0, MANIPULATOR_PATH)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(1, PROJECT_ROOT)

os.chdir(PROJECT_ROOT)

# ================= 模块导入与模式配置 =================

# --- ⚙️ 配置开关 (在此处切换模型) ---
USE_YOLO_WORLD = True  # True: 使用 YOLO-World (vlm_process)
                       # False: 使用 Qwen-VL (vlm_process_old)

try:
    # 1. 导入 YOLO 版本 (vlm_process.py)
    try:
        import vlm_process as vlm_yolo
        print("✅ 已加载 YOLO-World 模块 (vlm_process)")
    except ImportError as e:
        print(f"⚠️ YOLO 模块加载失败: {e}")
        vlm_yolo = None

    # 2. 导入 Qwen 版本 (vlm_process_old.py)
    try:
        import vlm_process_old as vlm_qwen
        print("✅ 已加载 Qwen-VL 模块 (vlm_process_old)")
    except ImportError as e:
        print(f"⚠️ Qwen 模块加载失败: {e}")
        vlm_qwen = None

    # 3. 导入 GraspNet 推理 (通用)
    from grasp_process_old import run_grasp_inference
    print("✅ 已加载 GraspNet 逻辑模块")

except ImportError as e:
    print(f"❌ 严重导入错误: {e}")
    sys.exit(1)

# ==========================================================

class VlmManagerNode(Node):
    def __init__(self):
        super().__init__('vlm_manager')
        
        # 根据开关状态显示启动日志
        self.mode_name = "YOLO-World (本地)" if USE_YOLO_WORLD else "Qwen-VL (云端)"
        self.get_logger().info(f'VLM Manager 启动中... 当前模式: [{self.mode_name}]')
        
        self.bridge = CvBridge()
        self.img_lock = threading.Lock()
        self.latest_rgb = None
        self.latest_depth = None
        
        # 使用 Sensor Data QoS 保证图像实时性
        from rclpy.qos import qos_profile_sensor_data
        self.create_subscription(Image, '/camera/color', self.rgb_cb, qos_profile_sensor_data)
        self.create_subscription(Image, '/camera/depth', self.depth_cb, qos_profile_sensor_data)
        
        # 发布计算好的位姿数据
        self.pose_pub = self.create_publisher(Float64MultiArray, '/robot/target_pose', 10)
        
        # 启动后台大脑线程
        threading.Thread(target=self.brain_loop, daemon=True).start()

    def rgb_cb(self, msg):
        cv_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        with self.img_lock: self.latest_rgb = cv_img

    def depth_cb(self, msg):
        cv_depth = self.bridge.imgmsg_to_cv2(msg, '32FC1')
        with self.img_lock: self.latest_depth = cv_depth

    def brain_loop(self):
        while rclpy.ok():
            if self.latest_rgb is None or self.latest_depth is None:
                time.sleep(1)
                continue
            
            print("\n" + "="*50)
            print(f"🚀 就绪模式: {self.mode_name}")
            print("👉 请确保相关权重文件已准备好")
            input("⌨️  按 [回车] 键开始一次新的抓取任务...")
            
            # 锁定并复制当前帧
            with self.img_lock:
                rgb_now = self.latest_rgb.copy()
                depth_now = self.latest_depth.copy()

            try:
                self.get_logger().info(f'📸 正在执行视觉感知 ({self.mode_name})...')
                
                mask = None
                target_name = ""  # 初始化物体名称
                
                # ================= 核心分支逻辑 =================
                if USE_YOLO_WORLD:
                    # 1. 使用 YOLO-World (vlm_process)
                    if vlm_yolo is None:
                        self.get_logger().error("❌ 无法运行: vlm_process 模块未正确加载")
                        continue
                    
                    # 【修改点 1】接收两个返回值：掩码 + 物体名
                    mask, target_name = vlm_yolo.segment_image(rgb_now)
                
                else:
                    # 2. 使用 Qwen-VL (vlm_process_old)
                    # 注意：如果您没有修改 vlm_process_old.py，这里还是只返回 mask
                    if vlm_qwen is None:
                        self.get_logger().error("❌ 无法运行: vlm_process_old 模块未正确加载")
                        continue
                        
                    # 兼容旧版代码：如果 Qwen 版本没改，就只接 mask
                    result = vlm_qwen.segment_image(rgb_now)
                    if isinstance(result, tuple):
                        mask, target_name = result
                    else:
                        mask = result
                        target_name = "unknown"
                # ===============================================

                # 统一的后处理逻辑
                if mask is None:
                    self.get_logger().warn('⚠️ 流程中断: 分割未返回有效掩码 (未检测到物体或取消)')
                    continue

                self.get_logger().info(f'🧠 正在计算抓取位姿 (GraspNet), 目标: {target_name} ...')
                # GraspNet 的输入接口是一样的，直接复用
                gg = run_grasp_inference(rgb_now, depth_now, mask)
                
                if gg is not None and len(gg) > 0:
                    best_translation = gg.translations[0]    # [x, y, z]
                    best_rotation = gg.rotation_matrices[0]  # 3x3 矩阵
                    
                    self.get_logger().info(f'✅ 找到最佳抓取点，得分: {gg.scores[0]:.4f}')
                    
                    # =================【修改点 2】判断物体类型并生成标志位 =================
                    # 定义薄物体关键词 (支持中英文)
                    thin_keywords = ['hammer', 'mouse', 'card', 'knife', 'chuizi', 'shubiao', '锤子', '鼠标']
                    
                    # 检查 target_name 是否包含关键词
                    is_thin = any(k in target_name.lower() for k in thin_keywords)
                    
                    # 标志位: 1.0 表示需要特殊抬升，0.0 表示普通抓取
                    flag_val = 1.0 if is_thin else 0.0
                    
                    if is_thin:
                        self.get_logger().info(f"🛠️ 检测到薄物体 [{target_name}]，发送【抬升指令】")
                    else:
                        self.get_logger().info(f"📦 检测到普通物体 [{target_name}]，发送标准指令")
                    # ==================================================================
                    
                    # 构造消息并发布
                    msg = Float64MultiArray()
                    # 数据结构: [9个旋转矩阵元素] + [3个位置坐标] + [1个标志位]
                    data = best_rotation.flatten().tolist() + best_translation.tolist() + [flag_val]
                    msg.data = data
                    self.pose_pub.publish(msg)
                else:
                    self.get_logger().error('❌ GraspNet 未找到有效抓取位姿')
                    
            except Exception as e:
                self.get_logger().error(f'大脑逻辑运行异常: {e}')
                import traceback
                traceback.print_exc()

def main():
    rclpy.init()
    node = VlmManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()