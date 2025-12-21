import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/xingyu/projects/VLM_Grasp_Interactive/ros2_ws/install/vlm_ros2_interface'
