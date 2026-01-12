import os
from glob import glob # <--- 1. 确保导入 glob
from setuptools import setup

package_name = 'vlm_ros2_interface'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        
        # 2. 添加下面这一行，注册 launch 文件：
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        
        # 如果你有 urdf 文件夹，最好也加上这一行：
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.xacro')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='xingyu',
    maintainer_email='xingyu@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # 你的节点配置应该在这里，保持原样即可
            'mujoco_bridge = vlm_ros2_interface.mujoco_bridge:main',
            'vlm_client = vlm_ros2_interface.vlm_client:main',
            'test_grasp_duck = vlm_ros2_interface.test_grasp_duck:main',
        ],
    },
)