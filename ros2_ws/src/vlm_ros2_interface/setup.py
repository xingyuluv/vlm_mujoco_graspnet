from setuptools import find_packages, setup

package_name = 'vlm_ros2_interface'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='YourName',
    maintainer_email='your@email.com',
    description='ROS2 Interface for VLM MuJoCo Graspnet',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mujoco_bridge = vlm_ros2_interface.mujoco_bridge:main',
            'vlm_client = vlm_ros2_interface.vlm_client:main',
        ],
    },
)