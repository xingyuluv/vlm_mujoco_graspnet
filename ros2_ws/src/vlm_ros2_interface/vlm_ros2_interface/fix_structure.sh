# 1. Create the missing config file
echo "[develop]
script_dir=\$base/lib/vlm_ros2_interface
[install]
install_scripts=\$base/lib/vlm_ros2_interface" > ~/projects/VLM_Grasp_Interactive/ros2_ws/src/vlm_ros2_interface/setup.cfg

# 2. Clean and Rebuild
cd ~/projects/VLM_Grasp_Interactive/ros2_ws
rm -rf build install log
colcon build --symlink-install

# 3. Run
source install/setup.bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/..
ros2 run vlm_ros2_interface mujoco_bridge