#!/usr/bin/env bash
# One-time setup: run from ur_llm_ws/
set -e
source /opt/ros/jazzy/setup.bash
sudo apt update
sudo apt install -y ros-jazzy-ur ros-jazzy-ur-simulation-gz ros-jazzy-moveit \
  ros-jazzy-ros-gz ros-jazzy-ros2-control ros-jazzy-ros2-controllers \
  ros-jazzy-gz-ros2-control ros-jazzy-cv-bridge python3-opencv python3-numpy
[ -d src/pymoveit2 ] || git clone https://github.com/AndrejOrsula/pymoveit2.git src/pymoveit2
# Robotiq 2F-85 description only (the driver packages in that repo are not needed in simulation)
if [ ! -d src/robotiq_description ]; then
  git clone --depth 1 https://github.com/PickNikRobotics/ros2_robotiq_gripper /tmp/ros2_robotiq_gripper
  cp -r /tmp/ros2_robotiq_gripper/robotiq_description src/robotiq_description
fi
rosdep install -r --from-paths src -i -y || true
colcon build
echo "Done. Run: source install/setup.bash"