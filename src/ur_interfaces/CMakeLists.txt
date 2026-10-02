cmake_minimum_required(VERSION 3.8)
project(ur_interfaces)
find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(action_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/DetectObjects.srv"
  "srv/DetectOpen.srv"
  "action/MoveHome.action"
  "action/Pick.action"
  "action/Place.action"
  DEPENDENCIES geometry_msgs action_msgs)
ament_package()
