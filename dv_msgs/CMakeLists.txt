cmake_minimum_required(VERSION 3.8)
project(dv_msgs)

find_package(ament_cmake REQUIRED)
find_package(std_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "action/SetMission.action"
  "action/RuntimeControl.action"
  "msg/LifecycleProgress.msg"
  "msg/ConeArray.msg"
  "srv/ActivateMode.srv"
  "srv/Setup.srv"
  "srv/StartBag.srv"
  "srv/StopBag.srv"
  DEPENDENCIES std_msgs builtin_interfaces
)

ament_package()
