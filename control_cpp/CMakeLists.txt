cmake_minimum_required(VERSION 3.16)
project(control_cpp C CXX)

# control_core: the controller in plain C99 (no heap, no OS, no ROS) so the
# same file builds for the uDV STM32H7. control_node: the rclcpp shell.

if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(node_base_cpp REQUIRED)
find_package(dv_msgs REQUIRED)
find_package(fs_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)

add_library(control_core STATIC src/control_core.c)
target_include_directories(control_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
# Same arithmetic as the Python reference: no fused multiply-add.
target_compile_options(control_core PRIVATE -Wall -Wextra -Wpedantic -ffp-contract=off)
set_target_properties(control_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(control_core m)

add_executable(control_node src/control_node.cpp)
target_compile_options(control_node PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(control_node control_core node_base_cpp::node_base_cpp)
ament_target_dependencies(control_node
  rclcpp rclcpp_lifecycle dv_msgs fs_msgs nav_msgs std_msgs)

install(TARGETS control_node DESTINATION lib/${PROJECT_NAME})
install(TARGETS control_core ARCHIVE DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
# The C sources, for the uDV firmware build.
install(FILES src/control_core.c DESTINATION share/${PROJECT_NAME}/src)

ament_package()
