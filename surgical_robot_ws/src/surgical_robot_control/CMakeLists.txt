cmake_minimum_required(VERSION 3.8)
project(surgical_robot_control)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# 消息生成
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/TrajectoryPoint.msg"
  "msg/RobotState.msg"
  DEPENDENCIES std_msgs
)

# 让可执行文件依赖已生成的接口
add_executable(hello_control src/hello_control.cpp)
ament_target_dependencies(hello_control rclcpp std_msgs)
rosidl_target_interfaces(hello_control ${PROJECT_NAME} "rosidl_typesupport_cpp")

# 添加trajectory_player可执行文件
add_executable(trajectory_player src/trajectory_player.cpp)
ament_target_dependencies(trajectory_player rclcpp std_msgs)
rosidl_target_interfaces(trajectory_player ${PROJECT_NAME} "rosidl_typesupport_cpp")

install(TARGETS hello_control trajectory_player DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_export_dependencies(rosidl_default_runtime)
ament_package()
