set(A3_ROBOT_IO_SOURCES
  src/robot_io/backend_factory.cpp
  src/robot_io/layouts.cpp
  src/robot_io/a3_layout_extra.cpp
  src/robot_io/a3_aimrt_backend.cpp
  src/a3_io/subscriber_manager.cpp
  src/a3_io/subscribers/arm_state_sub.cpp
  src/a3_io/subscribers/leg_state_sub.cpp
  src/a3_io/subscribers/neck_state_sub.cpp
  src/a3_io/subscribers/pelvis_imu_sub.cpp
  src/a3_io/subscribers/torso_imu_sub.cpp
  src/a3_io/subscribers/waist_state_sub.cpp
  src/a3_io/publisher_manager.cpp
  src/a3_io/publishers/arm_cmd_pub.cpp
  src/a3_io/publishers/leg_cmd_pub.cpp
  src/a3_io/publishers/neck_cmd_pub.cpp
  src/a3_io/publishers/waist_cmd_pub.cpp
  src/a3_sync/a3_sync_loop.cpp)

# The vendor factory deliberately provides a weak fallback in
# backend_factory.cpp and a strong A3 implementation in a separate translation
# unit. Keep the sources as an object library so both objects are placed on the
# final executable link line; a normal static archive can satisfy the symbol
# from the weak object and never extract the strong backend object.
add_library(a3_robot_io OBJECT ${A3_ROBOT_IO_SOURCES})
add_library(a3::robot_io ALIAS a3_robot_io)

target_include_directories(a3_robot_io PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

target_compile_definitions(a3_robot_io PRIVATE
  ENABLE_A3_AIMRT_BACKEND=1
  HAS_A3_ROS_MSGS=1)

target_link_libraries(a3_robot_io
  PUBLIC
    Eigen3::Eigen
  PRIVATE
    Threads::Threads
    aimrt::runtime::core
    aimrt::interface::aimrt_module_cpp_interface
    aimrt::interface::aimrt_module_ros2_interface
    joint_msgs::joint_msgs__rosidl_generator_cpp
    joint_msgs::joint_msgs__rosidl_typesupport_cpp
    joint_msgs::joint_msgs__rosidl_typesupport_fastrtps_cpp
    joint_msgs::joint_msgs__rosidl_typesupport_introspection_cpp
    sensor_msgs::sensor_msgs__rosidl_generator_cpp
    sensor_msgs::sensor_msgs__rosidl_typesupport_cpp
    sensor_msgs::sensor_msgs__rosidl_typesupport_fastrtps_cpp
    sensor_msgs::sensor_msgs__rosidl_typesupport_introspection_cpp)

set_target_properties(a3_robot_io PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF)
