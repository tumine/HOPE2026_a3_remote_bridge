add_executable(a3_hdu_command_dry_relay src/main.cpp)

target_link_libraries(a3_hdu_command_dry_relay PRIVATE
  rclcpp::rclcpp
  std_msgs::std_msgs__rosidl_generator_cpp
  std_msgs::std_msgs__rosidl_typesupport_cpp
  joint_msgs::joint_msgs__rosidl_generator_cpp
  joint_msgs::joint_msgs__rosidl_typesupport_cpp)

set_target_properties(a3_hdu_command_dry_relay PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/dist"
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH "$ORIGIN;/opt/ros/jazzy/lib;/opt/ros/humble/lib"
  INSTALL_RPATH "$ORIGIN;/opt/ros/jazzy/lib;/opt/ros/humble/lib")

