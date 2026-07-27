add_executable(a3_mdu_state_bridge src/main.cpp)

target_link_libraries(a3_mdu_state_bridge PRIVATE
  a3::robot_io
  rclcpp::rclcpp
  sensor_msgs::sensor_msgs__rosidl_generator_cpp
  sensor_msgs::sensor_msgs__rosidl_typesupport_cpp
  std_msgs::std_msgs__rosidl_generator_cpp
  std_msgs::std_msgs__rosidl_typesupport_cpp
  joint_msgs::joint_msgs__rosidl_generator_cpp
  joint_msgs::joint_msgs__rosidl_typesupport_cpp
  Threads::Threads)

set_target_properties(a3_mdu_state_bridge PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/dist"
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH "$ORIGIN;/opt/ros/jazzy/lib;/opt/ros/humble/lib"
  INSTALL_RPATH "$ORIGIN;/opt/ros/jazzy/lib;/opt/ros/humble/lib")

# AimRT defines the plugin before this directory is added. Override the
# build-tree path inherited from AimRT so the copied deployment plugin never
# refers to the Docker-only /work/build directory.
if(TARGET aimrt_plugins_iceoryx_plugin)
  set_target_properties(aimrt_plugins_iceoryx_plugin PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    BUILD_RPATH "$ORIGIN;/opt/ros/jazzy/lib;/opt/ros/humble/lib"
    INSTALL_RPATH "$ORIGIN;/opt/ros/jazzy/lib;/opt/ros/humble/lib")
endif()

add_custom_command(TARGET a3_mdu_state_bridge POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_FILE:aimrt_plugins_iceoryx_plugin>
    "$<TARGET_FILE_DIR:a3_mdu_state_bridge>"
  COMMENT "Copying AimRT iceoryx plugin")

foreach(_joint_target
    joint_msgs__rosidl_generator_c
    joint_msgs__rosidl_typesupport_cpp
    joint_msgs__rosidl_typesupport_fastrtps_cpp
    joint_msgs__rosidl_typesupport_introspection_cpp
    joint_msgs__rosidl_typesupport_c
    joint_msgs__rosidl_typesupport_fastrtps_c
    joint_msgs__rosidl_typesupport_introspection_c)
  if(TARGET ${_joint_target})
    set_target_properties(${_joint_target} PROPERTIES
      BUILD_WITH_INSTALL_RPATH TRUE
      BUILD_RPATH "$ORIGIN;/opt/ros/jazzy/lib;/opt/ros/humble/lib"
      INSTALL_RPATH "$ORIGIN;/opt/ros/jazzy/lib;/opt/ros/humble/lib")
    add_custom_command(TARGET a3_mdu_state_bridge POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:${_joint_target}>
        "$<TARGET_FILE_DIR:a3_mdu_state_bridge>"
      COMMENT "Copying ${_joint_target}")
  endif()
endforeach()
