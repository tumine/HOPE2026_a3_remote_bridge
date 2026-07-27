cmake_minimum_required(VERSION 3.16)

project(a3_remote_bridge VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(FetchContent)
FetchContent_Declare(fmt
  URL https://github.com/fmtlib/fmt/archive/10.2.1.tar.gz
  URL_HASH SHA256=1250e4cc58bf06ee631567523f48848dc4596133e163f02615c97f78bab6c811
  OVERRIDE_FIND_PACKAGE)

find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)

add_subdirectory(thirdparty/joint_msgs)

foreach(_suffix
    rosidl_generator_cpp
    rosidl_typesupport_introspection_cpp)
  if(TARGET joint_msgs__${_suffix}
     AND NOT TARGET joint_msgs::joint_msgs__${_suffix})
    add_library(joint_msgs::joint_msgs__${_suffix} ALIAS
                joint_msgs__${_suffix})
  endif()
endforeach()

include(cmake/GetAimRT.cmake)

add_subdirectory(src/a3_robot_io)
add_subdirectory(src/mdu_state_bridge)
add_subdirectory(src/hdu_command_relay)
