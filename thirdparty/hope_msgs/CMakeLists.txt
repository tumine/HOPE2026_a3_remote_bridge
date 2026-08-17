cmake_minimum_required(VERSION 3.8)

project(hope_msgs)

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

include(GSRosidl OPTIONAL)
if(COMMAND gs_disable_rosidl_generator_py_if_requested)
  gs_disable_rosidl_generator_py_if_requested()
endif()

set(_hope_msgs_build_shared_libs ${BUILD_SHARED_LIBS})
set(BUILD_SHARED_LIBS ON)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/RacketCommand.msg"
  DEPENDENCIES geometry_msgs std_msgs)

set(BUILD_SHARED_LIBS ${_hope_msgs_build_shared_libs})

ament_export_dependencies(rosidl_default_runtime)
ament_package()
