set(A3_POLICY_ONNX_SOURCE
  "${CMAKE_SOURCE_DIR}/26.7.25发球部署/pingpang_ustc-srf/a3_deploy/a3_pingpong_deploy_bundle/policy/hope_pingpong.onnx"
  CACHE FILEPATH "Frozen model_21500 ONNX actor")
set(A3_POLICY_ONNX_SHA256
  "6e2fcf9c9793a568f0583ae9b6e7b0439fb83df004c356e85af85841afc2d074")
if(NOT EXISTS "${A3_POLICY_ONNX_SOURCE}")
  message(FATAL_ERROR "model_21500 ONNX not found: ${A3_POLICY_ONNX_SOURCE}")
endif()
file(SHA256 "${A3_POLICY_ONNX_SOURCE}" _a3_policy_onnx_actual_sha256)
if(NOT _a3_policy_onnx_actual_sha256 STREQUAL A3_POLICY_ONNX_SHA256)
  message(FATAL_ERROR
    "model_21500 ONNX SHA256 mismatch: ${_a3_policy_onnx_actual_sha256}")
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  FetchContent_Declare(a3_onnxruntime_aarch64
    URL
      "https://github.com/microsoft/onnxruntime/releases/download/v1.19.2/onnxruntime-linux-aarch64-1.19.2.tgz"
    URL_HASH
      "SHA256=5e30145277d6d6fcb0e8f14f0d0ab5048af7b13ffd608023bb1e2875621fab07"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  FetchContent_MakeAvailable(a3_onnxruntime_aarch64)
  set(_a3_onnxruntime_include "${a3_onnxruntime_aarch64_SOURCE_DIR}/include")
  # Use the SONAME path so the copied runtime name matches DT_NEEDED.
  set(_a3_onnxruntime_library
    "${a3_onnxruntime_aarch64_SOURCE_DIR}/lib/libonnxruntime.so.1")
else()
  find_path(_a3_onnxruntime_include onnxruntime_cxx_api.h
    HINTS /opt/onnxruntime/include REQUIRED)
  find_library(_a3_onnxruntime_library_unresolved onnxruntime
    HINTS /opt/onnxruntime/lib REQUIRED)
  get_filename_component(_a3_onnxruntime_library
    "${_a3_onnxruntime_library_unresolved}" REALPATH)
endif()

add_library(a3_onnxruntime SHARED IMPORTED)
set_target_properties(a3_onnxruntime PROPERTIES
  IMPORTED_LOCATION "${_a3_onnxruntime_library}"
  INTERFACE_INCLUDE_DIRECTORIES "${_a3_onnxruntime_include}")

add_library(a3_pingpong_planner_input STATIC
  src/manual_control.cpp
  src/planner_input.cpp
  src/swing_lifecycle.cpp)

target_include_directories(a3_pingpong_planner_input PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_compile_features(a3_pingpong_planner_input PUBLIC cxx_std_20)

add_library(a3_pingpong_observation_builder STATIC
  src/pingpong_observation_builder.cpp)

target_include_directories(a3_pingpong_observation_builder PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_compile_features(a3_pingpong_observation_builder PUBLIC cxx_std_20)
target_link_libraries(a3_pingpong_observation_builder PUBLIC
  a3_pingpong_planner_input
  a3::robot_io)

add_library(a3_pingpong_onnx_actor STATIC
  src/onnx_actor.cpp)

target_include_directories(a3_pingpong_onnx_actor PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_compile_features(a3_pingpong_onnx_actor PUBLIC cxx_std_20)
target_link_libraries(a3_pingpong_onnx_actor PUBLIC
  a3_pingpong_observation_builder
  a3_onnxruntime)

add_library(a3_pingpong_action_adapter STATIC
  src/pingpong_action_adapter.cpp)

target_include_directories(a3_pingpong_action_adapter PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_compile_features(a3_pingpong_action_adapter PUBLIC cxx_std_20)
target_link_libraries(a3_pingpong_action_adapter PUBLIC
  a3_pingpong_observation_builder
  a3::robot_io)

add_library(a3_pingpong_receive_controller STATIC
  src/receive_controller.cpp)

target_include_directories(a3_pingpong_receive_controller PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_compile_features(a3_pingpong_receive_controller PUBLIC cxx_std_20)
target_link_libraries(a3_pingpong_receive_controller PUBLIC
  a3_pingpong_planner_input
  a3::robot_io)

add_executable(a3_mdu_planner_receiver
  src/mdu_planner_receiver.cpp)

target_link_libraries(a3_mdu_planner_receiver PRIVATE
  a3_pingpong_action_adapter
  a3_pingpong_onnx_actor
  a3_pingpong_observation_builder
  a3_pingpong_receive_controller
  a3::robot_io
  rclcpp::rclcpp
  geometry_msgs::geometry_msgs__rosidl_generator_cpp
  geometry_msgs::geometry_msgs__rosidl_typesupport_cpp
  hope_msgs__rosidl_generator_cpp
  hope_msgs__rosidl_typesupport_cpp)

set_target_properties(a3_mdu_planner_receiver PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/dist"
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH "$ORIGIN;/opt/ros/jazzy/lib;/opt/ros/humble/lib"
  INSTALL_RPATH "$ORIGIN;/opt/ros/jazzy/lib;/opt/ros/humble/lib")

add_custom_command(TARGET a3_mdu_planner_receiver POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/models"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${A3_POLICY_ONNX_SOURCE}"
    "${CMAKE_BINARY_DIR}/models/hope_pingpong.onnx"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "$<TARGET_FILE:a3_onnxruntime>"
    "$<TARGET_FILE_DIR:a3_mdu_planner_receiver>/$<TARGET_FILE_NAME:a3_onnxruntime>"
  COMMENT "Copying model_21500 and ONNX Runtime")

# hope_msgs is not installed on the robot image. Keep its generated C/C++
# libraries beside the receiver so Fast DDS can load the matching type support.
foreach(_hope_target
    hope_msgs__rosidl_generator_c
    hope_msgs__rosidl_typesupport_cpp
    hope_msgs__rosidl_typesupport_fastrtps_cpp
    hope_msgs__rosidl_typesupport_introspection_cpp
    hope_msgs__rosidl_typesupport_c
    hope_msgs__rosidl_typesupport_fastrtps_c
    hope_msgs__rosidl_typesupport_introspection_c)
  if(TARGET ${_hope_target})
    set_target_properties(${_hope_target} PROPERTIES
      BUILD_WITH_INSTALL_RPATH TRUE
      BUILD_RPATH "$ORIGIN;/opt/ros/jazzy/lib;/opt/ros/humble/lib"
      INSTALL_RPATH "$ORIGIN;/opt/ros/jazzy/lib;/opt/ros/humble/lib")
    add_custom_command(TARGET a3_mdu_planner_receiver POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:${_hope_target}>
        "$<TARGET_FILE_DIR:a3_mdu_planner_receiver>"
      COMMENT "Copying ${_hope_target}")
  endif()
endforeach()

if(A3_BUILD_TESTS)
  add_executable(a3_planner_input_test tests/planner_input_test.cpp)
  target_link_libraries(a3_planner_input_test PRIVATE a3_pingpong_planner_input)
  add_test(NAME a3_planner_input_test COMMAND a3_planner_input_test)

  add_executable(a3_swing_lifecycle_test tests/swing_lifecycle_test.cpp)
  target_link_libraries(a3_swing_lifecycle_test PRIVATE
    a3_pingpong_planner_input)
  add_test(NAME a3_swing_lifecycle_test COMMAND a3_swing_lifecycle_test)

  add_executable(a3_manual_control_test tests/manual_control_test.cpp)
  target_link_libraries(a3_manual_control_test PRIVATE
    a3_pingpong_planner_input)
  add_test(NAME a3_manual_control_test COMMAND a3_manual_control_test)

  add_executable(a3_receive_controller_test
    tests/receive_controller_test.cpp)
  target_link_libraries(a3_receive_controller_test PRIVATE
    a3_pingpong_receive_controller)
  add_test(NAME a3_receive_controller_test COMMAND a3_receive_controller_test)

  add_executable(a3_pingpong_observation_builder_test
    tests/pingpong_observation_builder_test.cpp)
  target_link_libraries(a3_pingpong_observation_builder_test PRIVATE
    a3_pingpong_observation_builder)
  target_compile_definitions(a3_pingpong_observation_builder_test PRIVATE
    "A3_OBSERVATION_GOLDEN_PATH=\"${CMAKE_CURRENT_SOURCE_DIR}/tests/observation_golden.txt\"")
  add_test(NAME a3_pingpong_observation_builder_test
    COMMAND a3_pingpong_observation_builder_test)

  add_executable(a3_pingpong_onnx_actor_test
    tests/onnx_actor_test.cpp)
  target_link_libraries(a3_pingpong_onnx_actor_test PRIVATE
    a3_pingpong_onnx_actor)
  target_compile_definitions(a3_pingpong_onnx_actor_test PRIVATE
    "A3_POLICY_ONNX_PATH=\"${A3_POLICY_ONNX_SOURCE}\""
    "A3_OBSERVATION_GOLDEN_PATH=\"${CMAKE_CURRENT_SOURCE_DIR}/tests/observation_golden.txt\""
    "A3_ACTION_GOLDEN_PATH=\"${CMAKE_CURRENT_SOURCE_DIR}/tests/onnx_action_golden.txt\"")
  set_target_properties(a3_pingpong_onnx_actor_test PROPERTIES
    BUILD_RPATH "/opt/onnxruntime/lib")
  add_test(NAME a3_pingpong_onnx_actor_test
    COMMAND a3_pingpong_onnx_actor_test)

  add_executable(a3_pingpong_action_adapter_test
    tests/pingpong_action_adapter_test.cpp)
  target_link_libraries(a3_pingpong_action_adapter_test PRIVATE
    a3_pingpong_action_adapter
    a3_pingpong_receive_controller)
  target_compile_definitions(a3_pingpong_action_adapter_test PRIVATE
    "A3_ACTION_GOLDEN_PATH=\"${CMAKE_CURRENT_SOURCE_DIR}/tests/onnx_action_golden.txt\""
    "A3_Q_DES_GOLDEN_PATH=\"${CMAKE_CURRENT_SOURCE_DIR}/tests/action_q_des_golden.txt\"")
  add_test(NAME a3_pingpong_action_adapter_test
    COMMAND a3_pingpong_action_adapter_test)

  add_executable(a3_planner_input_smoke_sender
    tests/planner_input_smoke_sender.cpp)
  target_link_libraries(a3_planner_input_smoke_sender PRIVATE
    rclcpp::rclcpp
    geometry_msgs::geometry_msgs__rosidl_generator_cpp
    geometry_msgs::geometry_msgs__rosidl_typesupport_cpp
    hope_msgs__rosidl_generator_cpp
    hope_msgs__rosidl_typesupport_cpp)
endif()
