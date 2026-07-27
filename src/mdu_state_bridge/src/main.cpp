#include "robot_io/robot_io_backend.hpp"

#include <joint_msgs/msg/joint_command.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/u_int32.hpp>

#include <Eigen/Core>

#include <atomic>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace {

struct Options {
  std::string aimrt_cfg;
  std::string topic_prefix{"/a3_internal"};
  double publish_hz{100.0};
  bool command_dry_run{false};
  bool command_enable{false};
  double command_watchdog_ms{100.0};
  double max_state_age_ms{100.0};
  double max_position_delta_rad{0.02};
  double max_session_excursion_rad{0.05};
  double max_abs_velocity{0.0};
  double max_abs_effort{0.0};
  double max_kp{0.0};
  double max_kd{0.0};
};

void Usage(const char* program) {
  std::cout
      << "Usage: " << program << " --aimrt-cfg PATH [options]\n\n"
      << "Read synchronized A3 state from MDU-local iceoryx and publish it "
         "over ROS 2.\n"
      << "The backend is read-only unless --command-enable is explicitly "
         "combined with A3_ACTUATION_CONFIRM=ENABLE_A3_ACTUATION.\n\n"
      << "Options:\n"
      << "  --aimrt-cfg PATH\n"
      << "  --topic-prefix PREFIX  (default: /a3_internal)\n"
      << "  --publish-hz HZ        (default: 100)\n"
      << "  --command-dry-run      validate /joint_command_dry_run and ACK; "
         "never send\n"
      << "  --command-enable       enable guarded /joint_command_control output\n"
      << "  --command-watchdog-ms MS       (default: 100)\n"
      << "  --max-state-age-ms MS          (default: 100)\n"
      << "  --max-position-delta-rad RAD   (default: 0.02)\n"
      << "  --max-session-excursion-rad RAD (default: 0.05)\n"
      << "  --max-abs-velocity VALUE       (default: 0)\n"
      << "  --max-abs-effort VALUE         (default: 0)\n"
      << "  --max-kp VALUE                 (default: 0)\n"
      << "  --max-kd VALUE                 (default: 0)\n"
      << "  -h, --help\n";
}

std::optional<std::string> EqualsValue(const std::string& arg,
                                       const std::string& key) {
  const std::string prefix = key + "=";
  if (arg.rfind(prefix, 0) != 0) return std::nullopt;
  return arg.substr(prefix.size());
}

Options ParseOptions(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "-h" || arg == "--help") {
      Usage(argv[0]);
      std::exit(0);
    }
    if (arg == "--aimrt-cfg" && i + 1 < argc) {
      options.aimrt_cfg = argv[++i];
    } else if (auto value = EqualsValue(arg, "--aimrt-cfg")) {
      options.aimrt_cfg = *value;
    } else if (arg == "--topic-prefix" && i + 1 < argc) {
      options.topic_prefix = argv[++i];
    } else if (auto value = EqualsValue(arg, "--topic-prefix")) {
      options.topic_prefix = *value;
    } else if (arg == "--publish-hz" && i + 1 < argc) {
      options.publish_hz = std::stod(argv[++i]);
    } else if (auto value = EqualsValue(arg, "--publish-hz")) {
      options.publish_hz = std::stod(*value);
    } else if (arg == "--command-dry-run") {
      options.command_dry_run = true;
    } else if (arg == "--command-enable") {
      options.command_enable = true;
    } else if (arg == "--command-watchdog-ms" && i + 1 < argc) {
      options.command_watchdog_ms = std::stod(argv[++i]);
    } else if (auto value = EqualsValue(arg, "--command-watchdog-ms")) {
      options.command_watchdog_ms = std::stod(*value);
    } else if (arg == "--max-state-age-ms" && i + 1 < argc) {
      options.max_state_age_ms = std::stod(argv[++i]);
    } else if (auto value = EqualsValue(arg, "--max-state-age-ms")) {
      options.max_state_age_ms = std::stod(*value);
    } else if (arg == "--max-position-delta-rad" && i + 1 < argc) {
      options.max_position_delta_rad = std::stod(argv[++i]);
    } else if (auto value = EqualsValue(arg, "--max-position-delta-rad")) {
      options.max_position_delta_rad = std::stod(*value);
    } else if (arg == "--max-session-excursion-rad" && i + 1 < argc) {
      options.max_session_excursion_rad = std::stod(argv[++i]);
    } else if (auto value = EqualsValue(arg, "--max-session-excursion-rad")) {
      options.max_session_excursion_rad = std::stod(*value);
    } else if (arg == "--max-abs-velocity" && i + 1 < argc) {
      options.max_abs_velocity = std::stod(argv[++i]);
    } else if (auto value = EqualsValue(arg, "--max-abs-velocity")) {
      options.max_abs_velocity = std::stod(*value);
    } else if (arg == "--max-abs-effort" && i + 1 < argc) {
      options.max_abs_effort = std::stod(argv[++i]);
    } else if (auto value = EqualsValue(arg, "--max-abs-effort")) {
      options.max_abs_effort = std::stod(*value);
    } else if (arg == "--max-kp" && i + 1 < argc) {
      options.max_kp = std::stod(argv[++i]);
    } else if (auto value = EqualsValue(arg, "--max-kp")) {
      options.max_kp = std::stod(*value);
    } else if (arg == "--max-kd" && i + 1 < argc) {
      options.max_kd = std::stod(argv[++i]);
    } else if (auto value = EqualsValue(arg, "--max-kd")) {
      options.max_kd = std::stod(*value);
    } else {
      throw std::runtime_error("unknown or incomplete option: " + arg);
    }
  }

  if (options.aimrt_cfg.empty()) {
    throw std::runtime_error("--aimrt-cfg is required");
  }
  if (!std::isfinite(options.publish_hz) || options.publish_hz <= 0.0 ||
      options.publish_hz > 1000.0) {
    throw std::runtime_error("--publish-hz must be in (0, 1000]");
  }
  if (options.topic_prefix.empty() || options.topic_prefix.front() != '/') {
    throw std::runtime_error("--topic-prefix must start with '/'");
  }
  while (options.topic_prefix.size() > 1 &&
         options.topic_prefix.back() == '/') {
    options.topic_prefix.pop_back();
  }
  if (options.command_dry_run && options.command_enable) {
    throw std::runtime_error(
        "--command-dry-run and --command-enable are mutually exclusive");
  }
  const double limits[] = {
      options.command_watchdog_ms, options.max_state_age_ms,
      options.max_position_delta_rad, options.max_session_excursion_rad,
      options.max_abs_velocity, options.max_abs_effort, options.max_kp,
      options.max_kd};
  for (double value : limits) {
    if (!std::isfinite(value) || value < 0.0) {
      throw std::runtime_error("command limits must be finite and non-negative");
    }
  }
  if (options.command_watchdog_ms <= 0.0 || options.max_state_age_ms <= 0.0 ||
      options.max_position_delta_rad <= 0.0 ||
      options.max_session_excursion_rad <= 0.0) {
    throw std::runtime_error(
        "watchdog/state-age/position limits must be positive");
  }
  if (options.command_enable) {
    const char* confirmation = std::getenv("A3_ACTUATION_CONFIRM");
    if (!confirmation ||
        std::string(confirmation) != "ENABLE_A3_ACTUATION") {
      throw std::runtime_error(
          "--command-enable requires "
          "A3_ACTUATION_CONFIRM=ENABLE_A3_ACTUATION");
    }
  }
  return options;
}

bool ValidState(const robot_io::RobotState& state, int dof) {
  return state.sync_complete && state.sync_aligned &&
         state.q.size() == dof && state.dq.size() == dof &&
         state.tau_est.size() == dof && state.q.allFinite() &&
         state.dq.allFinite() && state.tau_est.allFinite() &&
         state.imu_quat_wxyz.allFinite() && state.imu_gyro.allFinite() &&
         state.imu_accel.allFinite() && state.has_secondary_imu &&
         state.sec_imu_quat_wxyz.allFinite() &&
         state.sec_imu_gyro.allFinite() && state.sec_imu_accel.allFinite();
}

sensor_msgs::msg::Imu ToImu(const rclcpp::Time& stamp,
                            const std::string& frame_id,
                            const Eigen::Vector4d& quat_wxyz,
                            const Eigen::Vector3d& gyro,
                            const Eigen::Vector3d& accel) {
  sensor_msgs::msg::Imu msg;
  msg.header.stamp = stamp;
  msg.header.frame_id = frame_id;
  msg.orientation.w = quat_wxyz[0];
  msg.orientation.x = quat_wxyz[1];
  msg.orientation.y = quat_wxyz[2];
  msg.orientation.z = quat_wxyz[3];
  msg.angular_velocity.x = gyro[0];
  msg.angular_velocity.y = gyro[1];
  msg.angular_velocity.z = gyro[2];
  msg.linear_acceleration.x = accel[0];
  msg.linear_acceleration.y = accel[1];
  msg.linear_acceleration.z = accel[2];
  return msg;
}

struct DryCommandValidation {
  bool ok{false};
  std::uint32_t sequence{0};
  double max_position_delta{0.0};
  std::string reason;
};

struct ControlCommandValidation {
  bool ok{false};
  std::uint32_t sequence{0};
  double max_position_delta{0.0};
  double max_session_excursion{0.0};
  robot_io::RobotCommand command;
  std::string reason;
};

void BuildSafeHalt(const robot_io::RobotState& state,
                   robot_io::RobotCommand& command) {
  constexpr int kDof = 31;
  command.q_des = state.q.size() == kDof ? state.q
                                         : Eigen::VectorXd::Zero(kDof);
  command.dq_des = Eigen::VectorXd::Zero(kDof);
  command.tau_ff = Eigen::VectorXd::Zero(kDof);
  command.kp = Eigen::VectorXd::Zero(kDof);
  command.kd = Eigen::VectorXd::Zero(kDof);
}

ControlCommandValidation ValidateControlCommand(
    const joint_msgs::msg::JointCommand& message,
    const robot_io::JointLayout& layout,
    const robot_io::RobotState& state,
    const std::optional<Eigen::VectorXd>& session_anchor,
    const Options& options) {
  ControlCommandValidation result;
  constexpr const char* kSessionPrefix = "a3_remote_control/";
  const std::size_t dof = layout.names.size();

  if (message.header.frame_id.rfind(kSessionPrefix, 0) != 0) {
    result.reason = "invalid control session marker";
    return result;
  }
  if (message.joints.size() != dof || state.q.size() != static_cast<int>(dof) ||
      !state.q.allFinite()) {
    result.reason = "expected a valid 31-DOF state and command";
    return result;
  }
  if (session_anchor &&
      session_anchor->size() != static_cast<int>(dof)) {
    result.reason = "invalid session anchor";
    return result;
  }

  result.command.q_des.resize(dof);
  result.command.dq_des.resize(dof);
  result.command.tau_ff.resize(dof);
  result.command.kp.resize(dof);
  result.command.kd.resize(dof);
  bool have_sequence = false;
  for (std::size_t i = 0; i < dof; ++i) {
    const auto& row = message.joints[i];
    if (row.name != layout.names[i]) {
      result.reason = "non-canonical joint name/order at index " +
                      std::to_string(i);
      return result;
    }
    if (!have_sequence) {
      result.sequence = row.sequence;
      have_sequence = true;
    } else if (row.sequence != result.sequence) {
      result.reason = "inconsistent per-joint sequence";
      return result;
    }
    const double values[] = {row.position, row.velocity, row.effort,
                             row.stiffness, row.damping};
    if (!std::all_of(std::begin(values), std::end(values),
                     [](double value) { return std::isfinite(value); })) {
      result.reason = "non-finite command value: " + row.name;
      return result;
    }
    if (row.stiffness < 0.0 || row.damping < 0.0 ||
        std::abs(row.velocity) > options.max_abs_velocity ||
        std::abs(row.effort) > options.max_abs_effort ||
        row.stiffness > options.max_kp || row.damping > options.max_kd) {
      result.reason = "velocity/effort/gain limit exceeded: " + row.name;
      return result;
    }
    const double delta = std::abs(row.position - state.q[i]);
    result.max_position_delta = std::max(result.max_position_delta, delta);
    if (delta > options.max_position_delta_rad) {
      result.reason = "position delta limit exceeded: " + row.name;
      return result;
    }
    const bool actively_controlled =
        row.stiffness > 0.0 || row.damping > 0.0 ||
        std::abs(row.velocity) > 0.0 || std::abs(row.effort) > 0.0;
    if (session_anchor && actively_controlled) {
      const double excursion =
          std::abs(row.position - (*session_anchor)[static_cast<int>(i)]);
      result.max_session_excursion =
          std::max(result.max_session_excursion, excursion);
      if (excursion > options.max_session_excursion_rad) {
        result.reason = "session excursion limit exceeded: " + row.name;
        return result;
      }
    }
    result.command.q_des[i] = row.position;
    result.command.dq_des[i] = row.velocity;
    result.command.tau_ff[i] = row.effort;
    result.command.kp[i] = row.stiffness;
    result.command.kd[i] = row.damping;
  }
  result.ok = true;
  return result;
}

DryCommandValidation ValidateDryCommand(
    const joint_msgs::msg::JointCommand& message,
    const robot_io::JointLayout& layout,
    const std::optional<robot_io::RobotState>& latest_state) {
  DryCommandValidation result;
  constexpr double kMaxPositionDeltaRad = 0.05;
  constexpr const char* kSessionPrefix = "a3_remote_dry/";

  if (message.header.frame_id.rfind(kSessionPrefix, 0) != 0) {
    result.reason = "invalid dry-run session marker";
    return result;
  }
  if (message.joints.size() != layout.names.size()) {
    result.reason = "expected 31 command rows";
    return result;
  }
  if (!latest_state || latest_state->q.size() != 31 ||
      !latest_state->q.allFinite()) {
    result.reason = "no valid current state for position-delta check";
    return result;
  }

  std::unordered_map<std::string, std::size_t> layout_index;
  for (std::size_t i = 0; i < layout.names.size(); ++i) {
    layout_index.emplace(layout.names[i], i);
  }
  std::unordered_set<std::string> seen;
  bool have_sequence = false;
  std::uint32_t sequence = 0;
  for (const auto& row : message.joints) {
    const auto found = layout_index.find(row.name);
    if (found == layout_index.end() || !seen.insert(row.name).second) {
      result.reason = "unknown or duplicate joint name: " + row.name;
      return result;
    }
    if (!have_sequence) {
      sequence = row.sequence;
      have_sequence = true;
    } else if (row.sequence != sequence) {
      result.reason = "inconsistent per-joint sequence";
      return result;
    }
    if (!std::isfinite(row.position) || !std::isfinite(row.velocity) ||
        !std::isfinite(row.effort) || !std::isfinite(row.stiffness) ||
        !std::isfinite(row.damping)) {
      result.reason = "non-finite command value: " + row.name;
      return result;
    }
    if (row.velocity != 0.0 || row.effort != 0.0 ||
        row.stiffness != 0.0 || row.damping != 0.0) {
      result.reason = "dry-run command must have zero velocity/effort/gains";
      return result;
    }
    const double delta =
        std::abs(row.position - latest_state->q[found->second]);
    result.max_position_delta = std::max(result.max_position_delta, delta);
    if (delta > kMaxPositionDeltaRad) {
      result.reason = "position differs from current state by more than 0.05 rad";
      return result;
    }
  }

  result.ok = true;
  result.sequence = sequence;
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  try {
    options = ParseOptions(argc, argv);
  } catch (const std::exception& e) {
    std::cerr << "[a3_mdu_state_bridge] " << e.what() << "\n";
    Usage(argv[0]);
    return 64;
  }

  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("a3_mdu_state_bridge");
  auto backend = robot_io::CreateBackend("a3");
  if (!backend) {
    RCLCPP_ERROR(node->get_logger(), "A3 RobotIOBackend is unavailable");
    rclcpp::shutdown();
    return 1;
  }

  const auto& layout = backend->GetLayout();
  const int dof = layout.dof();
  if (dof != 31) {
    RCLCPP_ERROR(node->get_logger(), "expected 31 joints, got %d", dof);
    rclcpp::shutdown();
    return 1;
  }

  std::mutex mutex;
  std::optional<robot_io::RobotState> latest;
  std::optional<std::chrono::steady_clock::time_point> latest_received_at;
  std::atomic<std::uint64_t> received{0};
  std::atomic<std::uint64_t> rejected{0};
  std::atomic<std::uint64_t> dry_command_received{0};
  std::atomic<std::uint64_t> dry_command_accepted{0};
  std::atomic<std::uint64_t> dry_command_rejected{0};
  std::atomic<std::uint64_t> control_command_received{0};
  std::atomic<std::uint64_t> control_command_accepted{0};
  std::atomic<std::uint64_t> control_command_rejected{0};
  std::atomic<std::uint64_t> send_command_calls{0};
  std::atomic<std::uint64_t> watchdog_halts{0};

  backend->RegisterStateCallback([&](const robot_io::RobotState& state) {
    received.fetch_add(1, std::memory_order_relaxed);
    if (!ValidState(state, dof)) {
      rejected.fetch_add(1, std::memory_order_relaxed);
      return;
    }
    std::lock_guard<std::mutex> lock(mutex);
    latest = state;
    latest_received_at = std::chrono::steady_clock::now();
  });

  const std::string backend_options =
      "cfg_file_path=" + options.aimrt_cfg + ",publish_enabled=" +
      (options.command_enable ? "true" : "false");
  if (!backend->Init(backend_options) || !backend->Start()) {
    RCLCPP_ERROR(node->get_logger(), "failed to start RobotIOBackend");
    rclcpp::shutdown();
    return 1;
  }

  const auto qos = rclcpp::QoS(rclcpp::KeepLast(16))
                       .reliable()
                       .durability_volatile();
  auto joints_pub = node->create_publisher<sensor_msgs::msg::JointState>(
      options.topic_prefix + "/joint_states", qos);
  auto pelvis_pub = node->create_publisher<sensor_msgs::msg::Imu>(
      options.topic_prefix + "/pelvis_imu", qos);
  auto torso_pub = node->create_publisher<sensor_msgs::msg::Imu>(
      options.topic_prefix + "/torso_imu", qos);

  rclcpp::Publisher<std_msgs::msg::UInt32>::SharedPtr command_ack_pub;
  rclcpp::Subscription<joint_msgs::msg::JointCommand>::SharedPtr
      dry_command_sub;
  rclcpp::Publisher<std_msgs::msg::UInt32>::SharedPtr control_ack_pub;
  rclcpp::Subscription<joint_msgs::msg::JointCommand>::SharedPtr
      control_command_sub;
  std::string last_dry_session;
  std::optional<std::uint32_t> last_dry_sequence;
  if (options.command_dry_run) {
    const auto command_qos = rclcpp::QoS(rclcpp::KeepLast(64))
                                 .reliable()
                                 .durability_volatile();
    command_ack_pub = node->create_publisher<std_msgs::msg::UInt32>(
        options.topic_prefix + "/joint_command_ack", command_qos);
    dry_command_sub =
        node->create_subscription<joint_msgs::msg::JointCommand>(
            options.topic_prefix + "/joint_command_dry_run", command_qos,
            [&](const joint_msgs::msg::JointCommand::SharedPtr message) {
              dry_command_received.fetch_add(1, std::memory_order_relaxed);
              if (!message) {
                dry_command_rejected.fetch_add(1,
                                               std::memory_order_relaxed);
                return;
              }
              std::optional<robot_io::RobotState> state_snapshot;
              {
                std::lock_guard<std::mutex> lock(mutex);
                state_snapshot = latest;
              }
              auto validation =
                  ValidateDryCommand(*message, layout, state_snapshot);
              if (validation.ok) {
                const std::string& session = message->header.frame_id;
                if (session != last_dry_session) {
                  last_dry_session = session;
                  last_dry_sequence.reset();
                }
                if (last_dry_sequence &&
                    validation.sequence <= *last_dry_sequence) {
                  validation.ok = false;
                  validation.reason = "non-increasing sequence";
                }
              }
              if (!validation.ok) {
                const auto count = dry_command_rejected.fetch_add(
                                       1, std::memory_order_relaxed) +
                                   1;
                if (count <= 5 || count % 100 == 0) {
                  RCLCPP_WARN(node->get_logger(),
                              "dry command rejected: %s",
                              validation.reason.c_str());
                }
                return;
              }

              last_dry_sequence = validation.sequence;
              const auto accepted_count = dry_command_accepted.fetch_add(
                                              1, std::memory_order_relaxed) +
                                          1;
              std_msgs::msg::UInt32 ack;
              ack.data = validation.sequence;
              command_ack_pub->publish(ack);
              if (accepted_count <= 5 || accepted_count % 100 == 0) {
                RCLCPP_INFO(node->get_logger(),
                            "dry command accepted: seq=%u max_delta=%.6frad "
                            "accepted=%llu SendCommand_calls=0",
                            validation.sequence,
                            validation.max_position_delta,
                            static_cast<unsigned long long>(accepted_count));
              }
            });
  }

  std::string control_session;
  std::optional<std::uint32_t> last_control_sequence;
  std::optional<Eigen::VectorXd> control_session_anchor;
  std::optional<std::chrono::steady_clock::time_point> last_control_at;
  bool watchdog_latched = false;
  bool watchdog_reported = false;
  if (options.command_enable) {
    const auto command_qos = rclcpp::QoS(rclcpp::KeepLast(64))
                                 .reliable()
                                 .durability_volatile();
    control_ack_pub = node->create_publisher<std_msgs::msg::UInt32>(
        options.topic_prefix + "/joint_command_control_ack", command_qos);
    control_command_sub =
        node->create_subscription<joint_msgs::msg::JointCommand>(
            options.topic_prefix + "/joint_command_control", command_qos,
            [&](const joint_msgs::msg::JointCommand::SharedPtr message) {
              control_command_received.fetch_add(1,
                                                 std::memory_order_relaxed);
              auto reject = [&](const std::string& reason) {
                const auto count = control_command_rejected.fetch_add(
                                       1, std::memory_order_relaxed) +
                                   1;
                if (count <= 5 || count % 100 == 0) {
                  RCLCPP_WARN(node->get_logger(),
                              "control command rejected: %s",
                              reason.c_str());
                }
              };
              if (!message) {
                reject("null message");
                return;
              }
              if (watchdog_latched) {
                reject("watchdog is latched; restart gateway to re-arm");
                return;
              }

              std::optional<robot_io::RobotState> state_snapshot;
              std::optional<std::chrono::steady_clock::time_point>
                  state_received_at;
              {
                std::lock_guard<std::mutex> lock(mutex);
                state_snapshot = latest;
                state_received_at = latest_received_at;
              }
              const auto now = std::chrono::steady_clock::now();
              if (!state_snapshot || !state_received_at) {
                reject("no valid synchronized state");
                return;
              }
              const double state_age_ms =
                  std::chrono::duration<double, std::milli>(
                      now - *state_received_at)
                      .count();
              if (state_age_ms > options.max_state_age_ms) {
                reject("synchronized state is stale");
                return;
              }

              const std::string& session = message->header.frame_id;
              if (!control_session.empty() && session != control_session) {
                reject("gateway is locked to a different control session");
                return;
              }
              auto validation = ValidateControlCommand(
                  *message, layout, *state_snapshot, control_session_anchor,
                  options);
              if (!validation.ok) {
                reject(validation.reason);
                return;
              }
              if (last_control_sequence &&
                  validation.sequence <= *last_control_sequence) {
                reject("non-increasing sequence");
                return;
              }
              if (!backend->SendCommand(validation.command)) {
                reject("RobotIOBackend::SendCommand failed");
                return;
              }

              if (control_session.empty()) {
                control_session = session;
                control_session_anchor = state_snapshot->q;
              }
              last_control_sequence = validation.sequence;
              last_control_at = now;
              send_command_calls.fetch_add(1, std::memory_order_relaxed);
              const auto accepted = control_command_accepted.fetch_add(
                                        1, std::memory_order_relaxed) +
                                    1;
              std_msgs::msg::UInt32 ack;
              ack.data = validation.sequence;
              control_ack_pub->publish(ack);
              if (accepted <= 5 || accepted % 100 == 0) {
                RCLCPP_INFO(node->get_logger(),
                            "CONTROL SENT: seq=%u max_delta=%.6frad "
                            "session_excursion=%.6frad accepted=%llu",
                            validation.sequence,
                            validation.max_position_delta,
                            validation.max_session_excursion,
                            static_cast<unsigned long long>(accepted));
              }
            });
  }

  RCLCPP_INFO(node->get_logger(), "%s bridge started: prefix=%s rate=%.1fHz dof=%d",
              options.command_enable ? "CONTROL-ENABLED" : "read-only",
              options.topic_prefix.c_str(), options.publish_hz, dof);
  RCLCPP_INFO(node->get_logger(), "command publishers %s: publish_enabled=%s",
              options.command_enable ? "ENABLED" : "disabled",
              options.command_enable ? "true" : "false");
  RCLCPP_INFO(node->get_logger(),
              "direct DDS QoS: state=reliable/depth16 "
              "command=reliable/depth64");
  if (options.command_dry_run) {
    RCLCPP_INFO(node->get_logger(),
                "dry command validation enabled: topic=%s "
                "SendCommand_calls=0",
                (options.topic_prefix + "/joint_command_dry_run").c_str());
  }
  if (options.command_enable) {
    RCLCPP_WARN(node->get_logger(),
                "REAL CONTROL ARMED: topic=%s watchdog=%.1fms "
                "state_age=%.1fms delta=%.4frad excursion=%.4frad "
                "|velocity|<=%.3f |effort|<=%.3f kp<=%.3f kd<=%.3f",
                (options.topic_prefix + "/joint_command_control").c_str(),
                options.command_watchdog_ms, options.max_state_age_ms,
                options.max_position_delta_rad,
                options.max_session_excursion_rad,
                options.max_abs_velocity, options.max_abs_effort,
                options.max_kp, options.max_kd);
  }

  std::int64_t last_tick = std::numeric_limits<std::int64_t>::min();
  std::uint64_t published = 0;
  auto last_log = std::chrono::steady_clock::now();
  rclcpp::WallRate rate(options.publish_hz);

  while (rclcpp::ok()) {
    rclcpp::spin_some(node);

    const auto control_now = std::chrono::steady_clock::now();
    if (options.command_enable && last_control_at &&
        std::chrono::duration<double, std::milli>(control_now - *last_control_at)
                .count() > options.command_watchdog_ms) {
      watchdog_latched = true;
      std::optional<robot_io::RobotState> halt_state;
      {
        std::lock_guard<std::mutex> lock(mutex);
        halt_state = latest;
      }
      if (halt_state && halt_state->q.size() == dof &&
          halt_state->q.allFinite()) {
        robot_io::RobotCommand halt;
        BuildSafeHalt(*halt_state, halt);
        if (backend->SendCommand(halt)) {
          send_command_calls.fetch_add(1, std::memory_order_relaxed);
          watchdog_halts.fetch_add(1, std::memory_order_relaxed);
        }
      }
      if (!watchdog_reported) {
        RCLCPP_ERROR(node->get_logger(),
                     "CONTROL WATCHDOG LATCHED: zero-gain safe-halt active; "
                     "restart gateway to re-arm");
        watchdog_reported = true;
      }
    }

    std::optional<robot_io::RobotState> state;
    {
      std::lock_guard<std::mutex> lock(mutex);
      if (latest && latest->tick != last_tick) state = *latest;
    }

    if (state) {
      const auto stamp = node->now();
      sensor_msgs::msg::JointState joints;
      joints.header.stamp = stamp;
      joints.header.frame_id = "a3_base";
      joints.name = layout.names;
      joints.position.assign(state->q.data(), state->q.data() + dof);
      joints.velocity.assign(state->dq.data(), state->dq.data() + dof);
      joints.effort.assign(state->tau_est.data(), state->tau_est.data() + dof);

      joints_pub->publish(joints);
      pelvis_pub->publish(ToImu(stamp, "pelvis_imu", state->imu_quat_wxyz,
                                state->imu_gyro, state->imu_accel));
      torso_pub->publish(ToImu(
          stamp, "torso_imu", state->sec_imu_quat_wxyz,
          state->sec_imu_gyro, state->sec_imu_accel));
      last_tick = state->tick;
      ++published;
    }

    const auto now = std::chrono::steady_clock::now();
    if (now - last_log >= std::chrono::seconds(1)) {
      RCLCPP_INFO(node->get_logger(),
                  "received=%llu published=%llu rejected=%llu last_tick=%lld "
                  "dry_cmd={received:%llu accepted:%llu rejected:%llu} "
                  "control_cmd={received:%llu accepted:%llu rejected:%llu} "
                  "SendCommand_calls=%llu watchdog_halts=%llu",
                  static_cast<unsigned long long>(received.load()),
                  static_cast<unsigned long long>(published),
                  static_cast<unsigned long long>(rejected.load()),
                  static_cast<long long>(last_tick),
                  static_cast<unsigned long long>(dry_command_received.load()),
                  static_cast<unsigned long long>(dry_command_accepted.load()),
                  static_cast<unsigned long long>(dry_command_rejected.load()),
                  static_cast<unsigned long long>(control_command_received.load()),
                  static_cast<unsigned long long>(control_command_accepted.load()),
                  static_cast<unsigned long long>(control_command_rejected.load()),
                  static_cast<unsigned long long>(send_command_calls.load()),
                  static_cast<unsigned long long>(watchdog_halts.load()));
      last_log = now;
    }
    rate.sleep();
  }

  if (options.command_enable && last_control_at) {
    std::optional<robot_io::RobotState> halt_state;
    {
      std::lock_guard<std::mutex> lock(mutex);
      halt_state = latest;
    }
    if (halt_state && halt_state->q.size() == dof && halt_state->q.allFinite()) {
      robot_io::RobotCommand halt;
      BuildSafeHalt(*halt_state, halt);
      backend->SendCommand(halt);
    }
  }

  backend->RegisterStateCallback({});
  backend->Stop();
  rclcpp::shutdown();
  return 0;
}
