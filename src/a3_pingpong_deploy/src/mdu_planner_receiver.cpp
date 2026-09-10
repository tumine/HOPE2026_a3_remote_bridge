#include "a3_pingpong/a3_leg_limits.hpp"
#include "a3_pingpong/external_observation_guard.hpp"
#include "a3_pingpong/lateral_station.hpp"
#include "a3_pingpong/onnx_actor.hpp"
#include "a3_pingpong/manual_control.hpp"
#include "a3_pingpong/pingpong_action_adapter.hpp"
#include "a3_pingpong/pingpong_observation_builder.hpp"
#include "a3_pingpong/planner_input.hpp"
#include "a3_pingpong/planner_udp_receiver.hpp"
#include "a3_pingpong/receive_controller.hpp"
#include "a3_pingpong/swing_lifecycle.hpp"
#include "a3_pingpong/upper_body_serve.hpp"

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <hope_msgs/msg/racket_command.hpp>
#include <rclcpp/rclcpp.hpp>
#include <robot_io/robot_io_backend.hpp>

#include <poll.h>
#include <termios.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

struct Options {
  std::string input_transport{"udp"};
  std::string command_topic{"/racket/command"};
  std::string base_pose_topic{"/a3_mocap/pelvis_pose"};
  std::string expected_frame{"hope_table"};
  std::string udp_bind_address{"192.168.1.100"};
  std::string udp_source_address{"192.168.1.11"};
  std::uint16_t udp_port{15001};
  std::string aimrt_cfg;
  double command_timeout_ms{150.0};
  double base_pose_timeout_ms{100.0};
  double external_fallback_ms{0.0};
  double state_timeout_ms{0.0};
  double control_hz{50.0};
  double status_period_s{5.0};
  double leg_soft_scale{0.90};
  double leg_damping_kd{2.0};
  std::vector<std::string> leg_soft_limit_specs;
  bool observation_probe{false};
  bool onnx_requested{false};
  bool action_dry_run{false};
  bool upper_body_serve_dry_run{false};
  bool serve_vcf{false};
  std::string serve_tracks_dir;
  bool gripper_http{false};
  a3_pingpong::GripperHttpConfig gripper_config;
  bool manual_control{false};
  bool publish_commands{false};
  std::string onnx_model;
  a3_pingpong::ReceiveControllerOptions::LegDampingSafety leg_damping_safety;
  a3_pingpong::ReceiveControllerOptions::WaistPitchSafety waist_pitch_safety;
};

void Usage(const char* program) {
  std::cout
      << "Usage: " << program << " [options]\n\n"
      << "Receive PC-side HOPE planner inputs and run the MDU policy.\n"
      << "Command publishing is disabled unless explicitly requested.\n\n"
      << "Options:\n"
      << "  --input-transport MODE     udp (default) or ros2\n"
      << "  --command-topic TOPIC       (default: /racket/command)\n"
      << "  --base-pose-topic TOPIC     (default: /a3_mocap/pelvis_pose)\n"
      << "  --expected-frame FRAME      (default: hope_table)\n"
      << "  --udp-bind-address IPv4     (default: 192.168.1.100)\n"
      << "  --udp-source-address IPv4   allowed PC (default: 192.168.1.11)\n"
      << "  --udp-port PORT             (default: 15001)\n"
      << "  --aimrt-cfg PATH            enable the RobotIO backend\n"
      << "  --observation-probe         build 111-D observations; no inference\n"
      << "  --onnx-model PATH           run model_72500 inference\n"
      << "  --action-dry-run            build full RobotCommand; do not send\n"
      << "  --serve-vcf                 MuJoCo-aligned V/C/F serve state machine\n"
      << "  --serve-tracks-dir PATH     hot-reloaded serve YAML directory\n"
      << "  --gripper-http              actuate proven HalHandService endpoint\n"
      << "  --gripper-host HOST         (default: 10.42.10.12)\n"
      << "  --gripper-port PORT         (default: 56422)\n"
      << "  --upper-body-serve-dry-run  compatibility alias for --serve-vcf\n"
      << "  --manual-control            P/S/M/V/C/F/G/R/X keyboard state machine\n"
      << "  --publish-commands          enable official RobotIO SendCommand path\n"
      << "  --command-timeout-ms MS     (default: 150)\n"
      << "  --base-pose-timeout-ms MS   (default: 100)\n"
      << "  --external-fallback-ms MS   0=hold last pose indefinitely (default: 0)\n"
      << "  --state-timeout-ms MS       0=disable RobotIO age/sync watchdog (default: 0)\n"
      << "  --control-hz HZ             (default: 50)\n"
      << "  --leg-soft-scale SCALE     URDF endpoint scale (default: 0.90)\n"
      << "  --leg-soft-limit SPEC      NAME:LOWER:UPPER; repeat per joint\n"
      << "  --leg-damping-kd KD        leg Kd in latched damping (default: 2.0)\n"
      << "  --waist-pitch-guard        enable optional positive-pitch virtual wall\n"
      << "  --no-waist-pitch-guard     disable waist-pitch virtual wall (default)\n"
      << "  --waist-pitch-enter RAD    trigger angle (default: 0.35)\n"
      << "  --waist-pitch-release RAD  hysteresis release angle (default: 0.27)\n"
      << "  --waist-pitch-target RAD   recovery target (default: 0.00)\n"
      << "  --waist-pitch-kp KP        recovery Kp, max 500 (default: 400)\n"
      << "  --waist-pitch-kd KD        recovery Kd, max 8 (default: 8)\n"
      << "  --status-period-s SEC       status log period (default: 5)\n"
      << "  -h, --help\n";
}

std::optional<std::string> EqualsValue(const std::string& argument,
                                       const std::string& key) {
  const std::string prefix = key + "=";
  if (argument.rfind(prefix, 0) != 0) return std::nullopt;
  return argument.substr(prefix.size());
}

Options ParseOptions(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument(argv[index]);
    if (argument == "-h" || argument == "--help") {
      Usage(argv[0]);
      std::exit(0);
    }
    if (argument == "--observation-probe") {
      options.observation_probe = true;
      continue;
    }
    if (argument == "--action-dry-run") {
      options.action_dry_run = true;
      continue;
    }
    if (argument == "--upper-body-serve-dry-run") {
      options.upper_body_serve_dry_run = true;
      options.serve_vcf = true;
      options.observation_probe = true;
      options.manual_control = true;
      continue;
    }
    if (argument == "--serve-vcf") {
      options.serve_vcf = true;
      options.observation_probe = true;
      options.manual_control = true;
      continue;
    }
    if (argument == "--gripper-http") {
      options.gripper_http = true;
      continue;
    }
    if (argument == "--manual-control") {
      options.manual_control = true;
      continue;
    }
    if (argument == "--publish-commands") {
      options.publish_commands = true;
      options.observation_probe = true;
      continue;
    }
    if (argument == "--waist-pitch-guard") {
      options.waist_pitch_safety.enabled = true;
      continue;
    }
    if (argument == "--no-waist-pitch-guard") {
      options.waist_pitch_safety.enabled = false;
      continue;
    }
    auto take_value = [&](const std::string& key, std::string& output) {
      if (argument == key && index + 1 < argc) {
        output = argv[++index];
        return true;
      }
      if (auto value = EqualsValue(argument, key)) {
        output = *value;
        return true;
      }
      return false;
    };
    std::string numeric;
    if (take_value("--input-transport", options.input_transport) ||
        take_value("--command-topic", options.command_topic) ||
        take_value("--base-pose-topic", options.base_pose_topic) ||
        take_value("--expected-frame", options.expected_frame) ||
        take_value("--udp-bind-address", options.udp_bind_address) ||
        take_value("--udp-source-address", options.udp_source_address) ||
        take_value("--aimrt-cfg", options.aimrt_cfg) ||
        take_value("--serve-tracks-dir", options.serve_tracks_dir) ||
        take_value("--gripper-host", options.gripper_config.host) ||
        take_value("--gripper-path", options.gripper_config.path)) {
      continue;
    }
    if (take_value("--onnx-model", options.onnx_model)) {
      options.observation_probe = true;
      options.onnx_requested = true;
      continue;
    }
    if (take_value("--command-timeout-ms", numeric)) {
      options.command_timeout_ms = std::stod(numeric);
      continue;
    }
    if (take_value("--base-pose-timeout-ms", numeric)) {
      options.base_pose_timeout_ms = std::stod(numeric);
      continue;
    }
    if (take_value("--external-fallback-ms", numeric)) {
      options.external_fallback_ms = std::stod(numeric);
      continue;
    }
    if (take_value("--state-timeout-ms", numeric)) {
      options.state_timeout_ms = std::stod(numeric);
      continue;
    }
    if (take_value("--control-hz", numeric)) {
      options.control_hz = std::stod(numeric);
      continue;
    }
    if (take_value("--leg-soft-scale", numeric)) {
      options.leg_soft_scale = std::stod(numeric);
      continue;
    }
    if (take_value("--leg-soft-limit", numeric)) {
      options.leg_soft_limit_specs.push_back(numeric);
      continue;
    }
    if (take_value("--leg-damping-kd", numeric)) {
      options.leg_damping_kd = std::stod(numeric);
      continue;
    }
    if (take_value("--waist-pitch-enter", numeric)) {
      options.waist_pitch_safety.enter_rad = std::stod(numeric);
      continue;
    }
    if (take_value("--waist-pitch-release", numeric)) {
      options.waist_pitch_safety.release_rad = std::stod(numeric);
      continue;
    }
    if (take_value("--waist-pitch-target", numeric)) {
      options.waist_pitch_safety.recovery_target_rad = std::stod(numeric);
      continue;
    }
    if (take_value("--waist-pitch-kp", numeric)) {
      options.waist_pitch_safety.recovery_kp = std::stod(numeric);
      continue;
    }
    if (take_value("--waist-pitch-kd", numeric)) {
      options.waist_pitch_safety.recovery_kd = std::stod(numeric);
      continue;
    }
    if (take_value("--status-period-s", numeric)) {
      options.status_period_s = std::stod(numeric);
      continue;
    }
    if (take_value("--udp-port", numeric)) {
      const int port = std::stoi(numeric);
      if (port < 1 || port > 65535) {
        throw std::runtime_error("--udp-port must be in [1, 65535]");
      }
      options.udp_port = static_cast<std::uint16_t>(port);
      continue;
    }
    if (take_value("--gripper-port", numeric)) {
      const int port = std::stoi(numeric);
      if (port < 1 || port > 65535) {
        throw std::runtime_error("--gripper-port must be in [1, 65535]");
      }
      options.gripper_config.port = static_cast<std::uint16_t>(port);
      continue;
    }
    if (take_value("--gripper-open-position", numeric)) {
      options.gripper_config.open_position = std::stoi(numeric);
      continue;
    }
    if (take_value("--gripper-close-position", numeric)) {
      options.gripper_config.close_position = std::stoi(numeric);
      continue;
    }
    throw std::runtime_error("unknown or incomplete option: " + argument);
  }

  const auto valid_topic = [](const std::string& value) {
    return !value.empty() && value.front() == '/';
  };
  if (options.input_transport != "udp" && options.input_transport != "ros2") {
    throw std::runtime_error("--input-transport must be udp or ros2");
  }
  if (options.input_transport == "ros2" &&
      (!valid_topic(options.command_topic) ||
       !valid_topic(options.base_pose_topic))) {
    throw std::runtime_error("topics must be absolute ROS names");
  }
  if (options.udp_bind_address.empty() || options.udp_source_address.empty()) {
    throw std::runtime_error("UDP IPv4 addresses cannot be empty");
  }
  if (options.expected_frame.empty()) {
    throw std::runtime_error("--expected-frame cannot be empty");
  }
  if (options.serve_vcf && options.onnx_model.empty()) {
    throw std::runtime_error("--serve-vcf requires --onnx-model");
  }
  if (options.serve_vcf && options.serve_tracks_dir.empty()) {
    throw std::runtime_error("--serve-vcf requires --serve-tracks-dir");
  }
  if (options.gripper_http &&
      (!options.serve_vcf || !options.publish_commands)) {
    throw std::runtime_error(
        "--gripper-http requires --serve-vcf and --publish-commands");
  }
  if (options.observation_probe && options.aimrt_cfg.empty()) {
    throw std::runtime_error(
        "--observation-probe requires --aimrt-cfg");
  }
  if (options.manual_control && !options.observation_probe) {
    throw std::runtime_error(
        "--manual-control requires --observation-probe");
  }
  if (options.onnx_requested && options.onnx_model.empty()) {
    throw std::runtime_error("--onnx-model cannot be empty");
  }
  if (options.action_dry_run && options.onnx_model.empty()) {
    throw std::runtime_error(
        "--action-dry-run requires --onnx-model");
  }
  if (options.publish_commands &&
      (!options.manual_control || options.onnx_model.empty())) {
    throw std::runtime_error(
        "--publish-commands requires --manual-control and --onnx-model");
  }
  if (!std::isfinite(options.command_timeout_ms) ||
      options.command_timeout_ms <= 0.0 ||
      !std::isfinite(options.base_pose_timeout_ms) ||
      options.base_pose_timeout_ms <= 0.0 ||
      !std::isfinite(options.external_fallback_ms) ||
      options.external_fallback_ms < 0.0 ||
      (options.external_fallback_ms > 0.0 &&
       options.external_fallback_ms <= options.base_pose_timeout_ms) ||
      !std::isfinite(options.state_timeout_ms) ||
      options.state_timeout_ms < 0.0 ||
      !std::isfinite(options.control_hz) || options.control_hz <= 0.0 ||
      !std::isfinite(options.leg_soft_scale) ||
      options.leg_soft_scale <= 0.0 || options.leg_soft_scale > 1.0 ||
      !std::isfinite(options.leg_damping_kd) ||
      options.leg_damping_kd <= 0.0 ||
      !std::isfinite(options.status_period_s) ||
      options.status_period_s <= 0.0) {
    throw std::runtime_error("timeouts must be finite and positive");
  }

  // Scale each signed URDF endpoint toward zero. This preserves asymmetric
  // joint ranges such as hip-roll and knee exactly as represented in URDF.
  for (std::size_t index = 0; index < a3_pingpong::kA3LegDof; ++index) {
    options.leg_damping_safety.lower[index] =
        a3_pingpong::kA3LegUrdfLower[index] * options.leg_soft_scale;
    options.leg_damping_safety.upper[index] =
        a3_pingpong::kA3LegUrdfUpper[index] * options.leg_soft_scale;
  }
  for (const std::string& spec : options.leg_soft_limit_specs) {
    const std::size_t first_separator = spec.find(':');
    const std::size_t second_separator =
        first_separator == std::string::npos
            ? std::string::npos
            : spec.find(':', first_separator + 1);
    if (first_separator == std::string::npos ||
        second_separator == std::string::npos ||
        spec.find(':', second_separator + 1) != std::string::npos) {
      throw std::runtime_error(
          "--leg-soft-limit must be NAME:LOWER:UPPER");
    }
    const std::string name = spec.substr(0, first_separator);
    const double lower = std::stod(
        spec.substr(first_separator + 1,
                    second_separator - first_separator - 1));
    const double upper = std::stod(spec.substr(second_separator + 1));
    std::size_t leg_index = a3_pingpong::kA3LegDof;
    for (std::size_t index = 0; index < a3_pingpong::kA3LegDof; ++index) {
      if (name == a3_pingpong::kA3LegJointNames[index]) {
        leg_index = index;
        break;
      }
    }
    if (leg_index == a3_pingpong::kA3LegDof || !std::isfinite(lower) ||
        !std::isfinite(upper) || lower >= upper ||
        lower < a3_pingpong::kA3LegUrdfLower[leg_index] ||
        upper > a3_pingpong::kA3LegUrdfUpper[leg_index]) {
      throw std::runtime_error(
          "--leg-soft-limit is invalid or outside the A3 mechanical range: " +
          spec);
    }
    options.leg_damping_safety.lower[leg_index] = lower;
    options.leg_damping_safety.upper[leg_index] = upper;
  }
  options.leg_damping_safety.damping_kd = options.leg_damping_kd;
  if (!a3_pingpong::ValidateWaistPitchSafety(options.waist_pitch_safety)) {
    throw std::runtime_error(
        "invalid waist-pitch guard: require -0.488692<=target<release<"
        "enter<=0.418879, 0<Kp<=500 and 0<Kd<=8");
  }
  return options;
}

std::int64_t StampNanoseconds(const builtin_interfaces::msg::Time& stamp) {
  return static_cast<std::int64_t>(stamp.sec) * 1'000'000'000LL +
         static_cast<std::int64_t>(stamp.nanosec);
}

const char* ResultName(a3_pingpong::InputUpdateResult result) {
  switch (result) {
    case a3_pingpong::InputUpdateResult::kAccepted:
      return "accepted";
    case a3_pingpong::InputUpdateResult::kDuplicate:
      return "duplicate";
    case a3_pingpong::InputUpdateResult::kStale:
      return "stale";
    case a3_pingpong::InputUpdateResult::kInvalid:
      return "invalid";
  }
  return "unknown";
}

const char* TickResultName(a3_pingpong::ReceiveTickResult result) {
  switch (result) {
    case a3_pingpong::ReceiveTickResult::kNoState:
      return "no_state";
    case a3_pingpong::ReceiveTickResult::kPlannerInputNotReady:
      return "planner_not_ready";
    case a3_pingpong::ReceiveTickResult::kStateStale:
      return "state_stale";
    case a3_pingpong::ReceiveTickResult::kObservationRejected:
      return "observation_rejected";
    case a3_pingpong::ReceiveTickResult::kPolicyUnavailable:
      return "policy_unavailable";
    case a3_pingpong::ReceiveTickResult::kPolicyRejected:
      return "policy_rejected";
    case a3_pingpong::ReceiveTickResult::kCommandInvalid:
      return "command_invalid";
    case a3_pingpong::ReceiveTickResult::kLegLimitDamping:
      return "leg_limit_damping";
    case a3_pingpong::ReceiveTickResult::kCommandSent:
      return "command_sent";
    case a3_pingpong::ReceiveTickResult::kDryRun:
      return "dry_run";
  }
  return "unknown";
}

const char* LegJointName(int flat_index) {
  if (flat_index < 19 || flat_index >= 31) return "unknown";
  return a3_pingpong::kA3LegJointNames[
      static_cast<std::size_t>(flat_index - a3_pingpong::kA3LegCommandStart)];
}

const char* SwingPhaseName(a3_pingpong::SwingPhase phase) {
  switch (phase) {
    case a3_pingpong::SwingPhase::kReady: return "ready";
    case a3_pingpong::SwingPhase::kSwing: return "swing";
    case a3_pingpong::SwingPhase::kFollowThrough: return "follow_through";
    case a3_pingpong::SwingPhase::kRecovery: return "recovery";
  }
  return "unknown";
}

struct PolicyLifecycleDiagnostics {
  a3_pingpong::SwingPhase phase{a3_pingpong::SwingPhase::kReady};
  std::optional<std::uint64_t> active_task_id;
  std::uint32_t active_revision{0};
  std::int8_t swing_side{1};
  std::array<double, 3> target_position_w{};
  std::array<double, 3> target_velocity_w{};
  std::array<double, 2> nominal_station_xy{};
  std::array<double, 2> base_target_xy{};
  std::array<double, 2> current_base_xy{};
  std::array<double, 2> station_error_xy{};
  double observed_tts{1.0};
  // Canonical A3 waist order: yaw, roll, pitch. policy_raw is the
  // dimensionless ONNX output, q_* is in rad, and tau_* is in N*m.
  bool waist_policy_valid{false};
  std::array<double, 3> waist_policy_raw{};
  std::array<double, 3> waist_q_command{};
  std::array<double, 3> waist_q_feedback{};
  std::array<double, 3> waist_tau_theoretical{};
  std::array<double, 3> waist_tau_feedback{};
};

enum class ServeGripperState : int {
  kOpen,
  kClosing,
  kClosed,
  kOpening,
  kFault,
};

const char* ServeGripperStateName(ServeGripperState state) noexcept {
  switch (state) {
    case ServeGripperState::kOpen: return "open";
    case ServeGripperState::kClosing: return "closing";
    case ServeGripperState::kClosed: return "closed";
    case ServeGripperState::kOpening: return "opening";
    case ServeGripperState::kFault: return "fault";
  }
  return "unknown";
}

double SmoothStep01(double value) noexcept {
  const double t = std::clamp(value, 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}

class ObservationProbe {
 public:
  ObservationProbe(const std::string& onnx_model, bool command_output_enabled,
                   bool manual_control, bool upper_body_serve_enabled,
                   const std::string& serve_tracks_dir,
                   bool gripper_http,
                   const a3_pingpong::GripperHttpConfig& gripper_config,
                   double control_hz, double command_timeout_s,
                   double base_pose_timeout_s, double external_fallback_s)
      : builder_(a3_pingpong::Model50000ObservationConfig()),
        action_adapter_(a3_pingpong::Model72500ActionAdapterConfig()),
        command_output_enabled_(command_output_enabled),
        manual_control_enabled_(manual_control),
        upper_body_serve_enabled_(upper_body_serve_enabled),
        serve_tracks_dir_(serve_tracks_dir),
        control_dt_s_(1.0 / control_hz),
        command_timeout_s_(command_timeout_s),
        base_pose_timeout_s_(base_pose_timeout_s),
        lifecycle_config_(a3_pingpong::Model50000SwingLifecycleConfig()),
        station_config_(a3_pingpong::Model50000LateralStationConfig()),
        lifecycle_(lifecycle_config_),
        external_guard_({base_pose_timeout_s, external_fallback_s,
                         kExternalRecoveryFrames}) {
    if (!onnx_model.empty()) {
      actor_ = std::make_unique<a3_pingpong::OnnxActor>(onnx_model);
    }
    if (upper_body_serve_enabled_) {
      std::string reason;
      const auto profile = a3_pingpong::LoadUpperBodyServeProfile(
          ServeProfilePath(1), &reason);
      if (!profile) {
        throw std::runtime_error("initial serve YAML invalid: " + reason);
      }
      active_serve_profile_ = *profile;
      upper_body_serve_ = a3_pingpong::UpperBodyServeTrajectory(
          active_serve_profile_.trajectory);
      LogServeProfile(1, active_serve_profile_);
    }
    if (gripper_http) {
      gripper_client_ =
          std::make_unique<a3_pingpong::GripperHttpClient>(gripper_config);
      std::string reason;
      if (!gripper_client_->Start(&reason)) {
        throw std::runtime_error("failed to start gripper client: " + reason);
      }
    }
  }

  bool Evaluate(const a3_pingpong::PlannerInputSnapshot& planner,
                const robot_io::RobotState& state) {
    latest_command_ready_ = false;

    if (manual_control_enabled_) {
      a3_pingpong::ManualMode mode;
      {
        std::lock_guard<std::mutex> lock(manual_mutex_);
        if (manual_control_.epoch() != observed_manual_epoch_) {
          observed_manual_epoch_ = manual_control_.epoch();
          lifecycle_.Reset();
          last_action_.fill(0.0F);
          nominal_station_initialized_ = false;
          phase_report_initialized_ = false;
          pd_stand_initialized_ = false;
          pd_stand_elapsed_ticks_ = 0;
          {
            std::lock_guard<std::mutex> external_lock(external_mutex_);
            external_guard_.Reset();
          }
          external_fallback_pd_initialized_ = false;
          external_fallback_pd_elapsed_ticks_ = 0;
          external_fallback_pd_ready_.store(false);
          {
            std::lock_guard<std::mutex> serve_lock(serve_mutex_);
            ResetServeLocked();
          }
        }
        mode = manual_control_.mode();
      }

      if (mode == a3_pingpong::ManualMode::kPdStand) {
        const auto& default_q = builder_.config().default_q;
        if (state.q.size() != static_cast<Eigen::Index>(default_q.size()) ||
            !state.q.array().isFinite().all()) {
          rejected_count_.fetch_add(1, std::memory_order_relaxed);
          return false;
        }
        if (!pd_stand_initialized_) {
          for (std::size_t index = 0; index < pd_stand_start_q_.size();
               ++index) {
            pd_stand_start_q_[index] =
                state.q[static_cast<Eigen::Index>(index)];
          }
          pd_stand_initialized_ = true;
        }
        bool ready = pd_stand_elapsed_ticks_ >= kPdStandRampTicks;
        if (command_output_enabled_) {
          std::string reason;
          if (!a3_pingpong::BuildPdStandCommand(
                  pd_stand_start_q_, default_q, pd_stand_elapsed_ticks_,
                  kPdStandRampTicks, latest_command_, &ready, &reason)) {
            rejected_count_.fetch_add(1, std::memory_order_relaxed);
            return false;
          }
          latest_command_ready_ = true;
        }
        {
          std::lock_guard<std::mutex> lock(manual_mutex_);
          manual_control_.SetPdStandReady(ready);
        }
        if (pd_stand_elapsed_ticks_ < kPdStandRampTicks) {
          ++pd_stand_elapsed_ticks_;
        }
        return true;
      }

      if (mode != a3_pingpong::ManualMode::kMotion) {
        if (command_output_enabled_) {
          if (mode == a3_pingpong::ManualMode::kPassive) {
            PreparePassiveDampingCommand(state);
          } else {
            PrepareSafeHaltCommand(state);
          }
        }
        return true;
      }
    }

    if (external_resume_requested_.exchange(false)) {
      lifecycle_.Reset();
      last_action_.fill(0.0F);
      nominal_station_initialized_ = false;
      phase_report_initialized_ = false;
      external_fallback_pd_initialized_ = false;
      external_fallback_pd_elapsed_ticks_ = 0;
      external_fallback_pd_ready_.store(false);
      std::lock_guard<std::mutex> serve_lock(serve_mutex_);
      ResetServeLocked();
      std::clog << "[external_observation] M accepted: lifecycle reset; "
                   "receive restarts from READY\n";
    }

    a3_pingpong::ExternalObservationStatus external_status;
    {
      std::lock_guard<std::mutex> external_lock(external_mutex_);
      external_status = external_guard_.Update(
          planner, a3_pingpong::SteadyClock::now());
    }
    external_mode_.store(external_status.mode, std::memory_order_relaxed);
    external_stale_s_.store(external_status.stale_duration_s,
                            std::memory_order_relaxed);
    external_recovery_streak_.store(external_status.recovery_streak,
                                    std::memory_order_relaxed);
    if (external_status.mode != last_external_mode_) {
      std::clog << "[external_observation] mode="
                << a3_pingpong::ExternalObservationModeName(
                       external_status.mode)
                << " stale_ms=" << external_status.stale_duration_s * 1000.0
                << " recovery_streak=" << external_status.recovery_streak
                << '\n';
      last_external_mode_ = external_status.mode;
    }

    if (external_status.fallback_latched) {
      if (state.q.size() !=
              static_cast<Eigen::Index>(pd_stand_start_q_.size()) ||
          !state.q.array().isFinite().all()) {
        rejected_count_.fetch_add(1, std::memory_order_relaxed);
        return false;
      }
      if (!external_fallback_pd_initialized_) {
        for (std::size_t index = 0; index < external_fallback_pd_start_q_.size();
             ++index) {
          external_fallback_pd_start_q_[index] =
              state.q[static_cast<Eigen::Index>(index)];
        }
        external_fallback_pd_initialized_ = true;
        external_fallback_pd_elapsed_ticks_ = 0;
        lifecycle_.Reset();
        last_action_.fill(0.0F);
        nominal_station_initialized_ = false;
        phase_report_initialized_ = false;
        latest_tts_.store(
            static_cast<float>(lifecycle_config_.ready_time_to_strike_s),
            std::memory_order_relaxed);
        {
          std::lock_guard<std::mutex> diagnostics_lock(diagnostics_mutex_);
          diagnostics_.phase = a3_pingpong::SwingPhase::kReady;
          diagnostics_.active_task_id.reset();
          diagnostics_.active_revision = 0;
          diagnostics_.swing_side = lifecycle_config_.ready_swing_side;
          diagnostics_.target_position_w = {};
          diagnostics_.target_velocity_w = {};
          diagnostics_.observed_tts =
              lifecycle_config_.ready_time_to_strike_s;
          diagnostics_.waist_policy_valid = false;
        }
        {
          std::lock_guard<std::mutex> serve_lock(serve_mutex_);
          ResetServeLocked();
        }
        std::clog << "[external_observation] long outage: old task and VCF "
                     "cleared; entering receive-default full-body PD\n";
      }
      bool ready =
          external_fallback_pd_elapsed_ticks_ >= kPdStandRampTicks;
      if (command_output_enabled_) {
        std::string reason;
        if (!a3_pingpong::BuildPdStandCommand(
                external_fallback_pd_start_q_, builder_.config().default_q,
                external_fallback_pd_elapsed_ticks_, kPdStandRampTicks,
                latest_command_, &ready, &reason)) {
          rejected_count_.fetch_add(1, std::memory_order_relaxed);
          return false;
        }
        latest_command_ready_ = true;
        if (state.dq.size() >= 3 && state.tau_est.size() >= 3 &&
            state.dq.head(3).array().isFinite().all() &&
            state.tau_est.head(3).array().isFinite().all()) {
          const auto gains = a3_pingpong::A3PdStandGains();
          std::lock_guard<std::mutex> diagnostics_lock(diagnostics_mutex_);
          diagnostics_.waist_policy_raw = {};
          for (std::size_t index = 0; index < 3; ++index) {
            const auto state_index = static_cast<Eigen::Index>(index);
            diagnostics_.waist_q_command[index] =
                latest_command_.q_des[state_index];
            diagnostics_.waist_q_feedback[index] = state.q[state_index];
            diagnostics_.waist_tau_theoretical[index] =
                gains.kp[index] *
                    (latest_command_.q_des[state_index] -
                     state.q[state_index]) -
                gains.kd[index] * state.dq[state_index];
            diagnostics_.waist_tau_feedback[index] =
                state.tau_est[state_index];
          }
        }
      }
      external_fallback_pd_ready_.store(ready, std::memory_order_relaxed);
      if (external_fallback_pd_elapsed_ticks_ < kPdStandRampTicks) {
        ++external_fallback_pd_elapsed_ticks_;
      }
      return true;
    }

    if (!external_status.base_pose) {
      rejected_count_.fetch_add(1, std::memory_order_relaxed);
      return false;
    }

    auto held_planner = planner;
    held_planner.base_pose = external_status.base_pose;
    held_planner.base_pose_age_s = 0.0;
    const auto live_base_w = held_planner.base_pose->position_w;
    if (!nominal_station_initialized_) {
      // The policy's READY target is the calibrated table-centred deployment
      // station, not the pelvis location on the first inference tick. This also
      // keeps active-strike lateral targets independent of the startup offset.
      const auto& ready_station = lifecycle_.config().ready_reference_base_w;
      nominal_station_xy_ = {ready_station[0], ready_station[1]};
      base_target_xy_ = nominal_station_xy_;
      nominal_station_initialized_ = true;
    }

    bool force_ready_observation = false;
    {
      std::lock_guard<std::mutex> lock(serve_mutex_);
      UpdateGripperResultLocked();
      if (serve_pending_ &&
          lifecycle_.phase() == a3_pingpong::SwingPhase::kReady) {
        std::string reason;
        if (upper_body_serve_.BeginHoming(state, &reason)) {
          serve_pending_ = false;
          serve_cancel_requested_ = false;
          serve_transition_active_ = false;
          serve_waist_pitch_initialized_ = false;
          for (std::size_t i = 0; i < 2; ++i) {
            serve_head_hold_[i] = state.q[static_cast<Eigen::Index>(3 + i)];
          }
          std::clog << "[serve_vcf] V accepted: homing; observation=READY, "
                       "track=" << active_serve_track_
                    << " waist/legs=model_72500\n";
        } else {
          serve_pending_ = false;
          upper_body_serve_rejected_count_.fetch_add(1);
          std::clog << "[serve_vcf] V rejected at control tick: " << reason
                    << '\n';
        }
      }
      force_ready_observation =
          upper_body_serve_.phase() !=
              a3_pingpong::UpperBodyServePhase::kIdle ||
          serve_transition_active_;
    }

    std::optional<a3_pingpong::RacketTargetInput> fresh_command;
    if (!force_ready_observation && planner.command &&
        planner.command_age_s <= command_timeout_s_) {
      fresh_command = planner.command;
    }
    auto policy_input = held_planner;
    policy_input.command = lifecycle_.Update(fresh_command, live_base_w);
    policy_input.command_age_s = 0.0;
    if (lifecycle_.phase() == a3_pingpong::SwingPhase::kReady ||
        lifecycle_.phase() == a3_pingpong::SwingPhase::kRecovery) {
      base_target_xy_ = nominal_station_xy_;
    } else if (!a3_pingpong::DeriveLateralBaseTarget(
                   policy_input.command->position_w,
                   policy_input.command->swing_side, nominal_station_xy_,
                   station_config_, base_target_xy_)) {
      rejected_count_.fetch_add(1, std::memory_order_relaxed);
      return false;
    }
    if (!phase_report_initialized_ ||
        lifecycle_.phase() != last_reported_phase_ ||
        lifecycle_.active_task_id() != last_reported_task_id_) {
      std::clog << "[model_72500 lifecycle] phase="
                << SwingPhaseName(lifecycle_.phase()) << " active_task=";
      if (lifecycle_.active_task_id()) {
        std::clog << *lifecycle_.active_task_id();
      } else {
        std::clog << "none";
      }
      std::clog << " revision=" << lifecycle_.applied_revision()
                << " side=" << static_cast<int>(policy_input.command->swing_side)
                << " tts=" << policy_input.command->time_to_strike_s
                << " nominal_station=[" << nominal_station_xy_[0] << ','
                << nominal_station_xy_[1] << "] base_target=["
                << base_target_xy_[0] << ',' << base_target_xy_[1] << "]\n";
      last_reported_phase_ = lifecycle_.phase();
      last_reported_task_id_ = lifecycle_.active_task_id();
      phase_report_initialized_ = true;
    }
    a3_pingpong::PingpongObservation observation{};
    std::string reason;
    if (!builder_.Build(state, policy_input, last_action_, base_target_xy_,
                        observation, &reason)) {
      rejected_count_.fetch_add(1, std::memory_order_relaxed);
      return false;
    }

    float max_abs = 0.0F;
    for (const float value : observation) {
      max_abs = std::max(max_abs, std::abs(value));
    }
    latest_max_abs_.store(max_abs, std::memory_order_relaxed);
    latest_tts_.store(observation[109], std::memory_order_relaxed);
    built_count_.fetch_add(1, std::memory_order_relaxed);
    {
      std::lock_guard<std::mutex> lock(diagnostics_mutex_);
      diagnostics_.phase = lifecycle_.phase();
      diagnostics_.active_task_id = lifecycle_.active_task_id();
      diagnostics_.active_revision = lifecycle_.applied_revision();
      diagnostics_.swing_side = policy_input.command->swing_side;
      diagnostics_.target_position_w = policy_input.command->position_w;
      diagnostics_.target_velocity_w = policy_input.command->velocity_w;
      diagnostics_.nominal_station_xy = nominal_station_xy_;
      diagnostics_.base_target_xy = base_target_xy_;
      diagnostics_.current_base_xy = {live_base_w[0], live_base_w[1]};
      diagnostics_.station_error_xy = {
          static_cast<double>(observation[101]),
          static_cast<double>(observation[102])};
      diagnostics_.observed_tts = observation[109];
    }

    if (actor_) {
      a3_pingpong::PingpongAction raw_action{};
      const auto started_at = std::chrono::steady_clock::now();
      if (!actor_->Run(observation, raw_action, &reason)) {
        inference_rejected_count_.fetch_add(1, std::memory_order_relaxed);
        return false;
      }
      const auto latency_ns =
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - started_at)
              .count();
      inference_total_latency_ns_.fetch_add(
          static_cast<std::uint64_t>(latency_ns), std::memory_order_relaxed);
      const auto previous_max =
          inference_max_latency_ns_.load(std::memory_order_relaxed);
      if (static_cast<std::uint64_t>(latency_ns) > previous_max) {
        inference_max_latency_ns_.store(
            static_cast<std::uint64_t>(latency_ns),
            std::memory_order_relaxed);
      }

      float raw_max_abs = 0.0F;
      for (const float value : raw_action) {
        raw_max_abs = std::max(raw_max_abs, std::abs(value));
      }
      latest_raw_max_abs_.store(raw_max_abs, std::memory_order_relaxed);
      inference_count_.fetch_add(1, std::memory_order_relaxed);

      a3_pingpong::PingpongAction applied_action{};
      a3_pingpong::PingpongActionDiagnostics diagnostics;
      std::array<double, a3_pingpong::kPingpongActionDim> q_des{};
      if (!action_adapter_.Decode(raw_action, applied_action, q_des,
                                  &diagnostics, &reason)) {
        action_rejected_count_.fetch_add(1, std::memory_order_relaxed);
        return false;
      }
      last_action_ = applied_action;
      action_decode_count_.fetch_add(1, std::memory_order_relaxed);
      raw_clip_total_.fetch_add(diagnostics.raw_clip_count,
                                std::memory_order_relaxed);
      position_clip_total_.fetch_add(diagnostics.position_clip_count,
                                     std::memory_order_relaxed);
      latest_position_clip_count_.store(diagnostics.position_clip_count,
                                        std::memory_order_relaxed);
      latest_q_des_max_abs_.store(diagnostics.q_des_max_abs,
                                  std::memory_order_relaxed);
      double max_tracking_error = 0.0;
      for (std::size_t index = 0; index < q_des.size(); ++index) {
        max_tracking_error = std::max(
            max_tracking_error,
            std::abs(q_des[index] -
                     state.q[static_cast<Eigen::Index>(index)]));
      }
      latest_max_tracking_error_.store(max_tracking_error,
                                       std::memory_order_relaxed);

      // Report the exact waist command represented by the decoded policy
      // action. RobotCommand uses dq_des=0 and tau_ff=0, so this is the
      // theoretical PD torque before any drive-side saturation or filtering.
      if (state.q.size() >= 3 && state.dq.size() >= 3 &&
          state.tau_est.size() >= 3 &&
          state.q.head(3).array().isFinite().all() &&
          state.dq.head(3).array().isFinite().all() &&
          state.tau_est.head(3).array().isFinite().all()) {
        const auto gains = a3_pingpong::Model72500PolicyGains();
        std::lock_guard<std::mutex> lock(diagnostics_mutex_);
        diagnostics_.waist_policy_valid = true;
        for (std::size_t index = 0; index < 3; ++index) {
          const auto state_index = static_cast<Eigen::Index>(index);
          diagnostics_.waist_policy_raw[index] = raw_action[index];
          diagnostics_.waist_q_command[index] = q_des[index];
          diagnostics_.waist_q_feedback[index] = state.q[state_index];
          diagnostics_.waist_tau_theoretical[index] =
              gains.kp[index] * (q_des[index] - state.q[state_index]) -
              gains.kd[index] * state.dq[state_index];
          diagnostics_.waist_tau_feedback[index] =
              state.tau_est[state_index];
        }
      }

      if (command_output_enabled_) {
        robot_io::RobotCommand command;
        if (!action_adapter_.BuildPolicyCommand(
                raw_action, applied_action, command, nullptr, &reason)) {
          action_rejected_count_.fetch_add(1, std::memory_order_relaxed);
          return false;
        }
        latest_command_ = std::move(command);
        {
          std::lock_guard<std::mutex> lock(serve_mutex_);
          if (!ApplyServeOverrideLocked(state, latest_command_, &reason)) {
            upper_body_serve_rejected_count_.fetch_add(
                1, std::memory_order_relaxed);
            return false;
          }
          // ApplyServeOverrideLocked owns the actual executable waist target
          // during V/C/F. Keep waist_diag aligned with the command sent to
          // RobotIO, while policy_raw continues to report the unmodified ONNX
          // output for comparison.
          if (state.q.size() >= 3 && state.dq.size() >= 3) {
            std::lock_guard<std::mutex> diagnostics_lock(diagnostics_mutex_);
            diagnostics_.waist_q_command[2] = latest_command_.q_des[2];
            diagnostics_.waist_tau_theoretical[2] =
                latest_command_.kp[2] *
                    (latest_command_.q_des[2] - state.q[2]) -
                latest_command_.kd[2] * state.dq[2];
          }
        }
        latest_command_ready_ = true;
      }
    }
    lifecycle_.Advance();
    return true;
  }

  a3_pingpong::ManualActionResult HandleManualKey(char key) {
    if (!manual_control_enabled_) {
      return a3_pingpong::ManualActionResult::kIgnored;
    }
    const auto parsed = a3_pingpong::ParseManualKey(key);
    const int requested_track = a3_pingpong::ManualServeTrackNumber(parsed);
    const bool is_track_key = requested_track != 0;
    std::lock_guard<std::mutex> manual_lock(manual_mutex_);
    if (manual_control_.mode() == a3_pingpong::ManualMode::kMotion) {
      std::lock_guard<std::mutex> external_lock(external_mutex_);
      if (external_guard_.status().fallback_latched) {
        if (parsed == a3_pingpong::ManualKey::kMotion) {
          if (!external_guard_.RequestResume()) {
            return a3_pingpong::ManualActionResult::
                kRejectedExternalNotReady;
          }
          external_resume_requested_.store(true, std::memory_order_release);
          return a3_pingpong::ManualActionResult::kReceiveRequested;
        }
        if (parsed == a3_pingpong::ManualKey::kUpperBodyServe ||
            is_track_key ||
            parsed == a3_pingpong::ManualKey::kServeClose ||
            parsed == a3_pingpong::ManualKey::kServeFire ||
            parsed == a3_pingpong::ManualKey::kGripperOpen) {
          return a3_pingpong::ManualActionResult::
              kRejectedExternalNotReady;
        }
      }
    }
    if (parsed == a3_pingpong::ManualKey::kUpperBodyServe ||
        is_track_key ||
        parsed == a3_pingpong::ManualKey::kServeClose ||
        parsed == a3_pingpong::ManualKey::kServeFire ||
        parsed == a3_pingpong::ManualKey::kGripperOpen ||
        parsed == a3_pingpong::ManualKey::kServeCancel ||
        (parsed == a3_pingpong::ManualKey::kMotion &&
         manual_control_.mode() == a3_pingpong::ManualMode::kMotion)) {
      if (!upper_body_serve_enabled_ &&
          parsed != a3_pingpong::ManualKey::kMotion) {
        return a3_pingpong::ManualActionResult::kRejectedServeDisabled;
      }
      if (manual_control_.mode() != a3_pingpong::ManualMode::kMotion) {
        return parsed == a3_pingpong::ManualKey::kMotion
                   ? manual_control_.Apply(parsed)
                   : a3_pingpong::ManualActionResult::kRejectedNeedMotion;
      }
      std::lock_guard<std::mutex> serve_lock(serve_mutex_);
      if (parsed == a3_pingpong::ManualKey::kUpperBodyServe || is_track_key) {
        if (serve_pending_ || serve_transition_active_) {
          return a3_pingpong::ManualActionResult::kRejectedServeState;
        }
        const auto current_phase = upper_body_serve_.phase();
        if ((!is_track_key && current_phase !=
                                  a3_pingpong::UpperBodyServePhase::kIdle) ||
            (is_track_key &&
             current_phase != a3_pingpong::UpperBodyServePhase::kIdle &&
             current_phase != a3_pingpong::UpperBodyServePhase::kReady)) {
          return a3_pingpong::ManualActionResult::kRejectedServeState;
        }
        // V is deliberately stateless: it always returns to track 1 instead
        // of inheriting the last explicit 2/3/4/5 selection.
        const int track = is_track_key ? requested_track : 1;
        const auto profile = LoadServeProfileLocked(track);
        if (!profile) {
          return a3_pingpong::ManualActionResult::kRejectedServeState;
        }
        if (current_phase == a3_pingpong::UpperBodyServePhase::kReady) {
          // A READY-to-READY numbered switch starts from the old commanded
          // Home, never measured q. This preserves the arm-supporting PD
          // error/torque and eliminates the one-tick target drop seen in the
          // standalone lower-body serve controller before its optimization.
          const auto previous_home = upper_body_serve_.config().home_upper;
          upper_body_serve_ = a3_pingpong::UpperBodyServeTrajectory(
              profile->trajectory);
          std::string reason;
          if (!upper_body_serve_.BeginHomingFromTarget(previous_home,
                                                       &reason)) {
            return a3_pingpong::ManualActionResult::kRejectedServeState;
          }
          active_serve_profile_ = *profile;
          active_serve_track_ = track;
          serve_waist_pitch_from_rad_ = serve_waist_pitch_command_rad_;
          serve_waist_pitch_initialized_ = true;
          active_serve_track_public_.store(track,
                                           std::memory_order_relaxed);
          LogServeProfile(track, active_serve_profile_);
          return a3_pingpong::ManualActionResult::kServePending;
        }
        upper_body_serve_ = a3_pingpong::UpperBodyServeTrajectory(
            profile->trajectory);
        active_serve_profile_ = *profile;
        active_serve_track_ = track;
        active_serve_track_public_.store(track, std::memory_order_relaxed);
        LogServeProfile(track, active_serve_profile_);
        serve_pending_ = true;
        return a3_pingpong::ManualActionResult::kServePending;
      }
      if (parsed == a3_pingpong::ManualKey::kServeClose) {
        if (!upper_body_serve_.ready()) {
          return a3_pingpong::ManualActionResult::kRejectedServeState;
        }
        if (!RefreshServeActuationLocked()) {
          return a3_pingpong::ManualActionResult::kRejectedServeState;
        }
        if (!RequestGripperLocked(a3_pingpong::GripperAction::kClose)) {
          return a3_pingpong::ManualActionResult::kRejectedGripperBusy;
        }
        return a3_pingpong::ManualActionResult::kGripperRequested;
      }
      if (parsed == a3_pingpong::ManualKey::kServeFire) {
        if (!upper_body_serve_.ready_to_fire()) {
          return a3_pingpong::ManualActionResult::kRejectedServeState;
        }
        if (serve_gripper_state_ != ServeGripperState::kClosed) {
          return a3_pingpong::ManualActionResult::kRejectedGripperNotClosed;
        }
        if (!RefreshServeActuationLocked()) {
          return a3_pingpong::ManualActionResult::kRejectedServeState;
        }
        std::string reason;
        if (!upper_body_serve_.Fire(&reason)) {
          return a3_pingpong::ManualActionResult::kRejectedServeState;
        }
        return a3_pingpong::ManualActionResult::kServeFireRequested;
      }
      if (parsed == a3_pingpong::ManualKey::kGripperOpen) {
        if (!RefreshServeActuationLocked()) {
          return a3_pingpong::ManualActionResult::kRejectedServeState;
        }
        if (!RequestGripperLocked(a3_pingpong::GripperAction::kOpen)) {
          return a3_pingpong::ManualActionResult::kRejectedGripperBusy;
        }
        return a3_pingpong::ManualActionResult::kGripperRequested;
      }
      // R is the dedicated serve-cancel key. M retains the same compatibility
      // behavior while already in motion. Active windup/swing/settle/return
      // is deliberately not interruptible.
      const auto phase = upper_body_serve_.phase();
      if (serve_pending_) {
        serve_pending_ = false;
        return a3_pingpong::ManualActionResult::kReceiveRequested;
      }
      if (phase == a3_pingpong::UpperBodyServePhase::kPrepare ||
          phase == a3_pingpong::UpperBodyServePhase::kReady) {
        serve_cancel_requested_ = true;
        return a3_pingpong::ManualActionResult::kReceiveRequested;
      }
      if (phase != a3_pingpong::UpperBodyServePhase::kIdle ||
          serve_transition_active_) {
        return a3_pingpong::ManualActionResult::kRejectedServeState;
      }
      return a3_pingpong::ManualActionResult::kAccepted;
    }
    return manual_control_.Apply(parsed);
  }

  a3_pingpong::ManualMode manual_mode() const {
    std::lock_guard<std::mutex> lock(manual_mutex_);
    return manual_control_.mode();
  }

  bool pd_stand_ready() const {
    std::lock_guard<std::mutex> lock(manual_mutex_);
    return manual_control_.pd_stand_ready();
  }

  a3_pingpong::ExternalObservationMode external_mode() const noexcept {
    return external_mode_.load(std::memory_order_relaxed);
  }
  double external_stale_s() const noexcept {
    return external_stale_s_.load(std::memory_order_relaxed);
  }
  std::uint64_t external_recovery_streak() const noexcept {
    return external_recovery_streak_.load(std::memory_order_relaxed);
  }
  bool external_fallback_pd_ready() const noexcept {
    return external_fallback_pd_ready_.load(std::memory_order_relaxed);
  }

  bool TakeCommand(robot_io::RobotCommand& command) {
    if (!latest_command_ready_) return false;
    command = latest_command_;
    latest_command_ready_ = false;
    dry_run_command_count_.fetch_add(1, std::memory_order_relaxed);
    return true;
  }

  std::uint64_t built_count() const noexcept {
    return built_count_.load(std::memory_order_relaxed);
  }
  std::uint64_t rejected_count() const noexcept {
    return rejected_count_.load(std::memory_order_relaxed);
  }
  float latest_max_abs() const noexcept {
    return latest_max_abs_.load(std::memory_order_relaxed);
  }
  float latest_tts() const noexcept {
    return latest_tts_.load(std::memory_order_relaxed);
  }
  bool inference_enabled() const noexcept { return actor_ != nullptr; }
  std::uint64_t inference_count() const noexcept {
    return inference_count_.load(std::memory_order_relaxed);
  }
  std::uint64_t inference_rejected_count() const noexcept {
    return inference_rejected_count_.load(std::memory_order_relaxed);
  }
  double inference_average_latency_ms() const noexcept {
    const auto count = inference_count();
    if (count == 0) return 0.0;
    return static_cast<double>(
               inference_total_latency_ns_.load(std::memory_order_relaxed)) /
           static_cast<double>(count) * 1.0e-6;
  }
  double inference_max_latency_ms() const noexcept {
    return static_cast<double>(
               inference_max_latency_ns_.load(std::memory_order_relaxed)) *
           1.0e-6;
  }
  float latest_raw_max_abs() const noexcept {
    return latest_raw_max_abs_.load(std::memory_order_relaxed);
  }
  std::uint64_t action_decode_count() const noexcept {
    return action_decode_count_.load(std::memory_order_relaxed);
  }
  std::uint64_t action_rejected_count() const noexcept {
    return action_rejected_count_.load(std::memory_order_relaxed);
  }
  std::uint64_t dry_run_command_count() const noexcept {
    return dry_run_command_count_.load(std::memory_order_relaxed);
  }
  std::uint64_t raw_clip_total() const noexcept {
    return raw_clip_total_.load(std::memory_order_relaxed);
  }
  std::uint64_t position_clip_total() const noexcept {
    return position_clip_total_.load(std::memory_order_relaxed);
  }
  std::uint32_t latest_position_clip_count() const noexcept {
    return latest_position_clip_count_.load(std::memory_order_relaxed);
  }
  double latest_q_des_max_abs() const noexcept {
    return latest_q_des_max_abs_.load(std::memory_order_relaxed);
  }
  double latest_max_tracking_error() const noexcept {
    return latest_max_tracking_error_.load(std::memory_order_relaxed);
  }
  PolicyLifecycleDiagnostics lifecycle_diagnostics() const {
    std::lock_guard<std::mutex> lock(diagnostics_mutex_);
    return diagnostics_;
  }
  bool upper_body_serve_enabled() const noexcept {
    return upper_body_serve_enabled_;
  }
  a3_pingpong::UpperBodyServePhase upper_body_serve_phase() const noexcept {
    return upper_body_serve_phase_.load(std::memory_order_relaxed);
  }
  int upper_body_serve_track() const noexcept {
    return active_serve_track_public_.load(std::memory_order_relaxed);
  }
  std::uint64_t upper_body_serve_tick() const noexcept {
    return upper_body_serve_tick_.load(std::memory_order_relaxed);
  }
  std::uint64_t upper_body_serve_command_count() const noexcept {
    return upper_body_serve_command_count_.load(std::memory_order_relaxed);
  }
  std::uint64_t upper_body_serve_rejected_count() const noexcept {
    return upper_body_serve_rejected_count_.load(std::memory_order_relaxed);
  }
  std::uint64_t upper_body_serve_release_count() const noexcept {
    return upper_body_serve_release_count_.load(std::memory_order_relaxed);
  }
  ServeGripperState serve_gripper_state() const noexcept {
    return serve_gripper_state_public_.load(std::memory_order_relaxed);
  }
  bool serve_pending() const noexcept {
    return serve_pending_public_.load(std::memory_order_relaxed);
  }

 private:
  std::string ServeProfilePath(int track) const {
    return serve_tracks_dir_ + "/" + std::to_string(track) + ".yaml";
  }

  static void LogServeProfile(
      int track, const a3_pingpong::UpperBodyServeProfile& profile) {
    std::clog << "[发球YAML] 已加载 " << track
              << " 号轨迹：open=" << profile.gripper_open_position
              << " close=" << profile.gripper_close_position
              << " right=" << profile.gripper_right_position
              << " home=" << profile.trajectory.prepare_duration_s << "s"
              << " windup=" << profile.trajectory.windup_duration_s << "s"
              << " swing=" << profile.trajectory.swing_duration_s << "s"
              << " release=" << profile.trajectory.release_time_s << "s"
              << " waist_pitch="
              << profile.waist_pitch_target_rad * 57.29577951308232
              << "deg\n";
  }

  std::optional<a3_pingpong::UpperBodyServeProfile> LoadServeProfileLocked(
      int track) const {
    std::string reason;
    auto profile = a3_pingpong::LoadUpperBodyServeProfile(
        ServeProfilePath(track), &reason);
    if (!profile) {
      std::clog << "[发球YAML] 读取失败，保持当前控制：" << reason << '\n';
    }
    return profile;
  }

  bool RefreshServeActuationLocked() {
    const auto profile = LoadServeProfileLocked(active_serve_track_);
    if (!profile) return false;
    active_serve_profile_.arm_kp = profile->arm_kp;
    active_serve_profile_.arm_kd = profile->arm_kd;
    active_serve_profile_.gripper_open_position =
        profile->gripper_open_position;
    active_serve_profile_.gripper_close_position =
        profile->gripper_close_position;
    active_serve_profile_.gripper_right_position =
        profile->gripper_right_position;
    LogServeProfile(active_serve_track_, active_serve_profile_);
    return true;
  }

  bool RequestGripperLocked(a3_pingpong::GripperAction action) {
    if (!gripper_client_) {
      serve_gripper_state_ = action == a3_pingpong::GripperAction::kOpen
                                 ? ServeGripperState::kOpen
                                 : ServeGripperState::kClosed;
      serve_gripper_state_public_.store(serve_gripper_state_);
      return true;
    }
    const int left_position = action == a3_pingpong::GripperAction::kOpen
                                  ? active_serve_profile_.gripper_open_position
                                  : active_serve_profile_.gripper_close_position;
    const auto request_id = gripper_client_->Enqueue(
        action, left_position, active_serve_profile_.gripper_right_position);
    if (!request_id) return false;
    std::clog << "[发球YAML] 夹爪命令：action="
              << a3_pingpong::GripperActionName(action)
              << " left=" << left_position
              << " right=" << active_serve_profile_.gripper_right_position
              << '\n';
    gripper_request_id_ = *request_id;
    serve_gripper_state_ = action == a3_pingpong::GripperAction::kOpen
                               ? ServeGripperState::kOpening
                               : ServeGripperState::kClosing;
    serve_gripper_state_public_.store(serve_gripper_state_);
    return true;
  }

  void UpdateGripperResultLocked() {
    if (!gripper_client_) return;
    const auto result = gripper_client_->result();
    if (result.request_id == 0 || result.request_id <= gripper_result_id_) return;
    gripper_result_id_ = result.request_id;
    if (!result.success) {
      serve_gripper_state_ = ServeGripperState::kFault;
    } else {
      serve_gripper_state_ =
          result.action == a3_pingpong::GripperAction::kOpen
              ? ServeGripperState::kOpen
              : ServeGripperState::kClosed;
    }
    serve_gripper_state_public_.store(serve_gripper_state_);
  }

  void ResetServeLocked() {
    serve_pending_ = false;
    serve_pending_public_.store(false);
    serve_cancel_requested_ = false;
    serve_transition_active_ = false;
    serve_transition_elapsed_s_ = 0.0;
    serve_waist_pitch_initialized_ = false;
    serve_waist_pitch_from_rad_ = 0.0;
    serve_waist_pitch_command_rad_ = 0.0;
    serve_transition_from_waist_pitch_rad_ = 0.0;
    upper_body_serve_.Reset();
    upper_body_serve_phase_.store(a3_pingpong::UpperBodyServePhase::kIdle);
  }

  bool ApplyServeOverrideLocked(const robot_io::RobotState& state,
                                robot_io::RobotCommand& command,
                                std::string* reason) {
    serve_pending_public_.store(serve_pending_, std::memory_order_relaxed);
    if (command.q_des.size() != 31 || command.kp.size() != 31 ||
        command.kd.size() != 31) {
      if (reason) *reason = "serve override requires a 31-DOF policy command";
      return false;
    }
    std::array<double, 16> policy_upper{};
    std::array<double, 14> policy_arm_kp{};
    std::array<double, 14> policy_arm_kd{};
    const double policy_waist_pitch = command.q_des[2];
    for (std::size_t i = 0; i < policy_upper.size(); ++i) {
      policy_upper[i] = command.q_des[static_cast<Eigen::Index>(3 + i)];
    }
    for (std::size_t i = 0; i < 14; ++i) {
      policy_arm_kp[i] = command.kp[static_cast<Eigen::Index>(5 + i)];
      policy_arm_kd[i] = command.kd[static_cast<Eigen::Index>(5 + i)];
    }
    const auto phase_before = upper_body_serve_.phase();
    if (phase_before != a3_pingpong::UpperBodyServePhase::kIdle) {
      a3_pingpong::UpperBodyServeTarget arms{};
      a3_pingpong::UpperBodyServeDiagnostics diagnostics;
      if (!upper_body_serve_.Step(state, control_dt_s_, arms, &diagnostics,
                                  reason)) return false;
      upper_body_serve_phase_.store(diagnostics.phase);
      upper_body_serve_tick_.store(diagnostics.tick);
      if (!serve_waist_pitch_initialized_) {
        // The first Home command starts from this tick's decoded policy target,
        // preserving the pre-V waist torque instead of briefly switching to
        // measured q or zeroing the PD error.
        serve_waist_pitch_from_rad_ = policy_waist_pitch;
        serve_waist_pitch_command_rad_ = policy_waist_pitch;
        serve_waist_pitch_initialized_ = true;
      }
      if (diagnostics.phase ==
          a3_pingpong::UpperBodyServePhase::kPrepare) {
        const double alpha = SmoothStep01(
            diagnostics.phase_elapsed_s /
            upper_body_serve_.config().prepare_duration_s);
        serve_waist_pitch_command_rad_ =
            serve_waist_pitch_from_rad_ +
            alpha * (active_serve_profile_.waist_pitch_target_rad -
                     serve_waist_pitch_from_rad_);
      } else {
        serve_waist_pitch_command_rad_ =
            active_serve_profile_.waist_pitch_target_rad;
      }
      command.q_des[2] = serve_waist_pitch_command_rad_;
      command.q_des[3] = serve_head_hold_[0];
      command.q_des[4] = serve_head_hold_[1];
      const bool home_gain_boost =
          diagnostics.phase == a3_pingpong::UpperBodyServePhase::kPrepare;
      for (std::size_t i = 0; i < arms.size(); ++i) {
        const auto index = static_cast<Eigen::Index>(5 + i);
        command.q_des[index] = arms[i];
        command.kp[index] = home_gain_boost
                                ? std::min(1.25 * active_serve_profile_.arm_kp[i],
                                           250.0)
                                : active_serve_profile_.arm_kp[i];
        command.kd[index] = active_serve_profile_.arm_kd[i];
      }
      upper_body_serve_command_count_.fetch_add(1);
      if (phase_before == a3_pingpong::UpperBodyServePhase::kPrepare &&
          diagnostics.phase == a3_pingpong::UpperBodyServePhase::kReady) {
        // Reference controller guarantees an open claw in READY. C is
        // rejected while this asynchronous request is still in flight.
        RequestGripperLocked(a3_pingpong::GripperAction::kOpen);
      }
      if (diagnostics.release_requested) {
        upper_body_serve_release_count_.fetch_add(1);
        if (!RequestGripperLocked(a3_pingpong::GripperAction::kOpen)) {
          std::clog << "[serve_vcf] release open request rejected: gripper busy\n";
        }
      }
      const bool cancellable =
          diagnostics.phase == a3_pingpong::UpperBodyServePhase::kPrepare ||
          diagnostics.phase == a3_pingpong::UpperBodyServePhase::kReady;
      if (diagnostics.complete || (serve_cancel_requested_ && cancellable)) {
        for (std::size_t i = 0; i < serve_transition_from_upper_.size(); ++i) {
          serve_transition_from_upper_[i] =
              command.q_des[static_cast<Eigen::Index>(3 + i)];
        }
        serve_transition_from_waist_pitch_rad_ = command.q_des[2];
        upper_body_serve_.Reset();
        serve_transition_active_ = true;
        serve_transition_elapsed_s_ = 0.0;
        serve_cancel_requested_ = false;
      }
    }

    if (serve_transition_active_) {
      const double alpha = SmoothStep01(
          serve_transition_elapsed_s_ /
          upper_body_serve_.config().receive_transition_s);
      for (std::size_t i = 0; i < serve_transition_from_upper_.size(); ++i) {
        command.q_des[static_cast<Eigen::Index>(3 + i)] =
            serve_transition_from_upper_[i] +
            alpha * (policy_upper[i] - serve_transition_from_upper_[i]);
      }
      command.q_des[2] = serve_transition_from_waist_pitch_rad_ +
                         alpha * (policy_waist_pitch -
                                  serve_transition_from_waist_pitch_rad_);
      serve_waist_pitch_command_rad_ = command.q_des[2];
      for (std::size_t i = 0; i < 14; ++i) {
        const auto index = static_cast<Eigen::Index>(5 + i);
        command.kp[index] = active_serve_profile_.arm_kp[i] +
                            alpha * (policy_arm_kp[i] -
                                     active_serve_profile_.arm_kp[i]);
        command.kd[index] = active_serve_profile_.arm_kd[i] +
                            alpha * (policy_arm_kd[i] -
                                     active_serve_profile_.arm_kd[i]);
      }
      serve_transition_elapsed_s_ += control_dt_s_;
      if (serve_transition_elapsed_s_ + 1.0e-12 >=
          upper_body_serve_.config().receive_transition_s) {
        serve_transition_active_ = false;
        serve_waist_pitch_initialized_ = false;
        upper_body_serve_phase_.store(a3_pingpong::UpperBodyServePhase::kIdle);
        std::clog << "[serve_vcf] receive transition complete; "
                     "model_72500 owns all 31 joints\n";
      }
    }
    if (reason) *reason = "valid model_72500 command with VCF upper override";
    return true;
  }

  void PreparePassiveDampingCommand(const robot_io::RobotState& state) {
    // P is a controlled free-fall mode.  Keep the actual position as q_des
    // and only add mild velocity damping to arms and legs.  X remains the
    // separate zero-gain safe-halt path below.
    if (!a3_pingpong::BuildPassiveDampingCommand(
            state, kPassiveDampingKd, latest_command_)) {
      a3_pingpong::BuildSafeHaltCommand(state, latest_command_);
    }
    latest_command_ready_ = true;
  }

  void PrepareSafeHaltCommand(const robot_io::RobotState& state) {
    a3_pingpong::BuildSafeHaltCommand(state, latest_command_);
    latest_command_ready_ = true;
  }

  static constexpr double kPassiveDampingKd = 2.0;
  static constexpr std::uint64_t kPdStandRampTicks = 150;
  static constexpr std::uint64_t kExternalRecoveryFrames = 10;

  a3_pingpong::PingpongObservationBuilder builder_;
  a3_pingpong::PingpongActionAdapter action_adapter_;
  const bool command_output_enabled_;
  const bool manual_control_enabled_;
  const bool upper_body_serve_enabled_;
  const std::string serve_tracks_dir_;
  const double control_dt_s_;
  const double command_timeout_s_;
  const double base_pose_timeout_s_;
  const a3_pingpong::SwingLifecycleConfig lifecycle_config_;
  const a3_pingpong::LateralStationConfig station_config_;
  a3_pingpong::SwingLifecycle lifecycle_;
  mutable std::mutex external_mutex_;
  a3_pingpong::ExternalObservationGuard external_guard_;
  a3_pingpong::ExternalObservationMode last_external_mode_{
      a3_pingpong::ExternalObservationMode::kWaiting};
  std::atomic<a3_pingpong::ExternalObservationMode> external_mode_{
      a3_pingpong::ExternalObservationMode::kWaiting};
  std::atomic<double> external_stale_s_{0.0};
  std::atomic<std::uint64_t> external_recovery_streak_{0};
  std::atomic<bool> external_resume_requested_{false};
  bool external_fallback_pd_initialized_{false};
  std::uint64_t external_fallback_pd_elapsed_ticks_{0};
  std::array<double, a3_pingpong::kPingpongActionDim>
      external_fallback_pd_start_q_{};
  std::atomic<bool> external_fallback_pd_ready_{false};
  bool nominal_station_initialized_{false};
  std::array<double, 2> nominal_station_xy_{};
  std::array<double, 2> base_target_xy_{};
  bool phase_report_initialized_{false};
  a3_pingpong::SwingPhase last_reported_phase_{
      a3_pingpong::SwingPhase::kReady};
  std::optional<std::uint64_t> last_reported_task_id_;
  mutable std::mutex diagnostics_mutex_;
  PolicyLifecycleDiagnostics diagnostics_;
  mutable std::mutex manual_mutex_;
  mutable std::mutex serve_mutex_;
  a3_pingpong::ManualControl manual_control_;
  std::uint64_t observed_manual_epoch_{0};
  bool pd_stand_initialized_{false};
  std::uint64_t pd_stand_elapsed_ticks_{0};
  std::array<double, a3_pingpong::kPingpongActionDim> pd_stand_start_q_{};
  a3_pingpong::UpperBodyServeTrajectory upper_body_serve_;
  a3_pingpong::UpperBodyServeProfile active_serve_profile_;
  int active_serve_track_{1};
  std::atomic<int> active_serve_track_public_{1};
  std::unique_ptr<a3_pingpong::GripperHttpClient> gripper_client_;
  bool serve_pending_{false};
  bool serve_cancel_requested_{false};
  bool serve_transition_active_{false};
  double serve_transition_elapsed_s_{0.0};
  std::array<double, 2> serve_head_hold_{};
  std::array<double, 16> serve_transition_from_upper_{};
  double serve_waist_pitch_from_rad_{0.0};
  double serve_waist_pitch_command_rad_{0.0};
  double serve_transition_from_waist_pitch_rad_{0.0};
  bool serve_waist_pitch_initialized_{false};
  ServeGripperState serve_gripper_state_{ServeGripperState::kOpen};
  std::uint64_t gripper_request_id_{0};
  std::uint64_t gripper_result_id_{0};
  std::atomic<ServeGripperState> serve_gripper_state_public_{
      ServeGripperState::kOpen};
  std::atomic<bool> serve_pending_public_{false};
  std::atomic<a3_pingpong::UpperBodyServePhase> upper_body_serve_phase_{
      a3_pingpong::UpperBodyServePhase::kIdle};
  std::atomic<std::uint64_t> upper_body_serve_tick_{0};
  std::atomic<std::uint64_t> upper_body_serve_command_count_{0};
  std::atomic<std::uint64_t> upper_body_serve_rejected_count_{0};
  std::atomic<std::uint64_t> upper_body_serve_release_count_{0};
  std::unique_ptr<a3_pingpong::OnnxActor> actor_;
  a3_pingpong::PingpongAction last_action_{};
  std::atomic<std::uint64_t> built_count_{0};
  std::atomic<std::uint64_t> rejected_count_{0};
  std::atomic<float> latest_max_abs_{0.0F};
  std::atomic<float> latest_tts_{0.0F};
  std::atomic<std::uint64_t> inference_count_{0};
  std::atomic<std::uint64_t> inference_rejected_count_{0};
  std::atomic<std::uint64_t> inference_total_latency_ns_{0};
  std::atomic<std::uint64_t> inference_max_latency_ns_{0};
  std::atomic<float> latest_raw_max_abs_{0.0F};
  robot_io::RobotCommand latest_command_;
  bool latest_command_ready_{false};
  std::atomic<std::uint64_t> action_decode_count_{0};
  std::atomic<std::uint64_t> action_rejected_count_{0};
  std::atomic<std::uint64_t> dry_run_command_count_{0};
  std::atomic<std::uint64_t> raw_clip_total_{0};
  std::atomic<std::uint64_t> position_clip_total_{0};
  std::atomic<std::uint32_t> latest_position_clip_count_{0};
  std::atomic<double> latest_q_des_max_abs_{0.0};
  std::atomic<double> latest_max_tracking_error_{0.0};
};

class PlannerReceiverNode : public rclcpp::Node {
 public:
  explicit PlannerReceiverNode(const Options& options)
      : Node("a3_mdu_planner_receiver"),
        options_(options),
        mailbox_(options.expected_frame) {
    if (options_.input_transport == "ros2") {
      const auto command_qos = rclcpp::QoS(rclcpp::KeepLast(10))
                                   .reliable()
                                   .durability_volatile();
      const auto pose_qos = rclcpp::QoS(rclcpp::KeepLast(5))
                                .best_effort()
                                .durability_volatile();

      command_subscription_ =
          create_subscription<hope_msgs::msg::RacketCommand>(
              options_.command_topic, command_qos,
              [this](const hope_msgs::msg::RacketCommand::SharedPtr message) {
                OnCommand(message);
              });
      pose_subscription_ =
          create_subscription<geometry_msgs::msg::PoseStamped>(
              options_.base_pose_topic, pose_qos,
              [this](const geometry_msgs::msg::PoseStamped::SharedPtr message) {
                OnBasePose(message);
              });
    } else {
      a3_pingpong::udp::ReceiverOptions udp_options;
      udp_options.bind_address = options_.udp_bind_address;
      udp_options.allowed_source_address = options_.udp_source_address;
      udp_options.port = options_.udp_port;
      udp_options.expected_frame = options_.expected_frame;
      udp_receiver_ = std::make_unique<a3_pingpong::udp::PlannerUdpReceiver>(
          mailbox_, std::move(udp_options));
      std::string reason;
      if (!udp_receiver_->Start(&reason)) {
        throw std::runtime_error("failed to start planner UDP receiver: " +
                                 reason);
      }
    }
    status_timer_ = create_wall_timer(
        std::chrono::duration<double>(options_.status_period_s),
        [this]() {
          LogStatus();
          LogWaistStatus();
        });

    if (options_.input_transport == "udp") {
      RCLCPP_WARN(get_logger(),
                  "MDU planner receiver started: transport=A3PP/UDP "
                  "bind=%s:%u allowed_source=%s frame=%s packet=96B; "
                  "body_drive_publishers=0",
                  options_.udp_bind_address.c_str(), options_.udp_port,
                  options_.udp_source_address.c_str(),
                  options_.expected_frame.c_str());
    } else {
      RCLCPP_WARN(get_logger(),
                  "MDU planner receiver started: transport=ROS2 command=%s "
                  "pose=%s frame=%s; body_drive_publishers=0",
                  options_.command_topic.c_str(),
                  options_.base_pose_topic.c_str(),
                  options_.expected_frame.c_str());
    }
  }

  a3_pingpong::PlannerInputMailbox& mailbox() { return mailbox_; }

  void SetController(a3_pingpong::ReceiveController* controller) {
    controller_ = controller;
  }

  void SetObservationProbe(ObservationProbe* probe) {
    observation_probe_ = probe;
  }

 private:
  void LogWaistStatus() {
    const PolicyLifecycleDiagnostics diagnostics =
        observation_probe_ ? observation_probe_->lifecycle_diagnostics()
                           : PolicyLifecycleDiagnostics{};
    const auto manual_mode =
        observation_probe_ ? observation_probe_->manual_mode()
                           : a3_pingpong::ManualMode::kPassive;
    const auto tick_result =
        controller_ ? controller_->last_result()
                    : a3_pingpong::ReceiveTickResult::kNoState;
    const auto external_mode = observation_probe_
                                   ? observation_probe_->external_mode()
                                   : a3_pingpong::ExternalObservationMode::
                                         kWaiting;
    const bool policy_active =
        diagnostics.waist_policy_valid &&
        (external_mode == a3_pingpong::ExternalObservationMode::kLive ||
         external_mode == a3_pingpong::ExternalObservationMode::kHold) &&
        (!options_.manual_control ||
         manual_mode == a3_pingpong::ManualMode::kMotion) &&
        (tick_result == a3_pingpong::ReceiveTickResult::kCommandSent ||
         tick_result == a3_pingpong::ReceiveTickResult::kDryRun);
    const bool waist_guard_active =
        controller_ && controller_->waist_pitch_guard_active();
    const auto waist_guard_triggers =
        controller_ ? controller_->waist_pitch_guard_trigger_count() : 0;
    const double waist_guard_output =
        controller_ ? controller_->waist_pitch_guard_output_rad() : 0.0;

    RCLCPP_INFO(
        get_logger(),
        "waist_diag mode=%s result=%s active=%s order=[yaw,roll,pitch] "
        "policy_raw=[%.3f,%.3f,%.3f] "
        "q_exec_rad=[%.3f,%.3f,%.3f] "
        "q_feedback_rad=[%.3f,%.3f,%.3f] "
        "tau_theoretical_nm=[%.3f,%.3f,%.3f] "
        "tau_feedback_nm=[%.3f,%.3f,%.3f] "
        "pitch_guard=[active=%s,triggers=%llu,q_out=%.3f] "
        "external=[mode=%s,stale_ms=%.1f,recovery=%llu/10,pd_ready=%s]",
        observation_probe_ ? a3_pingpong::ManualModeName(manual_mode)
                           : "disabled",
        TickResultName(tick_result), policy_active ? "yes" : "no",
        diagnostics.waist_policy_raw[0], diagnostics.waist_policy_raw[1],
        diagnostics.waist_policy_raw[2], diagnostics.waist_q_command[0],
        diagnostics.waist_q_command[1], diagnostics.waist_q_command[2],
        diagnostics.waist_q_feedback[0], diagnostics.waist_q_feedback[1],
        diagnostics.waist_q_feedback[2],
        diagnostics.waist_tau_theoretical[0],
        diagnostics.waist_tau_theoretical[1],
        diagnostics.waist_tau_theoretical[2],
        diagnostics.waist_tau_feedback[0],
        diagnostics.waist_tau_feedback[1],
        diagnostics.waist_tau_feedback[2],
        waist_guard_active ? "yes" : "no",
        static_cast<unsigned long long>(waist_guard_triggers),
        waist_guard_output,
        observation_probe_
            ? a3_pingpong::ExternalObservationModeName(
                  observation_probe_->external_mode())
            : "disabled",
        observation_probe_ ? observation_probe_->external_stale_s() * 1000.0
                           : 0.0,
        static_cast<unsigned long long>(
            observation_probe_
                ? observation_probe_->external_recovery_streak()
                : 0),
        observation_probe_ && observation_probe_->external_fallback_pd_ready()
            ? "yes"
            : "no");
  }

  void OnCommand(const hope_msgs::msg::RacketCommand::SharedPtr& message) {
    if (!message) return;
    a3_pingpong::RacketTargetInput input;
    input.frame_id = message->header.frame_id;
    input.source_stamp_ns = StampNanoseconds(message->header.stamp);
    input.task_id = message->task_id;
    input.task_revision = message->task_revision;
    input.swing_side = message->swing_side;
    input.position_w = {message->position.x, message->position.y,
                        message->position.z};
    input.velocity_w = {message->velocity.x, message->velocity.y,
                        message->velocity.z};
    input.time_to_strike_s = message->time_to_strike;

    std::string reason;
    const auto result = mailbox_.UpdateCommand(
        std::move(input), now().nanoseconds(), a3_pingpong::SteadyClock::now(),
        &reason);
    if (result == a3_pingpong::InputUpdateResult::kAccepted) {
      accepted_commands_.fetch_add(1, std::memory_order_relaxed);
    } else if (result != a3_pingpong::InputUpdateResult::kDuplicate) {
      rejected_commands_.fetch_add(1, std::memory_order_relaxed);
      RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(),
          static_cast<std::int64_t>(options_.status_period_s * 1000.0),
          "RacketCommand %s: %s", ResultName(result), reason.c_str());
    }
  }

  void OnBasePose(const geometry_msgs::msg::PoseStamped::SharedPtr& message) {
    if (!message) return;
    a3_pingpong::BasePoseInput input;
    input.frame_id = message->header.frame_id;
    input.source_stamp_ns = StampNanoseconds(message->header.stamp);
    input.position_w = {message->pose.position.x, message->pose.position.y,
                        message->pose.position.z};
    input.quaternion_wxyz = {message->pose.orientation.w,
                             message->pose.orientation.x,
                             message->pose.orientation.y,
                             message->pose.orientation.z};

    std::string reason;
    const auto result = mailbox_.UpdateBasePose(
        std::move(input), a3_pingpong::SteadyClock::now(), &reason);
    if (result == a3_pingpong::InputUpdateResult::kAccepted) {
      accepted_poses_.fetch_add(1, std::memory_order_relaxed);
    } else {
      rejected_poses_.fetch_add(1, std::memory_order_relaxed);
      RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(),
          static_cast<std::int64_t>(options_.status_period_s * 1000.0),
          "BasePose %s: %s", ResultName(result), reason.c_str());
    }
  }

  void LogStatus() {
    const auto snapshot = mailbox_.Snapshot(a3_pingpong::SteadyClock::now());
    const bool pose_ready =
        snapshot.base_pose &&
        snapshot.base_pose_age_s <= options_.base_pose_timeout_ms * 1.0e-3;
    const bool command_fresh =
        snapshot.Ready(options_.command_timeout_ms * 1.0e-3,
                       options_.base_pose_timeout_ms * 1.0e-3);
    const auto task_id = snapshot.command ? snapshot.command->task_id : 0;
    const auto revision =
        snapshot.command ? snapshot.command->task_revision : 0;
    const double command_age_ms =
        snapshot.command ? snapshot.command_age_s * 1000.0 : -1.0;
    const double pose_age_ms =
        snapshot.base_pose ? snapshot.base_pose_age_s * 1000.0 : -1.0;
    const double tts =
        snapshot.command ? snapshot.command->time_to_strike_s : -1.0;
    const auto controller_ticks = controller_ ? controller_->tick_count() : 0;
    const auto safe_halt_count =
        controller_ ? controller_->safe_halt_count() : 0;
    const auto state_stale_count =
        controller_ ? controller_->result_count(
                          a3_pingpong::ReceiveTickResult::kStateStale)
                    : 0;
    const auto safe_halt_delta = safe_halt_count - last_logged_safe_halts_;
    const auto state_stale_delta =
        state_stale_count - last_logged_state_stale_;
    const double robot_state_age_ms =
        controller_ && controller_->last_state_age_ns() >= 0
            ? static_cast<double>(controller_->last_state_age_ns()) * 1.0e-6
            : -1.0;
    const auto result_count = [this](a3_pingpong::ReceiveTickResult result) {
      return controller_ ? controller_->result_count(result) : 0;
    };
    const auto command_send_failures =
        controller_ ? controller_->send_failure_count() : 0;
    const auto halt_send_failures =
        controller_ ? controller_->safe_halt_send_failure_count() : 0;
    const char* controller_result =
        controller_ ? TickResultName(controller_->last_result()) : "disabled";
    const bool leg_damping_active =
        controller_ && controller_->leg_limit_damping_active();
    const int leg_damping_joint =
        controller_ ? controller_->leg_limit_joint_index() : -1;
    const auto leg_damping_count =
        controller_ ? controller_->leg_limit_damping_count() : 0;
    const bool waist_guard_active =
        controller_ && controller_->waist_pitch_guard_active();
    const auto waist_guard_triggers =
        controller_ ? controller_->waist_pitch_guard_trigger_count() : 0;
    const double waist_guard_output =
        controller_ ? controller_->waist_pitch_guard_output_rad() : 0.0;
    const auto observation_built =
        observation_probe_ ? observation_probe_->built_count() : 0;
    const auto observation_rejected =
        observation_probe_ ? observation_probe_->rejected_count() : 0;
    const double observation_max_abs =
        observation_probe_ ? observation_probe_->latest_max_abs() : 0.0;
    const double observation_tts =
        observation_probe_ ? observation_probe_->latest_tts() : 0.0;
    const bool inference_enabled =
        observation_probe_ && observation_probe_->inference_enabled();
    const auto inference_count =
        observation_probe_ ? observation_probe_->inference_count() : 0;
    const auto inference_rejected = observation_probe_
                                        ? observation_probe_
                                              ->inference_rejected_count()
                                        : 0;
    const double inference_average_ms = observation_probe_
                                            ? observation_probe_
                                                  ->inference_average_latency_ms()
                                            : 0.0;
    const double inference_max_ms = observation_probe_
                                        ? observation_probe_
                                              ->inference_max_latency_ms()
                                        : 0.0;
    const double raw_max_abs = observation_probe_
                                   ? observation_probe_->latest_raw_max_abs()
                                   : 0.0;
    const PolicyLifecycleDiagnostics lifecycle_diagnostics =
        observation_probe_ ? observation_probe_->lifecycle_diagnostics()
                           : PolicyLifecycleDiagnostics{};
    const auto current_manual_mode =
        observation_probe_ ? observation_probe_->manual_mode()
                           : a3_pingpong::ManualMode::kPassive;
    const auto current_tick_result =
        controller_ ? controller_->last_result()
                    : a3_pingpong::ReceiveTickResult::kNoState;
    const auto current_external_mode =
        observation_probe_ ? observation_probe_->external_mode()
                           : a3_pingpong::ExternalObservationMode::kWaiting;
    const bool waist_policy_active =
        lifecycle_diagnostics.waist_policy_valid &&
        (current_external_mode ==
             a3_pingpong::ExternalObservationMode::kLive ||
         current_external_mode ==
             a3_pingpong::ExternalObservationMode::kHold) &&
        (!options_.manual_control ||
         current_manual_mode == a3_pingpong::ManualMode::kMotion) &&
        (current_tick_result ==
             a3_pingpong::ReceiveTickResult::kCommandSent ||
         current_tick_result == a3_pingpong::ReceiveTickResult::kDryRun);
    const std::uint64_t active_task_id =
        lifecycle_diagnostics.active_task_id.value_or(0);
    const auto serve_phase = observation_probe_
                                 ? observation_probe_->upper_body_serve_phase()
                                 : a3_pingpong::UpperBodyServePhase::kIdle;
    const auto serve_tick = observation_probe_
                                ? observation_probe_->upper_body_serve_tick()
                                : 0;
    const auto serve_commands =
        observation_probe_
            ? observation_probe_->upper_body_serve_command_count()
            : 0;
    const auto serve_rejected =
        observation_probe_
            ? observation_probe_->upper_body_serve_rejected_count()
            : 0;
    const auto serve_releases =
        observation_probe_
            ? observation_probe_->upper_body_serve_release_count()
            : 0;
    std::uint64_t accepted_commands = accepted_commands_.load();
    std::uint64_t rejected_commands = rejected_commands_.load();
    std::uint64_t accepted_poses = accepted_poses_.load();
    std::uint64_t rejected_poses = rejected_poses_.load();
    a3_pingpong::udp::ReceiverStats udp_stats;
    if (udp_receiver_) {
      udp_stats = udp_receiver_->stats();
      accepted_commands = udp_stats.accepted_commands;
      rejected_commands = udp_stats.rejected_commands;
      accepted_poses = udp_stats.accepted_poses;
      rejected_poses = udp_stats.rejected_poses;
    }

    RCLCPP_INFO(get_logger(),
                "planner_input transport=%s ready=%s command_fresh=%s task=%llu "
                "revision=%u tts=%.3fs "
                "command_age=%.1fms pose_age=%.1fms accepted=(%llu,%llu) "
                "rejected=(%llu,%llu) robot_io=(ticks=%llu,result=%s,"
                "safe_halts=%llu,delta=%llu,state_age_ms=%.1f,"
                "sync_complete=%s,sync_aligned=%s,"
                "send_failures=%llu,halt_send_failures=%llu,"
                "results=[no_state:%llu,planner:%llu,state_stale:%llu/+%llu,"
                "observation:%llu,policy_missing:%llu,policy_rejected:%llu,"
                "invalid:%llu,leg_damping:%llu,sent:%llu,dry:%llu]) "
                "leg_damping=(active=%s,joint=%s,commands=%llu) "
                "pitch_guard=(active=%s,triggers=%llu,q_out=%.3f) "
                "manual=(enabled=%s,mode=%s,pd_stand_ready=%s) "
                "external=(mode=%s,stale_ms=%.1f,recovery=%llu/10,"
                "pd_ready=%s) "
                "serve=(enabled=%s,pending=%s,track=%d,phase=%s,gripper=%s,tick=%llu,"
                "commands=%llu,rejected=%llu,releases=%llu,"
                "lower=model_72500_ready,gains=Serve_A3_leg_model) "
                "observation=(enabled=%s,built=%llu,rejected=%llu,"
                "max_abs=%.3f,tts=%.3f) "
                "lifecycle=(phase=%s,active=%s,task=%llu,revision=%u,side=%d) "
                "target=(p=[%.3f,%.3f,%.3f],v=[%.3f,%.3f,%.3f]) "
                "station=(nominal=[%.3f,%.3f],target=[%.3f,%.3f],"
                "base=[%.3f,%.3f],obs101_102=[%.3f,%.3f]) "
                "inference=(enabled=%s,runs=%llu,rejected=%llu,"
                "avg_ms=%.3f,max_ms=%.3f,raw_max_abs=%.3f) "
                "waist=(order=[yaw,roll,pitch],active=%s,"
                "policy_raw=[%.3f,%.3f,%.3f],"
                "q_cmd_rad=[%.3f,%.3f,%.3f],"
                "q_fb_rad=[%.3f,%.3f,%.3f],"
                "tau_pd_nm=[%.3f,%.3f,%.3f],"
                "tau_fb_nm=[%.3f,%.3f,%.3f]) "
                "udp=(accepted=%llu,rejected=%llu,reordered=%llu,"
                "age_ms=%.1f,max_gap_ms=%.1f) "
                "body_drive_publishers=%s",
                options_.input_transport.c_str(),
                pose_ready ? "yes" : "no",
                command_fresh ? "yes" : "no",
                static_cast<unsigned long long>(task_id), revision, tts,
                command_age_ms, pose_age_ms,
                static_cast<unsigned long long>(accepted_commands),
                static_cast<unsigned long long>(accepted_poses),
                static_cast<unsigned long long>(rejected_commands),
                static_cast<unsigned long long>(rejected_poses),
                static_cast<unsigned long long>(controller_ticks),
                controller_result,
                static_cast<unsigned long long>(safe_halt_count),
                static_cast<unsigned long long>(safe_halt_delta),
                robot_state_age_ms,
                controller_ && controller_->last_sync_complete() ? "yes"
                                                                  : "no",
                controller_ && controller_->last_sync_aligned() ? "yes"
                                                                 : "no",
                static_cast<unsigned long long>(command_send_failures),
                static_cast<unsigned long long>(halt_send_failures),
                static_cast<unsigned long long>(result_count(
                    a3_pingpong::ReceiveTickResult::kNoState)),
                static_cast<unsigned long long>(result_count(
                    a3_pingpong::ReceiveTickResult::kPlannerInputNotReady)),
                static_cast<unsigned long long>(state_stale_count),
                static_cast<unsigned long long>(state_stale_delta),
                static_cast<unsigned long long>(result_count(
                    a3_pingpong::ReceiveTickResult::kObservationRejected)),
                static_cast<unsigned long long>(result_count(
                    a3_pingpong::ReceiveTickResult::kPolicyUnavailable)),
                static_cast<unsigned long long>(result_count(
                    a3_pingpong::ReceiveTickResult::kPolicyRejected)),
                static_cast<unsigned long long>(result_count(
                    a3_pingpong::ReceiveTickResult::kCommandInvalid)),
                static_cast<unsigned long long>(result_count(
                    a3_pingpong::ReceiveTickResult::kLegLimitDamping)),
                static_cast<unsigned long long>(result_count(
                    a3_pingpong::ReceiveTickResult::kCommandSent)),
                static_cast<unsigned long long>(result_count(
                    a3_pingpong::ReceiveTickResult::kDryRun)),
                leg_damping_active ? "yes" : "no",
                LegJointName(leg_damping_joint),
                static_cast<unsigned long long>(leg_damping_count),
                waist_guard_active ? "yes" : "no",
                static_cast<unsigned long long>(waist_guard_triggers),
                waist_guard_output,
                options_.manual_control ? "yes" : "no",
                observation_probe_
                    ? a3_pingpong::ManualModeName(
                          observation_probe_->manual_mode())
                    : "disabled",
                observation_probe_ && observation_probe_->pd_stand_ready()
                    ? "yes" : "no",
                observation_probe_
                    ? a3_pingpong::ExternalObservationModeName(
                          observation_probe_->external_mode())
                    : "disabled",
                observation_probe_
                    ? observation_probe_->external_stale_s() * 1000.0
                    : 0.0,
                static_cast<unsigned long long>(
                    observation_probe_
                        ? observation_probe_->external_recovery_streak()
                        : 0),
                observation_probe_ &&
                        observation_probe_->external_fallback_pd_ready()
                    ? "yes"
                    : "no",
                options_.serve_vcf ? "yes" : "no",
                observation_probe_ && observation_probe_->serve_pending()
                    ? "yes" : "no",
                observation_probe_
                    ? observation_probe_->upper_body_serve_track()
                    : 1,
                a3_pingpong::UpperBodyServePhaseName(serve_phase),
                observation_probe_
                    ? ServeGripperStateName(
                          observation_probe_->serve_gripper_state())
                    : "disabled",
                static_cast<unsigned long long>(serve_tick),
                static_cast<unsigned long long>(serve_commands),
                static_cast<unsigned long long>(serve_rejected),
                static_cast<unsigned long long>(serve_releases),
                observation_probe_ ? "yes" : "no",
                static_cast<unsigned long long>(observation_built),
                static_cast<unsigned long long>(observation_rejected),
                observation_max_abs, observation_tts,
                SwingPhaseName(lifecycle_diagnostics.phase),
                lifecycle_diagnostics.active_task_id ? "yes" : "no",
                static_cast<unsigned long long>(active_task_id),
                lifecycle_diagnostics.active_revision,
                static_cast<int>(lifecycle_diagnostics.swing_side),
                lifecycle_diagnostics.target_position_w[0],
                lifecycle_diagnostics.target_position_w[1],
                lifecycle_diagnostics.target_position_w[2],
                lifecycle_diagnostics.target_velocity_w[0],
                lifecycle_diagnostics.target_velocity_w[1],
                lifecycle_diagnostics.target_velocity_w[2],
                lifecycle_diagnostics.nominal_station_xy[0],
                lifecycle_diagnostics.nominal_station_xy[1],
                lifecycle_diagnostics.base_target_xy[0],
                lifecycle_diagnostics.base_target_xy[1],
                lifecycle_diagnostics.current_base_xy[0],
                lifecycle_diagnostics.current_base_xy[1],
                lifecycle_diagnostics.station_error_xy[0],
                lifecycle_diagnostics.station_error_xy[1],
                inference_enabled ? "yes" : "no",
                static_cast<unsigned long long>(inference_count),
                static_cast<unsigned long long>(inference_rejected),
                inference_average_ms, inference_max_ms, raw_max_abs,
                waist_policy_active ? "yes" : "no",
                lifecycle_diagnostics.waist_policy_raw[0],
                lifecycle_diagnostics.waist_policy_raw[1],
                lifecycle_diagnostics.waist_policy_raw[2],
                lifecycle_diagnostics.waist_q_command[0],
                lifecycle_diagnostics.waist_q_command[1],
                lifecycle_diagnostics.waist_q_command[2],
                lifecycle_diagnostics.waist_q_feedback[0],
                lifecycle_diagnostics.waist_q_feedback[1],
                lifecycle_diagnostics.waist_q_feedback[2],
                lifecycle_diagnostics.waist_tau_theoretical[0],
                lifecycle_diagnostics.waist_tau_theoretical[1],
                lifecycle_diagnostics.waist_tau_theoretical[2],
                lifecycle_diagnostics.waist_tau_feedback[0],
                lifecycle_diagnostics.waist_tau_feedback[1],
                lifecycle_diagnostics.waist_tau_feedback[2],
                static_cast<unsigned long long>(udp_stats.accepted_packets),
                static_cast<unsigned long long>(udp_stats.rejected_packets),
                static_cast<unsigned long long>(
                    udp_stats.out_of_order_packets),
                static_cast<double>(udp_stats.current_packet_age_ns) * 1.0e-6,
                static_cast<double>(udp_stats.max_packet_gap_ns) * 1.0e-6,
                options_.publish_commands ? "enabled" : "disabled");
    last_logged_safe_halts_ = safe_halt_count;
    last_logged_state_stale_ = state_stale_count;
  }

  Options options_;
  a3_pingpong::PlannerInputMailbox mailbox_;
  std::unique_ptr<a3_pingpong::udp::PlannerUdpReceiver> udp_receiver_;
  a3_pingpong::ReceiveController* controller_{nullptr};
  ObservationProbe* observation_probe_{nullptr};
  std::atomic<std::uint64_t> accepted_commands_{0};
  std::atomic<std::uint64_t> rejected_commands_{0};
  std::atomic<std::uint64_t> accepted_poses_{0};
  std::atomic<std::uint64_t> rejected_poses_{0};
  std::uint64_t last_logged_safe_halts_{0};
  std::uint64_t last_logged_state_stale_{0};
  rclcpp::Subscription<hope_msgs::msg::RacketCommand>::SharedPtr
      command_subscription_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr
      pose_subscription_;
  rclcpp::TimerBase::SharedPtr status_timer_;
};

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = ParseOptions(argc, argv);
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PlannerReceiverNode>(options);
    std::unique_ptr<robot_io::RobotIOBackend> backend;
    std::unique_ptr<ObservationProbe> observation_probe;
    std::unique_ptr<a3_pingpong::ReceiveController> controller;

    if (!options.aimrt_cfg.empty()) {
      backend = robot_io::CreateBackend("a3");
      if (!backend) {
        throw std::runtime_error("A3 RobotIOBackend is unavailable");
      }
      const std::string backend_options =
          "cfg_file_path=" + options.aimrt_cfg +
          ",publish_enabled=" +
          std::string(options.publish_commands ? "true" : "false") +
          ",sync_mode=min_skew_pair,sync_hz=100";
      if (!backend->Init(backend_options)) {
        throw std::runtime_error("failed to initialize RobotIOBackend");
      }
      if (options.observation_probe) {
        std::string reason;
        if (!a3_pingpong::ValidatePingpongJointLayout(backend->GetLayout(),
                                                       &reason)) {
          throw std::runtime_error("observation joint contract invalid: " +
                                   reason);
        }
        observation_probe = std::make_unique<ObservationProbe>(
            options.onnx_model,
            options.action_dry_run || options.serve_vcf ||
                options.publish_commands,
            options.manual_control,
            options.serve_vcf,
            options.serve_tracks_dir,
            options.gripper_http,
            options.gripper_config,
            options.control_hz,
            options.command_timeout_ms * 1.0e-3,
            options.base_pose_timeout_ms * 1.0e-3,
            options.external_fallback_ms * 1.0e-3);
        node->SetObservationProbe(observation_probe.get());
      }

      a3_pingpong::ReceiveControllerOptions controller_options;
      controller_options.control_hz = options.control_hz;
      controller_options.command_timeout_s =
          options.command_timeout_ms * 1.0e-3;
      controller_options.base_pose_timeout_s =
          options.base_pose_timeout_ms * 1.0e-3;
      controller_options.max_state_age_ns = static_cast<std::int64_t>(
          options.state_timeout_ms * 1.0e6);
      controller_options.require_fresh_command = false;
      // ObservationProbe owns short hold and long-outage fallback. Keeping a
      // second freshness gate here would bypass that state machine and emit a
      // zero-gain safe halt before the probe can react.
      controller_options.require_fresh_base_pose = !options.observation_probe;
      controller_options.publish_commands = options.publish_commands;
      controller_options.leg_damping_safety.damping_kd =
          options.leg_damping_kd;
      // ParseOptions has already applied the requested margin to the default
      // A3 mechanical ranges.
      controller_options.leg_damping_safety.lower =
          options.leg_damping_safety.lower;
      controller_options.leg_damping_safety.upper =
          options.leg_damping_safety.upper;
      controller_options.waist_pitch_safety = options.waist_pitch_safety;
      a3_pingpong::ReceivePolicyFn policy;
      if (options.action_dry_run || options.serve_vcf ||
          options.publish_commands) {
        policy = [&observation_probe](const auto&, const auto&,
                                      robot_io::RobotCommand& command) {
          return observation_probe->TakeCommand(command);
        };
      }
      controller = std::make_unique<a3_pingpong::ReceiveController>(
          *backend, node->mailbox(), std::move(policy), controller_options);
      if (observation_probe && !controller->SetObservationProbe(
                                   [&observation_probe](
                                       const auto& planner,
                                       const auto& state) {
                                     return observation_probe->Evaluate(
                                         planner, state);
                                   })) {
        throw std::runtime_error("failed to install observation probe");
      }
      node->SetController(controller.get());
      if (!controller->Start()) {
        throw std::runtime_error("failed to start RobotIO controller");
      }
      RCLCPP_WARN(node->get_logger(),
                  "RobotIO controller enabled; observation=%s "
                  "inference=%s command_output=%s manual=%s upper_serve=%s "
                  "gains=model_72500/pd_stand_production "
                  "external_observation=(hold_after=%.0fms,"
                  "fallback_enabled=%s,default_pd_after=%.0fms,"
                  "recovery_frames=10) "
                  "robot_state_watchdog=(enabled=%s,timeout=%.0fms) "
                  "waist_pitch_guard=(enabled=%s,enter=%.3f,release=%.3f,"
                  "target=%.3f,kp=%.1f,kd=%.1f) "
                  "publish_enabled=%s",
                  observation_probe ? "111d" : "disabled",
                  options.onnx_model.empty() ? "disabled" : "model_72500",
                  options.publish_commands
                      ? "send"
                      : ((options.action_dry_run || options.serve_vcf)
                             ? "full_dry_run"
                             : "disabled"),
                  options.manual_control ? "P/S/M/V/C/F/G/R/X"
                                         : "automatic_probe",
                  options.serve_vcf
                      ? "Serve_A3_leg_model+model_72500_ready"
                      : "disabled",
                  options.base_pose_timeout_ms,
                  options.external_fallback_ms > 0.0 ? "yes" : "no",
                  options.external_fallback_ms,
                  options.state_timeout_ms > 0.0 ? "yes" : "no",
                  options.state_timeout_ms,
                  options.waist_pitch_safety.enabled ? "yes" : "no",
                  options.waist_pitch_safety.enter_rad,
                  options.waist_pitch_safety.release_rad,
                  options.waist_pitch_safety.recovery_target_rad,
                  options.waist_pitch_safety.recovery_kp,
                  options.waist_pitch_safety.recovery_kd,
                  options.publish_commands ? "true" : "false");
      if (options.serve_vcf) {
        RCLCPP_WARN(node->get_logger(),
                    "发球YAML热加载已启用：目录=%s；V/1-5重载完整轨迹，"
                    "C/F/G会重新读取当前轨迹的夹爪位置和双臂增益",
                    options.serve_tracks_dir.c_str());
      }
    }

    std::thread keyboard_thread;
    if (options.manual_control && observation_probe) {
      keyboard_thread = std::thread([&node, &observation_probe, &options]() {
        termios original{};
        const bool is_terminal = ::isatty(STDIN_FILENO) == 1 &&
                                 ::tcgetattr(STDIN_FILENO, &original) == 0;
        if (is_terminal) {
          termios raw = original;
          raw.c_lflag &= static_cast<tcflag_t>(~(ICANON | ECHO));
          raw.c_cc[VMIN] = 0;
          raw.c_cc[VTIME] = 0;
          ::tcsetattr(STDIN_FILENO, TCSANOW, &raw);
        }
        RCLCPP_WARN(node->get_logger(),
                    "manual keys: P=passive S=pd_stand M=receive "
                    "V=serve_track1 1/2/3/4/5=select_serve_track "
                    "C=close_gripper F=fire G=open_gripper "
                    "R=cancel_serve I=status X=halt H=help Q=quit(passive only)");
        while (rclcpp::ok()) {
          pollfd descriptor{STDIN_FILENO, POLLIN, 0};
          const int poll_result = ::poll(&descriptor, 1, 100);
          if (poll_result <= 0 || !(descriptor.revents & POLLIN)) continue;
          char key = 0;
          if (::read(STDIN_FILENO, &key, 1) != 1) continue;
          const auto result = observation_probe->HandleManualKey(key);
          const auto mode = observation_probe->manual_mode();
          if (result == a3_pingpong::ManualActionResult::kRejectedNeedPdStand) {
            RCLCPP_ERROR(node->get_logger(),
                         "M/V rejected: press S and wait for pd_stand_ready=yes");
          } else if (result ==
                     a3_pingpong::ManualActionResult::kRejectedNeedMotion) {
            RCLCPP_ERROR(node->get_logger(),
                         "发球按键被拒绝：请先按 S 等待站稳，再按 M 进入接球策略");
          } else if (result ==
                     a3_pingpong::ManualActionResult::kRejectedServeState) {
            RCLCPP_ERROR(node->get_logger(),
                         "按键被拒绝：当前发球阶段不允许该操作（I 查看状态）");
          } else if (result ==
                     a3_pingpong::ManualActionResult::kRejectedGripperBusy) {
            RCLCPP_ERROR(node->get_logger(),
                         "夹爪命令被拒绝：上一条 HTTP 请求尚未完成");
          } else if (result == a3_pingpong::ManualActionResult::
                                   kRejectedGripperNotClosed) {
            RCLCPP_ERROR(node->get_logger(),
                         "F 被拒绝：必须先按 C，并等待夹爪状态变为 closed");
          } else if (result == a3_pingpong::ManualActionResult::
                                   kRejectedExternalNotReady) {
            RCLCPP_ERROR(
                node->get_logger(),
                "按键被拒绝：外部位姿尚未连续恢复 10 帧；I 查看 external 状态");
          } else if (result ==
                     a3_pingpong::ManualActionResult::kRejectedServeDisabled) {
            RCLCPP_ERROR(node->get_logger(),
                         "V/C/F/G/R rejected: start with --serve-vcf");
          } else if (result ==
                     a3_pingpong::ManualActionResult::kRejectedQuitWhileActive) {
            RCLCPP_ERROR(node->get_logger(),
                         "Q rejected: return to passive with P first");
          } else if (result ==
                     a3_pingpong::ManualActionResult::kHelpRequested) {
            RCLCPP_INFO(node->get_logger(),
                        "P=passive S=pd_stand M=receive V=serve_track1 "
                        "1/2/3/4/5=select_serve_track "
                        "C=close_gripper F=fire G=open_gripper "
                        "R=cancel_serve I=status X=halt H=help "
                        "Q=quit(passive only)");
          } else if (result ==
                     a3_pingpong::ManualActionResult::kStatusRequested) {
            RCLCPP_INFO(node->get_logger(),
                        "manual mode=%s pd_stand_ready=%s serve_track=%d "
                        "serve_phase=%s "
                        "gripper=%s pending=%s external=%s stale_ms=%.1f "
                        "recovery=%llu/10 fallback_pd_ready=%s",
                        a3_pingpong::ManualModeName(mode),
                        observation_probe->pd_stand_ready() ? "yes" : "no",
                        observation_probe->upper_body_serve_track(),
                        a3_pingpong::UpperBodyServePhaseName(
                            observation_probe->upper_body_serve_phase()),
                        ServeGripperStateName(
                            observation_probe->serve_gripper_state()),
                        observation_probe->serve_pending() ? "yes" : "no",
                        a3_pingpong::ExternalObservationModeName(
                            observation_probe->external_mode()),
                        observation_probe->external_stale_s() * 1000.0,
                        static_cast<unsigned long long>(
                            observation_probe->external_recovery_streak()),
                        observation_probe->external_fallback_pd_ready()
                            ? "yes"
                            : "no");
          } else if (result ==
                     a3_pingpong::ManualActionResult::kServePending) {
            RCLCPP_WARN(node->get_logger(),
                        "发球轨迹 %d 已从 YAML 重载并开始连续切换至 Home；"
                        "V 始终选择并重载 1 号",
                        observation_probe->upper_body_serve_track());
          } else if (result ==
                     a3_pingpong::ManualActionResult::kGripperRequested) {
            RCLCPP_WARN(node->get_logger(),
                        "夹爪请求已提交；请按 I 确认 open/closed 后继续");
          } else if (result == a3_pingpong::ManualActionResult::
                                   kServeFireRequested) {
            RCLCPP_WARN(node->get_logger(),
                        "F 已接受：执行引拍、挥拍、释放，并自动平滑切回接球");
          } else if (result ==
                     a3_pingpong::ManualActionResult::kReceiveRequested) {
            RCLCPP_WARN(node->get_logger(),
                        "%c 已接受：取消待发球/归位，平滑回到接球策略",
                        static_cast<char>(std::toupper(
                            static_cast<unsigned char>(key))));
          } else if (result ==
                     a3_pingpong::ManualActionResult::kQuitRequested) {
            RCLCPP_WARN(node->get_logger(), "manual quit requested");
            rclcpp::shutdown();
          } else if (result == a3_pingpong::ManualActionResult::kAccepted) {
            RCLCPP_WARN(node->get_logger(), "manual mode -> %s (%s)",
                        a3_pingpong::ManualModeName(mode),
                        options.publish_commands ? "command publishing"
                                                 : "shadow only");
            if (mode == a3_pingpong::ManualMode::kHalted) rclcpp::shutdown();
          }
        }
        if (is_terminal) ::tcsetattr(STDIN_FILENO, TCSANOW, &original);
      });
    }

    rclcpp::spin(node);
    if (controller) controller->Stop();
    if (keyboard_thread.joinable()) keyboard_thread.join();
    rclcpp::shutdown();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "a3_mdu_planner_receiver: " << error.what() << '\n';
    if (rclcpp::ok()) rclcpp::shutdown();
    return 64;
  }
}
