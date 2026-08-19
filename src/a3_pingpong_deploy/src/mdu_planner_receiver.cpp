#include "a3_pingpong/a3_leg_limits.hpp"
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
  double state_timeout_ms{50.0};
  double control_hz{50.0};
  double status_period_s{5.0};
  double leg_soft_scale{0.90};
  double leg_damping_kd{2.0};
  std::vector<std::string> leg_soft_limit_specs;
  bool observation_probe{false};
  bool onnx_requested{false};
  bool action_dry_run{false};
  bool upper_body_serve_dry_run{false};
  bool manual_control{false};
  bool publish_commands{false};
  std::string onnx_model;
  a3_pingpong::ReceiveControllerOptions::LegDampingSafety leg_damping_safety;
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
      << "  --onnx-model PATH           run model_53000 inference\n"
      << "  --action-dry-run            build full RobotCommand; do not send\n"
      << "  --upper-body-serve-dry-run  V-key upper-body serve; hold lower body\n"
      << "  --manual-control            P/S/M/V/X keyboard control state machine\n"
      << "  --publish-commands          enable official RobotIO SendCommand path\n"
      << "  --command-timeout-ms MS     (default: 150)\n"
      << "  --base-pose-timeout-ms MS   (default: 100)\n"
      << "  --state-timeout-ms MS       (default: 50)\n"
      << "  --control-hz HZ             (default: 50)\n"
      << "  --leg-soft-scale SCALE     URDF endpoint scale (default: 0.90)\n"
      << "  --leg-soft-limit SPEC      NAME:LOWER:UPPER; repeat per joint\n"
      << "  --leg-damping-kd KD        leg Kd in latched damping (default: 2.0)\n"
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
      options.observation_probe = true;
      options.manual_control = true;
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
        take_value("--aimrt-cfg", options.aimrt_cfg)) {
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
  if (options.upper_body_serve_dry_run && options.publish_commands) {
    throw std::runtime_error(
        "upper-body serve phase 1 is dry-run only; command publishing is refused");
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
      !std::isfinite(options.state_timeout_ms) ||
      options.state_timeout_ms <= 0.0 ||
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

class ObservationProbe {
 public:
  ObservationProbe(const std::string& onnx_model, bool command_output_enabled,
                   bool manual_control, bool upper_body_serve_enabled,
                   double control_hz, double command_timeout_s,
                   double base_pose_timeout_s)
      : builder_(a3_pingpong::Model50000ObservationConfig()),
        action_adapter_(a3_pingpong::Model50000ActionAdapterConfig()),
        command_output_enabled_(command_output_enabled),
        manual_control_enabled_(manual_control),
        upper_body_serve_enabled_(upper_body_serve_enabled),
        control_dt_s_(1.0 / control_hz),
        command_timeout_s_(command_timeout_s),
        base_pose_timeout_s_(base_pose_timeout_s),
        lifecycle_config_(a3_pingpong::Model50000SwingLifecycleConfig()),
        station_config_(a3_pingpong::Model50000LateralStationConfig()),
        lifecycle_(lifecycle_config_) {
    if (!onnx_model.empty()) {
      actor_ = std::make_unique<a3_pingpong::OnnxActor>(onnx_model);
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
          upper_body_serve_.Reset();
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

      if (mode == a3_pingpong::ManualMode::kUpperBodyServe) {
        if (!upper_body_serve_enabled_) {
          rejected_count_.fetch_add(1, std::memory_order_relaxed);
          return false;
        }
        a3_pingpong::UpperBodyServeTarget upper_body{};
        a3_pingpong::UpperBodyServeDiagnostics diagnostics;
        std::string reason;
        if (!upper_body_serve_.Step(state, control_dt_s_, upper_body,
                                    &diagnostics, &reason)) {
          upper_body_serve_rejected_count_.fetch_add(
              1, std::memory_order_relaxed);
          return false;
        }
        upper_body_serve_phase_.store(diagnostics.phase,
                                      std::memory_order_relaxed);
        upper_body_serve_tick_.store(diagnostics.tick,
                                     std::memory_order_relaxed);
        if (diagnostics.release_requested) {
          upper_body_serve_release_count_.fetch_add(
              1, std::memory_order_relaxed);
        }
        if (command_output_enabled_) {
          if (!serve_composer_.Build(state, upper_body, std::nullopt,
                                     latest_command_, &reason)) {
            upper_body_serve_rejected_count_.fetch_add(
                1, std::memory_order_relaxed);
            return false;
          }
          latest_command_ready_ = true;
          upper_body_serve_command_count_.fetch_add(
              1, std::memory_order_relaxed);
        }
        if (diagnostics.complete) {
          std::lock_guard<std::mutex> lock(manual_mutex_);
          manual_control_.CompleteUpperBodyServe();
        }
        return true;
      }

      if (mode != a3_pingpong::ManualMode::kMotion) {
        if (command_output_enabled_) PreparePassiveCommand(state);
        return true;
      }
    }

    if (!planner.base_pose ||
        planner.base_pose_age_s > base_pose_timeout_s_) {
      rejected_count_.fetch_add(1, std::memory_order_relaxed);
      return false;
    }

    const auto live_base_w = planner.base_pose->position_w;
    if (!nominal_station_initialized_) {
      nominal_station_xy_ = {live_base_w[0], live_base_w[1]};
      base_target_xy_ = nominal_station_xy_;
      nominal_station_initialized_ = true;
    }

    std::optional<a3_pingpong::RacketTargetInput> fresh_command;
    if (planner.command && planner.command_age_s <= command_timeout_s_) {
      fresh_command = planner.command;
    }
    auto policy_input = planner;
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
      std::clog << "[model_53000 lifecycle] phase="
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
        const auto gains = a3_pingpong::Model50000PolicyGains();
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
    std::lock_guard<std::mutex> lock(manual_mutex_);
    const auto parsed = a3_pingpong::ParseManualKey(key);
    if (parsed == a3_pingpong::ManualKey::kUpperBodyServe &&
        !upper_body_serve_enabled_) {
      return a3_pingpong::ManualActionResult::kRejectedServeDisabled;
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

 private:
  void PreparePassiveCommand(const robot_io::RobotState& state) {
    a3_pingpong::BuildSafeHaltCommand(state, latest_command_);
    latest_command_ready_ = true;
  }

  static constexpr std::uint64_t kPdStandRampTicks = 150;

  a3_pingpong::PingpongObservationBuilder builder_;
  a3_pingpong::PingpongActionAdapter action_adapter_;
  const bool command_output_enabled_;
  const bool manual_control_enabled_;
  const bool upper_body_serve_enabled_;
  const double control_dt_s_;
  const double command_timeout_s_;
  const double base_pose_timeout_s_;
  const a3_pingpong::SwingLifecycleConfig lifecycle_config_;
  const a3_pingpong::LateralStationConfig station_config_;
  a3_pingpong::SwingLifecycle lifecycle_;
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
  a3_pingpong::ManualControl manual_control_;
  std::uint64_t observed_manual_epoch_{0};
  bool pd_stand_initialized_{false};
  std::uint64_t pd_stand_elapsed_ticks_{0};
  std::array<double, a3_pingpong::kPingpongActionDim> pd_stand_start_q_{};
  a3_pingpong::UpperBodyServeTrajectory upper_body_serve_;
  a3_pingpong::FullBodyServeComposer serve_composer_;
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
        [this]() { LogWaistStatus(); });

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
    const bool policy_active =
        diagnostics.waist_policy_valid &&
        (!options_.manual_control ||
         manual_mode == a3_pingpong::ManualMode::kMotion) &&
        (tick_result == a3_pingpong::ReceiveTickResult::kCommandSent ||
         tick_result == a3_pingpong::ReceiveTickResult::kDryRun);

    RCLCPP_INFO(
        get_logger(),
        "waist_diag mode=%s result=%s active=%s order=[yaw,roll,pitch] "
        "policy_raw=[%.3f,%.3f,%.3f] "
        "q_exec_rad=[%.3f,%.3f,%.3f] "
        "q_feedback_rad=[%.3f,%.3f,%.3f] "
        "tau_theoretical_nm=[%.3f,%.3f,%.3f] "
        "tau_feedback_nm=[%.3f,%.3f,%.3f]",
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
        diagnostics.waist_tau_feedback[2]);
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
    const char* controller_result =
        controller_ ? TickResultName(controller_->last_result()) : "disabled";
    const bool leg_damping_active =
        controller_ && controller_->leg_limit_damping_active();
    const int leg_damping_joint =
        controller_ ? controller_->leg_limit_joint_index() : -1;
    const auto leg_damping_count =
        controller_ ? controller_->leg_limit_damping_count() : 0;
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
    const bool waist_policy_active =
        lifecycle_diagnostics.waist_policy_valid &&
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
                "rejected=(%llu,%llu) robot_io=(ticks=%llu,result=%s) "
                "leg_damping=(active=%s,joint=%s,commands=%llu) "
                "manual=(enabled=%s,mode=%s,pd_stand_ready=%s) "
                "serve=(enabled=%s,phase=%s,tick=%llu,commands=%llu,"
                "rejected=%llu,releases=%llu,lower=hold,gains=pd_stand) "
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
                "udp=(accepted=%llu,rejected=%llu,reordered=%llu) "
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
                leg_damping_active ? "yes" : "no",
                LegJointName(leg_damping_joint),
                static_cast<unsigned long long>(leg_damping_count),
                options_.manual_control ? "yes" : "no",
                observation_probe_
                    ? a3_pingpong::ManualModeName(
                          observation_probe_->manual_mode())
                    : "disabled",
                observation_probe_ && observation_probe_->pd_stand_ready()
                    ? "yes" : "no",
                options_.upper_body_serve_dry_run ? "yes" : "no",
                a3_pingpong::UpperBodyServePhaseName(serve_phase),
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
                options_.publish_commands ? "enabled" : "disabled");
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
            options.action_dry_run || options.upper_body_serve_dry_run ||
                options.publish_commands,
            options.manual_control,
            options.upper_body_serve_dry_run,
            options.control_hz,
            options.command_timeout_ms * 1.0e-3,
            options.base_pose_timeout_ms * 1.0e-3);
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
      controller_options.require_fresh_base_pose = !options.manual_control;
      controller_options.publish_commands = options.publish_commands;
      controller_options.leg_damping_safety.damping_kd =
          options.leg_damping_kd;
      // ParseOptions has already applied the requested margin to the default
      // A3 mechanical ranges.
      controller_options.leg_damping_safety.lower =
          options.leg_damping_safety.lower;
      controller_options.leg_damping_safety.upper =
          options.leg_damping_safety.upper;
      a3_pingpong::ReceivePolicyFn policy;
      if (options.action_dry_run || options.upper_body_serve_dry_run ||
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
                  "gains=model_53000/pd_stand_production "
                  "publish_enabled=%s",
                  observation_probe ? "111d" : "disabled",
                  options.onnx_model.empty() ? "disabled" : "model_53000",
                  options.publish_commands
                      ? "send"
                      : ((options.action_dry_run ||
                          options.upper_body_serve_dry_run)
                             ? "full_dry_run"
                             : "disabled"),
                  options.manual_control ? "P/S/M/V/X" : "automatic_probe",
                  options.upper_body_serve_dry_run ? "right_arm+hold_lower"
                                                   : "disabled",
                  options.publish_commands ? "true" : "false");
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
                    "manual shadow keys: P=passive S=pd_stand M=motion "
                    "V=upper_body_serve "
                    "I=status X=halt H=help Q=quit(passive only)");
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
                     a3_pingpong::ManualActionResult::kRejectedServeDisabled) {
            RCLCPP_ERROR(node->get_logger(),
                         "V rejected: start with --upper-body-serve-dry-run");
          } else if (result ==
                     a3_pingpong::ManualActionResult::kRejectedQuitWhileActive) {
            RCLCPP_ERROR(node->get_logger(),
                         "Q rejected: return to passive with P first");
          } else if (result ==
                     a3_pingpong::ManualActionResult::kHelpRequested) {
            RCLCPP_INFO(node->get_logger(),
                        "P=passive S=pd_stand M=motion V=upper_body_serve "
                        "I=status X=halt "
                        "H=help Q=quit(passive only)");
          } else if (result ==
                     a3_pingpong::ManualActionResult::kStatusRequested) {
            RCLCPP_INFO(node->get_logger(), "manual mode=%s pd_stand_ready=%s",
                        a3_pingpong::ManualModeName(mode),
                        observation_probe->pd_stand_ready() ? "yes" : "no");
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
