#include "a3_pingpong/onnx_actor.hpp"
#include "a3_pingpong/manual_control.hpp"
#include "a3_pingpong/pingpong_action_adapter.hpp"
#include "a3_pingpong/pingpong_observation_builder.hpp"
#include "a3_pingpong/planner_input.hpp"
#include "a3_pingpong/receive_controller.hpp"
#include "a3_pingpong/swing_lifecycle.hpp"

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

namespace {

struct Options {
  std::string command_topic{"/racket/command"};
  std::string base_pose_topic{"/a3_mocap/pelvis_pose"};
  std::string expected_frame{"hope_table"};
  std::string aimrt_cfg;
  double command_timeout_ms{150.0};
  double base_pose_timeout_ms{100.0};
  double state_timeout_ms{50.0};
  double control_hz{50.0};
  bool observation_probe{false};
  bool onnx_requested{false};
  bool action_dry_run{false};
  bool manual_control{false};
  bool publish_commands{false};
  std::string onnx_model;
};

void Usage(const char* program) {
  std::cout
      << "Usage: " << program << " [options]\n\n"
      << "Receive PC-side HOPE planner inputs and run the MDU policy.\n"
      << "Command publishing is disabled unless explicitly requested.\n\n"
      << "Options:\n"
      << "  --command-topic TOPIC       (default: /racket/command)\n"
      << "  --base-pose-topic TOPIC     (default: /a3_mocap/pelvis_pose)\n"
      << "  --expected-frame FRAME      (default: hope_table)\n"
      << "  --aimrt-cfg PATH            enable the RobotIO backend\n"
      << "  --observation-probe         build 111-D observations; no inference\n"
      << "  --onnx-model PATH           run model_21500 inference\n"
      << "  --action-dry-run            build full RobotCommand; do not send\n"
      << "  --manual-control            P/S/M/X keyboard control state machine\n"
      << "  --publish-commands          enable official RobotIO SendCommand path\n"
      << "  --command-timeout-ms MS     (default: 150)\n"
      << "  --base-pose-timeout-ms MS   (default: 100)\n"
      << "  --state-timeout-ms MS       (default: 50)\n"
      << "  --control-hz HZ             (default: 50)\n"
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
    if (take_value("--command-topic", options.command_topic) ||
        take_value("--base-pose-topic", options.base_pose_topic) ||
        take_value("--expected-frame", options.expected_frame) ||
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
    throw std::runtime_error("unknown or incomplete option: " + argument);
  }

  const auto valid_topic = [](const std::string& value) {
    return !value.empty() && value.front() == '/';
  };
  if (!valid_topic(options.command_topic) ||
      !valid_topic(options.base_pose_topic)) {
    throw std::runtime_error("topics must be absolute ROS names");
  }
  if (options.expected_frame.empty()) {
    throw std::runtime_error("--expected-frame cannot be empty");
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
      !std::isfinite(options.control_hz) || options.control_hz <= 0.0) {
    throw std::runtime_error("timeouts must be finite and positive");
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
    case a3_pingpong::ReceiveTickResult::kCommandSent:
      return "command_sent";
    case a3_pingpong::ReceiveTickResult::kDryRun:
      return "dry_run";
  }
  return "unknown";
}

class ObservationProbe {
 public:
  ObservationProbe(const std::string& onnx_model, bool command_output_enabled,
                   bool manual_control, double command_timeout_s,
                   double base_pose_timeout_s)
      : builder_(a3_pingpong::Model21500ObservationConfig()),
        action_adapter_(a3_pingpong::Model21500ActionAdapterConfig()),
        command_output_enabled_(command_output_enabled),
        manual_control_enabled_(manual_control),
        command_timeout_s_(command_timeout_s),
        base_pose_timeout_s_(base_pose_timeout_s),
        lifecycle_config_(a3_pingpong::Model21500SwingLifecycleConfig()),
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
          pd_stand_initialized_ = false;
          pd_stand_elapsed_ticks_ = 0;
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
        if (command_output_enabled_) PreparePassiveCommand(state);
        return true;
      }
    }

    if (!planner.base_pose ||
        planner.base_pose_age_s > base_pose_timeout_s_) {
      rejected_count_.fetch_add(1, std::memory_order_relaxed);
      return false;
    }

    std::optional<a3_pingpong::RacketTargetInput> fresh_command;
    if (planner.command && planner.command_age_s <= command_timeout_s_ &&
        planner.command->time_to_strike_s >= 0.0) {
      fresh_command = planner.command;
    }
    auto policy_input = planner;
    policy_input.command = lifecycle_.Update(fresh_command);
    policy_input.command_age_s = 0.0;
    const std::array<double, 2> fixed_station_xy = {
        lifecycle_config_.ready_reference_base_w[0],
        lifecycle_config_.ready_reference_base_w[1]};

    a3_pingpong::PingpongObservation observation{};
    std::string reason;
    if (!builder_.Build(state, policy_input, last_action_, fixed_station_xy,
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
    return manual_control_.Apply(a3_pingpong::ParseManualKey(key));
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
  const double command_timeout_s_;
  const double base_pose_timeout_s_;
  const a3_pingpong::SwingLifecycleConfig lifecycle_config_;
  a3_pingpong::SwingLifecycle lifecycle_;
  mutable std::mutex manual_mutex_;
  a3_pingpong::ManualControl manual_control_;
  std::uint64_t observed_manual_epoch_{0};
  bool pd_stand_initialized_{false};
  std::uint64_t pd_stand_elapsed_ticks_{0};
  std::array<double, a3_pingpong::kPingpongActionDim> pd_stand_start_q_{};
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
    const auto command_qos = rclcpp::QoS(rclcpp::KeepLast(10))
                                 .reliable()
                                 .durability_volatile();
    const auto pose_qos = rclcpp::QoS(rclcpp::KeepLast(5))
                              .best_effort()
                              .durability_volatile();

    command_subscription_ = create_subscription<hope_msgs::msg::RacketCommand>(
        options_.command_topic, command_qos,
        [this](const hope_msgs::msg::RacketCommand::SharedPtr message) {
          OnCommand(message);
        });
    pose_subscription_ = create_subscription<geometry_msgs::msg::PoseStamped>(
        options_.base_pose_topic, pose_qos,
        [this](const geometry_msgs::msg::PoseStamped::SharedPtr message) {
          OnBasePose(message);
        });
    status_timer_ = create_wall_timer(std::chrono::seconds(1),
                                      [this]() { LogStatus(); });

    RCLCPP_WARN(get_logger(),
                "MDU planner receiver started: command=%s pose=%s frame=%s; "
                "body_drive_publishers=0",
                options_.command_topic.c_str(), options_.base_pose_topic.c_str(),
                options_.expected_frame.c_str());
  }

  a3_pingpong::PlannerInputMailbox& mailbox() { return mailbox_; }

  void SetController(a3_pingpong::ReceiveController* controller) {
    controller_ = controller;
  }

  void SetObservationProbe(ObservationProbe* probe) {
    observation_probe_ = probe;
  }

 private:
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
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                           "RacketCommand %s: %s", ResultName(result),
                           reason.c_str());
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
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                           "BasePose %s: %s", ResultName(result),
                           reason.c_str());
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
    const auto action_decoded =
        observation_probe_ ? observation_probe_->action_decode_count() : 0;
    const auto action_rejected =
        observation_probe_ ? observation_probe_->action_rejected_count() : 0;
    const auto dry_run_commands =
        observation_probe_ ? observation_probe_->dry_run_command_count() : 0;
    const auto raw_clip_total =
        observation_probe_ ? observation_probe_->raw_clip_total() : 0;
    const auto position_clip_total =
        observation_probe_ ? observation_probe_->position_clip_total() : 0;
    const auto latest_position_clips = observation_probe_
                                           ? observation_probe_
                                                 ->latest_position_clip_count()
                                           : 0;
    const double q_des_max_abs = observation_probe_
                                     ? observation_probe_->latest_q_des_max_abs()
                                     : 0.0;
    const double max_tracking_error =
        observation_probe_
            ? observation_probe_->latest_max_tracking_error()
            : 0.0;

    RCLCPP_INFO(get_logger(),
                "planner_input ready=%s command_fresh=%s task=%llu "
                "revision=%u tts=%.3fs "
                "command_age=%.1fms pose_age=%.1fms accepted=(%llu,%llu) "
                "rejected=(%llu,%llu) robot_io=(ticks=%llu,result=%s) "
                "manual=(enabled=%s,mode=%s,pd_stand_ready=%s) "
                "observation=(enabled=%s,built=%llu,rejected=%llu,"
                "max_abs=%.3f,tts=%.3f) "
                "inference=(enabled=%s,runs=%llu,rejected=%llu,"
                "avg_ms=%.3f,max_ms=%.3f,raw_max_abs=%.3f) "
                "action=(output=%s,decoded=%llu,rejected=%llu,commands=%llu,"
                "raw_clips=%llu,pos_clips=%llu/latest:%u,q_des_max_abs=%.3f,"
                "max_error=%.3f,gains=model_21500) "
                "body_drive_publishers=%s",
                pose_ready ? "yes" : "no",
                command_fresh ? "yes" : "no",
                static_cast<unsigned long long>(task_id), revision, tts,
                command_age_ms, pose_age_ms,
                static_cast<unsigned long long>(accepted_commands_.load()),
                static_cast<unsigned long long>(accepted_poses_.load()),
                static_cast<unsigned long long>(rejected_commands_.load()),
                static_cast<unsigned long long>(rejected_poses_.load()),
                static_cast<unsigned long long>(controller_ticks),
                controller_result,
                options_.manual_control ? "yes" : "no",
                observation_probe_
                    ? a3_pingpong::ManualModeName(
                          observation_probe_->manual_mode())
                    : "disabled",
                observation_probe_ && observation_probe_->pd_stand_ready()
                    ? "yes" : "no",
                observation_probe_ ? "yes" : "no",
                static_cast<unsigned long long>(observation_built),
                static_cast<unsigned long long>(observation_rejected),
                observation_max_abs, observation_tts,
                inference_enabled ? "yes" : "no",
                static_cast<unsigned long long>(inference_count),
                static_cast<unsigned long long>(inference_rejected),
                inference_average_ms, inference_max_ms, raw_max_abs,
                options_.publish_commands
                    ? "send"
                    : (options_.action_dry_run ? "dry_run" : "disabled"),
                static_cast<unsigned long long>(action_decoded),
                static_cast<unsigned long long>(action_rejected),
                static_cast<unsigned long long>(dry_run_commands),
                static_cast<unsigned long long>(raw_clip_total),
                static_cast<unsigned long long>(position_clip_total),
                latest_position_clips, q_des_max_abs, max_tracking_error,
                options_.publish_commands ? "enabled" : "disabled");
  }

  Options options_;
  a3_pingpong::PlannerInputMailbox mailbox_;
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
            options.action_dry_run || options.publish_commands,
            options.manual_control,
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
      a3_pingpong::ReceivePolicyFn policy;
      if (options.action_dry_run || options.publish_commands) {
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
                  "inference=%s command_output=%s manual=%s "
                  "gains=model_21500/pd_stand_production "
                  "publish_enabled=%s",
                  observation_probe ? "111d" : "disabled",
                  options.onnx_model.empty() ? "disabled" : "model_21500",
                  options.publish_commands
                      ? "send"
                      : (options.action_dry_run ? "full_dry_run" : "disabled"),
                  options.manual_control ? "P/S/M/X" : "automatic_probe",
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
                         "M rejected: press S and wait for pd_stand_ready=yes");
          } else if (result ==
                     a3_pingpong::ManualActionResult::kRejectedQuitWhileActive) {
            RCLCPP_ERROR(node->get_logger(),
                         "Q rejected: return to passive with P first");
          } else if (result ==
                     a3_pingpong::ManualActionResult::kHelpRequested) {
            RCLCPP_INFO(node->get_logger(),
                        "P=passive S=pd_stand M=motion I=status X=halt "
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
