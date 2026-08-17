#pragma once

#include "a3_pingpong/planner_input.hpp"
#include "robot_io/robot_io_backend.hpp"

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <thread>

namespace a3_pingpong {

// The policy is deliberately injected at this boundary.  This keeps the
// RobotIO/control-loop contract testable without a model runtime and lets the
// receive policy be replaced by ONNX/RKNN later.
using ReceivePolicyFn = std::function<bool(
    const PlannerInputSnapshot&, const robot_io::RobotState&,
    robot_io::RobotCommand&)>;

// Optional read-only hook executed after state/planner freshness checks and
// before policy inference. It must never publish or mutate RobotIO state.
using ObservationProbeFn = std::function<bool(
    const PlannerInputSnapshot&, const robot_io::RobotState&)>;

struct ReceiveControllerOptions {
  double control_hz{50.0};
  double command_timeout_s{0.150};
  double base_pose_timeout_s{0.100};
  std::int64_t max_state_age_ns{50'000'000};

  // The ping-pong lifecycle can run its fixed READY target without a live
  // RacketCommand. Keep true for callers that require a command every tick;
  // the model_21500 lifecycle sets this false and still requires live pose.
  bool require_fresh_command{true};

  // Manual PASSIVE/PD_STAND must be usable before the PC planner/mocap path is
  // online. When false, the observation/policy callback owns pose validation.
  bool require_fresh_base_pose{true};

  // False is the required bring-up default.  The policy and safety path can
  // be exercised without registering body-drive command publishers.
  bool publish_commands{false};
};

enum class ReceiveTickResult {
  kNoState,
  kPlannerInputNotReady,
  kStateStale,
  kObservationRejected,
  kPolicyUnavailable,
  kPolicyRejected,
  kCommandInvalid,
  kCommandSent,
  kDryRun,
};

// Build a compliant hold command using the latest measured q.  This mirrors
// HOPE's safe-halt rule: keep the command stream shape but set all gains,
// velocity and feed-forward terms to zero.
void BuildSafeHaltCommand(const robot_io::RobotState& state,
                          robot_io::RobotCommand& output);

bool ValidateRobotCommand(const robot_io::RobotCommand& command,
                          int expected_dof);

class ReceiveController {
 public:
  ReceiveController(robot_io::RobotIOBackend& backend,
                    PlannerInputMailbox& planner_mailbox,
                    ReceivePolicyFn policy,
                    ReceiveControllerOptions options = {});
  ~ReceiveController();

  ReceiveController(const ReceiveController&) = delete;
  ReceiveController& operator=(const ReceiveController&) = delete;

  // The caller must call backend.Init() first. Start registers the state
  // callback, starts the backend I/O, and launches the 50 Hz control thread.
  // The observation probe may only be installed while stopped.
  bool SetObservationProbe(ObservationProbeFn probe);
  bool Start();
  void Stop();

  bool running() const noexcept {
    return running_.load(std::memory_order_acquire);
  }
  std::uint64_t tick_count() const noexcept {
    return tick_count_.load(std::memory_order_relaxed);
  }
  std::uint64_t command_sent_count() const noexcept {
    return command_sent_count_.load(std::memory_order_relaxed);
  }
  std::uint64_t safe_halt_count() const noexcept {
    return safe_halt_count_.load(std::memory_order_relaxed);
  }
  ReceiveTickResult last_result() const noexcept {
    return last_result_.load(std::memory_order_relaxed);
  }

 private:
  void OnState(const robot_io::RobotState& state) noexcept;
  void Run();
  ReceiveTickResult RunOneTick(std::int64_t now_ns);
  void MaybeSendSafeHalt(const robot_io::RobotState& state) noexcept;

  robot_io::RobotIOBackend& backend_;
  PlannerInputMailbox& planner_mailbox_;
  ReceivePolicyFn policy_;
  ObservationProbeFn observation_probe_;
  ReceiveControllerOptions options_;

  std::shared_ptr<const robot_io::RobotState> latest_state_;
  std::thread control_thread_;
  std::atomic<bool> running_{false};
  std::atomic<std::uint64_t> tick_count_{0};
  std::atomic<std::uint64_t> command_sent_count_{0};
  std::atomic<std::uint64_t> safe_halt_count_{0};
  std::atomic<ReceiveTickResult> last_result_{ReceiveTickResult::kNoState};
};

}  // namespace a3_pingpong
