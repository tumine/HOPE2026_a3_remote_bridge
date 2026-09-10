#pragma once

#include "a3_pingpong/a3_leg_limits.hpp"
#include "a3_pingpong/planner_input.hpp"
#include "robot_io/robot_io_backend.hpp"

#include <atomic>
#include <array>
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
  // Zero disables the RobotIO age/sync watchdog. The latest assembled state is
  // then kept in control until the operator changes mode manually.
  std::int64_t max_state_age_ns{50'000'000};

  // The ping-pong lifecycle can run its fixed READY target without a live
  // RacketCommand. Keep true for callers that require a command every tick;
  // the model_50000 lifecycle sets this false and still requires live pose.
  bool require_fresh_command{true};

  // Manual PASSIVE/PD_STAND must be usable before the PC planner/mocap path is
  // online. When false, the observation/policy callback owns pose validation.
  bool require_fresh_base_pose{true};

  // False is the required bring-up default.  The policy and safety path can
  // be exercised without registering body-drive command publishers.
  bool publish_commands{false};

  // A3 SDK leg order: left hip-p/r/y, knee, ankle-p/r, then right leg. The
  // default commissioning guard protects only both hip-pitch joints. The
  // remaining leg joints are intentionally not checked by this guard.
  struct LegDampingSafety {
    bool enabled{true};
    std::array<double, kA3LegDof> lower{};
    std::array<double, kA3LegDof> upper{};
    std::array<bool, kA3LegDof> protected_joints{};
    double damping_kd{2.0};

    LegDampingSafety();
  } leg_damping_safety;

  // Optional legacy positive-pitch virtual wall. It is disabled by default for
  // model_72500 because training locks waist roll/pitch near zero. When enabled,
  // it replaces only the actuator command with a stiff recovery PD target.
  struct WaistPitchSafety {
    bool enabled{false};
    double enter_rad{0.35};
    double release_rad{0.27};
    double recovery_target_rad{0.0};
    double recovery_kp{400.0};
    double recovery_kd{8.0};
  } waist_pitch_safety;
};

enum class ReceiveTickResult {
  kNoState,
  kPlannerInputNotReady,
  kStateStale,
  kObservationRejected,
  kPolicyUnavailable,
  kPolicyRejected,
  kCommandInvalid,
  kLegLimitDamping,
  kCommandSent,
  kDryRun,
};

// Build a compliant hold command using the latest measured q.  This mirrors
// HOPE's safe-halt rule: keep the command stream shape but set all gains,
// velocity and feed-forward terms to zero.
void BuildSafeHaltCommand(const robot_io::RobotState& state,
                          robot_io::RobotCommand& output);

// Build a zero-position-error damping command. q_des follows the measured
// state, Kp/tau_ff are zero, and Kd damps measured joint velocity.
bool BuildDampingCommand(const robot_io::RobotState& state, double damping_kd,
                         robot_io::RobotCommand& output);

// Build the manual P-mode command: q_des follows the measured state, Kp and
// feed-forward terms stay zero, while both arms and legs receive mild velocity
// damping.  Waist and neck remain zero-gain.
bool BuildPassiveDampingCommand(const robot_io::RobotState& state,
                                double damping_kd,
                                robot_io::RobotCommand& output);

bool ValidateRobotCommand(const robot_io::RobotCommand& command,
                          int expected_dof);

bool ValidateWaistPitchSafety(
    const ReceiveControllerOptions::WaistPitchSafety& safety);

// Apply the waist-pitch virtual wall to a complete 31-DOF actuator command.
// Returns false for malformed state/command data. `active` carries the
// hysteresis latch between control ticks.
bool ApplyWaistPitchSafety(
    const robot_io::RobotState& state,
    const ReceiveControllerOptions::WaistPitchSafety& safety,
    bool& active, robot_io::RobotCommand& command);

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
  std::uint64_t leg_limit_damping_count() const noexcept {
    return leg_limit_damping_count_.load(std::memory_order_relaxed);
  }
  bool leg_limit_damping_active() const noexcept {
    return leg_limit_damping_active_.load(std::memory_order_acquire);
  }
  int leg_limit_joint_index() const noexcept {
    return leg_limit_joint_index_.load(std::memory_order_relaxed);
  }
  bool waist_pitch_guard_active() const noexcept {
    return waist_pitch_guard_active_.load(std::memory_order_acquire);
  }
  std::uint64_t waist_pitch_guard_trigger_count() const noexcept {
    return waist_pitch_guard_trigger_count_.load(std::memory_order_relaxed);
  }
  double waist_pitch_guard_output_rad() const noexcept {
    return waist_pitch_guard_output_rad_.load(std::memory_order_relaxed);
  }
  ReceiveTickResult last_result() const noexcept {
    return last_result_.load(std::memory_order_relaxed);
  }
  std::uint64_t result_count(ReceiveTickResult result) const noexcept {
    const auto index = static_cast<std::size_t>(result);
    return index < result_counts_.size()
               ? result_counts_[index].load(std::memory_order_relaxed)
               : 0;
  }
  std::int64_t last_state_age_ns() const noexcept {
    return last_state_age_ns_.load(std::memory_order_relaxed);
  }
  bool last_sync_complete() const noexcept {
    return last_sync_complete_.load(std::memory_order_relaxed);
  }
  bool last_sync_aligned() const noexcept {
    return last_sync_aligned_.load(std::memory_order_relaxed);
  }
  std::uint64_t send_failure_count() const noexcept {
    return send_failure_count_.load(std::memory_order_relaxed);
  }
  std::uint64_t safe_halt_send_failure_count() const noexcept {
    return safe_halt_send_failure_count_.load(std::memory_order_relaxed);
  }

 private:
  void OnState(const robot_io::RobotState& state) noexcept;
  void Run();
  ReceiveTickResult RunOneTick(std::int64_t now_ns);
  void MaybeSendSafeHalt(const robot_io::RobotState& state) noexcept;
  void MaybeSendLegDamping(const robot_io::RobotState& state) noexcept;
  bool FindLegLimitViolation(const robot_io::RobotState& state,
                             const robot_io::RobotCommand* command,
                             int* joint_index) const noexcept;
  void LatchLegLimit(int joint_index) noexcept;

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
  std::atomic<std::uint64_t> leg_limit_damping_count_{0};
  std::atomic<bool> leg_limit_damping_active_{false};
  std::atomic<int> leg_limit_joint_index_{-1};
  std::atomic<bool> waist_pitch_guard_active_{false};
  std::atomic<std::uint64_t> waist_pitch_guard_trigger_count_{0};
  std::atomic<double> waist_pitch_guard_output_rad_{0.0};
  std::atomic<ReceiveTickResult> last_result_{ReceiveTickResult::kNoState};
  static constexpr std::size_t kReceiveTickResultCount =
      static_cast<std::size_t>(ReceiveTickResult::kDryRun) + 1;
  std::array<std::atomic<std::uint64_t>, kReceiveTickResultCount>
      result_counts_{};
  std::atomic<std::int64_t> last_state_age_ns_{-1};
  std::atomic<bool> last_sync_complete_{false};
  std::atomic<bool> last_sync_aligned_{false};
  std::atomic<std::uint64_t> send_failure_count_{0};
  std::atomic<std::uint64_t> safe_halt_send_failure_count_{0};
};

}  // namespace a3_pingpong
