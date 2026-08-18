#include "a3_pingpong/receive_controller.hpp"

#include "robot_io/a3_layout_extra.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <utility>

namespace a3_pingpong {
namespace {

std::int64_t SystemNowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

bool FiniteVector(const Eigen::VectorXd& value, int expected_dof) {
  return value.size() == expected_dof && value.array().isFinite().all();
}

bool ValidLegDampingSafety(
    const ReceiveControllerOptions::LegDampingSafety& safety) {
  if (!safety.enabled) return true;
  if (!std::isfinite(safety.damping_kd) || safety.damping_kd <= 0.0) {
    return false;
  }
  for (std::size_t index = 0; index < kA3LegDof; ++index) {
    if (!std::isfinite(safety.lower[index]) ||
        !std::isfinite(safety.upper[index]) ||
        safety.lower[index] >= safety.upper[index]) {
      return false;
    }
  }
  return true;
}

}  // namespace

ReceiveControllerOptions::LegDampingSafety::LegDampingSafety()
    : lower(ScaleA3LegLimits(kA3LegUrdfLower, 0.90)),
      upper(ScaleA3LegLimits(kA3LegUrdfUpper, 0.90)) {
  // Only left/right hip-pitch remain protected for the current deployment.
  protected_joints.fill(false);
  protected_joints[0] = true;
  protected_joints[6] = true;
}

void BuildSafeHaltCommand(const robot_io::RobotState& state,
                          robot_io::RobotCommand& output) {
  constexpr int dof = robot_io::kA3Dof;
  output.q_des = Eigen::VectorXd::Zero(dof);
  output.dq_des = Eigen::VectorXd::Zero(dof);
  output.tau_ff = Eigen::VectorXd::Zero(dof);
  output.kp = Eigen::VectorXd::Zero(dof);
  output.kd = Eigen::VectorXd::Zero(dof);
  if (state.q.size() == dof && state.q.array().isFinite().all()) {
    output.q_des = state.q;
  }
}

bool BuildDampingCommand(const robot_io::RobotState& state, double damping_kd,
                         robot_io::RobotCommand& output) {
  constexpr int dof = robot_io::kA3Dof;
  if (!std::isfinite(damping_kd) || damping_kd <= 0.0 ||
      !FiniteVector(state.q, dof)) {
    return false;
  }
  output.q_des = state.q;
  output.dq_des = Eigen::VectorXd::Zero(dof);
  output.tau_ff = Eigen::VectorXd::Zero(dof);
  output.kp = Eigen::VectorXd::Zero(dof);
  output.kd = Eigen::VectorXd::Zero(dof);
  // Keep the upper body passive while damping every leg joint that triggered
  // the safety mode. q_des remains measured for the complete 31-DOF packet.
  output.kd.segment(kA3LegCommandStart, static_cast<Eigen::Index>(kA3LegDof))
      .setConstant(damping_kd);
  return true;
}

bool ValidateRobotCommand(const robot_io::RobotCommand& command,
                          int expected_dof) {
  return expected_dof > 0 && FiniteVector(command.q_des, expected_dof) &&
         FiniteVector(command.dq_des, expected_dof) &&
         FiniteVector(command.tau_ff, expected_dof) &&
         FiniteVector(command.kp, expected_dof) &&
         FiniteVector(command.kd, expected_dof);
}

ReceiveController::ReceiveController(robot_io::RobotIOBackend& backend,
                                     PlannerInputMailbox& planner_mailbox,
                                     ReceivePolicyFn policy,
                                     ReceiveControllerOptions options)
    : backend_(backend),
      planner_mailbox_(planner_mailbox),
      policy_(std::move(policy)),
      options_(options) {}

ReceiveController::~ReceiveController() { Stop(); }

bool ReceiveController::SetObservationProbe(ObservationProbeFn probe) {
  if (running_.load(std::memory_order_acquire)) return false;
  observation_probe_ = std::move(probe);
  return true;
}

bool ReceiveController::Start() {
  if (running_.exchange(true, std::memory_order_acq_rel)) return false;
  if (!std::isfinite(options_.control_hz) || options_.control_hz <= 0.0 ||
      !std::isfinite(options_.command_timeout_s) ||
      options_.command_timeout_s <= 0.0 ||
      !std::isfinite(options_.base_pose_timeout_s) ||
      options_.base_pose_timeout_s <= 0.0 || options_.max_state_age_ns <= 0) {
    running_.store(false, std::memory_order_release);
    return false;
  }
  if (!ValidLegDampingSafety(options_.leg_damping_safety)) {
    running_.store(false, std::memory_order_release);
    return false;
  }

  backend_.RegisterStateCallback(
      [this](const robot_io::RobotState& state) { OnState(state); });
  if (!backend_.Start()) {
    running_.store(false, std::memory_order_release);
    return false;
  }

  try {
    control_thread_ = std::thread([this]() { Run(); });
  } catch (...) {
    backend_.Stop();
    running_.store(false, std::memory_order_release);
    throw;
  }
  return true;
}

void ReceiveController::Stop() {
  if (!running_.exchange(false, std::memory_order_acq_rel)) return;
  if (control_thread_.joinable()) control_thread_.join();
  backend_.Stop();
}

void ReceiveController::OnState(const robot_io::RobotState& state) noexcept {
  auto copy = std::make_shared<robot_io::RobotState>(state);
  std::atomic_store_explicit(&latest_state_,
                             std::shared_ptr<const robot_io::RobotState>(
                                 std::move(copy)),
                             std::memory_order_release);
}

void ReceiveController::Run() {
  const auto period = std::chrono::duration<double>(1.0 / options_.control_hz);
  auto next = std::chrono::steady_clock::now();
  while (running_.load(std::memory_order_acquire)) {
    next += std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        period);
    const auto result = RunOneTick(SystemNowNs());
    last_result_.store(result, std::memory_order_relaxed);
    tick_count_.fetch_add(1, std::memory_order_relaxed);
    std::this_thread::sleep_until(next);
  }
}

ReceiveTickResult ReceiveController::RunOneTick(std::int64_t now_ns) {
  const auto state = std::atomic_load_explicit(
      &latest_state_, std::memory_order_acquire);
  if (!state) return ReceiveTickResult::kNoState;
  if (state->timestamp_ns <= 0 ||
      now_ns - state->timestamp_ns > options_.max_state_age_ns ||
      now_ns + options_.max_state_age_ns < state->timestamp_ns ||
      !state->sync_complete || !state->sync_aligned) {
    MaybeSendSafeHalt(*state);
    return ReceiveTickResult::kStateStale;
  }

  if (leg_limit_damping_active_.load(std::memory_order_acquire)) {
    MaybeSendLegDamping(*state);
    return ReceiveTickResult::kLegLimitDamping;
  }
  int leg_limit_joint = -1;
  if (FindLegLimitViolation(*state, nullptr, &leg_limit_joint)) {
    LatchLegLimit(leg_limit_joint);
    MaybeSendLegDamping(*state);
    return ReceiveTickResult::kLegLimitDamping;
  }

  const auto planner = planner_mailbox_.Snapshot(SteadyClock::now());
  const bool base_pose_ready =
      planner.base_pose.has_value() &&
      planner.base_pose_age_s <= options_.base_pose_timeout_s;
  const bool planner_ready =
      (!options_.require_fresh_base_pose || base_pose_ready) &&
      (!options_.require_fresh_command ||
       planner.Ready(options_.command_timeout_s,
                     options_.base_pose_timeout_s));
  if (!planner_ready) {
    MaybeSendSafeHalt(*state);
    return ReceiveTickResult::kPlannerInputNotReady;
  }
  if (observation_probe_) {
    try {
      if (!observation_probe_(planner, *state)) {
        MaybeSendSafeHalt(*state);
        return ReceiveTickResult::kObservationRejected;
      }
    } catch (...) {
      MaybeSendSafeHalt(*state);
      return ReceiveTickResult::kObservationRejected;
    }
  }
  if (!policy_) {
    MaybeSendSafeHalt(*state);
    return ReceiveTickResult::kPolicyUnavailable;
  }

  robot_io::RobotCommand command;
  try {
    if (!policy_(planner, *state, command)) {
      MaybeSendSafeHalt(*state);
      return ReceiveTickResult::kPolicyRejected;
    }
  } catch (...) {
    MaybeSendSafeHalt(*state);
    return ReceiveTickResult::kPolicyRejected;
  }
  if (!ValidateRobotCommand(command, backend_.GetLayout().dof())) {
    MaybeSendSafeHalt(*state);
    return ReceiveTickResult::kCommandInvalid;
  }
  leg_limit_joint = -1;
  if (FindLegLimitViolation(*state, &command, &leg_limit_joint)) {
    LatchLegLimit(leg_limit_joint);
    MaybeSendLegDamping(*state);
    return ReceiveTickResult::kLegLimitDamping;
  }
  if (!options_.publish_commands) return ReceiveTickResult::kDryRun;
  if (!backend_.SendCommand(command)) return ReceiveTickResult::kCommandInvalid;
  command_sent_count_.fetch_add(1, std::memory_order_relaxed);
  return ReceiveTickResult::kCommandSent;
}

void ReceiveController::MaybeSendSafeHalt(
    const robot_io::RobotState& state) noexcept {
  if (!options_.publish_commands ||
      backend_.GetLayout().dof() != robot_io::kA3Dof) {
    return;
  }
  robot_io::RobotCommand halt;
  BuildSafeHaltCommand(state, halt);
  if (ValidateRobotCommand(halt, robot_io::kA3Dof) &&
      backend_.SendCommand(halt)) {
    safe_halt_count_.fetch_add(1, std::memory_order_relaxed);
  }
}

void ReceiveController::MaybeSendLegDamping(
    const robot_io::RobotState& state) noexcept {
  if (!options_.publish_commands ||
      backend_.GetLayout().dof() != robot_io::kA3Dof) {
    return;
  }
  robot_io::RobotCommand damping;
  if (BuildDampingCommand(state, options_.leg_damping_safety.damping_kd,
                          damping) &&
      ValidateRobotCommand(damping, robot_io::kA3Dof) &&
      backend_.SendCommand(damping)) {
    leg_limit_damping_count_.fetch_add(1, std::memory_order_relaxed);
  }
}

bool ReceiveController::FindLegLimitViolation(
    const robot_io::RobotState& state, const robot_io::RobotCommand* command,
    int* joint_index) const noexcept {
  if (!options_.leg_damping_safety.enabled ||
      backend_.GetLayout().dof() != robot_io::kA3Dof ||
      state.q.size() != robot_io::kA3Dof ||
      !state.q.array().isFinite().all()) {
    return false;
  }
  if (command &&
      (command->q_des.size() != robot_io::kA3Dof ||
       !command->q_des.array().isFinite().all())) {
    return false;
  }
  for (std::size_t leg = 0; leg < kA3LegDof; ++leg) {
    if (!options_.leg_damping_safety.protected_joints[leg]) continue;
    const int index = kA3LegCommandStart + static_cast<int>(leg);
    const double measured = state.q[index];
    if (measured < options_.leg_damping_safety.lower[leg] ||
        measured > options_.leg_damping_safety.upper[leg]) {
      if (joint_index) *joint_index = index;
      return true;
    }
    if (command) {
      const double target = (*command).q_des[index];
      if (target < options_.leg_damping_safety.lower[leg] ||
          target > options_.leg_damping_safety.upper[leg]) {
        if (joint_index) *joint_index = index;
        return true;
      }
    }
  }
  return false;
}

void ReceiveController::LatchLegLimit(int joint_index) noexcept {
  bool expected = false;
  if (leg_limit_damping_active_.compare_exchange_strong(
          expected, true, std::memory_order_acq_rel,
          std::memory_order_acquire)) {
    leg_limit_joint_index_.store(joint_index, std::memory_order_release);
  }
}

}  // namespace a3_pingpong
