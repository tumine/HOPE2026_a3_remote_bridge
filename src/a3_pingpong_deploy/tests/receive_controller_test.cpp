#include "a3_pingpong/receive_controller.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#define CHECK(condition)      \
  do {                        \
    if (!(condition)) return __LINE__; \
  } while (false)

namespace {

std::int64_t SystemNowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

class FakeBackend final : public robot_io::RobotIOBackend {
 public:
  bool Init(const std::string&) override { return true; }
  bool Start() override {
    started = true;
    if (callback) callback(state);
    return true;
  }
  void Stop() override { started = false; }
  const robot_io::JointLayout& GetLayout() const override { return layout; }
  void RegisterStateCallback(StateCallback value) override {
    callback = std::move(value);
  }
  bool SendCommand(const robot_io::RobotCommand& command) override {
    last_command = command;
    ++send_count;
    return true;
  }
  std::string Name() const override { return "fake_a3"; }
  double StateRateHz() const override { return 100.0; }

  robot_io::JointLayout layout{std::vector<std::string>(31, "joint")};
  robot_io::RobotState state;
  StateCallback callback;
  robot_io::RobotCommand last_command;
  std::atomic<int> send_count{0};
  bool started{false};
};

}  // namespace

int main() {
  const a3_pingpong::ReceiveControllerOptions::LegDampingSafety urdf_safety;
  for (std::size_t index = 0; index < a3_pingpong::kA3LegDof; ++index) {
    CHECK(std::abs(urdf_safety.lower[index] -
                   a3_pingpong::kA3LegUrdfLower[index] * 0.90) <= 1.0e-12);
    CHECK(std::abs(urdf_safety.upper[index] -
                   a3_pingpong::kA3LegUrdfUpper[index] * 0.90) <= 1.0e-12);
    CHECK(urdf_safety.protected_joints[index] == (index == 0 || index == 6));
  }

  robot_io::RobotState state;
  state.q = Eigen::VectorXd::LinSpaced(31, -0.5, 0.5);
  robot_io::RobotCommand halt;
  a3_pingpong::BuildSafeHaltCommand(state, halt);
  CHECK(a3_pingpong::ValidateRobotCommand(halt, 31));
  CHECK((halt.q_des - state.q).norm() == 0.0);
  CHECK(halt.dq_des.norm() == 0.0);
  CHECK(halt.tau_ff.norm() == 0.0);
  CHECK(halt.kp.norm() == 0.0);
  CHECK(halt.kd.norm() == 0.0);

  // model_72500 locks waist pitch to zero, so its legacy positive-pitch virtual
  // wall is disabled by default and must leave the policy command untouched.
  a3_pingpong::ReceiveControllerOptions::WaistPitchSafety pitch_safety;
  CHECK(!pitch_safety.enabled);
  CHECK(pitch_safety.recovery_target_rad == 0.0);
  CHECK(a3_pingpong::ValidateWaistPitchSafety(pitch_safety));
  robot_io::RobotState pitch_state;
  pitch_state.q = Eigen::VectorXd::Zero(31);
  pitch_state.dq = Eigen::VectorXd::Zero(31);
  pitch_state.q[2] = 0.36;
  robot_io::RobotCommand pitch_command;
  a3_pingpong::BuildSafeHaltCommand(pitch_state, pitch_command);
  pitch_command.q_des[2] = 0.0;
  pitch_command.kp.setConstant(100.0);
  pitch_command.kd.setConstant(2.0);
  bool pitch_guard_active = false;
  CHECK(a3_pingpong::ApplyWaistPitchSafety(
      pitch_state, pitch_safety, pitch_guard_active, pitch_command));
  CHECK(!pitch_guard_active);
  CHECK(pitch_command.q_des[2] == 0.0);
  CHECK(pitch_command.kp[2] == 100.0);
  CHECK(pitch_command.kd[2] == 2.0);

  // The explicit opt-in path remains available for legacy diagnostics.
  pitch_safety.enabled = true;
  CHECK(a3_pingpong::ApplyWaistPitchSafety(
      pitch_state, pitch_safety, pitch_guard_active, pitch_command));
  CHECK(pitch_guard_active);
  CHECK(pitch_command.q_des[2] == 0.0);
  CHECK(pitch_command.kp[2] == 400.0);
  CHECK(pitch_command.kd[2] == 8.0);

  robot_io::RobotCommand invalid = halt;
  invalid.kp[0] = std::numeric_limits<double>::quiet_NaN();
  CHECK(!a3_pingpong::ValidateRobotCommand(invalid, 31));
  CHECK(!a3_pingpong::ValidateRobotCommand(halt, 29));

  a3_pingpong::PlannerInputMailbox mailbox("hope_table");
  a3_pingpong::RacketTargetInput target;
  target.frame_id = "hope_table";
  target.source_stamp_ns = SystemNowNs();
  target.task_id = 1;
  target.task_revision = 0;
  target.swing_side = 1;
  target.position_w = {0.1, -1.0, 0.3};
  target.velocity_w = {1.0, 0.0, 0.5};
  target.time_to_strike_s = 2.0;
  CHECK(mailbox.UpdateCommand(target, SystemNowNs(),
                              a3_pingpong::SteadyClock::now()) ==
        a3_pingpong::InputUpdateResult::kAccepted);

  a3_pingpong::BasePoseInput pose;
  pose.frame_id = "hope_table";
  pose.source_stamp_ns = SystemNowNs();
  pose.position_w = {-0.5, -0.7, 0.3};
  pose.quaternion_wxyz = {1.0, 0.0, 0.0, 0.0};
  CHECK(mailbox.UpdateBasePose(pose, a3_pingpong::SteadyClock::now()) ==
        a3_pingpong::InputUpdateResult::kAccepted);

  FakeBackend backend;
  backend.state.timestamp_ns = SystemNowNs();
  backend.state.q = Eigen::VectorXd::Zero(31);
  backend.state.dq = Eigen::VectorXd::Zero(31);
  backend.state.tau_est = Eigen::VectorXd::Zero(31);
  backend.state.sync_complete = true;
  backend.state.sync_aligned = true;

  auto policy = [](const a3_pingpong::PlannerInputSnapshot&,
                   const robot_io::RobotState& live_state,
                   robot_io::RobotCommand& command) {
    a3_pingpong::BuildSafeHaltCommand(live_state, command);
    command.kp.setConstant(10.0);
    command.kd.setConstant(1.0);
    return true;
  };
  a3_pingpong::ReceiveControllerOptions options;
  options.command_timeout_s = 1.0;
  options.base_pose_timeout_s = 1.0;
  options.max_state_age_ns = 1'000'000'000;
  options.publish_commands = false;
  // Keep the baseline controller tests independent from the deployment
  // envelope; production startup uses the guard by default.
  options.leg_damping_safety.enabled = false;
  a3_pingpong::ReceiveController controller(backend, mailbox, policy, options);
  std::atomic<int> observation_count{0};
  CHECK(controller.SetObservationProbe(
      [&observation_count](const a3_pingpong::PlannerInputSnapshot&,
                           const robot_io::RobotState&) {
        ++observation_count;
        return true;
      }));
  CHECK(controller.Start());
  CHECK(!controller.SetObservationProbe({}));
  std::this_thread::sleep_for(std::chrono::milliseconds(70));
  controller.Stop();
  CHECK(controller.tick_count() >= 2);
  CHECK(controller.last_result() == a3_pingpong::ReceiveTickResult::kDryRun);
  CHECK(controller.command_sent_count() == 0);
  CHECK(controller.safe_halt_count() == 0);
  CHECK(observation_count.load() >= 2);
  CHECK(backend.send_count.load() == 0);
  CHECK(!backend.started);

  // model_48000 supplies its own READY target when no ball/planner command is
  // active. The controller must still require a fresh pelvis pose, but it must
  // not reject the tick merely because RacketCommand is absent.
  a3_pingpong::PlannerInputMailbox ready_mailbox("hope_table");
  CHECK(ready_mailbox.UpdateBasePose(
            pose, a3_pingpong::SteadyClock::now()) ==
        a3_pingpong::InputUpdateResult::kAccepted);
  backend.state.timestamp_ns = SystemNowNs();
  options.require_fresh_command = false;
  a3_pingpong::ReceiveController ready_controller(
      backend, ready_mailbox, policy, options);
  std::atomic<int> ready_observations{0};
  CHECK(ready_controller.SetObservationProbe(
      [&ready_observations](const a3_pingpong::PlannerInputSnapshot& planner,
                            const robot_io::RobotState&) {
        if (!planner.base_pose || planner.command) return false;
        ++ready_observations;
        return true;
      }));
  CHECK(ready_controller.Start());
  std::this_thread::sleep_for(std::chrono::milliseconds(30));
  ready_controller.Stop();
  CHECK(ready_controller.last_result() ==
        a3_pingpong::ReceiveTickResult::kDryRun);
  CHECK(ready_observations.load() >= 1);

  backend.state.timestamp_ns = SystemNowNs();
  a3_pingpong::ReceiveController rejecting_controller(
      backend, mailbox, a3_pingpong::ReceivePolicyFn{}, options);
  CHECK(rejecting_controller.SetObservationProbe(
      [](const a3_pingpong::PlannerInputSnapshot&,
         const robot_io::RobotState&) { return false; }));
  CHECK(rejecting_controller.Start());
  std::this_thread::sleep_for(std::chrono::milliseconds(30));
  rejecting_controller.Stop();
  CHECK(rejecting_controller.last_result() ==
        a3_pingpong::ReceiveTickResult::kObservationRejected);
  CHECK(backend.send_count.load() == 0);

  // The enabled path must call the official RobotIO SendCommand seam with the
  // validated policy packet, while stale state must use the safe-halt packet.
  FakeBackend publishing_backend;
  publishing_backend.state = backend.state;
  publishing_backend.state.timestamp_ns = SystemNowNs();
  auto publishing_options = options;
  publishing_options.publish_commands = true;
  a3_pingpong::ReceiveController publishing_controller(
      publishing_backend, mailbox, policy, publishing_options);
  CHECK(publishing_controller.Start());
  std::this_thread::sleep_for(std::chrono::milliseconds(30));
  publishing_controller.Stop();
  CHECK(publishing_controller.last_result() ==
        a3_pingpong::ReceiveTickResult::kCommandSent);
  CHECK(publishing_controller.command_sent_count() >= 1);
  CHECK(publishing_backend.send_count.load() >= 1);
  CHECK(publishing_backend.last_command.kp[0] == 10.0);
  CHECK(publishing_backend.last_command.kd[0] == 1.0);

  // A leg target outside the configured soft envelope must never reach the
  // backend. The controller latches the fault and sends leg damping instead.
  FakeBackend target_limit_backend;
  target_limit_backend.state = backend.state;
  target_limit_backend.state.timestamp_ns = SystemNowNs();
  auto target_limit_options = publishing_options;
  target_limit_options.leg_damping_safety.enabled = true;
  target_limit_options.leg_damping_safety.lower.fill(-1.0);
  target_limit_options.leg_damping_safety.upper.fill(1.0);
  target_limit_options.leg_damping_safety.damping_kd = 2.5;
  auto violating_policy = [](const a3_pingpong::PlannerInputSnapshot&,
                             const robot_io::RobotState& live_state,
                             robot_io::RobotCommand& command) {
    a3_pingpong::BuildSafeHaltCommand(live_state, command);
    command.q_des[19] = 1.25;
    command.kp.setConstant(10.0);
    command.kd.setConstant(1.0);
    return true;
  };
  a3_pingpong::ReceiveController target_limit_controller(
      target_limit_backend, mailbox, violating_policy, target_limit_options);
  CHECK(target_limit_controller.Start());
  std::this_thread::sleep_for(std::chrono::milliseconds(30));
  target_limit_controller.Stop();
  CHECK(target_limit_controller.leg_limit_damping_active());
  CHECK(target_limit_controller.leg_limit_joint_index() == 19);
  CHECK(target_limit_controller.last_result() ==
        a3_pingpong::ReceiveTickResult::kLegLimitDamping);
  CHECK(target_limit_controller.leg_limit_damping_count() >= 1);
  CHECK(target_limit_backend.last_command.q_des[19] == 0.0);
  CHECK(target_limit_backend.last_command.kp.norm() == 0.0);
  CHECK(target_limit_backend.last_command.kd[19] == 2.5);
  CHECK(target_limit_backend.last_command.kd[0] == 0.0);

  // Non-hip-pitch leg joints are outside the current protection scope. An
  // otherwise valid command with an out-of-envelope knee target must pass
  // through instead of entering the hip-pitch damping latch.
  FakeBackend unprotected_target_backend;
  unprotected_target_backend.state = backend.state;
  unprotected_target_backend.state.timestamp_ns = SystemNowNs();
  auto unprotected_target_policy =
      [](const a3_pingpong::PlannerInputSnapshot&,
         const robot_io::RobotState& live_state,
         robot_io::RobotCommand& command) {
        a3_pingpong::BuildSafeHaltCommand(live_state, command);
        command.q_des[20] = 1.25;
        command.kp.setConstant(10.0);
        command.kd.setConstant(1.0);
        return true;
      };
  a3_pingpong::ReceiveController unprotected_target_controller(
      unprotected_target_backend, mailbox, unprotected_target_policy,
      target_limit_options);
  CHECK(unprotected_target_controller.Start());
  std::this_thread::sleep_for(std::chrono::milliseconds(30));
  unprotected_target_controller.Stop();
  CHECK(!unprotected_target_controller.leg_limit_damping_active());
  CHECK(unprotected_target_controller.last_result() ==
        a3_pingpong::ReceiveTickResult::kCommandSent);
  CHECK(unprotected_target_backend.last_command.q_des[20] == 1.25);

  // The measured position is checked before policy/planner evaluation too.
  FakeBackend measured_limit_backend;
  measured_limit_backend.state = backend.state;
  measured_limit_backend.state.timestamp_ns = SystemNowNs();
  measured_limit_backend.state.q[19] = 1.25;
  auto measured_policy = [](const a3_pingpong::PlannerInputSnapshot&,
                            const robot_io::RobotState& live_state,
                            robot_io::RobotCommand& command) {
    a3_pingpong::BuildSafeHaltCommand(live_state, command);
    command.kp.setConstant(10.0);
    command.kd.setConstant(1.0);
    return true;
  };
  a3_pingpong::ReceiveController measured_limit_controller(
      measured_limit_backend, mailbox, measured_policy, target_limit_options);
  CHECK(measured_limit_controller.Start());
  std::this_thread::sleep_for(std::chrono::milliseconds(30));
  measured_limit_controller.Stop();
  CHECK(measured_limit_controller.leg_limit_damping_active());
  CHECK(measured_limit_controller.leg_limit_joint_index() == 19);
  CHECK(measured_limit_backend.last_command.q_des[19] == 1.25);
  CHECK(measured_limit_backend.last_command.kp.norm() == 0.0);

  FakeBackend stale_backend;
  stale_backend.state = backend.state;
  stale_backend.state.timestamp_ns = 1;
  a3_pingpong::ReceiveController stale_controller(
      stale_backend, mailbox, policy, publishing_options);
  CHECK(stale_controller.Start());
  std::this_thread::sleep_for(std::chrono::milliseconds(30));
  stale_controller.Stop();
  CHECK(stale_controller.last_result() ==
        a3_pingpong::ReceiveTickResult::kStateStale);
  CHECK(stale_controller.safe_halt_count() >= 1);
  CHECK(stale_backend.send_count.load() >= 1);
  CHECK(stale_backend.last_command.kp.norm() == 0.0);
  CHECK(stale_backend.last_command.kd.norm() == 0.0);
  return 0;
}
