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

  // model_21500 supplies its own READY target when no ball/planner command is
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
