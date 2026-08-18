#include "a3_pingpong/upper_body_serve.hpp"

#include <Eigen/Core>

#include <array>
#include <cmath>
#include <optional>

#define CHECK(condition)                \
  do {                                  \
    if (!(condition)) return __LINE__;  \
  } while (false)

namespace {

robot_io::RobotState State31() {
  robot_io::RobotState state;
  state.q = Eigen::VectorXd(31);
  state.dq = Eigen::VectorXd::Zero(31);
  state.tau_est = Eigen::VectorXd::Zero(31);
  for (Eigen::Index index = 0; index < state.q.size(); ++index) {
    state.q[index] = 0.01 * static_cast<double>(index);
  }
  return state;
}

bool Near(double lhs, double rhs) {
  return std::abs(lhs - rhs) < 1.0e-12;
}

}  // namespace

int main() {
  using namespace a3_pingpong;

  auto state = State31();
  UpperBodyServeConfig config;
  config.prepare_duration_s = 0.04;
  config.windup_duration_s = 0.04;
  config.swing_duration_s = 0.04;
  config.settle_duration_s = 0.04;
  config.return_duration_s = 0.04;
  UpperBodyServeTrajectory trajectory(config);

  UpperBodyServeTarget upper{};
  UpperBodyServeDiagnostics diagnostics;
  std::string reason;
  CHECK(trajectory.Step(state, 0.02, upper, &diagnostics, &reason));
  CHECK(diagnostics.phase == UpperBodyServePhase::kPrepare);
  for (std::size_t index = 0; index < upper.size(); ++index) {
    CHECK(Near(upper[index], state.q[static_cast<Eigen::Index>(5 + index)]));
  }

  int release_count = 0;
  bool saw_windup = false;
  bool saw_swing = false;
  bool saw_settle = false;
  bool saw_return = false;
  for (int tick = 0; tick < 30 && !diagnostics.complete; ++tick) {
    CHECK(trajectory.Step(state, 0.02, upper, &diagnostics, &reason));
    saw_windup |= diagnostics.phase == UpperBodyServePhase::kWindup;
    saw_swing |= diagnostics.phase == UpperBodyServePhase::kSwing;
    saw_settle |= diagnostics.phase == UpperBodyServePhase::kSettle;
    saw_return |= diagnostics.phase == UpperBodyServePhase::kReturn;
    if (diagnostics.release_requested) ++release_count;
  }
  CHECK(diagnostics.complete);
  CHECK(saw_windup && saw_swing && saw_settle && saw_return);
  CHECK(release_count == 1);
  for (std::size_t index = 0; index < upper.size(); ++index) {
    CHECK(Near(upper[index], config.home_upper[index]));
  }

  FullBodyServeComposer composer;
  robot_io::RobotCommand command;
  CHECK(composer.Build(state, upper, std::nullopt, command, &reason));
  CHECK(command.q_des.size() == 31);
  CHECK(Near(command.q_des[3], state.q[3]));
  CHECK(Near(command.q_des[4], state.q[4]));
  CHECK(Near(command.q_des[5], upper[0]));
  CHECK(Near(command.q_des[18], upper[13]));
  CHECK(Near(command.q_des[19], state.q[19]));
  CHECK(Near(command.q_des[30], state.q[30]));

  LowerBodyServeTarget lower{};
  for (std::size_t index = 0; index < lower.size(); ++index) {
    lower[index] = -0.02 * static_cast<double>(index + 1);
  }
  CHECK(composer.Build(state, upper, lower, command, &reason));
  CHECK(Near(command.q_des[0], lower[0]));
  CHECK(Near(command.q_des[2], lower[2]));
  CHECK(Near(command.q_des[19], lower[3]));
  CHECK(Near(command.q_des[30], lower[14]));
  CHECK(Near(command.q_des[12], upper[7]));

  state.q[0] = std::nan("");
  CHECK(!composer.Build(state, upper, std::nullopt, command, &reason));
  return 0;
}
