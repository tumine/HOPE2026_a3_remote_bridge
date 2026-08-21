#include "a3_pingpong/pingpong_action_adapter.hpp"
#include "a3_pingpong/a3_leg_limits.hpp"
#include "a3_pingpong/receive_controller.hpp"

#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>

#define CHECK(condition)              \
  do {                                \
    if (!(condition)) return __LINE__; \
  } while (false)

#ifndef A3_ACTION_GOLDEN_PATH
#error "A3_ACTION_GOLDEN_PATH must be defined"
#endif

#ifndef A3_Q_DES_GOLDEN_PATH
#error "A3_Q_DES_GOLDEN_PATH must be defined"
#endif

int main() {
  const auto adapter_config = a3_pingpong::Model72500ActionAdapterConfig();
  CHECK(adapter_config.upper[2] == 0.418879);
  CHECK(adapter_config.locked[1]);
  CHECK(adapter_config.locked[2]);
  for (std::size_t index = 0; index < a3_pingpong::kA3LegDof; ++index) {
    const auto flat = static_cast<Eigen::Index>(
        a3_pingpong::kA3LegCommandStart + index);
    CHECK(std::abs(adapter_config.lower[flat] -
                   a3_pingpong::kA3LegUrdfLower[index]) <= 1.0e-12);
    CHECK(std::abs(adapter_config.upper[flat] -
                   a3_pingpong::kA3LegUrdfUpper[index]) <= 1.0e-12);
  }

  a3_pingpong::PingpongAction raw_action{};
  std::ifstream action_fixture(A3_ACTION_GOLDEN_PATH);
  CHECK(action_fixture.good());
  for (float& value : raw_action) {
    double parsed = 0.0;
    CHECK(static_cast<bool>(action_fixture >> parsed));
    value = static_cast<float>(parsed);
  }

  a3_pingpong::PingpongActionAdapter adapter(
      a3_pingpong::Model72500ActionAdapterConfig());
  a3_pingpong::PingpongAction applied{};
  robot_io::RobotCommand command;
  a3_pingpong::PingpongActionDiagnostics diagnostics;
  std::string reason;
  CHECK(adapter.BuildZeroGainDryRunCommand(
      raw_action, applied, command, &diagnostics, &reason));
  CHECK(diagnostics.raw_clip_count == 0);
  CHECK(diagnostics.position_clip_count == 2);
  CHECK(applied[1] == 0.0F);
  CHECK(applied[2] == 0.0F);
  CHECK(applied[3] == 0.0F);
  CHECK(applied[4] == 0.0F);

  std::ifstream q_des_fixture(A3_Q_DES_GOLDEN_PATH);
  CHECK(q_des_fixture.good());
  CHECK(command.q_des.size() == 31);
  for (Eigen::Index index = 0; index < command.q_des.size(); ++index) {
    double expected = 0.0;
    CHECK(static_cast<bool>(q_des_fixture >> expected));
    if (std::abs(command.q_des[index] - expected) > 1.0e-7) {
      std::cerr << "q_des mismatch index=" << index
                << " actual=" << command.q_des[index]
                << " expected=" << expected << '\n';
      return __LINE__;
    }
  }
  CHECK(command.dq_des.norm() == 0.0);
  CHECK(command.tau_ff.norm() == 0.0);
  CHECK(command.kp.norm() == 0.0);
  CHECK(command.kd.norm() == 0.0);
  CHECK(a3_pingpong::ValidateRobotCommand(command, 31));

  // Even deliberately non-zero raw waist roll/pitch values are never executed
  // or fed back to the next observation.
  raw_action[1] = 1.25F;
  raw_action[2] = -2.5F;
  CHECK(adapter.BuildPolicyCommand(
      raw_action, applied, command, &diagnostics, &reason));
  CHECK(applied[1] == 0.0F);
  CHECK(applied[2] == 0.0F);
  CHECK(command.q_des[1] == 0.0);
  CHECK(command.q_des[2] == 0.0);
  CHECK(command.kp[0] == 85.0);
  CHECK(command.kp[1] == 500.0);
  CHECK(command.kp[2] == 500.0);

  CHECK(adapter.BuildPolicyCommand(
      raw_action, applied, command, &diagnostics, &reason));
  CHECK(command.kp[0] == 85.0);
  CHECK(command.kd[0] == 3.0);
  CHECK(command.kp[1] == 500.0);
  CHECK(command.kd[1] == 2.0);
  CHECK(command.kp[2] == 500.0);
  CHECK(command.kd[2] == 2.0);
  CHECK(command.kp[3] == 40.0);
  CHECK(command.kd[4] == 2.0);
  CHECK(command.kp[20] == 120.0);
  CHECK(command.kp[22] == 250.0);
  CHECK(command.kd[22] == 8.0);
  CHECK(command.kp[28] == 250.0);
  CHECK(command.dq_des.norm() == 0.0);
  CHECK(command.tau_ff.norm() == 0.0);
  CHECK(a3_pingpong::ValidateRobotCommand(command, 31));

  std::array<double, a3_pingpong::kPingpongActionDim> start_q{};
  start_q.fill(1.0);
  const auto target_q =
      a3_pingpong::Model72500ActionAdapterConfig().default_q;
  bool stand_ready = true;
  CHECK(a3_pingpong::BuildPdStandCommand(
      start_q, target_q, 0, 150, command, &stand_ready, &reason));
  CHECK(!stand_ready);
  CHECK(command.q_des[0] == 1.0);
  CHECK(command.q_des[3] == 0.0);
  CHECK(command.kp[0] == 400.0);
  CHECK(command.kp[3] == 40.0);
  CHECK(command.kp[22] == 2000.0);
  CHECK(command.kd[22] == 8.0);
  CHECK(a3_pingpong::BuildPdStandCommand(
      start_q, target_q, 75, 150, command, &stand_ready, &reason));
  CHECK(!stand_ready);
  CHECK(std::abs(command.q_des[0] - 0.5) <= 1.0e-12);
  CHECK(std::abs(command.q_des[5] - 0.6) <= 1.0e-12);
  CHECK(a3_pingpong::BuildPdStandCommand(
      start_q, target_q, 150, 150, command, &stand_ready, &reason));
  CHECK(stand_ready);
  for (std::size_t index = 0; index < target_q.size(); ++index) {
    CHECK(std::abs(command.q_des[static_cast<Eigen::Index>(index)] -
                   target_q[index]) <= 1.0e-12);
  }

  raw_action[0] = 150.0F;
  CHECK(adapter.BuildZeroGainDryRunCommand(
      raw_action, applied, command, &diagnostics, &reason));
  CHECK(diagnostics.raw_clip_count == 1);
  CHECK(applied[0] == 100.0F);

  raw_action[0] = std::numeric_limits<float>::quiet_NaN();
  CHECK(!adapter.BuildPolicyCommand(
      raw_action, applied, command, &diagnostics, &reason));
  return 0;
}
