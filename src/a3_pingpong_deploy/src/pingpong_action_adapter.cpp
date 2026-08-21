#include "a3_pingpong/pingpong_action_adapter.hpp"

#include "a3_pingpong/a3_leg_limits.hpp"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <utility>

namespace a3_pingpong {
namespace {

constexpr std::size_t kHeadYawIndex = 3;
constexpr std::size_t kHeadPitchIndex = 4;
constexpr std::size_t kWaistRollIndex = 1;
constexpr std::size_t kWaistPitchIndex = 2;

void SetReason(std::string* output, std::string value) {
  if (output) *output = std::move(value);
}

}  // namespace

PingpongActionAdapterConfig Model50000ActionAdapterConfig() {
  PingpongActionAdapterConfig config;
  config.default_q = Model50000ObservationConfig().default_q;
  config.lower = {
      -2.61799,  -0.349066, -0.488692, -1.0472,   -0.436332, -2.87979,
      -0.0872665, -2.79253, -0.959931, -2.79253, -1.62316,  -1.62316,
      -2.87979,  -2.61799,  -2.79253,  -0.959931, -2.79253,  -1.62316,
      -1.62316,  -2.51327,  -0.523599, -2.72271,  -0.122173, -0.907571,
      -0.349066, -2.51327,  -1.6057,   -2.72271,  -0.122173, -0.907571,
      -0.349066,
  };
  config.upper = {
      2.61799,  0.349066, 0.418879, 1.0472,   0.261799, 2.87979,
      2.61799,  2.79253,  1.74533,  2.79253,  1.62316,  1.62316,
      2.87979,  0.0872665, 2.79253, 1.74533,  2.79253,  1.62316,
      1.62316,  2.93215,  1.6057,   2.72271,  2.49582,  0.523599,
      0.349066, 2.93215,  0.523599, 2.72271,  2.49582,  0.523599,
      0.349066,
  };
  for (std::size_t index = 0; index < kA3LegDof; ++index) {
    config.lower[kA3LegCommandStart + index] = kA3LegUrdfLower[index];
    config.upper[kA3LegCommandStart + index] = kA3LegUrdfUpper[index];
  }
  return config;
}

PingpongActionAdapterConfig Model72500ActionAdapterConfig() {
  auto config = Model50000ActionAdapterConfig();
  config.locked[kWaistRollIndex] = true;
  config.locked[kWaistPitchIndex] = true;
  return config;
}

PingpongActionAdapterConfig Model48000ActionAdapterConfig() {
  return Model50000ActionAdapterConfig();
}

PingpongActionAdapterConfig Model41500ActionAdapterConfig() {
  return Model48000ActionAdapterConfig();
}

PingpongActionAdapterConfig Model21500ActionAdapterConfig() {
  return Model48000ActionAdapterConfig();
}

PingpongCommandGains Model50000PolicyGains() {
  PingpongCommandGains gains;
  // Real-robot deployment gains in canonical A3 SDK order:
  // waist, head, left arm, right arm, left leg, right leg.
  // model_53000 simulation gains with explicitly requested real-robot waist
  // stiffness overrides (yaw/roll/pitch = 150/50/100). Other joints stay exact.
  gains.kp = {
      150.0, 50.0, 100.0, 40.0, 40.0,
      40.0, 40.0, 30.0, 30.0, 30.0, 20.0, 20.0,
      40.0, 40.0, 30.0, 30.0, 30.0, 20.0, 20.0,
      80.0, 120.0, 80.0, 250.0, 50.0, 50.0,
      80.0, 120.0, 80.0, 250.0, 50.0, 50.0,
  };
  gains.kd = {
      3.0, 2.0, 2.0, 2.0, 2.0,
      3.0, 3.0, 2.0, 2.0, 2.0, 2.0, 2.0,
      3.0, 3.0, 2.0, 2.0, 2.0, 2.0, 2.0,
      3.0, 4.0, 3.0, 8.0, 2.0, 2.0,
      3.0, 4.0, 3.0, 8.0, 2.0, 2.0,
  };
  return gains;
}

PingpongCommandGains Model72500PolicyGains() {
  // model_72500 freezes waist roll/pitch at q_des=0 and was trained with the
  // nominal waist gains below. Keep every non-waist real-robot gain unchanged
  // while aligning the three waist axes with the bundle contract.
  auto gains = Model50000PolicyGains();
  gains.kp[0] = 85.0;   // waist_yaw_joint
  gains.kp[1] = 500.0;  // waist_roll_joint, locked q_des=0
  gains.kp[2] = 500.0;  // waist_pitch_joint, locked q_des=0
  gains.kd[0] = 3.0;
  gains.kd[1] = 2.0;
  gains.kd[2] = 2.0;
  return gains;
}

PingpongCommandGains Model48000PolicyGains() {
  return Model50000PolicyGains();
}

PingpongCommandGains Model41500PolicyGains() {
  return Model48000PolicyGains();
}

PingpongCommandGains Model21500PolicyGains() {
  return Model48000PolicyGains();
}

PingpongCommandGains A3PdStandGains() {
  PingpongCommandGains gains;
  // Production PD_STAND gains from the official A3 deployment reference.
  // Head is held at zero by the same 40/2 gains used by ExpandToBackend().
  gains.kp = {
      400.0, 500.0, 500.0, 40.0, 40.0,
      200.0, 200.0, 100.0, 200.0, 100.0, 50.0, 50.0,
      200.0, 200.0, 100.0, 200.0, 100.0, 50.0, 50.0,
      1500.0, 400.0, 300.0, 2000.0, 500.0, 500.0,
      1500.0, 400.0, 300.0, 2000.0, 500.0, 500.0,
  };
  gains.kd = {
      4.0, 4.0, 4.0, 2.0, 2.0,
      2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0,
      2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0,
      8.0, 7.0, 7.0, 8.0, 5.0, 5.0,
      8.0, 7.0, 7.0, 8.0, 5.0, 5.0,
  };
  return gains;
}

bool BuildPositionCommand(
    const std::array<double, kPingpongActionDim>& q_des,
    const PingpongCommandGains& gains,
    robot_io::RobotCommand& command,
    std::string* reason) {
  const auto dof = static_cast<Eigen::Index>(kPingpongActionDim);
  command.q_des = Eigen::VectorXd(dof);
  command.dq_des = Eigen::VectorXd::Zero(dof);
  command.tau_ff = Eigen::VectorXd::Zero(dof);
  command.kp = Eigen::VectorXd(dof);
  command.kd = Eigen::VectorXd(dof);
  for (std::size_t index = 0; index < q_des.size(); ++index) {
    if (!std::isfinite(q_des[index]) || !std::isfinite(gains.kp[index]) ||
        !std::isfinite(gains.kd[index]) || gains.kp[index] < 0.0 ||
        gains.kd[index] < 0.0) {
      SetReason(reason, "position command contains an invalid value");
      return false;
    }
    const auto eigen_index = static_cast<Eigen::Index>(index);
    command.q_des[eigen_index] = q_des[index];
    command.kp[eigen_index] = gains.kp[index];
    command.kd[eigen_index] = gains.kd[index];
  }
  SetReason(reason, "valid position command");
  return true;
}

bool BuildPdStandCommand(
    const std::array<double, kPingpongActionDim>& start_q,
    const std::array<double, kPingpongActionDim>& target_q,
    std::uint64_t elapsed_ticks,
    std::uint64_t ramp_ticks,
    robot_io::RobotCommand& command,
    bool* ready,
    std::string* reason) {
  const double alpha = ramp_ticks == 0
                           ? 1.0
                           : std::min(1.0, static_cast<double>(elapsed_ticks) /
                                              static_cast<double>(ramp_ticks));
  std::array<double, kPingpongActionDim> q_des{};
  for (std::size_t index = 0; index < q_des.size(); ++index) {
    q_des[index] = start_q[index] + alpha * (target_q[index] - start_q[index]);
  }
  // The official 29->31 adapter owns the neck rather than the policy. Both
  // model_50000 neck targets are zero, so make that contract explicit here.
  q_des[kHeadYawIndex] = 0.0;
  q_des[kHeadPitchIndex] = 0.0;
  if (ready) *ready = alpha >= 1.0;
  return BuildPositionCommand(q_des, A3PdStandGains(), command, reason);
}

PingpongActionAdapter::PingpongActionAdapter(
    PingpongActionAdapterConfig config)
    : config_(std::move(config)) {}

bool PingpongActionAdapter::Decode(
    const PingpongAction& raw_action,
    PingpongAction& applied_action,
    std::array<double, kPingpongActionDim>& q_des,
    PingpongActionDiagnostics* diagnostics,
    std::string* reason) const {
  if (!std::isfinite(config_.action_scale) || config_.action_scale <= 0.0 ||
      !std::isfinite(config_.action_clip_lower) ||
      !std::isfinite(config_.action_clip_upper) ||
      config_.action_clip_lower > config_.action_clip_upper) {
    SetReason(reason, "action adapter scale or clip is invalid");
    return false;
  }
  PingpongActionDiagnostics local;
  for (std::size_t index = 0; index < raw_action.size(); ++index) {
    const double raw = raw_action[index];
    if (!std::isfinite(raw) || !std::isfinite(config_.default_q[index]) ||
        !std::isfinite(config_.lower[index]) ||
        !std::isfinite(config_.upper[index]) ||
        config_.lower[index] > config_.upper[index]) {
      SetReason(reason, "action adapter contains an invalid value");
      return false;
    }
    local.raw_max_abs = std::max(local.raw_max_abs, std::abs(raw));
    const double clipped = std::clamp(
        raw, config_.action_clip_lower, config_.action_clip_upper);
    if (clipped != raw) ++local.raw_clip_count;
    applied_action[index] = static_cast<float>(clipped);
  }

  applied_action[kHeadYawIndex] = 0.0F;
  applied_action[kHeadPitchIndex] = 0.0F;
  for (std::size_t index = 0; index < applied_action.size(); ++index) {
    if (config_.locked[index]) applied_action[index] = 0.0F;
  }
  for (std::size_t index = 0; index < raw_action.size(); ++index) {
    const double applied = applied_action[index];
    local.applied_max_abs =
        std::max(local.applied_max_abs, std::abs(applied));
    const double unclipped =
        config_.default_q[index] + applied * config_.action_scale;
    q_des[index] =
        std::clamp(unclipped, config_.lower[index], config_.upper[index]);
    if (q_des[index] != unclipped) ++local.position_clip_count;
    local.q_des_max_abs =
        std::max(local.q_des_max_abs, std::abs(q_des[index]));
  }
  q_des[kHeadYawIndex] = config_.default_q[kHeadYawIndex];
  q_des[kHeadPitchIndex] = config_.default_q[kHeadPitchIndex];
  for (std::size_t index = 0; index < q_des.size(); ++index) {
    if (config_.locked[index]) q_des[index] = config_.default_q[index];
  }

  if (diagnostics) *diagnostics = local;
  SetReason(reason, "valid");
  return true;
}

bool PingpongActionAdapter::BuildZeroGainDryRunCommand(
    const PingpongAction& raw_action,
    PingpongAction& applied_action,
    robot_io::RobotCommand& command,
    PingpongActionDiagnostics* diagnostics,
    std::string* reason) const {
  std::array<double, kPingpongActionDim> q_des{};
  if (!Decode(raw_action, applied_action, q_des, diagnostics, reason)) {
    return false;
  }

  const auto dof = static_cast<Eigen::Index>(kPingpongActionDim);
  command.q_des = Eigen::VectorXd(dof);
  command.dq_des = Eigen::VectorXd::Zero(dof);
  command.tau_ff = Eigen::VectorXd::Zero(dof);
  command.kp = Eigen::VectorXd::Zero(dof);
  command.kd = Eigen::VectorXd::Zero(dof);
  for (std::size_t index = 0; index < q_des.size(); ++index) {
    command.q_des[static_cast<Eigen::Index>(index)] = q_des[index];
  }
  SetReason(reason, "valid zero-gain dry-run command");
  return true;
}

bool PingpongActionAdapter::BuildPolicyCommand(
    const PingpongAction& raw_action,
    PingpongAction& applied_action,
    robot_io::RobotCommand& command,
    PingpongActionDiagnostics* diagnostics,
    std::string* reason) const {
  std::array<double, kPingpongActionDim> q_des{};
  if (!Decode(raw_action, applied_action, q_des, diagnostics, reason)) {
    return false;
  }
  if (!BuildPositionCommand(q_des, Model72500PolicyGains(), command, reason)) {
    return false;
  }
  SetReason(reason, "valid model_72500 policy command");
  return true;
}

}  // namespace a3_pingpong
