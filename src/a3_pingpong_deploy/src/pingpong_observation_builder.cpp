#include "a3_pingpong/pingpong_observation_builder.hpp"

#include <Eigen/Core>

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>

namespace a3_pingpong {
namespace {

constexpr std::array<const char*, kPingpongActionDim> kJointNames = {
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "head_yaw_joint",
    "head_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
};

void SetReason(std::string* output, std::string value) {
  if (output) *output = std::move(value);
}

template <typename Range>
bool AllFinite(const Range& values) {
  for (const auto value : values) {
    if (!std::isfinite(static_cast<double>(value))) return false;
  }
  return true;
}

bool NormalizeQuaternion(std::array<double, 4>& quaternion) {
  if (!AllFinite(quaternion)) return false;
  double squared_norm = 0.0;
  for (const double value : quaternion) squared_norm += value * value;
  const double norm = std::sqrt(squared_norm);
  if (!std::isfinite(norm) || norm < 1.0e-9) return false;
  for (double& value : quaternion) value /= norm;
  return true;
}

std::array<double, 3> Rotate(const std::array<double, 4>& quaternion,
                             const std::array<double, 3>& vector,
                             bool inverse) {
  const double w = quaternion[0];
  const std::array<double, 3> xyz = {
      quaternion[1], quaternion[2], quaternion[3]};
  const std::array<double, 3> cross = {
      xyz[1] * vector[2] - xyz[2] * vector[1],
      xyz[2] * vector[0] - xyz[0] * vector[2],
      xyz[0] * vector[1] - xyz[1] * vector[0],
  };
  const double dot = xyz[0] * vector[0] + xyz[1] * vector[1] +
                     xyz[2] * vector[2];
  std::array<double, 3> output{};
  for (std::size_t index = 0; index < output.size(); ++index) {
    const double a = vector[index] * (2.0 * w * w - 1.0);
    const double b = cross[index] * (2.0 * w);
    const double c = xyz[index] * (2.0 * dot);
    output[index] = inverse ? a - b + c : a + b + c;
  }
  return output;
}

}  // namespace

PingpongObservationConfig Model21500ObservationConfig() {
  PingpongObservationConfig config;
  config.default_q = {
      0.0,   0.0,  0.0, 0.0, 0.0,  0.20, 0.15, 0.0,
      0.30,  0.0,  0.0, 0.0, 0.20, -0.15, 0.0, 0.30,
      0.0,   0.0,  0.0, -0.15, 0.0, 0.0, 0.30, -0.15,
      0.0,  -0.15, 0.0, 0.0, 0.30, -0.15, 0.0,
  };
  return config;
}

bool ValidatePingpongJointLayout(const robot_io::JointLayout& layout,
                                 std::string* reason) {
  if (layout.names.size() != kJointNames.size()) {
    SetReason(reason, "RobotIO layout must contain exactly 31 joints");
    return false;
  }
  for (std::size_t index = 0; index < kJointNames.size(); ++index) {
    if (layout.names[index] != kJointNames[index]) {
      SetReason(reason, "RobotIO joint order mismatch at index " +
                            std::to_string(index) + ": expected " +
                            kJointNames[index] + ", got " +
                            layout.names[index]);
      return false;
    }
  }
  SetReason(reason, "valid");
  return true;
}

PingpongObservationBuilder::PingpongObservationBuilder(
    PingpongObservationConfig config)
    : config_(std::move(config)) {}

bool PingpongObservationBuilder::Build(
    const robot_io::RobotState& state,
    const PlannerInputSnapshot& planner,
    const PingpongAction& last_action,
    const std::array<double, 2>& fixed_station_xy,
    PingpongObservation& output,
    std::string* reason) const {
  if (!planner.command || !planner.base_pose) {
    SetReason(reason, "planner command and base pose are required");
    return false;
  }
  if (state.q.size() != static_cast<Eigen::Index>(kPingpongActionDim) ||
      state.dq.size() != static_cast<Eigen::Index>(kPingpongActionDim) ||
      !state.q.array().isFinite().all() ||
      !state.dq.array().isFinite().all()) {
    SetReason(reason, "RobotState q/dq must contain 31 finite values");
    return false;
  }
  if (!state.imu_gyro.array().isFinite().all() ||
      !AllFinite(last_action) || !AllFinite(fixed_station_xy) ||
      !AllFinite(planner.base_pose->position_w) ||
      !AllFinite(planner.command->position_w) ||
      !AllFinite(planner.command->velocity_w) ||
      !std::isfinite(planner.command->time_to_strike_s) ||
      (planner.command->swing_side != 1 &&
       planner.command->swing_side != -1)) {
    SetReason(reason, "observation source contains an invalid value");
    return false;
  }

  std::array<double, 4> gravity_quaternion = {
      state.imu_quat_wxyz[0], state.imu_quat_wxyz[1],
      state.imu_quat_wxyz[2], state.imu_quat_wxyz[3]};
  std::array<double, 4> heading_quaternion =
      planner.base_pose->quaternion_wxyz;
  if (!NormalizeQuaternion(gravity_quaternion) ||
      !NormalizeQuaternion(heading_quaternion)) {
    SetReason(reason, "gravity or heading quaternion is invalid");
    return false;
  }

  const auto gravity_body = Rotate(
      gravity_quaternion, std::array<double, 3>{0.0, 0.0, -1.0}, true);
  const auto forward_world = Rotate(
      heading_quaternion, std::array<double, 3>{1.0, 0.0, 0.0}, false);
  const double forward_norm =
      std::hypot(forward_world[0], forward_world[1]) + 1.0e-6;

  for (std::size_t index = 0; index < 3; ++index) {
    output[index] = static_cast<float>(state.imu_gyro[index]);
  }
  for (std::size_t index = 0; index < kPingpongActionDim; ++index) {
    output[3 + index] =
        static_cast<float>(state.q[static_cast<Eigen::Index>(index)] -
                           config_.default_q[index]);
    output[34 + index] =
        static_cast<float>(state.dq[static_cast<Eigen::Index>(index)]);
    output[65 + index] = last_action[index];
  }
  for (std::size_t index = 0; index < 3; ++index) {
    output[96 + index] = static_cast<float>(gravity_body[index]);
  }
  output[99] = static_cast<float>(forward_world[0] / forward_norm);
  output[100] = static_cast<float>(forward_world[1] / forward_norm);
  output[101] = static_cast<float>(fixed_station_xy[0] -
                                   planner.base_pose->position_w[0]);
  output[102] = static_cast<float>(fixed_station_xy[1] -
                                   planner.base_pose->position_w[1]);
  for (std::size_t index = 0; index < 3; ++index) {
    output[103 + index] = static_cast<float>(
        planner.command->position_w[index] -
        planner.base_pose->position_w[index]);
    output[106 + index] =
        static_cast<float>(planner.command->velocity_w[index]);
  }
  output[109] = static_cast<float>(planner.command->time_to_strike_s);
  output[110] = static_cast<float>(planner.command->swing_side);

  if (!AllFinite(output)) {
    SetReason(reason, "constructed observation contains an invalid value");
    return false;
  }
  SetReason(reason, "valid");
  return true;
}

}  // namespace a3_pingpong
