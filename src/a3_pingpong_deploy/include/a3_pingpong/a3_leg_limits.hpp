#pragma once

#include <array>
#include <cstddef>

namespace a3_pingpong {

// Canonical A3 31-DOF command slots [19..30]. Values are copied from the
// supplied A3T2.5 URDF revolute-joint limits, in radians.
inline constexpr int kA3LegCommandStart = 19;
inline constexpr std::size_t kA3LegDof = 12;

inline constexpr std::array<const char*, kA3LegDof> kA3LegJointNames = {
    "left_hip_pitch_joint",   "left_hip_roll_joint",
    "left_hip_yaw_joint",     "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint",  "right_hip_roll_joint",
    "right_hip_yaw_joint",    "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint"};

inline constexpr std::array<double, kA3LegDof> kA3LegUrdfLower = {
    -2.5132741228718345, -0.5235987755982988, -2.722713633111154,
    -0.12217304763960307, -0.9075712110370514, -0.3490658503988659,
    -2.5132741228718345, -1.6057029118347832, -2.722713633111154,
    -0.12217304763960307, -0.9075712110370514, -0.3490658503988659};

inline constexpr std::array<double, kA3LegDof> kA3LegUrdfUpper = {
    2.9321531433504737, 1.6057029118347832, 2.722713633111154,
    2.4958208303518914, 0.5235987755982988, 0.3490658503988659,
    2.9321531433504737, 0.5235987755982988, 2.722713633111154,
    2.4958208303518914, 0.5235987755982988, 0.3490658503988659};

constexpr std::array<double, kA3LegDof> ScaleA3LegLimits(
    const std::array<double, kA3LegDof>& limits, double scale) {
  std::array<double, kA3LegDof> output{};
  for (std::size_t index = 0; index < kA3LegDof; ++index) {
    output[index] = limits[index] * scale;
  }
  return output;
}

}  // namespace a3_pingpong
