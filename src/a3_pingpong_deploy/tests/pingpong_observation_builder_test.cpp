#include "a3_pingpong/pingpong_observation_builder.hpp"
#include "robot_io/a3_layout_extra.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <string>

#define CHECK(condition)              \
  do {                                \
    if (!(condition)) return __LINE__; \
  } while (false)

#ifndef A3_OBSERVATION_GOLDEN_PATH
#error "A3_OBSERVATION_GOLDEN_PATH must be defined"
#endif

int main() {
  std::string reason;
  CHECK(a3_pingpong::ValidatePingpongJointLayout(
      robot_io::MakeA3Layout31(), &reason));

  robot_io::JointLayout wrong_layout = robot_io::MakeA3Layout31();
  std::swap(wrong_layout.names[0], wrong_layout.names[1]);
  CHECK(!a3_pingpong::ValidatePingpongJointLayout(wrong_layout, &reason));

  const auto config = a3_pingpong::Model50000ObservationConfig();
  a3_pingpong::PingpongObservationBuilder builder(config);

  robot_io::RobotState state;
  state.q = Eigen::VectorXd(31);
  state.dq = Eigen::VectorXd(31);
  for (std::size_t index = 0; index < 31; ++index) {
    state.q[static_cast<Eigen::Index>(index)] =
        -0.25 + static_cast<double>(index) * 0.025;
    state.dq[static_cast<Eigen::Index>(index)] =
        -0.3 + static_cast<double>(index) * 0.02;
  }
  state.imu_quat_wxyz = Eigen::Vector4d(0.9, 0.1, -0.2, 0.3);
  state.imu_gyro = Eigen::Vector3d(0.11, -0.22, 0.33);

  a3_pingpong::PlannerInputSnapshot planner;
  planner.base_pose.emplace();
  planner.base_pose->position_w = {0.2, -0.4, 0.7};
  planner.base_pose->quaternion_wxyz = {0.7, -0.1, 0.2, 0.65};
  planner.command.emplace();
  planner.command->position_w = {0.45, -1.1, 0.25};
  planner.command->velocity_w = {2.2, 0.4, 0.9};
  planner.command->time_to_strike_s = 0.7345;
  planner.command->swing_side = -1;

  a3_pingpong::PingpongAction last_action{};
  for (std::size_t index = 0; index < last_action.size(); ++index) {
    last_action[index] =
        static_cast<float>((static_cast<double>(index) - 15.0) * 0.03);
  }

  a3_pingpong::PingpongObservation observation{};
  CHECK(builder.Build(state, planner, last_action, {-0.5, -0.7625},
                      observation, &reason));

  // The two attitude sources are intentional: IMU drives gravity and PPMocap
  // drives world/table heading.
  const auto original_gravity =
      std::array<float, 3>{observation[96], observation[97], observation[98]};
  const auto original_heading =
      std::array<float, 2>{observation[99], observation[100]};
  state.imu_quat_wxyz = Eigen::Vector4d(1.0, 0.0, 0.0, 0.0);
  CHECK(builder.Build(state, planner, last_action, {-0.5, -0.7625},
                      observation, &reason));
  CHECK(std::abs(observation[96]) <= 1.0e-6F);
  CHECK(std::abs(observation[97]) <= 1.0e-6F);
  CHECK(std::abs(observation[98] + 1.0F) <= 1.0e-6F);
  CHECK(std::abs(observation[99] - original_heading[0]) <= 1.0e-6F);
  CHECK(std::abs(observation[100] - original_heading[1]) <= 1.0e-6F);
  CHECK(std::abs(original_gravity[0] - observation[96]) > 1.0e-3F ||
        std::abs(original_gravity[1] - observation[97]) > 1.0e-3F ||
        std::abs(original_gravity[2] - observation[98]) > 1.0e-3F);
  state.imu_quat_wxyz = Eigen::Vector4d(0.9, 0.1, -0.2, 0.3);
  CHECK(builder.Build(state, planner, last_action, {-0.5, -0.7625},
                      observation, &reason));

  std::ifstream fixture(A3_OBSERVATION_GOLDEN_PATH);
  CHECK(fixture.good());
  for (std::size_t index = 0; index < observation.size(); ++index) {
    double expected = 0.0;
    CHECK(static_cast<bool>(fixture >> expected));
    CHECK(std::abs(static_cast<double>(observation[index]) - expected) <=
          1.0e-6);
  }
  double extra = 0.0;
  CHECK(!(fixture >> extra));

  state.imu_quat_wxyz = Eigen::Vector4d::Zero();
  CHECK(!builder.Build(state, planner, last_action, {-0.5, -0.7625},
                       observation, &reason));
  state.imu_quat_wxyz = Eigen::Vector4d(0.9, 0.1, -0.2, 0.3);

  state.q[0] = std::numeric_limits<double>::quiet_NaN();
  CHECK(!builder.Build(state, planner, last_action, {-0.5, -0.7625},
                       observation, &reason));
  state.q[0] = -0.25;
  planner.command->swing_side = 0;
  CHECK(!builder.Build(state, planner, last_action, {-0.5, -0.7625},
                       observation, &reason));
  return 0;
}
