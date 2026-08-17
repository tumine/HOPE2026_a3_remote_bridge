#pragma once

#include "a3_pingpong/planner_input.hpp"
#include "robot_io/robot_io_backend.hpp"

#include <array>
#include <cstddef>
#include <string>

namespace a3_pingpong {

constexpr std::size_t kPingpongActionDim = 31;
constexpr std::size_t kPingpongObservationDim = 111;

using PingpongAction = std::array<float, kPingpongActionDim>;
using PingpongObservation = std::array<float, kPingpongObservationDim>;

struct PingpongObservationConfig {
  std::array<double, kPingpongActionDim> default_q{};
};

// Frozen model_21500 values from the HOPE bundle's action_adapter.yaml.
PingpongObservationConfig Model21500ObservationConfig();

// Validate that RobotIO's 31 slots match the training/ONNX contract exactly.
bool ValidatePingpongJointLayout(const robot_io::JointLayout& layout,
                                 std::string* reason = nullptr);

class PingpongObservationBuilder {
 public:
  explicit PingpongObservationBuilder(PingpongObservationConfig config);

  // Reproduces A3RLContract.build_observation() exactly. The planner snapshot
  // must already have transport/queue age subtracted from time_to_strike.
  bool Build(const robot_io::RobotState& state,
             const PlannerInputSnapshot& planner,
             const PingpongAction& last_action,
             const std::array<double, 2>& fixed_station_xy,
             PingpongObservation& output,
             std::string* reason = nullptr) const;

  const PingpongObservationConfig& config() const noexcept { return config_; }

 private:
  PingpongObservationConfig config_;
};

}  // namespace a3_pingpong
