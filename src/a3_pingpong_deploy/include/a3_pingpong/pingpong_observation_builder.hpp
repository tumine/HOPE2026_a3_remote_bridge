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

// Frozen model_50000 values from the sim2sim bundle action adapter.
PingpongObservationConfig Model50000ObservationConfig();

// Compatibility aliases retained for existing callers.
PingpongObservationConfig Model48000ObservationConfig();

// Compatibility alias for callers that still use the previous checkpoint name.
PingpongObservationConfig Model41500ObservationConfig();

// Compatibility alias for callers that still use the previous policy name.
PingpongObservationConfig Model21500ObservationConfig();

// Validate that RobotIO's 31 slots match the training/ONNX contract exactly.
bool ValidatePingpongJointLayout(const robot_io::JointLayout& layout,
                                 std::string* reason = nullptr);

class PingpongObservationBuilder {
 public:
  explicit PingpongObservationBuilder(PingpongObservationConfig config);

  // Reproduces the hardware A3RLContract observation. Projected gravity uses
  // the pelvis IMU while world/table heading uses calibrated PPMocap. The
  // planner snapshot must already have transport/queue age subtracted from
  // time_to_strike.
  bool Build(const robot_io::RobotState& state,
             const PlannerInputSnapshot& planner,
             const PingpongAction& last_action,
             const std::array<double, 2>& base_target_xy,
             PingpongObservation& output,
             std::string* reason = nullptr) const;

  const PingpongObservationConfig& config() const noexcept { return config_; }

 private:
  PingpongObservationConfig config_;
};

}  // namespace a3_pingpong
