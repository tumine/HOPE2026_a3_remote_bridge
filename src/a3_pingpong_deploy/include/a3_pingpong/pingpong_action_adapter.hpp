#pragma once

#include "a3_pingpong/pingpong_observation_builder.hpp"
#include "robot_io/robot_io_backend.hpp"

#include <array>
#include <cstdint>
#include <string>

namespace a3_pingpong {

struct PingpongActionAdapterConfig {
  std::array<double, kPingpongActionDim> default_q{};
  std::array<double, kPingpongActionDim> lower{};
  std::array<double, kPingpongActionDim> upper{};
  double action_scale{0.25};
  double action_clip_lower{-100.0};
  double action_clip_upper{100.0};
};

struct PingpongActionDiagnostics {
  std::uint32_t raw_clip_count{0};
  std::uint32_t position_clip_count{0};
  double raw_max_abs{0.0};
  double applied_max_abs{0.0};
  double q_des_max_abs{0.0};
};

struct PingpongCommandGains {
  std::array<double, kPingpongActionDim> kp{};
  std::array<double, kPingpongActionDim> kd{};
};

PingpongActionAdapterConfig Model21500ActionAdapterConfig();
PingpongCommandGains Model21500PolicyGains();
PingpongCommandGains A3PdStandGains();

bool BuildPositionCommand(
    const std::array<double, kPingpongActionDim>& q_des,
    const PingpongCommandGains& gains,
    robot_io::RobotCommand& command,
    std::string* reason = nullptr);

bool BuildPdStandCommand(
    const std::array<double, kPingpongActionDim>& start_q,
    const std::array<double, kPingpongActionDim>& target_q,
    std::uint64_t elapsed_ticks,
    std::uint64_t ramp_ticks,
    robot_io::RobotCommand& command,
    bool* ready = nullptr,
    std::string* reason = nullptr);

class PingpongActionAdapter {
 public:
  explicit PingpongActionAdapter(PingpongActionAdapterConfig config);

  bool Decode(const PingpongAction& raw_action,
              PingpongAction& applied_action,
              std::array<double, kPingpongActionDim>& q_des,
              PingpongActionDiagnostics* diagnostics = nullptr,
              std::string* reason = nullptr) const;

  // Convert the 31-D model output into the exact RobotCommand consumed by the
  // official A3 RobotIO backend. dq_des and tau_ff are zero; q_des, Kp and Kd
  // use the frozen model_21500 contract and its deployment gains.
  bool BuildPolicyCommand(
      const PingpongAction& raw_action,
      PingpongAction& applied_action,
      robot_io::RobotCommand& command,
      PingpongActionDiagnostics* diagnostics = nullptr,
      std::string* reason = nullptr) const;

  // Retained for a narrow command-shape regression test. Runtime dry-run uses
  // BuildPolicyCommand so it validates the same packet that would be sent.
  bool BuildZeroGainDryRunCommand(
      const PingpongAction& raw_action,
      PingpongAction& applied_action,
      robot_io::RobotCommand& command,
      PingpongActionDiagnostics* diagnostics = nullptr,
      std::string* reason = nullptr) const;

 private:
  PingpongActionAdapterConfig config_;
};

}  // namespace a3_pingpong
