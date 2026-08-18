#pragma once

#include "a3_pingpong/pingpong_action_adapter.hpp"
#include "robot_io/robot_io_backend.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <string>

namespace a3_pingpong {

constexpr std::size_t kServeUpperBodyDim = 14;
constexpr std::size_t kServeRightArmDim = 7;
constexpr std::size_t kServeLowerBodyDim = 15;  // waist 3 + legs 12

using UpperBodyServeTarget = std::array<double, kServeUpperBodyDim>;
using LowerBodyServeTarget = std::array<double, kServeLowerBodyDim>;

enum class UpperBodyServePhase {
  kIdle,
  kPrepare,
  kWindup,
  kSwing,
  kSettle,
  kReturn,
  kComplete,
};

const char* UpperBodyServePhaseName(UpperBodyServePhase phase) noexcept;

struct UpperBodyServeConfig {
  // Canonical A3 arm order: left arm [0..6], right arm [7..13].
  UpperBodyServeTarget home_upper{
      -1.30, 0.00, 0.00, 1.30, 1.57, -1.57, 0.00,
      -0.57, -0.59, 0.65, 0.00, 0.01, 0.04, 0.70};
  std::array<double, kServeRightArmDim> windup_right{
      -0.33, 0.00, 0.85, 0.10, -0.30, -0.22, 0.53};
  std::array<double, kServeRightArmDim> hit_through_right{
      -0.80, -0.60, 0.35, 0.80, -0.43, 0.05, 0.80};
  double prepare_duration_s{2.0};
  double windup_duration_s{0.50};
  double swing_duration_s{0.04};
  double settle_duration_s{1.0};
  double return_duration_s{1.0};
};

struct UpperBodyServeDiagnostics {
  UpperBodyServePhase phase{UpperBodyServePhase::kIdle};
  std::uint64_t tick{0};
  bool release_requested{false};
  bool complete{false};
};

// One-shot upper-body trajectory. Start position is captured from the latest
// synchronized RobotState. No command is published by this class.
class UpperBodyServeTrajectory {
 public:
  explicit UpperBodyServeTrajectory(
      UpperBodyServeConfig config = UpperBodyServeConfig{});

  void Reset() noexcept;
  bool Step(const robot_io::RobotState& state, double dt_s,
            UpperBodyServeTarget& output,
            UpperBodyServeDiagnostics* diagnostics = nullptr,
            std::string* reason = nullptr);

  UpperBodyServePhase phase() const noexcept { return phase_; }
  bool complete() const noexcept {
    return phase_ == UpperBodyServePhase::kComplete;
  }

 private:
  bool Start(const robot_io::RobotState& state, std::string* reason);
  void AdvancePhase() noexcept;

  UpperBodyServeConfig config_;
  UpperBodyServeTarget start_upper_{};
  UpperBodyServePhase phase_{UpperBodyServePhase::kIdle};
  double phase_elapsed_s_{0.0};
  std::uint64_t tick_{0};
  bool release_emitted_{false};
};

// Maps the upper-body trajectory and an optional lower-body policy target into
// the exact A3 31-DOF command layout. Without a lower-body target, waist,
// neck, and legs hold the latest measured positions. A future lower-body
// policy supplies waist[0..2] + legs[0..11] through LowerBodyServeTarget.
class FullBodyServeComposer {
 public:
  explicit FullBodyServeComposer(
      PingpongCommandGains gains = A3PdStandGains());

  bool Build(const robot_io::RobotState& state,
             const UpperBodyServeTarget& upper_body,
             const std::optional<LowerBodyServeTarget>& lower_body,
             robot_io::RobotCommand& command,
             std::string* reason = nullptr) const;

 private:
  PingpongCommandGains gains_;
};

}  // namespace a3_pingpong
