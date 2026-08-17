#pragma once

#include "a3_pingpong/planner_input.hpp"

#include <array>
#include <cstdint>
#include <optional>

namespace a3_pingpong {

enum class SwingPhase {
  kReady,
  kSwing,
  kFollowThrough,
  kRecovery,
};

struct SwingLifecycleConfig {
  double control_hz{50.0};
  double follow_through_s{0.8};
  double recovery_s{0.0};
  double ready_time_to_strike_s{1.0};
  std::int8_t ready_swing_side{1};
  std::array<double, 3> ready_reference_base_w{};
  std::array<double, 3> ready_target_rel_base_w{};
  std::array<double, 3> ready_target_velocity_w{};
};

// Frozen model_21500 lifecycle values from the HOPE deployment bundle.
SwingLifecycleConfig Model21500SwingLifecycleConfig();

// C++ migration of pc_tools/a3_rl_contract.py::SwingLifecycle. A planner
// command is accepted only once per new task_id. Once engaged, its TTS is
// advanced locally at the policy rate, followed by the configured follow-
// through and recovery phases. With no eligible command, the lifecycle emits
// the fixed model_21500 READY target.
class SwingLifecycle {
 public:
  explicit SwingLifecycle(SwingLifecycleConfig config);

  RacketTargetInput Update(const std::optional<RacketTargetInput>& command);
  void Advance();
  void Reset();

  SwingPhase phase() const noexcept { return phase_; }
  std::optional<std::uint64_t> active_task_id() const noexcept {
    return active_task_id_;
  }
  std::optional<std::uint64_t> last_engaged_task_id() const noexcept {
    return last_engaged_task_id_;
  }
  std::uint32_t applied_revision() const noexcept { return applied_revision_; }
  const SwingLifecycleConfig& config() const noexcept { return config_; }

 private:
  RacketTargetInput ReadyTarget() const;
  bool CommandEligible(const RacketTargetInput& command) const;

  SwingLifecycleConfig config_;
  SwingPhase phase_{SwingPhase::kReady};
  std::optional<std::uint64_t> active_task_id_;
  std::optional<std::uint64_t> last_engaged_task_id_;
  std::uint32_t applied_revision_{0};
  RacketTargetInput active_target_;
  double follow_time_s_{0.0};
  double recovery_time_s_{0.0};
};

}  // namespace a3_pingpong
