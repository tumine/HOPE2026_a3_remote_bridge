#pragma once

#include "a3_pingpong/planner_input.hpp"

#include <chrono>
#include <cstdint>
#include <optional>

namespace a3_pingpong {

enum class ExternalObservationMode {
  kWaiting,
  kLive,
  kHold,
  kFallbackPd,
  kRecoveredWait,
};

const char* ExternalObservationModeName(ExternalObservationMode mode) noexcept;

struct ExternalObservationGuardConfig {
  double freshness_timeout_s{0.100};
  double fallback_timeout_s{0.500};
  std::uint64_t recovery_frames{10};
};

struct ExternalObservationStatus {
  ExternalObservationMode mode{ExternalObservationMode::kWaiting};
  std::optional<BasePoseInput> base_pose;
  double stale_duration_s{0.0};
  std::uint64_t recovery_streak{0};
  bool fallback_latched{false};
};

// Keeps only the last valid exteroceptive base pose. Proprioception is never
// cached here and therefore remains live in the policy observation. A long
// outage latches fallback until fresh poses have remained stable and the
// operator explicitly requests resume.
class ExternalObservationGuard {
 public:
  explicit ExternalObservationGuard(ExternalObservationGuardConfig config);

  ExternalObservationStatus Update(const PlannerInputSnapshot& planner,
                                   SteadyClock::time_point now);
  bool RequestResume();
  void Reset();

  const ExternalObservationStatus& status() const noexcept { return status_; }

 private:
  ExternalObservationGuardConfig config_;
  ExternalObservationStatus status_;
  std::optional<SteadyClock::time_point> last_fresh_at_;
};

}  // namespace a3_pingpong
