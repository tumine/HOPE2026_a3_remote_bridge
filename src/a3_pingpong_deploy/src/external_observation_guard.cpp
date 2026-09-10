#include "a3_pingpong/external_observation_guard.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace a3_pingpong {

const char* ExternalObservationModeName(
    ExternalObservationMode mode) noexcept {
  switch (mode) {
    case ExternalObservationMode::kWaiting: return "waiting";
    case ExternalObservationMode::kLive: return "live";
    case ExternalObservationMode::kHold: return "hold";
    case ExternalObservationMode::kFallbackPd: return "fallback_pd";
    case ExternalObservationMode::kRecoveredWait: return "recovered_wait";
  }
  return "unknown";
}

ExternalObservationGuard::ExternalObservationGuard(
    ExternalObservationGuardConfig config)
    : config_(std::move(config)) {
  if (!std::isfinite(config_.freshness_timeout_s) ||
      config_.freshness_timeout_s <= 0.0 ||
      !std::isfinite(config_.fallback_timeout_s) ||
      config_.fallback_timeout_s < 0.0 ||
      (config_.fallback_timeout_s > 0.0 &&
       config_.fallback_timeout_s <= config_.freshness_timeout_s) ||
      config_.recovery_frames == 0) {
    throw std::invalid_argument("invalid external observation guard config");
  }
}

ExternalObservationStatus ExternalObservationGuard::Update(
    const PlannerInputSnapshot& planner, SteadyClock::time_point now) {
  const bool have_pose = planner.base_pose.has_value();
  const bool fresh = have_pose && std::isfinite(planner.base_pose_age_s) &&
                     planner.base_pose_age_s <= config_.freshness_timeout_s;
  const bool fallback_enabled = config_.fallback_timeout_s > 0.0;

  // The mailbox retains its last accepted pose. Cache even a stale first
  // snapshot so a newly entered motion mode can measure the real outage age.
  if (have_pose && !status_.base_pose) status_.base_pose = planner.base_pose;
  if (fresh) {
    status_.base_pose = planner.base_pose;
    last_fresh_at_ =
        now - std::chrono::duration_cast<SteadyClock::duration>(
                  std::chrono::duration<double>(planner.base_pose_age_s));
  } else if (!last_fresh_at_ && have_pose &&
             std::isfinite(planner.base_pose_age_s)) {
    last_fresh_at_ =
        now - std::chrono::duration_cast<SteadyClock::duration>(
                  std::chrono::duration<double>(planner.base_pose_age_s));
  }

  status_.stale_duration_s =
      last_fresh_at_
          ? std::max(0.0,
                     std::chrono::duration<double>(now - *last_fresh_at_)
                         .count())
          : (fallback_enabled ? config_.fallback_timeout_s : 0.0);

  if (status_.fallback_latched) {
    status_.recovery_streak = fresh ? status_.recovery_streak + 1 : 0;
    status_.mode = status_.recovery_streak >= config_.recovery_frames
                       ? ExternalObservationMode::kRecoveredWait
                       : ExternalObservationMode::kFallbackPd;
    return status_;
  }

  status_.recovery_streak = 0;
  if (fresh) {
    status_.mode = ExternalObservationMode::kLive;
  } else if (status_.base_pose &&
             (!fallback_enabled ||
              status_.stale_duration_s <= config_.fallback_timeout_s)) {
    status_.mode = ExternalObservationMode::kHold;
  } else if (fallback_enabled) {
    status_.fallback_latched = true;
    status_.mode = ExternalObservationMode::kFallbackPd;
  } else {
    status_.mode = ExternalObservationMode::kWaiting;
  }
  return status_;
}

bool ExternalObservationGuard::RequestResume() {
  if (!status_.fallback_latched ||
      status_.mode != ExternalObservationMode::kRecoveredWait) {
    return false;
  }
  status_.fallback_latched = false;
  status_.recovery_streak = 0;
  status_.stale_duration_s = 0.0;
  status_.mode = ExternalObservationMode::kLive;
  return true;
}

void ExternalObservationGuard::Reset() {
  status_ = ExternalObservationStatus{};
  last_fresh_at_.reset();
}

}  // namespace a3_pingpong
