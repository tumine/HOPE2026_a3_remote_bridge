#include "a3_pingpong/external_observation_guard.hpp"
#include "a3_pingpong/swing_lifecycle.hpp"

#include <chrono>
#include <cstdlib>
#include <iostream>

#define CHECK(expression)                                                   \
  do {                                                                      \
    if (!(expression)) {                                                    \
      std::cerr << "CHECK failed at " << __FILE__ << ':' << __LINE__       \
                << ": " #expression << '\n';                              \
      return EXIT_FAILURE;                                                  \
    }                                                                       \
  } while (false)

int main() {
  using a3_pingpong::ExternalObservationMode;
  using Clock = a3_pingpong::SteadyClock;

  a3_pingpong::ExternalObservationGuardConfig config;
  config.freshness_timeout_s = 0.1;
  config.fallback_timeout_s = 0.5;
  config.recovery_frames = 3;
  a3_pingpong::ExternalObservationGuard guard(config);
  const auto start = Clock::time_point{};

  a3_pingpong::PlannerInputSnapshot planner;
  planner.base_pose = a3_pingpong::BasePoseInput{};
  planner.base_pose_age_s = 0.0;
  CHECK(guard.Update(planner, start).mode == ExternalObservationMode::kLive);

  planner.base_pose.reset();
  auto status = guard.Update(planner, start + std::chrono::milliseconds(120));
  CHECK(status.mode == ExternalObservationMode::kHold);
  CHECK(status.base_pose.has_value());

  // Holding base pose must not freeze the locally advanced strike clock.
  a3_pingpong::SwingLifecycleConfig lifecycle_config;
  lifecycle_config.control_hz = 50.0;
  lifecycle_config.follow_through_s = 0.8;
  lifecycle_config.ready_time_to_strike_s = 1.0;
  a3_pingpong::SwingLifecycle lifecycle(lifecycle_config);
  a3_pingpong::RacketTargetInput command;
  command.task_id = 1;
  command.task_revision = 1;
  command.swing_side = 1;
  command.time_to_strike_s = 0.4;
  const std::array<double, 3> held_base{};
  CHECK(lifecycle.Update(command, held_base).time_to_strike_s == 0.4);
  lifecycle.Advance();
  CHECK(lifecycle.Update(std::nullopt, held_base).time_to_strike_s < 0.4);

  status = guard.Update(planner, start + std::chrono::milliseconds(520));
  CHECK(status.mode == ExternalObservationMode::kFallbackPd);
  CHECK(status.fallback_latched);
  CHECK(!guard.RequestResume());

  planner.base_pose = a3_pingpong::BasePoseInput{};
  planner.base_pose_age_s = 0.0;
  CHECK(guard.Update(planner, start + std::chrono::milliseconds(540)).mode ==
        ExternalObservationMode::kFallbackPd);
  CHECK(guard.Update(planner, start + std::chrono::milliseconds(560)).mode ==
        ExternalObservationMode::kFallbackPd);
  CHECK(guard.Update(planner, start + std::chrono::milliseconds(580)).mode ==
        ExternalObservationMode::kRecoveredWait);
  CHECK(guard.RequestResume());
  CHECK(guard.status().mode == ExternalObservationMode::kLive);

  guard.Reset();
  CHECK(guard.Update({}, start).mode == ExternalObservationMode::kFallbackPd);

  // Deployment mode: no automatic external-pose fallback. Once a valid pose
  // exists it is held indefinitely, and a missing first pose remains waiting.
  config.fallback_timeout_s = 0.0;
  a3_pingpong::ExternalObservationGuard hold_forever(config);
  CHECK(hold_forever.Update({}, start).mode == ExternalObservationMode::kWaiting);
  planner.base_pose = a3_pingpong::BasePoseInput{};
  planner.base_pose_age_s = 0.0;
  CHECK(hold_forever.Update(planner, start).mode == ExternalObservationMode::kLive);
  planner.base_pose.reset();
  status = hold_forever.Update(planner, start + std::chrono::seconds(30));
  CHECK(status.mode == ExternalObservationMode::kHold);
  CHECK(status.base_pose.has_value());
  CHECK(!status.fallback_latched);
  return EXIT_SUCCESS;
}
