#include "a3_pingpong/swing_lifecycle.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <optional>

#define CHECK(condition)                  \
  do {                                    \
    if (!(condition)) return __LINE__;    \
  } while (false)

namespace {

a3_pingpong::RacketTargetInput Command(std::uint64_t task_id,
                                        std::uint32_t revision,
                                        double tts,
                                        std::int8_t side = 1) {
  a3_pingpong::RacketTargetInput command;
  command.frame_id = "hope_table";
  command.task_id = task_id;
  command.task_revision = revision;
  command.swing_side = side;
  command.position_w = {0.4 + 0.01 * revision, -1.0, 0.2};
  command.velocity_w = {2.0, 0.5, 1.0};
  command.time_to_strike_s = tts;
  return command;
}

}  // namespace

int main() {
  const auto config = a3_pingpong::Model50000SwingLifecycleConfig();
  a3_pingpong::SwingLifecycle lifecycle(config);
  const std::array<double, 3> base0{-0.5, -0.7625, 0.3064};

  const auto ready = lifecycle.Update(std::nullopt, base0);
  CHECK(lifecycle.phase() == a3_pingpong::SwingPhase::kReady);
  CHECK(std::abs(ready.position_w[0] - (-0.05)) < 1.0e-12);
  CHECK(std::abs(ready.position_w[1] - (-1.0125)) < 1.0e-12);
  CHECK(std::abs(ready.position_w[2] - 0.3864) < 1.0e-12);
  const std::array<double, 3> zero_velocity{0.0, 0.0, 0.0};
  CHECK(ready.velocity_w == zero_velocity);
  CHECK(std::abs(ready.time_to_strike_s - 1.0) < 1.0e-12);
  CHECK(ready.swing_side == 0);
  const std::array<double, 3> moved_base{0.1, -0.2, 0.3};
  const auto moved_ready = lifecycle.Update(std::nullopt, moved_base);
  CHECK(std::abs(moved_ready.position_w[0] - 0.55) < 1.0e-12);
  CHECK(std::abs(moved_ready.position_w[1] - (-0.45)) < 1.0e-12);
  CHECK(std::abs(moved_ready.position_w[2] - 0.38) < 1.0e-12);
  for (std::size_t index = 0; index < 3; ++index) {
    CHECK(std::abs(moved_ready.position_w[index] - moved_base[index] -
                   config.ready_target_rel_base_w[index]) < 1.0e-12);
  }
  lifecycle.Advance();
  CHECK(std::abs(lifecycle.Update(std::nullopt, moved_base).time_to_strike_s - 1.0) <
        1.0e-12);

  auto target = lifecycle.Update(Command(1, 0, 1.0), moved_base);
  CHECK(lifecycle.phase() == a3_pingpong::SwingPhase::kSwing);
  CHECK(lifecycle.active_task_id() == std::optional<std::uint64_t>(1));
  CHECK(std::abs(target.position_w[0] - 0.4) < 1.0e-12);

  lifecycle.Advance();
  target = lifecycle.Update(Command(1, 0, 0.9), moved_base);
  CHECK(std::abs(target.time_to_strike_s - 0.98) < 1.0e-12);

  target = lifecycle.Update(Command(1, 1, 0.85), moved_base);
  CHECK(lifecycle.applied_revision() == 1);
  CHECK(std::abs(target.position_w[0] - 0.41) < 1.0e-12);
  CHECK(std::abs(target.time_to_strike_s - 0.85) < 1.0e-12);

  // A different task cannot interrupt an active swing.
  target = lifecycle.Update(Command(2, 0, 0.8, -1), moved_base);
  CHECK(lifecycle.active_task_id() == std::optional<std::uint64_t>(1));
  CHECK(target.swing_side == 1);

  for (int index = 0; index < 43; ++index) lifecycle.Advance();
  CHECK(lifecycle.phase() == a3_pingpong::SwingPhase::kFollowThrough);
  // Match the old half-tick phase-boundary rule: transition after 0.82 s.
  for (int index = 0; index < 41; ++index) lifecycle.Advance();
  CHECK(lifecycle.phase() == a3_pingpong::SwingPhase::kReady);
  CHECK(!lifecycle.active_task_id());

  // The completed task cannot re-engage; the next task can.
  target = lifecycle.Update(Command(1, 2, 0.7), moved_base);
  CHECK(lifecycle.phase() == a3_pingpong::SwingPhase::kReady);
  target = lifecycle.Update(Command(2, 0, 0.7, -1), moved_base);
  CHECK(lifecycle.phase() == a3_pingpong::SwingPhase::kSwing);
  CHECK(lifecycle.active_task_id() == std::optional<std::uint64_t>(2));
  CHECK(target.swing_side == -1);

  lifecycle.Reset();
  CHECK(lifecycle.phase() == a3_pingpong::SwingPhase::kReady);
  CHECK(!lifecycle.last_engaged_task_id());

  // A stale new id is consumed but never engages; its later revision cannot
  // trigger a partial swing.
  auto stale = Command(3, 0, -0.01);
  target = lifecycle.Update(stale, moved_base);
  CHECK(lifecycle.phase() == a3_pingpong::SwingPhase::kReady);
  CHECK(lifecycle.last_engaged_task_id() ==
        std::optional<std::uint64_t>(3));
  target = lifecycle.Update(Command(3, 1, 0.5), moved_base);
  CHECK(lifecycle.phase() == a3_pingpong::SwingPhase::kReady);
  return 0;
}
