#include "a3_pingpong/planner_input.hpp"

#include <cassert>
#include <chrono>
#include <cmath>
#include <string>

namespace {

a3_pingpong::RacketTargetInput Command(std::uint64_t task,
                                       std::uint32_t revision,
                                       std::int8_t side = 1) {
  a3_pingpong::RacketTargetInput value;
  value.frame_id = "hope_table";
  value.source_stamp_ns = 9'900'000'000LL;
  value.task_id = task;
  value.task_revision = revision;
  value.swing_side = side;
  value.position_w = {0.1, -0.2, 0.3};
  value.velocity_w = {2.0, 0.5, 0.8};
  value.time_to_strike_s = 0.5;
  return value;
}

a3_pingpong::BasePoseInput Pose() {
  a3_pingpong::BasePoseInput value;
  value.frame_id = "hope_table";
  value.source_stamp_ns = 10'000'000'000LL;
  value.position_w = {-0.5, -0.7625, 0.3064};
  value.quaternion_wxyz = {2.0, 0.0, 0.0, 0.0};
  return value;
}

}  // namespace

int main() {
  using a3_pingpong::InputUpdateResult;
  using a3_pingpong::PlannerInputMailbox;
  using a3_pingpong::SteadyClock;

  PlannerInputMailbox mailbox("hope_table");
  const auto start = SteadyClock::time_point(std::chrono::seconds(1));
  std::string reason;

  assert(mailbox.UpdateBasePose(Pose(), start, &reason) ==
         InputUpdateResult::kAccepted);
  assert(mailbox.UpdateCommand(Command(1, 0), 10'000'000'000LL, start,
                               &reason) == InputUpdateResult::kAccepted);

  const auto snapshot = mailbox.Snapshot(start + std::chrono::milliseconds(50));
  assert(snapshot.Ready(0.2, 0.2));
  assert(snapshot.command.has_value());
  assert(std::abs(snapshot.command->time_to_strike_s - 0.35) < 1.0e-9);
  assert(snapshot.base_pose.has_value());
  assert(std::abs(snapshot.base_pose->quaternion_wxyz[0] - 1.0) < 1.0e-12);

  assert(mailbox.UpdateCommand(Command(1, 0), 10'000'000'000LL, start,
                               &reason) == InputUpdateResult::kDuplicate);
  assert(mailbox.UpdateCommand(Command(1, 1, -1), 10'000'000'000LL, start,
                               &reason) == InputUpdateResult::kInvalid);
  assert(mailbox.UpdateCommand(Command(2, 0, -1), 10'000'000'000LL, start,
                               &reason) == InputUpdateResult::kAccepted);
  assert(mailbox.UpdateCommand(Command(1, 2), 10'000'000'000LL, start,
                               &reason) == InputUpdateResult::kStale);

  auto bad_frame = Command(3, 0);
  bad_frame.frame_id = "map";
  assert(mailbox.UpdateCommand(bad_frame, 10'000'000'000LL, start, &reason) ==
         InputUpdateResult::kInvalid);

  auto expired = Command(3, 0);
  expired.time_to_strike_s = 0.05;
  assert(mailbox.UpdateCommand(expired, 10'000'000'000LL, start, &reason) ==
         InputUpdateResult::kStale);
  return 0;
}
