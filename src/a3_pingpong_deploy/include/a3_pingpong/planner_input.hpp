#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>

namespace a3_pingpong {

using SteadyClock = std::chrono::steady_clock;

struct RacketTargetInput {
  std::string frame_id;
  std::int64_t source_stamp_ns{0};
  std::uint64_t task_id{0};
  std::uint32_t task_revision{0};
  std::int8_t swing_side{0};
  std::array<double, 3> position_w{};
  std::array<double, 3> velocity_w{};
  double time_to_strike_s{0.0};
};

struct BasePoseInput {
  std::string frame_id;
  std::int64_t source_stamp_ns{0};
  std::array<double, 3> position_w{};
  std::array<double, 4> quaternion_wxyz{};
};

enum class InputUpdateResult {
  kAccepted,
  kDuplicate,
  kStale,
  kInvalid,
};

struct PlannerInputSnapshot {
  std::optional<RacketTargetInput> command;
  std::optional<BasePoseInput> base_pose;
  double command_age_s{0.0};
  double base_pose_age_s{0.0};

  bool Ready(double command_timeout_s, double base_pose_timeout_s) const;
};

class PlannerInputMailbox {
 public:
  explicit PlannerInputMailbox(std::string expected_frame);

  InputUpdateResult UpdateCommand(RacketTargetInput input,
                                  std::int64_t now_ros_ns,
                                  SteadyClock::time_point now,
                                  std::string* reason = nullptr);
  InputUpdateResult UpdateBasePose(BasePoseInput input,
                                   SteadyClock::time_point now,
                                   std::string* reason = nullptr);
  PlannerInputSnapshot Snapshot(SteadyClock::time_point now) const;
  void Reset();

  const std::string& expected_frame() const { return expected_frame_; }

 private:
  struct TimedCommand {
    RacketTargetInput value;
    SteadyClock::time_point received_at;
  };
  struct TimedPose {
    BasePoseInput value;
    SteadyClock::time_point received_at;
  };

  std::string expected_frame_;
  mutable std::mutex mutex_;
  std::optional<TimedCommand> command_;
  std::optional<TimedPose> base_pose_;
  std::uint64_t last_task_id_{0};
  std::uint32_t last_task_revision_{0};
  std::int8_t locked_swing_side_{0};
  bool have_task_{false};
};

}  // namespace a3_pingpong
