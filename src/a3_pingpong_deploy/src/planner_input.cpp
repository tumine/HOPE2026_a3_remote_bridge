#include "a3_pingpong/planner_input.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

namespace a3_pingpong {
namespace {

template <std::size_t N>
bool AllFinite(const std::array<double, N>& values) {
  for (double value : values) {
    if (!std::isfinite(value)) return false;
  }
  return true;
}

void SetReason(std::string* output, std::string value) {
  if (output) *output = std::move(value);
}

double AgeSeconds(SteadyClock::time_point now,
                  SteadyClock::time_point received_at) {
  return std::max(
      0.0, std::chrono::duration<double>(now - received_at).count());
}

}  // namespace

bool PlannerInputSnapshot::Ready(double command_timeout_s,
                                 double base_pose_timeout_s) const {
  return command.has_value() && base_pose.has_value() &&
         command_age_s <= command_timeout_s &&
         base_pose_age_s <= base_pose_timeout_s &&
         command->time_to_strike_s > 0.0;
}

PlannerInputMailbox::PlannerInputMailbox(std::string expected_frame)
    : expected_frame_(std::move(expected_frame)) {}

InputUpdateResult PlannerInputMailbox::UpdateCommand(
    RacketTargetInput input, std::int64_t now_ros_ns,
    SteadyClock::time_point now, std::string* reason) {
  if (input.frame_id != expected_frame_) {
    SetReason(reason, "unexpected command frame: " + input.frame_id);
    return InputUpdateResult::kInvalid;
  }
  if (input.swing_side != 1 && input.swing_side != -1) {
    SetReason(reason, "swing_side must be +1 or -1");
    return InputUpdateResult::kInvalid;
  }
  if (!AllFinite(input.position_w) || !AllFinite(input.velocity_w) ||
      !std::isfinite(input.time_to_strike_s) ||
      input.time_to_strike_s < 0.0) {
    SetReason(reason, "command contains a non-finite value or negative TTS");
    return InputUpdateResult::kInvalid;
  }

  double transport_age_s = 0.0;
  if (input.source_stamp_ns > 0 && now_ros_ns >= input.source_stamp_ns) {
    transport_age_s =
        static_cast<double>(now_ros_ns - input.source_stamp_ns) * 1.0e-9;
  }
  input.time_to_strike_s -= transport_age_s;
  if (input.time_to_strike_s <= 0.0) {
    SetReason(reason, "command expired before reaching the MDU");
    return InputUpdateResult::kStale;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  if (have_task_) {
    if (input.task_id < last_task_id_) {
      SetReason(reason, "task_id is older than the latest accepted task");
      return InputUpdateResult::kStale;
    }
    if (input.task_id == last_task_id_) {
      if (input.swing_side != locked_swing_side_) {
        SetReason(reason, "swing_side changed within one task");
        return InputUpdateResult::kInvalid;
      }
      if (input.task_revision <= last_task_revision_) {
        SetReason(reason, "task revision is duplicate or out of order");
        return input.task_revision == last_task_revision_
                   ? InputUpdateResult::kDuplicate
                   : InputUpdateResult::kStale;
      }
    }
  }

  if (!have_task_ || input.task_id > last_task_id_) {
    last_task_id_ = input.task_id;
    locked_swing_side_ = input.swing_side;
  }
  last_task_revision_ = input.task_revision;
  have_task_ = true;
  command_ = TimedCommand{std::move(input), now};
  SetReason(reason, "accepted");
  return InputUpdateResult::kAccepted;
}

InputUpdateResult PlannerInputMailbox::UpdateBasePose(
    BasePoseInput input, SteadyClock::time_point now, std::string* reason) {
  if (input.frame_id != expected_frame_) {
    SetReason(reason, "unexpected base-pose frame: " + input.frame_id);
    return InputUpdateResult::kInvalid;
  }
  if (!AllFinite(input.position_w) || !AllFinite(input.quaternion_wxyz)) {
    SetReason(reason, "base pose contains a non-finite value");
    return InputUpdateResult::kInvalid;
  }
  double squared_norm = 0.0;
  for (double value : input.quaternion_wxyz) squared_norm += value * value;
  const double norm = std::sqrt(squared_norm);
  if (!std::isfinite(norm) || norm < 1.0e-9) {
    SetReason(reason, "base-pose quaternion has zero norm");
    return InputUpdateResult::kInvalid;
  }
  for (double& value : input.quaternion_wxyz) value /= norm;

  std::lock_guard<std::mutex> lock(mutex_);
  base_pose_ = TimedPose{std::move(input), now};
  SetReason(reason, "accepted");
  return InputUpdateResult::kAccepted;
}

PlannerInputSnapshot PlannerInputMailbox::Snapshot(
    SteadyClock::time_point now) const {
  std::lock_guard<std::mutex> lock(mutex_);
  PlannerInputSnapshot output;
  if (command_) {
    output.command_age_s = AgeSeconds(now, command_->received_at);
    output.command = command_->value;
    output.command->time_to_strike_s -= output.command_age_s;
  }
  if (base_pose_) {
    output.base_pose_age_s = AgeSeconds(now, base_pose_->received_at);
    output.base_pose = base_pose_->value;
  }
  return output;
}

void PlannerInputMailbox::Reset() {
  std::lock_guard<std::mutex> lock(mutex_);
  command_.reset();
  base_pose_.reset();
  last_task_id_ = 0;
  last_task_revision_ = 0;
  locked_swing_side_ = 0;
  have_task_ = false;
}

}  // namespace a3_pingpong
