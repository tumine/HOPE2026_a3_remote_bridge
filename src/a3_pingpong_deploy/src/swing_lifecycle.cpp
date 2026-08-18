#include "a3_pingpong/swing_lifecycle.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace a3_pingpong {
namespace {

template <std::size_t N>
bool AllFinite(const std::array<double, N>& values) {
  for (const double value : values) {
    if (!std::isfinite(value)) return false;
  }
  return true;
}

}  // namespace

SwingLifecycleConfig Model50000SwingLifecycleConfig() {
  SwingLifecycleConfig config;
  config.control_hz = 50.0;
  config.follow_through_s = 0.8;
  config.recovery_s = 0.0;
  config.ready_time_to_strike_s = 1.0;
  config.ready_swing_side = 1;
  config.ready_target_tracks_live_base = true;
  // Compatibility fallback only. model_50000 READY targets are rebuilt from
  // the live pelvis each tick, exactly like the reference runner.
  config.ready_reference_base_w = {-0.5, -0.7625, 0.3064};
  config.ready_target_rel_base_w = {
      0.3201315701007843,
      -0.6952511668205261,
      -0.057795558124780655,
  };
  config.ready_target_velocity_w = {
      2.3803303241729736,
      0.6271703243255615,
      1.0374835729599,
  };
  return config;
}

SwingLifecycleConfig Model48000SwingLifecycleConfig() {
  return Model50000SwingLifecycleConfig();
}

SwingLifecycleConfig Model21500SwingLifecycleConfig() {
  return Model48000SwingLifecycleConfig();
}

SwingLifecycleConfig Model41500SwingLifecycleConfig() {
  return Model48000SwingLifecycleConfig();
}

SwingLifecycle::SwingLifecycle(SwingLifecycleConfig config)
    : config_(std::move(config)) {
  if (!std::isfinite(config_.control_hz) || config_.control_hz <= 0.0 ||
      !std::isfinite(config_.follow_through_s) ||
      config_.follow_through_s < 0.0 ||
      !std::isfinite(config_.recovery_s) || config_.recovery_s < 0.0 ||
      !std::isfinite(config_.ready_time_to_strike_s) ||
      config_.ready_time_to_strike_s <= 0.0 ||
      (config_.ready_swing_side != 1 && config_.ready_swing_side != -1) ||
      !AllFinite(config_.ready_reference_base_w) ||
      !AllFinite(config_.ready_target_rel_base_w) ||
      !AllFinite(config_.ready_target_velocity_w)) {
    throw std::invalid_argument("invalid swing lifecycle configuration");
  }
  active_target_ = ReadyTarget(config_.ready_reference_base_w);
}

RacketTargetInput SwingLifecycle::ReadyTarget(
    const std::array<double, 3>& live_base_position_w) const {
  RacketTargetInput target;
  target.frame_id = "hope_table";
  target.swing_side = config_.ready_swing_side;
  target.time_to_strike_s = config_.ready_time_to_strike_s;
  const auto& base = config_.ready_target_tracks_live_base
                         ? live_base_position_w
                         : config_.ready_reference_base_w;
  for (std::size_t index = 0; index < 3; ++index) {
    target.position_w[index] =
        base[index] + config_.ready_target_rel_base_w[index];
    target.velocity_w[index] = config_.ready_target_velocity_w[index];
  }
  return target;
}

void SwingLifecycle::SetReadyReferenceBase(
    const std::array<double, 3>& base_position_w) {
  if (!AllFinite(base_position_w)) {
    throw std::invalid_argument("READY reference base contains invalid values");
  }
  config_.ready_reference_base_w = base_position_w;
  if (phase_ == SwingPhase::kReady) {
    active_target_ = ReadyTarget(base_position_w);
  }
}

bool SwingLifecycle::CommandStructurallyValid(
    const RacketTargetInput& command) const {
  return (command.swing_side == 1 || command.swing_side == -1) &&
         AllFinite(command.position_w) && AllFinite(command.velocity_w);
}

RacketTargetInput SwingLifecycle::Update(
    const std::optional<RacketTargetInput>& command,
    const std::array<double, 3>& live_base_position_w) {
  if (command && CommandStructurallyValid(*command)) {
    const bool new_task =
        !last_engaged_task_id_ || command->task_id > *last_engaged_task_id_;
    if (new_task &&
        (phase_ == SwingPhase::kReady || phase_ == SwingPhase::kRecovery)) {
      // Consume every strictly newer id, even if its clock has already expired.
      // This prevents a later stale revision of one ball from triggering a
      // partial swing, matching the reference lifecycle.
      last_engaged_task_id_ = command->task_id;
      if (std::isfinite(command->time_to_strike_s) &&
          command->time_to_strike_s >= 0.0) {
        active_task_id_ = command->task_id;
        applied_revision_ = command->task_revision;
        active_target_ = *command;
        phase_ = SwingPhase::kSwing;
        follow_time_s_ = 0.0;
        recovery_time_s_ = 0.0;
      }
    } else if (active_task_id_ &&
               command->task_id == *active_task_id_ &&
               phase_ == SwingPhase::kSwing &&
               active_target_.time_to_strike_s > 0.0 &&
               std::isfinite(command->time_to_strike_s) &&
               command->time_to_strike_s >= 0.0 &&
               command->task_revision > applied_revision_) {
      applied_revision_ = command->task_revision;
      active_target_.source_stamp_ns = command->source_stamp_ns;
      active_target_.task_revision = command->task_revision;
      active_target_.position_w = command->position_w;
      active_target_.velocity_w = command->velocity_w;
      active_target_.time_to_strike_s = command->time_to_strike_s;
    }
  }

  if (phase_ == SwingPhase::kSwing ||
      phase_ == SwingPhase::kFollowThrough) {
    return active_target_;
  }
  return ReadyTarget(live_base_position_w);
}

RacketTargetInput SwingLifecycle::Update(
    const std::optional<RacketTargetInput>& command) {
  return Update(command, config_.ready_reference_base_w);
}

void SwingLifecycle::Advance() {
  const double dt = 1.0 / config_.control_hz;
  if (phase_ == SwingPhase::kSwing) {
    active_target_.time_to_strike_s -= dt;
    if (active_target_.time_to_strike_s <= 1.0e-9) {
      if (std::abs(active_target_.time_to_strike_s) <= 1.0e-9) {
        active_target_.time_to_strike_s = 0.0;
      }
      phase_ = SwingPhase::kFollowThrough;
      follow_time_s_ = 0.0;
    }
    return;
  }
  if (phase_ == SwingPhase::kFollowThrough) {
    active_target_.time_to_strike_s -= dt;
    follow_time_s_ += dt;
    if (follow_time_s_ - config_.follow_through_s >= 0.5 * dt) {
      if (config_.recovery_s <= 0.0) {
        phase_ = SwingPhase::kReady;
        active_task_id_.reset();
      } else {
        phase_ = SwingPhase::kRecovery;
        recovery_time_s_ = 0.0;
      }
    }
    return;
  }
  if (phase_ == SwingPhase::kRecovery) {
    recovery_time_s_ += dt;
    if (recovery_time_s_ >= config_.recovery_s) {
      phase_ = SwingPhase::kReady;
      active_task_id_.reset();
    }
  }
}

void SwingLifecycle::Reset() {
  phase_ = SwingPhase::kReady;
  active_task_id_.reset();
  last_engaged_task_id_.reset();
  applied_revision_ = 0;
  active_target_ = ReadyTarget(config_.ready_reference_base_w);
  follow_time_s_ = 0.0;
  recovery_time_s_ = 0.0;
}

}  // namespace a3_pingpong
