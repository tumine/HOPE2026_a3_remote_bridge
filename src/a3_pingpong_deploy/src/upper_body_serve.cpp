#include "a3_pingpong/upper_body_serve.hpp"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <utility>

namespace a3_pingpong {
namespace {

constexpr std::size_t kArmStart = 5;
constexpr std::size_t kRightArmUpperStart = 7;
constexpr std::size_t kLegStart = 19;

void SetReason(std::string* output, std::string value) {
  if (output) *output = std::move(value);
}

double SmoothStep(double value) {
  const double t = std::clamp(value, 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}

double Progress(double elapsed_s, double duration_s, bool smooth) {
  const double linear = duration_s <= 0.0 ? 1.0 : elapsed_s / duration_s;
  return smooth ? SmoothStep(linear) : std::clamp(linear, 0.0, 1.0);
}

double PhaseDuration(const UpperBodyServeConfig& config,
                     UpperBodyServePhase phase) {
  switch (phase) {
    case UpperBodyServePhase::kPrepare: return config.prepare_duration_s;
    case UpperBodyServePhase::kWindup: return config.windup_duration_s;
    case UpperBodyServePhase::kSwing: return config.swing_duration_s;
    case UpperBodyServePhase::kSettle: return config.settle_duration_s;
    case UpperBodyServePhase::kReturn: return config.return_duration_s;
    case UpperBodyServePhase::kIdle:
    case UpperBodyServePhase::kComplete: return 0.0;
  }
  return 0.0;
}

bool FiniteArray(const UpperBodyServeTarget& values) {
  return std::all_of(values.begin(), values.end(),
                     [](double value) { return std::isfinite(value); });
}

}  // namespace

const char* UpperBodyServePhaseName(UpperBodyServePhase phase) noexcept {
  switch (phase) {
    case UpperBodyServePhase::kIdle: return "idle";
    case UpperBodyServePhase::kPrepare: return "prepare";
    case UpperBodyServePhase::kWindup: return "windup";
    case UpperBodyServePhase::kSwing: return "swing";
    case UpperBodyServePhase::kSettle: return "settle";
    case UpperBodyServePhase::kReturn: return "return";
    case UpperBodyServePhase::kComplete: return "complete";
  }
  return "unknown";
}

UpperBodyServeTrajectory::UpperBodyServeTrajectory(UpperBodyServeConfig config)
    : config_(std::move(config)) {}

void UpperBodyServeTrajectory::Reset() noexcept {
  start_upper_.fill(0.0);
  phase_ = UpperBodyServePhase::kIdle;
  phase_elapsed_s_ = 0.0;
  tick_ = 0;
  release_emitted_ = false;
}

bool UpperBodyServeTrajectory::Start(const robot_io::RobotState& state,
                                     std::string* reason) {
  if (state.q.size() != static_cast<Eigen::Index>(kPingpongActionDim) ||
      !state.q.array().isFinite().all()) {
    SetReason(reason, "serve start requires a finite 31-DOF RobotState");
    return false;
  }
  if (!FiniteArray(config_.home_upper) ||
      !std::all_of(config_.windup_right.begin(), config_.windup_right.end(),
                   [](double value) { return std::isfinite(value); }) ||
      !std::all_of(config_.hit_through_right.begin(),
                   config_.hit_through_right.end(),
                   [](double value) { return std::isfinite(value); })) {
    SetReason(reason, "serve trajectory contains a non-finite target");
    return false;
  }
  const std::array<double, 5> durations{
      config_.prepare_duration_s, config_.windup_duration_s,
      config_.swing_duration_s, config_.settle_duration_s,
      config_.return_duration_s};
  if (!std::all_of(durations.begin(), durations.end(), [](double value) {
        return std::isfinite(value) && value >= 0.0;
      })) {
    SetReason(reason, "serve trajectory durations must be finite and nonnegative");
    return false;
  }
  for (std::size_t index = 0; index < start_upper_.size(); ++index) {
    start_upper_[index] =
        state.q[static_cast<Eigen::Index>(kArmStart + index)];
  }
  phase_ = UpperBodyServePhase::kPrepare;
  phase_elapsed_s_ = 0.0;
  tick_ = 0;
  release_emitted_ = false;
  SetReason(reason, "upper-body serve started");
  return true;
}

void UpperBodyServeTrajectory::AdvancePhase() noexcept {
  switch (phase_) {
    case UpperBodyServePhase::kIdle:
      phase_ = UpperBodyServePhase::kPrepare;
      break;
    case UpperBodyServePhase::kPrepare:
      phase_ = UpperBodyServePhase::kWindup;
      break;
    case UpperBodyServePhase::kWindup:
      phase_ = UpperBodyServePhase::kSwing;
      break;
    case UpperBodyServePhase::kSwing:
      phase_ = UpperBodyServePhase::kSettle;
      break;
    case UpperBodyServePhase::kSettle:
      phase_ = UpperBodyServePhase::kReturn;
      break;
    case UpperBodyServePhase::kReturn:
    case UpperBodyServePhase::kComplete:
      phase_ = UpperBodyServePhase::kComplete;
      break;
  }
  phase_elapsed_s_ = 0.0;
}

bool UpperBodyServeTrajectory::Step(
    const robot_io::RobotState& state, double dt_s,
    UpperBodyServeTarget& output, UpperBodyServeDiagnostics* diagnostics,
    std::string* reason) {
  if (!std::isfinite(dt_s) || dt_s <= 0.0) {
    SetReason(reason, "serve dt must be finite and positive");
    return false;
  }
  if (phase_ == UpperBodyServePhase::kIdle && !Start(state, reason)) {
    return false;
  }

  // Move to the next phase before generating this tick. Adjacent phases share
  // their boundary target, so this produces every endpoint without a jump.
  while (phase_ != UpperBodyServePhase::kComplete &&
         phase_elapsed_s_ + 1.0e-12 >= PhaseDuration(config_, phase_)) {
    AdvancePhase();
  }

  const UpperBodyServePhase output_phase = phase_;
  output = config_.home_upper;
  bool release_requested = false;
  double duration_s = 0.0;

  if (phase_ == UpperBodyServePhase::kPrepare) {
    duration_s = config_.prepare_duration_s;
    const double alpha = Progress(phase_elapsed_s_, duration_s, true);
    for (std::size_t index = 0; index < output.size(); ++index) {
      output[index] = start_upper_[index] +
                      alpha * (config_.home_upper[index] - start_upper_[index]);
    }
  } else if (phase_ == UpperBodyServePhase::kWindup) {
    duration_s = config_.windup_duration_s;
    const double alpha = Progress(phase_elapsed_s_, duration_s, true);
    for (std::size_t index = 0; index < kServeRightArmDim; ++index) {
      const std::size_t upper_index = kRightArmUpperStart + index;
      output[upper_index] = config_.home_upper[upper_index] +
          alpha * (config_.windup_right[index] -
                   config_.home_upper[upper_index]);
    }
  } else if (phase_ == UpperBodyServePhase::kSwing) {
    duration_s = config_.swing_duration_s;
    const double alpha = Progress(phase_elapsed_s_, duration_s, false);
    for (std::size_t index = 0; index < kServeRightArmDim; ++index) {
      output[kRightArmUpperStart + index] = config_.windup_right[index] +
          alpha * (config_.hit_through_right[index] -
                   config_.windup_right[index]);
    }
    if (!release_emitted_) {
      release_requested = true;
      release_emitted_ = true;
    }
  } else if (phase_ == UpperBodyServePhase::kSettle) {
    duration_s = config_.settle_duration_s;
    for (std::size_t index = 0; index < kServeRightArmDim; ++index) {
      output[kRightArmUpperStart + index] =
          config_.hit_through_right[index];
    }
  } else if (phase_ == UpperBodyServePhase::kReturn) {
    duration_s = config_.return_duration_s;
    const double alpha = Progress(phase_elapsed_s_, duration_s, true);
    for (std::size_t index = 0; index < kServeRightArmDim; ++index) {
      const std::size_t upper_index = kRightArmUpperStart + index;
      output[upper_index] = config_.hit_through_right[index] +
          alpha * (config_.home_upper[upper_index] -
                   config_.hit_through_right[index]);
    }
  } else if (phase_ == UpperBodyServePhase::kComplete) {
    duration_s = 0.0;
  }

  ++tick_;
  if (phase_ != UpperBodyServePhase::kComplete) {
    phase_elapsed_s_ += dt_s;
  }

  if (diagnostics) {
    diagnostics->phase = output_phase;
    diagnostics->tick = tick_;
    diagnostics->release_requested = release_requested;
    diagnostics->complete = output_phase == UpperBodyServePhase::kComplete;
  }
  SetReason(reason, "valid upper-body serve target");
  return true;
}

FullBodyServeComposer::FullBodyServeComposer(PingpongCommandGains gains)
    : gains_(std::move(gains)) {}

bool FullBodyServeComposer::Build(
    const robot_io::RobotState& state,
    const UpperBodyServeTarget& upper_body,
    const std::optional<LowerBodyServeTarget>& lower_body,
    robot_io::RobotCommand& command, std::string* reason) const {
  if (state.q.size() != static_cast<Eigen::Index>(kPingpongActionDim) ||
      !state.q.array().isFinite().all() || !FiniteArray(upper_body)) {
    SetReason(reason, "serve composition requires finite 31-DOF state and upper target");
    return false;
  }

  std::array<double, kPingpongActionDim> q_des{};
  for (std::size_t index = 0; index < q_des.size(); ++index) {
    q_des[index] = state.q[static_cast<Eigen::Index>(index)];
  }
  for (std::size_t index = 0; index < upper_body.size(); ++index) {
    q_des[kArmStart + index] = upper_body[index];
  }

  if (lower_body) {
    if (!std::all_of(lower_body->begin(), lower_body->end(),
                     [](double value) { return std::isfinite(value); })) {
      SetReason(reason, "lower-body serve target contains a non-finite value");
      return false;
    }
    for (std::size_t index = 0; index < 3; ++index) {
      q_des[index] = (*lower_body)[index];
    }
    for (std::size_t index = 0; index < 12; ++index) {
      q_des[kLegStart + index] = (*lower_body)[3 + index];
    }
  }

  if (!BuildPositionCommand(q_des, gains_, command, reason)) return false;
  SetReason(reason, lower_body ? "valid full-body serve command"
                               : "valid upper-body serve hold command");
  return true;
}

}  // namespace a3_pingpong
