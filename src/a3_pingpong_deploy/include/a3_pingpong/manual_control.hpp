#pragma once

#include <cstdint>

namespace a3_pingpong {

enum class ManualMode {
  kPassive,
  kPdStand,
  kMotion,
  kUpperBodyServe,
  kHalted,
};

enum class ManualKey {
  kPassive,
  kPdStand,
  kMotion,
  kUpperBodyServe,
  kHalt,
  kStatus,
  kHelp,
  kQuit,
  kUnknown,
};

enum class ManualActionResult {
  kAccepted,
  kRejectedNeedPdStand,
  kRejectedServeDisabled,
  kRejectedQuitWhileActive,
  kStatusRequested,
  kHelpRequested,
  kQuitRequested,
  kIgnored,
};

ManualKey ParseManualKey(char key) noexcept;
const char* ManualModeName(ManualMode mode) noexcept;

// On-robot control sequence: P=passive, S=pd_stand, then M=motion or
// V=upper-body serve. Active modes are deliberately unreachable until the
// official 150-tick PD_STAND interpolation has completed.
class ManualControl {
 public:
  ManualActionResult Apply(ManualKey key) noexcept;
  void SetPdStandReady(bool ready) noexcept;
  void CompleteUpperBodyServe() noexcept;

  ManualMode mode() const noexcept { return mode_; }
  bool pd_stand_ready() const noexcept { return pd_stand_ready_; }
  bool policy_enabled() const noexcept { return mode_ == ManualMode::kMotion; }
  bool upper_body_serve_enabled() const noexcept {
    return mode_ == ManualMode::kUpperBodyServe;
  }
  std::uint64_t epoch() const noexcept { return epoch_; }

 private:
  void Enter(ManualMode mode) noexcept;

  ManualMode mode_{ManualMode::kPassive};
  bool pd_stand_ready_{false};
  std::uint64_t epoch_{0};
};

}  // namespace a3_pingpong
