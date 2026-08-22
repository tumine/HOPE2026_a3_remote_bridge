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
  kServeTrack1,
  kServeTrack2,
  kServeTrack3,
  kServeTrack4,
  kServeClose,
  kServeFire,
  kGripperOpen,
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
  kRejectedNeedMotion,
  kRejectedServeState,
  kRejectedGripperBusy,
  kRejectedGripperNotClosed,
  kRejectedExternalNotReady,
  kServePending,
  kGripperRequested,
  kServeFireRequested,
  kReceiveRequested,
  kRejectedQuitWhileActive,
  kStatusRequested,
  kHelpRequested,
  kQuitRequested,
  kIgnored,
};

ManualKey ParseManualKey(char key) noexcept;
int ManualServeTrackNumber(ManualKey key) noexcept;
const char* ManualModeName(ManualMode mode) noexcept;

// On-robot control sequence: P=passive, S=pd_stand, M=motion. V/C/F/G are
// consumed by ObservationProbe while mode remains kMotion so the receive
// policy continues to own waist and legs during the upper-body serve.
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
