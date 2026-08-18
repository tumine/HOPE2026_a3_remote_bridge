#include "a3_pingpong/manual_control.hpp"

namespace a3_pingpong {

ManualKey ParseManualKey(char key) noexcept {
  switch (key) {
    case 'p':
    case 'P': return ManualKey::kPassive;
    case 's':
    case 'S': return ManualKey::kPdStand;
    case 'm':
    case 'M': return ManualKey::kMotion;
    case 'v':
    case 'V': return ManualKey::kUpperBodyServe;
    case 'x':
    case 'X': return ManualKey::kHalt;
    case 'i':
    case 'I': return ManualKey::kStatus;
    case 'h':
    case 'H': return ManualKey::kHelp;
    case 'q':
    case 'Q': return ManualKey::kQuit;
    default: return ManualKey::kUnknown;
  }
}

const char* ManualModeName(ManualMode mode) noexcept {
  switch (mode) {
    case ManualMode::kPassive: return "passive";
    case ManualMode::kPdStand: return "pd_stand";
    case ManualMode::kMotion: return "motion";
    case ManualMode::kUpperBodyServe: return "upper_body_serve";
    case ManualMode::kHalted: return "halted";
  }
  return "unknown";
}

void ManualControl::Enter(ManualMode mode) noexcept {
  if (mode_ != mode) {
    mode_ = mode;
    ++epoch_;
  }
}

ManualActionResult ManualControl::Apply(ManualKey key) noexcept {
  switch (key) {
    case ManualKey::kPassive:
      pd_stand_ready_ = false;
      Enter(ManualMode::kPassive);
      return ManualActionResult::kAccepted;
    case ManualKey::kPdStand:
      pd_stand_ready_ = false;
      if (mode_ == ManualMode::kPdStand) {
        ++epoch_;
      } else {
        Enter(ManualMode::kPdStand);
      }
      return ManualActionResult::kAccepted;
    case ManualKey::kMotion:
      if (mode_ != ManualMode::kPdStand || !pd_stand_ready_) {
        return ManualActionResult::kRejectedNeedPdStand;
      }
      pd_stand_ready_ = false;
      Enter(ManualMode::kMotion);
      return ManualActionResult::kAccepted;
    case ManualKey::kUpperBodyServe:
      if (mode_ != ManualMode::kPdStand || !pd_stand_ready_) {
        return ManualActionResult::kRejectedNeedPdStand;
      }
      pd_stand_ready_ = false;
      Enter(ManualMode::kUpperBodyServe);
      return ManualActionResult::kAccepted;
    case ManualKey::kHalt:
      pd_stand_ready_ = false;
      Enter(ManualMode::kHalted);
      return ManualActionResult::kAccepted;
    case ManualKey::kStatus:
      return ManualActionResult::kStatusRequested;
    case ManualKey::kHelp:
      return ManualActionResult::kHelpRequested;
    case ManualKey::kQuit:
      return mode_ == ManualMode::kPassive
                 ? ManualActionResult::kQuitRequested
                 : ManualActionResult::kRejectedQuitWhileActive;
    case ManualKey::kUnknown:
      return ManualActionResult::kIgnored;
  }
  return ManualActionResult::kIgnored;
}

void ManualControl::SetPdStandReady(bool ready) noexcept {
  pd_stand_ready_ = mode_ == ManualMode::kPdStand && ready;
}

void ManualControl::CompleteUpperBodyServe() noexcept {
  if (mode_ != ManualMode::kUpperBodyServe) return;
  pd_stand_ready_ = false;
  Enter(ManualMode::kPdStand);
}

}  // namespace a3_pingpong
