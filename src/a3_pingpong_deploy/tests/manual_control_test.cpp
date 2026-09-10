#include "a3_pingpong/manual_control.hpp"

#define CHECK(condition)                \
  do {                                  \
    if (!(condition)) return __LINE__;  \
  } while (false)

int main() {
  using namespace a3_pingpong;

  CHECK(ParseManualKey('s') == ManualKey::kPdStand);
  CHECK(ParseManualKey('M') == ManualKey::kMotion);
  CHECK(ParseManualKey('v') == ManualKey::kUpperBodyServe);
  CHECK(ParseManualKey('1') == ManualKey::kServeTrack1);
  CHECK(ParseManualKey('4') == ManualKey::kServeTrack4);
  CHECK(ManualServeTrackNumber(ParseManualKey('1')) == 1);
  CHECK(ManualServeTrackNumber(ParseManualKey('2')) == 2);
  CHECK(ManualServeTrackNumber(ParseManualKey('3')) == 3);
  CHECK(ManualServeTrackNumber(ParseManualKey('4')) == 4);
  CHECK(ManualServeTrackNumber(ParseManualKey('5')) == 5);
  CHECK(ManualServeTrackNumber(ParseManualKey('V')) == 0);
  CHECK(ParseManualKey('C') == ManualKey::kServeClose);
  CHECK(ParseManualKey('f') == ManualKey::kServeFire);
  CHECK(ParseManualKey('G') == ManualKey::kGripperOpen);
  CHECK(ParseManualKey('r') == ManualKey::kServeCancel);
  CHECK(ParseManualKey('?') == ManualKey::kUnknown);

  ManualControl control;
  CHECK(control.mode() == ManualMode::kPassive);
  CHECK(control.Apply(ManualKey::kMotion) ==
        ManualActionResult::kRejectedNeedPdStand);
  CHECK(!control.policy_enabled());

  CHECK(control.Apply(ManualKey::kPdStand) == ManualActionResult::kAccepted);
  CHECK(control.mode() == ManualMode::kPdStand);
  CHECK(!control.pd_stand_ready());
  CHECK(control.Apply(ManualKey::kMotion) ==
        ManualActionResult::kRejectedNeedPdStand);

  control.SetPdStandReady(true);
  CHECK(control.pd_stand_ready());
  const auto stand_epoch = control.epoch();
  CHECK(control.Apply(ManualKey::kPdStand) == ManualActionResult::kAccepted);
  CHECK(control.epoch() == stand_epoch + 1);
  CHECK(!control.pd_stand_ready());
  control.SetPdStandReady(true);
  CHECK(control.Apply(ManualKey::kMotion) == ManualActionResult::kAccepted);
  CHECK(control.mode() == ManualMode::kMotion);
  CHECK(control.policy_enabled());
  CHECK(control.Apply(ManualKey::kQuit) ==
        ManualActionResult::kRejectedQuitWhileActive);

  CHECK(control.Apply(ManualKey::kPassive) == ManualActionResult::kAccepted);
  CHECK(control.mode() == ManualMode::kPassive);
  CHECK(control.Apply(ManualKey::kQuit) == ManualActionResult::kQuitRequested);

  CHECK(control.Apply(ManualKey::kPdStand) == ManualActionResult::kAccepted);
  control.SetPdStandReady(true);
  CHECK(control.Apply(ManualKey::kUpperBodyServe) ==
        ManualActionResult::kAccepted);
  CHECK(control.mode() == ManualMode::kUpperBodyServe);
  CHECK(control.upper_body_serve_enabled());
  control.CompleteUpperBodyServe();
  CHECK(control.mode() == ManualMode::kPdStand);
  CHECK(!control.pd_stand_ready());
  control.SetPdStandReady(true);
  CHECK(control.Apply(ManualKey::kHalt) == ManualActionResult::kAccepted);
  CHECK(control.mode() == ManualMode::kHalted);
  CHECK(!control.pd_stand_ready());
  CHECK(!control.policy_enabled());
  return 0;
}
