#include "a3_pingpong/onnx_actor.hpp"

#include <cmath>
#include <fstream>
#include <limits>

#define CHECK(condition)              \
  do {                                \
    if (!(condition)) return __LINE__; \
  } while (false)

#ifndef A3_POLICY_ONNX_PATH
#error "A3_POLICY_ONNX_PATH must be defined"
#endif

#ifndef A3_OBSERVATION_GOLDEN_PATH
#error "A3_OBSERVATION_GOLDEN_PATH must be defined"
#endif

#ifndef A3_ACTION_GOLDEN_PATH
#error "A3_ACTION_GOLDEN_PATH must be defined"
#endif

int main() {
  a3_pingpong::PingpongObservation observation{};
  std::ifstream observation_fixture(A3_OBSERVATION_GOLDEN_PATH);
  CHECK(observation_fixture.good());
  for (float& value : observation) {
    double parsed = 0.0;
    CHECK(static_cast<bool>(observation_fixture >> parsed));
    value = static_cast<float>(parsed);
  }

  a3_pingpong::OnnxActor actor(A3_POLICY_ONNX_PATH);
  CHECK(actor.input_name() == "observation");
  CHECK(actor.output_name() == "raw_action");

  a3_pingpong::PingpongAction raw_action{};
  std::string reason;
  CHECK(actor.Run(observation, raw_action, &reason));

  std::ifstream action_fixture(A3_ACTION_GOLDEN_PATH);
  CHECK(action_fixture.good());
  for (const float value : raw_action) {
    double expected = 0.0;
    CHECK(static_cast<bool>(action_fixture >> expected));
    CHECK(std::abs(static_cast<double>(value) - expected) <= 2.0e-5);
  }
  double extra = 0.0;
  CHECK(!(action_fixture >> extra));

  observation[0] = std::numeric_limits<float>::quiet_NaN();
  CHECK(!actor.Run(observation, raw_action, &reason));
  return 0;
}
