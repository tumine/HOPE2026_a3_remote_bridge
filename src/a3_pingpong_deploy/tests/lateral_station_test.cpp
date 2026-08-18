#include "a3_pingpong/lateral_station.hpp"

#include <array>
#include <cmath>

#define CHECK(condition)               \
  do {                                 \
    if (!(condition)) return __LINE__; \
  } while (false)

int main() {
  const auto config = a3_pingpong::Model50000LateralStationConfig();
  const std::array<double, 2> nominal{-0.5, -0.7625};
  std::array<double, 2> target{};

  CHECK(a3_pingpong::DeriveLateralBaseTarget(
      {-0.2, -1.20, 0.30}, 1, nominal, config, target));
  CHECK(std::abs(target[0] - nominal[0]) < 1.0e-12);
  CHECK(std::abs(target[1] - (-0.56)) < 1.0e-12);

  CHECK(a3_pingpong::DeriveLateralBaseTarget(
      {0.0, -0.20, 0.30}, -1, nominal, config, target));
  CHECK(std::abs(target[1] - (-0.10)) < 1.0e-12);

  // Clamp both ends to the exact station offset range learned in training.
  CHECK(a3_pingpong::DeriveLateralBaseTarget(
      {0.0, -3.0, 0.30}, 1, nominal, config, target));
  CHECK(std::abs(target[1] - (-1.1125)) < 1.0e-12);
  CHECK(a3_pingpong::DeriveLateralBaseTarget(
      {0.0, 3.0, 0.30}, -1, nominal, config, target));
  CHECK(std::abs(target[1] - (-0.10)) < 1.0e-12);
  return 0;
}
