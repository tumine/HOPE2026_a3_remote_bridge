#include "a3_pingpong/lateral_station.hpp"

#include <algorithm>
#include <cmath>

namespace a3_pingpong {

LateralStationConfig Model50000LateralStationConfig() {
  return LateralStationConfig{};
}

bool DeriveLateralBaseTarget(
    const std::array<double, 3>& target_position_w,
    std::int8_t swing_side,
    const std::array<double, 2>& nominal_station_xy,
    const LateralStationConfig& config,
    std::array<double, 2>& base_target_xy) noexcept {
  if ((swing_side != 1 && swing_side != -1) ||
      !std::isfinite(target_position_w[0]) ||
      !std::isfinite(target_position_w[1]) ||
      !std::isfinite(target_position_w[2]) ||
      !std::isfinite(nominal_station_xy[0]) ||
      !std::isfinite(nominal_station_xy[1]) ||
      !std::isfinite(config.base_target_y_range[0]) ||
      !std::isfinite(config.base_target_y_range[1]) ||
      config.base_target_y_range[0] > config.base_target_y_range[1] ||
      !std::isfinite(config.forehand_reach_y) ||
      !std::isfinite(config.backhand_reach_y)) {
    return false;
  }

  base_target_xy = nominal_station_xy;
  if (!config.enabled) return true;

  const double reach_y =
      swing_side >= 0 ? config.forehand_reach_y : config.backhand_reach_y;
  const double desired_offset_y =
      target_position_w[1] - nominal_station_xy[1] - reach_y;
  base_target_xy[1] +=
      std::clamp(desired_offset_y, config.base_target_y_range[0],
                 config.base_target_y_range[1]);
  return true;
}

}  // namespace a3_pingpong
