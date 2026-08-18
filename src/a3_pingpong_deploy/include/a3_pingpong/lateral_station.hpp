#pragma once

#include <array>
#include <cstdint>

namespace a3_pingpong {

struct LateralStationConfig {
  bool enabled{true};
  std::array<double, 2> base_target_y_range{-0.35, 0.6625};
  double forehand_reach_y{-0.64};
  double backhand_reach_y{-0.10};
};

// Frozen model_50000 moving-station geometry from
// config/hope_pingpong_runtime.yaml.
LateralStationConfig Model50000LateralStationConfig();

// Invert the training command geometry. X stays at the nominal startup
// station; Y follows the active strike target within the trained offset range.
bool DeriveLateralBaseTarget(
    const std::array<double, 3>& target_position_w,
    std::int8_t swing_side,
    const std::array<double, 2>& nominal_station_xy,
    const LateralStationConfig& config,
    std::array<double, 2>& base_target_xy) noexcept;

}  // namespace a3_pingpong
