#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

namespace a3_pingpong::udp {

inline constexpr std::uint32_t kMagic = 0x50503341U;  // "A3PP" on the wire.
inline constexpr std::uint8_t kVersion = 1;
inline constexpr std::size_t kPacketSize = 96;
inline constexpr std::uint8_t kPoseValid = 1U << 0U;
inline constexpr std::uint8_t kCommandValid = 1U << 1U;

struct PlannerPacket {
  std::uint8_t flags{0};
  std::uint32_t session_id{0};
  std::uint32_t packet_sequence{0};
  std::uint32_t pose_sequence{0};
  std::array<float, 3> base_position_w{};
  std::array<float, 4> base_quaternion_wxyz{};
  std::uint64_t task_id{0};
  std::uint32_t task_revision{0};
  std::int8_t swing_side{0};
  std::int32_t time_to_strike_us{0};
  std::array<float, 3> target_position_w{};
  std::array<float, 3> target_velocity_w{};
};

std::array<std::uint8_t, kPacketSize> Encode(const PlannerPacket& packet);

bool Decode(std::span<const std::uint8_t> bytes, PlannerPacket& packet,
            std::string* reason = nullptr);

std::uint32_t Crc32(std::span<const std::uint8_t> bytes);

// Handles uint32 wrap-around and rejects equal sequence numbers.
bool SequenceIsNewer(std::uint32_t candidate, std::uint32_t previous);

}  // namespace a3_pingpong::udp
