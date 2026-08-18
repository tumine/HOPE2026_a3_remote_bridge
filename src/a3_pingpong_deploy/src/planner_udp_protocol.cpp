#include "a3_pingpong/planner_udp_protocol.hpp"

#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <type_traits>

namespace a3_pingpong::udp {
namespace {

constexpr std::size_t kCrcOffset = kPacketSize - sizeof(std::uint32_t);

template <typename T>
void WriteLittle(std::uint8_t*& output, T value) {
  static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>);
  using Bits = std::conditional_t<sizeof(T) == 1, std::uint8_t,
               std::conditional_t<sizeof(T) == 4, std::uint32_t,
                                  std::uint64_t>>;
  Bits bits{};
  std::memcpy(&bits, &value, sizeof(T));
  for (std::size_t index = 0; index < sizeof(T); ++index) {
    *output++ = static_cast<std::uint8_t>(bits >> (index * 8U));
  }
}

template <typename T>
T ReadLittle(const std::uint8_t*& input) {
  static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>);
  using Bits = std::conditional_t<sizeof(T) == 1, std::uint8_t,
               std::conditional_t<sizeof(T) == 4, std::uint32_t,
                                  std::uint64_t>>;
  Bits bits{};
  for (std::size_t index = 0; index < sizeof(T); ++index) {
    bits |= static_cast<Bits>(*input++) << (index * 8U);
  }
  T value{};
  std::memcpy(&value, &bits, sizeof(T));
  return value;
}

template <std::size_t N>
bool AllFinite(const std::array<float, N>& values) {
  for (float value : values) {
    if (!std::isfinite(value)) return false;
  }
  return true;
}

void SetReason(std::string* reason, const char* value) {
  if (reason) *reason = value;
}

}  // namespace

std::uint32_t Crc32(std::span<const std::uint8_t> bytes) {
  std::uint32_t crc = 0xffffffffU;
  for (const std::uint8_t byte : bytes) {
    crc ^= byte;
    for (int bit = 0; bit < 8; ++bit) {
      const std::uint32_t mask = 0U - (crc & 1U);
      crc = (crc >> 1U) ^ (0xedb88320U & mask);
    }
  }
  return ~crc;
}

std::array<std::uint8_t, kPacketSize> Encode(const PlannerPacket& packet) {
  std::array<std::uint8_t, kPacketSize> bytes{};
  std::uint8_t* output = bytes.data();
  WriteLittle(output, kMagic);
  WriteLittle(output, kVersion);
  WriteLittle(output, packet.flags);
  WriteLittle(output, static_cast<std::uint16_t>(kPacketSize));
  WriteLittle(output, packet.session_id);
  WriteLittle(output, packet.packet_sequence);
  WriteLittle(output, packet.pose_sequence);
  for (float value : packet.base_position_w) WriteLittle(output, value);
  for (float value : packet.base_quaternion_wxyz) WriteLittle(output, value);
  WriteLittle(output, packet.task_id);
  WriteLittle(output, packet.task_revision);
  WriteLittle(output, packet.swing_side);
  WriteLittle(output, std::uint8_t{0});
  WriteLittle(output, std::uint8_t{0});
  WriteLittle(output, std::uint8_t{0});
  WriteLittle(output, packet.time_to_strike_us);
  for (float value : packet.target_position_w) WriteLittle(output, value);
  for (float value : packet.target_velocity_w) WriteLittle(output, value);
  const std::uint32_t crc = Crc32(std::span(bytes).first(kCrcOffset));
  WriteLittle(output, crc);
  return bytes;
}

bool Decode(std::span<const std::uint8_t> bytes, PlannerPacket& packet,
            std::string* reason) {
  if (bytes.size() != kPacketSize) {
    SetReason(reason, "wrong packet length");
    return false;
  }
  const std::uint8_t* input = bytes.data();
  if (ReadLittle<std::uint32_t>(input) != kMagic) {
    SetReason(reason, "wrong magic");
    return false;
  }
  if (ReadLittle<std::uint8_t>(input) != kVersion) {
    SetReason(reason, "unsupported version");
    return false;
  }
  packet.flags = ReadLittle<std::uint8_t>(input);
  if ((packet.flags & ~(kPoseValid | kCommandValid)) != 0U) {
    SetReason(reason, "unknown flag bits");
    return false;
  }
  if (ReadLittle<std::uint16_t>(input) != kPacketSize) {
    SetReason(reason, "encoded length mismatch");
    return false;
  }
  const std::uint32_t expected_crc = Crc32(bytes.first(kCrcOffset));
  const std::uint8_t* crc_input = bytes.data() + kCrcOffset;
  if (ReadLittle<std::uint32_t>(crc_input) != expected_crc) {
    SetReason(reason, "CRC32 mismatch");
    return false;
  }

  packet.session_id = ReadLittle<std::uint32_t>(input);
  packet.packet_sequence = ReadLittle<std::uint32_t>(input);
  packet.pose_sequence = ReadLittle<std::uint32_t>(input);
  for (float& value : packet.base_position_w) value = ReadLittle<float>(input);
  for (float& value : packet.base_quaternion_wxyz) value = ReadLittle<float>(input);
  packet.task_id = ReadLittle<std::uint64_t>(input);
  packet.task_revision = ReadLittle<std::uint32_t>(input);
  packet.swing_side = ReadLittle<std::int8_t>(input);
  if (input[0] != 0U || input[1] != 0U || input[2] != 0U) {
    SetReason(reason, "reserved bytes are non-zero");
    return false;
  }
  input += 3;
  packet.time_to_strike_us = ReadLittle<std::int32_t>(input);
  for (float& value : packet.target_position_w) value = ReadLittle<float>(input);
  for (float& value : packet.target_velocity_w) value = ReadLittle<float>(input);

  if ((packet.flags & kPoseValid) != 0U &&
      (!AllFinite(packet.base_position_w) ||
       !AllFinite(packet.base_quaternion_wxyz))) {
    SetReason(reason, "pose contains a non-finite value");
    return false;
  }
  if ((packet.flags & kCommandValid) != 0U &&
      (!AllFinite(packet.target_position_w) ||
       !AllFinite(packet.target_velocity_w) ||
       (packet.swing_side != 1 && packet.swing_side != -1) ||
       packet.time_to_strike_us <= 0)) {
    SetReason(reason, "command fields are invalid or expired");
    return false;
  }
  SetReason(reason, "accepted");
  return true;
}

bool SequenceIsNewer(std::uint32_t candidate, std::uint32_t previous) {
  const std::uint32_t distance = candidate - previous;
  return distance != 0U && distance < 0x80000000U;
}

static_assert(kPacketSize == 96);

}  // namespace a3_pingpong::udp
