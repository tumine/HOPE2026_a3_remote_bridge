#include "a3_pingpong/planner_udp_protocol.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <string>

int main() {
  using namespace a3_pingpong::udp;
  PlannerPacket input;
  input.flags = kPoseValid | kCommandValid;
  input.session_id = 0x10203040U;
  input.packet_sequence = 0xfffffff0U;
  input.pose_sequence = 77;
  input.base_position_w = {-0.25F, 0.5F, 0.75F};
  input.base_quaternion_wxyz = {1.0F, 0.0F, 0.1F, -0.1F};
  input.task_id = 0x0102030405060708ULL;
  input.task_revision = 29;
  input.swing_side = -1;
  input.time_to_strike_us = 345678;
  input.target_position_w = {1.25F, -2.5F, 3.75F};
  input.target_velocity_w = {-4.0F, 5.0F, -6.0F};

  const auto encoded = Encode(input);
  static_assert(encoded.size() == 96);
  assert(encoded[0] == 'A' && encoded[1] == '3' &&
         encoded[2] == 'P' && encoded[3] == 'P');

  PlannerPacket output;
  std::string reason;
  assert(Decode(encoded, output, &reason));
  assert(reason == "accepted");
  assert(output.flags == input.flags);
  assert(output.session_id == input.session_id);
  assert(output.packet_sequence == input.packet_sequence);
  assert(output.pose_sequence == input.pose_sequence);
  assert(output.base_position_w == input.base_position_w);
  assert(output.base_quaternion_wxyz == input.base_quaternion_wxyz);
  assert(output.task_id == input.task_id);
  assert(output.task_revision == input.task_revision);
  assert(output.swing_side == input.swing_side);
  assert(output.time_to_strike_us == input.time_to_strike_us);
  assert(output.target_position_w == input.target_position_w);
  assert(output.target_velocity_w == input.target_velocity_w);

  auto corrupt = encoded;
  corrupt[24] ^= 0x01U;
  assert(!Decode(corrupt, output, &reason));
  assert(reason == "CRC32 mismatch");

  auto wrong_version = encoded;
  wrong_version[4] = 2;
  const auto crc = Crc32(std::span(wrong_version).first(92));
  wrong_version[92] = static_cast<std::uint8_t>(crc);
  wrong_version[93] = static_cast<std::uint8_t>(crc >> 8U);
  wrong_version[94] = static_cast<std::uint8_t>(crc >> 16U);
  wrong_version[95] = static_cast<std::uint8_t>(crc >> 24U);
  assert(!Decode(wrong_version, output, &reason));
  assert(reason == "unsupported version");

  PlannerPacket invalid = input;
  invalid.target_velocity_w[1] = NAN;
  assert(!Decode(Encode(invalid), output, &reason));
  assert(reason == "command fields are invalid or expired");

  assert(SequenceIsNewer(2, 1));
  assert(!SequenceIsNewer(1, 1));
  assert(!SequenceIsNewer(1, 2));
  assert(SequenceIsNewer(0, 0xffffffffU));
  return 0;
}
