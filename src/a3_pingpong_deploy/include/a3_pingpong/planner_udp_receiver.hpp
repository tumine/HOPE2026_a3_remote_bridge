#pragma once

#include "a3_pingpong/planner_input.hpp"

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>

namespace a3_pingpong::udp {

struct ReceiverOptions {
  std::string bind_address{"192.168.1.100"};
  std::string allowed_source_address{"192.168.1.11"};
  std::uint16_t port{15001};
  std::string expected_frame{"hope_table"};
};

struct ReceiverStats {
  std::uint64_t datagrams{0};
  std::uint64_t accepted_packets{0};
  std::uint64_t rejected_packets{0};
  std::uint64_t out_of_order_packets{0};
  std::uint64_t accepted_commands{0};
  std::uint64_t rejected_commands{0};
  std::uint64_t accepted_poses{0};
  std::uint64_t rejected_poses{0};
  std::uint64_t current_packet_age_ns{0};
  std::uint64_t max_packet_gap_ns{0};
};

class PlannerUdpReceiver {
 public:
  PlannerUdpReceiver(PlannerInputMailbox& mailbox, ReceiverOptions options);
  ~PlannerUdpReceiver();

  PlannerUdpReceiver(const PlannerUdpReceiver&) = delete;
  PlannerUdpReceiver& operator=(const PlannerUdpReceiver&) = delete;

  bool Start(std::string* reason = nullptr);
  void Stop();
  ReceiverStats stats() const;
  std::string last_error() const;

 private:
  void Run();

  PlannerInputMailbox& mailbox_;
  ReceiverOptions options_;
  int socket_fd_{-1};
  std::uint32_t allowed_source_network_order_{0};
  std::atomic<bool> running_{false};
  std::thread thread_;
  std::atomic<std::uint64_t> datagrams_{0};
  std::atomic<std::uint64_t> accepted_packets_{0};
  std::atomic<std::uint64_t> rejected_packets_{0};
  std::atomic<std::uint64_t> out_of_order_packets_{0};
  std::atomic<std::uint64_t> accepted_commands_{0};
  std::atomic<std::uint64_t> rejected_commands_{0};
  std::atomic<std::uint64_t> accepted_poses_{0};
  std::atomic<std::uint64_t> rejected_poses_{0};
  std::atomic<std::uint64_t> last_packet_at_ns_{0};
  std::atomic<std::uint64_t> max_packet_gap_ns_{0};
  mutable std::mutex error_mutex_;
  std::string last_error_;
};

}  // namespace a3_pingpong::udp
