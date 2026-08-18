#include "a3_pingpong/planner_udp_receiver.hpp"

#include "a3_pingpong/planner_udp_protocol.hpp"

#include <arpa/inet.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstring>
#include <span>
#include <utility>

namespace a3_pingpong::udp {
namespace {

void SetReason(std::string* output, std::string value) {
  if (output) *output = std::move(value);
}

}  // namespace

PlannerUdpReceiver::PlannerUdpReceiver(PlannerInputMailbox& mailbox,
                                       ReceiverOptions options)
    : mailbox_(mailbox), options_(std::move(options)) {}

PlannerUdpReceiver::~PlannerUdpReceiver() { Stop(); }

bool PlannerUdpReceiver::Start(std::string* reason) {
  if (running_.load()) {
    SetReason(reason, "already running");
    return true;
  }
  in_addr bind_address{};
  in_addr source_address{};
  if (::inet_pton(AF_INET, options_.bind_address.c_str(), &bind_address) != 1) {
    SetReason(reason, "invalid UDP bind IPv4 address");
    return false;
  }
  if (::inet_pton(AF_INET, options_.allowed_source_address.c_str(),
                  &source_address) != 1) {
    SetReason(reason, "invalid allowed-source IPv4 address");
    return false;
  }
  allowed_source_network_order_ = source_address.s_addr;
  socket_fd_ = ::socket(AF_INET, SOCK_DGRAM | SOCK_CLOEXEC, 0);
  if (socket_fd_ < 0) {
    SetReason(reason, std::string("socket: ") + std::strerror(errno));
    return false;
  }
  int receive_buffer = 64 * 1024;
  (void)::setsockopt(socket_fd_, SOL_SOCKET, SO_RCVBUF, &receive_buffer,
                     sizeof(receive_buffer));
  sockaddr_in local{};
  local.sin_family = AF_INET;
  local.sin_port = htons(options_.port);
  local.sin_addr = bind_address;
  if (::bind(socket_fd_, reinterpret_cast<const sockaddr*>(&local),
             sizeof(local)) != 0) {
    SetReason(reason, std::string("bind ") + options_.bind_address + ":" +
                          std::to_string(options_.port) + ": " +
                          std::strerror(errno));
    ::close(socket_fd_);
    socket_fd_ = -1;
    return false;
  }
  running_.store(true);
  thread_ = std::thread(&PlannerUdpReceiver::Run, this);
  SetReason(reason, "started");
  return true;
}

void PlannerUdpReceiver::Stop() {
  running_.store(false);
  if (thread_.joinable()) thread_.join();
  if (socket_fd_ >= 0) {
    ::close(socket_fd_);
    socket_fd_ = -1;
  }
}

ReceiverStats PlannerUdpReceiver::stats() const {
  return ReceiverStats{
      datagrams_.load(),          accepted_packets_.load(),
      rejected_packets_.load(),  out_of_order_packets_.load(),
      accepted_commands_.load(), rejected_commands_.load(),
      accepted_poses_.load(),    rejected_poses_.load()};
}

std::string PlannerUdpReceiver::last_error() const {
  std::lock_guard<std::mutex> lock(error_mutex_);
  return last_error_;
}

void PlannerUdpReceiver::Run() {
  bool have_session = false;
  bool have_packet_sequence = false;
  bool have_pose_sequence = false;
  std::uint32_t session_id = 0;
  std::uint32_t packet_sequence = 0;
  std::uint32_t pose_sequence = 0;
  std::array<std::uint8_t, 256> buffer{};

  while (running_.load()) {
    pollfd descriptor{socket_fd_, POLLIN, 0};
    const int poll_result = ::poll(&descriptor, 1, 100);
    if (!running_.load()) break;
    if (poll_result < 0) {
      if (errno == EINTR) continue;
      std::lock_guard<std::mutex> lock(error_mutex_);
      last_error_ = std::string("poll: ") + std::strerror(errno);
      rejected_packets_.fetch_add(1);
      continue;
    }
    if (poll_result == 0 || (descriptor.revents & POLLIN) == 0) continue;

    sockaddr_in remote{};
    socklen_t remote_size = sizeof(remote);
    const ssize_t received = ::recvfrom(
        socket_fd_, buffer.data(), buffer.size(), 0,
        reinterpret_cast<sockaddr*>(&remote), &remote_size);
    if (received < 0) {
      if (errno == EINTR) continue;
      std::lock_guard<std::mutex> lock(error_mutex_);
      last_error_ = std::string("recvfrom: ") + std::strerror(errno);
      rejected_packets_.fetch_add(1);
      continue;
    }
    datagrams_.fetch_add(1);
    if (remote.sin_family != AF_INET ||
        remote.sin_addr.s_addr != allowed_source_network_order_) {
      rejected_packets_.fetch_add(1);
      std::lock_guard<std::mutex> lock(error_mutex_);
      last_error_ = "datagram from disallowed source address";
      continue;
    }

    PlannerPacket packet;
    std::string decode_reason;
    if (!Decode(std::span(buffer.data(), static_cast<std::size_t>(received)),
                packet, &decode_reason)) {
      rejected_packets_.fetch_add(1);
      std::lock_guard<std::mutex> lock(error_mutex_);
      last_error_ = std::move(decode_reason);
      continue;
    }

    if (!have_session || packet.session_id != session_id) {
      session_id = packet.session_id;
      have_session = true;
      have_packet_sequence = false;
      have_pose_sequence = false;
      mailbox_.Reset();
    }
    if (have_packet_sequence &&
        !SequenceIsNewer(packet.packet_sequence, packet_sequence)) {
      out_of_order_packets_.fetch_add(1);
      continue;
    }
    packet_sequence = packet.packet_sequence;
    have_packet_sequence = true;
    accepted_packets_.fetch_add(1);
    const auto now = SteadyClock::now();

    if ((packet.flags & kPoseValid) != 0U &&
        (!have_pose_sequence ||
         SequenceIsNewer(packet.pose_sequence, pose_sequence))) {
      BasePoseInput input;
      input.frame_id = options_.expected_frame;
      input.position_w = {packet.base_position_w[0],
                          packet.base_position_w[1],
                          packet.base_position_w[2]};
      input.quaternion_wxyz = {packet.base_quaternion_wxyz[0],
                               packet.base_quaternion_wxyz[1],
                               packet.base_quaternion_wxyz[2],
                               packet.base_quaternion_wxyz[3]};
      std::string update_reason;
      if (mailbox_.UpdateBasePose(std::move(input), now, &update_reason) ==
          InputUpdateResult::kAccepted) {
        pose_sequence = packet.pose_sequence;
        have_pose_sequence = true;
        accepted_poses_.fetch_add(1);
      } else {
        rejected_poses_.fetch_add(1);
        std::lock_guard<std::mutex> lock(error_mutex_);
        last_error_ = std::move(update_reason);
      }
    }

    if ((packet.flags & kCommandValid) != 0U) {
      RacketTargetInput input;
      input.frame_id = options_.expected_frame;
      input.task_id = packet.task_id;
      input.task_revision = packet.task_revision;
      input.swing_side = packet.swing_side;
      input.time_to_strike_s = packet.time_to_strike_us * 1.0e-6;
      input.position_w = {packet.target_position_w[0],
                          packet.target_position_w[1],
                          packet.target_position_w[2]};
      input.velocity_w = {packet.target_velocity_w[0],
                          packet.target_velocity_w[1],
                          packet.target_velocity_w[2]};
      std::string update_reason;
      const auto result = mailbox_.UpdateCommand(std::move(input), 0, now,
                                                  &update_reason);
      if (result == InputUpdateResult::kAccepted) {
        accepted_commands_.fetch_add(1);
      } else if (result != InputUpdateResult::kDuplicate) {
        rejected_commands_.fetch_add(1);
        std::lock_guard<std::mutex> lock(error_mutex_);
        last_error_ = std::move(update_reason);
      }
    }
  }
}

}  // namespace a3_pingpong::udp
