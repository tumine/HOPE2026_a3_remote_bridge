#include "a3_pingpong/upper_body_serve.hpp"

#include <Eigen/Core>

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <chrono>
#include <cmath>
#include <optional>
#include <string>
#include <thread>

#define CHECK(condition)                \
  do {                                  \
    if (!(condition)) return __LINE__;  \
  } while (false)

namespace {

robot_io::RobotState State31() {
  robot_io::RobotState state;
  state.q = Eigen::VectorXd(31);
  state.dq = Eigen::VectorXd::Zero(31);
  state.tau_est = Eigen::VectorXd::Zero(31);
  for (Eigen::Index index = 0; index < state.q.size(); ++index) {
    state.q[index] = 0.01 * static_cast<double>(index);
  }
  return state;
}

bool Near(double lhs, double rhs) {
  return std::abs(lhs - rhs) < 1.0e-12;
}

bool GripperHttpRoundTrip() {
  const int listener = ::socket(AF_INET, SOCK_STREAM, 0);
  if (listener < 0) return false;
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  address.sin_port = 0;
  if (::bind(listener, reinterpret_cast<sockaddr*>(&address),
             sizeof(address)) != 0 ||
      ::listen(listener, 1) != 0) {
    ::close(listener);
    return false;
  }
  socklen_t address_length = sizeof(address);
  if (::getsockname(listener, reinterpret_cast<sockaddr*>(&address),
                    &address_length) != 0) {
    ::close(listener);
    return false;
  }

  std::thread server([listener] {
    const int client = ::accept(listener, nullptr, nullptr);
    if (client >= 0) {
      std::array<char, 2048> request{};
      (void)::recv(client, request.data(), request.size(), 0);
      constexpr char response[] =
          "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
          "Content-Length: 23\r\nConnection: close\r\n\r\n"
          "{\"header\":{\"code\":\"0\"}}";
      (void)::send(client, response, sizeof(response) - 1, MSG_NOSIGNAL);
      ::close(client);
    }
    ::close(listener);
  });

  a3_pingpong::GripperHttpConfig config;
  config.host = "127.0.0.1";
  config.port = ntohs(address.sin_port);
  config.connect_timeout_ms = 500;
  config.response_timeout_ms = 500;
  a3_pingpong::GripperHttpClient client(config);
  std::string reason;
  const bool started = client.Start(&reason);
  const auto request_id = client.Enqueue(a3_pingpong::GripperAction::kClose);
  for (int attempt = 0; attempt < 200 && client.busy(); ++attempt) {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  const auto result = client.result();
  client.Stop();
  server.join();
  return started && request_id && result.request_id == *request_id &&
         result.action == a3_pingpong::GripperAction::kClose && result.success;
}

}  // namespace

int main() {
  using namespace a3_pingpong;

  const auto track1 = NumberedUpperBodyServeConfig(1);
  const auto track2 = NumberedUpperBodyServeConfig(2);
  const auto track3 = NumberedUpperBodyServeConfig(3);
  const auto track4 = NumberedUpperBodyServeConfig(4);
  CHECK(track1 && track2 && track3 && track4);
  CHECK(!NumberedUpperBodyServeConfig(0));
  CHECK(!NumberedUpperBodyServeConfig(5));
  CHECK(Near(track1->home_upper[0], -1.191947170859));
  CHECK(Near(track2->swing_duration_s, 0.06));
  CHECK(Near(track3->release_time_s, -0.17));
  CHECK(Near(track4->hit_through_right[6], 1.112887970476));
  CHECK(Near(track1->settle_duration_s, 0.05));
  CHECK(Near(track1->return_duration_s, 0.0));
  CHECK(Near(track1->receive_transition_s, 0.20));

  GripperHttpConfig gripper_config;
  CHECK(BuildGripperCommandJson(GripperAction::kClose, gripper_config) ==
        "{\"data\":{\"left\":{\"agi_claw_cmd\":{\"cmd\":0,\"pos\":1200,"
        "\"vel\":20,\"force\":20,\"clamp_method\":2,\"finger_pos\":0}},"
        "\"right\":{\"agi_claw_cmd\":{\"cmd\":0,\"pos\":0,\"vel\":20,"
        "\"force\":20,\"clamp_method\":2,\"finger_pos\":0}}}}");
  CHECK(BuildGripperCommandJson(GripperAction::kOpen, gripper_config).find(
            "\"cmd\":0,\"pos\":4096,\"vel\":20,\"force\":20") !=
        std::string::npos);
  CHECK(GripperHttpRoundTrip());

  auto state = State31();
  UpperBodyServeConfig config;
  config.prepare_duration_s = 0.04;
  config.ready_dwell_s = 0.04;
  config.windup_duration_s = 0.04;
  config.swing_duration_s = 0.04;
  config.release_time_s = 0.04;
  config.settle_duration_s = 0.04;
  config.return_duration_s = 0.04;
  UpperBodyServeTrajectory trajectory(config);

  UpperBodyServeTarget upper{};
  UpperBodyServeDiagnostics diagnostics;
  std::string reason;
  CHECK(!trajectory.Step(state, 0.02, upper, &diagnostics, &reason));
  CHECK(trajectory.BeginHoming(state, &reason));
  CHECK(trajectory.Step(state, 0.02, upper, &diagnostics, &reason));
  CHECK(diagnostics.phase == UpperBodyServePhase::kPrepare);
  for (std::size_t index = 0; index < upper.size(); ++index) {
    CHECK(Near(upper[index], state.q[static_cast<Eigen::Index>(5 + index)]));
  }

  int release_count = 0;
  bool saw_windup = false;
  bool saw_swing = false;
  bool saw_settle = false;
  bool saw_return = false;
  for (int tick = 0; tick < 4; ++tick) {
    CHECK(trajectory.Step(state, 0.02, upper, &diagnostics, &reason));
  }
  CHECK(trajectory.ready());
  CHECK(trajectory.ready_to_fire());
  CHECK(trajectory.Fire(&reason));
  for (int tick = 0; tick < 30 && !diagnostics.complete; ++tick) {
    CHECK(trajectory.Step(state, 0.02, upper, &diagnostics, &reason));
    saw_windup |= diagnostics.phase == UpperBodyServePhase::kWindup;
    saw_swing |= diagnostics.phase == UpperBodyServePhase::kSwing;
    saw_settle |= diagnostics.phase == UpperBodyServePhase::kSettle;
    saw_return |= diagnostics.phase == UpperBodyServePhase::kReturn;
    if (diagnostics.release_requested) ++release_count;
  }
  CHECK(diagnostics.complete);
  CHECK(saw_windup && saw_swing && saw_settle && saw_return);
  CHECK(release_count == 1);
  for (std::size_t index = 0; index < upper.size(); ++index) {
    CHECK(Near(upper[index], config.home_upper[index]));
  }

  // The combined loop uses the numbered configuration without a Return
  // phase: after 50 ms at hit-through it completes there, ready for the
  // receiver's direct 200 ms interpolation to the live policy target.
  UpperBodyServeTrajectory direct(*track1);
  CHECK(direct.BeginHoming(state, &reason));
  while (!direct.ready()) {
    CHECK(direct.Step(state, 0.02, upper, &diagnostics, &reason));
  }
  CHECK(direct.Step(state, 0.50, upper, &diagnostics, &reason));
  CHECK(direct.ready_to_fire());
  CHECK(direct.Fire(&reason));
  bool direct_saw_return = false;
  for (int tick = 0; tick < 100 && !diagnostics.complete; ++tick) {
    CHECK(direct.Step(state, 0.02, upper, &diagnostics, &reason));
    direct_saw_return |= diagnostics.phase == UpperBodyServePhase::kReturn;
  }
  CHECK(diagnostics.complete);
  CHECK(!direct_saw_return);
  for (std::size_t index = 0; index < kServeRightArmDim; ++index) {
    CHECK(Near(upper[7 + index], track1->hit_through_right[index]));
  }

  FullBodyServeComposer composer;
  robot_io::RobotCommand command;
  CHECK(composer.Build(state, upper, std::nullopt, command, &reason));
  CHECK(command.q_des.size() == 31);
  CHECK(Near(command.q_des[3], state.q[3]));
  CHECK(Near(command.q_des[4], state.q[4]));
  CHECK(Near(command.q_des[5], upper[0]));
  CHECK(Near(command.q_des[18], upper[13]));
  CHECK(Near(command.q_des[19], state.q[19]));
  CHECK(Near(command.q_des[30], state.q[30]));

  LowerBodyServeTarget lower{};
  for (std::size_t index = 0; index < lower.size(); ++index) {
    lower[index] = -0.02 * static_cast<double>(index + 1);
  }
  CHECK(composer.Build(state, upper, lower, command, &reason));
  CHECK(Near(command.q_des[0], lower[0]));
  CHECK(Near(command.q_des[2], lower[2]));
  CHECK(Near(command.q_des[19], lower[3]));
  CHECK(Near(command.q_des[30], lower[14]));
  CHECK(Near(command.q_des[12], upper[7]));

  state.q[0] = std::nan("");
  CHECK(!composer.Build(state, upper, std::nullopt, command, &reason));
  return 0;
}
