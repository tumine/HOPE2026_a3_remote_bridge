#include "a3_pingpong/upper_body_serve.hpp"

#include <Eigen/Core>

#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstring>
#include <iostream>
#include <sstream>
#include <utility>

namespace a3_pingpong {
namespace {

constexpr std::size_t kArmStart = 5;
constexpr std::size_t kRightArmUpperStart = 7;
constexpr std::size_t kLegStart = 19;

void SetReason(std::string* output, std::string value) {
  if (output) *output = std::move(value);
}

double SmoothStep(double value) {
  const double t = std::clamp(value, 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}

double Progress(double elapsed_s, double duration_s, bool smooth) {
  const double linear = duration_s <= 0.0 ? 1.0 : elapsed_s / duration_s;
  return smooth ? SmoothStep(linear) : std::clamp(linear, 0.0, 1.0);
}

double PhaseDuration(const UpperBodyServeConfig& config,
                     UpperBodyServePhase phase) {
  switch (phase) {
    case UpperBodyServePhase::kPrepare: return config.prepare_duration_s;
    case UpperBodyServePhase::kReady: return 0.0;
    case UpperBodyServePhase::kWindup: return config.windup_duration_s;
    case UpperBodyServePhase::kSwing: return config.swing_duration_s;
    case UpperBodyServePhase::kSettle: return config.settle_duration_s;
    case UpperBodyServePhase::kReturn: return config.return_duration_s;
    case UpperBodyServePhase::kIdle:
    case UpperBodyServePhase::kComplete: return 0.0;
  }
  return 0.0;
}

bool FiniteArray(const UpperBodyServeTarget& values) {
  return std::all_of(values.begin(), values.end(),
                     [](double value) { return std::isfinite(value); });
}

}  // namespace

const char* UpperBodyServePhaseName(UpperBodyServePhase phase) noexcept {
  switch (phase) {
    case UpperBodyServePhase::kIdle: return "idle";
    case UpperBodyServePhase::kPrepare: return "prepare";
    case UpperBodyServePhase::kReady: return "ready";
    case UpperBodyServePhase::kWindup: return "windup";
    case UpperBodyServePhase::kSwing: return "swing";
    case UpperBodyServePhase::kSettle: return "settle";
    case UpperBodyServePhase::kReturn: return "return";
    case UpperBodyServePhase::kComplete: return "complete";
  }
  return "unknown";
}

std::optional<UpperBodyServeConfig> NumberedUpperBodyServeConfig(
    int track) noexcept {
  UpperBodyServeConfig config;
  switch (track) {
    case 1:
      // UpperBodyServeConfig defaults are the exact numbered track 1.
      return config;
    case 2:
      config.home_upper = {
          -1.273641478967, 0.163693671459, -0.419944012499,
          0.384910344245, 0.115719728466, -0.895945374305,
          -1.021938992525,
          -0.57, -0.59, 0.65, 0.0, 0.01, 0.04, 0.70};
      config.windup_right = {
          -0.008261136630, -0.508973522035, 0.576516751668,
          -0.282309917701, 0.398359035293, 0.136361784391,
          0.662183771585};
      config.hit_through_right = {
          -0.706465858083, -0.541027532092, 0.530585720180,
          0.238550245950, 0.470531787375, 0.460047591154,
          1.029143336004};
      config.prepare_duration_s = 5.40;
      config.swing_duration_s = 0.06;
      config.release_time_s = -0.18;
      return config;
    case 3:
      config.home_upper = {
          -1.273641478967, 0.163693671459, -0.419944012499,
          0.384910344245, 0.115719728466, -0.895945374305,
          -1.021938992525,
          -0.57, -0.59, 0.65, 0.0, 0.01, 0.04, 0.70};
      config.windup_right = {
          -0.008261136630, -0.508973522035, 0.576516751668,
          -0.282309917701, 0.398359035293, 0.136361784391,
          0.662183771585};
      config.hit_through_right = {
          -0.706465858083, -0.541027532092, 0.530585720180,
          0.238550245950, 0.470531787375, 0.460047591154,
          1.029143336004};
      config.prepare_duration_s = 5.40;
      config.swing_duration_s = 0.10;
      config.release_time_s = -0.17;
      return config;
    case 4:
      config.home_upper = {
          -1.227535374051, 0.021952965340, -0.362970269047,
          0.417042234922, 0.287261482443, -0.914976259720,
          -1.005917653318,
          -0.57, -0.59, 0.65, 0.0, 0.01, 0.04, 0.70};
      config.windup_right = {
          -0.424269722288, -0.278503596243, 0.809542912603,
          -0.163748191625, -0.341913864255, 0.126565193141,
          0.771753460264};
      config.hit_through_right = {
          -0.634716002155, -0.744613944223, 0.451040981275,
          0.321645848425, 0.377963868020, 0.009638231988,
          1.112887970476};
      config.prepare_duration_s = 5.0;
      config.swing_duration_s = 0.13;
      config.release_time_s = -0.15;
      return config;
    default:
      return std::nullopt;
  }
}

UpperBodyServeTrajectory::UpperBodyServeTrajectory(UpperBodyServeConfig config)
    : config_(std::move(config)) {}

void UpperBodyServeTrajectory::Reset() noexcept {
  start_upper_.fill(0.0);
  phase_ = UpperBodyServePhase::kIdle;
  phase_elapsed_s_ = 0.0;
  tick_ = 0;
  release_emitted_ = false;
}

bool UpperBodyServeTrajectory::BeginHoming(
    const robot_io::RobotState& state, std::string* reason) {
  if (state.q.size() != static_cast<Eigen::Index>(kPingpongActionDim) ||
      !state.q.array().isFinite().all()) {
    SetReason(reason, "serve start requires a finite 31-DOF RobotState");
    return false;
  }
  if (!FiniteArray(config_.home_upper) ||
      !std::all_of(config_.windup_right.begin(), config_.windup_right.end(),
                   [](double value) { return std::isfinite(value); }) ||
      !std::all_of(config_.hit_through_right.begin(),
                   config_.hit_through_right.end(),
                   [](double value) { return std::isfinite(value); })) {
    SetReason(reason, "serve trajectory contains a non-finite target");
    return false;
  }
  const std::array<double, 7> durations{
      config_.prepare_duration_s, config_.ready_dwell_s,
      config_.windup_duration_s, config_.swing_duration_s,
      config_.settle_duration_s,
      config_.return_duration_s, config_.receive_transition_s};
  if (!std::all_of(durations.begin(), durations.end(), [](double value) {
        return std::isfinite(value) && value >= 0.0;
      })) {
    SetReason(reason, "serve trajectory durations must be finite and nonnegative");
    return false;
  }
  if (!std::isfinite(config_.release_time_s) ||
      config_.release_time_s < -config_.windup_duration_s ||
      config_.release_time_s > config_.swing_duration_s) {
    SetReason(reason,
              "serve release time must lie within windup/swing interval");
    return false;
  }
  for (std::size_t index = 0; index < start_upper_.size(); ++index) {
    start_upper_[index] =
        state.q[static_cast<Eigen::Index>(kArmStart + index)];
  }
  phase_ = UpperBodyServePhase::kPrepare;
  phase_elapsed_s_ = 0.0;
  tick_ = 0;
  release_emitted_ = false;
  SetReason(reason, "serve homing started; waiting for C then F");
  return true;
}

bool UpperBodyServeTrajectory::Fire(std::string* reason) noexcept {
  if (!ready_to_fire()) {
    SetReason(reason, ready() ? "serve READY dwell is not complete"
                              : "serve is not in READY");
    return false;
  }
  phase_ = UpperBodyServePhase::kWindup;
  phase_elapsed_s_ = 0.0;
  release_emitted_ = false;
  SetReason(reason, "serve fire accepted");
  return true;
}

double UpperBodyServeTrajectory::ready_remaining_s() const noexcept {
  if (!ready()) return 0.0;
  return std::max(0.0, config_.ready_dwell_s - phase_elapsed_s_);
}

void UpperBodyServeTrajectory::AdvancePhase() noexcept {
  switch (phase_) {
    case UpperBodyServePhase::kIdle:
      phase_ = UpperBodyServePhase::kPrepare;
      break;
    case UpperBodyServePhase::kPrepare:
      phase_ = UpperBodyServePhase::kReady;
      break;
    case UpperBodyServePhase::kReady:
      // READY is an explicit latched phase; only Fire() may leave it.
      break;
    case UpperBodyServePhase::kWindup:
      phase_ = UpperBodyServePhase::kSwing;
      break;
    case UpperBodyServePhase::kSwing:
      phase_ = UpperBodyServePhase::kSettle;
      break;
    case UpperBodyServePhase::kSettle:
      phase_ = config_.return_duration_s > 0.0
                   ? UpperBodyServePhase::kReturn
                   : UpperBodyServePhase::kComplete;
      break;
    case UpperBodyServePhase::kReturn:
    case UpperBodyServePhase::kComplete:
      phase_ = UpperBodyServePhase::kComplete;
      break;
  }
  phase_elapsed_s_ = 0.0;
}

bool UpperBodyServeTrajectory::Step(
    const robot_io::RobotState& state, double dt_s,
    UpperBodyServeTarget& output, UpperBodyServeDiagnostics* diagnostics,
    std::string* reason) {
  if (!std::isfinite(dt_s) || dt_s <= 0.0) {
    SetReason(reason, "serve dt must be finite and positive");
    return false;
  }
  if (phase_ == UpperBodyServePhase::kIdle) {
    SetReason(reason, "serve has not been armed with V");
    return false;
  }

  bool boundary_release = false;
  // Move to the next phase before generating this tick. Adjacent phases share
  // their boundary target, so this produces every endpoint without a jump.
  while (phase_ != UpperBodyServePhase::kReady &&
         phase_ != UpperBodyServePhase::kComplete &&
         phase_elapsed_s_ + 1.0e-12 >= PhaseDuration(config_, phase_)) {
    if (phase_ == UpperBodyServePhase::kSwing && !release_emitted_ &&
        phase_elapsed_s_ + 1.0e-12 >= config_.release_time_s) {
      boundary_release = true;
      release_emitted_ = true;
    }
    AdvancePhase();
  }

  const UpperBodyServePhase output_phase = phase_;
  const double output_elapsed_s = phase_elapsed_s_;
  output = config_.home_upper;
  bool release_requested = boundary_release;
  double duration_s = 0.0;

  if (phase_ == UpperBodyServePhase::kPrepare) {
    duration_s = config_.prepare_duration_s;
    const double alpha = Progress(phase_elapsed_s_, duration_s, true);
    for (std::size_t index = 0; index < output.size(); ++index) {
      output[index] = start_upper_[index] +
                      alpha * (config_.home_upper[index] - start_upper_[index]);
    }
  } else if (phase_ == UpperBodyServePhase::kWindup) {
    duration_s = config_.windup_duration_s;
    const double alpha = Progress(phase_elapsed_s_, duration_s, true);
    for (std::size_t index = 0; index < kServeRightArmDim; ++index) {
      const std::size_t upper_index = kRightArmUpperStart + index;
      output[upper_index] = config_.home_upper[upper_index] +
          alpha * (config_.windup_right[index] -
                   config_.home_upper[upper_index]);
    }
  } else if (phase_ == UpperBodyServePhase::kSwing) {
    duration_s = config_.swing_duration_s;
    const double alpha = Progress(phase_elapsed_s_, duration_s, false);
    for (std::size_t index = 0; index < kServeRightArmDim; ++index) {
      output[kRightArmUpperStart + index] = config_.windup_right[index] +
          alpha * (config_.hit_through_right[index] -
                   config_.windup_right[index]);
    }
    if (!release_emitted_ &&
        phase_elapsed_s_ + 1.0e-12 >= config_.release_time_s) {
      release_requested = true;
      release_emitted_ = true;
    }
  } else if (phase_ == UpperBodyServePhase::kSettle) {
    duration_s = config_.settle_duration_s;
    for (std::size_t index = 0; index < kServeRightArmDim; ++index) {
      output[kRightArmUpperStart + index] =
          config_.hit_through_right[index];
    }
  } else if (phase_ == UpperBodyServePhase::kReturn) {
    duration_s = config_.return_duration_s;
    const double alpha = Progress(phase_elapsed_s_, duration_s, true);
    for (std::size_t index = 0; index < kServeRightArmDim; ++index) {
      const std::size_t upper_index = kRightArmUpperStart + index;
      output[upper_index] = config_.hit_through_right[index] +
          alpha * (config_.home_upper[upper_index] -
                   config_.hit_through_right[index]);
    }
  } else if (phase_ == UpperBodyServePhase::kComplete) {
    duration_s = 0.0;
    if (config_.return_duration_s <= 0.0) {
      for (std::size_t index = 0; index < kServeRightArmDim; ++index) {
        output[kRightArmUpperStart + index] =
            config_.hit_through_right[index];
      }
    }
  }

  ++tick_;
  if (phase_ != UpperBodyServePhase::kComplete) {
    phase_elapsed_s_ += dt_s;
  }

  if (diagnostics) {
    diagnostics->phase = output_phase;
    diagnostics->tick = tick_;
    diagnostics->phase_elapsed_s = output_elapsed_s;
    diagnostics->release_requested = release_requested;
    diagnostics->complete = output_phase == UpperBodyServePhase::kComplete;
  }
  SetReason(reason, "valid upper-body serve target");
  return true;
}

const char* GripperActionName(GripperAction action) noexcept {
  switch (action) {
    case GripperAction::kNone: return "none";
    case GripperAction::kOpen: return "open";
    case GripperAction::kClose: return "close";
  }
  return "unknown";
}

std::string BuildGripperCommandJson(
    GripperAction action, const GripperHttpConfig& config) {
  const int left_position = action == GripperAction::kOpen
                                ? config.open_position
                                : config.close_position;
  auto append_claw = [&config](std::ostringstream& body, int position) {
    body << "{\"agi_claw_cmd\":{\"cmd\":" << config.command
         << ",\"pos\":" << position
         << ",\"vel\":" << config.velocity
         << ",\"force\":" << config.force
         << ",\"clamp_method\":" << config.clamp_method
         << ",\"finger_pos\":" << config.finger_position << "}}";
  };
  std::ostringstream body;
  body << "{\"data\":{\"left\":";
  append_claw(body, left_position);
  body << ",\"right\":";
  append_claw(body, config.right_position);
  body << "}}";
  return body.str();
}

GripperHttpClient::GripperHttpClient(GripperHttpConfig config)
    : config_(std::move(config)) {}

GripperHttpClient::~GripperHttpClient() { Stop(); }

bool GripperHttpClient::Start(std::string* reason) {
  if (worker_.joinable()) {
    SetReason(reason, "gripper worker already started");
    return true;
  }
  if (config_.host.empty() || config_.path.empty() || config_.port == 0 ||
      config_.connect_timeout_ms <= 0 || config_.response_timeout_ms <= 0) {
    SetReason(reason, "invalid gripper HTTP configuration");
    return false;
  }
  stop_.store(false, std::memory_order_release);
  worker_ = std::thread(&GripperHttpClient::Worker, this);
  SetReason(reason, "gripper HTTP worker started");
  return true;
}

void GripperHttpClient::Stop() noexcept {
  stop_.store(true, std::memory_order_release);
  const int socket_fd = active_socket_.exchange(-1, std::memory_order_acq_rel);
  if (socket_fd >= 0) ::shutdown(socket_fd, SHUT_RDWR);
  wake_cv_.notify_all();
  if (worker_.joinable()) worker_.join();
  busy_.store(false, std::memory_order_release);
}

std::optional<std::uint64_t> GripperHttpClient::Enqueue(
    GripperAction action) noexcept {
  if (action == GripperAction::kNone || !worker_.joinable() ||
      stop_.load(std::memory_order_acquire)) return std::nullopt;
  bool expected = false;
  if (!busy_.compare_exchange_strong(expected, true,
                                     std::memory_order_acq_rel)) {
    return std::nullopt;
  }
  const auto request_id = next_request_id_.fetch_add(1);
  pending_request_id_.store(request_id, std::memory_order_relaxed);
  pending_.store(action, std::memory_order_release);
  wake_cv_.notify_one();
  return request_id;
}

GripperResult GripperHttpClient::result() const noexcept {
  return GripperResult{
      completed_request_id_.load(std::memory_order_acquire),
      completed_action_.load(std::memory_order_relaxed),
      completed_success_.load(std::memory_order_relaxed)};
}

void GripperHttpClient::Worker() {
  while (!stop_.load(std::memory_order_acquire)) {
    std::unique_lock<std::mutex> lock(wake_mutex_);
    wake_cv_.wait(lock, [this] {
      return stop_.load(std::memory_order_acquire) ||
             pending_.load(std::memory_order_acquire) != GripperAction::kNone;
    });
    lock.unlock();
    if (stop_.load(std::memory_order_acquire)) break;
    const auto action = pending_.exchange(GripperAction::kNone,
                                          std::memory_order_acq_rel);
    const auto request_id = pending_request_id_.load(std::memory_order_relaxed);
    std::string detail;
    const bool success = Send(action, detail);
    completed_action_.store(action, std::memory_order_relaxed);
    completed_success_.store(success, std::memory_order_relaxed);
    completed_request_id_.store(request_id, std::memory_order_release);
    busy_.store(false, std::memory_order_release);
    std::clog << "[gripper_http] id=" << request_id
              << " action=" << GripperActionName(action)
              << " success=" << (success ? "yes" : "no")
              << " detail=" << detail << '\n';
  }
}

bool GripperHttpClient::Send(GripperAction action,
                             std::string& detail) noexcept {
  addrinfo hints{};
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  addrinfo* addresses = nullptr;
  const std::string port = std::to_string(config_.port);
  const int gai = ::getaddrinfo(config_.host.c_str(), port.c_str(), &hints,
                                &addresses);
  if (gai != 0) {
    detail = std::string("getaddrinfo: ") + ::gai_strerror(gai);
    return false;
  }

  int fd = -1;
  for (addrinfo* address = addresses; address; address = address->ai_next) {
    fd = ::socket(address->ai_family, address->ai_socktype,
                  address->ai_protocol);
    if (fd < 0) continue;
    const int original_flags = ::fcntl(fd, F_GETFL, 0);
    if (original_flags < 0 || ::fcntl(fd, F_SETFL, original_flags | O_NONBLOCK) < 0) {
      ::close(fd);
      fd = -1;
      continue;
    }
    int rc = ::connect(fd, address->ai_addr, address->ai_addrlen);
    if (rc < 0 && errno == EINPROGRESS) {
      fd_set write_set;
      FD_ZERO(&write_set);
      FD_SET(fd, &write_set);
      timeval timeout{config_.connect_timeout_ms / 1000,
                      (config_.connect_timeout_ms % 1000) * 1000};
      rc = ::select(fd + 1, nullptr, &write_set, nullptr, &timeout);
      if (rc > 0) {
        int socket_error = 0;
        socklen_t length = sizeof(socket_error);
        if (::getsockopt(fd, SOL_SOCKET, SO_ERROR, &socket_error, &length) < 0 ||
            socket_error != 0) {
          rc = -1;
        } else {
          rc = 0;
        }
      } else {
        // select() returns zero on timeout. Keep it distinct from connect()'s
        // zero success result so a timed-out socket is never used.
        rc = -1;
      }
    }
    if (rc == 0 && ::fcntl(fd, F_SETFL, original_flags) == 0) break;
    ::close(fd);
    fd = -1;
  }
  ::freeaddrinfo(addresses);
  if (fd < 0) {
    detail = "connect failed or timed out";
    return false;
  }
  active_socket_.store(fd, std::memory_order_release);
  timeval io_timeout{config_.response_timeout_ms / 1000,
                     (config_.response_timeout_ms % 1000) * 1000};
  ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &io_timeout, sizeof(io_timeout));
  ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &io_timeout, sizeof(io_timeout));

  const std::string payload = BuildGripperCommandJson(action, config_);
  std::ostringstream request_stream;
  request_stream << "POST " << config_.path << " HTTP/1.1\r\n"
                 << "Host: " << config_.host << ':' << config_.port << "\r\n"
                 << "Content-Type: application/json\r\n"
                 << "Timeout: 60000\r\n"
                 << "Content-Length: " << payload.size() << "\r\n"
                 << "Connection: close\r\n\r\n" << payload;
  const std::string request = request_stream.str();
  std::size_t sent = 0;
  while (sent < request.size()) {
    const ssize_t count = ::send(fd, request.data() + sent,
                                 request.size() - sent, MSG_NOSIGNAL);
    if (count <= 0) {
      detail = std::string("send: ") + std::strerror(errno);
      active_socket_.store(-1, std::memory_order_release);
      ::close(fd);
      return false;
    }
    sent += static_cast<std::size_t>(count);
  }
  std::string response;
  std::array<char, 2048> buffer{};
  while (response.size() < 64 * 1024) {
    const ssize_t count = ::recv(fd, buffer.data(), buffer.size(), 0);
    if (count == 0) break;
    if (count < 0) {
      detail = std::string("recv: ") + std::strerror(errno);
      active_socket_.store(-1, std::memory_order_release);
      ::close(fd);
      return false;
    }
    response.append(buffer.data(), static_cast<std::size_t>(count));
  }
  active_socket_.store(-1, std::memory_order_release);
  ::close(fd);
  const auto line_end = response.find("\r\n");
  const std::string status = response.substr(0, line_end);
  const bool http_ok = status.find(" 2") != std::string::npos;
  std::string compact = response;
  compact.erase(std::remove_if(compact.begin(), compact.end(),
                               [](unsigned char value) {
                                 return std::isspace(value) != 0;
                               }),
                compact.end());
  const bool code_ok = compact.find("\"code\":0") != std::string::npos ||
                       compact.find("\"code\":\"0\"") != std::string::npos;
  detail = status.empty() ? "empty HTTP response" : status;
  return http_ok && code_ok;
}

FullBodyServeComposer::FullBodyServeComposer(PingpongCommandGains gains)
    : gains_(std::move(gains)) {}

bool FullBodyServeComposer::Build(
    const robot_io::RobotState& state,
    const UpperBodyServeTarget& upper_body,
    const std::optional<LowerBodyServeTarget>& lower_body,
    robot_io::RobotCommand& command, std::string* reason) const {
  if (state.q.size() != static_cast<Eigen::Index>(kPingpongActionDim) ||
      !state.q.array().isFinite().all() || !FiniteArray(upper_body)) {
    SetReason(reason, "serve composition requires finite 31-DOF state and upper target");
    return false;
  }

  std::array<double, kPingpongActionDim> q_des{};
  for (std::size_t index = 0; index < q_des.size(); ++index) {
    q_des[index] = state.q[static_cast<Eigen::Index>(index)];
  }
  for (std::size_t index = 0; index < upper_body.size(); ++index) {
    q_des[kArmStart + index] = upper_body[index];
  }

  if (lower_body) {
    if (!std::all_of(lower_body->begin(), lower_body->end(),
                     [](double value) { return std::isfinite(value); })) {
      SetReason(reason, "lower-body serve target contains a non-finite value");
      return false;
    }
    for (std::size_t index = 0; index < 3; ++index) {
      q_des[index] = (*lower_body)[index];
    }
    for (std::size_t index = 0; index < 12; ++index) {
      q_des[kLegStart + index] = (*lower_body)[3 + index];
    }
  }

  if (!BuildPositionCommand(q_des, gains_, command, reason)) return false;
  SetReason(reason, lower_body ? "valid full-body serve command"
                               : "valid upper-body serve hold command");
  return true;
}

}  // namespace a3_pingpong
