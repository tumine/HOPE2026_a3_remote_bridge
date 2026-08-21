#pragma once

#include "a3_pingpong/pingpong_action_adapter.hpp"
#include "robot_io/robot_io_backend.hpp"

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

namespace a3_pingpong {

constexpr std::size_t kServeUpperBodyDim = 14;
constexpr std::size_t kServeRightArmDim = 7;
constexpr std::size_t kServeLowerBodyDim = 15;  // waist 3 + legs 12

using UpperBodyServeTarget = std::array<double, kServeUpperBodyDim>;
using LowerBodyServeTarget = std::array<double, kServeLowerBodyDim>;

enum class UpperBodyServePhase {
  kIdle,
  kPrepare,
  kReady,
  kWindup,
  kSwing,
  kSettle,
  kReturn,
  kComplete,
};

const char* UpperBodyServePhaseName(UpperBodyServePhase phase) noexcept;

struct UpperBodyServeConfig {
  // Canonical A3 arm order: left arm [0..6], right arm [7..13].
  UpperBodyServeTarget home_upper{
      -1.10, 0.00, 0.00, 0.90, 1.57, -1.57, 0.00,
      -0.57, -0.59, 0.65, 0.00, 0.01, 0.04, 0.70};
  std::array<double, kServeRightArmDim> windup_right{
      -0.33, -0.55, 0.71, -0.07, -0.20, -0.22, 0.73};
  std::array<double, kServeRightArmDim> hit_through_right{
      -0.55, -0.59, 0.56, 0.03, 0.01, 0.05, 0.73};
  // Frozen from Serve_A3_leg_model/config/a3_lower_body.yaml and shared with
  // the MuJoCo V->C->F visualization.
  double prepare_duration_s{5.0};
  double ready_dwell_s{1.0};
  double windup_duration_s{0.50};
  double swing_duration_s{0.10};
  double release_time_s{0.10};
  double settle_duration_s{1.0};
  double return_duration_s{1.0};
  double receive_transition_s{0.60};
};

struct UpperBodyServeDiagnostics {
  UpperBodyServePhase phase{UpperBodyServePhase::kIdle};
  std::uint64_t tick{0};
  double phase_elapsed_s{0.0};
  bool release_requested{false};
  bool complete{false};
};

// Explicit V->C->F upper-body trajectory. V calls BeginHoming(), C is owned by
// the gripper client, and F calls Fire() only after the close result is
// confirmed. No command is published by this class.
class UpperBodyServeTrajectory {
 public:
  explicit UpperBodyServeTrajectory(
      UpperBodyServeConfig config = UpperBodyServeConfig{});

  void Reset() noexcept;
  bool BeginHoming(const robot_io::RobotState& state,
                   std::string* reason = nullptr);
  bool Fire(std::string* reason = nullptr) noexcept;
  bool Step(const robot_io::RobotState& state, double dt_s,
            UpperBodyServeTarget& output,
            UpperBodyServeDiagnostics* diagnostics = nullptr,
            std::string* reason = nullptr);

  UpperBodyServePhase phase() const noexcept { return phase_; }
  bool complete() const noexcept {
    return phase_ == UpperBodyServePhase::kComplete;
  }
  bool ready() const noexcept { return phase_ == UpperBodyServePhase::kReady; }
  bool ready_to_fire() const noexcept {
    return ready() && phase_elapsed_s_ >= config_.ready_dwell_s;
  }
  double ready_remaining_s() const noexcept;
  const UpperBodyServeConfig& config() const noexcept { return config_; }

 private:
  void AdvancePhase() noexcept;

  UpperBodyServeConfig config_;
  UpperBodyServeTarget start_upper_{};
  UpperBodyServePhase phase_{UpperBodyServePhase::kIdle};
  double phase_elapsed_s_{0.0};
  std::uint64_t tick_{0};
  bool release_emitted_{false};
};

enum class GripperAction : int { kNone = 0, kOpen, kClose };

const char* GripperActionName(GripperAction action) noexcept;

struct GripperHttpConfig {
  std::string host{"10.42.10.12"};
  std::uint16_t port{56422};
  std::string path{"/rpc/aimdk.protocol.HalHandService/SetHandCommand"};
  int open_position{4096};
  int close_position{1200};
  int right_position{0};
  int command{0};
  int velocity{20};
  int force{20};
  int clamp_method{2};
  int finger_position{0};
  int connect_timeout_ms{3000};
  int response_timeout_ms{65000};
};

// Builds the exact dual-claw frame accepted by the deployed A3_T3D0
// HalHandService. Both sides are mandatory even when only the left claw moves.
std::string BuildGripperCommandJson(
    GripperAction action, const GripperHttpConfig& config);

struct GripperResult {
  std::uint64_t request_id{0};
  GripperAction action{GripperAction::kNone};
  bool success{false};
};

// Non-blocking wrapper for the proven HOPE HalHandService HTTP endpoint.
// The keyboard/control thread only enqueues; all DNS/socket work happens in
// the worker. Exactly one request may be outstanding at a time.
class GripperHttpClient {
 public:
  explicit GripperHttpClient(GripperHttpConfig config = {});
  ~GripperHttpClient();

  GripperHttpClient(const GripperHttpClient&) = delete;
  GripperHttpClient& operator=(const GripperHttpClient&) = delete;

  bool Start(std::string* reason = nullptr);
  void Stop() noexcept;
  std::optional<std::uint64_t> Enqueue(GripperAction action) noexcept;
  bool busy() const noexcept { return busy_.load(std::memory_order_acquire); }
  GripperResult result() const noexcept;
  const GripperHttpConfig& config() const noexcept { return config_; }

 private:
  void Worker();
  bool Send(GripperAction action, std::string& detail) noexcept;

  GripperHttpConfig config_;
  std::atomic<GripperAction> pending_{GripperAction::kNone};
  std::atomic<bool> busy_{false};
  std::atomic<bool> stop_{false};
  std::atomic<int> active_socket_{-1};
  std::atomic<std::uint64_t> next_request_id_{1};
  std::atomic<std::uint64_t> pending_request_id_{0};
  std::atomic<std::uint64_t> completed_request_id_{0};
  std::atomic<GripperAction> completed_action_{GripperAction::kNone};
  std::atomic<bool> completed_success_{false};
  std::mutex wake_mutex_;
  std::condition_variable wake_cv_;
  std::thread worker_;
};

// Maps the upper-body trajectory and an optional lower-body policy target into
// the exact A3 31-DOF command layout. Without a lower-body target, waist,
// neck, and legs hold the latest measured positions. A future lower-body
// policy supplies waist[0..2] + legs[0..11] through LowerBodyServeTarget.
class FullBodyServeComposer {
 public:
  explicit FullBodyServeComposer(
      PingpongCommandGains gains = A3PdStandGains());

  bool Build(const robot_io::RobotState& state,
             const UpperBodyServeTarget& upper_body,
             const std::optional<LowerBodyServeTarget>& lower_body,
             robot_io::RobotCommand& command,
             std::string* reason = nullptr) const;

 private:
  PingpongCommandGains gains_;
};

}  // namespace a3_pingpong
