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
      -1.191947170859, 0.160059961944, -0.413680271815,
      0.362461292346, 0.176261448817, -0.944081736121,
      -0.988417746825,
      -0.57, -0.59, 0.65, 0.00, 0.01, 0.04, 0.70};
  std::array<double, kServeRightArmDim> windup_right{
      0.002435028829, -0.471091091169, 0.593553488245,
      -0.132633432142, 0.354979439390, 0.004177040662,
      0.700382116774};
  std::array<double, kServeRightArmDim> hit_through_right{
      -0.932245543163, -0.513271234914, 0.793750783000,
      0.501023846375, 0.405425286567, 0.357815376051,
      1.208649592179};
  // Track 1 is the only implicit V default. Tracks 1-5 are copied exactly
  // from Serve_A3_leg_model/tracks. The combined loop shortens only Home and
  // post-impact recovery; windup/swing/release remain robot-tuned values.
  double prepare_duration_s{1.35};
  double ready_dwell_s{0.50};
  double windup_duration_s{0.50};
  double swing_duration_s{0.12};
  double release_time_s{-0.15};
  double settle_duration_s{0.05};
  double return_duration_s{0.0};
  double receive_transition_s{0.20};
};

// Returns one of the five built-in reference tracks for simulation/tests.
// Real MDU V/1-5 control uses the hot-reloaded YAML profiles below.
std::optional<UpperBodyServeConfig> NumberedUpperBodyServeConfig(
    int track) noexcept;

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
  // Used for READY-to-READY numbered switching: interpolation starts from the
  // previous commanded Home, never from feedback, preserving target/torque
  // continuity exactly like the standalone lower-body serve controller.
  bool BeginHomingFromTarget(const UpperBodyServeTarget& start_upper,
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
  int close_position{350};
  int right_position{0};
  int command{0};
  int velocity{20};
  int force{20};
  int clamp_method{2};
  int finger_position{0};
  int connect_timeout_ms{3000};
  int response_timeout_ms{65000};
};

// Complete runtime-tunable serve profile. The MDU reloads the selected YAML
// when V/1-5 is pressed. Gripper positions and arm gains are also refreshed
// immediately before C/F/G, so those values can be commissioned without
// stopping the receive policy.
struct UpperBodyServeProfile {
  UpperBodyServeConfig trajectory;
  // Final waist-pitch target while serving. The MDU interpolates from the
  // live policy command during Home and returns to policy ownership during
  // receive_transition_s. Negative is backward, away from the table.
  double waist_pitch_target_rad{-0.06981317007977318};  // -4 degrees
  std::array<double, kServeUpperBodyDim> arm_kp{
      200.0, 150.0, 80.0, 100.0, 100.0, 80.0, 80.0,
      200.0, 150.0, 80.0, 100.0, 100.0, 80.0, 80.0};
  std::array<double, kServeUpperBodyDim> arm_kd{
      2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
      2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
  int gripper_open_position{4096};
  int gripper_close_position{350};
  int gripper_right_position{0};
};

std::optional<UpperBodyServeProfile> LoadUpperBodyServeProfile(
    const std::string& path, std::string* reason = nullptr) noexcept;

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
  std::optional<std::uint64_t> Enqueue(GripperAction action,
                                       int left_position,
                                       int right_position) noexcept;
  bool busy() const noexcept { return busy_.load(std::memory_order_acquire); }
  GripperResult result() const noexcept;
  const GripperHttpConfig& config() const noexcept { return config_; }

 private:
  void Worker();
  bool Send(GripperAction action, int left_position, int right_position,
            std::string& detail) noexcept;

  GripperHttpConfig config_;
  std::atomic<GripperAction> pending_{GripperAction::kNone};
  std::atomic<bool> busy_{false};
  std::atomic<bool> stop_{false};
  std::atomic<int> active_socket_{-1};
  std::atomic<std::uint64_t> next_request_id_{1};
  std::atomic<std::uint64_t> pending_request_id_{0};
  std::atomic<int> pending_left_position_{0};
  std::atomic<int> pending_right_position_{0};
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
