#include <joint_msgs/msg/joint_command.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/u_int32.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("a3_hdu_command_dry_relay");

  const auto qos = rclcpp::QoS(rclcpp::KeepLast(1))
                       .best_effort()
                       .durability_volatile();
  auto command_pub = node->create_publisher<joint_msgs::msg::JointCommand>(
      "/a3_internal/joint_command_dry_run", qos);
  auto ack_pub = node->create_publisher<std_msgs::msg::UInt32>(
      "/a3_remote/joint_command_ack", qos);

  std::atomic<std::uint64_t> forwarded{0};
  std::atomic<std::uint64_t> returned{0};
  auto command_sub = node->create_subscription<joint_msgs::msg::JointCommand>(
      "/a3_remote/joint_command_dry_run", qos,
      [command_pub, &forwarded](
          const joint_msgs::msg::JointCommand::SharedPtr msg) {
        if (!msg || msg->joints.size() != 31) return;
        command_pub->publish(*msg);
        forwarded.fetch_add(1, std::memory_order_relaxed);
      });
  auto ack_sub = node->create_subscription<std_msgs::msg::UInt32>(
      "/a3_internal/joint_command_ack", qos,
      [ack_pub, &returned](const std_msgs::msg::UInt32::SharedPtr msg) {
        if (!msg) return;
        ack_pub->publish(*msg);
        returned.fetch_add(1, std::memory_order_relaxed);
      });

  auto log_timer = node->create_wall_timer(
      std::chrono::seconds(1), [&]() {
        RCLCPP_INFO(node->get_logger(),
                    "dry relay: pc_to_mdu=%llu mdu_to_pc_ack=%llu "
                    "body_drive_publishers=0",
                    static_cast<unsigned long long>(forwarded.load()),
                    static_cast<unsigned long long>(returned.load()));
      });

  RCLCPP_INFO(node->get_logger(),
              "dry command relay started; isolated topics only, "
              "body_drive_publishers=0");
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}

