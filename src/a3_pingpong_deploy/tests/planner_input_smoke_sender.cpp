#include <geometry_msgs/msg/pose_stamped.hpp>
#include <hope_msgs/msg/racket_command.hpp>
#include <rclcpp/rclcpp.hpp>

#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <thread>

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("a3_planner_input_smoke_sender");
  auto command_publisher = node->create_publisher<hope_msgs::msg::RacketCommand>(
      "/racket/command",
      rclcpp::QoS(rclcpp::KeepLast(10)).reliable().durability_volatile());
  auto pose_publisher =
      node->create_publisher<geometry_msgs::msg::PoseStamped>(
          "/a3_mocap/pelvis_pose",
          rclcpp::QoS(rclcpp::KeepLast(5))
              .best_effort()
              .durability_volatile());

  const auto discovery_deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (std::chrono::steady_clock::now() < discovery_deadline &&
         (command_publisher->get_subscription_count() == 0 ||
          pose_publisher->get_subscription_count() == 0)) {
    rclcpp::spin_some(node);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }

  const auto command_matches = command_publisher->get_subscription_count();
  const auto pose_matches = pose_publisher->get_subscription_count();
  std::cout << "matched subscriptions: command=" << command_matches
            << " pose=" << pose_matches << std::endl;
  if (command_matches == 0 || pose_matches == 0) {
    std::cerr << "planner receiver was not discovered within 5 seconds"
              << std::endl;
    rclcpp::shutdown();
    return 2;
  }

  for (std::uint32_t revision = 0; revision < 30; ++revision) {
    const auto stamp = node->now();
    geometry_msgs::msg::PoseStamped pose;
    pose.header.stamp = stamp;
    pose.header.frame_id = "hope_table";
    pose.pose.position.x = -0.5;
    pose.pose.position.y = -0.7625;
    pose.pose.position.z = 0.3064;
    pose.pose.orientation.w = 1.0;
    pose_publisher->publish(pose);

    hope_msgs::msg::RacketCommand command;
    command.header.stamp = stamp;
    command.header.frame_id = "hope_table";
    command.task_id = 1;
    command.task_revision = revision;
    command.swing_side = hope_msgs::msg::RacketCommand::FOREHAND;
    command.position.x = 0.2;
    command.position.y = -1.3;
    command.position.z = 0.35;
    command.velocity.x = 2.0;
    command.velocity.y = 0.5;
    command.velocity.z = 0.8;
    command.time_to_strike = 2.0;
    command_publisher->publish(command);

    rclcpp::spin_some(node);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }

  // Keep the participant alive long enough for the reliable command writer
  // to receive ACKs and for the best-effort pose writer to flush its tail.
  (void)command_publisher->wait_for_all_acked(std::chrono::seconds(2));
  const auto flush_deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
  while (std::chrono::steady_clock::now() < flush_deadline) {
    rclcpp::spin_some(node);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }

  std::cout << "published: command=30 pose=30 task=1 final_revision=29"
            << std::endl;

  rclcpp::shutdown();
  return 0;
}
