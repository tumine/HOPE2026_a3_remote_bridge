// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from hope_msgs:msg/RacketCommand.idl
// generated code does not contain a copyright notice

#ifndef HOPE_MSGS__MSG__DETAIL__RACKET_COMMAND__BUILDER_HPP_
#define HOPE_MSGS__MSG__DETAIL__RACKET_COMMAND__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "hope_msgs/msg/detail/racket_command__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace hope_msgs
{

namespace msg
{

namespace builder
{

class Init_RacketCommand_time_to_strike
{
public:
  explicit Init_RacketCommand_time_to_strike(::hope_msgs::msg::RacketCommand & msg)
  : msg_(msg)
  {}
  ::hope_msgs::msg::RacketCommand time_to_strike(::hope_msgs::msg::RacketCommand::_time_to_strike_type arg)
  {
    msg_.time_to_strike = std::move(arg);
    return std::move(msg_);
  }

private:
  ::hope_msgs::msg::RacketCommand msg_;
};

class Init_RacketCommand_velocity
{
public:
  explicit Init_RacketCommand_velocity(::hope_msgs::msg::RacketCommand & msg)
  : msg_(msg)
  {}
  Init_RacketCommand_time_to_strike velocity(::hope_msgs::msg::RacketCommand::_velocity_type arg)
  {
    msg_.velocity = std::move(arg);
    return Init_RacketCommand_time_to_strike(msg_);
  }

private:
  ::hope_msgs::msg::RacketCommand msg_;
};

class Init_RacketCommand_position
{
public:
  explicit Init_RacketCommand_position(::hope_msgs::msg::RacketCommand & msg)
  : msg_(msg)
  {}
  Init_RacketCommand_velocity position(::hope_msgs::msg::RacketCommand::_position_type arg)
  {
    msg_.position = std::move(arg);
    return Init_RacketCommand_velocity(msg_);
  }

private:
  ::hope_msgs::msg::RacketCommand msg_;
};

class Init_RacketCommand_swing_side
{
public:
  explicit Init_RacketCommand_swing_side(::hope_msgs::msg::RacketCommand & msg)
  : msg_(msg)
  {}
  Init_RacketCommand_position swing_side(::hope_msgs::msg::RacketCommand::_swing_side_type arg)
  {
    msg_.swing_side = std::move(arg);
    return Init_RacketCommand_position(msg_);
  }

private:
  ::hope_msgs::msg::RacketCommand msg_;
};

class Init_RacketCommand_task_revision
{
public:
  explicit Init_RacketCommand_task_revision(::hope_msgs::msg::RacketCommand & msg)
  : msg_(msg)
  {}
  Init_RacketCommand_swing_side task_revision(::hope_msgs::msg::RacketCommand::_task_revision_type arg)
  {
    msg_.task_revision = std::move(arg);
    return Init_RacketCommand_swing_side(msg_);
  }

private:
  ::hope_msgs::msg::RacketCommand msg_;
};

class Init_RacketCommand_task_id
{
public:
  explicit Init_RacketCommand_task_id(::hope_msgs::msg::RacketCommand & msg)
  : msg_(msg)
  {}
  Init_RacketCommand_task_revision task_id(::hope_msgs::msg::RacketCommand::_task_id_type arg)
  {
    msg_.task_id = std::move(arg);
    return Init_RacketCommand_task_revision(msg_);
  }

private:
  ::hope_msgs::msg::RacketCommand msg_;
};

class Init_RacketCommand_header
{
public:
  Init_RacketCommand_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_RacketCommand_task_id header(::hope_msgs::msg::RacketCommand::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_RacketCommand_task_id(msg_);
  }

private:
  ::hope_msgs::msg::RacketCommand msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::hope_msgs::msg::RacketCommand>()
{
  return hope_msgs::msg::builder::Init_RacketCommand_header();
}

}  // namespace hope_msgs

#endif  // HOPE_MSGS__MSG__DETAIL__RACKET_COMMAND__BUILDER_HPP_
