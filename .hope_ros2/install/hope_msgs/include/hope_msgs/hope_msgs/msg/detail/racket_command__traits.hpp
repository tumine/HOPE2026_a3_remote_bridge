// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from hope_msgs:msg/RacketCommand.idl
// generated code does not contain a copyright notice

#ifndef HOPE_MSGS__MSG__DETAIL__RACKET_COMMAND__TRAITS_HPP_
#define HOPE_MSGS__MSG__DETAIL__RACKET_COMMAND__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "hope_msgs/msg/detail/racket_command__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__traits.hpp"
// Member 'position'
#include "geometry_msgs/msg/detail/point__traits.hpp"
// Member 'velocity'
#include "geometry_msgs/msg/detail/vector3__traits.hpp"

namespace hope_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const RacketCommand & msg,
  std::ostream & out)
{
  out << "{";
  // member: header
  {
    out << "header: ";
    to_flow_style_yaml(msg.header, out);
    out << ", ";
  }

  // member: task_id
  {
    out << "task_id: ";
    rosidl_generator_traits::value_to_yaml(msg.task_id, out);
    out << ", ";
  }

  // member: task_revision
  {
    out << "task_revision: ";
    rosidl_generator_traits::value_to_yaml(msg.task_revision, out);
    out << ", ";
  }

  // member: swing_side
  {
    out << "swing_side: ";
    rosidl_generator_traits::value_to_yaml(msg.swing_side, out);
    out << ", ";
  }

  // member: position
  {
    out << "position: ";
    to_flow_style_yaml(msg.position, out);
    out << ", ";
  }

  // member: velocity
  {
    out << "velocity: ";
    to_flow_style_yaml(msg.velocity, out);
    out << ", ";
  }

  // member: time_to_strike
  {
    out << "time_to_strike: ";
    rosidl_generator_traits::value_to_yaml(msg.time_to_strike, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const RacketCommand & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: header
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "header:\n";
    to_block_style_yaml(msg.header, out, indentation + 2);
  }

  // member: task_id
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "task_id: ";
    rosidl_generator_traits::value_to_yaml(msg.task_id, out);
    out << "\n";
  }

  // member: task_revision
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "task_revision: ";
    rosidl_generator_traits::value_to_yaml(msg.task_revision, out);
    out << "\n";
  }

  // member: swing_side
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "swing_side: ";
    rosidl_generator_traits::value_to_yaml(msg.swing_side, out);
    out << "\n";
  }

  // member: position
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "position:\n";
    to_block_style_yaml(msg.position, out, indentation + 2);
  }

  // member: velocity
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "velocity:\n";
    to_block_style_yaml(msg.velocity, out, indentation + 2);
  }

  // member: time_to_strike
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "time_to_strike: ";
    rosidl_generator_traits::value_to_yaml(msg.time_to_strike, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const RacketCommand & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace msg

}  // namespace hope_msgs

namespace rosidl_generator_traits
{

[[deprecated("use hope_msgs::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const hope_msgs::msg::RacketCommand & msg,
  std::ostream & out, size_t indentation = 0)
{
  hope_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use hope_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const hope_msgs::msg::RacketCommand & msg)
{
  return hope_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<hope_msgs::msg::RacketCommand>()
{
  return "hope_msgs::msg::RacketCommand";
}

template<>
inline const char * name<hope_msgs::msg::RacketCommand>()
{
  return "hope_msgs/msg/RacketCommand";
}

template<>
struct has_fixed_size<hope_msgs::msg::RacketCommand>
  : std::integral_constant<bool, has_fixed_size<geometry_msgs::msg::Point>::value && has_fixed_size<geometry_msgs::msg::Vector3>::value && has_fixed_size<std_msgs::msg::Header>::value> {};

template<>
struct has_bounded_size<hope_msgs::msg::RacketCommand>
  : std::integral_constant<bool, has_bounded_size<geometry_msgs::msg::Point>::value && has_bounded_size<geometry_msgs::msg::Vector3>::value && has_bounded_size<std_msgs::msg::Header>::value> {};

template<>
struct is_message<hope_msgs::msg::RacketCommand>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // HOPE_MSGS__MSG__DETAIL__RACKET_COMMAND__TRAITS_HPP_
