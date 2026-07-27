// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from hope_msgs:msg/RacketCommand.idl
// generated code does not contain a copyright notice

#ifndef HOPE_MSGS__MSG__DETAIL__RACKET_COMMAND__STRUCT_HPP_
#define HOPE_MSGS__MSG__DETAIL__RACKET_COMMAND__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__struct.hpp"
// Member 'position'
#include "geometry_msgs/msg/detail/point__struct.hpp"
// Member 'velocity'
#include "geometry_msgs/msg/detail/vector3__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__hope_msgs__msg__RacketCommand __attribute__((deprecated))
#else
# define DEPRECATED__hope_msgs__msg__RacketCommand __declspec(deprecated)
#endif

namespace hope_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct RacketCommand_
{
  using Type = RacketCommand_<ContainerAllocator>;

  explicit RacketCommand_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_init),
    position(_init),
    velocity(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->task_id = 0ull;
      this->task_revision = 0ul;
      this->swing_side = 0;
      this->time_to_strike = 0.0;
    }
  }

  explicit RacketCommand_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_alloc, _init),
    position(_alloc, _init),
    velocity(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->task_id = 0ull;
      this->task_revision = 0ul;
      this->swing_side = 0;
      this->time_to_strike = 0.0;
    }
  }

  // field types and members
  using _header_type =
    std_msgs::msg::Header_<ContainerAllocator>;
  _header_type header;
  using _task_id_type =
    uint64_t;
  _task_id_type task_id;
  using _task_revision_type =
    uint32_t;
  _task_revision_type task_revision;
  using _swing_side_type =
    int8_t;
  _swing_side_type swing_side;
  using _position_type =
    geometry_msgs::msg::Point_<ContainerAllocator>;
  _position_type position;
  using _velocity_type =
    geometry_msgs::msg::Vector3_<ContainerAllocator>;
  _velocity_type velocity;
  using _time_to_strike_type =
    double;
  _time_to_strike_type time_to_strike;

  // setters for named parameter idiom
  Type & set__header(
    const std_msgs::msg::Header_<ContainerAllocator> & _arg)
  {
    this->header = _arg;
    return *this;
  }
  Type & set__task_id(
    const uint64_t & _arg)
  {
    this->task_id = _arg;
    return *this;
  }
  Type & set__task_revision(
    const uint32_t & _arg)
  {
    this->task_revision = _arg;
    return *this;
  }
  Type & set__swing_side(
    const int8_t & _arg)
  {
    this->swing_side = _arg;
    return *this;
  }
  Type & set__position(
    const geometry_msgs::msg::Point_<ContainerAllocator> & _arg)
  {
    this->position = _arg;
    return *this;
  }
  Type & set__velocity(
    const geometry_msgs::msg::Vector3_<ContainerAllocator> & _arg)
  {
    this->velocity = _arg;
    return *this;
  }
  Type & set__time_to_strike(
    const double & _arg)
  {
    this->time_to_strike = _arg;
    return *this;
  }

  // constant declarations
  static constexpr int8_t FOREHAND =
    1;
  static constexpr int8_t BACKHAND =
    -1;

  // pointer types
  using RawPtr =
    hope_msgs::msg::RacketCommand_<ContainerAllocator> *;
  using ConstRawPtr =
    const hope_msgs::msg::RacketCommand_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<hope_msgs::msg::RacketCommand_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<hope_msgs::msg::RacketCommand_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      hope_msgs::msg::RacketCommand_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<hope_msgs::msg::RacketCommand_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      hope_msgs::msg::RacketCommand_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<hope_msgs::msg::RacketCommand_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<hope_msgs::msg::RacketCommand_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<hope_msgs::msg::RacketCommand_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__hope_msgs__msg__RacketCommand
    std::shared_ptr<hope_msgs::msg::RacketCommand_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__hope_msgs__msg__RacketCommand
    std::shared_ptr<hope_msgs::msg::RacketCommand_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const RacketCommand_ & other) const
  {
    if (this->header != other.header) {
      return false;
    }
    if (this->task_id != other.task_id) {
      return false;
    }
    if (this->task_revision != other.task_revision) {
      return false;
    }
    if (this->swing_side != other.swing_side) {
      return false;
    }
    if (this->position != other.position) {
      return false;
    }
    if (this->velocity != other.velocity) {
      return false;
    }
    if (this->time_to_strike != other.time_to_strike) {
      return false;
    }
    return true;
  }
  bool operator!=(const RacketCommand_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct RacketCommand_

// alias to use template instance with default allocator
using RacketCommand =
  hope_msgs::msg::RacketCommand_<std::allocator<void>>;

// constant definitions
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr int8_t RacketCommand_<ContainerAllocator>::FOREHAND;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr int8_t RacketCommand_<ContainerAllocator>::BACKHAND;
#endif  // __cplusplus < 201703L

}  // namespace msg

}  // namespace hope_msgs

#endif  // HOPE_MSGS__MSG__DETAIL__RACKET_COMMAND__STRUCT_HPP_
