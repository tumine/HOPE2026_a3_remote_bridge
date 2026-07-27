// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from hope_msgs:msg/RacketCommand.idl
// generated code does not contain a copyright notice

#ifndef HOPE_MSGS__MSG__DETAIL__RACKET_COMMAND__STRUCT_H_
#define HOPE_MSGS__MSG__DETAIL__RACKET_COMMAND__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Constant 'FOREHAND'.
/**
  * swing_side values.
 */
enum
{
  hope_msgs__msg__RacketCommand__FOREHAND = 1
};

/// Constant 'BACKHAND'.
enum
{
  hope_msgs__msg__RacketCommand__BACKHAND = -1
};

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__struct.h"
// Member 'position'
#include "geometry_msgs/msg/detail/point__struct.h"
// Member 'velocity'
#include "geometry_msgs/msg/detail/vector3__struct.h"

/// Struct defined in msg/RacketCommand in the package hope_msgs.
/**
  * Racket target command published by the HOPE planner.
  *
  * One task_id is created per incoming ball. Pre-strike updates keep the same
  * task_id and increase task_revision. swing_side is chosen once per task_id and
  * stays fixed within it. All fields refer to the world frame (+x forward,
  * +y left, +z up; metres, seconds).
 */
typedef struct hope_msgs__msg__RacketCommand
{
  std_msgs__msg__Header header;
  /// Identifies the incoming ball this command belongs to (new ball -> new id).
  uint64_t task_id;
  /// Monotonically increasing within a task_id as the pre-strike plan is refined.
  uint32_t task_revision;
  /// Selected swing side for this task (FOREHAND or BACKHAND). Formal side channel.
  int8_t swing_side;
  /// The header stamp is the capture/planning time at which time_to_strike is
  /// defined. Consumers subtract transport and queue age before observation use.
  /// A zero/invalid/future stamp must never be used to increase time_to_strike.
  /// Target racket position at the strike, world frame (m).
  geometry_msgs__msg__Point position;
  /// Target racket velocity at the strike, world frame (m/s).
  geometry_msgs__msg__Vector3 velocity;
  /// Remaining time from header.stamp until the strike (s).
  double time_to_strike;
} hope_msgs__msg__RacketCommand;

// Struct for a sequence of hope_msgs__msg__RacketCommand.
typedef struct hope_msgs__msg__RacketCommand__Sequence
{
  hope_msgs__msg__RacketCommand * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} hope_msgs__msg__RacketCommand__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // HOPE_MSGS__MSG__DETAIL__RACKET_COMMAND__STRUCT_H_
