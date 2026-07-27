// generated from rosidl_generator_c/resource/idl__functions.h.em
// with input from hope_msgs:msg/RacketCommand.idl
// generated code does not contain a copyright notice

#ifndef HOPE_MSGS__MSG__DETAIL__RACKET_COMMAND__FUNCTIONS_H_
#define HOPE_MSGS__MSG__DETAIL__RACKET_COMMAND__FUNCTIONS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "rosidl_runtime_c/visibility_control.h"
#include "hope_msgs/msg/rosidl_generator_c__visibility_control.h"

#include "hope_msgs/msg/detail/racket_command__struct.h"

/// Initialize msg/RacketCommand message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * hope_msgs__msg__RacketCommand
 * )) before or use
 * hope_msgs__msg__RacketCommand__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_hope_msgs
bool
hope_msgs__msg__RacketCommand__init(hope_msgs__msg__RacketCommand * msg);

/// Finalize msg/RacketCommand message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_hope_msgs
void
hope_msgs__msg__RacketCommand__fini(hope_msgs__msg__RacketCommand * msg);

/// Create msg/RacketCommand message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * hope_msgs__msg__RacketCommand__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_hope_msgs
hope_msgs__msg__RacketCommand *
hope_msgs__msg__RacketCommand__create();

/// Destroy msg/RacketCommand message.
/**
 * It calls
 * hope_msgs__msg__RacketCommand__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_hope_msgs
void
hope_msgs__msg__RacketCommand__destroy(hope_msgs__msg__RacketCommand * msg);

/// Check for msg/RacketCommand message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_hope_msgs
bool
hope_msgs__msg__RacketCommand__are_equal(const hope_msgs__msg__RacketCommand * lhs, const hope_msgs__msg__RacketCommand * rhs);

/// Copy a msg/RacketCommand message.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source message pointer.
 * \param[out] output The target message pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer is null
 *   or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_hope_msgs
bool
hope_msgs__msg__RacketCommand__copy(
  const hope_msgs__msg__RacketCommand * input,
  hope_msgs__msg__RacketCommand * output);

/// Initialize array of msg/RacketCommand messages.
/**
 * It allocates the memory for the number of elements and calls
 * hope_msgs__msg__RacketCommand__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_hope_msgs
bool
hope_msgs__msg__RacketCommand__Sequence__init(hope_msgs__msg__RacketCommand__Sequence * array, size_t size);

/// Finalize array of msg/RacketCommand messages.
/**
 * It calls
 * hope_msgs__msg__RacketCommand__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_hope_msgs
void
hope_msgs__msg__RacketCommand__Sequence__fini(hope_msgs__msg__RacketCommand__Sequence * array);

/// Create array of msg/RacketCommand messages.
/**
 * It allocates the memory for the array and calls
 * hope_msgs__msg__RacketCommand__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_hope_msgs
hope_msgs__msg__RacketCommand__Sequence *
hope_msgs__msg__RacketCommand__Sequence__create(size_t size);

/// Destroy array of msg/RacketCommand messages.
/**
 * It calls
 * hope_msgs__msg__RacketCommand__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_hope_msgs
void
hope_msgs__msg__RacketCommand__Sequence__destroy(hope_msgs__msg__RacketCommand__Sequence * array);

/// Check for msg/RacketCommand message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_hope_msgs
bool
hope_msgs__msg__RacketCommand__Sequence__are_equal(const hope_msgs__msg__RacketCommand__Sequence * lhs, const hope_msgs__msg__RacketCommand__Sequence * rhs);

/// Copy an array of msg/RacketCommand messages.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source array pointer.
 * \param[out] output The target array pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer
 *   is null or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_hope_msgs
bool
hope_msgs__msg__RacketCommand__Sequence__copy(
  const hope_msgs__msg__RacketCommand__Sequence * input,
  hope_msgs__msg__RacketCommand__Sequence * output);

#ifdef __cplusplus
}
#endif

#endif  // HOPE_MSGS__MSG__DETAIL__RACKET_COMMAND__FUNCTIONS_H_
