// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from hope_msgs:msg/RacketCommand.idl
// generated code does not contain a copyright notice
#include "hope_msgs/msg/detail/racket_command__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/detail/header__functions.h"
// Member `position`
#include "geometry_msgs/msg/detail/point__functions.h"
// Member `velocity`
#include "geometry_msgs/msg/detail/vector3__functions.h"

bool
hope_msgs__msg__RacketCommand__init(hope_msgs__msg__RacketCommand * msg)
{
  if (!msg) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__init(&msg->header)) {
    hope_msgs__msg__RacketCommand__fini(msg);
    return false;
  }
  // task_id
  // task_revision
  // swing_side
  // position
  if (!geometry_msgs__msg__Point__init(&msg->position)) {
    hope_msgs__msg__RacketCommand__fini(msg);
    return false;
  }
  // velocity
  if (!geometry_msgs__msg__Vector3__init(&msg->velocity)) {
    hope_msgs__msg__RacketCommand__fini(msg);
    return false;
  }
  // time_to_strike
  return true;
}

void
hope_msgs__msg__RacketCommand__fini(hope_msgs__msg__RacketCommand * msg)
{
  if (!msg) {
    return;
  }
  // header
  std_msgs__msg__Header__fini(&msg->header);
  // task_id
  // task_revision
  // swing_side
  // position
  geometry_msgs__msg__Point__fini(&msg->position);
  // velocity
  geometry_msgs__msg__Vector3__fini(&msg->velocity);
  // time_to_strike
}

bool
hope_msgs__msg__RacketCommand__are_equal(const hope_msgs__msg__RacketCommand * lhs, const hope_msgs__msg__RacketCommand * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__are_equal(
      &(lhs->header), &(rhs->header)))
  {
    return false;
  }
  // task_id
  if (lhs->task_id != rhs->task_id) {
    return false;
  }
  // task_revision
  if (lhs->task_revision != rhs->task_revision) {
    return false;
  }
  // swing_side
  if (lhs->swing_side != rhs->swing_side) {
    return false;
  }
  // position
  if (!geometry_msgs__msg__Point__are_equal(
      &(lhs->position), &(rhs->position)))
  {
    return false;
  }
  // velocity
  if (!geometry_msgs__msg__Vector3__are_equal(
      &(lhs->velocity), &(rhs->velocity)))
  {
    return false;
  }
  // time_to_strike
  if (lhs->time_to_strike != rhs->time_to_strike) {
    return false;
  }
  return true;
}

bool
hope_msgs__msg__RacketCommand__copy(
  const hope_msgs__msg__RacketCommand * input,
  hope_msgs__msg__RacketCommand * output)
{
  if (!input || !output) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__copy(
      &(input->header), &(output->header)))
  {
    return false;
  }
  // task_id
  output->task_id = input->task_id;
  // task_revision
  output->task_revision = input->task_revision;
  // swing_side
  output->swing_side = input->swing_side;
  // position
  if (!geometry_msgs__msg__Point__copy(
      &(input->position), &(output->position)))
  {
    return false;
  }
  // velocity
  if (!geometry_msgs__msg__Vector3__copy(
      &(input->velocity), &(output->velocity)))
  {
    return false;
  }
  // time_to_strike
  output->time_to_strike = input->time_to_strike;
  return true;
}

hope_msgs__msg__RacketCommand *
hope_msgs__msg__RacketCommand__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  hope_msgs__msg__RacketCommand * msg = (hope_msgs__msg__RacketCommand *)allocator.allocate(sizeof(hope_msgs__msg__RacketCommand), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(hope_msgs__msg__RacketCommand));
  bool success = hope_msgs__msg__RacketCommand__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
hope_msgs__msg__RacketCommand__destroy(hope_msgs__msg__RacketCommand * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    hope_msgs__msg__RacketCommand__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
hope_msgs__msg__RacketCommand__Sequence__init(hope_msgs__msg__RacketCommand__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  hope_msgs__msg__RacketCommand * data = NULL;

  if (size) {
    data = (hope_msgs__msg__RacketCommand *)allocator.zero_allocate(size, sizeof(hope_msgs__msg__RacketCommand), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = hope_msgs__msg__RacketCommand__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        hope_msgs__msg__RacketCommand__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
hope_msgs__msg__RacketCommand__Sequence__fini(hope_msgs__msg__RacketCommand__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      hope_msgs__msg__RacketCommand__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

hope_msgs__msg__RacketCommand__Sequence *
hope_msgs__msg__RacketCommand__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  hope_msgs__msg__RacketCommand__Sequence * array = (hope_msgs__msg__RacketCommand__Sequence *)allocator.allocate(sizeof(hope_msgs__msg__RacketCommand__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = hope_msgs__msg__RacketCommand__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
hope_msgs__msg__RacketCommand__Sequence__destroy(hope_msgs__msg__RacketCommand__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    hope_msgs__msg__RacketCommand__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
hope_msgs__msg__RacketCommand__Sequence__are_equal(const hope_msgs__msg__RacketCommand__Sequence * lhs, const hope_msgs__msg__RacketCommand__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!hope_msgs__msg__RacketCommand__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
hope_msgs__msg__RacketCommand__Sequence__copy(
  const hope_msgs__msg__RacketCommand__Sequence * input,
  hope_msgs__msg__RacketCommand__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(hope_msgs__msg__RacketCommand);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    hope_msgs__msg__RacketCommand * data =
      (hope_msgs__msg__RacketCommand *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!hope_msgs__msg__RacketCommand__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          hope_msgs__msg__RacketCommand__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!hope_msgs__msg__RacketCommand__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
