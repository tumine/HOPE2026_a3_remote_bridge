// generated from rosidl_generator_py/resource/_idl_support.c.em
// with input from hope_msgs:msg/RacketCommand.idl
// generated code does not contain a copyright notice
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <stdbool.h>
#ifndef _WIN32
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include "numpy/ndarrayobject.h"
#ifndef _WIN32
# pragma GCC diagnostic pop
#endif
#include "rosidl_runtime_c/visibility_control.h"
#include "hope_msgs/msg/detail/racket_command__struct.h"
#include "hope_msgs/msg/detail/racket_command__functions.h"

ROSIDL_GENERATOR_C_IMPORT
bool std_msgs__msg__header__convert_from_py(PyObject * _pymsg, void * _ros_message);
ROSIDL_GENERATOR_C_IMPORT
PyObject * std_msgs__msg__header__convert_to_py(void * raw_ros_message);
ROSIDL_GENERATOR_C_IMPORT
bool geometry_msgs__msg__point__convert_from_py(PyObject * _pymsg, void * _ros_message);
ROSIDL_GENERATOR_C_IMPORT
PyObject * geometry_msgs__msg__point__convert_to_py(void * raw_ros_message);
ROSIDL_GENERATOR_C_IMPORT
bool geometry_msgs__msg__vector3__convert_from_py(PyObject * _pymsg, void * _ros_message);
ROSIDL_GENERATOR_C_IMPORT
PyObject * geometry_msgs__msg__vector3__convert_to_py(void * raw_ros_message);

ROSIDL_GENERATOR_C_EXPORT
bool hope_msgs__msg__racket_command__convert_from_py(PyObject * _pymsg, void * _ros_message)
{
  // check that the passed message is of the expected Python class
  {
    char full_classname_dest[44];
    {
      char * class_name = NULL;
      char * module_name = NULL;
      {
        PyObject * class_attr = PyObject_GetAttrString(_pymsg, "__class__");
        if (class_attr) {
          PyObject * name_attr = PyObject_GetAttrString(class_attr, "__name__");
          if (name_attr) {
            class_name = (char *)PyUnicode_1BYTE_DATA(name_attr);
            Py_DECREF(name_attr);
          }
          PyObject * module_attr = PyObject_GetAttrString(class_attr, "__module__");
          if (module_attr) {
            module_name = (char *)PyUnicode_1BYTE_DATA(module_attr);
            Py_DECREF(module_attr);
          }
          Py_DECREF(class_attr);
        }
      }
      if (!class_name || !module_name) {
        return false;
      }
      snprintf(full_classname_dest, sizeof(full_classname_dest), "%s.%s", module_name, class_name);
    }
    assert(strncmp("hope_msgs.msg._racket_command.RacketCommand", full_classname_dest, 43) == 0);
  }
  hope_msgs__msg__RacketCommand * ros_message = _ros_message;
  {  // header
    PyObject * field = PyObject_GetAttrString(_pymsg, "header");
    if (!field) {
      return false;
    }
    if (!std_msgs__msg__header__convert_from_py(field, &ros_message->header)) {
      Py_DECREF(field);
      return false;
    }
    Py_DECREF(field);
  }
  {  // task_id
    PyObject * field = PyObject_GetAttrString(_pymsg, "task_id");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->task_id = PyLong_AsUnsignedLongLong(field);
    Py_DECREF(field);
  }
  {  // task_revision
    PyObject * field = PyObject_GetAttrString(_pymsg, "task_revision");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->task_revision = PyLong_AsUnsignedLong(field);
    Py_DECREF(field);
  }
  {  // swing_side
    PyObject * field = PyObject_GetAttrString(_pymsg, "swing_side");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->swing_side = (int8_t)PyLong_AsLong(field);
    Py_DECREF(field);
  }
  {  // position
    PyObject * field = PyObject_GetAttrString(_pymsg, "position");
    if (!field) {
      return false;
    }
    if (!geometry_msgs__msg__point__convert_from_py(field, &ros_message->position)) {
      Py_DECREF(field);
      return false;
    }
    Py_DECREF(field);
  }
  {  // velocity
    PyObject * field = PyObject_GetAttrString(_pymsg, "velocity");
    if (!field) {
      return false;
    }
    if (!geometry_msgs__msg__vector3__convert_from_py(field, &ros_message->velocity)) {
      Py_DECREF(field);
      return false;
    }
    Py_DECREF(field);
  }
  {  // time_to_strike
    PyObject * field = PyObject_GetAttrString(_pymsg, "time_to_strike");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->time_to_strike = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }

  return true;
}

ROSIDL_GENERATOR_C_EXPORT
PyObject * hope_msgs__msg__racket_command__convert_to_py(void * raw_ros_message)
{
  /* NOTE(esteve): Call constructor of RacketCommand */
  PyObject * _pymessage = NULL;
  {
    PyObject * pymessage_module = PyImport_ImportModule("hope_msgs.msg._racket_command");
    assert(pymessage_module);
    PyObject * pymessage_class = PyObject_GetAttrString(pymessage_module, "RacketCommand");
    assert(pymessage_class);
    Py_DECREF(pymessage_module);
    _pymessage = PyObject_CallObject(pymessage_class, NULL);
    Py_DECREF(pymessage_class);
    if (!_pymessage) {
      return NULL;
    }
  }
  hope_msgs__msg__RacketCommand * ros_message = (hope_msgs__msg__RacketCommand *)raw_ros_message;
  {  // header
    PyObject * field = NULL;
    field = std_msgs__msg__header__convert_to_py(&ros_message->header);
    if (!field) {
      return NULL;
    }
    {
      int rc = PyObject_SetAttrString(_pymessage, "header", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // task_id
    PyObject * field = NULL;
    field = PyLong_FromUnsignedLongLong(ros_message->task_id);
    {
      int rc = PyObject_SetAttrString(_pymessage, "task_id", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // task_revision
    PyObject * field = NULL;
    field = PyLong_FromUnsignedLong(ros_message->task_revision);
    {
      int rc = PyObject_SetAttrString(_pymessage, "task_revision", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // swing_side
    PyObject * field = NULL;
    field = PyLong_FromLong(ros_message->swing_side);
    {
      int rc = PyObject_SetAttrString(_pymessage, "swing_side", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // position
    PyObject * field = NULL;
    field = geometry_msgs__msg__point__convert_to_py(&ros_message->position);
    if (!field) {
      return NULL;
    }
    {
      int rc = PyObject_SetAttrString(_pymessage, "position", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // velocity
    PyObject * field = NULL;
    field = geometry_msgs__msg__vector3__convert_to_py(&ros_message->velocity);
    if (!field) {
      return NULL;
    }
    {
      int rc = PyObject_SetAttrString(_pymessage, "velocity", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // time_to_strike
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->time_to_strike);
    {
      int rc = PyObject_SetAttrString(_pymessage, "time_to_strike", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }

  // ownership of _pymessage is transferred to the caller
  return _pymessage;
}
