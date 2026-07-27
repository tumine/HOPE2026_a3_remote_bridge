# generated from rosidl_generator_py/resource/_idl.py.em
# with input from hope_msgs:msg/RacketCommand.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import math  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_RacketCommand(type):
    """Metaclass of message 'RacketCommand'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
        'FOREHAND': 1,
        'BACKHAND': -1,
    }

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('hope_msgs')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'hope_msgs.msg.RacketCommand')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__racket_command
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__racket_command
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__racket_command
            cls._TYPE_SUPPORT = module.type_support_msg__msg__racket_command
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__racket_command

            from geometry_msgs.msg import Point
            if Point.__class__._TYPE_SUPPORT is None:
                Point.__class__.__import_type_support__()

            from geometry_msgs.msg import Vector3
            if Vector3.__class__._TYPE_SUPPORT is None:
                Vector3.__class__.__import_type_support__()

            from std_msgs.msg import Header
            if Header.__class__._TYPE_SUPPORT is None:
                Header.__class__.__import_type_support__()

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
            'FOREHAND': cls.__constants['FOREHAND'],
            'BACKHAND': cls.__constants['BACKHAND'],
        }

    @property
    def FOREHAND(self):
        """Message constant 'FOREHAND'."""
        return Metaclass_RacketCommand.__constants['FOREHAND']

    @property
    def BACKHAND(self):
        """Message constant 'BACKHAND'."""
        return Metaclass_RacketCommand.__constants['BACKHAND']


class RacketCommand(metaclass=Metaclass_RacketCommand):
    """
    Message class 'RacketCommand'.

    Constants:
      FOREHAND
      BACKHAND
    """

    __slots__ = [
        '_header',
        '_task_id',
        '_task_revision',
        '_swing_side',
        '_position',
        '_velocity',
        '_time_to_strike',
    ]

    _fields_and_field_types = {
        'header': 'std_msgs/Header',
        'task_id': 'uint64',
        'task_revision': 'uint32',
        'swing_side': 'int8',
        'position': 'geometry_msgs/Point',
        'velocity': 'geometry_msgs/Vector3',
        'time_to_strike': 'double',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.NamespacedType(['std_msgs', 'msg'], 'Header'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint64'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint32'),  # noqa: E501
        rosidl_parser.definition.BasicType('int8'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['geometry_msgs', 'msg'], 'Point'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['geometry_msgs', 'msg'], 'Vector3'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        from std_msgs.msg import Header
        self.header = kwargs.get('header', Header())
        self.task_id = kwargs.get('task_id', int())
        self.task_revision = kwargs.get('task_revision', int())
        self.swing_side = kwargs.get('swing_side', int())
        from geometry_msgs.msg import Point
        self.position = kwargs.get('position', Point())
        from geometry_msgs.msg import Vector3
        self.velocity = kwargs.get('velocity', Vector3())
        self.time_to_strike = kwargs.get('time_to_strike', float())

    def __repr__(self):
        typename = self.__class__.__module__.split('.')
        typename.pop()
        typename.append(self.__class__.__name__)
        args = []
        for s, t in zip(self.__slots__, self.SLOT_TYPES):
            field = getattr(self, s)
            fieldstr = repr(field)
            # We use Python array type for fields that can be directly stored
            # in them, and "normal" sequences for everything else.  If it is
            # a type that we store in an array, strip off the 'array' portion.
            if (
                isinstance(t, rosidl_parser.definition.AbstractSequence) and
                isinstance(t.value_type, rosidl_parser.definition.BasicType) and
                t.value_type.typename in ['float', 'double', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']
            ):
                if len(field) == 0:
                    fieldstr = '[]'
                else:
                    assert fieldstr.startswith('array(')
                    prefix = "array('X', "
                    suffix = ')'
                    fieldstr = fieldstr[len(prefix):-len(suffix)]
            args.append(s[1:] + '=' + fieldstr)
        return '%s(%s)' % ('.'.join(typename), ', '.join(args))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if self.header != other.header:
            return False
        if self.task_id != other.task_id:
            return False
        if self.task_revision != other.task_revision:
            return False
        if self.swing_side != other.swing_side:
            return False
        if self.position != other.position:
            return False
        if self.velocity != other.velocity:
            return False
        if self.time_to_strike != other.time_to_strike:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def header(self):
        """Message field 'header'."""
        return self._header

    @header.setter
    def header(self, value):
        if __debug__:
            from std_msgs.msg import Header
            assert \
                isinstance(value, Header), \
                "The 'header' field must be a sub message of type 'Header'"
        self._header = value

    @builtins.property
    def task_id(self):
        """Message field 'task_id'."""
        return self._task_id

    @task_id.setter
    def task_id(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'task_id' field must be of type 'int'"
            assert value >= 0 and value < 18446744073709551616, \
                "The 'task_id' field must be an unsigned integer in [0, 18446744073709551615]"
        self._task_id = value

    @builtins.property
    def task_revision(self):
        """Message field 'task_revision'."""
        return self._task_revision

    @task_revision.setter
    def task_revision(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'task_revision' field must be of type 'int'"
            assert value >= 0 and value < 4294967296, \
                "The 'task_revision' field must be an unsigned integer in [0, 4294967295]"
        self._task_revision = value

    @builtins.property
    def swing_side(self):
        """Message field 'swing_side'."""
        return self._swing_side

    @swing_side.setter
    def swing_side(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'swing_side' field must be of type 'int'"
            assert value >= -128 and value < 128, \
                "The 'swing_side' field must be an integer in [-128, 127]"
        self._swing_side = value

    @builtins.property
    def position(self):
        """Message field 'position'."""
        return self._position

    @position.setter
    def position(self, value):
        if __debug__:
            from geometry_msgs.msg import Point
            assert \
                isinstance(value, Point), \
                "The 'position' field must be a sub message of type 'Point'"
        self._position = value

    @builtins.property
    def velocity(self):
        """Message field 'velocity'."""
        return self._velocity

    @velocity.setter
    def velocity(self, value):
        if __debug__:
            from geometry_msgs.msg import Vector3
            assert \
                isinstance(value, Vector3), \
                "The 'velocity' field must be a sub message of type 'Vector3'"
        self._velocity = value

    @builtins.property
    def time_to_strike(self):
        """Message field 'time_to_strike'."""
        return self._time_to_strike

    @time_to_strike.setter
    def time_to_strike(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'time_to_strike' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'time_to_strike' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._time_to_strike = value
