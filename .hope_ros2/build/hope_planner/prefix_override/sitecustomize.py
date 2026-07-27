import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/bth/workspace/a3_remote_bridge/.hope_ros2/install/hope_planner'
