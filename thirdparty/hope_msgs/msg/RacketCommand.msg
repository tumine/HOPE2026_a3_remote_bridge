# Racket target command published by the local HOPE planner.
std_msgs/Header header

# A new incoming ball gets a new task id. Revisions refine the same task.
uint64 task_id
uint32 task_revision

int8 FOREHAND=1
int8 BACKHAND=-1
int8 swing_side

# All values use the canonical table frame from header.frame_id.
geometry_msgs/Point position
geometry_msgs/Vector3 velocity

# Remaining seconds from header.stamp until racket impact.
float64 time_to_strike
