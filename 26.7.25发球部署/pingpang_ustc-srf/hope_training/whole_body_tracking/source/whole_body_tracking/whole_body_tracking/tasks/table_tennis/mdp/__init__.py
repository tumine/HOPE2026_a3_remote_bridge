"""MDP terms for the table-tennis environment.

Re-exports the generic Isaac Lab MDP terms (joint-position action, proprioceptive observations,
``reset_root_state_uniform`` / ``reset_joints_by_scale`` events, ``is_alive`` / ``action_rate_l2``
rewards, ``time_out`` termination) plus the table-tennis-specific terms defined here.
"""

from isaaclab.envs.mdp import *  # noqa: F401, F403

from .events import *  # noqa: F401, F403
from .observations import *  # noqa: F401, F403
from .rewards import *  # noqa: F401, F403
from .terminations import *  # noqa: F401, F403
