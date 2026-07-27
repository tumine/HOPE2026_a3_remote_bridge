"""rsl_rl PPO runner glue for HOPE training.

The subclass intentionally keeps :class:`rsl_rl.runners.OnPolicyRunner`'s standard
logging implementation. With ``logger: tensorboard`` in the agent configuration,
rsl_rl creates a ``torch.utils.tensorboard.SummaryWriter`` in the run directory and
writes losses, rewards, episode metrics and performance scalars while retaining the
normal local checkpoint behavior.
"""

from __future__ import annotations

from rsl_rl.runners import OnPolicyRunner


class HOPEOnPolicyRunner(OnPolicyRunner):
    """HOPE runner using rsl_rl's configured logger and checkpoint implementation."""

    pass
