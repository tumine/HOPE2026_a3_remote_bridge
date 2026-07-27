# Robot-specific table-tennis configs live in sub-packages (e.g. ``agibot_a3``). This file only makes
# ``config`` an importable package so ``isaaclab_tasks.utils.import_packages`` can discover and import
# the robot sub-packages (which perform the Gym registration). No configs are exposed here directly.
