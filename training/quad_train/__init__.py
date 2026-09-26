"""Training the quadruped's walking policy (#377). Registers the tasks with
mjlab; `python -m quad_train.train <task> ...` runs mjlab's own trainer."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from quad_train.rl import ppo_runner_cfg
from quad_train.task import flat_env_cfg

register_mjlab_task(
  task_id="Pluggy-Quad-Flat",
  env_cfg=flat_env_cfg(),
  play_env_cfg=flat_env_cfg(play=True),
  rl_cfg=ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
