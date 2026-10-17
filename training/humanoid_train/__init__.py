"""Training the humanoid's policies (#507). Registers the tasks with mjlab;
`python -m quad_train.train <task> ...` runs mjlab's own trainer on them."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from humanoid_train.rl import ppo_runner_cfg
from humanoid_train.stand import stand_env_cfg
from humanoid_train.walk import rough_env_cfg, walk_env_cfg

register_mjlab_task(
  task_id="Pluggy-Humanoid-Stand",
  env_cfg=stand_env_cfg(),
  play_env_cfg=stand_env_cfg(play=True),
  rl_cfg=ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Pluggy-Humanoid-Walk",
  env_cfg=walk_env_cfg(),
  play_env_cfg=walk_env_cfg(play=True),
  rl_cfg=ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Pluggy-Humanoid-Walk-Rough",
  env_cfg=rough_env_cfg(),
  play_env_cfg=rough_env_cfg(play=True),
  rl_cfg=ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
