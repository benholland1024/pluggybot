"""PPO for the humanoid: the quadruped's (mjlab's Go1 and G1 recipe, rsl_rl),
its own experiment name. The actor is the network the served sim runs in
numpy (`pluggybot.humanoid.policy`): 512-256-128, ELU, a running
observation normaliser."""

from dataclasses import replace

from quad_train.rl import ppo_runner_cfg as _quad


def ppo_runner_cfg():
  return replace(_quad(), experiment_name="pluggy_humanoid")
