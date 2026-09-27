"""What the stairs curriculum adds to mjlab's velocity task (#388): a command
that walks the robot OUT along a flight, and a reward for clearing the step
nose ahead of a swinging foot.

Why the command: a robot is promoted a level for ending an episode 4 m from
its tile's centre (`terrain_levels_vel`), and mjlab's commands turn and
sidestep every 3-8 s, so a capable robot seldom gets there: across rough3's
6000 iterations the climbing column's level sat at 1.7 of 10 (risers of
~0.08 m), and even the FLAT column's at 4.3, where the level changes nothing.
On a stair tile most commands here are straight ahead, aimed along the
nearest of the pyramid's four flights, so a robot that can climb is promoted.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp.velocity_command import (
  UniformVelocityCommand,
  UniformVelocityCommandCfg,
)
from mjlab.utils.lab_api.math import wrap_to_pi

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class StairsCommand(UniformVelocityCommand):
  """mjlab's twist, except on the stair tiles: there a share of the commands
  walk straight out of the tile's centre, heading along the nearest flight
  (give or take `jitter`), at a speed from `speed`."""

  cfg: StairsCommandCfg

  def __init__(self, cfg: StairsCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    terrain = env.scene.terrain
    gen = terrain.cfg.terrain_generator if terrain is not None else None
    names = list(gen.sub_terrains) if gen is not None and gen.curriculum else []
    self.stair_cols = torch.tensor(
      [names.index(n) for n in cfg.stairs if n in names], device=self.device,
      dtype=torch.long)
    #: Envs whose heading target is set on the next update: a reset
    #: resamples before `sim.forward()`, so the pose read here is stale.
    self.aim = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    terrain = self._env.scene.terrain
    if len(self.stair_cols) == 0 or terrain is None or terrain.terrain_types is None:
      return
    on_stairs = torch.isin(terrain.terrain_types[env_ids], self.stair_cols)
    pick = torch.rand(len(env_ids), device=self.device) < self.cfg.forward
    ids = env_ids[on_stairs & pick]
    if len(ids) == 0:
      return
    self.vel_command_b[ids, 0] = torch.empty(len(ids), device=self.device).uniform_(
      *self.cfg.speed)
    self.vel_command_b[ids, 1:] = 0.0
    self.is_standing_env[ids] = False
    self.is_world_env[ids] = False
    self.is_forward_env[ids] = True
    self.is_heading_env[ids] = True
    self.aim[ids] = True

  def _update_command(self) -> None:
    ids = self.aim.nonzero(as_tuple=False).flatten()
    if len(ids) > 0:
      # Out of the centre: the nearest axis to the way out, or, right at the
      # centre, to the way the robot faces.
      offset = (self.robot.data.root_link_pos_w[ids, :2]
                - self._env.scene.env_origins[ids, :2])
      out = torch.atan2(offset[:, 1], offset[:, 0])
      near = offset.norm(dim=1) < self.cfg.centre_m
      way = torch.where(near, self.robot.data.heading_w[ids], out)
      axis = torch.round(way / (math.pi / 2)) * (math.pi / 2)
      jitter = torch.empty(len(ids), device=self.device).uniform_(
        -self.cfg.jitter, self.cfg.jitter)
      self.heading_target[ids] = wrap_to_pi(axis + jitter)
      self.aim[ids] = False
    super()._update_command()


@dataclass(kw_only=True)
class StairsCommandCfg(UniformVelocityCommandCfg):
  stairs: tuple[str, ...] = ()
  """The sub-terrains (curriculum columns) that are stairs."""
  forward: float = 0.75
  """The share of a stair tile's commands that walk straight out."""
  speed: tuple[float, float] = (0.3, 0.7)
  """Their forward speed, m/s."""
  jitter: float = 0.25
  """How far off the flight's axis they may head, rad."""
  centre_m: float = 0.3
  """Closer than this to the centre, the way out is the way the robot faces."""

  def build(self, env: ManagerBasedRlEnv) -> StairsCommand:
    return StairsCommand(self, env)


def foot_nose_clearance(
  env: ManagerBasedRlEnv,
  target_height: float,
  height_sensor_name: str,
  command_name: str,
  command_threshold: float = 0.05,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """A moving foot less than `target_height` over the highest terrain within
  the sensor's ring, summed over the feet and weighted by each foot's speed.

  One-sided, unlike mjlab's `feet_clearance`, which also charges a foot
  swung HIGH -- the lift a riser needs. With the ring half a tread wide, a
  foot closing on a riser answers to the tread above it before it gets
  there (a ray starting inside the step reads zero), and a foot stepping
  down answers to the tread it leaves until it is half a tread past the
  nose."""
  asset = env.scene[asset_cfg.name]
  heights = env.scene[height_sensor_name].data.heights  # [B, F]
  speed = torch.norm(asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2], dim=-1)
  short = torch.clamp(target_height - heights, min=0.0)
  cost = torch.sum(short * speed, dim=1)
  command = env.command_manager.get_command(command_name)
  assert command is not None
  moving = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
  cost = cost * (moving > command_threshold).float()
  swinging = speed > 0.1
  env.extras["log"]["Metrics/nose_shortfall_mean"] = (
    torch.sum(short * swinging) / torch.clamp(torch.sum(swinging), min=1))
  return cost
