"""Posture (#377 item 3): walking with the torso's height, pitch and roll
commanded too -- reaching low, tilting a tool, ducking under a table.

The posture command is (height offset from the stand, pitch, roll); the
tilt is tracked as gravity's direction in the body frame, which for a roll
r then a pitch p is (sin p, -sin r cos p, -cos r cos p). It replaces the
stock "stay level" reward, and the joint-pose reward is loosened so a crouch
is not paid against. The actor's observation grows by the three numbers
(`observation_names` in the exported file says so, and
`pluggybot.legs.policy` builds the vector from those names).
"""

from dataclasses import dataclass

import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp

from quad_train import robot
from quad_train.task import flat_env_cfg

#: Height offsets from the stand (m), pitch and roll (rad). The low end is a
#: deep crouch; lying on the belly is the rest routine's, not the policy's.
HEIGHT = (-0.14, 0.03)
PITCH = (-0.25, 0.25)
ROLL = (-0.2, 0.2)
_ROBOT = SceneEntityCfg("robot")


@dataclass(kw_only=True)
class PostureCommandCfg(CommandTermCfg):
  entity_name: str = "robot"
  height: tuple[float, float] = HEIGHT
  pitch: tuple[float, float] = PITCH
  roll: tuple[float, float] = ROLL
  #: Share of commands at the nominal stance, so walking stays the default.
  rel_nominal_envs: float = 0.4

  def build(self, env) -> "PostureCommand":
    return PostureCommand(self, env)


class PostureCommand(CommandTerm):
  cfg: PostureCommandCfg

  def __init__(self, cfg: PostureCommandCfg, env):
    super().__init__(cfg, env)
    self.robot = env.scene[cfg.entity_name]
    self.posture = torch.zeros(self.num_envs, 3, device=self.device)
    self.metrics["error_height"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_tilt"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.posture

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    r = torch.empty(len(env_ids), device=self.device)
    self.posture[env_ids, 0] = r.uniform_(*self.cfg.height)
    self.posture[env_ids, 1] = r.uniform_(*self.cfg.pitch)
    self.posture[env_ids, 2] = r.uniform_(*self.cfg.roll)
    nominal = r.uniform_(0.0, 1.0) <= self.cfg.rel_nominal_envs
    self.posture[env_ids[nominal]] = 0.0

  def _update_command(self) -> None:
    pass

  def _update_metrics(self) -> None:
    steps = self.cfg.resampling_time_range[1] / self._env.step_dt
    self.metrics["error_height"] += height_error(self._env, self.posture).abs() / steps
    self.metrics["error_tilt"] += tilt_error(self._env, self.posture) / steps


def gravity_target(posture: torch.Tensor) -> torch.Tensor:
  pitch, roll = posture[:, 1], posture[:, 2]
  return torch.stack([torch.sin(pitch), -torch.sin(roll) * torch.cos(pitch),
                      -torch.cos(roll) * torch.cos(pitch)], dim=-1)


def height_error(env, posture: torch.Tensor, asset_cfg: SceneEntityCfg = _ROBOT):
  z = env.scene[asset_cfg.name].data.root_link_pos_w[:, 2]
  return z - (robot.BODY["stand_height"] + posture[:, 0])


def tilt_error(env, posture: torch.Tensor, asset_cfg: SceneEntityCfg = _ROBOT):
  g = env.scene[asset_cfg.name].data.projected_gravity_b
  return torch.norm(g - gravity_target(posture), dim=-1)


def track_height(env, command_name: str, std: float) -> torch.Tensor:
  posture = env.command_manager.get_command(command_name)
  return torch.exp(-torch.square(height_error(env, posture)) / std ** 2)


def track_tilt(env, command_name: str, std: float) -> torch.Tensor:
  posture = env.command_manager.get_command(command_name)
  return torch.exp(-torch.square(tilt_error(env, posture)) / std ** 2)


def posture_env_cfg(play: bool = False):
  cfg = flat_env_cfg(play=play)
  cfg.commands["posture"] = PostureCommandCfg(resampling_time_range=(3.0, 8.0))
  for group in ("actor", "critic"):
    cfg.observations[group].terms["posture"] = ObservationTermCfg(
      func=mdp.generated_commands, params={"command_name": "posture"})
  cfg.rewards.pop("upright", None)
  cfg.rewards["track_height"] = RewardTermCfg(
    func=track_height, weight=1.0, params={"command_name": "posture", "std": 0.04})
  cfg.rewards["track_tilt"] = RewardTermCfg(
    func=track_tilt, weight=1.0, params={"command_name": "posture", "std": 0.15})
  pose = cfg.rewards["pose"]
  pose.weight = 0.3
  loose = {r".*_(hip_abd|hip_flex)": 0.5, r".*_knee": 1.0}
  pose.params["std_standing"] = pose.params["std_walking"] = loose
  pose.params["std_running"] = loose
  return cfg
