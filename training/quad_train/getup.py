"""Getting up (#377 item 3), rising rather than springing (#389): from a
fall, and from lying on the belly.

A policy of its own, beside the walking one (Playground's Go1 splits them the
same way): every episode starts either LYING on the belly pack in the rest
pose (`models/quadruped.json`'s `lie_qpos`), or dropped from 0.35-0.55 m in
a random orientation with its joints anywhere in range, and is paid for
being upright, at the standing height, in the standing pose, quietly. The
served controller switches to it when the body is down and back to walking
when it stands.

#377's policy stood from the belly in 0.2 s by driving every joint to its
22 N*m peak: paid for every step it stood, the sooner the better, it sprang.
Here the stand pays only from `BUDGET_S` after the release, so standing
sooner earns nothing, and charges set the pace: a joint's torque over the
motor's continuous rating, and harder over `TORQUE_CAP` (#389's bound), the
joints' speed and acceleration, and the torso rising faster than
`RISE_FREE_MPS`. The charges ramp up over training (`GENTLE_STAGES`).
Being the right way up still pays from the release: with it held back to
`BUDGET_S` too, no run ever learnt to roll off its back (SimNotes, "A
gentler get-up").

⚠ mjlab's `upright` reward reads only the sideways tilt, so it scores a body
lying on its BACK as perfectly upright; this task's `belly_down` reads the
sign of gravity in the body frame.
"""

import math

import torch
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers import TerminationTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from quad_train import robot
from quad_train.task import _common

#: Episode seconds: a fall, then time to get up and stand still.
EPISODE_S = 6.0
#: Share of episodes that start lying on the belly (the rest pose) rather
#: than fallen at random.
LIE_FRACTION = 0.3
#: The stand pays from this long after the release, s: a stand-up that
#: takes this long earns what a spring earns.
BUDGET_S = 2.0
#: #389's bound on a joint's torque while getting up: 60 % of the peak.
TORQUE_CAP = 0.6 * robot.MOTOR["peak_torque"]
#: The torso may rise this fast for nothing, m/s: from the belly to the
#: stand (0.22 m) at this speed is ~1 s.
RISE_FREE_MPS = 0.25
#: The charges for getting up hard, per second, at full weight.
GENTLE = {
  "torque_over_rating": -1.0,
  "torque_over_cap": -10.0,
  "joint_vel_l2": -0.01,
  "joint_acc_l2": -2.5e-7,
  "rise_speed": -10.0,
  "action_rate_l2": -0.02,
}
#: The charges' share of full weight, from each iteration on.
GENTLE_STAGES = ((0, 0.1), (300, 0.25), (600, 0.5), (900, 1.0))
#: Environment steps an iteration (`rl.ppo_runner_cfg`): the curriculum
#: counts steps.
STEPS_PER_ITERATION = 24

_ROBOT = SceneEntityCfg("robot")


def _quat_from_rpy(roll, pitch, yaw):
  cr, sr = torch.cos(roll / 2), torch.sin(roll / 2)
  cp, sp = torch.cos(pitch / 2), torch.sin(pitch / 2)
  cy, sy = torch.cos(yaw / 2), torch.sin(yaw / 2)
  return torch.stack([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                      cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy],
                     dim=-1)


def reset_down(env, env_ids, lie_fraction: float = LIE_FRACTION,
               asset_cfg: SceneEntityCfg = _ROBOT) -> None:
  """Lying on the belly, or dropped in a random orientation."""
  env_ids = envs_mdp.events.resolve_env_ids(env, env_ids)
  asset = env.scene[asset_cfg.name]
  n, dev = len(env_ids), env.device
  lie = torch.rand(n, device=dev) < lie_fraction
  roll = torch.empty(n, device=dev).uniform_(-math.pi, math.pi)
  pitch = torch.empty(n, device=dev).uniform_(-math.pi / 2, math.pi / 2)
  yaw = torch.empty(n, device=dev).uniform_(-math.pi, math.pi)
  roll = torch.where(lie, torch.zeros_like(roll), roll)
  pitch = torch.where(lie, torch.zeros_like(pitch), pitch)
  pos = env.scene.env_origins[env_ids].clone()
  drop = torch.empty(n, device=dev).uniform_(0.35, 0.55)
  pos[:, 2] = torch.where(lie, torch.full_like(drop, robot.BODY["belly_depth"] + 0.003),
                          drop)
  root = torch.cat([pos, _quat_from_rpy(roll, pitch, yaw),
                    torch.zeros(n, 6, device=dev)], dim=-1)
  asset.write_root_state_to_sim(root, env_ids)
  limits = asset.data.soft_joint_pos_limits[env_ids]
  q = limits[..., 0] + torch.rand_like(limits[..., 0]) * (limits[..., 1] - limits[..., 0])
  lie_q = torch.tensor(robot.BODY["lie_qpos"], device=dev).expand(n, -1)
  q = torch.where(lie[:, None], lie_q, q)
  asset.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)


def _belly_down(env, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
  """1 standing the right way up, 0 on the side, -1 on the back."""
  return -env.scene[asset_cfg.name].data.projected_gravity_b[:, 2]


def _since_release(env) -> torch.Tensor:
  return env.episode_length_buf * env.step_dt


def _paid(env, budget_s: float) -> torch.Tensor:
  """1 from `budget_s` after the release, 0 before."""
  return (_since_release(env) >= budget_s).float()


def belly_down(env, budget_s: float, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
  return _paid(env, budget_s) * torch.square((1.0 + _belly_down(env, asset_cfg)) / 2.0)


def stand_height(env, std: float, budget_s: float,
                 asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
  """The torso at the standing height, paid only the right way up."""
  z = env.scene[asset_cfg.name].data.root_link_pos_w[:, 2]
  gate = (_belly_down(env, asset_cfg) > 0.8).float()
  return (_paid(env, budget_s) * gate
          * torch.exp(-torch.square(z - robot.BODY["stand_height"]) / std ** 2))


def stand_pose(env, std: float, budget_s: float,
               asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
  """The legs in the standing pose, paid only the right way up."""
  asset = env.scene[asset_cfg.name]
  err = torch.sum(torch.square(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
  gate = (_belly_down(env, asset_cfg) > 0.8).float()
  return _paid(env, budget_s) * gate * torch.exp(-err / std ** 2)


def torque_over_rating(env, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
  """Each joint's torque past the motor's continuous rating, in ratings,
  squared: a stance is free, a joint at the peak ~5."""
  tau = env.scene[asset_cfg.name].data.actuator_force
  rated = robot.MOTOR["rated_torque"]
  return torch.sum(torch.square(torch.clamp(tau.abs() - rated, min=0.0) / rated), dim=1)


def torque_over_cap(env, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
  """Each joint's torque past `TORQUE_CAP`, in caps."""
  tau = env.scene[asset_cfg.name].data.actuator_force
  return torch.sum(torch.clamp(tau.abs() - TORQUE_CAP, min=0.0) / TORQUE_CAP, dim=1)


def rise_speed(env, free: float, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
  """The torso rising faster than `free`, m/s, squared (falling is free:
  every drop starts with one)."""
  vz = env.scene[asset_cfg.name].data.root_link_lin_vel_w[:, 2]
  return torch.square(torch.clamp(vz - free, min=0.0))


def root_height(env, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
  return env.scene[asset_cfg.name].data.root_link_pos_w[:, 2:3]


def since_release(env) -> torch.Tensor:
  """The critic's clock, in episodes."""
  return _since_release(env)[:, None] / EPISODE_S


def leg_torque(env, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
  return env.scene[asset_cfg.name].data.actuator_force


def gentle_curriculum(stages=GENTLE_STAGES, weights=None) -> dict:
  """One reward curriculum per charge: its weight times each stage's share."""
  weights = GENTLE if weights is None else weights
  return {f"gentle_{name}": CurriculumTermCfg(
    func=envs_mdp.reward_curriculum,
    params={"reward_name": name, "stages": [
      {"step": it * STEPS_PER_ITERATION, "weight": w * share} for it, share in stages]})
    for name, w in weights.items()}


def getup_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = _common(make_velocity_env_cfg())
  cfg.episode_length_s = EPISODE_S
  cfg.sim.njmax = 600
  cfg.sim.nconmax = None
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None
  cfg.scene.sensors = tuple(s for s in cfg.scene.sensors
                            if s.name not in ("terrain_scan",))

  # The actor: the walking policy's senses without a command (42 numbers).
  actor = cfg.observations["actor"].terms
  for name in ("height_scan", "command"):
    actor.pop(name, None)
  critic = cfg.observations["critic"].terms
  for name in ("height_scan", "command"):
    critic.pop(name, None)
  # The critic, which never runs on the robot, is shown what the reward
  # reads: the height, the clock the stand pays by, the torques.
  critic["root_height"] = ObservationTermCfg(func=root_height)
  critic["since_release"] = ObservationTermCfg(func=since_release)
  critic["leg_torque"] = ObservationTermCfg(
    func=leg_torque, scale=1.0 / robot.MOTOR["peak_torque"])

  # Nothing to track: a still command (the twist term stays for the managers
  # that expect it) and no velocity curriculum.
  twist = cfg.commands["twist"]
  twist.ranges.lin_vel_x = twist.ranges.lin_vel_y = twist.ranges.ang_vel_z = (0.0, 0.0)
  twist.rel_standing_envs = 1.0

  cfg.events["reset_base"] = EventTermCfg(func=reset_down, mode="reset",
                                          params={"lie_fraction": LIE_FRACTION})
  cfg.events.pop("reset_robot_joints", None)
  cfg.events.pop("push_robot", None)

  share = GENTLE_STAGES[0][1]
  gentle = {
    "torque_over_rating": (torque_over_rating, {}),
    "torque_over_cap": (torque_over_cap, {}),
    "joint_vel_l2": (envs_mdp.rewards.joint_vel_l2, {}),
    "joint_acc_l2": (envs_mdp.rewards.joint_acc_l2, {}),
    "rise_speed": (rise_speed, {"free": RISE_FREE_MPS}),
    "action_rate_l2": (mdp.action_rate_l2, {}),
  }
  assert gentle.keys() == GENTLE.keys()
  cfg.rewards = {
    "belly_down": RewardTermCfg(func=belly_down, weight=1.0,
                                params={"budget_s": 0.0}),
    "stand_height": RewardTermCfg(func=stand_height, weight=1.0,
                                  params={"std": 0.05, "budget_s": BUDGET_S}),
    "stand_pose": RewardTermCfg(func=stand_pose, weight=0.5,
                                params={"std": 0.6, "budget_s": BUDGET_S}),
    **{name: RewardTermCfg(func=func, weight=GENTLE[name] * share, params=params)
       for name, (func, params) in gentle.items()},
    "joint_torques_l2": RewardTermCfg(func=envs_mdp.rewards.joint_torques_l2,
                                      weight=-1e-4),
    "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
  }
  cfg.curriculum = gentle_curriculum()
  cfg.terminations = {"time_out": TerminationTermCfg(func=mdp.time_out,
                                                     time_out=True)}
  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}
  return cfg
