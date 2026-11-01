"""Falling and getting up (#507, stage 5): the humanoid from the floor to its
stand -- lying on its back, on its front, dropped any way up, or tipping
over from standing -- by a policy of its own, which the served body hands
its legs AND arms to once it is falling, and back once it stands.

The quadruped's get-up (#389, at `quadruped-final`) is the template, with
what a biped needs:
  - all twenty-four joints are the policy's: a body on the floor pushes off
    with its hands, and its arms meet its trunk and legs
    (`models/humanoid.xml`'s pairs), so a push is one a real arm could make;
  - a LIFT at the pelvis, a share of the body's weight that falls to
    nothing over training (`ASSIST_STAGES`): standing up from the floor is
    found far sooner held up than not (HoST, Huang et al. 2025, pulled its
    G1 up so);
  - the rise is paid as it comes, the pelvis's height off the floor, and
    being upright from the release (a reward held back killed the
    quadruped's roll, #389); the STAND -- the height, the pose, both soles
    down -- only from `BUDGET_S`, so standing sooner earns nothing;
  - a fall is part of it: a quarter of the episodes start TIPPING, the body
    upright and turning over at a shove's speed, and a hit harder than
    `IMPACT_FREE` body weights on the trunk, the head or a limb is charged,
    so the policy that takes a falling body over learns to land on its
    arms and legs rather than its head.

The charges set the pace, as the quadruped's do: torque past each part's
continuous rating and harder past 60 % of its peak, the joints' speed and
acceleration, the pelvis rising faster than `RISE_FREE_MPS`, the action's
rate -- ramped up over training (`GENTLE_STAGES`).
"""

import math

import torch
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.terminations import nan_detection
from mjlab.managers import TerminationTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from humanoid_train import robot
from humanoid_train.walk import blown_up

_ROBOT = SceneEntityCfg("robot")
JOINTS = robot.LEGS + robot.ARMS

#: Episode seconds: a fall or a body lying still, then time to get up and
#: stand quietly.
EPISODE_S = 10.0
#: How each episode starts, and the share of them: on its back, on its
#: front or on a side (each a few centimetres up, its joints near the
#: stand's), dropped from 0.85-1.05 m any way up with its joints anywhere
#: near the stand's, or TIPPING from its stand at a shove's speed.
KINDS = ("back", "front", "side", "dropped", "tipping")
STARTS = {"back": 0.3, "front": 0.25, "side": 0.0, "dropped": 0.2, "tipping": 0.25}
#: A side start's pelvis, m: clear of the floor with an arm under it.
SIDE_M = 0.22
#: A tipping start's speed at the pelvis, m/s, turning the body over its
#: feet (0.23 m/s is a 10 N*s shove, which the standing policy stands
#: through: `humanoid_spike.py --shove`).
TIP_MPS = (0.6, 1.4)
#: The stand pays from this long after the release, s.
BUDGET_S = 3.0
#: The pelvis's height lying, m: the rise is paid from here to the stand's.
LYING_M = 0.10
#: Past 60 % of a part's peak is charged harder (#389's bound).
CAP_SHARE = 0.6
#: The pelvis may rise this fast for nothing, m/s.
RISE_FREE_MPS = 0.5
#: A hit on anything but a sole up to this many body weights is free; past
#: it each body weight is charged (`impact`).
IMPACT_FREE = 2.0
#: The charges for getting up hard, at full weight, and their share of it
#: from each iteration on.
GENTLE = {
  "torque_over_rating": -0.5,
  "torque_over_cap": -5.0,
  "joint_vel_l2": -0.005,
  "joint_acc_l2": -1e-7,
  "rise_speed": -5.0,
  "action_rate_l2": -0.02,
  "impact": -0.2,
}
GENTLE_STAGES = ((0, 0.1), (500, 0.25), (1000, 0.5), (1500, 1.0))
#: The lift at the pelvis, a share of the body's weight, from each
#: iteration on: held up, the body finds its way to its feet; then it must
#: carry itself.
ASSIST_STAGES = ((0, 0.5), (800, 0.3), (1600, 0.15), (2400, 0.0))
STEPS_PER_ITERATION = 24
#: `-Getup-Head`, a get-up that stands carried on: the head's own hits
#: charged past `HEAD_FREE` body weights at `HEAD_WEIGHT` a body weight
#: (the rewards are scaled by the step: a 5 kN hit costs a tenth of an
#: episode's pay), and more of the episodes a fall (`HEAD_STARTS`). ⚠ At
#: `impact`'s charge alone, taking a falling body over it let the head
#: strike at 5.5-6.9 m/s falling forward or sideways, harder than the
#: standing policy's own falls (`humanoid_spike.py --fall`).
HEAD_FREE, HEAD_WEIGHT = 0.5, -20.0
#: ⚠ With no side among its starts (dropped any way up was 0.2), it lost
#: getting up off one side, which the get-up before it had (getup2).
HEAD_STARTS = {"back": 0.15, "front": 0.15, "side": 0.15, "dropped": 0.15, "tipping": 0.4}

#: An action of 1 moves a joint's target this far, rad: the joints a
#: get-up swings through their ranges (the hips' pitch, the knees, the
#: shoulders, the elbows) the furthest.
ACTION_SCALE = {
  j: (0.6 if any(k in j for k in ("hip_pitch", "knee", "shoulder_pitch", "shoulder_roll",
                                   "elbow"))
      else 0.15 if "ankle_roll" in j else 0.3)
  for j in JOINTS
}
#: The trunk's and the limbs' geoms, whose hits on the floor are charged.
STRUCK = (("pelvis", "torso", "head")
          + tuple(f"{s}_{g}" for s in ("left", "right")
                  for g in ("thigh", "shank", "upper_arm", "forearm", "gripper")))


def _up(env, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
  """1 standing, 0 on its side or lying, -1 on its head."""
  return -env.scene[asset_cfg.name].data.projected_gravity_b[:, 2]


def _z(env, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
  return env.scene[asset_cfg.name].data.root_link_pos_w[:, 2]


def _since_release(env) -> torch.Tensor:
  return env.episode_length_buf * env.step_dt


def _paid(env, budget_s: float) -> torch.Tensor:
  return (_since_release(env) >= budget_s).float()


# ---- the starts --------------------------------------------------------------------

def _quat_from_rpy(roll, pitch, yaw):
  cr, sr = torch.cos(roll / 2), torch.sin(roll / 2)
  cp, sp = torch.cos(pitch / 2), torch.sin(pitch / 2)
  cy, sy = torch.cos(yaw / 2), torch.sin(yaw / 2)
  return torch.stack([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                      cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy],
                     dim=-1)


def reset_fallen(env, env_ids, starts: dict | None = None,
                 asset_cfg: SceneEntityCfg = _ROBOT) -> None:
  """Each episode's start, drawn by `starts` (`STARTS`'s kinds)."""
  env_ids = envs_mdp.events.resolve_env_ids(env, env_ids)
  asset = env.scene[asset_cfg.name]
  shares = [(starts or STARTS).get(k, 0.0) for k in KINDS]
  n, dev = len(env_ids), env.device
  u = torch.rand(n, device=dev) * sum(shares)
  edges = torch.cumsum(torch.tensor(shares, device=dev), 0)
  kind = torch.searchsorted(edges, u.contiguous()).clamp(max=len(KINDS) - 1)
  back, front, side, dropped, tipping = (kind == k for k in range(len(KINDS)))
  yaw = torch.empty(n, device=dev).uniform_(-math.pi, math.pi)
  roll = torch.empty(n, device=dev).uniform_(-0.3, 0.3)
  pitch = torch.where(back, torch.full_like(yaw, -math.pi / 2),
                      torch.full_like(yaw, math.pi / 2))
  flip = torch.where(torch.rand(n, device=dev) < 0.5, 1.0, -1.0)
  roll = torch.where(side, flip * math.pi / 2 + roll, roll)
  pitch = torch.where(side, torch.zeros_like(pitch), pitch)
  roll = torch.where(dropped, torch.empty(n, device=dev).uniform_(-math.pi, math.pi), roll)
  pitch = torch.where(dropped, torch.empty(n, device=dev).uniform_(-math.pi / 2, math.pi / 2),
                      pitch)
  roll = torch.where(tipping, torch.zeros_like(roll), roll)
  pitch = torch.where(tipping, torch.zeros_like(pitch), pitch)
  pos = env.scene.env_origins[env_ids].clone()
  pos[:, 2] = torch.where(dropped, torch.empty(n, device=dev).uniform_(0.85, 1.05),
                          torch.full_like(yaw, 0.12))
  pos[:, 2] = torch.where(tipping, torch.full_like(yaw, robot.BODY["stand_height"] + 0.01),
                          pos[:, 2])
  # On a side the arm under it reaches 0.2 m from the pelvis.
  pos[:, 2] = torch.where(side, torch.full_like(yaw, SIDE_M), pos[:, 2])
  vel = torch.zeros(n, 6, device=dev)
  way = torch.empty(n, device=dev).uniform_(-math.pi, math.pi)
  speed = torch.empty(n, device=dev).uniform_(*TIP_MPS) * tipping.float()
  vel[:, 0], vel[:, 1] = speed * torch.cos(way), speed * torch.sin(way)
  turn = speed / robot.BODY["stand_height"]
  vel[:, 3], vel[:, 4] = -turn * torch.sin(way), turn * torch.cos(way)
  asset.write_root_state_to_sim(
    torch.cat([pos, _quat_from_rpy(roll, pitch, yaw), vel], dim=-1), env_ids)
  q0 = asset.data.default_joint_pos[env_ids]
  spread = torch.where(dropped, 0.6, torch.where(tipping, 0.0, 0.3))[:, None]
  q = q0 + spread * (2 * torch.rand_like(q0) - 1)
  limits = asset.data.soft_joint_pos_limits[env_ids]
  q = torch.max(torch.min(q, limits[..., 1]), limits[..., 0])
  asset.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)


def assist(env, env_ids, share: float, asset_cfg: SceneEntityCfg) -> None:
  """The lift at the pelvis for the episodes starting: `share` of the
  body's weight, upward, held until their next start. ⚠ `asset_cfg` must be
  in the event's params, where the manager resolves its body: a default
  here stays unresolved, and the lift went on every body (the robot flew)."""
  env_ids = envs_mdp.events.resolve_env_ids(env, env_ids)
  asset = env.scene[asset_cfg.name]
  forces = torch.zeros(len(env_ids), 1, 3, device=env.device)
  forces[..., 2] = share * robot.MASS_KG * 9.81
  asset.write_external_wrench_to_sim(forces, torch.zeros_like(forces), env_ids=env_ids,
                                     body_ids=asset_cfg.body_ids)


def assist_curriculum(env, env_ids, stages=ASSIST_STAGES):
  """The lift's share for the iteration training is at."""
  it = env.common_step_counter // STEPS_PER_ITERATION
  share = [s for at, s in stages if it >= at][-1]
  env.event_manager.get_term_cfg("assist").params["share"] = share
  return torch.tensor(share)


# ---- rewards -----------------------------------------------------------------------

def upright(env) -> torch.Tensor:
  """The right way up, from the release."""
  return torch.square((1.0 + _up(env)) / 2.0)


def rise(env) -> torch.Tensor:
  """The pelvis off the floor, from lying to the stand's height, from the
  release: kneeling is half way."""
  return torch.clamp((_z(env) - LYING_M) / (robot.BODY["stand_height"] - LYING_M), 0.0, 1.0)


def stand_height(env, std: float, budget_s: float) -> torch.Tensor:
  gate = (_up(env) > 0.9).float()
  return _paid(env, budget_s) * gate * torch.exp(
    -torch.square(_z(env) - robot.BODY["stand_height"]) / std ** 2)


def stand_pose(env, std: float, budget_s: float, asset_cfg: SceneEntityCfg = _ROBOT):
  asset = env.scene[asset_cfg.name]
  err = torch.sum(torch.square(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
  gate = (_up(env) > 0.9).float()
  return _paid(env, budget_s) * gate * torch.exp(-err / std ** 2)


def feet_down(env, sensor_name: str, budget_s: float) -> torch.Tensor:
  """Both soles on the floor, standing tall."""
  found = env.scene[sensor_name].data.found
  tall = (_z(env) > robot.BODY["stand_height"] - 0.1).float()
  return _paid(env, budget_s) * tall * (found > 0).all(dim=1).float()


def _ratio(env, asset_cfg: SceneEntityCfg, of: torch.Tensor) -> torch.Tensor:
  tau = env.scene[asset_cfg.name].data.actuator_force[:, asset_cfg.actuator_ids]
  return tau.abs() / of.to(tau.device)


def torque_over_rating(env, asset_cfg: SceneEntityCfg, rated: torch.Tensor):
  """Each joint's torque past its part's continuous rating, in ratings,
  squared."""
  return torch.sum(torch.square(torch.clamp(_ratio(env, asset_cfg, rated) - 1.0, min=0.0)), 1)


def torque_over_cap(env, asset_cfg: SceneEntityCfg, cap: torch.Tensor):
  """Each joint's torque past `CAP_SHARE` of its part's peak, in caps."""
  return torch.sum(torch.clamp(_ratio(env, asset_cfg, cap) - 1.0, min=0.0), 1)


def rise_speed(env, free: float, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
  """The pelvis rising faster than `free`, squared (falling is free)."""
  vz = env.scene[asset_cfg.name].data.root_link_lin_vel_w[:, 2]
  return torch.square(torch.clamp(vz - free, min=0.0))


def impact(env, sensor_name: str, free: float) -> torch.Tensor:
  """The hardest hit this policy step on each struck part past `free` body
  weights, in body weights, summed over the parts."""
  hist = env.scene[sensor_name].data.force_history          # [B, N, H, 3]
  peak = torch.norm(hist, dim=-1).amax(dim=-1) / (robot.MASS_KG * 9.81)
  return torch.sum(torch.clamp(peak - free, min=0.0), dim=1)


def gentle_curriculum(stages=GENTLE_STAGES) -> dict:
  return {f"gentle_{name}": CurriculumTermCfg(
    func=envs_mdp.reward_curriculum,
    params={"reward_name": name, "stages": [
      {"step": it * STEPS_PER_ITERATION, "weight": w * share} for it, share in stages]})
    for name, w in GENTLE.items()}


# ---- the critic's extras -------------------------------------------------------------

def root_height(env) -> torch.Tensor:
  return _z(env)[:, None]


def since_release(env) -> torch.Tensor:
  return _since_release(env)[:, None] / EPISODE_S


def joint_torque(env, asset_cfg: SceneEntityCfg, peak: torch.Tensor) -> torch.Tensor:
  tau = env.scene[asset_cfg.name].data.actuator_force[:, asset_cfg.actuator_ids]
  return tau / peak.to(tau.device)


def getup_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = make_velocity_env_cfg()
  cfg.scene.entities = {"robot": robot.get_robot_cfg()}
  cfg.episode_length_s = EPISODE_S
  cfg.sim.njmax = 600
  cfg.sim.nconmax = None
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None
  terrain = ContactMatch(mode="body", pattern="terrain")
  cfg.scene.sensors = (
    ContactSensorCfg(
      name="feet_ground_contact",
      primary=ContactMatch(mode="geom", pattern=robot.SOLES, entity="robot"),
      secondary=terrain, fields=("found", "force"), reduce="netforce", num_slots=1),
    ContactSensorCfg(
      name="struck",
      primary=ContactMatch(mode="geom", pattern=STRUCK, entity="robot"),
      secondary=terrain, fields=("found", "force"), reduce="netforce", num_slots=1,
      history_length=robot.DECIMATION),
  )

  # Nothing to track: the twist stays at zero for the stock terms that read it.
  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  twist.ranges.lin_vel_x = twist.ranges.lin_vel_y = twist.ranges.ang_vel_z = (0.0, 0.0)
  twist.rel_standing_envs = 1.0
  twist.rel_heading_envs = 0.0
  twist.rel_forward_envs = 0.0
  twist.heading_command = False
  twist.ranges.heading = None

  action = cfg.actions["joint_pos"]
  assert isinstance(action, JointPositionActionCfg)
  action.actuator_names = JOINTS
  action.preserve_order = True
  action.scale = dict(ACTION_SCALE)

  actor = cfg.observations["actor"].terms
  for name in ("base_lin_vel", "command", "height_scan"):
    del actor[name]
  critic = cfg.observations["critic"].terms
  for name in ("command", "height_scan", "foot_height", "foot_air_time", "foot_contact",
               "foot_contact_forces"):
    critic.pop(name, None)
  driven = SceneEntityCfg("robot", actuator_names=JOINTS, preserve_order=True)
  peak = torch.tensor([robot.JOINTS[j]["peak_torque"] for j in JOINTS])
  rated = torch.tensor([robot.JOINTS[j]["rated_torque"] for j in JOINTS])
  critic["root_height"] = ObservationTermCfg(func=root_height)
  critic["since_release"] = ObservationTermCfg(func=since_release)
  critic["joint_torque"] = ObservationTermCfg(
    func=joint_torque, params={"asset_cfg": driven, "peak": peak})

  joints = SceneEntityCfg("robot", joint_names=(".*",))
  events = {k: cfg.events[k] for k in ("foot_friction", "encoder_bias", "base_com")}
  events["foot_friction"].params["asset_cfg"].geom_names = robot.SOLES
  events["base_com"].params["asset_cfg"].body_names = (robot.ROOT,)
  events["rotor_inertia"] = EventTermCfg(
    mode="startup", func=dr.joint_armature,
    params={"asset_cfg": joints, "operation": "scale", "ranges": (0.8, 1.2)})
  events["joint_friction"] = EventTermCfg(
    mode="startup", func=dr.joint_friction,
    params={"asset_cfg": joints, "operation": "scale", "ranges": (0.5, 2.0)})
  events["body_mass"] = EventTermCfg(
    mode="startup", func=dr.body_mass,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=(robot.ROOT,)),
            "operation": "scale", "ranges": (0.9, 1.1)})
  events["effort_limit"] = EventTermCfg(
    mode="startup", func=dr.effort_limits,
    params={"asset_cfg": SceneEntityCfg("robot", actuator_names=(".*",)),
            "operation": "scale", "effort_limit_range": (0.9, 1.1)})
  events["reset_base"] = EventTermCfg(func=reset_fallen, mode="reset")
  events["assist"] = EventTermCfg(
    func=assist, mode="reset",
    params={"share": ASSIST_STAGES[0][1],
            "asset_cfg": SceneEntityCfg("robot", body_names=(robot.ROOT,))})
  cfg.events = events

  share = GENTLE_STAGES[0][1]
  gentle = {
    "torque_over_rating": (torque_over_rating, {"asset_cfg": driven, "rated": rated}),
    "torque_over_cap": (torque_over_cap, {"asset_cfg": driven, "cap": CAP_SHARE * peak}),
    "joint_vel_l2": (envs_mdp.rewards.joint_vel_l2, {}),
    "joint_acc_l2": (envs_mdp.rewards.joint_acc_l2, {}),
    "rise_speed": (rise_speed, {"free": RISE_FREE_MPS}),
    "action_rate_l2": (mdp.action_rate_l2, {}),
    "impact": (impact, {"sensor_name": "struck", "free": IMPACT_FREE}),
  }
  assert gentle.keys() == GENTLE.keys()
  cfg.rewards = {
    "upright": RewardTermCfg(func=upright, weight=1.0),
    "rise": RewardTermCfg(func=rise, weight=1.0),
    "stand_height": RewardTermCfg(func=stand_height, weight=1.0,
                                  params={"std": 0.05, "budget_s": BUDGET_S}),
    "stand_pose": RewardTermCfg(func=stand_pose, weight=0.5,
                                params={"std": 0.8, "budget_s": BUDGET_S}),
    "feet_down": RewardTermCfg(func=feet_down, weight=0.5,
                               params={"sensor_name": "feet_ground_contact",
                                       "budget_s": BUDGET_S}),
    **{name: RewardTermCfg(func=func, weight=GENTLE[name] * share, params=params)
       for name, (func, params) in gentle.items()},
    "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
  }
  cfg.curriculum = {**gentle_curriculum(),
                    "assist": CurriculumTermCfg(func=assist_curriculum, params={})}
  cfg.terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "nan": TerminationTermCfg(func=nan_detection),
    "blown_up": TerminationTermCfg(func=blown_up),
  }
  cfg.viewer.body_name = robot.ROOT
  cfg.viewer.distance = 3.0
  cfg.viewer.elevation = -10.0
  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}
    cfg.events["assist"].params["share"] = 0.0
  return cfg


def getup_head_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """`-Getup-Head`: the get-up carried on, falling without its head
  (`HEAD_FREE`): no lift and every charge at its full weight from the
  start, as the get-up ended."""
  cfg = getup_env_cfg(play)
  cfg.scene.sensors = cfg.scene.sensors + (
    ContactSensorCfg(
      name="head_struck", primary=ContactMatch(mode="geom", pattern=("head",), entity="robot"),
      secondary=ContactMatch(mode="body", pattern="terrain"), fields=("found", "force"),
      reduce="netforce", num_slots=1, history_length=robot.DECIMATION),
  )
  cfg.rewards["head_struck"] = RewardTermCfg(
    func=impact, weight=HEAD_WEIGHT, params={"sensor_name": "head_struck", "free": HEAD_FREE})
  for name, w in GENTLE.items():
    cfg.rewards[name].weight = w
  cfg.curriculum = {}
  cfg.events["assist"].params["share"] = 0.0
  cfg.events["reset_base"].params["starts"] = dict(HEAD_STARTS)
  return cfg
