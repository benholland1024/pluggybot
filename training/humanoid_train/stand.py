"""Standing (#507, stage 2): the humanoid balancing on planted feet at a
commanded posture -- the pelvis's height and the trunk's pitch, a crouch and
a bow that take the hands from the floor to a high shelf -- while its arms
reach where they are sent and its hands carry and pull.

The legs are the policy's; the arms are NOT. On the robot they follow an
inverse-kinematics reach (`pluggybot.humanoid.arms`), so here they follow a
script that sends them, joint by joint, somewhere new every few seconds
(`ScriptedArms`), with their own weight fed forward as the reach's commands
carry it. The policy sees their encoders and so where their mass is; it is
paid for the posture it was asked, for its feet staying where they are, and
charged for torque past a part's continuous rating. What disturbs it:
shoves to the trunk, a load in each hand (`HAND_KG`), and pulls and pushes
at a hand (`HAND_N`, `hand_pull`) -- a handle gripped and drawn back -- each
ramped up over the first iterations (`DISTURBANCE_STAGES`).

What differs from mjlab's velocity task, and why:
  - no walking: the twist command is held at zero (it stays, because the
    stock terms that key off it read "standing" from it);
  - the actor sees no base linear velocity (the body measures none) and no
    twist; it sees the posture it is asked, ramped as the reach ramps it;
  - "upright" is the commanded tilt, and the joint-pose reward keeps only
    the legs' lateral joints near the stand: a crouch and a bow are paid,
    not charged;
  - a fall is the pelvis under `FALLEN_M` or the tilt `TILT_LOST` off what
    was asked: a bow of 1.15 rad is past the stock 70-degree fall.
"""

from dataclasses import dataclass
import math

import torch
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, ObjRef, RingPatternCfg, TerrainHeightSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from humanoid_train import robot

_ROBOT = SceneEntityCfg("robot")

# ---- the posture command ------------------------------------------------------------

#: The pelvis's height off the stand's, m, and the trunk's pitch forward,
#: rad: from standing to a crouch and a bow that bring the shoulders to
#: 0.67 m, where a hand reaches the floor's 0.09 m (the static solve in
#: SimNotes, "The humanoid stands, reaches and grasps"). The hip folds to
#: -2.34 rad there, its thigh a centimetre off the torso.
HEIGHT = (-0.25, 0.0)
PITCH = (-0.1, 1.1)
#: How fast the asked posture moves toward a new one, m/s and rad/s: the
#: reach ramps it so (`pluggybot.humanoid.reach`), never a step.
HEIGHT_RATE, PITCH_RATE = 0.15, 0.6


@dataclass(kw_only=True)
class PostureCommandCfg(CommandTermCfg):
  entity_name: str = "robot"
  height: tuple[float, float] = HEIGHT
  pitch: tuple[float, float] = PITCH
  #: Share of goals at the stand itself, the posture most of a day is in.
  rel_nominal_envs: float = 0.3

  def build(self, env) -> "PostureCommand":
    return PostureCommand(self, env)


class PostureCommand(CommandTerm):
  """(height offset, pitch): a goal resampled now and then, and the command
  the policy is shown walking toward it at the reach's rates."""

  cfg: PostureCommandCfg

  def __init__(self, cfg: PostureCommandCfg, env):
    super().__init__(cfg, env)
    self.goal = torch.zeros(self.num_envs, 2, device=self.device)
    self.posture = torch.zeros(self.num_envs, 2, device=self.device)
    self.rate = torch.tensor([HEIGHT_RATE, PITCH_RATE], device=self.device)
    self.metrics["error_height"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_tilt"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.posture

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    r = torch.empty(len(env_ids), device=self.device)
    self.goal[env_ids, 0] = r.uniform_(*self.cfg.height)
    self.goal[env_ids, 1] = r.uniform_(*self.cfg.pitch)
    nominal = r.uniform_(0.0, 1.0) <= self.cfg.rel_nominal_envs
    self.goal[env_ids[nominal]] = 0.0

  def _update_command(self) -> None:
    step = self.rate * self._env.step_dt
    self.posture += torch.clamp(self.goal - self.posture, -step, step)

  def reset(self, env_ids=None):
    out = super().reset(env_ids)
    ids = slice(None) if env_ids is None else env_ids
    # An episode starts standing; the first goal is walked to.
    self.posture[ids] = 0.0
    return out

  def _update_metrics(self) -> None:
    steps = self.cfg.resampling_time_range[1] / self._env.step_dt
    self.metrics["error_height"] += height_error(self._env, self.posture).abs() / steps
    self.metrics["error_tilt"] += tilt_error(self._env, self.posture) / steps


def gravity_target(posture: torch.Tensor) -> torch.Tensor:
  """Gravity in the trunk's frame when it is pitched `posture[:, 1]`
  forward and not rolled."""
  pitch = posture[:, 1]
  return torch.stack([torch.sin(pitch), torch.zeros_like(pitch), -torch.cos(pitch)], dim=-1)


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


# ---- the arms: sent somewhere new every few seconds ------------------------------------

#: Where the script sends each arm joint, the LEFT side's range, rad (the
#: right mirrors roll and yaw): every reach the scenes make and more -- an
#: arm up past the head, down to the floor ahead, out to the side, folded
#: across -- inside each joint's range.
ARM_GOALS = {
  "shoulder_pitch": (-2.7, 0.6), "shoulder_roll": (0.0, 1.5),
  "shoulder_yaw": (-1.3, 1.3), "elbow": (0.0, 2.0),
  "wrist_pitch": (-1.4, 1.4), "wrist_roll": (-1.6, 1.6),
}
MIRRORED = ("shoulder_roll", "shoulder_yaw", "wrist_roll")
#: Seconds an arm takes to its next goal at most, and the fastest a joint
#: is sent, rad/s (the reach ramps its targets no faster, `arms.MAX_RATE`).
ARM_MOVE_S = (1.0, 4.0)
ARM_RATE = 1.5
#: Share of goals that are the stand's own hanging arms.
ARM_REST_SHARE = 0.25


@dataclass(kw_only=True)
class ScriptedArmsCfg(ActionTermCfg):
  entity_name: str = "robot"
  #: Share of goals that are the arms hanging, and how far toward each
  #: joint's `ARM_GOALS` range a goal may go from the hanging pose (1: all
  #: of it) -- walking carries its arms more quietly than a reach does.
  rest_share: float = ARM_REST_SHARE
  reach: float = 1.0

  def build(self, env) -> "ScriptedArms":
    return ScriptedArms(self, env)


class ScriptedArms(ActionTerm):
  """The arms as the policy meets them: not its to move (no action), each
  sent to a new goal every `ARM_MOVE_S`, its targets ramped at `ARM_RATE`,
  its weight fed forward (the joints' share of the bias force, as the
  reach's commands carry the arm's own weight off its model)."""

  def __init__(self, cfg: ScriptedArmsCfg, env):
    super().__init__(cfg, env)
    ids, names = self._entity.find_joints(robot.ARMS, preserve_order=True)
    self.ids = torch.tensor(ids, device=self.device, dtype=torch.long)
    lo, hi = [], []
    for name in names:
      side, kind = name.split("_", 1)
      a, b = ARM_GOALS[kind]
      if side == "right" and kind in MIRRORED:
        a, b = -b, -a
      lo.append(a)
      hi.append(b)
    self.rest = self._entity.data.default_joint_pos[:, self.ids].clone()
    mid = self.rest[0]
    self.lo = mid + cfg.reach * (torch.tensor(lo, device=self.device) - mid)
    self.hi = mid + cfg.reach * (torch.tensor(hi, device=self.device) - mid)
    self.target = self.rest.clone()
    self.goal = self.rest.clone()
    self.timer = torch.zeros(self.num_envs, device=self.device)
    self._raw = torch.zeros(self.num_envs, 0, device=self.device)

  @property
  def action_dim(self) -> int:
    return 0

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw

  def _new_goals(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    u = torch.rand(n, len(self.ids), device=self.device)
    goal = self.lo + u * (self.hi - self.lo)
    rest = torch.rand(n, 2, device=self.device) < self.cfg.rest_share
    half = len(self.ids) // 2
    goal[:, :half] = torch.where(rest[:, :1], self.rest[env_ids, :half], goal[:, :half])
    goal[:, half:] = torch.where(rest[:, 1:], self.rest[env_ids, half:], goal[:, half:])
    self.goal[env_ids] = goal
    self.timer[env_ids] = torch.empty(n, device=self.device).uniform_(*ARM_MOVE_S)

  def process_actions(self, actions: torch.Tensor) -> None:
    dt = self._env.step_dt
    self.timer -= dt
    due = torch.nonzero(self.timer <= 0).flatten()
    if len(due):
      self._new_goals(due)
    step = ARM_RATE * dt
    self.target += torch.clamp(self.goal - self.target, -step, step)

  def apply_actions(self) -> None:
    self._entity.set_joint_position_target(self.target, joint_ids=self.ids)
    bias = self._entity.data._joint_dof_field("qfrc_bias")[:, self.ids]
    self._entity.set_joint_effort_target(bias, joint_ids=self.ids)

  def reset(self, env_ids=None) -> None:
    ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
    if isinstance(ids, slice):
      ids = torch.arange(self.num_envs, device=self.device)[ids]
    self.target[ids] = self.rest[ids]
    self.goal[ids] = self.rest[ids]
    self.timer[ids] = torch.empty(len(ids), device=self.device).uniform_(0.0, ARM_MOVE_S[0])


# ---- feet that stay where they are -------------------------------------------------------

def feet_still(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """The soles' speed, squared and summed: a standing robot's feet stay
  where they are, and a step is the last resort (the reach's frame is the
  stance's)."""
  v = env.scene[asset_cfg.name].data.site_lin_vel_w[:, asset_cfg.site_ids]
  return torch.sum(torch.square(v).sum(dim=-1), dim=1)


class feet_anchor:
  """Each sole's distance across the floor from where the episode's stance
  put it, m, summed, whether it is down or lifted: a robot reaching from
  where it stands keeps its feet there, and a step a shove forces is
  charged until it steps back. ⚠ A charge on the soles' SPEED alone
  (`feet_still`) let a policy creep its feet 10 cm in 8 s -- 0.5 cm/s
  squared is nothing -- and an anchor RENEWED at each touchdown made a step
  free: bowing, a policy stepped its feet 40 cm. A reach's stance frame
  moves with them."""

  def __init__(self, cfg: RewardTermCfg, env):
    self.entity = env.scene["robot"]
    ids, _ = self.entity.find_sites(robot.FEET, preserve_order=True)
    self.ids = torch.tensor(ids, device=env.device, dtype=torch.long)
    self.anchor = torch.zeros(env.num_envs, len(ids), 2, device=env.device)
    self.valid = torch.zeros(env.num_envs, len(ids), dtype=torch.bool, device=env.device)

  def reset(self, env_ids=None) -> None:
    self.valid[slice(None) if env_ids is None else env_ids] = False

  def __call__(self, env, sensor_name: str) -> torch.Tensor:
    pos = self.entity.data.site_pos_w[:, self.ids, :2]
    down = env.scene[sensor_name].data.found > 0
    # The stance is where each sole first comes down in an episode.
    first = down & ~self.valid
    self.anchor = torch.where(first[..., None], pos, self.anchor)
    self.valid = self.valid | down
    return (torch.norm(pos - self.anchor, dim=-1) * self.valid).sum(dim=1)


def feet_down(env, sensor_name: str) -> torch.Tensor:
  """1 while both soles touch the ground."""
  found = env.scene[sensor_name].data.found
  return (found > 0).all(dim=1).float()


def torque_over_rating(env, asset_cfg: SceneEntityCfg, rated: torch.Tensor) -> torch.Tensor:
  """Each driven joint's torque past its part's continuous rating, as a
  share of it, summed: a pose held there heats its winding past its limit
  (SimNotes, "The humanoid body": a held torque is a stalled motor)."""
  tau = env.scene[asset_cfg.name].data.actuator_force[:, asset_cfg.actuator_ids]
  return torch.sum(torch.clamp(tau.abs() / rated.to(tau.device) - 1.0, min=0.0), dim=1)


# ---- falls ----------------------------------------------------------------------------

#: The pelvis's lowest before it is a fall, m (a crouch puts it at 0.48),
#: and the tilt's furthest off the command: |g - g_asked|, 0.9 is 54 deg.
FALLEN_M = 0.36
TILT_LOST = 0.9


def tilt_lost(env, command_name: str, limit: float) -> torch.Tensor:
  posture = env.command_manager.get_command(command_name)
  return tilt_error(env, posture) > limit


# ---- the disturbances, ramped ----------------------------------------------------------

#: A shove: the trunk's velocity changed by up to this, m/s and rad/s, every
#: `PUSH_EVERY_S` -- a hand on the shoulder, not a tackle: the balance margin
#: under shoves is measured in our physics after (`humanoid_spike.py --shove`).
PUSH = {"x": (-0.4, 0.4), "y": (-0.4, 0.4), "roll": (-0.3, 0.3), "pitch": (-0.3, 0.3),
        "yaw": (-0.4, 0.4)}
PUSH_EVERY_S = (2.0, 5.0)
#: What a hand carries, kg: from nothing to a full kettle (the shelves'
#: objects), each hand its own, held at the hand.
HAND_KG = (0.0, 2.0)
#: A pull or a push at a hand, N, held a while: a handle gripped and drawn
#: back, a door leaned on -- up to the `pull` verb's own most (its `force`,
#: `procedure.steps`), along any way, as the verb's is, at one hand at a
#: time (`hand_pull`, #544). ⚠ At 30 N a component, both hands at once, a
#: robot's 55 N pull toppled it, and it noted to keep under 50.
HAND_N = 60.0
HAND_PULL_S, HAND_REST_S = (0.5, 3.0), (1.0, 4.0)
#: The shoves' and the pulls' share of their full size, from each iteration
#: on: a policy shoved at full size from the first iteration learns to fall.
DISTURBANCE_STAGES = ((0, 0.25), (400, 0.5), (800, 0.75), (1200, 1.0))
STEPS_PER_ITERATION = 24


class hand_pull(envs_mdp.apply_body_impulse):
  """A pull or a push at ONE hand, drawn uniformly from the ball of radius
  `force_n`: any way, any size up to it, as the `pull` verb pulls one hand
  along any (dx, dy, dz) with at most its `force`. mjlab's impulse keeps
  its clocks -- the hold, then the rest -- and only the draw is ours: its
  own, uniform per component, loads both hands at once. With a finite
  `moment_nm`, a pull whose tipping moment about the middle of the feet,
  on the ground, is past it is scaled down to it as it begins: 60 N across
  a hand 1.5 m up tips any body whose feet stay put. A positive
  `longest_s` holds a pull up to that long (`HAND_PULL_S`'s upper end; a
  flag, where tyro reads the tuple as one number)."""

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    feet, _ = self._asset.find_sites(robot.FEET, preserve_order=True)
    self._feet = torch.tensor(feet, device=self._device, dtype=torch.long)

  def __call__(self, env, env_ids, force_n: float, duration_s, cooldown_s, asset_cfg,
               moment_nm: float = math.inf, longest_s: float = 0.0) -> None:
    was = self._active.clone()
    if longest_s > 0.0:
      duration_s = (duration_s[0], longest_s)
    super().__call__(env, env_ids, (0.0, 0.0), (0.0, 0.0), duration_s, cooldown_s, asset_cfg)
    began = (self._active & ~was).nonzero(as_tuple=False).squeeze(-1)
    n = len(began)
    if not n:
      return
    way = torch.randn(n, 3, device=self._device)
    way = way / way.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    size = force_n * torch.rand(n, 1, device=self._device) ** (1.0 / 3.0)
    forces = torch.zeros(n, self._num_bodies, 3, device=self._device)
    hand = torch.randint(self._num_bodies, (n,), device=self._device)
    forces[torch.arange(n, device=self._device), hand] = way * size
    if math.isfinite(moment_nm):
      mid = self._asset.data.site_pos_w[began][:, self._feet].mean(dim=1)
      mid[:, 2] = 0.0
      arm = self._asset.data.body_link_pos_w[began][:, self._body_ids] - mid[:, None, :]
      tip = torch.linalg.cross(arm, forces, dim=-1)[..., :2].norm(dim=-1)
      forces = forces * torch.clamp(moment_nm / tip.clamp_min(1e-9), max=1.0)[..., None]
    self._asset.write_external_wrench_to_sim(forces, torch.zeros_like(forces),
                                             env_ids=began, body_ids=self._body_ids)


def disturbance_scale(env, env_ids, stages=DISTURBANCE_STAGES, scale: float = 1.0):
  """Sets the shoves' and the pulls' size for the iteration training is at:
  the stage's share of their full size, times `scale` (a variant's flag,
  `--env.curriculum.disturbances.params.scale 0.6`)."""
  it = env.common_step_counter // STEPS_PER_ITERATION
  share = [s for at, s in stages if it >= at][-1] * scale
  push = env.event_manager.get_term_cfg("push_robot")
  push.params["velocity_range"] = {k: (lo * share, hi * share) for k, (lo, hi) in PUSH.items()}
  pull = env.event_manager.get_term_cfg("hand_pull")
  pull.params["force_n"] = HAND_N * share
  return torch.tensor(share)


# ---- the task -------------------------------------------------------------------------

def stand_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = make_velocity_env_cfg()
  cfg.scene.entities = {"robot": robot.get_robot_cfg()}

  # Flat ground, as the house's floors are (the stairs are stage 5's).
  cfg.sim.njmax = 300
  cfg.sim.nconmax = None
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None
  cfg.scene.sensors = tuple(s for s in cfg.scene.sensors if s.name != "terrain_scan")
  for sensor in cfg.scene.sensors:
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(ObjRef(type="site", name=f, entity="robot") for f in robot.FEET)
      sensor.pattern = RingPatternCfg.single_ring(radius=0.05, num_samples=6)
  feet_contact = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=robot.SOLES, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"), reduce="netforce", num_slots=1, track_air_time=True)
  cfg.scene.sensors = cfg.scene.sensors + (feet_contact,)

  # No walking: the twist stays at zero, which every stock term reads as
  # standing.
  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  twist.ranges.lin_vel_x = twist.ranges.lin_vel_y = twist.ranges.ang_vel_z = (0.0, 0.0)
  twist.rel_standing_envs = 1.0
  twist.rel_heading_envs = 0.0
  twist.rel_forward_envs = 0.0
  twist.heading_command = False
  twist.ranges.heading = None
  cfg.commands["posture"] = PostureCommandCfg(resampling_time_range=(3.0, 8.0))

  # The actions: the legs' twelve; the arms are scripted.
  action = cfg.actions["joint_pos"]
  assert isinstance(action, JointPositionActionCfg)
  action.actuator_names = robot.LEGS
  action.scale = robot.ACTION_SCALE
  action.preserve_order = True
  cfg.actions["arms"] = ScriptedArmsCfg()

  # The observations: no base velocity, no twist; the posture asked; every
  # joint's encoder, the arms' too (where their mass is).
  actor = cfg.observations["actor"].terms
  for name in ("base_lin_vel", "command", "height_scan"):
    del actor[name]
  actor["posture"] = ObservationTermCfg(
    func=mdp.generated_commands, params={"command_name": "posture"})
  critic = cfg.observations["critic"].terms
  for name in ("command", "height_scan"):
    del critic[name]
  critic["posture"] = ObservationTermCfg(
    func=mdp.generated_commands, params={"command_name": "posture"})
  critic["root_height"] = ObservationTermCfg(func=_root_height)

  # Randomisation: every number no datasheet gives, and what a hand carries.
  legs_and_arms = SceneEntityCfg("robot", joint_names=(".*",))
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = robot.SOLES
  cfg.events["foot_friction"].params["ranges"] = (0.4, 1.2)
  cfg.events["base_com"].params["asset_cfg"].body_names = (robot.ROOT,)
  cfg.events["rotor_inertia"] = EventTermCfg(
    mode="startup", func=dr.joint_armature,
    params={"asset_cfg": legs_and_arms, "operation": "scale", "ranges": (0.8, 1.2)})
  cfg.events["joint_friction"] = EventTermCfg(
    mode="startup", func=dr.joint_friction,
    params={"asset_cfg": legs_and_arms, "operation": "scale", "ranges": (0.5, 2.0)})
  cfg.events["body_mass"] = EventTermCfg(
    mode="startup", func=dr.body_mass,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=(robot.ROOT,)),
            "operation": "scale", "ranges": (0.9, 1.1)})
  cfg.events["hand_kg"] = EventTermCfg(
    mode="startup", func=dr.body_mass,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=robot.HANDS),
            "operation": "add", "ranges": HAND_KG})
  cfg.events["effort_limit"] = EventTermCfg(
    mode="startup", func=dr.effort_limits,
    params={"asset_cfg": SceneEntityCfg("robot", actuator_names=(".*",)),
            "operation": "scale", "effort_limit_range": (0.9, 1.1)})
  cfg.events["push_robot"].interval_range_s = PUSH_EVERY_S
  cfg.events["push_robot"].params["velocity_range"] = dict(PUSH)
  cfg.events["hand_pull"] = EventTermCfg(
    mode="step", func=hand_pull,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=robot.HANDS),
            "force_n": HAND_N, "moment_nm": math.inf, "longest_s": 0.0, "duration_s": HAND_PULL_S,
            "cooldown_s": HAND_REST_S})
  cfg.curriculum = {"disturbances": CurriculumTermCfg(func=disturbance_scale,
                                                      params={"scale": 1.0})}

  # Rewards.
  rewards = cfg.rewards
  for name in ("track_linear_velocity", "upright", "air_time", "foot_clearance",
               "foot_swing_height", "foot_slip", "soft_landing"):
    rewards.pop(name, None)
  rewards["track_angular_velocity"].weight = 0.5
  rewards["track_height"] = RewardTermCfg(
    func=track_height, weight=1.5, params={"command_name": "posture", "std": 0.05})
  rewards["track_tilt"] = RewardTermCfg(
    func=track_tilt, weight=1.5, params={"command_name": "posture", "std": 0.15})
  lateral = (".*_hip_roll", ".*_hip_yaw", ".*_ankle_roll")
  rewards["pose"].params["asset_cfg"] = SceneEntityCfg("robot", joint_names=lateral)
  for key in ("std_standing", "std_walking", "std_running"):
    rewards["pose"].params[key] = {r".*_hip_roll": 0.15, r".*_hip_yaw": 0.15,
                                   r".*_ankle_roll": 0.1}
  rewards["pose"].weight = 0.5
  rewards["body_ang_vel"].params["asset_cfg"].body_names = (robot.ROOT,)
  rewards["body_ang_vel"].weight = -0.05
  rewards["angular_momentum"].weight = -0.02
  rewards["feet_still"] = RewardTermCfg(
    func=feet_still, weight=-2.0,
    params={"asset_cfg": SceneEntityCfg("robot", site_names=robot.FEET)})
  rewards["feet_anchor"] = RewardTermCfg(
    func=feet_anchor, weight=-10.0, params={"sensor_name": "feet_ground_contact"})
  rewards["feet_down"] = RewardTermCfg(
    func=feet_down, weight=0.5, params={"sensor_name": "feet_ground_contact"})
  legs = SceneEntityCfg("robot", actuator_names=robot.LEGS, preserve_order=True)
  rated = [robot.JOINTS[j]["rated_torque"] for j in robot.LEGS]
  rewards["torque_over_rating"] = RewardTermCfg(
    func=torque_over_rating, weight=-0.5,
    params={"asset_cfg": legs, "rated": torch.tensor(rated)})

  # Falls.
  cfg.terminations.pop("out_of_terrain_bounds", None)
  cfg.terminations["fell_over"] = TerminationTermCfg(
    func=envs_mdp.root_height_below_minimum, params={"minimum_height": FALLEN_M})
  cfg.terminations["tilt_lost"] = TerminationTermCfg(
    func=tilt_lost, params={"command_name": "posture", "limit": TILT_LOST})

  cfg.viewer.body_name = robot.ROOT
  cfg.viewer.distance = 3.0
  cfg.viewer.elevation = -10.0
  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}
  return cfg


def _root_height(env, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
  return env.scene[asset_cfg.name].data.root_link_pos_w[:, 2:3]
