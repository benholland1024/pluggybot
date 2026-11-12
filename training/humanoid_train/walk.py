"""Walking (#507, stage 4): mjlab's G1 velocity recipe on our humanoid,
made honest where the stock recipe is not -- the quadruped's changes
(#377, its walking task, at `quadruped-final`), carried over:

  - the ACTOR sees no base linear velocity (the body measures none: an IMU
    integrates acceleration, and its legs' odometry is the body's); the
    critic, which never runs on the robot, keeps it;
  - each joint is driven by its own RobStride part (`robot.actuators`), the
    numbers no sheet gives randomised (`stand.py`'s ranges);
  - the arms are not the policy's: they are scripted (`stand.ScriptedArms`)
    more quietly than a reach moves them, hanging half the time, as the
    served arms are driven by whatever carries them;
  - the legs meet each other (`models/humanoid.xml`'s contact pairs), so a
    stride is charged for a knock as the G1's self-collision is.

`Pluggy-Humanoid-Walk` is flat ground: the served walker. `-Rough` is the
flat walker trained on (`--agent.resume`) over the house's floors' cousins
-- blocks, low steps and tilted plates, built of boxes (`TERRAINS`) --
under mjlab's terrain curriculum: a robot goes up a row for ending its
episode 4 m from its tile's centre, and down one for walking under half its
command's distance. Its observations are the flat task's, so the one
checkpoint carries over: the critic is blind too. ⚠ BLIND, IT GIVES UP
WHAT RISKS A FALL: rough4 (this recipe) never falls on the courses, but it
stands still where asked to walk backward or sideways, on flat ground too
(SimNotes, "The humanoid walks"), so it is not the served walker. Every
difficulty at once (rough3) narrowed it further; sharper tracking widths
(0.2 m/s and 0.3 rad/s, rough5) paid nothing for a walk it could not track
precisely, and it stood still at 1 m/s too.

`-Seeing` (stage 5) is a walker that SEES the ground: the actor is shown a
height scan, the terrain's height under a grid round its feet and ahead
(`SCAN`), and the terrain adds flights up and down at the house's tread
(`SEEING_TERRAINS`), with #388's stairs recipe (`stairs.py`): on a
stair tile most commands walk straight out along a flight, and a swinging
foot must clear the nose ahead of it. On the robot the scan is the D435's
map, read at the same grid (`pluggybot.humanoid.scan`). A new policy, not a
fine-tune: its inputs are new.
"""

import math
from dataclasses import dataclass, fields, replace

import torch
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.terminations import nan_detection
from mjlab.managers import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, ObjRef, RingPatternCfg, TerrainHeightSensorCfg
from mjlab.sensor.raycast_sensor import GridPatternCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.terrains import config as terrain_cfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

from humanoid_train import robot
from humanoid_train.stairs import StairsCommandCfg, foot_nose_clearance
from humanoid_train.stand import (ARM_MOVE_S, ARM_RATE, HAND_KG, MIRRORED, ScriptedArms,
                                  ScriptedArmsCfg, torque_over_rating)

#: Top commanded speeds: forward and back, sideways (m/s), turning (rad/s),
#: and where the curriculum starts them. A person walks 1.2-1.4 m/s; the
#: T2's maker quotes 1.4.
LIN_X, LIN_Y, ANG_Z = (-0.8, 1.4), (-0.4, 0.4), (-1.0, 1.0)
START_X, START_Z = (-0.5, 0.8), (-0.6, 0.6)
#: The joint-pose reward's tolerances while walking, rad (the G1's, our
#: joints): the knees and hip pitch loosest, the ankles' roll tightest --
#: and the hip's yaw twice the G1's. ⚠ A turn is the hips' yaw alone (the
#: waist, #560's stage (c3), is held home while it walks): at the G1's 0.15 a policy asked 0.5 rad/s turned 0.08, the
#: turn's reward still paying 60 % for not turning at all.
STD_WALKING = {r".*_hip_pitch": 0.3, r".*_hip_roll": 0.15, r".*_hip_yaw": 0.3,
               r".*_knee": 0.35, r".*_ankle_pitch": 0.25, r".*_ankle_roll": 0.1}
#: The turn's reward's width, rad/s: the stock sqrt(0.5) pays 61 % for
#: standing still when asked 0.5 rad/s, sqrt(0.25) 37 %.
TURN_STD = 0.5
#: The legs' parts that meet each other (`models/humanoid.xml`'s pairs).
LEFT_LEG, RIGHT_LEG = "left_hip_pitch_link", "right_hip_pitch_link"


def _common(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
  cfg.scene.entities = {"robot": robot.get_robot_cfg()}
  for sensor in cfg.scene.sensors:
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(ObjRef(type="site", name=f, entity="robot") for f in robot.FEET)
      sensor.pattern = RingPatternCfg.single_ring(radius=0.05, num_samples=6)
    if sensor.name == "terrain_scan":
      sensor.frame.name = robot.ROOT
  legs_meet = ContactSensorCfg(
    name="legs_meet",
    primary=ContactMatch(mode="subtree", pattern=LEFT_LEG, entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern=RIGHT_LEG, entity="robot"),
    fields=("found", "force"), reduce="none", num_slots=1, history_length=4)
  cfg.scene.sensors = cfg.scene.sensors + (
    ContactSensorCfg(
      name="feet_ground_contact",
      primary=ContactMatch(mode="geom", pattern=robot.SOLES, entity="robot"),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found", "force"), reduce="netforce", num_slots=1, track_air_time=True),
    legs_meet)

  # The actor gets no base velocity (the body cannot measure it).
  del cfg.observations["actor"].terms["base_lin_vel"]

  action = cfg.actions["joint_pos"]
  assert isinstance(action, JointPositionActionCfg)
  action.actuator_names = robot.LEGS
  action.preserve_order = True
  action.scale = {j: 0.25 * robot.JOINTS[j]["peak_torque"] / robot.JOINTS[j]["stiffness"]
                  for j in robot.LEGS}
  cfg.actions["arms"] = ScriptedArmsCfg(rest_share=0.5, reach=0.5)

  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  twist.ranges.lin_vel_x, twist.ranges.lin_vel_y, twist.ranges.ang_vel_z = START_X, LIN_Y, START_Z
  twist.rel_standing_envs = 0.15
  twist.viz.z_offset = 1.5
  cfg.curriculum["command_vel"].params["velocity_stages"] = [
    {"step": 0, "lin_vel_x": START_X, "ang_vel_z": START_Z},
    {"step": 1500 * 24, "lin_vel_x": LIN_X, "ang_vel_z": ANG_Z},
  ]

  # Randomisation: stand.py's, and what a hand carries while walking.
  joints = SceneEntityCfg("robot", joint_names=(".*",))
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = robot.SOLES
  cfg.events["base_com"].params["asset_cfg"].body_names = (robot.ROOT,)
  cfg.events["rotor_inertia"] = EventTermCfg(
    mode="startup", func=dr.joint_armature,
    params={"asset_cfg": joints, "operation": "scale", "ranges": (0.8, 1.2)})
  cfg.events["joint_friction"] = EventTermCfg(
    mode="startup", func=dr.joint_friction,
    params={"asset_cfg": joints, "operation": "scale", "ranges": (0.5, 2.0)})
  cfg.events["body_mass"] = EventTermCfg(
    mode="startup", func=dr.body_mass,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=(robot.ROOT,)),
            "operation": "scale", "ranges": (0.9, 1.1)})
  cfg.events["hand_kg"] = EventTermCfg(
    mode="startup", func=dr.body_mass,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=robot.HANDS),
            "operation": "add", "ranges": (0.0, HAND_KG[1] / 2)})
  cfg.events["effort_limit"] = EventTermCfg(
    mode="startup", func=dr.effort_limits,
    params={"asset_cfg": SceneEntityCfg("robot", actuator_names=(".*",)),
            "operation": "scale", "effort_limit_range": (0.9, 1.1)})

  # Rewards: the G1's set, pointed at our names.
  rewards = cfg.rewards
  rewards["upright"].params["asset_cfg"].body_names = (robot.ROOT,)
  rewards["body_ang_vel"].params["asset_cfg"].body_names = (robot.ROOT,)
  for name in ("foot_clearance", "foot_slip"):
    rewards[name].params["asset_cfg"].site_names = robot.FEET
  legs = SceneEntityCfg("robot", joint_names=robot.LEGS)
  rewards["pose"].params["asset_cfg"] = legs
  rewards["pose"].params["std_standing"] = {".*": 0.05}
  rewards["pose"].params["std_walking"] = dict(STD_WALKING)
  rewards["pose"].params["std_running"] = {k: 1.7 * v for k, v in STD_WALKING.items()}
  rewards["track_angular_velocity"].params["std"] = TURN_STD
  rewards["body_ang_vel"].weight = -0.05
  rewards["angular_momentum"].weight = -0.02
  rewards["air_time"].weight = 0.0
  rewards["legs_meet"] = RewardTermCfg(
    func=mdp.self_collision_cost, weight=-1.0,
    params={"sensor_name": "legs_meet", "force_threshold": 10.0})
  driven = SceneEntityCfg("robot", actuator_names=robot.LEGS, preserve_order=True)
  rated = torch.tensor([robot.JOINTS[j]["rated_torque"] for j in robot.LEGS])
  rewards["torque_over_rating"] = RewardTermCfg(
    func=torque_over_rating, weight=-0.5, params={"asset_cfg": driven, "rated": rated})

  cfg.viewer.body_name = robot.ROOT
  cfg.viewer.distance = 3.0
  cfg.viewer.elevation = -10.0
  return cfg


def _play(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
  cfg.episode_length_s = int(1e9)
  cfg.observations["actor"].enable_corruption = False
  cfg.events.pop("push_robot", None)
  cfg.curriculum = {}
  twist = cfg.commands["twist"]
  twist.ranges.lin_vel_x, twist.ranges.ang_vel_z = LIN_X, ANG_Z
  return cfg


def walk_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Flat ground, as the house's floors are."""
  cfg = _common(make_velocity_env_cfg())
  cfg.sim.njmax = 300
  cfg.sim.nconmax = None
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None
  cfg.scene.sensors = tuple(s for s in cfg.scene.sensors if s.name != "terrain_scan")
  for group in ("actor", "critic"):
    del cfg.observations[group].terms["height_scan"]
  cfg.curriculum.pop("terrain_levels", None)
  cfg.terminations.pop("out_of_terrain_bounds", None)
  cfg.terminations["fell_over"] = TerminationTermCfg(
    func=mdp.bad_orientation, params={"limit_angle": math.radians(70.0)})
  return _play(cfg) if play else cfg


#: Rough ground, both networks blind: what a doorstep, a garden path and a
#: rug give a walker -- a grid of blocks to +-8 cm, low stairs up and down
#: (risers to 8 cm) and a grid of tilted plates (to 10 deg). ⚠ BOXES ONLY:
#: the box soles on a height field blew up -- flown by the flat walker, about
#: one world in seven on `random_rough` within 20 s (root speeds to 1e5
#: m/s), 2-4 % on the slopes', none of 978 on the blocks -- and rough1 and
#: rough2 ended in it (a NaN; the value loss at 4e6).
TERRAINS = TerrainGeneratorCfg(
  # A column a terrain, a row a difficulty: 50 tiles. ⚠ 200 tiles of boxes
  # ran a 5090 out of memory at 4096 worlds.
  size=(8.0, 8.0), border_width=20.0, num_rows=10, num_cols=5, curriculum=True,
  sub_terrains={
    "flat": terrain_cfg.flat(proportion=0.2),
    "blocks": terrain_cfg.box_random_grid(proportion=0.3, grid_height_range=(0.0, 0.08),
                                          grid_width=0.45),
    "steps": terrain_cfg.pyramid_stairs(proportion=0.15, step_height_range=(0.0, 0.08),
                                        step_width=0.5),
    "steps_down": terrain_cfg.pyramid_stairs_inv(proportion=0.15,
                                                 step_height_range=(0.0, 0.08), step_width=0.5),
    "tilted": terrain_cfg.tilted_grid(proportion=0.2, tilt_range_deg=10.0, height_range=0.05),
  },
  add_lights=True,
)
#: A world whose body moves faster than this is reset as a NaN's is, m/s
#: and rad/s: on the boxes, 4096 worlds of the flat walker for 20 s reached
#: 4.8 m/s and 100 rad/s at worst, falls included; a blow-up, 1e5 and 6e7.
BLOWN_V, BLOWN_QD = 10.0, 200.0


def blown_up(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
  robot = env.scene[asset_cfg.name]
  v = torch.linalg.norm(robot.data.root_link_lin_vel_w, dim=1)
  qd = robot.data.joint_vel.abs().max(dim=1).values
  return ~torch.isfinite(v) | ~torch.isfinite(qd) | (v > BLOWN_V) | (qd > BLOWN_QD)


def rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = _common(make_velocity_env_cfg())
  # A world that blows up or goes NaN is reset, never the run ended. ⚠ At
  # 200 tiles, the stock G1 rough's 70 contacts a world ran a 5090 out of
  # memory at 4096 worlds; the default 35 fit.
  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.terminations["nan"] = TerminationTermCfg(func=nan_detection)
  cfg.terminations["blown_up"] = TerminationTermCfg(func=blown_up)
  cfg.scene.terrain.terrain_generator = replace(TERRAINS)
  cfg.scene.sensors = tuple(s for s in cfg.scene.sensors if s.name != "terrain_scan")
  for group in ("actor", "critic"):
    del cfg.observations[group].terms["height_scan"]
  cfg.scene.terrain.max_init_terrain_level = 2
  # Carried on from the flat walker, the commands start at their full range.
  cfg.curriculum.pop("command_vel", None)
  twist = cfg.commands["twist"]
  twist.ranges.lin_vel_x, twist.ranges.ang_vel_z = LIN_X, ANG_Z
  cfg.terminations["fell_over"] = TerminationTermCfg(
    func=mdp.bad_orientation, params={"limit_angle": math.radians(70.0)})
  if play:
    cfg = _play(cfg)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    gen = cfg.scene.terrain.terrain_generator
    gen.curriculum, gen.num_rows, gen.num_cols = False, 5, 5
  return cfg


# ---- seeing the ground (stage 5) ---------------------------------------------------

#: The house's flight (`pluggybot.home.world`): 0.18 m risers on 0.30 m
#: treads. The stair columns span risers from 0.02 to 0.23 m across their
#: ten rows (the house's near row 7; the UK allows a home 0.22), in flights
#: of eight on 2 m platforms with 0.5 m borders. ⚠ From 0.05 m, a walker
#: learning to walk shuffled its feet 1.5-2 cm high and never climbed out of
#: the easiest pit while every other column moved up.
TREAD_M = 0.30
RISERS_M = (0.02, 0.23)
FLIGHT = dict(step_height_range=RISERS_M, step_width=TREAD_M, platform_width=2.0,
              border_width=0.5)
#: The rough task's boxes and the flights: one column a terrain (mjlab's
#: curriculum), the share of robots each spawns. ⚠ BOXES ONLY, as the
#: rough task's (`TERRAINS`): the box soles blow up on a height field.
SEEING_TERRAINS = TerrainGeneratorCfg(
  size=(8.0, 8.0), border_width=20.0, num_rows=10, num_cols=5, curriculum=True,
  sub_terrains={
    "flat": terrain_cfg.flat(proportion=0.15),
    "stairs_up": terrain_cfg.pyramid_stairs_inv(proportion=0.3, **FLIGHT),
    "stairs_down": terrain_cfg.pyramid_stairs(proportion=0.25, **FLIGHT),
    "blocks": terrain_cfg.box_random_grid(proportion=0.15, grid_height_range=(0.0, 0.08),
                                          grid_width=0.45),
    "tilted": terrain_cfg.tilted_grid(proportion=0.15, tilt_range_deg=10.0, height_range=0.05),
  },
  add_lights=True,
)
#: ⚠ mjlab's `pyramid_stairs_inv` spawns the robot in a pit, so walking out
#: CLIMBS; `pyramid_stairs` on top, so it descends (#388).
STAIRS = ("stairs_up", "stairs_down")


@dataclass
class RaisedGridPatternCfg(GridPatternCfg):
  """mjlab's height-scan grid with its ray origins `height` above the body
  (still turned with the heading only), the quadruped's (#388). Cast from
  the body itself, a ray a metre ahead on a flight of 0.18 m risers starts
  INSIDE the step above the body and reports the floor under it; the scan's
  value is unchanged (the body's height over each hit)."""

  height: float = 1.0

  def generate_rays(self, mj_model, device):
    offsets, directions = super().generate_rays(mj_model, device)
    offsets[:, 2] = self.height
    return offsets, directions


@dataclass
class AheadGridPatternCfg(RaisedGridPatternCfg):
  """The raised height-scan grid with its middle `ahead` of the pelvis: a
  walker needs the treads in front more than the floor behind."""

  ahead: float = 0.3

  def generate_rays(self, mj_model, device):
    offsets, directions = super().generate_rays(mj_model, device)
    offsets[:, 0] += self.ahead
    return offsets, directions


#: What the seeing walker is shown: the terrain under a grid 1.6 m long and
#: 1.0 m wide at 0.1 m (187 points), from 0.5 m behind the pelvis to 1.1 m
#: ahead, cast from 1 m over it (a flight a metre ahead rises 0.6 m). On the
#: robot the D435 sees the floor from 0.48 m ahead and its map keeps what it
#: saw, the ground under and behind the feet included.
SCAN = AheadGridPatternCfg(size=(1.6, 1.0), resolution=0.1, height=1.0, ahead=0.3)
#: A swinging sole's middle must pass this far over the highest terrain
#: within half a tread of it, m (`stairs.foot_nose_clearance`): the box
#: sole's toe is 0.11 m ahead of it.
NOSE_CLEARANCE_M = 0.07
NOSE_RING = RingPatternCfg(rings=(RingPatternCfg.Ring(TREAD_M / 4, 8),
                                  RingPatternCfg.Ring(TREAD_M / 2, 8)))
#: The robot's geoms that must not touch a step: a thigh or shank costs a
#: quarter of what the trunk does.
LIMBS = tuple(f"{s}_{g}" for s in ("left", "right") for g in ("thigh", "shank"))
TRUNK = ("pelvis", "torso", "head")
#: On a flight the hip, the knee and the ankle bend further than a walk:
#: the joint-pose reward's sagittal tolerances widen (a 0.18 m riser takes
#: the knee a radian past the stand's), the lateral do not, and the reward
#: weighs less (`POSE_WEIGHT`). ⚠ At the walk's weight and half these
#: widths, with the action rate charged as the flat walker's, the transplant
#: kept walk2's 2.5 cm shuffle and stopped at the first riser, every column
#: of risers held at the curriculum's floor for 800 iterations (seeing3 and
#: seeing4), the swing's charges four times as heavy as the stock's or not.
STD_STAIRS = {**STD_WALKING, r".*_hip_pitch": 0.8, r".*_knee": 1.0, r".*_ankle_pitch": 0.5}
POSE_WEIGHT, ACTION_RATE_WEIGHT, AIR_TIME_WEIGHT = 0.4, -0.05, 0.5
#: The swing's charges, heavier than the stock's: a foot swung under
#: `NOSE_CLEARANCE_M` over the nose ahead, and its peak under `SWING_M`.
NOSE_WEIGHT, SWING_WEIGHT, SWING_M = -8.0, -1.0, 0.12


def seeing_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """The ground-seeing walker on flights, blocks and tilted plates."""
  cfg = _common(make_velocity_env_cfg())
  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.terminations["nan"] = TerminationTermCfg(func=nan_detection)
  cfg.terminations["blown_up"] = TerminationTermCfg(func=blown_up)
  cfg.terminations["fell_over"] = TerminationTermCfg(
    func=mdp.bad_orientation, params={"limit_angle": math.radians(70.0)})
  cfg.scene.terrain.terrain_generator = replace(SEEING_TERRAINS)
  cfg.scene.terrain.max_init_terrain_level = 0
  for sensor in cfg.scene.sensors:
    if sensor.name == "terrain_scan":
      sensor.pattern = SCAN

  twist = cfg.commands["twist"]
  cfg.commands["twist"] = StairsCommandCfg(
    **{f.name: getattr(twist, f.name) for f in fields(twist)}, stairs=STAIRS,
    speed=(0.3, 0.6))

  feet = tuple(ObjRef(type="site", name=f, entity="robot") for f in robot.FEET)
  terrain = ContactMatch(mode="body", pattern="terrain")
  cfg.scene.sensors = cfg.scene.sensors + (
    TerrainHeightSensorCfg(
      name="foot_nose_scan", frame=feet, pattern=NOSE_RING, ray_alignment="yaw",
      max_distance=1.0, exclude_parent_body=True, include_geom_groups=(0,)),
    ContactSensorCfg(
      name="limb_terrain",
      primary=ContactMatch(mode="geom", pattern=LIMBS, entity="robot"),
      secondary=terrain, fields=("found",), reduce="none", num_slots=1),
    ContactSensorCfg(
      name="trunk_terrain",
      primary=ContactMatch(mode="geom", pattern=TRUNK, entity="robot"),
      secondary=terrain, fields=("found",), reduce="none", num_slots=1),
  )
  rewards = cfg.rewards
  del rewards["foot_clearance"]
  rewards["nose_clearance"] = RewardTermCfg(
    func=foot_nose_clearance, weight=NOSE_WEIGHT,
    params={"target_height": NOSE_CLEARANCE_M, "height_sensor_name": "foot_nose_scan",
            "command_name": "twist",
            "asset_cfg": SceneEntityCfg("robot", site_names=robot.FEET)})
  # The swing's peak, read over the same ring: a lift over a riser peaks
  # over the tread above, not the one it left.
  rewards["foot_swing_height"].params["height_sensor_name"] = "foot_nose_scan"
  rewards["foot_swing_height"].params["target_height"] = SWING_M
  rewards["foot_swing_height"].weight = SWING_WEIGHT
  rewards["pose"].weight = POSE_WEIGHT
  rewards["action_rate_l2"].weight = ACTION_RATE_WEIGHT
  rewards["air_time"].weight = AIR_TIME_WEIGHT
  rewards["limb_contact"] = RewardTermCfg(
    func=mdp.self_collision_cost, weight=-0.25, params={"sensor_name": "limb_terrain"})
  rewards["trunk_contact"] = RewardTermCfg(
    func=mdp.self_collision_cost, weight=-1.0, params={"sensor_name": "trunk_terrain"})
  rewards["pose"].params["std_walking"] = dict(STD_STAIRS)
  rewards["pose"].params["std_running"] = {k: 1.7 * v for k, v in STD_STAIRS.items()}
  if play:
    cfg = _play(cfg)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    gen = cfg.scene.terrain.terrain_generator
    gen.curriculum, gen.num_rows, gen.num_cols = False, 5, 5
  return cfg


#: `-Walk-Seeing-Fine`: the seeing walker carried on (`--agent.resume`
#: from seeing7) with its tracking paid NARROWLY too, on top of the wide
#: terms: the planar velocity within `FINE_LIN_STD` and the turn within
#: `FINE_ANG_STD`. ⚠ The wide terms alone (0.5 m/s) paid 70 % for standing
#: still when asked to sidestep at 0.3 m/s, and seeing7 stood still there
#: and when asked back at 0.3, on flat ground too; ⚠ a narrow term ALONE
#: paid nothing for a walk it could not track precisely, and rough5 stood
#: still at 1 m/s. ⚠ Planar and yaw alone: mjlab's terms count a climb's
#: rise and a step's roll as error, which a narrow width would charge.
#: Its last round's (fine5a, from fine3; the walker is its iteration 12500):
#: at a turn width of 0.25 a turn asked 1 rad/s walked 0.82-0.85, at 0.10
#: 0.92. Each round's flags are `training/README.md`'s.
FINE_LIN_STD, FINE_ANG_STD = 0.15, 0.10
FINE_LIN_WEIGHT, FINE_ANG_WEIGHT = 2.0, 3.0


def track_planar_velocity(env, std: float, command_name: str,
                          asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
  """The commanded forward and sideways speed tracked, exp(-error^2 / std^2):
  its rise not counted."""
  asset = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  error = torch.sum(torch.square(command[:, :2] - asset.data.root_link_lin_vel_b[:, :2]), dim=1)
  return torch.exp(-error / std ** 2)


def track_yaw_rate(env, std: float, command_name: str,
                   asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
  """The commanded turn tracked, exp(-error^2 / std^2): its roll and pitch
  rates not counted."""
  asset = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  error = torch.square(command[:, 2] - asset.data.root_link_ang_vel_b[:, 2])
  return torch.exp(-error / std ** 2)


def seeing_fine_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """`-Walk-Seeing-Fine`: the seeing walker's task with its tracking paid
  narrowly too, carried on from a walker that has it: the commands at their
  full range from the start and the robots spread over the easier half of
  the rows."""
  cfg = seeing_env_cfg(play=False)
  cfg.rewards["track_planar_fine"] = RewardTermCfg(
    func=track_planar_velocity, weight=FINE_LIN_WEIGHT,
    params={"std": FINE_LIN_STD, "command_name": "twist"})
  cfg.rewards["track_yaw_fine"] = RewardTermCfg(
    func=track_yaw_rate, weight=FINE_ANG_WEIGHT,
    params={"std": FINE_ANG_STD, "command_name": "twist"})
  cfg.curriculum.pop("command_vel", None)
  twist = cfg.commands["twist"]
  twist.ranges.lin_vel_x, twist.ranges.ang_vel_z = LIN_X, ANG_Z
  cfg.scene.terrain.max_init_terrain_level = 5
  if play:
    cfg = _play(cfg)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    gen = cfg.scene.terrain.terrain_generator
    gen.curriculum, gen.num_rows, gen.num_cols = False, 5, 5
  return cfg


#: The stairs alone (`-Stairs`): a specialist for the house's flight beside
#: the seeing walker, the same scan and rewards over flights up and down
#: (risers 0.02-0.22 m, the house's 0.30 m tread) and a little flat floor,
#: nine commands in ten on a flight walking straight out along it.
STAIRS_TERRAINS = TerrainGeneratorCfg(
  size=(8.0, 8.0), border_width=20.0, num_rows=10, num_cols=3, curriculum=True,
  sub_terrains={
    "flat": terrain_cfg.flat(proportion=0.1),
    "stairs_up": terrain_cfg.pyramid_stairs_inv(
      proportion=0.5, step_height_range=(0.02, 0.22), step_width=TREAD_M,
      platform_width=2.0, border_width=0.5),
    "stairs_down": terrain_cfg.pyramid_stairs(
      proportion=0.4, step_height_range=(0.02, 0.22), step_width=TREAD_M,
      platform_width=2.0, border_width=0.5),
  },
  add_lights=True,
)


def stairs_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """The seeing walker's task on flights alone."""
  cfg = seeing_env_cfg(play=False)
  cfg.scene.terrain.terrain_generator = replace(STAIRS_TERRAINS)
  cfg.commands["twist"].forward = 0.9
  if play:
    cfg = _play(cfg)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    gen = cfg.scene.terrain.terrain_generator
    gen.curriculum, gen.num_rows, gen.num_cols = False, 5, 3
  return cfg


# ---- carrying (#682) ---------------------------------------------------------------

#: THE CARRY BOX, the LEFT arm's range for each joint, rad (the right mirrors
#: roll, yaw and the wrist's roll): every carry pose #682's instrument flew
#: (`scripts/carry_spike.py`'s `low`, `tray`, `out`, `close`, the wrist
#: levelling a piece) and the wrist's whole span, which levelling one uses.
#: walk2's arms went within half of `stand.ARM_GOALS` of hanging, and every
#: pose that holds a piece level is outside that: its wrist turned 0.6-1.5
#: rad, an elbow folded close to 2.0, and walk2 veered 4-22 deg toward the
#: carrying arm over 4 m (SimNotes, "Carrying while walking, measured").
CARRY_BOX = {"shoulder_pitch": (-1.2, 0.3), "shoulder_roll": (0.05, 0.6),
             "shoulder_yaw": (-0.5, 0.5), "elbow": (0.3, 2.05),
             "wrist_pitch": (-1.5, 1.5), "wrist_roll": (-1.6, 1.6)}
#: Each arm, on its own, is given a segment this long, s, and in it hangs,
#: moves as walk2's did (a new goal every `stand.ARM_MOVE_S`), or CARRIES --
#: a pose drawn from `CARRY_BOX` held the whole segment, as a carry is held
#: through a walk -- by these shares: both arms carry about a quarter of the
#: time.
CARRY_SEGMENT_S = (4.0, 15.0)
CARRY_SHARES = (0.3, 0.2, 0.5)
#: THE CARRY POSE the served walk takes (`pluggybot.humanoid.reach.CARRY_POSE`,
#: Ben's `close`: the LEFT arm's shoulder pitch, roll, yaw and elbow, rad; a
#: pluggybot test reads the two alike), and how far round it, rad, a carry
#: drawn near it lies (`CarryArmsCfg.close_share`), its wrist anywhere in the
#: carry box -- the wrist that levels a piece there depends on the piece.
CARRY_CLOSE = (0.15, 0.25, 0.0, 2.00)
CARRY_CLOSE_SPREAD = 0.15
#: What a carried piece weighs at each grip, kg, drawn at each reset
#: (`robot.PAYLOADS`): the table's dishware is 0.04-0.50 kg, and 1 kg a
#: stack or a full bowl.
PAYLOAD_KG = (0.0, 1.0)
#: The joints a carrying walker sees, in their order: walk2's.
CARRY_SEEN = robot.LEGS + robot.ARMS


@dataclass(kw_only=True)
class CarryArmsCfg(ScriptedArmsCfg):
  """`stand.ScriptedArmsCfg` with carries held (`CarryArms`)."""
  segment_s: tuple[float, float] = CARRY_SEGMENT_S
  shares: tuple[float, float, float] = CARRY_SHARES
  #: Share of carries drawn near `CARRY_CLOSE` rather than across the box
  #: (carry3: 0.5; carry2: 0).
  close_share: float = 0.0

  def build(self, env) -> "CarryArms":
    return CarryArms(self, env)


class CarryArms(ScriptedArms):
  """The arms as a carrying walker meets them (#682): each arm in segments
  of `segment_s`, hanging, moving as walk2's scripted arms did (within
  `reach` of `stand.ARM_GOALS`, hanging `rest_share` of its goals), or
  holding one pose drawn from `CARRY_BOX` the whole segment -- ramped there
  at `stand.ARM_RATE` as the served reach walks an arm in joint space, its
  weight fed forward as `ScriptedArms` feeds it."""

  HANG, MOVE, CARRY = 0, 1, 2

  def __init__(self, cfg: CarryArmsCfg, env):
    super().__init__(cfg, env)
    _, names = self._entity.find_joints(robot.ARMS, preserve_order=True)
    lo, hi = [], []
    for name in names:
      side, kind = name.split("_", 1)
      a, b = CARRY_BOX[kind]
      if side == "right" and kind in MIRRORED:
        a, b = -b, -a
      lo.append(a)
      hi.append(b)
    self.carry_lo = torch.tensor(lo, device=self.device)
    self.carry_hi = torch.tensor(hi, device=self.device)
    near_lo, near_hi = list(lo), list(hi)
    for i, name in enumerate(names):
      side, kind = name.split("_", 1)
      j = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow")
      if kind in j:
        c = CARRY_CLOSE[j.index(kind)]
        if side == "right" and kind in MIRRORED:
          c = -c
        near_lo[i] = max(lo[i], c - CARRY_CLOSE_SPREAD)
        near_hi[i] = min(hi[i], c + CARRY_CLOSE_SPREAD)
    self.near_lo = torch.tensor(near_lo, device=self.device)
    self.near_hi = torch.tensor(near_hi, device=self.device)
    self.half = len(self.ids) // 2
    self.mode = torch.zeros(self.num_envs, 2, dtype=torch.long, device=self.device)
    self.segment = torch.zeros(self.num_envs, 2, device=self.device)
    self.arm_timer = torch.zeros(self.num_envs, 2, device=self.device)
    shares = torch.tensor(cfg.shares, device=self.device, dtype=torch.float)
    self.cum = torch.cumsum(shares / shares.sum(), 0)

  def _arm(self, k: int) -> slice:
    return slice(0, self.half) if k == 0 else slice(self.half, 2 * self.half)

  def _moving_goal(self, env_ids: torch.Tensor, k: int) -> torch.Tensor:
    """walk2's draw for arm `k` of `env_ids`: within `reach` of
    `stand.ARM_GOALS`, or hanging `rest_share` of the time."""
    sl = self._arm(k)
    u = torch.rand(len(env_ids), self.half, device=self.device)
    goal = self.lo[sl] + u * (self.hi[sl] - self.lo[sl])
    rest = torch.rand(len(env_ids), 1, device=self.device) < self.cfg.rest_share
    return torch.where(rest, self.rest[env_ids, sl], goal)

  def _new_segments(self, env_ids: torch.Tensor, k: int) -> None:
    """Arm `k` of `env_ids` given a new segment: its mode drawn by `shares`,
    its length by `segment_s`, and its first goal."""
    n = len(env_ids)
    sl = self._arm(k)
    mode = torch.bucketize(torch.rand(n, device=self.device), self.cum).clamp(max=2)
    self.mode[env_ids, k] = mode
    self.segment[env_ids, k] = torch.empty(n, device=self.device).uniform_(*self.cfg.segment_s)
    near = (torch.rand(n, 1, device=self.device) < self.cfg.close_share)
    lo = torch.where(near, self.near_lo[sl], self.carry_lo[sl])
    hi = torch.where(near, self.near_hi[sl], self.carry_hi[sl])
    carry = lo + torch.rand(n, self.half, device=self.device) * (hi - lo)
    goal = torch.where((mode == self.CARRY)[:, None], carry, self.rest[env_ids, sl])
    moving = mode == self.MOVE
    if moving.any():
      goal[moving] = self._moving_goal(env_ids[moving], k)
    self.goal[env_ids, sl] = goal
    self.arm_timer[env_ids, k] = torch.empty(n, device=self.device).uniform_(*ARM_MOVE_S)

  def process_actions(self, actions: torch.Tensor) -> None:
    dt = self._env.step_dt
    self.segment -= dt
    self.arm_timer -= dt
    for k in (0, 1):
      due = torch.nonzero(self.segment[:, k] <= 0).flatten()
      if len(due):
        self._new_segments(due, k)
      # ...an arm moving as walk2's: a new goal every `ARM_MOVE_S`
      moved = torch.nonzero((self.mode[:, k] == self.MOVE) & (self.arm_timer[:, k] <= 0)).flatten()
      if len(moved):
        self.goal[moved, self._arm(k)] = self._moving_goal(moved, k)
        self.arm_timer[moved, k] = torch.empty(len(moved), device=self.device).uniform_(*ARM_MOVE_S)
    step = ARM_RATE * dt
    self.target += torch.clamp(self.goal - self.target, -step, step)

  def reset(self, env_ids=None) -> None:
    super().reset(env_ids)
    ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
    if isinstance(ids, slice):
      ids = torch.arange(self.num_envs, device=self.device)[ids]
    for k in (0, 1):
      self._new_segments(ids, k)


def carry_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """`Pluggy-Humanoid-Walk-Carry` (#682): the flat walker (`walk_env_cfg`)
  with its arms carrying (`CarryArms`) and a load at each grip
  (`robot.PAYLOADS`, `PAYLOAD_KG` drawn at every reset) in place of walk2's
  load at the hands' middles -- fine-tuned from walk2 (`--agent.resume True
  --agent.load-run '.*_walk2$' --agent.load-checkpoint model_2999.pt`):
  its observations are walk2's, so the checkpoint carries over, and its
  commands start at their full range."""
  cfg = walk_env_cfg(play)
  cfg.scene.entities = {"robot": robot.get_carry_robot_cfg()}
  cfg.actions["arms"] = CarryArmsCfg(rest_share=0.5, reach=0.5)
  cfg.events.pop("hand_kg", None)
  cfg.events["payload_kg"] = EventTermCfg(
    mode="reset", func=dr.body_mass,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=robot.PAYLOADS),
            "operation": "abs", "ranges": PAYLOAD_KG})
  # ...and it sees the joints walk2 saw, in walk2's order: the legs and the
  # arms, never the waist, which came after it (#560's stage (c3)) and is
  # held home while it walks -- seen, its 71 inputs took no checkpoint of
  # walk2's 69 (`export.py` records the joints a task sees)
  for group in ("actor", "critic"):
    for term in ("joint_pos", "joint_vel"):
      cfg.observations[group].terms[term].params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=CARRY_SEEN, preserve_order=True)
  cfg.curriculum.pop("command_vel", None)
  twist = cfg.commands["twist"]
  twist.ranges.lin_vel_x, twist.ranges.ang_vel_z = LIN_X, ANG_Z
  return cfg
