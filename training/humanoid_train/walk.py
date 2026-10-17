"""Walking (#507, stage 4): mjlab's G1 velocity recipe on our humanoid,
made honest where the stock recipe is not -- the quadruped's changes
(`quad_train.task`), carried over:

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

`Pluggy-Humanoid-Walk` is flat ground; `-Rough` adds the house's floors'
cousins -- rough patches, slopes and low blocks -- under the curriculum
mjlab's velocity task climbs. The stairs are stage 5's.
"""

import math
from dataclasses import replace

import torch
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, ObjRef, RingPatternCfg, TerrainHeightSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.terrains import config as terrain_cfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

from humanoid_train import robot
from humanoid_train.stand import HAND_KG, ScriptedArmsCfg, torque_over_rating

#: Top commanded speeds: forward and back, sideways (m/s), turning (rad/s),
#: and where the curriculum starts them. A person walks 1.2-1.4 m/s; the
#: T2's maker quotes 1.4.
LIN_X, LIN_Y, ANG_Z = (-0.8, 1.4), (-0.4, 0.4), (-1.0, 1.0)
START_X, START_Z = (-0.5, 0.8), (-0.6, 0.6)
#: The joint-pose reward's tolerances while walking, rad (the G1's, our
#: joints): the knees and hip pitch loosest, the ankles' roll tightest --
#: and the hip's yaw twice the G1's. ⚠ With no waist, a turn is the hips'
#: yaw alone: at the G1's 0.15 a policy asked 0.5 rad/s turned 0.08, the
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


#: Rough ground, BLIND (the critic sees the terrain, the actor does not):
#: what a garden path, a doorstep and a rug give a walker -- rough patches
#: to 6 cm, slopes to 0.3, a grid of blocks to 8 cm.
TERRAINS = TerrainGeneratorCfg(
  size=(8.0, 8.0), border_width=20.0, num_rows=10, num_cols=20, curriculum=True,
  sub_terrains={
    "flat": terrain_cfg.flat(proportion=0.25),
    "rough": terrain_cfg.random_rough(proportion=0.3, noise_range=(0.01, 0.06)),
    "slope": terrain_cfg.hf_pyramid_slope(proportion=0.2, slope_range=(0.0, 0.3)),
    "blocks": terrain_cfg.box_random_grid(proportion=0.25, grid_height_range=(0.0, 0.08),
                                          grid_width=0.45),
  },
  add_lights=True,
)


def rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = _common(make_velocity_env_cfg())
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.scene.terrain.terrain_generator = replace(TERRAINS)
  del cfg.observations["actor"].terms["height_scan"]
  cfg.terminations["fell_over"] = TerminationTermCfg(
    func=mdp.bad_orientation, params={"limit_angle": math.radians(70.0)})
  if play:
    cfg = _play(cfg)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    gen = cfg.scene.terrain.terrain_generator
    gen.curriculum, gen.num_rows, gen.num_cols = False, 5, 5
  return cfg
