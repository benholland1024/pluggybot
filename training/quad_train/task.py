"""The walking task (#377 item 3): mjlab's velocity task on our body, made
honest where the stock recipe is not.

What differs from mjlab's Go1 recipe, and why:
  - the ACTOR sees no base linear velocity. The real body measures none (an
    IMU integrates acceleration; legged odometry is #377 item 8); the critic,
    which never runs on the robot, keeps it.
  - the actuator is `robot.LEG_ACTUATOR`: the GIM8108-8's torque-speed line,
    its reflected inertia and a command delay up to `latency_s`.
  - what no datasheet gives is randomised across the ranges
    `models/quadruped.json` carries: the rotor inertia, joint friction, and
    the effort limit; the arm's mass and reach as the torso's mass and CoM.
  - commands stop at the body's measured speed limit (~1.2 m/s, SimNotes
    "The quadruped body"), not Go1's 3 m/s.
"""

from dataclasses import dataclass, fields, replace
import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.sensor.raycast_sensor import GridPatternCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.terrains import config as terrain_cfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

from quad_train import robot
from quad_train.stairs import StairsCommandCfg, foot_nose_clearance

#: Top commanded speeds: forward/back, sideways (m/s), turning (rad/s).
LIN_X = (-0.8, 1.2)
LIN_Y = (-0.5, 0.5)
ANG_Z = (-1.0, 1.0)


def _common(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
  """What every one of our tasks sets the same way: the robot, its feet, its
  commands, the randomisation and the rewards' names."""
  cfg.scene.entities = {"robot": robot.get_robot_cfg()}
  for sensor in cfg.scene.sensors:
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(ObjRef(type="site", name=f, entity="robot")
                           for f in robot.FEET)
      sensor.pattern = RingPatternCfg.single_ring(radius=0.04, num_samples=4)
    if sensor.name == "terrain_scan":
      sensor.frame.name = robot.ROOT
  cfg.scene.sensors = cfg.scene.sensors + (ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=robot.FEET, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  ),)

  # The actor gets no base velocity (the body cannot measure it).
  del cfg.observations["actor"].terms["base_lin_vel"]

  action = cfg.actions["joint_pos"]
  assert isinstance(action, JointPositionActionCfg)
  action.scale = robot.ACTION_SCALE

  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  twist.ranges.lin_vel_x = (-0.6, 0.8)
  twist.ranges.lin_vel_y = LIN_Y
  twist.ranges.ang_vel_z = (-0.5, 0.5)
  cfg.curriculum["command_vel"].params["velocity_stages"] = [
    {"step": 0, "lin_vel_x": (-0.6, 0.8), "ang_vel_z": (-0.5, 0.5)},
    {"step": 1500 * 24, "lin_vel_x": LIN_X, "ang_vel_z": ANG_Z},
  ]

  # Randomisation: every number no datasheet gives.
  joints = SceneEntityCfg("robot", joint_names=(".*",))
  root = SceneEntityCfg("robot", body_names=(robot.ROOT,))
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = robot.FEET
  cfg.events["base_com"].params["asset_cfg"].body_names = (robot.ROOT,)
  # The arm at full reach moves the CoM ~4 cm forward: cover it.
  cfg.events["base_com"].params["ranges"] = {
    0: (-0.05, 0.05), 1: (-0.025, 0.025), 2: (-0.03, 0.03)}
  cfg.events["rotor_inertia"] = EventTermCfg(
    mode="startup", func=envs_mdp.dr.joint_armature,
    params={"asset_cfg": joints, "operation": "abs",
            "ranges": tuple(robot.MOTOR["armature_range"])})
  cfg.events["joint_friction"] = EventTermCfg(
    mode="startup", func=envs_mdp.dr.joint_friction,
    params={"asset_cfg": joints, "operation": "abs",
            "ranges": tuple(robot.BODY["friction_range"])})
  cfg.events["body_mass"] = EventTermCfg(
    mode="startup", func=envs_mdp.dr.body_mass,
    params={"asset_cfg": root, "operation": "scale", "ranges": (0.85, 1.15)})
  cfg.events["effort_limit"] = EventTermCfg(
    mode="startup", func=envs_mdp.dr.effort_limits,
    params={"asset_cfg": SceneEntityCfg("robot", actuator_names=(".*",)),
            "operation": "scale", "effort_limit_range": (0.9, 1.1)})

  # Rewards: the stock set, pointed at our names.
  cfg.rewards["upright"].params["asset_cfg"].body_names = (robot.ROOT,)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = (robot.ROOT,)
  for name in ("foot_clearance", "foot_slip"):
    cfg.rewards[name].params["asset_cfg"].site_names = robot.FEET
  cfg.rewards["pose"].params["std_standing"] = {
    r".*_(hip_abd|hip_flex)": 0.05, r".*_knee": 0.1}
  cfg.rewards["pose"].params["std_walking"] = {
    r".*_(hip_abd|hip_flex)": 0.3, r".*_knee": 0.6}
  cfg.rewards["pose"].params["std_running"] = {
    r".*_(hip_abd|hip_flex)": 0.3, r".*_knee": 0.6}
  cfg.rewards["body_ang_vel"].weight = 0.0
  cfg.rewards["angular_momentum"].weight = 0.0
  cfg.rewards["air_time"].weight = 0.0

  cfg.viewer.body_name = robot.ROOT
  cfg.viewer.distance = 2.0
  cfg.viewer.elevation = -10.0
  return cfg


def _play(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
  cfg.episode_length_s = int(1e9)
  cfg.observations["actor"].enable_corruption = False
  cfg.events.pop("push_robot", None)
  cfg.curriculum = {}
  twist = cfg.commands["twist"]
  twist.ranges.lin_vel_x = LIN_X
  twist.ranges.ang_vel_z = ANG_Z
  return cfg


def flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = _common(make_velocity_env_cfg())
  cfg.sim.njmax = 300
  cfg.sim.nconmax = None
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None
  cfg.scene.sensors = tuple(s for s in cfg.scene.sensors
                            if s.name != "terrain_scan")
  for group in ("actor", "critic"):
    del cfg.observations[group].terms["height_scan"]
  cfg.rewards["upright"].params.pop("terrain_sensor_names", None)
  cfg.curriculum.pop("terrain_levels", None)
  cfg.terminations.pop("out_of_terrain_bounds", None)
  cfg.terminations["fell_over"] = TerminationTermCfg(
    func=mdp.bad_orientation, params={"limit_angle": math.radians(70.0)})
  return _play(cfg) if play else cfg


#: A house stair's tread, and the risers the stair columns span across their
#: ten levels (#388): the house's 0.18 m in the middle, between levels 4 and
#: 5; the UK allows a home 0.22.
TREAD_M = 0.28
RISERS_M = (0.10, 0.26)
#: Flights of eight: 2 m platforms and 0.5 m borders on the 8 m tiles (mjlab's
#: 3 m and 1 m leave five), so a pit is nine risers deep.
FLIGHT = dict(step_height_range=RISERS_M, step_width=TREAD_M,
              platform_width=2.0, border_width=0.5)

#: The world #280 builds, as terrain the curriculum climbs through: the
#: stairs, a curb and garden rocks as a grid of blocks to 0.15, rough ground,
#: slopes. ⚠ mjlab's `pyramid_stairs_inv` spawns the robot in a pit, so
#: walking out CLIMBS, and `pyramid_stairs` on the top, so it DESCENDS.
TERRAINS = TerrainGeneratorCfg(
  size=(8.0, 8.0), border_width=20.0, num_rows=10, num_cols=20,
  curriculum=True,
  sub_terrains={
    "flat": terrain_cfg.flat(proportion=0.1),
    "stairs_up": terrain_cfg.pyramid_stairs_inv(proportion=0.35, **FLIGHT),
    "stairs_down": terrain_cfg.pyramid_stairs(proportion=0.25, **FLIGHT),
    "blocks": terrain_cfg.box_random_grid(
      proportion=0.15, grid_height_range=(0.0, 0.15), grid_width=0.45),
    "rough": terrain_cfg.random_rough(proportion=0.08, noise_range=(0.02, 0.08)),
    "slope": terrain_cfg.hf_pyramid_slope(proportion=0.07, slope_range=(0.0, 0.4)),
  },
  add_lights=True,
)
STAIRS = ("stairs_up", "stairs_down")

#: A swinging foot's centre must pass this far over the highest terrain
#: within half a tread of it, m (`stairs.foot_nose_clearance`): the ball's
#: underside ~5 cm over the nose. The #377 policies swung 3.6 cm.
NOSE_CLEARANCE_M = 0.07
#: The ring that reading comes from: the foot, and eight rays at a quarter
#: and at half a tread.
NOSE_RING = RingPatternCfg(rings=(RingPatternCfg.Ring(TREAD_M / 4, 8),
                                  RingPatternCfg.Ring(TREAD_M / 2, 8)))
#: The robot's geoms that must not touch a step (`models/quadruped.xml`).
LIMBS = tuple(f"{leg}_{part}" for leg in robot.BODY["legs"]
              for part in ("thigh", "shank"))
TRUNK = ("torso", "belly")


def _stairs(cfg: ManagerBasedRlEnvCfg) -> None:
  """What makes the curriculum reach a house's riser (#388; `stairs.py`)."""
  twist = cfg.commands["twist"]
  cfg.commands["twist"] = StairsCommandCfg(
    **{f.name: getattr(twist, f.name) for f in fields(twist)}, stairs=STAIRS)

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

  # A flight is 33 degrees: level against gravity, the hind legs cannot
  # reach it, and the stock reward kept a quarter of its value for a torso
  # parallel to the flight. Measured against the terrain's plane, as mjlab's
  # Go1 recipe does.
  cfg.rewards["upright"].params["terrain_sensor_names"] = ("terrain_scan",)
  # mjlab's clearance also charges a foot swung high, the lift a riser needs.
  del cfg.rewards["foot_clearance"]
  cfg.rewards["nose_clearance"] = RewardTermCfg(
    func=foot_nose_clearance, weight=-2.0,
    params={"target_height": NOSE_CLEARANCE_M, "height_sensor_name": "foot_nose_scan",
            "command_name": "twist",
            "asset_cfg": SceneEntityCfg("robot", site_names=robot.FEET)})
  # The swing's peak, read over the same ring: a lift over a riser peaks
  # over the tread above, not the one it left.
  cfg.rewards["foot_swing_height"].params["height_sensor_name"] = "foot_nose_scan"
  cfg.rewards["limb_contact"] = RewardTermCfg(
    func=mdp.self_collision_cost, weight=-0.25, params={"sensor_name": "limb_terrain"})
  cfg.rewards["trunk_contact"] = RewardTermCfg(
    func=mdp.self_collision_cost, weight=-1.0, params={"sensor_name": "trunk_terrain"})
  # On a flight the legs sit ~0.4 rad of hip flexion off the stand: the
  # walking tolerance widens there, the lateral one does not.
  wide = {r".*_hip_abd": 0.3, r".*_hip_flex": 0.5, r".*_knee": 0.8}
  cfg.rewards["pose"].params["std_walking"] = wide
  cfg.rewards["pose"].params["std_running"] = wide


@dataclass
class RaisedGridPatternCfg(GridPatternCfg):
  """mjlab's height-scan grid with its ray origins `height` above the body
  (still turned with the heading only). Cast from the body itself, a ray a
  metre ahead on a flight of 0.18 m risers starts INSIDE the step above the
  body and reports the floor under it; the scan's value is unchanged (the
  body's height over each hit)."""

  height: float = 1.0

  def generate_rays(self, mj_model, device):
    offsets, directions = super().generate_rays(mj_model, device)
    offsets[:, 2] = self.height
    return offsets, directions


#: What the perceptive policy is shown: the terrain's height under a grid
#: 1.6 m long and 1.0 m wide at 0.1 m (187 points) around the body. On the
#: robot that is the D435's height map (`perception/heightmap.py`) sampled at
#: these points: the camera sees ahead, and the map keeps what it saw.
SCAN = RaisedGridPatternCfg(size=(1.6, 1.0), resolution=0.1)


def rough_env_cfg(play: bool = False, perceptive: bool = False) -> ManagerBasedRlEnvCfg:
  """Stairs, curbs and rocks: BLIND by default (the actor has no height scan,
  the critic does; #377 asks for the tallest riser cleared blind first), or
  `perceptive`, the actor shown the height scan."""
  cfg = _common(make_velocity_env_cfg())
  for sensor in cfg.scene.sensors:
    if sensor.name == "terrain_scan":
      sensor.pattern = SCAN
  # Memory on a 6 GB card: the stock 500 CCD iterations wanted a 1.7 GB
  # scratch array at 4096 envs (scratchpad trainer report). The 4090 runs
  # pass `--env.sim.mujoco.ccd-iterations 500`.
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.scene.terrain.terrain_generator = replace(TERRAINS)
  _stairs(cfg)
  if not perceptive:
    del cfg.observations["actor"].terms["height_scan"]
  cfg.terminations["fell_over"] = TerminationTermCfg(
    func=mdp.bad_orientation, params={"limit_angle": math.radians(70.0)})
  if play:
    cfg = _play(cfg)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    gen = cfg.scene.terrain.terrain_generator
    gen.curriculum, gen.num_rows, gen.num_cols = False, 5, 5
  return cfg
