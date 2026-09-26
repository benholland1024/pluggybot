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

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from quad_train import robot

#: Top commanded speeds: forward/back, sideways (m/s), turning (rad/s).
LIN_X = (-0.8, 1.2)
LIN_Y = (-0.5, 0.5)
ANG_Z = (-1.0, 1.0)


def flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = make_velocity_env_cfg()
  cfg.sim.njmax = 300
  cfg.sim.nconmax = None
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.scene.entities = {"robot": robot.get_robot_cfg()}

  # Flat ground.
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None
  cfg.scene.sensors = tuple(s for s in cfg.scene.sensors
                            if s.name != "terrain_scan")
  for sensor in cfg.scene.sensors:
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(ObjRef(type="site", name=f, entity="robot")
                           for f in robot.FEET)
      sensor.pattern = RingPatternCfg.single_ring(radius=0.04, num_samples=4)
  feet_contact = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=robot.FEET, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  cfg.scene.sensors = cfg.scene.sensors + (feet_contact,)

  # Observations: no terrain scan on flat; no base velocity for the actor.
  for group in ("actor", "critic"):
    del cfg.observations[group].terms["height_scan"]
  del cfg.observations["actor"].terms["base_lin_vel"]

  action = cfg.actions["joint_pos"]
  assert isinstance(action, JointPositionActionCfg)
  action.scale = robot.ACTION_SCALE

  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  twist.ranges.lin_vel_x = (-0.6, 0.8)
  twist.ranges.lin_vel_y = LIN_Y
  twist.ranges.ang_vel_z = (-0.5, 0.5)
  cfg.curriculum.pop("terrain_levels", None)
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
  cfg.rewards["upright"].params.pop("terrain_sensor_names", None)
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

  cfg.terminations.pop("out_of_terrain_bounds", None)
  cfg.terminations["fell_over"] = TerminationTermCfg(
    func=mdp.bad_orientation, params={"limit_angle": math.radians(70.0)})

  cfg.viewer.body_name = robot.ROOT
  cfg.viewer.distance = 2.0
  cfg.viewer.elevation = -10.0

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    twist.ranges.lin_vel_x = LIN_X
    twist.ranges.ang_vel_z = ANG_Z
  return cfg
