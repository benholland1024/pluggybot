"""The quadruped as a `Body` (issue #387; `body.py` is the contract).

#377's body, walking on its policy, reckoning with its legs (corrected by
scan matching, #386, and by the dock's board, #378), resting by reflex and
getting up from a fall, in a world the rover lived in. Three layers, the
rover's shape (`mission/rover.py`):

  QuadStepper   what steps the physics: the command is `(vx, vy, yaw
                rate)`, and the stepper turns it into joint targets through
                the walking policy -- or, beneath any command, lies the body
                down, stands it up or gets it up (the posture machine below)
  QuadMission   the `Navigator` over that body: the map, the planner and
                the drive, and what only this body does -- the depth
                camera's obstacles under the LIDAR's plane, the dock
  QuadBody      the `Body` the loop reaches it through

THE POSTURE, below every command. `standing` walks what it is told. After
`posture.T_REST_S` with no motion command it lies down (`lying_down`, then
`lying`): the rest reflex (Ben, 2026-09-27). A motion command to a lying
body stands it up first (`standing_up`). A torso tilted past
`posture.FALL_TILT_RAD`, or slumped, is a fall, and the get-up policy
drives until the body stands (`getting_up`); how long that may take before
it is a death is the lifecycle's (`Body.stuck_after_s`). The posture rides
the wire (`QuadBody.posture`).

⚠ THE COMMAND'S DEAD BAND (#378): the policy walks nothing below ~0.2 m/s
and barely turns below ~0.2 rad/s, so a small command is raised to
`V_MIN` / `W_MIN` (`command_for`) and the rover's driving law, which tapers
to zero at a goal, still arrives.

⚠ THE LIDAR'S PLANE IS 0.51 m UP, on the rear mast (clear of the stowed
arm): the couch (0.50 m) and the bed (0.40 m) are under it. The nose's
D435 sees them, and what it sees there is the planner's (`_look_step`,
`_planning_grid`) -- the first thing on this robot that decides off the
depth camera (CLAUDE.md, the near-field bullet).
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from pluggybot import tick
from pluggybot.body import Body
from pluggybot.legs import dock as dk
from pluggybot.legs import posture as pz
from pluggybot.legs.actuator import BUS_V_NOMINAL, JointLimits
from pluggybot.legs.model import CHOSEN, ELECTRONICS_W, LEGS
from pluggybot.legs.odometry import LegOdometry
from pluggybot.legs.policy import POLICY_NPZ, PolicyDriver, Twist, WalkingPolicy
from pluggybot.legs.scripted import Command, VirtualModel
from pluggybot.mapping.frontier import OCC_THRESH
from pluggybot.navigator import Navigator, gave_up
from pluggybot.perception.depth import PERIOD as DEPTH_PERIOD
from pluggybot.perception.depth import DepthCamera, DepthFrame
from pluggybot.power import Pack
from pluggybot.rack.coupling import (
  HUB_PEG_Z, PEG_R, RACK_HANG_X, STATION_YS, TRAY_VERTEX_DROP,
)
from pluggybot.rack.localize import RackPose
from pluggybot.robot import FIRST, RobotHandle
from pluggybot.tick import Routine

#: The get-up policy (`training/`, `Pluggy-Quad-Getup`), beside the walker.
GETUP_NPZ = POLICY_NPZ.with_name("quadruped_getup.npz")
#: The command that holds the body where it is.
STILL = (0.0, 0.0, 0.0)
#: The postures, as the wire carries them.
STANDING, LYING_DOWN, LYING, STANDING_UP, GETTING_UP = POSTURES = (
  "standing", "lying_down", "lying", "standing_up", "getting_up")
#: A command component smaller than this is no command.
MOTION_EPS = 1e-3
#: The policy's dead band (#378, measured on all four committed policies):
#: 0.15 m/s commanded walks 5 mm/s, and yaw below 0.2 rad/s barely turns. A
#: walking command is raised to V_MIN; a turn on the spot to W_MIN.
V_MIN, W_MIN = 0.25, 0.3
#: What the walking policy was trained to track (`training/quad_train/
#: task.py`: LIN_X, LIN_Y, ANG_Z); a command is clamped into it.
VX_RANGE, VY_MAX, W_MAX = (-0.8, 1.2), 0.5, 1.0
#: How long `stand_routine` / `rest_routine` wait for the posture before
#: they give up and say so, s: 2.2 s is a stand-up; a fall's get-up is the
#: lifecycle's to bound.
POSTURE_BUDGET_S = 15.0

#: THE DEPTH CAMERA'S OBSTACLES (the module docstring). A point this high
#: over the floor, and no higher, is something the body walks into: above
#: `LOW_MIN_Z` a foot does not step over it (the flat policy is trained on
#: flat ground), below `LOW_MAX_Z` the torso and the LIDAR's puck reach it.
LOW_MIN_Z, LOW_MAX_Z = 0.08, 0.60
#: ...and a point under this is the floor, which clears a cell.
FLOOR_Z = 0.03
#: Only points this near, m (horizontally, from the torso): the stereo
#: error grows with z^2, and at the part's 3 m a floor pixel read 5 cm high
#: often enough to leave single obstacle cells in the middle of doorways --
#: MEASURED, a 1.0 m door closed to the planner by one of them (at the
#: inflation's 0.35 m a side, 0.30 m of a door is plannable). At 1.8 m
#: sigma is 12 mm.
LOW_RANGE_M = 1.8
#: ...and never within this many cells of what the LIDAR maps: walls are the
#: LIDAR's (matched, #386), and the camera's view of their bases through a
#: drifting pose only thickens them.
LOW_WALL_CELLS = 3
#: The layer's evidence per frame (log-odds, as the LIDAR's grid), its
#: clamp, and the level at which a cell is an obstacle to the planner:
#: three frames that saw it.
LOW_HIT, LOW_MISS, LOW_CLAMP, LOW_OCC = 0.9, 0.4, 4.0, 2.5
#: What an obstacle cell reads as, to the planner (`OCC_THRESH` and more).
LOW_BURN = OCC_THRESH + 2.0


def command_for(vx: float, vy: float, w: float) -> tuple[float, float, float]:
  """A velocity as this body can walk it: out of the policy's dead band,
  and clamped into what it was trained to track. A forward speed under
  `V_MIN` with a turn to make is a PIVOT, not a faster arc: the driving law
  crawls while it turns, and raised to V_MIN the crawl swept 0.25 m arcs
  into the walls it was turning away from (MEASURED, #387)."""
  if abs(vx) < MOTION_EPS and abs(vy) < MOTION_EPS and abs(w) < MOTION_EPS:
    return STILL
  if MOTION_EPS <= abs(vx) < V_MIN and abs(w) >= W_MIN:
    vx = 0.0
  if abs(vx) >= MOTION_EPS:
    vx = math.copysign(max(abs(vx), V_MIN), vx)
  elif abs(w) >= MOTION_EPS and abs(vy) < MOTION_EPS:
    w = math.copysign(max(abs(w), W_MIN), w)
  return (min(max(vx, VX_RANGE[0]), VX_RANGE[1]),
          min(max(vy, -VY_MAX), VY_MAX), min(max(w, -W_MAX), W_MAX))


def is_motion(command) -> bool:
  return any(abs(c) >= MOTION_EPS for c in command)


_POLICIES: dict = {}


def policies() -> tuple[WalkingPolicy, WalkingPolicy]:
  """The walker and the get-up policy, loaded once a process: stateless,
  so every body shares them (a `PolicyDriver` holds each body's state)."""
  if not _POLICIES:
    _POLICIES["walk"] = WalkingPolicy(POLICY_NPZ)
    _POLICIES["getup"] = WalkingPolicy(GETUP_NPZ)
  return _POLICIES["walk"], _POLICIES["getup"]


def _site_in_root(model, site: int, root: int) -> np.ndarray:
  """A site's position in its root body's frame, composed through the
  bodies between (all fixed to it)."""
  pos = np.array(model.site_pos[site], dtype=float)
  body = int(model.site_bodyid[site])
  while body != root:
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, model.body_quat[body])
    pos = model.body_pos[body] + mat.reshape(3, 3) @ pos
    body = int(model.body_parentid[body])
  return pos


class LegPack(Pack):
  """The quadruped's pack (#377's energy table): the electronics, always,
  and what the twelve drivers put into their windings (1.5 R I^2) and out
  through their shafts (positive work only; no credit for regeneration),
  off the torques they apply. Charged at the dock's rate."""

  def __init__(self, model, capacity_wh: float, charge_scale: float = 1.0,
               prefix: str = "") -> None:
    super().__init__(capacity_wh, charge_scale=charge_scale)
    self.charge_w = dk.CHARGE_W
    joints = pz.Joints.of(model, prefix)
    self._vadr = joints.vadr
    self._act = np.array([model.actuator(f"{prefix}{n}").id for n in
                          (f"{leg}_{j}" for leg in LEGS
                           for j in ("hip_abd", "hip_flex", "knee"))])
    self._lim = JointLimits.of(CHOSEN.motor, CHOSEN.knee_ratio)
    self.base_w = float(sum(ELECTRONICS_W.values()))

  def power_draw(self, data) -> float:
    tau = data.actuator_force[self._act]
    qd = data.qvel[self._vadr]
    return (self.base_w + float(self._lim.copper_w(tau).sum())
            + float(np.clip(tau * qd, 0.0, None).sum()))


class QuadStepper:
  """Steps one quadruped's physics (`tick.py`'s stepper): the posture
  machine and the policy before the world steps, the reckoning and the
  hooks after."""

  STILL = STILL

  def __init__(self, mission: "QuadMission") -> None:
    self.mission = mission

  @property
  def model(self):
    return self.mission.model

  @property
  def data(self):
    return self.mission.data

  def apply(self, command) -> None:
    self.mission._before_step(command)

  def after_step(self) -> None:
    self.mission._after_physics()

  def step(self, command) -> None:
    self.apply(command)
    mujoco.mj_step(self.model, self.data)
    self.after_step()

  def run(self, routine, name: str = ""):
    return tick.run(self, routine, name)


class QuadMission(Navigator):
  """The Navigator over a quadruped (the module docstring)."""

  #: The body's own sizes (`scripts/quad_spike.py`; SimNotes, "The first
  #: quadruped deploy"). Standing, its outline reaches 0.38 m from the torso's
  #: centre (the hind knees, 0.36 m behind it) and 0.20 m to either side;
  #: lying, 0.43 m (the hind knees further back); fallen, 0.63 m from the
  #: middle of its footprint. The map's inflation covers the half-width
  #: walking forward (0.35 m square on, 0.25 m on a diagonal), as the
  #: rover's did its swing: a turn on the spot within 0.38 m of a wall
  #: can touch it, and the front stop and the bumper are what answer that.
  INFLATION_CELLS = 7
  #: The disc round another robot, cells: its outline (0.43 m lying) plus
  #: this one's half-width and 5 cm -- 0.68 m, so 14; fallen, from the
  #: middle of its footprint, 0.63 + 0.25 = 0.88 m, so 18.
  OTHER_ROBOT_CELLS = 14
  DOWN_ROBOT_CELLS = 18
  #: The front stop, from the LIDAR on the rear mast. ⚠ UNDER THE CLEARANCE
  #: THE PLANNER GRANTS: a centre 0.35 m from a wall (the inflation) puts
  #: the LIDAR, 0.15 m behind it, 0.50 m from the wall, and a stop beyond
  #: that fires on every waypoint the planner lays along one -- MEASURED at
  #: 0.64 m, a drive toward the bedroom's divider backed off every second
  #: for 135 s. At 0.45 m the nose (0.39 m ahead of the LIDAR) is 6 cm off.
  FRONT_STOP_RANGE = 0.45
  #: ...tested over the CORRIDOR ahead, not the rover's 0.35 rad cone: the
  #: torso is 0.10 m either side of its line, and at 0.45 m the cone reaches
  #: 0.15 m out -- MEASURED, a wall alongside at 0.16 m fired it every scan,
  #: and the robot backed 2.3 m into a corner.
  FRONT_HALF_M = 0.12
  #: Backing off at the rover's 0.15 m/s would walk nothing (the dead band).
  BACKOFF_V = 0.3
  #: The depth camera's peer corridor (issue #328) in the torso's frame:
  #: the feet are 0.16 m either side, the nose 0.24 m ahead.
  PEER_STOP_AHEAD_M = 0.75
  PEER_STOP_HALF_M = 0.22
  PEER_CLEARANCE_M = 0.34
  #: A turn on the spot ends inside this: the policy tracks no finer.
  FACING_TOLERANCE = math.radians(4.0)
  #: Its drive reads progress along the route (`Navigator`): a detour round
  #: a wall is progress.
  PROGRESS_ALONG_ROUTE = True
  #: The turn's rate, rad/s per rad of error, and its budget, s.
  FACE_GAIN, FACE_BUDGET_S = 1.5, 12.0

  def __init__(self, model, data, viewer=None, realtime: bool = True,
               rack=None, grid_bounds=(-3, -3, 7, 7),
               handle: RobotHandle = FIRST) -> None:
    self.stepper = QuadStepper(self)
    root = model.body(handle.root).id
    lidar = _site_in_root(model, model.site(handle.el("lidar")).id, root)
    #: The LIDAR's place on the torso: the scan's origin for the map.
    self.LIDAR_ORIGIN = (float(lidar[0]), float(lidar[1]))
    super().__init__(model, data, handle=handle, grid_bounds=grid_bounds,
                     viewer=viewer, realtime=realtime, match=True)
    self.root = root
    walk, getup = policies()
    self.walker = PolicyDriver(model, data, walk, prefix=handle.prefix)
    self.getup = PolicyDriver(model, data, getup, prefix=handle.prefix)
    # ONE set of drivers for everything that commands the legs: a policy
    # re-sends its gains after anything else took them (`Drivers.torqued`).
    self.getup.drivers = self.drivers = self.walker.drivers
    self.drivers.set_bus(BUS_V_NOMINAL)
    self.joints = pz.Joints.of(model, handle.prefix)
    self.feet = [model.site(handle.el(f"{leg}_foot")).id for leg in LEGS]
    self.foot_r = float(model.geom_size[model.geom(handle.el("FL_foot")).id][0])
    self._vm: VirtualModel | None = None
    self.odo = LegOdometry(model, data, prefix=handle.prefix)
    self.posture = STANDING
    #: What a routine has asked of the posture: "lie", "stand" or None.
    self.want: str | None = None
    self.last_motion_t = float(data.time)
    self._move = None
    self._slumped_since: float | None = None
    self._stood_since: float | None = None
    self.falls = 0
    #: Held on the dock: the legs' reckoning holds its position.
    self.docked = False
    #: The depth camera, its latest frame and when, and the layer of what it
    #: sees under the LIDAR's plane (log-odds, the grid's cells).
    self.depth = DepthCamera(model, handle, mount="body")
    self._frame: DepthFrame | None = None
    self._frame_t = -math.inf
    self._frame_used_t = -math.inf
    self.low = np.zeros_like(self.grid.grid, dtype=np.float32)
    self._walls_t, self._walls_mask = None, None
    # THE DOCK: the commissioned pose (its board's drawing and where it was
    # installed -- the map frame is defined by it, as the rover's by its
    # rack) and what the robot believes of it this approach.
    self.dock_prior = self._dock_pose(model)
    rack_at = self.dock_prior if self.dock_prior is not None else (0.0, 0.0, 0.0)
    #: The loop's charge logic works round a "rack": for this body that is
    #: the DOCK, origin at the board, +x out into the room.
    self.rack = self.rack_prior = self._dock_as_rack(rack_at)
    self.dock_seen: tuple[float, float, float] | None = None
    self._board = None                # the nose camera's tag detector, lazily
    self.last_charge: dict | None = None
    self.press_steps = 0
    self._pressing = False
    self._contact_gids = self.body_gids[np.isin(
      self.body_gids, [model.geom(handle.el(f"{leg}_foot")).id for leg in LEGS],
      invert=True)]
    # ...the floors, which a leg brushes: every plane, and the generated
    # worlds' floor and ground slabs (`home_floor_geom`, `garden_ground_geom`)
    self._floor_gids = np.array(
      [g for g in range(model.ngeom)
       if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE
       or "_floor" in model.geom(g).name or "_ground" in model.geom(g).name],
      dtype=np.int32)
    self._dock_gids = np.array(
      [g for g in range(model.ngeom) if model.geom(g).name.startswith("dock_")],
      dtype=np.int32)
    #: ...as lookups by geom id, for the check that runs every step
    self._is_limb = np.zeros(model.ngeom, dtype=bool)
    self._is_limb[self._contact_gids] = True
    self._is_ignored = np.zeros(model.ngeom, dtype=bool)
    self._is_ignored[np.concatenate((self._floor_gids, self._dock_gids,
                                     self.body_gids))] = True

  # ---- the dock's frame -----------------------------------------------------

  @staticmethod
  def _dock_pose(model) -> tuple[float, float, float] | None:
    """Where the dock was installed: its body's pose, the commissioning."""
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "dock") < 0:
      return None
    b = model.body("dock")
    w, _, _, z = b.quat
    return float(b.pos[0]), float(b.pos[1]), 2.0 * math.atan2(float(z), float(w))

  @staticmethod
  def _dock_as_rack(pose) -> RackPose:
    x, y, yaw = pose
    bx, by = x + dk.DEFAULT.board_x * math.cos(yaw), y + dk.DEFAULT.board_x * math.sin(yaw)
    return RackPose(bx, by, dk._wrap(yaw + math.pi))

  def charge_standoff(self) -> tuple[float, float, float]:
    """Where the approach begins: on the dock's axis, `STANDOFF_M` behind the
    seat, facing the board -- off the commissioned pose, which every lie on
    the dock re-anchors the map to."""
    x, y, yaw = self.dock_prior
    return dk.compose((x, y, yaw), (-dk.STANDOFF_M, 0.0, 0.0))

  # ---- what the Navigator asks ---------------------------------------------

  @property
  def pose(self) -> tuple[float, float, float]:
    return self.odo.x, self.odo.y, self.odo.yaw

  def pose_xy(self) -> tuple[float, float]:
    return self.odo.x, self.odo.y

  def _set_pose(self, x: float, y: float, theta: float) -> None:
    self.odo.correct(x, y, theta)

  def true_pose(self) -> tuple[float, float, float]:
    """The torso's centre and its forward axis on the floor."""
    d = self.data
    r = d.xmat[self.root].reshape(3, 3)
    return (float(d.xpos[self.root][0]), float(d.xpos[self.root][1]),
            math.atan2(float(r[1, 0]), float(r[0, 0])))

  def level(self) -> bool:
    """On its feet and level: the map's gate. Lying, its scan plane is at
    a different height (0.29 m), and a map of two heights is neither."""
    return self.posture == STANDING and self.odo.att.tilt() <= self.LEVEL_TILT

  #: The IMU's tilt under which a standing torso is level for the map: the
  #: rover's 1.5 deg (its policy tilts the torso under 0.4 deg walking).
  LEVEL_TILT = math.radians(1.5)

  @property
  def pressing(self) -> bool:
    return self._pressing

  def run(self, routine, name: str = ""):
    return tick.run(self.stepper, routine, name)

  def _nav_routine(self, v: float, w: float) -> Routine:
    yield from self._twist_routine(v, 0.0, w)

  def _twist_routine(self, vx: float, vy: float, w: float) -> Routine:
    """One physics step at a velocity, as the policy can walk it -- stood
    up first if the body is resting."""
    cmd = command_for(vx, vy, w)
    if cmd != STILL and self.posture != STANDING:
      yield from self.stand_routine()
    yield cmd
    self._after_step()

  #: A plan asked for again this soon, for the same goal from within this of
  #: where the last was made, is that plan (s, m). MEASURED: at a stand-in
  #: (a goal off the plannable floor, the drive aimed at the nearest cell it
  #: could reach) the waypoints run out where the robot stands and the drive
  #: replanned EVERY STEP until its stagnation cut -- 4 190 plans in 90 s of
  #: a pair, 15 ms each; the map it planned over had not moved.
  REPLAN_HOLD_S, REPLAN_MOVED_M = 0.5, 0.05

  def _plan_to(self, wx: float, wy: float):
    t = float(self.data.time)
    x, y, _ = self.pose
    memo = self._plan_memo
    if (memo is not None and memo[0] == (wx, wy) and t - memo[1] < self.REPLAN_HOLD_S
        and math.hypot(x - memo[2][0], y - memo[2][1]) < self.REPLAN_MOVED_M):
      self._stand_in = memo[4]
      return None if memo[3] is None else list(memo[3])
    planned = super()._plan_to(wx, wy)
    self._plan_memo = ((wx, wy), t, (x, y),
                       None if planned is None else list(planned), self._stand_in)
    return planned

  _plan_memo = None

  def _front_blocked(self, angles, ranges) -> bool:
    ahead = np.cos(angles) * ranges
    side = np.sin(angles) * ranges
    return bool(((ahead > 0.0) & (ahead < self.FRONT_STOP_RANGE)
                 & (np.abs(side) <= self.FRONT_HALF_M)).any())

  def _planning_grid(self) -> np.ndarray:
    """The LIDAR's grid, and what the depth camera saw under its plane
    burned in as obstacles."""
    low = self.low > LOW_OCC
    if not low.any():
      return self.grid.grid
    out = self.grid.grid.copy()
    out[low] = np.maximum(out[low], LOW_BURN)
    return out

  # ---- the physics seam ------------------------------------------------------

  def _before_step(self, command) -> None:
    """The posture machine, then the command (the module docstring)."""
    t = float(self.data.time)
    moving = is_motion(command)
    if moving:
      self.last_motion_t = t
    self._fall_check(t)
    if self._move is not None:
      if self._step_move():
        return
    p = self.posture
    if p == STANDING:
      rest = t - self.last_motion_t >= pz.T_REST_S
      if self.want == "lie" or (not moving and rest and self.want != "stand"):
        self._begin(LYING_DOWN)
        self._step_move()
        return
      self.walker.command(Twist(*command) if moving else Twist())
    elif p == LYING:
      if moving or self.want == "stand":
        self._begin(STANDING_UP)
        self._step_move()
        return
      self.drivers.limp()
    elif p == GETTING_UP:
      self.getup.command(Twist())
      self._stood_check(t)

  def _after_physics(self) -> None:
    held = (self.odo.x, self.odo.y) if self.docked else None
    self.odo.step()
    if held is not None:
      self.odo.x, self.odo.y = held
    self._pressing = self._press_now()
    if self._pressing:
      self.press_steps += 1
    self._on_step()

  def _vm_now(self) -> VirtualModel:
    if self._vm is None:
      self._vm = VirtualModel(self.model, self.data, CHOSEN,
                              prefix=self.handle.prefix)
    return self._vm

  def _begin(self, posture: str) -> None:
    """Start a scripted move: LYING_DOWN ends LYING, STANDING_UP ends
    STANDING."""
    vm, d, j = self._vm_now(), self.data, self.joints

    def lie():
      yield from pz.lie_down_routine(CHOSEN, d, vm, j)
      t0 = d.time
      while d.time - t0 < pz.LIE_SETTLE_S:   # the drivers hold nothing
        yield None

    def stand():
      yield from pz.stand_up_routine(CHOSEN, d, vm, j)
      vm.reset()
      t0 = d.time
      while d.time - t0 < pz.STAND_HOLD_S:
        yield vm.torque(Command())

    self.posture = posture
    self._move = lie() if posture == LYING_DOWN else stand()

  def _step_move(self) -> bool:
    """One step of the scripted move in progress; False once it has ended
    (and the posture it ends in has begun)."""
    tau = next(self._move, StopIteration)
    if tau is StopIteration:
      self._move = None
      if self.posture == LYING_DOWN:
        self.posture = LYING
      else:
        self._handover(self.walker)
        self.posture = STANDING
        self.last_motion_t = float(self.data.time)
      return False
    if tau is None:
      self.drivers.limp()
    else:
      self.drivers.torque(tau)
    return True

  @staticmethod
  def _handover(driver: PolicyDriver) -> None:
    """A policy takes the body back from something that is not it: its last
    action is none, and it decides at once."""
    driver.last_action[:] = 0.0
    driver.steps = 0

  def _uprightness(self) -> float:
    """The torso's up axis against the world's (1 is upright)."""
    return float(self.data.xmat[self.root][8])

  def _height(self) -> float:
    """The torso over its lowest foot: how high the legs hold it (their
    kinematics, as the encoders give them)."""
    d = self.data
    return float(d.xpos[self.root][2] - min(d.site_xpos[f][2] for f in self.feet)
                 + self.foot_r)

  def _fall_check(self, t: float) -> None:
    if self.posture == GETTING_UP:
      return
    fallen = self._uprightness() < math.cos(pz.FALL_TILT_RAD)
    if self.posture == STANDING and self._height() < pz.SLUMP_Z_M:
      self._slumped_since = t if self._slumped_since is None else self._slumped_since
      fallen = fallen or t - self._slumped_since >= pz.SLUMP_S
    else:
      self._slumped_since = None
    if fallen:
      self._move = None
      self.posture = GETTING_UP
      self._stood_since = None
      self.falls += 1
      self._handover(self.getup)

  def _stood_check(self, t: float) -> None:
    up = (self._uprightness() > pz.UPRIGHT_COS
          and abs(self._height() - CHOSEN.stand_height) < pz.STAND_TOL_M)
    if not up:
      self._stood_since = None
      return
    self._stood_since = t if self._stood_since is None else self._stood_since
    if t - self._stood_since >= pz.STOOD_HOLD_S:
      self._handover(self.walker)
      self.posture = STANDING
      self.last_motion_t = t

  def _press_now(self) -> bool:
    """Something other than a foot touching anything but the floor, on its
    feet -- the rover's bumper, off contacts (a real one reads it off the
    drivers' current). Where it was touched, in the torso's frame, is kept
    for the retreat (`_backoff_routine`)."""
    if self.posture != STANDING or not self._contact_gids.size:
      return False
    d = self.data
    g = d.contact.geom[:d.ncon]
    if not len(g):
      return False
    # lookups over geom ids, not `np.isin` on the physics seam (rooftop #296)
    mine0, mine1 = self._is_limb[g[:, 0]], self._is_limb[g[:, 1]]
    other = np.where(mine0, g[:, 1], g[:, 0])
    hit = (mine0 | mine1) & ~self._is_ignored[other]
    if not hit.any():
      return False
    rot = d.xmat[self.root].reshape(3, 3)
    at = (d.contact.pos[:d.ncon][hit] - d.xpos[self.root]) @ rot
    self.pressed_at = (float(at[:, 0].mean()), float(at[:, 1].mean()))
    return True

  #: Where the body was last touched, (x ahead, y left) of the torso's
  #: centre: the retreat steps away from it.
  pressed_at: tuple[float, float] | None = None
  #: A touch this far from the centre along the body is at an end; nearer,
  #: a flank (the torso is 0.21 m long either side, 0.10 m wide).
  PRESS_END_M = 0.15

  def _backoff_routine(self) -> Routine:
    """The retreat, away from where it was touched: a legged body steps
    SIDEWAYS off a flank -- MEASURED, reversing off a thigh that scraped the
    kitchen counter replanned the same scrape, and the body backed until its
    hind knees met the far wall -- back off its nose, forward off its hind
    knees; after the LIDAR's front stop, straight back."""
    if self.pressed_at is not None:
      at, self.pressed_at = self.pressed_at, None
      if at[0] > self.PRESS_END_M:
        away = (-self.BACKOFF_V, 0.0)
      elif at[0] < -self.PRESS_END_M:
        away = (self.BACKOFF_V, 0.0)
      else:
        away = (0.0, -math.copysign(self.BACKOFF_V, at[1]))
      # ...for the whole of this retreat's window
      self._retreat = (away, self.backoff_until)
    away, until = self._retreat
    if until != self.backoff_until:
      away = (-self.BACKOFF_V, 0.0)
    yield from self._twist_routine(away[0], away[1], 0.0)

  _retreat: tuple = ((0.0, 0.0), None)

  # ---- the navigator's senses ------------------------------------------------

  def _look_step(self) -> None:
    """The depth camera, at its rate: what it sees under the LIDAR's plane
    goes into the planner's layer."""
    frame = self.latest_depth()
    if self._frame_t == self._frame_used_t or frame is None:
      return
    self._frame_used_t = self._frame_t
    if self.posture != STANDING:
      return
    self._fold_low(frame.points)

  def latest_depth(self) -> DepthFrame | None:
    """The newest depth frame, taken now if the last is a period old: ONE
    render feeds the planner's layer, the near-field map and the peer
    channel (`QuadBody.depth_camera`)."""
    t = float(self.data.time)
    if t - self._frame_t >= DEPTH_PERIOD - 1e-9:
      raw = self.depth.frame(self.data)
      self._frame = self._levelled(raw)
      self._frame_t = t
    return self._frame

  def _levelled(self, frame: DepthFrame) -> DepthFrame:
    """A frame from the torso's own axes into the robot frame the map and
    the peer channel read: level (the IMU's roll and pitch), x ahead of the
    torso's centre, z above the floor under the feet."""
    lv = self.odo.att.level()
    h = self._height()
    pts = frame.points @ lv.T
    pts[:, 2] += h
    peers = frame.peers @ lv.T if len(frame.peers) else frame.peers
    if len(peers):
      peers[:, 2] += h
    return DepthFrame(z=frame.z, points=pts, self_fraction=frame.self_fraction,
                      peers=peers, peer_geoms=frame.peer_geoms)

  def _fold_low(self, points: np.ndarray) -> None:
    if not len(points):
      return
    x, y, th = self.pose
    c, s = math.cos(th), math.sin(th)
    wx = x + c * points[:, 0] - s * points[:, 1]
    wy = y + s * points[:, 0] + c * points[:, 1]
    g = self.grid
    ix = np.floor((wx - g.x_min) / g.resolution).astype(np.int64)
    iy = np.floor((wy - g.y_min) / g.resolution).astype(np.int64)
    rows, cols = self.low.shape
    ok = (ix >= 0) & (ix < cols) & (iy >= 0) & (iy < rows)
    z = points[:, 2]
    near = np.hypot(points[:, 0], points[:, 1]) <= LOW_RANGE_M
    hit = ok & near & (z >= LOW_MIN_Z) & (z <= LOW_MAX_Z)
    floor = ok & near & (z < FLOOR_Z)
    if floor.any():
      np.subtract.at(self.low, (iy[floor], ix[floor]), LOW_MISS)
    if hit.any():
      # once per cell a frame: a couch's face is hundreds of points
      cells = np.unique(iy[hit] * cols + ix[hit])
      cells = cells[~self._walls().flat[cells]]
      self.low.flat[cells] += LOW_HIT
    np.clip(self.low, -LOW_CLAMP, LOW_CLAMP, out=self.low)

  def _walls(self) -> np.ndarray:
    """What the LIDAR maps, grown by `LOW_WALL_CELLS`, over the window the
    camera's points can land in (`LOW_RANGE_M` round the robot): the
    camera's layer stays out of it. Recomputed at most once a scan."""
    if self._walls_t != self._next_scan:
      from scipy import ndimage
      g = self.grid
      rows, cols = g.grid.shape
      cx, cy = g.world_to_cell(self.pose[0], self.pose[1])
      r = int(math.ceil(LOW_RANGE_M / g.resolution)) + LOW_WALL_CELLS + 1
      x0, x1 = max(cx - r, 0), min(cx + r + 1, cols)
      y0, y1 = max(cy - r, 0), min(cy + r + 1, rows)
      mask = np.zeros_like(g.grid, dtype=bool)
      if x0 < x1 and y0 < y1:
        occ = g.grid[y0:y1, x0:x1] > OCC_THRESH
        if occ.any():
          mask[y0:y1, x0:x1] = ndimage.binary_dilation(occ, iterations=LOW_WALL_CELLS)
      self._walls_mask, self._walls_t = mask, self._next_scan
    return self._walls_mask

  # ---- standing, lying, facing -------------------------------------------------

  def stand_routine(self) -> Routine:
    """Onto its feet, if it is not on them: True once standing."""
    t0 = float(self.data.time)
    self.want = "stand"
    try:
      while self.posture != STANDING:
        if self.data.time - t0 > POSTURE_BUDGET_S:
          return False
        yield STILL
        self._after_step()
    finally:
      self.want = None
    return True

  def rest_routine(self) -> Routine:
    """Down onto its belly, if it is not there: True once lying."""
    t0 = float(self.data.time)
    self.want = "lie"
    try:
      while self.posture != LYING:
        if self.data.time - t0 > POSTURE_BUDGET_S:
          return False
        yield STILL
        self._after_step()
    finally:
      self.want = None
    return True

  def face_routine(self, heading: float) -> Routine:
    """Turn on the spot to a heading, at a rate the policy tracks: False if
    its budget ran out first."""
    t0 = float(self.data.time)
    while True:
      err = dk._wrap(heading - self.pose[2])
      if abs(err) <= self.FACING_TOLERANCE:
        break
      if self.data.time - t0 > self.FACE_BUDGET_S:
        return False
      w = max(-W_MAX, min(W_MAX, self.FACE_GAIN * err))
      yield from self._twist_routine(0.0, 0.0, w)
    yield from self._drive_routine(dk.TURN_SETTLE_S, 0.0, 0.0)
    return True

  def travel_routine(self, distance: float, v: float) -> Routine:
    """`distance` of believed travel along its heading at `v` (negative
    backs up): "arrived", or "timeout" at twice the time it should take."""
    x0, y0 = self.pose_xy()
    speed = max(abs(v), V_MIN)
    budget = abs(distance) / speed * 2.0 + 2.0
    t0 = float(self.data.time)
    sign = math.copysign(1.0, distance) * math.copysign(1.0, v)
    while math.hypot(self.odo.x - x0, self.odo.y - y0) < abs(distance):
      if self.data.time - t0 > budget:
        return "timeout"
      yield from self._twist_routine(sign * speed, 0.0, 0.0)
    return "arrived"

  # ---- the dock (#378) ----------------------------------------------------------

  def _board_detector(self):
    if self._board is None:
      from pluggybot.rack.tags import DOCK_TAG_SIZE, TagDetector
      self._board = TagDetector(self.model, self.handle.el("nav_eye"),
                                tag_size=DOCK_TAG_SIZE)
    return self._board

  def detect_board(self) -> dict:
    """One decode from the nose camera."""
    return self._board_detector().detect(self.data)

  def look_at_board(self) -> dk.DockFix | None:
    """One look; a fit moves the dock's believed pose (odometry frame)."""
    seen = dk.seen_from(self.model, self.data, self.detect_board(),
                        self.handle.el("nav_eye"), self.root)
    fix = dk.fit_dock(seen)
    if fix is not None:
      self.dock_seen = dk.blend(self.dock_seen,
                                dk.compose(self.pose, (fix.x, fix.y, fix.yaw)))
    return fix

  def _dock_error(self) -> tuple[float, float, float]:
    return dk.relative(self.pose, self.dock_seen)

  def charging(self) -> bool:
    return (self.dock_prior is not None
            and dk.dock_charge_contact(self.model, self.data, self.handle.prefix))

  def _turn_by_routine(self, degrees: float) -> Routine:
    target = self.pose[2] + math.radians(degrees)
    t0 = float(self.data.time)
    while (abs(dk._wrap(target - self.pose[2])) > dk.TURN_TOL
           and self.data.time - t0 < dk.TURN_BUDGET_S):
      yield from self._twist_routine(
        0.0, 0.0, math.copysign(dk.TURN_W, dk._wrap(target - self.pose[2])))
    yield from self._drive_routine(dk.TURN_SETTLE_S, 0.0, 0.0)

  def _find_board_routine(self) -> Routine:
    if self.look_at_board() is not None:
      return True
    for step in dk.SEARCH_STEPS:
      yield from self._turn_by_routine(step)
      if self.look_at_board() is not None:
        return True
    return False

  def _walk_in_routine(self) -> Routine:
    t0 = last = float(self.data.time)
    while self.data.time - t0 < dk.WALK_IN_S:
      tw = dk.walk_in_twist(*self._dock_error())
      if tw == Twist():
        return "stopped"
      yield from self._twist_routine(tw.vx, tw.vy, tw.yaw_rate)
      if self.data.time - last >= dk.LOOK_EVERY_S:
        self.look_at_board()
        last = float(self.data.time)
    return "budget"

  def _back_out_routine(self) -> Routine:
    x0, y0 = self.pose_xy()
    t0 = float(self.data.time)
    while (math.hypot(self.odo.x - x0, self.odo.y - y0) < dk.BACK_OUT_M
           and self.data.time - t0 < dk.BACK_OUT_S):
      yield from self._twist_routine(-dk.APPROACH_V, 0.0, 0.0)
    yield from self._drive_routine(dk.BACK_OUT_SETTLE_S, 0.0, 0.0)

  def dock_routine(self) -> Routine:
    """From the standoff: find the board, walk in by it, stop, check the
    line-up, lie down; again from a backed-out start if the pads do not
    conduct (`legs/dock.py`; SimNotes, "The quadruped's dock"). Returns why
    it stopped: "docked", or the last attempt's failure."""
    rec = {"attempts": []}
    self.last_charge = rec
    why = "no dock"
    if self.dock_prior is None:
      return why
    self.dock_seen = None
    yield from self.stand_routine()
    # ...facing the board, as the standoff has it: the walk to the standoff
    # arrives facing wherever it came from, and the search turns reach only
    # 75 deg either side of where they start
    yield from self.face_routine(self.charge_standoff()[2])
    for _ in range(dk.TRIES):
      att = {"err": self.truth_error()}
      rec["attempts"].append(att)
      if not (yield from self._find_board_routine()):
        why = att["why"] = "no board"
        break
      walked = yield from self._walk_in_routine()
      yield from self._drive_routine(dk.STOPPED_S, 0.0, 0.0)
      att["fix"] = self.look_at_board() is not None
      ex, ey, eth = self._dock_error()
      att["stop"] = [round(ex, 3), round(ey, 3), round(math.degrees(eth), 1)]
      if (walked != "stopped" or abs(ey) > dk.LIE_ACROSS_M or abs(eth) > dk.LIE_YAW
          or abs(ex - dk.LIE_SHIFT_M) > dk.LIE_ALONG_M):
        why = att["why"] = "not lined up" if walked == "stopped" else f"walk {walked}"
        yield from self._back_out_routine()
        continue
      yield from self.rest_routine()
      if self.charging():
        why = att["why"] = "docked"
        return why
      why = att["why"] = "no contact"
      yield from self.stand_routine()
      yield from self._back_out_routine()
    return why

  def anchor_at_dock(self) -> None:
    """Lying on the dock: the board, read from where it lies, puts the body
    in the dock's frame to millimetres, and the dock's frame IS the map's
    (#378, "the dock is the map's origin"). With no decode, the seat itself:
    the funnel lays it within a few mm and 2 deg."""
    seen = dk.seen_from(self.model, self.data, self.detect_board(),
                        self.handle.el("nav_eye"), self.root)
    fix = dk.fit_dock(seen)
    x, y, yaw = self.dock_prior
    if fix is not None:
      # the robot in the dock's frame: the dock's origin seen from the robot,
      # inverted
      me = dk.relative((0.0, 0.0, 0.0), (fix.x, fix.y, fix.yaw))
    else:
      me = (0.0, 0.0, 0.0)
    wx, wy, wyaw = dk.compose((x, y, yaw), me)
    self.odo.correct(wx, wy, wyaw)
    if self.last_charge is not None:
      self.last_charge["anchor"] = "board" if fix is not None else "seat"

  def undock_routine(self) -> Routine:
    """Up off the dock and back out along its axis."""
    yield from self.stand_routine()
    yield from self._back_out_routine()
    return "arrived"

  # ---- tools: none, until the arm (#378) ---------------------------------------

  def module_state(self, module: str) -> dict:
    """Where a module is, off the world: never on this body (it has no arm
    yet); hung, if it sits in a bay as the rover's swap would have it."""
    d, m = self.data, self.model
    p = d.xpos[m.body(module).id]
    bid = m.body("rack").id
    rp, rm = d.xpos[bid], d.xmat[bid].reshape(3, 3)
    lx, ly, lz = (rm.T @ (p - rp)).tolist()
    peg_rest_z = HUB_PEG_Z - TRAY_VERTEX_DROP + PEG_R
    bay = min(range(len(STATION_YS)), key=lambda i: abs(ly - STATION_YS[i]))
    hung = (abs(lx - RACK_HANG_X) < 0.012 and abs(lz - (peg_rest_z - 0.022)) < 0.008
            and abs(ly - STATION_YS[bay]) < 0.030)
    return {"pos": [float(v) for v in p], "on_fork": False, "hung": hung,
            "bay": bay}

  def close(self) -> None:
    if self._board is not None:
      self._board.close()
      self._board = None

  def kept_state(self) -> tuple[dict, dict]:
    """What it believes, for a restart (issue #345): the reckoning, the
    posture it rests in, the dock and the clocks; the grid and the depth
    camera's layer as arrays."""
    o = self.odo
    match, match_arrays = (self.matcher.kept_state() if self.matcher is not None
                           else (None, {}))
    return ({"odometry": {"x": o.x, "y": o.y, "z": o.z, "v": o.v.tolist(),
                          "held": o.held, "quat": list(o.att.q),
                          "distance": o.distance,
                          "rng": o.rng.bit_generator.state,
                          "imu": o.imu.kept_state()},
             "posture": self.posture, "lastMotion": self.last_motion_t,
             "dockSeen": None if self.dock_seen is None else list(self.dock_seen),
             "clocks": {"scan": self._next_scan, "backoff": self.backoff_until},
             "peerSeen": [self.peer_seen_m, self.peer_seen_t],
             "lidarRng": [self.lidar.rng.bit_generator.state,
                          self.lidar.peer_rng.bit_generator.state],
             "depthRng": [self.depth.rng.bit_generator.state,
                          self.depth.peer_rng.bit_generator.state],
             "match": match},
            {"grid": self.grid.grid, "low": self.low, **match_arrays})

  def restore_kept(self, state: dict, arrays: dict) -> bool:
    o, s = self.odo, state["odometry"]
    o.x, o.y, o.z = s["x"], s["y"], s["z"]
    o.v = np.array(s["v"])
    o.held = int(s["held"])
    o.att.q = tuple(float(v) for v in s["quat"])
    o.yaw = o.att.yaw()
    o.distance = float(s["distance"])
    o.rng.bit_generator.state = s["rng"]
    o.imu.restore_kept(s["imu"])
    # A lying body is laid back down by the world's own bodies; a posture
    # mid-move goes on as the one it was heading for.
    self.posture = {LYING_DOWN: LYING, STANDING_UP: STANDING}.get(
      state["posture"], state["posture"])
    self.last_motion_t = float(state["lastMotion"])
    self.dock_seen = (None if state.get("dockSeen") is None
                      else tuple(state["dockSeen"]))
    clocks = state.get("clocks", {})
    self._next_scan = float(clocks.get("scan", 0.0))
    self.backoff_until = float(clocks.get("backoff", 0.0))
    self.peer_seen_m, self.peer_seen_t = state.get("peerSeen", [None, 0.0])
    self.lidar.rng.bit_generator.state = state["lidarRng"][0]
    self.lidar.peer_rng.bit_generator.state = state["lidarRng"][1]
    self.depth.rng.bit_generator.state = state["depthRng"][0]
    self.depth.peer_rng.bit_generator.state = state["depthRng"][1]
    grid = arrays.get("grid")
    mapped = grid is not None and grid.shape == self.grid.grid.shape
    if mapped:
      self.grid.grid[...] = grid
      if arrays.get("low") is not None:
        self.low[...] = arrays["low"]
    if self.matcher is not None and state.get("match") and mapped:
      self.matcher.restore_kept(state["match"], arrays)
    return mapped

  def rebind(self, model, data) -> None:
    """⚠ NOT DONE: a quadruped's world is never recompiled -- the seam hangs
    a built tool (issue #168), and this body has no arm to take one until
    #378's; its world has no workshop (`world_config`'s `built_bays`). Its
    drivers, policies, reckoning and pack hold ids by the dozen, and a
    rebind that missed one would drive another body's joints, so it is
    refused out loud, not half-done."""
    raise NotImplementedError("a quadruped's world is not recompiled: it hangs "
                              "no tool until its arm (#378)")

  def start_at(self, x: float, y: float, yaw: float) -> None:
    """Stand the body at a pose, on its feet, and tell its reckoning so --
    a mission's start, and a stand-up after a death. Steps a second."""
    from pluggybot.legs.world import stand
    stand(self.model, self.data, self.handle.prefix, x, y, yaw)
    self._move, self.want = None, None
    self.posture = STANDING
    self._slumped_since = self._stood_since = None
    self.docked = False
    self.odo = LegOdometry(self.model, self.data, prefix=self.handle.prefix)
    self.odo.correct(x, y, yaw)
    self._vm = None
    self._handover(self.walker)
    self.last_motion_t = float(self.data.time)
    self.run(self._drive_routine(1.0, 0.0, 0.0))


class QuadBody(Body):
  """The quadruped: `QuadMission` and everything it owns, as a `Body`.
  Every member hands the call to the mission by name at CALL time, as
  `RoverBody` does, so a stub on the mission is what the body runs."""

  STILL = STILL
  level_tilt_rad = QuadMission.LEVEL_TILT
  #: A fall is got up from (the get-up policy) until it has lasted this
  #: long, s: then it is the `stuck` death. MEASURED (`quad_spike.py
  #: --getup`, SimNotes "The first quadruped deploy").
  stuck_after_s = 20.0
  #: ...and this body rights itself.
  rights_itself = True
  swapping_at = None
  peer_at_bay_m = None
  bay_wait = None
  rack_discovered = False

  def __init__(self, model, data, viewer=None, realtime: bool = True,
               rack=None, grid_bounds=(-3, -3, 7, 7),
               handle: RobotHandle = FIRST) -> None:
    self.mission = QuadMission(model, data, viewer=viewer, realtime=realtime,
                               rack=rack, grid_bounds=grid_bounds, handle=handle)

  # ---- which robot, and the seam -------------------------------------------

  handle = property(lambda self: self.mission.handle)
  model = property(lambda self: self.mission.model)
  data = property(lambda self: self.mission.data)
  step_hooks = property(lambda self: self.mission.step_hooks)
  stepper = property(lambda self: self.mission.stepper)

  def run(self, routine: Routine, name: str = ""):
    return self.mission.run(routine, name)

  def rebind(self, model, data) -> None:
    self.mission.rebind(model, data)

  def kept_state(self):
    return self.mission.kept_state()

  def restore_kept(self, state, arrays) -> bool:
    return self.mission.restore_kept(state, arrays)

  def close(self) -> None:
    self.mission.close()

  # ---- going places --------------------------------------------------------

  def start_at(self, x, y, yaw) -> None:
    self.mission.start_at(x, y, yaw)

  def go_to_routine(self, x, y, timeout=None) -> Routine:
    if timeout is None:
      return self.mission.drive_to_routine(x, y)
    return self.mission.drive_to_routine(x, y, timeout=timeout)

  last_drive = property(lambda self: self.mission.last_drive)

  def gave_up(self, record, peer="the other robot") -> str:
    return gave_up(record, peer).replace("the drive gave up", "the walk gave up", 1)

  def face_routine(self, heading) -> Routine:
    return self.mission.face_routine(heading)

  def hold_routine(self, seconds) -> Routine:
    return self.mission._drive_routine(seconds, 0.0, 0.0)

  def look_around_routine(self) -> Routine:
    return self.mission._spin_routine()

  @property
  def grid(self):
    return _PlannedGrid(self.mission)

  def plan_frontier(self, blacklist):
    from pluggybot.behavior.navigation import plan
    return plan(self.grid, self.pose, blacklist, self.mission.INFLATION_CELLS,
                own_component=True)

  def reachable(self, points):
    return self.mission.reachable(points)

  def in_sight(self, x, y) -> bool:
    return self.mission.in_sight(x, y)

  # ---- where it is ---------------------------------------------------------

  @property
  def pose(self):
    return self.mission.pose

  def pose_xy(self):
    return self.mission.pose_xy()

  def true_pose(self):
    return self.mission.true_pose()

  def root_xy(self):
    q = self.handle.qpos_adr(self.model)
    return self.data.qpos[q:q + 2]

  def orientation(self):
    q = self.handle.qpos_adr(self.model)
    return self.data.qpos[q + 3:q + 7]

  def footprint_centre(self):
    return self.mission.footprint_centre()

  def as_seen(self, x, y):
    return self.mission.as_seen(x, y)

  def level(self) -> bool:
    return self.mission.level()

  # ---- the dock, as the loop's "rack" -----------------------------------------

  rack = property(lambda self: self.mission.rack)
  rack_prior = property(lambda self: self.mission.rack_prior)

  def start_discovery(self) -> None:
    pass

  def refresh_rack(self):
    return None

  def bay_standoff(self, station_y):
    # the tool rack's bays: this body has nothing to take from them
    x, y = RackPose.prior().to_world(0.5, station_y)
    return x, y, RackPose.prior().heading

  def fetch_tool_routine(self, station_y, module) -> Routine:
    return "no arm"
    yield

  def stow_tool_routine(self, station_y, module) -> Routine:
    return "no arm"
    yield

  def module_state(self, module) -> dict:
    return self.mission.module_state(module)

  def tool_powered(self, module) -> bool:
    return False

  def seated_on(self, module):
    return None

  def tool(self, module, **kw):
    return None

  def swap_trace(self) -> str:
    return "no arm yet: this body takes no tool"

  def charge_standoff(self):
    return self.mission.charge_standoff()

  def dock_routine(self) -> Routine:
    return self.mission.dock_routine()

  def charging(self) -> bool:
    return self.mission.charging()

  @property
  def docked(self) -> bool:
    return self.mission.docked

  @docked.setter
  def docked(self, held: bool) -> None:
    self.mission.docked = held

  def anchor_at_dock(self) -> None:
    self.mission.anchor_at_dock()

  def dock_hold_routine(self, seconds) -> Routine:
    return self.mission._drive_routine(seconds, 0.0, 0.0)

  def redock_routine(self) -> Routine:
    # lying on the pins, nothing to press: the contact settles or it does not
    return self.mission._drive_routine(1.0, 0.0, 0.0)

  def undock_routine(self) -> Routine:
    return self.mission.undock_routine()

  def charge_trace(self) -> str:
    rec = self.mission.last_charge
    if not rec or not rec.get("attempts"):
      return "no dock approach recorded"
    parts = []
    for i, a in enumerate(rec["attempts"], 1):
      e = a.get("err") or [0.0, 0.0, 0.0]
      parts.append(f"#{i} belief off {e[0]:+.0f},{e[1]:+.0f} mm {e[2]:+.1f} deg, "
                   f"stopped at {a.get('stop')} -> {a.get('why')}")
    return "; ".join(parts)

  # ---- senses --------------------------------------------------------------

  def pack(self, capacity_wh, charge_scale):
    return LegPack(self.model, capacity_wh, charge_scale, self.handle.prefix)

  def depth_camera(self):
    return _SharedDepth(self.mission)

  def know_peer(self, root) -> None:
    self.mission.lidar.exclude_robot(root)
    self.mission.depth.exclude_robot(root)

  def detect_tags(self) -> dict:
    return self.mission.detect_board()

  def spot(self, tag, at_height=None):
    return None

  pressing = property(lambda self: self.mission.pressing)

  # ---- the others, and collisions ------------------------------------------

  @property
  def others(self):
    return self.mission.others

  @others.setter
  def others(self, where) -> None:
    self.mission.others = where

  def peer_on_the_goal(self, x, y):
    return self.mission.peer_on_the_goal(x, y)

  def watch_for_peers(self, points):
    return self.mission.watch_for_peers(points)

  geom_ids = property(lambda self: self.mission.body_gids)
  peer_holds = property(lambda self: self.mission.peer_holds)
  collision_steps = property(lambda self: self.mission.collision_steps)
  press_steps = property(lambda self: self.mission.press_steps)

  # ---- the motor level -----------------------------------------------------

  def actuator(self, name) -> int:
    raise KeyError(f"no actuator {name!r} a procedure may move: the legs are "
                   "the walking policy's, and the arm is not built yet")

  def ramp_routine(self, act, target, speed, settle=0.0) -> Routine:
    return self.mission._drive_routine(settle, 0.0, 0.0)

  def settle_routine(self, seconds) -> Routine:
    for _ in range(round(seconds / self.model.opt.timestep)):
      yield STILL

  def velocity_routine(self, seconds, v, w) -> Routine:
    return self.mission._drive_routine(seconds, v, w)

  def travel_routine(self, distance, v) -> Routine:
    return self.mission.travel_routine(distance, v)

  def retract_arm_routine(self) -> Routine:
    return
    yield

  # ---- posture and rest ----------------------------------------------------

  @property
  def resting(self) -> bool:
    return self.mission.posture == LYING

  @property
  def posture(self) -> str:
    return self.mission.posture

  def rest_routine(self) -> Routine:
    return self.mission.rest_routine()

  def stand_routine(self) -> Routine:
    return self.mission.stand_routine()


class _PlannedGrid:
  """The body's map as the loop reads it -- the explore's planner and the
  stream: the LIDAR's grid with what the depth camera saw under its plane
  (`QuadMission._planning_grid`), so the frontier the explore picks is one
  the drive can reach, and the map on the site is the map it plans over.
  Read-only; the scans go into the LIDAR's own grid."""

  def __init__(self, mission: QuadMission) -> None:
    self._mission = mission

  @property
  def grid(self) -> np.ndarray:
    return self._mission._planning_grid()

  def to_image(self) -> np.ndarray:
    from pluggybot.mapping.occupancy_grid import OccupancyGrid
    return OccupancyGrid.to_image(self)

  def __getattr__(self, name):
    return getattr(self._mission.grid, name)


class _SharedDepth:
  """The body's depth camera as the loop's near-field reads it: the SAME
  frames the planner's layer is built from (`QuadMission.latest_depth`),
  levelled into the robot frame. Its draw is in the body's electronics
  (`LegPack`), so the loop adds none."""

  draw_w = 0.0

  def __init__(self, mission: QuadMission) -> None:
    self.mission = mission

  @property
  def rng(self):
    return self.mission.depth.rng

  @property
  def peer_rng(self):
    return self.mission.depth.peer_rng

  def frame(self, data) -> DepthFrame:
    return self.mission.latest_depth()

  def rebind(self, model) -> None:
    pass

  def exclude_robot(self, root_name: str) -> None:
    pass
