"""The humanoid on the map (issue #507, stage 4): the `Navigator` over a
humanoid -- the map, the planner, the LIDAR, the scan matcher and the drive
the quadruped walks by -- with what only this body says: where it believes
it is (its leg odometry and IMU, `humanoid.odometry`), how it walks a
velocity (its walking policy, `humanoid.walk`), its front stop, and what
its depth camera sees under its LIDAR's plane.

⚠ THE LIDAR IS ON THE HEAD, 1.41 m up, OVER every table, counter, couch
and bed in the house: a map of its plane alone plans straight through
them. So the D435's points between the floor and that plane are a layer of
their own the planner plans round (`_fold_low`, the quadruped's rule with
the humanoid's band, `LOW_MAX_Z`).

Its legs are one of two policies': the WALKING one's while it moves, the
STANDING one's while it stands to reach (`humanoid.policy`, `humanoid.
reach`). The legs change hands only at a standstill, the arms home
(`to_walk_routine`, `to_stand_routine`).

A FALL is its get-up policy's (stage 5, `humanoid.getup`): once its senses
say it is falling -- tipped 35 deg past the posture its legs hold (a bow
the standing policy was asked is no fall, #520), or folded down under 0.40
m over its soles -- every joint, arms too, goes to that policy until it
stands, and the legs come back to the standing policy (`legs` is "getup"
between).

A walker that SEES THE GROUND (`walk.SEEING_NPZ`) is shown the
D435's map at its grid (`humanoid.scan.HeadScan`): the scan then owns the
camera's frames, the low layer folds the scan's (one render, not two), and
the odometry keeps its height and finds its planted foot on the map
(`humanoid.odometry`).
"""

import math

import mujoco
import numpy as np

from pluggybot import tick
from pluggybot.humanoid import model as hm
from pluggybot.humanoid.arms import SelfModel, kept_value, value_back
from pluggybot.humanoid.getup import GETUP_NPZ, FallWatch, GetupDriver
from pluggybot.humanoid.odometry import StanceOdometry
from pluggybot.humanoid.policy import StandDriver, StandPolicy
from pluggybot.humanoid.reach import Reach
from pluggybot.humanoid.scan import HeadScan
from pluggybot.humanoid.walk import WALK_NPZ, WalkDriver
from pluggybot.humanoid.way import HumanoidWay
from pluggybot.mapping.frontier import OCC_THRESH
from pluggybot.navigator import Navigator
from pluggybot.perception.depth import UNCULLED, DepthCamera, DepthFrame
from pluggybot.robot import FIRST, RobotHandle

STILL = (0.0, 0.0, 0.0)

#: A velocity as the walking policy walks it (`command_for`): under these
#: it is no motion at all, m/s and rad/s; a forward speed under `V_MIN`
#: with a turn to make is a turn on the spot; and the fastest it trained
#: to track (`walk.LIN_X`, `LIN_Y`, `ANG_Z`).
MOTION_EPS = 1e-3
V_MIN, W_MIN = 0.2, 0.3
VX_RANGE, VY_MAX, W_MAX = (-0.8, 1.4), 0.4, 1.0
#: The standing policy's hand-back: the walking one is held still this
#: long first, s, its stride settled.
SETTLE_S = 0.8

#: THE DEPTH CAMERA'S OBSTACLES: a point this high over the floor, and no
#: higher, is something the body walks into -- over a foot's clearance
#: (the walking policy is trained on flat ground), under the LIDAR's plane
#: (1.41 m) less a margin. ...and a point under `FLOOR_Z` is floor.
LOW_MIN_Z, LOW_MAX_Z, FLOOR_Z = 0.08, 1.35, 0.03
#: The quadruped's rules for the layer (`legs.body`): points within this
#: of the trunk, three frames' evidence, never beside what the LIDAR maps.
LOW_RANGE_M, LOW_WALL_CELLS = 1.8, 3
LOW_HIT, LOW_MISS, LOW_CLAMP, LOW_OCC = 0.9, 0.4, 4.0, 2.5
LOW_BURN = OCC_THRESH + 2.0
#: A depth frame this often, s -- the ground's map's too (`humanoid.scan`):
#: at the part's 10 Hz, its ray-cast frames slowed one robot's real time
#: from 2.7x to 1.8x, and at 1 m/s two frames 0.2 m apart still overlap.
DEPTH_PERIOD = 0.2


#: What each policy's driver carries between its policy steps, beside its
#: step count (`HumanoidMission.kept_state`).
_DRIVER = {"stand": ("last_action", "goal", "posture", "target"),
           "walk": ("last_action", "twist", "target"),
           "getup": ("last_action", "target", "_still")}


def command_for(vx: float, vy: float, w: float) -> tuple[float, float, float]:
  """A velocity as the walking policy walks it: out of its dead band and
  inside what it trained on (the quadruped's rule, `legs.body.command_for`:
  a creep with a turn to make is a pivot)."""
  if abs(vx) < MOTION_EPS and abs(vy) < MOTION_EPS and abs(w) < MOTION_EPS:
    return STILL
  if MOTION_EPS <= abs(vx) < V_MIN and abs(w) >= W_MIN:
    vx = 0.0
  if abs(vx) >= MOTION_EPS:
    vx = math.copysign(max(abs(vx), V_MIN), vx)
  elif abs(w) >= MOTION_EPS and abs(vy) < MOTION_EPS:
    w = math.copysign(max(abs(w), W_MIN), w)
  return (min(max(vx, VX_RANGE[0]), VX_RANGE[1]), min(max(vy, -VY_MAX), VY_MAX),
          min(max(w, -W_MAX), W_MAX))


def _site_in_root(model, site: int, root: int) -> np.ndarray:
  pos = np.array(model.site_pos[site], dtype=float)
  body = int(model.site_bodyid[site])
  while body != root:
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, model.body_quat[body])
    pos = model.body_pos[body] + mat.reshape(3, 3) @ pos
    body = int(model.body_parentid[body])
  return pos


class HumanoidStepper:
  """Steps one humanoid's physics (`tick.py`'s stepper): its legs' policy,
  its arms and its senses before the world steps; its reckoning and its
  hooks after."""

  STILL = STILL

  def __init__(self, mission: "HumanoidMission") -> None:
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


class HumanoidMission(HumanoidWay, Navigator):
  """The Navigator over a humanoid (module docstring), making way for the
  other robot as it rests (`humanoid.way`, #559)."""

  #: Its sizes. Half its width with its hands hanging is 0.27 m: the map's
  #: inflation is that and 8 cm, as the quadruped's covers its half-width.
  INFLATION_CELLS = 7
  #: The disc round another robot, cells: its outline standing (0.27 m) and
  #: this one's and 10 cm; fallen, a 1.4 m body from the middle of its
  #: footprint, 0.75 m, and this one's half-width.
  OTHER_ROBOT_CELLS = 13
  DOWN_ROBOT_CELLS = 21
  #: The front stop, from the LIDAR over the pelvis: its toes and its hands
  #: hanging reach 0.2 m ahead of it, and it stops 0.2 m short of them --
  #: inside the clearance the planner grants (the inflation and 8 cm).
  FRONT_STOP_RANGE = 0.40
  FRONT_HALF_M = 0.27
  BACKOFF_V = 0.3
  PEER_STOP_AHEAD_M, PEER_STOP_HALF_M, PEER_CLEARANCE_M = 0.75, 0.27, 0.34
  FACING_TOLERANCE = math.radians(4.0)
  PROGRESS_ALONG_ROUTE = True
  OPTIMISTIC = True
  PROGRESS_MAP_GROWTH = True
  #: Its IMU's tilt under which the LIDAR's scan is a map of the room: a
  #: walking trunk rocks a degree or so, and a 1.41 m plane tilted 2 deg
  #: reaches the floor at 40 m, past the LIDAR's 8.
  LEVEL_TILT = math.radians(2.0)
  #: The odometry's tick, s.
  ODO_S = 0.01

  def __init__(self, model, data, handle: RobotHandle = FIRST,
               stand: StandPolicy | None = None, walk: StandPolicy | None = None,
               grid_bounds=(-3, -3, 7, 7), viewer=None, realtime: bool = False,
               getup: StandPolicy | None = None) -> None:
    self._getup_policy = getup if getup is not None else (
      StandPolicy(GETUP_NPZ) if GETUP_NPZ.exists() else None)
    self.stepper = HumanoidStepper(self)
    root = model.body(handle.root).id
    lidar = _site_in_root(model, model.site(handle.el("lidar")).id, root)
    #: The LIDAR's place on the trunk: the scan's origin for the map.
    self.LIDAR_ORIGIN = (float(lidar[0]), float(lidar[1]))
    super().__init__(model, data, handle=handle, grid_bounds=grid_bounds,
                     viewer=viewer, realtime=realtime, match=True)
    self.root = root
    self._build_drivers(stand, walk)
    self._odo_every = max(1, round(self.ODO_S / model.opt.timestep))
    self._steps = 0
    #: Whose the legs are: "stand" (the standing policy's), "walk", or
    #: "getup" (the get-up policy's, and the arms with them).
    self.legs = "stand"
    #: Falls the get-up policy has taken over.
    self.falls = 0
    #: A tag watched while it moves (`perception.watch.TagWatch`, the
    #: body's `watch`), stepped after every physics step; None for none.
    self.sight = None
    self.depth = DepthCamera(model, handle, mount="body", reach=UNCULLED)
    self._frame: DepthFrame | None = None
    self._frame_t = -math.inf
    self._frame_used_t = -math.inf
    self._frame_level = False
    self.low = np.zeros_like(self.grid.grid, dtype=np.float32)
    self._walls_t, self._walls_mask = None, None
    self._pressing = False
    self._init_way()

  # ---- what only a body says ------------------------------------------------

  @property
  def pose(self):
    return self.odo.x, self.odo.y, self.odo.yaw

  def pose_xy(self) -> tuple[float, float]:
    return self.odo.x, self.odo.y

  def _set_pose(self, x: float, y: float, theta: float) -> None:
    self.odo.correct(x, y, theta)

  def true_pose(self):
    d = self.data
    rot = d.xmat[self.root].reshape(3, 3)
    return (float(d.xpos[self.root][0]), float(d.xpos[self.root][1]),
            math.atan2(rot[1, 0], rot[0, 0]))

  def level(self) -> bool:
    return self.stand.trunk.attitude.tilt() <= self.LEVEL_TILT

  @property
  def pressing(self) -> bool:
    return self._pressing

  def run(self, routine, name: str = ""):
    return tick.run(self.stepper, routine, name)

  # ---- the physics seam ------------------------------------------------------

  def _before_step(self, command) -> None:
    if self._aside is not None:
      # Making way (#559, `humanoid.way`) beneath whatever routine holds it:
      # its legs to the walker where they are the standing policy's, its
      # posture back at the stand first, as `to_walk_routine` hands them
      # over -- a bow left held (a `spot` cut off) is no stance to walk from
      command = self._aside_command(command, float(self.data.time))
      if any(command) and self.legs == "stand":
        if self.reach.posture.any():
          self.reach.hold_posture(0.0, 0.0)
        if not self.stand.settled:
          command = STILL
        else:
          self.walker.last_action[:] = 0.0
          self.legs = "walk"
    if any(command):
      self.last_motion_t = float(self.data.time)
    if self.legs == "getup":
      # Getting up: every joint the get-up policy's, a walk asked meanwhile
      # walks nowhere.
      self.getup.step()
      if self.getup.stood:
        self._stood_up()
      return
    if self.legs == "walk":
      self.walker.ask(*command)
      self.walker.step()
    else:
      if any(command):
        raise RuntimeError("a walk was asked of the standing policy's legs: "
                           "`to_walk_routine` hands them over")
      self.stand.step()
    # ...a hand's force off the whole body only on both soles (`Reach.standing`)
    self.reach.standing = self.legs == "stand"
    self.reach.step()
    if (self.getup is not None and self._steps % self._fall_every == 0
        and self.fall.falling(self.data, self._held_pitch())):
      self._fell()

  def _held_pitch(self) -> float:
    """The trunk's pitch its legs' policy holds, rad: the standing policy's
    posture (`Reach.held_pitch`), the walker's upright."""
    return self.reach.held_pitch() if self.legs == "stand" else 0.0

  def _fell(self) -> None:
    """The get-up policy takes every joint: the arms' reach is let go."""
    self.falls += 1
    self.legs = "getup"
    self.getup.begin()

  def _stood_up(self) -> None:
    """Stood again: the legs back to the standing policy at the stand, the
    arms to a fresh reach from where they hang, the grippers as they were
    asked (no policy drives them; built afresh, a shut gripper read as
    never shut, #520) -- and, as at boot, its height the stand's over the
    ground it stands on, the ground's map laid anew. ⚠ Kept through a fall
    down the flight, the height was 0.41 m off at the foot, past what a
    frame registers, for good."""
    key = f"humanoid:{self.handle.prefix}"
    self.stand.last_action[:] = 0.0
    self.stand.posture[:] = 0.0
    self.stand.ask(0.0, 0.0)
    grippers = self.reach.grippers
    self.reach = Reach(self.model, self.data, self.stand, prefix=self.handle.prefix, key=key)
    self.reach.grippers.update(grippers)
    self.legs = "stand"
    if self.head is not None:
      self.head.forget()
      self.odo.z, self.odo.foot = hm.stand_height(), None

  def _after_physics(self) -> None:
    # The reckoning first: the scan lays its frame through it.
    if self._steps % self._odo_every == 0:
      q, _, _ = self.stand.joints.read(self.data)
      self.odo.step(q, self.stand.trunk.gravity())
    if self.head is not None:
      self.head.step()
    if self.sight is not None:
      # ...nothing seen while the get-up policy has every joint: no arm's
      # loop poses its model of itself, so no picture can be placed
      self.sight.step(float(self.data.time), sees=self.legs != "getup")
    self._steps += 1
    self._on_step()

  # ---- whose legs ------------------------------------------------------------

  def _getup_routine(self):
    """Held still while a fall's get-up has the legs, until it stands. ⚠
    Unheld, a walk asked of a body that fell with its arms home found them
    at rest and the posture settled, and took the legs off the floor. ⚠ No
    budget of its own, as the quadruped's get-up has none: a body that
    cannot stand is the lifecycle's `stuck` death after `stuck_after_s`,
    whose stand-up closes the routine; a script with no lifecycle bounds
    its own run."""
    while self.legs == "getup":
      yield STILL
      self._after_step()

  def to_walk_routine(self):
    """The legs to the walking policy: the arms home and the posture back to
    the stand first (a walk carries nothing yet), then handed over. ⚠ A
    fall as they came home went to the walker if its step was the one that
    settled them (#520): the get-up stands it first."""
    yield from self._getup_routine()
    if self.legs == "walk":
      return
    for s in hm.SIDES:
      if self.reach.phase[s] not in ("rest",):
        self.reach.retreat(s)
    self.reach.hold_posture(0.0, 0.0)
    while (any(p != "rest" for p in self.reach.phase.values())
           or not self.stand.settled):
      yield STILL
      self._after_step()
    yield from self._getup_routine()
    self.walker.last_action[:] = 0.0
    self.legs = "walk"

  def to_stand_routine(self):
    """The legs to the standing policy: the walking one held still
    `SETTLE_S` first, its stride settled, then handed over. ⚠ A fall while
    it settles is the get-up's until it stands (#520): at the settle's end
    the legs went to the standing policy whatever had happened in it. ⚠ It
    ends a step aside under way (#559): the aside walked the settle's STILL
    steps, and the legs went to the standing policy mid-stride."""
    yield from self._getup_routine()
    if self._aside is not None:
      self._end_aside("its own move", STILL)
    if self.legs == "stand":
      return
    for _ in range(int(round(SETTLE_S / self.model.opt.timestep))):
      if self.legs != "walk":
        break
      yield STILL
      self._after_step()
    if self.legs != "walk":
      yield from self._getup_routine()
      return
    self.stand.last_action[:] = 0.0
    self.stand.posture[:] = 0.0
    self.legs = "stand"

  # ---- the drive -------------------------------------------------------------

  def _nav_routine(self, v: float, w: float):
    yield from self._twist_routine(v, 0.0, w)

  def _twist_routine(self, vx: float, vy: float, w: float):
    """One physics step at a velocity, as the walking policy walks it --
    the legs handed to it first if they were the standing one's."""
    cmd = command_for(vx, vy, w)
    if cmd != STILL and self.legs != "walk":
      yield from self.to_walk_routine()
    yield cmd
    self._after_step()

  def travel_routine(self, distance: float, v: float):
    """`distance` of believed travel along its heading at `v` (negative
    backs up): "arrived", or "timeout" at twice the time it should take."""
    x0, y0, _ = self.pose
    speed = max(abs(v), V_MIN)
    budget = 2 * abs(distance) / speed
    t0 = float(self.data.time)
    while math.hypot(self.pose[0] - x0, self.pose[1] - y0) < abs(distance):
      if float(self.data.time) - t0 > budget:
        return "timeout"
      yield from self._twist_routine(math.copysign(speed, distance), 0.0, 0.0)
    return "arrived"

  def _build_drivers(self, stand: StandPolicy | None, walk: StandPolicy | None) -> None:
    """Its policies' drivers, its senses and its reckoning, where the body
    stands now: the IMU's filter from the boot's orientation, the odometry
    at the pelvis, and -- for a walker that sees the ground -- the scan."""
    model, data, handle = self.model, self.data, self.handle
    key = f"humanoid:{handle.prefix}"
    self.stand = StandDriver(model, data, stand, prefix=handle.prefix, key=key)
    walk = walk or StandPolicy(WALK_NPZ)
    att = self.stand.trunk.attitude
    self.head = None
    if "height_scan" in walk.observation_names:
      self.head = HeadScan(model, data, handle,
                           belief=lambda: (self.odo.x, self.odo.y, self.odo.z, self.odo.yaw),
                           level=att.level, correct=self._height_drifted, period=DEPTH_PERIOD)
    self.walker = WalkDriver(model, data, walk, prefix=handle.prefix, key=key,
                             trunk=self.stand.trunk, joints=self.stand.joints,
                             scan=None if self.head is None else self.head.scan)
    self.reach = Reach(model, data, self.stand, prefix=handle.prefix, key=key)
    self.getup = None if self._getup_policy is None else GetupDriver(
      model, data, self._getup_policy, prefix=handle.prefix, key=key, trunk=self.stand.trunk,
      joints=self.stand.joints)
    self.fall = FallWatch(self.stand.trunk, self.stand.joints)
    self._fall_every = max(1, round(self.stand.policy.period / model.opt.timestep))
    x, y, yaw = self.true_pose()
    # It boots standing on the floor it maps from: its pelvis at the stand's
    # height (the walker's scan reads it before the odometry's first tick).
    self.odo = StanceOdometry(SelfModel(), att, x, y, z=hm.stand_height(),
                              terrain=None if self.head is None else self.head.under,
                              felt=None if self.head is None else self.head.feel,
                              ground_at=None if self.head is None else self.head.ground_at)
    self.odo.correct(x, y, yaw)
    self.legs = "stand"

  def _height_drifted(self, dz: float) -> None:
    self.odo.z += dz

  # ---- a restart (#345; served since #559) ----------------------------------

  def _own_acts(self) -> np.ndarray:
    """Every actuator of this robot's: what its drivers hold lives in the
    MODEL (`legs.drivers.Drivers.pd`: the gains; a gripper's force limit,
    `arms.Gripper`), which a restart builds fresh."""
    m = self.model
    joint = int(mujoco.mjtTrn.mjTRN_JOINT)
    return np.array([a for a in range(m.nu) if int(m.actuator_trntype[a]) == joint
                     and self._under_root(int(m.jnt_bodyid[m.actuator_trnid[a, 0]]))],
                    dtype=int)

  def _under_root(self, body: int) -> bool:
    while body > 0:
      if body == self.root:
        return True
      body = int(self.model.body_parentid[body])
    return False

  def kept_state(self) -> tuple[dict, dict]:
    """What it believes and what its controllers hold, for a restart (#345,
    the quadruped's rule): its reckoning, the three policies' drivers and
    the gains they set in the model, its senses' noise, its reach (`Reach.
    kept_state`), the depth frame it last took, the plan it last made, its
    clocks; the grid, the camera's layer and the walls it is kept off as
    arrays. Never saved with a hand out or getting up (`HumanoidBody.
    working`, `posture`): a hand's move is a routine's. ⚠ The walker that
    sees the ground (not served) keeps no scan: it starts its ground anew,
    as after a fall. ⚠ Nor is the hands' depth cloud's noise kept
    (`HumanoidBody.depth_cloud`, `measure`): nothing served measures until
    the hands are (#559's stage 3)."""
    match, arrays = (self.matcher.kept_state() if self.matcher is not None else (None, {}))
    acts = self._own_acts()
    m = self.model
    arrays = {"grid": self.grid.grid, "low": self.low, **arrays,
              "gains": np.array(m.actuator_gainprm[acts], dtype=float),
              "bias": np.array(m.actuator_biasprm[acts], dtype=float),
              "forceRange": np.array(m.actuator_forcerange[acts], dtype=float)}
    if self._walls_mask is not None:
      arrays["walls"] = self._walls_mask
    frame = None
    if self._frame is not None:
      f = self._frame
      frame = {"t": self._frame_t, "used": self._frame_used_t, "self": f.self_fraction,
               "level": self._frame_level}
      arrays.update(frameZ=f.z, framePoints=f.points, framePeers=f.peers,
                    framePeerGeoms=f.peer_geoms)
    o, trunk = self.odo, self.stand.trunk
    drivers = {}
    for name, drv in (("stand", self.stand), ("walk", self.walker), ("getup", self.getup)):
      if drv is None:
        continue
      drivers[name] = {"steps": drv.steps, **{k: kept_value(getattr(drv, k)) for k in _DRIVER[name]}}
    torqued = {name: d.torqued for name, d in self._all_drivers().items()}
    memo = self._plan_memo
    return ({"legs": self.legs, "falls": self.falls, "steps": self._steps,
             "lastMotion": self.last_motion_t,
             "odometry": {"x": o.x, "y": o.y, "z": o.z, "offset": o.offset, "foot": o.foot,
                          "last": kept_value(o.last)},
             "trunk": {"imu": trunk.imu.kept_state(), "q": list(trunk.attitude.q),
                       "rate": kept_value(trunk.rate)},
             "jointsRng": self.stand.joints.rng.bit_generator.state,
             "drivers": drivers, "torqued": torqued,
             "reach": self.reach.kept_state(),
             "frame": frame,
             "memo": None if memo is None else
             [list(memo[0]), memo[1], list(memo[2]),
              None if memo[3] is None else [list(p) for p in memo[3]],
              None if memo[4] is None else list(memo[4])],
             "clocks": {"scan": self._next_scan, "backoff": self.backoff_until,
                        "walls": self._walls_t},
             "peerSeen": [self.peer_seen_m, self.peer_seen_t],
             "lidarRng": [self.lidar.rng.bit_generator.state,
                          self.lidar.peer_rng.bit_generator.state],
             "depthRng": [self.depth.rng.bit_generator.state,
                          self.depth.peer_rng.bit_generator.state],
             "match": match, "pressing": self._pressing},
            arrays)

  def restore_kept(self, state: dict, arrays: dict) -> bool:
    """`kept_state` put back into a fresh body built where the save stood;
    True where its map came back with it."""
    acts = self._own_acts()
    m = self.model
    if "gains" in arrays and len(arrays["gains"]) == len(acts):
      m.actuator_gainprm[acts] = arrays["gains"]
      m.actuator_biasprm[acts] = arrays["bias"]
      m.actuator_forcerange[acts] = arrays["forceRange"]
    self.legs = state["legs"]
    self.falls = int(state["falls"])
    self._steps = int(state["steps"])
    self.last_motion_t = float(state.get("lastMotion", 0.0))
    o, s = self.odo, state["odometry"]
    o.x, o.y, o.z, o.offset = s["x"], s["y"], s["z"], s["offset"]
    o.foot = s["foot"]
    o.last = value_back(s["last"])
    trunk = self.stand.trunk
    trunk.imu.restore_kept(state["trunk"]["imu"])
    trunk.attitude.q = tuple(float(v) for v in state["trunk"]["q"])
    trunk.rate = value_back(state["trunk"]["rate"])
    self.stand.joints.rng.bit_generator.state = state["jointsRng"]
    for name, drv in (("stand", self.stand), ("walk", self.walker), ("getup", self.getup)):
      kept = state["drivers"].get(name)
      if drv is None or kept is None:
        continue
      drv.steps = int(kept["steps"])
      for k in _DRIVER[name]:
        setattr(drv, k, value_back(kept[k]))
    for name, d in self._all_drivers().items():
      d.torqued = bool(state["torqued"].get(name, d.torqued))
    self.reach.restore_kept(state["reach"])
    frame = state.get("frame")
    if frame is not None:
      self._frame = DepthFrame(z=arrays["frameZ"], points=arrays["framePoints"],
                               self_fraction=float(frame["self"]),
                               peers=arrays["framePeers"],
                               peer_geoms=np.asarray(arrays["framePeerGeoms"], dtype=np.int32))
      self._frame_t, self._frame_used_t = float(frame["t"]), float(frame["used"])
      self._frame_level = bool(frame["level"])
    memo = state.get("memo")
    self._plan_memo = (None if memo is None else
                       (tuple(memo[0]), float(memo[1]), tuple(memo[2]),
                        None if memo[3] is None else [tuple(p) for p in memo[3]],
                        None if memo[4] is None else tuple(memo[4])))
    clocks = state["clocks"]
    self._next_scan = float(clocks["scan"])
    self.backoff_until = float(clocks["backoff"])
    self.peer_seen_m, self.peer_seen_t = state["peerSeen"]
    self.lidar.rng.bit_generator.state = state["lidarRng"][0]
    self.lidar.peer_rng.bit_generator.state = state["lidarRng"][1]
    self.depth.rng.bit_generator.state = state["depthRng"][0]
    self.depth.peer_rng.bit_generator.state = state["depthRng"][1]
    self._pressing = bool(state.get("pressing", False))
    if self.head is not None:
      self.head.forget()
    grid = arrays.get("grid")
    mapped = grid is not None and grid.shape == self.grid.grid.shape
    if mapped:
      self.grid.grid[...] = grid
      self.low[...] = arrays["low"]
      if "walls" in arrays:
        self._walls_mask, self._walls_t = np.array(arrays["walls"], dtype=bool), clocks["walls"]
      if self.matcher is not None and state.get("match"):
        self.matcher.restore_kept(state["match"], arrays)
    return mapped

  def _all_drivers(self) -> dict:
    """Every set of its drivers, by name: the legs' under each policy, each
    arm's."""
    out = {"stand": self.stand.legs, "walk": self.walker.legs}
    if self.getup is not None:
      out["getup"] = self.getup.drivers
    out.update({f"arm:{s}": arm.drivers for s, arm in self.reach.arms.items()})
    return out

  def fresh_drivers(self) -> None:
    """Its drivers and senses built again where the body stands now (a
    start, a stand-up). Its map stays."""
    self._build_drivers(self.stand.policy, self.walker.policy)

  def _front_blocked(self, angles, ranges) -> bool:
    ahead = np.cos(angles) * ranges
    side = np.sin(angles) * ranges
    return bool(((ahead > 0.0) & (ahead < self.FRONT_STOP_RANGE)
                 & (np.abs(side) <= self.FRONT_HALF_M)).any())

  def _backoff_routine(self):
    yield from self._twist_routine(-self.BACKOFF_V, 0.0, 0.0)

  def face_routine(self, heading: float, budget: float = 12.0):
    """Turn where it stands to a heading: a turn on the spot at a rate the
    error sets, then stood still. False if its budget ran out."""
    dt = self.model.opt.timestep
    for _ in range(int(budget / dt)):
      err = math.remainder(heading - self.pose[2], 2 * math.pi)
      if abs(err) < self.FACING_TOLERANCE:
        break
      yield from self._twist_routine(0.0, 0.0, max(-W_MAX, min(W_MAX, 1.5 * err)))
    else:
      return False
    for _ in range(int(0.5 / dt)):
      yield from self._twist_routine(0.0, 0.0, 0.0)
    return True

  # ---- what the depth camera sees under the LIDAR ---------------------------

  def _planning_grid(self, pads: bool = True) -> np.ndarray:
    """The LIDAR's grid, with what the depth camera saw between the floor
    and the LIDAR's plane burned in as obstacles."""
    low = self.low > LOW_OCC
    if not low.any():
      return self.grid.grid
    out = self.grid.grid.copy()
    out[low] = np.maximum(out[low], LOW_BURN)
    return out

  def _look_step(self) -> None:
    """The depth camera, at its rate: what it sees under the LIDAR's plane
    goes into the planner's layer, off a frame taken level and on its feet
    -- and none is taken otherwise but for the loop's near field
    (`latest_depth`): bowed for a reach, it renders nothing."""
    if self.legs == "getup" or not self.level():
      return
    frame = self.latest_depth()
    if frame is None or self._frame_t == self._frame_used_t or not self._frame_level:
      return
    self._frame_used_t = self._frame_t
    self._fold_low(frame.points)

  def latest_depth(self) -> DepthFrame | None:
    """The newest depth frame, levelled, taken now if the last is a period
    old: ONE render feeds the planner's layer, the near-field map and the
    peer channel (`HumanoidBody.depth_camera`, #559), as the quadruped's.
    ⚠ Only a frame taken level goes into the layer (`_frame_level`): one
    the near field asked for bowed is no map's."""
    t = float(self.data.time)
    if t - self._frame_t >= DEPTH_PERIOD - 1e-9:
      self._frame_t = t
      self._frame_level = self.legs != "getup" and self.level()
      raw = self.depth.frame(self.data) if self.head is None else self.head.frame
      self._frame = None if raw is None else self._levelled(raw)
    return self._frame

  def _levelled(self, frame: DepthFrame) -> DepthFrame:
    """A frame from the trunk's own axes into the robot's level frame: its
    IMU's roll and pitch taken out, z over the floor under its feet (the
    pelvis's height off its legs' encoders, its model of itself)."""
    att = self.stand.trunk.attitude
    lv = att.level()
    h = float(self.odo.sm.d.qpos[2])
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
    cells = np.unique(iy[hit] * cols + ix[hit]) if hit.any() else np.zeros(0, dtype=np.int64)
    if floor.any():
      # ⚠ A CELL A FRAME HITS IS NO FLOOR IN IT (#559): a thin thing fills a
      # part of its 5 cm cell and the floor round it the rest, and each of
      # the floor's points voted it down -- the garden light's 4 cm pole read
      # -0.4 to -1.0 from 1 m off, under the LIDAR's plane and never in the
      # layer, and a walk into it stood there all day. (The quadruped's
      # LIDAR, 0.51 m up, maps the pole.)
      fl = iy[floor] * cols + ix[floor]
      fl = fl[~np.isin(fl, cells)]
      np.subtract.at(self.low.reshape(-1), fl, LOW_MISS)
    if len(cells):
      cells = cells[~self._walls().flat[cells]]
      self.low.flat[cells] += LOW_HIT
    np.clip(self.low, -LOW_CLAMP, LOW_CLAMP, out=self.low)

  def _walls(self) -> np.ndarray:
    """What the LIDAR maps, grown by `LOW_WALL_CELLS`, over the window the
    camera's points can land in (`LOW_RANGE_M` round the robot), as the
    quadruped's: the camera's layer stays out of it. Recomputed at most once
    a scan. ⚠ Over the whole grid it was 3.6 s of a pair's 155 in 100 sim s
    (#558, #559), and it grows with the map."""
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
