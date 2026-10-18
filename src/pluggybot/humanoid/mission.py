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
from pluggybot.humanoid.arms import SelfModel
from pluggybot.humanoid.getup import GETUP_NPZ, FallWatch, GetupDriver
from pluggybot.humanoid.odometry import StanceOdometry
from pluggybot.humanoid.policy import StandDriver, StandPolicy
from pluggybot.humanoid.reach import Reach
from pluggybot.humanoid.scan import HeadScan
from pluggybot.humanoid.walk import WALK_NPZ, WalkDriver
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


class HumanoidMission(Navigator):
  """The Navigator over a humanoid (module docstring)."""

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
    self.depth = DepthCamera(model, handle, mount="body", reach=UNCULLED)
    self._frame: DepthFrame | None = None
    self._frame_t = -math.inf
    self._frame_used_t = -math.inf
    self.low = np.zeros_like(self.grid.grid, dtype=np.float32)
    self._walls_t, self._walls_mask = None, None
    self._pressing = False

  # ---- what only a body says ------------------------------------------------

  @property
  def pose(self):
    return self.odo.x, self.odo.y, self.odo.yaw

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
    the legs went to the standing policy whatever had happened in it."""
    yield from self._getup_routine()
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
    t = float(self.data.time)
    if t - self._frame_t < DEPTH_PERIOD - 1e-9:
      return
    self._frame_t = t
    if not self.level():
      return
    raw = self.depth.frame(self.data) if self.head is None else self.head.frame
    if raw is None:
      return
    self._frame = self._levelled(raw)
    self._fold_low(self._frame.points)

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
    if floor.any():
      np.subtract.at(self.low, (iy[floor], ix[floor]), LOW_MISS)
    if hit.any():
      cells = np.unique(iy[hit] * cols + ix[hit])
      cells = cells[~self._walls().flat[cells]]
      self.low.flat[cells] += LOW_HIT
    np.clip(self.low, -LOW_CLAMP, LOW_CLAMP, out=self.low)

  def _walls(self) -> np.ndarray:
    if self._walls_t != self._next_scan:
      from scipy import ndimage
      g = self.grid
      occ = g.grid > OCC_THRESH
      self._walls_mask = (ndimage.binary_dilation(occ, iterations=LOW_WALL_CELLS)
                          if occ.any() else np.zeros_like(occ))
      self._walls_t = self._next_scan
    return self._walls_mask
