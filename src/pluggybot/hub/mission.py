"""Hub-in-room navigation (milestone 8): drive across room_hub, swap a tool.

The mission stack, reusing every layer that already exists:

  map        look-around spin + scans -> occupancy grid (milestone 4)
  localize   the rack is FOUND, not assumed: its fiducial is watched for
             through every maneuver and confirmed by sighting count
             (hub/localize.py). The stored pose is only the boot-time prior
             -- what a robot that started docked knows -- and observation
             overrides it.
  plan       A* over the inflated grid to the believed hand-off pose
  drive      waypoints + safety reflex (armed in every maneuver, as learned)
  face       pivot to the rack heading, then back-up-and-retry until the
             lateral error is inside what the terminal servo can absorb
  terminal   creep in steering on the bay's tag, with the DISTANCE ranged
             off the rack's tag rather than integrated from odometry (which
             had drifted ~20 mm by the return leg -- twice the coupling's
             capture window)
  swap       HubSwap's verbs (hub/swap.py), timestep dropped to 1 ms

The markers are real tag36h11 AprilTags (hub/tags.py), so every reading is
tied to a decoded ID rather than to whichever blob looked most promising.
Range comes from each marker's own PnP pose -- no depth buffer, matching
what the hardware will actually have.
"""

import math
import time

import mujoco
import numpy as np

from pluggybot.behavior.navigation import (
  BACKOFF_TIME, FRONT_STOP_RANGE, W_SPIN, drive_toward, path_to_waypoints,
)
from pluggybot.control import turn_command, wheel_targets, wrap_angle
from pluggybot.hub.coupling import (
  CHARGE_BAY_Y, HUB_PEG_Z, HUB_STATION_YS, RACK_HANG_X, bay_tag_id,
  module_power_contact,
)
from pluggybot.hub.localize import TAG_LOCAL_X, RackFinder, RackPose
from pluggybot.hub.tags import (
  RACK_TAG_ID, SMALL_TAG_SIZE, TagDetector,
)
from pluggybot.hub.swap import (
  ARM_EXT, CARRY_OFFSET, DROOP_COMP, FORK_MOUNT_RAISE, PICK_OVERSHOOT,
  PLUG_LATERAL, STANDOFF, VERTEX_AHEAD_OF_AXLE, HubSwap,
)
from pluggybot.mapping.astar import astar
from pluggybot.mapping.frontier import traversable_mask
from pluggybot.mapping.occupancy_grid import OccupancyGrid
from pluggybot.perception.lidar import LIDAR_ORIGIN, LIDAR_PERIOD, Lidar

CHARGE_PIN_X = 0.114      # rack-local x of the pogo-pin faces
FACING_TOLERANCE = math.radians(0.5)
SWAP_TIMESTEP = 0.001     # mm-scale peg/V contacts (the spike's floor)
SERVO_PERIOD = 0.25       # s between fiducial looks (the detector cadence)
LOOK_PERIOD = 0.3         # s between rack-tag looks while navigating
SERVO_GAIN = 3.0          # rad/s per meter of lateral error (dock_eye's gain)


def rack_heading(rack: RackPose | None = None) -> float:
  """The heading a robot faces the rack with (into the rack's +x normal)."""
  return (rack or RackPose.prior()).heading


def charge_standoff(rack: RackPose | None = None,
                    distance: float = 0.42) -> tuple[float, float, float]:
  """(axle_x, axle_y, heading) for nosing into the charge bay.

  No plug-lateral offset, unlike a tool bay: charging presses the BUMPER
  against the pogo pins, so it is the chassis centreline that must line up,
  not the fork line. (And that is the whole point of putting charging on
  the bumper -- it works whatever the fork is carrying.)
  """
  r = rack or RackPose.prior()
  bx, by = r.to_world(CHARGE_PIN_X, CHARGE_BAY_Y)
  nx, ny = math.cos(r.yaw), math.sin(r.yaw)
  return bx + nx * distance, by + ny * distance, r.heading


def bay_standoff(station_y: float,
                 rack: RackPose | None = None) -> tuple[float, float, float]:
  """(axle_x, axle_y, heading) of the hand-off pose for one bay.

  Computed from the BELIEVED rack pose; the room's true placement is only
  the default prior (what a robot that booted docked would believe).
  """
  return (rack or RackPose.prior()).bay_standoff(station_y, STANDOFF,
                                                 PLUG_LATERAL)


class TagSpotter:
  """AprilTag reader on the dock camera: identity, not guesswork.

  Both terminal readings now come from decoded IDs, which is the whole
  reason for real tags:
    bay_error   lateral offset to the bay being mated with, selected BY ITS
                ID. The colour stand-in had to guess -- drop the biggest
                blob, take the one nearest the axis -- and it guessed wrong
                once, locking onto the charge bay's marker and dragging a
                module 22 cm toward the wrong bay. Not expressible now.
    rack_range  distance to the rack's marker, from its own PnP pose. No
                depth buffer, which is also what hardware will have.
  """

  def __init__(self, model) -> None:
    self.detector = TagDetector(model, "dock_eye", tag_size=SMALL_TAG_SIZE)

  def detect(self, data) -> dict:
    return self.detector.detect(data)

  def bay_error(self, data, tag_id: int) -> float | None:
    """Lateral offset (m) of one specific bay marker, or None if that tag
    is not in view (near-in occlusion: hold course)."""
    found = self.detector.detect(data)
    if tag_id not in found:
      return None
    return found[tag_id]["t"][0]        # camera-frame x is lateral

  def rack_range(self, data) -> float | None:
    """Perpendicular distance (m) to the rack's tag plane, or None."""
    found = self.detector.detect(data)
    if RACK_TAG_ID not in found:
      return None
    return found[RACK_TAG_ID]["t"][2]

  def close(self) -> None:
    self.detector.close()


class MissionAborted(RuntimeError):
  """The viewer window was closed mid-mission."""


class HubMission:
  """Navigate room_hub to the rack, swap a tool, bring it back.

  Pass a passive mujoco viewer to watch it live: every phase of both the
  mission and the swap bottoms out in HubSwap._step_once, so one hook there
  drives the whole run. Syncing is decimated to ~50 Hz of SIM time (a sync
  per 1 ms physics step would spend all its time in the renderer), and
  realtime paces sim seconds to wall seconds so the motion looks like the
  robot's actual speed.
  """

  VIEW_PERIOD = 0.02       # s of sim time between viewer syncs

  def __init__(self, model, data, viewer=None, realtime: bool = True,
               rack: RackPose | None = None) -> None:
    self.model, self.data = model, data
    self.swap = HubSwap(model, data)
    # What the robot believes about the rack. The prior is what a robot
    # that booted on its dock knows; discover_rack() replaces it with what
    # the robot has actually seen.
    self.rack = rack or RackPose.prior()
    self.finder: RackFinder | None = None
    self.rack_discovered = False
    self._next_look = 0.0
    self.viewer = viewer
    self.realtime = realtime
    self._next_sync = 0.0
    self._wall0 = time.time()
    # Callbacks run on EVERY physics step, whatever phase is driving (both
    # the mission and the swap bottom out in HubSwap._step_once). The
    # battery hooks in here: energy must drain in lockstep with the physics,
    # not per phase, or a long terminal creep is free.
    self.step_hooks: list = []
    self.swap.on_step = self._on_step
    self.lidar = Lidar(model)
    self._next_scan = 0.0
    self.grid = OccupancyGrid(x_min=-3, y_min=-3, x_max=7, y_max=7, resolution=0.05)
    self.tags = TagSpotter(model)
    self.cruise_timestep = model.opt.timestep
    self.backoff_until = 0.0
    self.step_count = 0
    self.collision_steps = 0
    self.chassis_gid = model.geom("chassis").id
    self._cam_id = model.camera("dock_eye").id
    self._charge_pin_gids = {model.geom("rack_pin_l").id,
                             model.geom("rack_pin_r").id}

  def _on_step(self) -> None:
    for hook in self.step_hooks:
      hook()
    if self.viewer is not None:
      self._sync()

  def _sync(self) -> None:
    """Viewer tick, called from every physics step (decimated + paced)."""
    if self.data.time < self._next_sync:
      return
    self._next_sync = self.data.time + self.VIEW_PERIOD
    if not self.viewer.is_running():
      raise MissionAborted("viewer closed")
    self.viewer.sync()
    if self.realtime:
      ahead = self.data.time - (time.time() - self._wall0)
      if ahead > 0:
        time.sleep(min(ahead, 0.05))

  @property
  def pose(self) -> tuple[float, float, float]:
    r = self.swap.reckoner
    return r.x, r.y, r.theta

  def start_at(self, x: float, y: float, yaw: float) -> None:
    """Place the robot and initialize odometry from the known start pose."""
    d = self.data
    d.qpos[0] = x + 0.08 * math.cos(yaw)
    d.qpos[1] = y + 0.08 * math.sin(yaw)
    d.qpos[2] = 0.045
    d.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    # swap's lift preset. Imported, NOT re-typed: these were duplicated as
    # bare 0.016/0.008 here, so raising the fork mount for the lean-pad would
    # have silently left this copy pointing at the old geometry.
    lift0 = HUB_PEG_Z - 0.145 - FORK_MOUNT_RAISE + DROOP_COMP
    d.qpos[self.model.joint("lift_joint").qposadr[0]] = lift0
    d.ctrl[self.model.actuator("lift").id] = lift0
    d.ctrl[self.model.actuator("arm").id] = 0.0     # stowed for driving
    d.qpos[self.model.joint("arm_joint").qposadr[0]] = 0.0
    mujoco.mj_forward(self.model, d)
    r = self.swap.reckoner
    r.x, r.y, r.theta = x, y, yaw
    r.update(float(d.qpos[self.swap.left_adr]), float(d.qpos[self.swap.right_adr]))
    self._drive(1.0, 0.0, 0.0)

  # ---- navigation plumbing -------------------------------------------------

  def _drive(self, seconds: float, v: float, w: float) -> None:
    tl, tr = wheel_targets(v, w)
    for _ in range(round(seconds / self.model.opt.timestep)):
      self._nav_step(tl, tr)

  def _nav_step(self, tl: float, tr: float) -> None:
    """One physics step with scanning, tag-looking + collision bookkeeping."""
    self.swap._step_once(tl, tr)
    self.step_count += 1
    # Look for the rack tag on a cadence through EVERY maneuver, not just
    # during the opening spin. Measured why: from the demo's start pose the
    # floor-box occludes the rack tag entirely (the sight line runs below
    # its top edge), so a single look-around sees nothing and concludes the
    # rack does not exist. Line of sight is a property of where you are
    # standing -- discovery has to ride along with the driving, exactly as
    # outlet landmarks did during exploration.
    if self.finder is not None and self.data.time >= self._next_look:
      self._next_look = self.data.time + LOOK_PERIOD
      self.finder.look(self.data, self.pose)
    # Scan on a TIME cadence at the part's real rate. The camera scanner ran
    # every 20 physics steps (50 Hz) because a depth render is free in sim; a
    # spinning mirror is not, and pretending otherwise would let the mapper
    # rely on data the hardware cannot deliver.
    if self.data.time >= self._next_scan:
      self._next_scan = self.data.time + LIDAR_PERIOD
      angles, ranges = self.lidar.scan(self.data)
      self.grid.update(self.pose, angles, ranges, self.lidar.max_range,
                       origin=LIDAR_ORIGIN)
      if self.data.time >= self.backoff_until:
        front = ranges[np.abs(angles) < 0.35]
        # front can be EMPTY: those bearings may all be self-occluded (the
        # arm crosses the scan plane at some lift heights). No reading is not
        # a clear path -- hold course rather than inventing one.
        if front.size and front.min() < FRONT_STOP_RANGE:
          self.backoff_until = self.data.time + BACKOFF_TIME
    for i in range(self.data.ncon):
      c = self.data.contact[i]
      if self.chassis_gid in (c.geom1, c.geom2):
        # The charge pins touch the bumper BY DESIGN -- counting that as a
        # collision made a successful charge look like 24 750 steps of
        # wall-grinding. Intended contact is not a collision.
        other = c.geom1 if c.geom2 == self.chassis_gid else c.geom2
        if other in self._charge_pin_gids:
          continue
        self.collision_steps += 1
        break

  def _spin(self) -> None:
    """A 360 look-around: seeds the map (and tag sightings, via _nav_step)."""
    remaining = 2 * math.pi
    tl, tr = wheel_targets(0.0, W_SPIN)
    while remaining > 0:
      self._nav_step(tl, tr)
      remaining -= W_SPIN * self.model.opt.timestep

  def start_discovery(self) -> None:
    """Begin watching for the rack tag (every maneuver from here on)."""
    if self.finder is None:
      self.finder = RackFinder(self.model)

  def refresh_rack(self) -> RackPose | None:
    """Adopt the discovered rack pose if the tag has been confirmed.

    Returns what was adopted, or None while unconfirmed -- in which case the
    believed pose stays as it was. A robot that cannot see its dock falls
    back on remembering where it was, which is exactly what the prior is
    for; seeing it is what corrects the memory's drift.
    """
    if self.finder is None:
      return None
    found = self.finder.estimate(self.grid)
    if found is not None:
      self.rack = found
      self.rack_discovered = True
    return found

  def _plan_to(self, wx: float, wy: float) -> list[tuple[float, float]] | None:
    trav = traversable_mask(self.grid.grid)
    rows, cols = trav.shape
    rix, riy = self.grid.world_to_cell(self.pose[0], self.pose[1])
    start = (min(max(rix, 0), cols - 1), min(max(riy, 0), rows - 1))
    if not trav[start[1], start[0]]:
      best, best_d = None, 1e9
      for dy in range(-10, 11):
        for dx in range(-10, 11):
          x2, y2 = start[0] + dx, start[1] + dy
          if 0 <= x2 < cols and 0 <= y2 < rows and trav[y2, x2]:
            if dx * dx + dy * dy < best_d:
              best, best_d = (x2, y2), dx * dx + dy * dy
      if best is None:
        return None
      start = best
    goal = self.grid.world_to_cell(wx, wy)
    goal = (min(max(goal[0], 0), cols - 1), min(max(goal[1], 0), rows - 1))
    if not trav[goal[1], goal[0]]:
      # The goal sits in unknown or inflated space. Plan to the KNOWN-FREE
      # cell nearest it instead: driving there grows the map toward the
      # goal, and the next replan gets closer. (A blind greedy advance was
      # tried first and measured awful: it rammed the floor-box's reflex
      # zone forever, ignoring the very map it was building.)
      ys, xs = np.nonzero(trav)
      if len(xs) == 0:
        return None
      d2 = (xs - goal[0]) ** 2 + (ys - goal[1]) ** 2
      k = int(np.argmin(d2))
      goal = (int(xs[k]), int(ys[k]))
    path = astar(trav, start, goal)
    return None if path is None else path_to_waypoints(self.grid, path)

  def drive_to(self, wx: float, wy: float, timeout: float = 90.0) -> bool:
    """A*-navigate to a world point, arriving within 8 cm. Plans through
    known space only, targeting the reachable cell nearest the goal until
    the goal itself becomes reachable. Gives up on stagnation (no progress
    toward the goal for 10 sim-seconds)."""
    waypoints: list[tuple[float, float]] = []
    next_replan = 0.0
    t0 = self.data.time
    best_dist = math.hypot(wx - self.pose[0], wy - self.pose[1])
    last_improve = t0
    while self.data.time - t0 < timeout:
      dist = math.hypot(wx - self.pose[0], wy - self.pose[1])
      if dist < 0.08 and not waypoints:
        return True
      if dist < best_dist - 0.02:
        best_dist, last_improve = dist, self.data.time
      elif self.data.time - last_improve > 10.0:
        return dist < 0.15               # stagnated: close enough or fail
      if self.data.time < self.backoff_until:
        self._drive(self.model.opt.timestep, -0.15, 0.0)
        waypoints = []
        continue
      if self.data.time >= next_replan or not waypoints:
        next_replan = self.data.time + 2.0
        planned = self._plan_to(wx, wy)
        if planned is None:
          return dist < 0.15             # no known space at all
        waypoints = planned
      while waypoints and math.hypot(waypoints[0][0] - self.pose[0],
                                     waypoints[0][1] - self.pose[1]) < 0.08:
        waypoints.pop(0)
      if waypoints:
        v, w = drive_toward(self.pose, waypoints[0])
      else:
        v, w = drive_toward(self.pose, (wx, wy))
      self._drive(self.model.opt.timestep, v, w)
    return False

  def face(self, heading: float) -> None:
    while abs(wrap_angle(heading - self.pose[2])) > FACING_TOLERANCE:
      err = wrap_angle(heading - self.pose[2])
      self._drive(self.model.opt.timestep, 0.0,
                  turn_command(err, gain=2.5, limit=1.0))
    self._drive(0.5, 0.0, 0.0)

  # ---- the mission ---------------------------------------------------------

  def steer_fn(self, tag_id: int):
    """Terminal-servo callback for HubSwap: steer on ONE named bay marker,
    re-looking every SERVO_PERIOD. Passing the id is the point -- the
    servo can no longer be captured by whatever fiducial happens to look
    most appealing in the frame."""
    state = {"next": 0.0, "w": 0.0}

    def fn() -> float:
      if self.data.time >= state["next"]:
        state["next"] = self.data.time + SERVO_PERIOD
        e = self.tags.bay_error(self.data, tag_id)
        state["w"] = 0.0 if e is None else max(-0.3, min(0.3, -SERVO_GAIN * e))
      return state["w"]
    return fn

  def refine_standoff(self, sx: float, sy: float, hd: float) -> float:
    """Kill the lateral arrival error before the terminal creep.

    drive_to's 8 cm arrival radius happily 'arrives' 7 cm off the bay line
    (measured: the fork went in that far off-centre and knocked the module
    to the floor), and the tag servo's steering authority over a 0.2 m
    creep is only ~1-2 cm. The fix is the oldest one in driving: back up
    and take another run at it -- drive_toward's P-controller converges
    laterally given a runway. Returns the final believed lateral error.
    """
    for _ in range(3):
      dx, dy = sx - self.pose[0], sy - self.pose[1]
      lat = -dx * math.sin(hd) + dy * math.cos(hd)
      if abs(lat) < 0.015:
        break
      self._drive(2.5, -0.15, 0.0)              # back off ~0.35 m
      while math.hypot(sx - self.pose[0], sy - self.pose[1]) > 0.05:
        v, w = drive_toward(self.pose, (sx, sy))
        self._drive(self.model.opt.timestep, v, w)
      self.face(hd)
    dx, dy = sx - self.pose[0], sy - self.pose[1]
    return -dx * math.sin(hd) + dy * math.cos(hd)

  def set_arm(self, extension: float, settle: float = 1.5) -> None:
    """Deploy or stow the fork. Stowed is the DRIVING configuration: an
    extended fork rides at module height and sweeps a whole rack clean --
    measured, after a successful stow, by the robot driving along the rack
    to the charge bay and knocking the module it had just put away off its
    trays. Tuck the arm before you drive."""
    self.data.ctrl[self.model.actuator("arm").id] = extension
    self._drive(settle, 0.0, 0.0)

  def _terminal_travel(self, station_y: float) -> float:
    """How far to creep so the fork vertex lands on the bay's hang line.

    Preferred source is the dock camera's range to the RACK TAG -- absolute,
    and immune to the odometry drift that accumulates over a mission (dead
    reckoning was measured ~20 mm out on the return leg, twice the
    coupling's capture window). Falls back to the believed rack pose plus
    odometry when the tag is not in view.
    """
    seen = self.tags.rack_range(self.data)
    if seen is not None:
      # Camera -> hang plane, then camera -> fork vertex, both along the
      # approach axis; the difference is what the base still has to drive.
      cam = self.data.cam_xpos[self._cam_id]
      vtx = self.data.site_xpos[self.model.site("fork_vertex").id]
      h = self.pose[2]
      vertex_ahead = ((float(vtx[0]) - float(cam[0])) * math.cos(h)
                      + (float(vtx[1]) - float(cam[1])) * math.sin(h))
      to_hang = seen - (TAG_LOCAL_X - RACK_HANG_X)
      return to_hang - vertex_ahead
    hx, hy = self.rack.to_world(RACK_HANG_X, station_y)
    d_along = ((hx - self.pose[0]) * math.cos(self.pose[2])
               + (hy - self.pose[1]) * math.sin(self.pose[2]))
    return d_along - VERTEX_AHEAD_OF_AXLE

  def swap_at_bay(self, station_y: float, verb: str,
                  module: str | None = None, tries: int = 2) -> str:
    """Navigate to a bay's hand-off pose and pick or return there.

    module + tries: VERIFY the outcome and take another run at it. The
    terminal travel is ranged off the rack tag, and that range scatters
    several mm run to run (measured against truth: -5 to +11 mm on
    otherwise identical missions) against a coupling capture window of
    about +/-11 mm -- so a single blind attempt is a dice roll that any
    mm-level physics change re-rolls (the issue-3 spike watched the same
    pick pass and fail across configurations that differ by less than a
    millimetre of wheel slip). The verdict is the criterion that already
    exists -- ELECTRICAL seating for a pick, hung-in-bay for a return --
    and the retry re-ranges the tag from the retreat pose, so the second
    attempt is a fresh measurement, not a repeat of the first one's error.
    Same doctrine as refine_standoff, one verb up: back up and take
    another run at it.
    """
    # Adopt whatever the tag has shown us so far, then aim at that. Driving
    # to the neighborhood is itself what buys line of sight, so the belief
    # is refreshed once more after arriving.
    self.refresh_rack()
    sx, sy, hd = bay_standoff(station_y, self.rack)
    # A route failure this early usually means the planner ran out of KNOWN
    # cells, not that the bay is unreachable -- the opening look-around's
    # coverage is pose-dependent and marginal from some starts. The
    # milestone-4 doctrine applies verbatim: when nothing is reachable, spin
    # to buy map (and possibly the rack tag) and try again.
    for _ in range(2):
      if self.drive_to(sx, sy):
        break
      self._spin()
      self.refresh_rack()
      sx, sy, hd = bay_standoff(station_y, self.rack)
    else:
      return "no-route"
    if self.refresh_rack() is not None:
      sx, sy, hd = bay_standoff(station_y, self.rack)
      self.drive_to(sx, sy, timeout=25.0)
    why = "no-attempt"
    for _ in range(max(tries, 1)):
      self.face(hd)
      self.refine_standoff(sx, sy, hd)
      self.set_arm(ARM_EXT)               # deploy only once lined up
      # Travel computed from the BELIEVED distance to the hang plane --
      # fixed travels assume a perfect standoff, and arrival is only good
      # to cm. Signs: picking OVERSHOOTS the peg line (the peg rides up the
      # near flank and the lift centres it); returning STOPS SHORT by the
      # carry offset, because the peg rides that far ahead of the vertex
      # and it is the PEG that must land over the tray line.
      travel = self._terminal_travel(station_y)
      tag_id = bay_tag_id(station_y)
      self.model.opt.timestep = SWAP_TIMESTEP
      try:
        if verb == "pick":
          why = self.swap.pick(steer_fn=self.steer_fn(tag_id),
                               dist=travel + PICK_OVERSHOOT)
        else:
          why = self.swap.put_back(steer_fn=self.steer_fn(tag_id),
                                   dist=travel - CARRY_OFFSET)
      finally:
        self.model.opt.timestep = self.cruise_timestep
      self.set_arm(0.0)                   # tuck it back before driving off
      if module is None:
        break
      st = self.swap.module_state(module)
      if verb == "pick":
        ok = st["on_fork"] and module_power_contact(self.model, self.data,
                                                    module)
      else:
        ok = st["hung"]
      if ok:
        break
    return why

  def close(self) -> None:
    self.tags.close()
    if self.finder is not None:
      self.finder.close()
      self.finder = None


def run_demo(start=(0.5, 3.0, math.pi / 2), station_y=HUB_STATION_YS[0],
             carry_to=(-1.2, 2.5), view: bool = False,
             realtime: bool = True, discover: bool = True) -> dict:
  """The full story: navigate, pick the LCD, carry it away, bring it back.

  view=True opens the passive MuJoCo viewer and paces the run to real time.
  discover=True finds the rack by its tag during the opening look-around
  and navigates to what it SAW; False keeps the boot-time prior.
  """
  model = mujoco.MjModel.from_xml_path("models/room_hub.xml")
  data = mujoco.MjData(model)
  viewer = None
  if view:
    # `from mujoco import viewer as ...`, NOT `import mujoco.viewer`: the
    # latter binds the name `mujoco` as a function-local, shadowing the
    # module-level import for the WHOLE function -- including the lines
    # above it (UnboundLocalError on the model load).
    from mujoco import viewer as mj_viewer
    viewer = mj_viewer.launch_passive(model, data)
  mission = HubMission(model, data, viewer=viewer, realtime=realtime)
  aborted = False
  found = None
  try:
    mission.start_at(*start)
    if discover:
      mission.start_discovery()              # watch for the tag from here on
    mission._spin()                          # seed the map

    mission.swap_at_bay(station_y, "pick", module="module_lcd")
    found = mission.rack if mission.rack_discovered else None
    picked = mission.swap.module_state("module_lcd")

    mission.drive_to(*carry_to)              # take the tool somewhere

    mission.swap_at_bay(station_y, "return", module="module_lcd")
    returned = mission.swap.module_state("module_lcd")
  except MissionAborted:
    aborted = True
    picked = returned = {"on_fork": False, "hung": False}
  finally:
    mission.close()
    if viewer is not None:
      viewer.close()
  pos_err, yaw_err = ((None, None) if found is None
                      else found.error_against(RackPose.prior()))
  return {
    "picked": picked["on_fork"],
    "returned": returned["hung"],
    "collision_steps": mission.collision_steps,
    "sim_time": float(data.time),
    "aborted": aborted,
    "discovered": found is not None,
    "rack_pos_err_m": pos_err,
    "rack_yaw_err_deg": None if yaw_err is None else math.degrees(yaw_err),
  }
