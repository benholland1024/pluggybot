"""Getting about: the half of a robot's navigation no body owns (issue #387).

A robot's map of the floor, the LIDAR that builds it, the scan matcher that
keeps it true (issue #386), the planner, the drive to a goal round the
other robots and the words for why a drive gave up -- everything
`HubMission` did that is not wheels, a lift or a fork. The rover
(`mission/mission.py`) and the quadruped (`legs/body.py`) are each a
`Navigator`, and each supplies what only it can say:

  pose, _set_pose   where it believes it is, and a correction to that
  _nav_routine      one physics step at a forward speed and a yaw rate --
                    the body's own command, yielded (the tick contract)
  level, true_pose  the IMU's gate on the map, and the sim's own check
  pressing          pressed against something it is moving into
  _look_step, _contact_step   its own senses either side of the scan
  run               the blocking twin of `yield from`

...and its own MEASURED sizes as class attributes (the planner's
inflation, the disc round another robot, the front stop, the corridor the
peer channel watches). The defaults are the rover's, measured on it and
documented at the constants below; a body overrides them with its own.

Moved out of `HubMission` whole, and the rover's day is the day it was
(`scripts/determinism_spike.py --compare`).
"""

import math
import time

import numpy as np
from scipy import ndimage

from pluggybot.behavior.navigation import (
  BACKOFF_TIME, FRONT_STOP_RANGE, W_SPIN, drive_toward, path_to_waypoints,
)
from pluggybot.body import KeepClear
from pluggybot.control import square_up_routine, wrap_angle
from pluggybot.mapping import optimistic
from pluggybot.mapping.astar import astar, nearest_traversable
from pluggybot.mapping.frontier import FREE_THRESH, OCC_THRESH, traversable_mask
from pluggybot.mapping.occupancy_grid import OccupancyGrid
from pluggybot.mapping.scan_match import ScanMatcher
from pluggybot.perception.lidar import LIDAR_ORIGIN, LIDAR_PERIOD, Lidar, robot_geoms
from pluggybot.robot import FIRST, RobotHandle
from pluggybot.tick import MissionAborted, Routine

def gave_up(rec: dict, peer: str = "the other robot") -> str:
  """Why a drive gave up (`HubMission.last_drive`), as the clause every
  failure line that follows one ends with (issue #350). `peer` names the
  robot a `peer` cause was about -- the lifecycle knows names, the mission
  only poses."""
  why = rec["why"]
  if why == "no_route":
    cause = "no route over the floor mapped so far"
  elif why == "stalled":
    cause = f"stalled, no progress for {STAGNATION_S:.0f} s"
  elif why == DRIVE_STOPPED:
    cause = "stopped by the robot's own interrupt"
  elif why == "peer":
    far, lies = rec.get("peerM"), rec.get("peerDown")
    if far is None:
      cause = f"{peer} in the way"
    elif rec.get("peerAt") == "goal":
      cause = (f"{peer} {'lying' if lies else 'standing'} {far:.1f} m from "
               "where it was going")
    else:
      cause = f"{peer}{' lying' if lies else ''} in the way, {far:.1f} m off"
  else:
    cause = "out of time"
  return (f"the drive gave up {rec['shortM']:.1f} m short after "
          f"{rec['seconds']:.0f} s ({cause})")


#: How far round another robot's believed centre A* keeps this robot's
#: centre (issue #167): the map's own inflation (traversable_mask, 7 cells =
#: 0.35 m, the armed robot's swing) plus the other robot's half-diagonal
#: (0.15 m bare, 0.27 m armed) -- two armed robots passing at 0.62 m.
OTHER_ROBOT_CELLS = 12
#: ...and round a robot LYING ON THE FLOOR (issue #365), whose disc is taken
#: round the middle of its body (`footprint_centre`), not round anything it
#: reports. From there it reaches further than a standing robot does from
#: its own centre: MEASURED 0.32-0.33 m on its side, its front or its back
#: (its geoms' bounding circles; the mast lies along the floor), against a
#: standing robot's 0.27 m armed -- plus the same 0.35 m swing, 0.68 m.
DOWN_ROBOT_CELLS = 14
#: A stagnated drive with another robot this close to us or to the goal is
#: a robot in the way, and the drive WAITS this long before looking again --
#: unless it is lying down, which it will not stop doing for being waited
#: for (issue #365).
OTHER_NEAR_M = 1.2
OTHER_WAIT_S = 2.0
#: How often a drive looks at which other robots are lying down (issue
#: #365). A fall or a stand-up moves that robot's disc, and the plan is
#: made again at once rather than at the next 2 s replan: the depth camera
#: does not hold for a robot on the floor, so nothing else covers that
#: window. MEASURED without it, a robot knocked flat 0.5 m ahead just after
#: a replan was met by the LIDAR's 0.25 m stop, the driver's axle 0.18 m
#: from its body; with it the replan comes 0.04-0.1 s after the fall.
DOWN_CHECK_S = 0.1
#: How close to the goal a stagnated drive counts as having arrived after
#: all -- the tolerance the bay approach then measures its way out of.
CLOSE_ENOUGH_M = 0.15
#: A drive with no progress toward its goal for this long has stagnated.
STAGNATION_S = 10.0
#: WHY A DRIVE GAVE UP (issue #350), `HubMission.last_drive["why"]`, one of
#: four: the planner could not route to the goal over the floor mapped so
#: far (no plan at all, or only to a stand-in the robot then stood at); no
#: progress for `STAGNATION_S` toward a goal the plan did reach; another
#: robot in the way or on the goal when it stopped; the time budget ran
#: out. The robot reads the cause back: "stopped 9.1 m short", with none,
#: was read as the pack running short, and both robots declined the lab.
DRIVE_GAVE_UP = ("no_route", "stalled", "peer", "timeout")
#: ...and the one way a drive ends that is not giving up (issue #381): its
#: caller's `stop` said so -- the robot's own hazard row, mid-walk. An abort,
#: never an error, and never said as one.
DRIVE_STOPPED = "interrupted"
#: How often a drive asks its `stop`, sim s: a walk is a safe point at every
#: step, and this is the latency of an interrupt against the cost of asking.
STOP_EVERY_S = 1.0
#: A stand-in plan's end this close means the drive reached all the floor
#: it could plan over -- a stagnation there is "no route", not a stall.
STAND_IN_REACHED_M = 0.5
#: THE MAP STILL GROWING IS PROGRESS (issue #381, a walk into the unknown):
#: this many more cells known -- free or wall, 1 m^2 at 5 cm -- since the
#: last progress restarts the stagnation clock, however the route's length
#: moved. Read every `MAP_LOOK_S`. MEASURED: see SimNotes, "Walking into
#: the unknown".
MAP_GROWTH_CELLS = 400
MAP_LOOK_S = 1.0
#: ANOTHER ROBOT'S BODY IN THE WAY (issue #328), off the near-field depth
#: camera's peer channel (`DepthFrame.peers`): a peer point nearer than
#: this, inside the corridor this robot is about to drive through, holds
#: the drive still until it clears.
#:
#: The numbers are MEASURED, not chosen. The corridor is the ROBOT: its
#: widest parts are the tyres at 0.150 m either side of centre and the fork
#: prongs at 0.143, so 0.20 leaves 50 mm and no more. The range is what the
#: camera can actually deliver: a peer's nearest point sits at about
#: (centre gap - 0.12) in this frame, so 0.60 m fires at a ~0.72 m gap
#: where the frame carries 300-600 points of it -- and it has to fire up
#: there, because below a ~0.4 m gap the peer's near face falls inside
#: `depth.MIN_Z` (0.28 m) and the camera stops seeing the thing it is
#: about to hit. The LIDAR's 0.25 m front stop is the floor under all of
#: this and is unchanged; it sees the mast alone (issue #316).
PEER_STOP_AHEAD_M = 0.60
PEER_STOP_HALF_M = 0.20
#: How long a sighting stays worth acting on: three frames at
#: `depth.PERIOD`. A hold that outlived the sighting would be a robot
#: standing still because something USED to be there, which is the mistake
#: the map makes and the reason the peer channel exists at all.
PEER_HOLD_S = 0.3
#: ⚠ AND THE HOLD IS AGAINST THE TRAVEL LEFT, NOT AGAINST THE CAMERA
#: (issue #328). A robot with 0.1 m still to drive cannot reach a peer
#: 0.5 m ahead, and holding for one is how a robot parked BESIDE the
#: charge bay stopped the other robot charging at all: measured, a peer
#: 0.50-0.56 m from the charge standoff took the approach from 96 s and a
#: dock to 201 s and none, because arriving at the standoff turns the
#: robot to face the rack and sweeps a body it will never travel into
#: through the corridor. So the drive holds only for a peer nearer than
#: what is left of the drive plus this: the robot's front face, which
#: rides 0.20 m ahead of the axle the points are measured from, and
#: 0.10 m of clearance behind it.
PEER_CLEARANCE_M = 0.30
#: A drive's LAST leg -- its final waypoint and the goal itself -- is a
#: terminal approach (`drive_toward(slow_radius=)`, navigation.py's rule),
#: tapering over this many metres. The path waypoints before it keep the
#: sweeping law. MEASURED (issue #277): a stow begun 13 cm from the bay's
#: standoff -- a procedure that fetches a tool, tries it where it stands
#: and stows it -- handed the plain law a goal closer than its overshoot
#: and orbited it, 690 deg of turning in 16 s; whether the stagnation cut
#: then landed inside `drive_to`'s 15 cm "close enough" was a coin the map
#: tossed, and the built-tool rail beside bay E turned it over. The
#: dispenser's own hops use the same 0.25 (`tools/dispenser.py`).
ARRIVAL_SLOW_RADIUS = 0.25
FACING_TOLERANCE = math.radians(0.5)


class Navigator:
  """The map, the planner and the drive, over a body that says how it moves
  and where it believes it is (the module docstring). Not a body itself:
  `mission.HubMission` and `legs.body.QuadMission` are, each wrapped as a
  `Body` (`mission/rover.py`, `legs/body.py`)."""

  VIEW_PERIOD = 0.02       # s of sim time between viewer syncs
  #: The LIDAR's place on the body, from the pose point (x ahead, y left).
  LIDAR_ORIGIN = LIDAR_ORIGIN
  #: The front stop: a scan return this near, dead ahead, backs the body off.
  FRONT_STOP_RANGE = FRONT_STOP_RANGE
  #: ...at this speed, m/s, for `BACKOFF_TIME`.
  BACKOFF_V = 0.15
  #: The map's obstacles grow by this many cells before the planner treats
  #: the body as a point (`traversable_mask`): its outermost geometry.
  INFLATION_CELLS = 7
  OTHER_ROBOT_CELLS = OTHER_ROBOT_CELLS
  DOWN_ROBOT_CELLS = DOWN_ROBOT_CELLS
  PEER_STOP_AHEAD_M = PEER_STOP_AHEAD_M
  PEER_STOP_HALF_M = PEER_STOP_HALF_M
  PEER_CLEARANCE_M = PEER_CLEARANCE_M
  FACING_TOLERANCE = FACING_TOLERANCE
  #: How a drive's progress is read (`STAGNATION_S`): in a straight line to
  #: the goal, the rover's, or ALONG THE ROUTE it plans (issue #387): from
  #: the bedroom's corner the only way to the hall door walks 2.5 m round
  #: the divider first, away from the goal, and read in a straight line
  #: every such drive "stalled" at 10 s -- MEASURED, 21 in a row.
  PROGRESS_ALONG_ROUTE = False
  #: ...and a replan whose route is this much longer than the best seen is
  #: a NEW route, whose progress is read from where it starts, m.
  NEW_ROUTE_M = 0.5
  #: PLANS THROUGH FLOOR IT HAS NOT SEEN (issue #381, `mapping/optimistic.py`),
  #: at `UNKNOWN_COST` a metre, or -- the rover's -- through mapped floor only,
  #: aiming at a stand-in for anything off it.
  OPTIMISTIC = False
  UNKNOWN_COST = optimistic.UNKNOWN_COST
  #: ...and counts the map still growing as progress (`MAP_GROWTH_CELLS`).
  PROGRESS_MAP_GROWTH = False

  def __init__(self, model, data, handle: RobotHandle = FIRST,
               grid_bounds: tuple[float, float, float, float] = (-3, -3, 7, 7),
               viewer=None, realtime: bool = True, match: bool = True) -> None:
    self.model, self.data = model, data
    #: WHICH ROBOT (issue #167): every element resolves through it.
    self.handle = handle
    self.viewer = viewer
    self.realtime = realtime
    self._next_sync = 0.0
    self._wall0 = time.time()
    # Callbacks run on EVERY physics step, whatever phase is driving. The
    # battery hooks in here: energy must drain in lockstep with the physics,
    # not per phase, or a long terminal creep is free.
    self.step_hooks: list = []
    self.lidar = Lidar(model, site_name=handle.el("lidar"),
                       robot_body=handle.root)
    self._next_scan = 0.0
    # Bounds are per-WORLD (issue #6): room_hub keeps its historical box,
    # home_world passes its own from the generator's meta -- a grid sized
    # for one room silently truncates every scan beyond its edge.
    gx0, gy0, gx1, gy1 = grid_bounds
    self.grid = OccupancyGrid(x_min=gx0, y_min=gy0, x_max=gx1, y_max=gy1,
                              resolution=0.05)
    #: Each scan aligned against the map before it is fused (issue #386);
    #: None flies on odometry alone, as every day did before it.
    self.matcher = (ScanMatcher(self.grid, origin=self.LIDAR_ORIGIN,
                                max_range=self.lidar.max_range)
                    if match else None)
    self.backoff_until = 0.0
    #: How the last `drive_to` ended (issue #350): `why` ("" arrived, else
    #: one of `DRIVE_GAVE_UP`), the goal, the seconds it took, how far short
    #: it stopped, and for a peer which body -- beside the robot (`here`)
    #: or at the goal, how far, where, and whether it lies down. `gave_up`
    #: is its sentence.
    self.last_drive: dict | None = None
    #: Where the last plan aimed INSTEAD of the goal -- the nearest cell of
    #: this robot's own component, when the goal was off it -- or None.
    self._stand_in: tuple[float, float] | None = None
    self._floor = None                        # `_plan_to`'s floor, unmasked
    #: The peer stop (issue #328): the last sighting in the corridor ahead
    #: -- how far off it was and when -- and how many times a drive has
    #: actually HELD for one. Episodes, not frames, and counted where the
    #: hold happens rather than where the sighting does: a robot that sees
    #: the other one while parked has not changed what it was doing, and
    #: "how often did this change anything" is the number that says
    #: whether any of it helps.
    self.peer_seen_m: float | None = None
    self.peer_seen_t = 0.0
    self.peer_holds = 0
    self.step_count = 0
    self.collision_steps = 0
    #: This robot's geoms, root and subtree: what another robot's sensors
    #: attribute to it (issue #365).
    self.body_gids = np.array(sorted(robot_geoms(model, handle.root)),
                              dtype=np.int32)
    #: THE OTHER ROBOTS (issue #167): callables returning each one's believed
    #: (x, y), read at plan time so A* routes round a footprint the lidar
    #: may not have marked yet. What a robot may know of another over the
    #: network is its reported pose -- odometry is a work order's kind of
    #: fact, not a sensor's (TaskPattern.md §2) -- and that is what is read.
    #: A callable may answer a `KeepClear` instead: a robot lying down,
    #: avoided round its body (issue #365, `HubLifecycle.keep_clear`).
    #: `_bodies` reads both.
    self.others: list = []

  def rebind(self, model, data) -> None:
    """A recompiled world (issue #168): the map and the belief are state and
    stay; the LIDAR and this body's geoms are found again by name."""
    self.model, self.data = model, data
    self.lidar.rebind(model)
    self.body_gids = np.array(sorted(robot_geoms(model, self.handle.root)),
                              dtype=np.int32)

  # ---- what the body says --------------------------------------------------

  @property
  def pose(self) -> tuple[float, float, float]:
    """Where the body BELIEVES it is: (x, y, heading)."""
    raise NotImplementedError

  def _set_pose(self, x: float, y: float, theta: float) -> None:
    """Move the belief to a pose from outside the body's own reckoning (a
    scan match)."""
    raise NotImplementedError

  def true_pose(self) -> tuple[float, float, float]:
    """Where it IS, in `pose`'s terms, off the world."""
    raise NotImplementedError

  def level(self) -> bool:
    """Level enough for a scan to be a map of the room (issue #339)."""
    raise NotImplementedError

  @property
  def pressing(self) -> bool:
    """Pressed against something it is moving into."""
    return False

  def _planning_grid(self) -> np.ndarray:
    """The map the planner plans over, as log-odds: the LIDAR's grid here.
    A body that senses below its scan plane adds what it saw there (the
    quadruped's depth camera, `legs.body.QuadMission`)."""
    return self.grid.grid

  def run(self, routine, name: str = ""):
    raise NotImplementedError

  # ---- the seam (issue #58) ------------------------------------------------

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

  def footprint_centre(self) -> tuple[float, float]:
    """The middle of the floor this robot's body covers, off the TRUE
    geometry: its geoms' bounding circles, boxed. MEASURED 0.08 m from the
    chassis origin upright, and 0.20-0.21 m along the mast lying down.
    ⚠ Read to ACT, unlike `true_pose`, and only for a robot lying on the
    floor (issue #365): what another robot keeps clear of, because a real
    one would see a robot-shaped lump there, and the one thing about a
    fallen robot its own odometry cannot say."""
    xy = self.data.geom_xpos[self.body_gids, :2]
    r = self.model.geom_rbound[self.body_gids][:, None]
    lo, hi = (xy - r).min(axis=0), (xy + r).max(axis=0)
    return float(lo[0] + hi[0]) / 2.0, float(lo[1] + hi[1]) / 2.0

  def as_seen(self, wx: float, wy: float) -> tuple[float, float]:
    """A TRUE world point where this robot's own sensors would put it:
    seen from where it truly stands, placed through where it BELIEVES it
    stands -- what the map does with every scan. Its own drift then cancels
    out of anything it plans round (issue #365: the pair's drifts ran
    0.24-0.55 m, against 0.37 m of floor between a detour and a robot lying
    down). Reads `true_pose` to act, as a ray cast does: the geometry is
    the sensor's, the placement is the belief's."""
    tx, ty, tth = self.true_pose()
    bx, by, bth = self.pose
    dx, dy = wx - tx, wy - ty
    c, s = math.cos(bth - tth), math.sin(bth - tth)
    return bx + c * dx - s * dy, by + s * dx + c * dy

  def truth_error(self) -> list[float]:
    """The belief minus the TRUE axle pose, (dx mm, dy mm, dyaw deg): what
    dead reckoning has drifted by, for a failed swap's trace (issue #264)."""
    ax, ay, yaw = self.true_pose()
    bx, by, bth = self.pose
    dyaw = (bth - yaw + math.pi) % (2 * math.pi) - math.pi
    return [round(1000 * (bx - ax), 1), round(1000 * (by - ay), 1),
            round(math.degrees(dyaw), 2)]

  def _drive(self, seconds: float, v: float, w: float) -> None:
    return self.run(self._drive_routine(seconds, v, w))

  def _drive_routine(self, seconds: float, v: float, w: float) -> Routine:
    for _ in range(round(seconds / self.model.opt.timestep)):
      yield from self._nav_routine(v, w)

  def _nav_routine(self, v: float, w: float) -> Routine:
    """`_nav_step` as a routine: the step is the yield, the bookkeeping runs
    when the driver resumes -- after the step, as before."""
    yield v, w
    self._after_step()

  def _after_step(self) -> None:
    self.step_count += 1
    self._look_step()
    self._scan_step()
    self._contact_step()

  def _look_step(self) -> None:
    """What the body looks for through every manoeuvre, before the scan:
    nothing here; the rover watches for its rack's tag."""

  def _scan_step(self) -> None:
    """The LIDAR, at its own rate: the map (level, and matched) and the
    front stop's reflex."""
    # Scan on a TIME cadence at the part's real rate. The camera scanner ran
    # every 20 physics steps (50 Hz) because a depth render is free in sim; a
    # spinning mirror is not, and pretending otherwise would let the mapper
    # rely on data the hardware cannot deliver.
    if self.data.time >= self._next_scan:
      self._next_scan = self.data.time + LIDAR_PERIOD
      # ⚠ ONE SCAN, TWO CONSUMERS WITH OPPOSITE NEEDS (issue #316). The MAP
      # must not contain another robot -- painted in and inflated, a robot
      # driving past walls in the robot it passed (issue #167) -- and the
      # REFLEX must, because the other robot is the only thing in the world
      # that moves. Excluding it from the scan silenced both, and the 0.25 m
      # stop that holds this robot off a wall, a doorpost and a bed was
      # blind to its pair: 9 `stuck` deaths in the seven days that found it.
      angles, ranges, peer_angles, peer_ranges = self.lidar.scan_split(self.data)
      if self.level():
        m = self._match(angles, ranges)
        if m is None or self.matcher.fuses(m, self.data.time):
          self.grid.update(self.pose, angles, ranges, self.lidar.max_range,
                           origin=self.LIDAR_ORIGIN)
          if m is not None:
            self.matcher.fused(self.pose, self.data.time)
      elif self.matcher is not None:
        self.matcher.fuse_next()
      if self.data.time >= self.backoff_until:
        all_angles = np.concatenate((angles, peer_angles))
        all_ranges = np.concatenate((ranges, peer_ranges))
        if self._front_blocked(all_angles, all_ranges):
          self.backoff_until = self.data.time + BACKOFF_TIME

  def _front_blocked(self, angles, ranges) -> bool:
    """The front stop's test, over one scan's bearings and ranges (the room
    and the other robots): a return inside `FRONT_STOP_RANGE` within 0.35
    rad of dead ahead."""
    front = ranges[np.abs(angles) < 0.35]
    # front can be EMPTY: those bearings may all be self-occluded (the
    # arm crosses the scan plane at some lift heights). No reading is not
    # a clear path -- hold course rather than inventing one.
    return bool(front.size and front.min() < self.FRONT_STOP_RANGE)

  def _contact_step(self) -> None:
    """The body's contacts, after the scan: nothing here; the rover
    counts its chassis's collisions."""

  def _match(self, angles, ranges):
    """Align the scan with the map it is about to go into (issue #386) and
    move the belief to where the walls put it: the pose the map is laid
    through, the planner plans from and the other robot is told. The match,
    or None with no matcher."""
    if self.matcher is None:
      return None
    m = self.matcher.match(self.pose, angles, ranges)
    if m.accepted:
      self._set_pose(*m.pose)
    return m

  def _spin(self) -> None:
    return self.run(self._spin_routine())

  def _spin_routine(self) -> Routine:
    """A 360 look-around: seeds the map (and tag sightings, via _nav_step)."""
    remaining = 2 * math.pi
    while remaining > 0:
      yield from self._nav_routine(0.0, W_SPIN)
      remaining -= W_SPIN * self.model.opt.timestep

  # ---- planning and driving ------------------------------------------------

  def _plan_to(self, wx: float, wy: float) -> list[tuple[float, float]] | None:
    """A* to the goal, or to its stand-in (below). The floor before the
    other robots are masked out is kept (`_floor`) for
    `_route_cut_by_others`. `OPTIMISTIC` plans through the unknown too
    (`_plan_optimistic`)."""
    if self.OPTIMISTIC:
      return self._plan_optimistic(wx, wy)
    self._floor = traversable_mask(self._planning_grid(), self.INFLATION_CELLS)
    trav = self._floor.copy()
    self._mask_others(trav)
    self._stand_in = None
    rows, cols = trav.shape
    # The halo escape, shared with `navigation.plan` since issue #92 -- this
    # inline version is where the idea was born, and exploration's planner
    # not having it is what let a sealed-in robot declare the house mapped.
    start = nearest_traversable(
      trav, self.grid.world_to_cell(self.pose[0], self.pose[1]))
    if start is None:
      return None
    goal = self.grid.world_to_cell(wx, wy)
    goal = (min(max(goal[0], 0), cols - 1), min(max(goal[1], 0), rows - 1))
    if not trav[goal[1], goal[0]]:
      # The goal sits in unknown or inflated space. Plan to the nearest
      # cell OF THIS ROBOT'S OWN COMPONENT instead -- the 4-connected
      # component its start cell is in, `astar`'s own neighbourhood:
      # driving there grows the map toward the goal, and the next replan
      # gets closer. (A blind greedy advance was tried first and measured
      # awful: it rammed the floor-box's reflex zone forever, ignoring the
      # very map it was building.)
      # ⚠ OWN COMPONENT, not the nearest free cell anywhere (issue #298):
      # the bedroom seen through its doorway leaves one-cell islands of
      # free space near whiteboard_b, the island nearest the board's use
      # pose was one cell nearer than the reachable wedge beside it,
      # `astar` answered None and the drive gave up in 0 s -- the board
      # paid nobody in 30 deployed hours, and never did in a single-robot
      # flight. ⚠ A TRAVERSABLE goal in another component still plans
      # None, deliberately: that is a frontier behind the other robot, and
      # explore's strike logic counts on the drive stepping nothing
      # (`test_explore_does_not_spin_when_the_other_robot_blocks_...`).
      labels, _ = ndimage.label(trav)
      ys, xs = np.nonzero(labels == labels[start[1], start[0]])
      if len(xs) == 0:
        return None
      d2 = (xs - goal[0]) ** 2 + (ys - goal[1]) ** 2
      k = int(np.argmin(d2))
      goal = (int(xs[k]), int(ys[k]))
      self._stand_in = self.grid.cell_to_world(*goal)
    path = astar(trav, start, goal)
    return None if path is None else path_to_waypoints(self.grid, path)

  def _plan_optimistic(self, wx: float, wy: float) -> list[tuple[float, float]] | None:
    """`_plan_to` through the floor it has not seen as well as the floor it
    has (issue #381): the lattice route of `mapping/optimistic.py`, the
    other robots' discs taken out of it as they are out of A*'s, the
    stand-in for a goal inside a wall's inflation, and no plan at all for
    a goal the walls it has seen shut it off from. `_floor` is the map's
    floor before the discs, as `_plan_to` keeps it."""
    cost = optimistic.map_costs(self._planning_grid(), self.INFLATION_CELLS,
                                self.UNKNOWN_COST)
    self._floor = np.isfinite(cost)
    floor = self._floor.copy()
    self._mask_others(floor)
    cost[~floor] = np.inf
    b = optimistic.BLOCK
    lattice = optimistic.coarsen(cost, b)
    graph = optimistic.lattice_for(lattice.shape, self.grid.resolution * b)
    sx, sy = self.grid.world_to_cell(self.pose[0], self.pose[1])
    gx, gy = self.grid.world_to_cell(wx, wy)
    cells, stand_in = optimistic.route(graph, lattice, (sx // b, sy // b),
                                       (gx // b, gy // b))
    self._stand_in = None
    if cells is None:
      return None
    g, side = self.grid, self.grid.resolution * b
    pts = [(g.x_min + (cx + 0.5) * side, g.y_min + (cy + 0.5) * side)
           for cx, cy in cells]
    if stand_in:
      self._stand_in = pts[-1]
    # every other cell, 0.2 m apart as A*'s every third, and always the end
    out = pts[2::2]
    if not out or out[-1] != pts[-1]:
      out.append(pts[-1])
    return out

  def in_sight(self, wx: float, wy: float) -> bool:
    """Can one drive plan to (wx, wy): is it within the LIDAR's reach and
    on the map, its cell seen (free or not)? Beyond either, `_plan_to`
    aims at a stand-in, and the procedure verb goes by the house's route
    instead (issue #353, `steps._drive_to`)."""
    if math.hypot(wx - self.pose[0], wy - self.pose[1]) > self.lidar.max_range:
      return False
    cx, cy = self.grid.world_to_cell(wx, wy)
    rows, cols = self.grid.grid.shape
    if not (0 <= cx < cols and 0 <= cy < rows):
      return False
    seen = self.grid.grid[cy, cx]
    return bool(seen < FREE_THRESH or seen > OCC_THRESH)

  def _route_cut_by_others(self, wx: float, wy: float) -> bool:
    """Would `_plan_to` have found a plan on the floor it last planned
    over, with no other robot masked out (issue #350)? Its own tests --
    a start cell, and the goal off the floor (a stand-in, which the start's
    component always offers) or in the start's component -- by labelling,
    never a second A*: a plan across the home loop is ~1.3 s of Python, on
    the physics thread, and this is asked of every drive that planned
    nothing beside another robot. `test_failure_words` holds it to a real
    plan."""
    trav, b, escape = self._floor, 1, 10
    if self.OPTIMISTIC:
      # ...on the planner's own lattice (`_plan_optimistic`), whose
      # components are its 4-connected ones
      b, escape = optimistic.BLOCK, optimistic.ESCAPE_CELLS
      trav = np.isfinite(optimistic.coarsen(np.where(trav, 1.0, np.inf), b))
    rows, cols = trav.shape
    sx, sy = self.grid.world_to_cell(self.pose[0], self.pose[1])
    start = nearest_traversable(trav, (sx // b, sy // b), radius=escape)
    if start is None:
      return False
    gx, gy = self.grid.world_to_cell(wx, wy)
    gx, gy = min(max(gx // b, 0), cols - 1), min(max(gy // b, 0), rows - 1)
    if not trav[gy, gx]:
      return True
    labels, _ = ndimage.label(trav)
    return bool(labels[gy, gx] == labels[start[1], start[0]])

  def drive_to(self, wx: float, wy: float, timeout: float = 90.0) -> bool:
    return self.run(self.drive_to_routine(wx, wy, timeout))

  def drive_to_routine(self, wx: float, wy: float, timeout: float = 90.0,
                       stop=None) -> Routine:
    """A*-navigate to a world point, arriving within 8 cm. Plans through
    known space only, targeting the reachable cell nearest the goal until
    the goal itself becomes reachable -- or, `OPTIMISTIC`, through the
    unknown as well. Gives up on stagnation (no progress toward the goal
    for `STAGNATION_S`). Why it gave up is `last_drive`.

    `timeout` is the walk's patience. `stop`, a callable, is asked every
    `STOP_EVERY_S` (issue #381): True ends the drive where it stands, as
    `DRIVE_STOPPED`, and the time it took to answer -- a question to the
    mind, standing still -- counts against neither the patience nor the
    stagnation clock."""
    waypoints: list[tuple[float, float]] = []
    next_replan = 0.0
    holding = False                    # is this drive standing for a peer
    waiting = False                    # ...or waiting on one it stagnated by
    downs, next_downs = (), 0.0        # who lies down, `DOWN_CHECK_S`
    self._stand_in = None
    t0 = self.data.time
    best_dist = math.hypot(wx - self.pose[0], wy - self.pose[1])
    last_improve = t0
    next_stop = t0 + STOP_EVERY_S
    known = self._known_cells() if self.PROGRESS_MAP_GROWTH else 0
    next_look = t0 + MAP_LOOK_S
    while self.data.time - t0 < timeout:
      dist = math.hypot(wx - self.pose[0], wy - self.pose[1])
      if dist < 0.08 and not waypoints:
        return self._drove(wx, wy, t0, "")
      if stop is not None and self.data.time >= next_stop:
        asked = float(self.data.time)
        if stop():
          return self._drove(wx, wy, t0, DRIVE_STOPPED)
        took = float(self.data.time) - asked
        t0, last_improve = t0 + took, last_improve + took
        next_stop = self.data.time + STOP_EVERY_S
        if took > 0.0:
          waypoints = []               # the world moved while it stood
      left = self._left(dist, waypoints, wx, wy)
      if self.PROGRESS_MAP_GROWTH and self.data.time >= next_look:
        next_look = self.data.time + MAP_LOOK_S
        now_known = self._known_cells()
        if now_known >= known + MAP_GROWTH_CELLS:
          known, last_improve = now_known, self.data.time
      if left < best_dist - 0.02:
        best_dist, last_improve = left, self.data.time
        waiting = False
      elif self.data.time - last_improve > STAGNATION_S:
        if self._other_in_the_way(wx, wy):
          # ANOTHER ROBOT IS WHERE THIS ONE NEEDS TO BE (issue #167). A
          # blocked route is a wait, not a failure: stand still, let it
          # move, look again -- bounded by `timeout`, which is the whole
          # of this robot's patience. Giving up here was measured: two
          # robots sent for the same bay, and the first to arrive reported
          # "no route" after 16 s with the other crossing its path.
          yield from self._drive_routine(OTHER_WAIT_S, 0.0, 0.0)
          last_improve = self.data.time
          waypoints = []
          waiting = True
          continue
        # stagnated: close enough, or fail -- and say which failure
        # ⚠ A BODY ON THE GOAL IS THE OTHER ROBOT'S, standing or LYING:
        # one lying down is never waited for (#365), and its disc turns
        # the goal into a stand-in that read as "no route" (#350 review)
        return self._drove(wx, wy, t0, (
          "" if dist < CLOSE_ENOUGH_M
          else "peer" if holding or self.peer_on_the_goal(wx, wy) is not None
          else "no_route" if self._at_stand_in() else "stalled"))
      peer_m = self.peer_sighting()
      if peer_m is not None and peer_m < dist + self.PEER_CLEARANCE_M:
        # ⚠ A ROBOT IS HELD FOR, NOT BACKED AWAY FROM (issue #328), and the
        # branch sits above the backoff for that reason. The wall reflex
        # reverses because a wall will still be there in a second and
        # reversing is what buys the room to plan round it; the other robot
        # is the one obstacle in this world that MOVES, so standing still
        # costs a second and solves it -- and the reverse is blind behind,
        # which is a poor thing to do near the only other thing that drives.
        # ⚠ ...AND ONLY FOR A BODY THIS DRIVE COULD REACH (`PEER_CLEARANCE_M`):
        # holding for one it stops short of is what took a charge from 96 s
        # to never.
        # Bounded by `timeout` like every other wait here, and the drive
        # stagnates honestly if the other robot never moves.
        if not holding:
          self.peer_holds += 1
          holding = True
        yield from self._nav_routine(0.0, 0.0)
        waypoints = []
        continue
      holding = False
      if self.data.time < self.backoff_until:
        yield from self._backoff_routine()
        waypoints = []
        continue
      if self.others and self.data.time >= next_downs:
        next_downs = self.data.time + DOWN_CHECK_S
        now_down = tuple(b.down for b in self._bodies())
        if now_down != downs:          # one fell, or got up: plan round it now
          downs, waypoints = now_down, []
      if self.data.time >= next_replan or not waypoints:
        next_replan = self.data.time + 2.0
        planned = self._plan_to(wx, wy)
        if planned is None:
          # no known space at all -- unless it is the other robot's disc
          # that cut the route, which a plan without it tells apart
          return self._drove(wx, wy, t0, (
            "" if dist < CLOSE_ENOUGH_M
            else "peer" if self.others and self._route_cut_by_others(wx, wy)
            else "no_route"))
        waypoints = planned
        if self.PROGRESS_ALONG_ROUTE:
          route = self._left(dist, waypoints, wx, wy)
          if route > best_dist + self.NEW_ROUTE_M:
            best_dist, last_improve = route, self.data.time
      while waypoints and math.hypot(waypoints[0][0] - self.pose[0],
                                     waypoints[0][1] - self.pose[1]) < 0.08:
        waypoints.pop(0)
      # The last waypoint is the goal's cell and the goal is 8 cm at most
      # beyond it: both are the final approach, and get the law that
      # cannot orbit (ARRIVAL_SLOW_RADIUS); every waypoint before them is
      # swept through.
      if len(waypoints) > 1:
        v, w = drive_toward(self.pose, waypoints[0])
      else:
        v, w = drive_toward(self.pose, waypoints[0] if waypoints else (wx, wy),
                            slow_radius=ARRIVAL_SLOW_RADIUS)
      yield from self._nav_routine(v, w)
      if self.pressing:
        # The BUMPER reflex (issue #94), the lidar reflex's twin for what
        # the scan plane (0.223 m) looks straight over: back off and replan
        # rather than grind. The reckoner already holds its travel through
        # a press, so a robot that keeps meeting the same unseen thing now
        # stagnates honestly above -- where before it "arrived" at a point
        # it never reached, 4 m of imaginary travel later.
        self.backoff_until = self.data.time + BACKOFF_TIME
    return self._drove(wx, wy, t0, (
      "peer" if waiting or holding or self.peer_on_the_goal(wx, wy) is not None
      else "timeout"))

  def _backoff_routine(self) -> Routine:
    """One step of the reflex's retreat, after the front stop or the
    bumper (`backoff_until`): straight back."""
    yield from self._nav_routine(-self.BACKOFF_V, 0.0)

  def _known_cells(self) -> int:
    """How much of the map is known -- free or wall -- in cells: what
    `PROGRESS_MAP_GROWTH` watches grow."""
    g = self.grid.grid
    return int(np.count_nonzero(g < FREE_THRESH) + np.count_nonzero(g > OCC_THRESH))

  def _left(self, dist: float, waypoints, wx: float, wy: float) -> float:
    """How far a drive has still to go: `dist`, the straight line, or --
    `PROGRESS_ALONG_ROUTE` -- along its waypoints to the goal."""
    if not self.PROGRESS_ALONG_ROUTE or not waypoints:
      return dist
    x, y = self.pose[0], self.pose[1]
    total = 0.0
    for px, py in (*waypoints, (wx, wy)):
      total += math.hypot(px - x, py - y)
      x, y = px, py
    return total

  def _drove(self, wx: float, wy: float, t0: float, why: str) -> bool:
    """Record how a drive ended in `last_drive` (issue #350) and answer it:
    True for `why == ""`, arrived."""
    px, py, _ = self.pose
    rec = {"why": why, "goal": (float(wx), float(wy)),
           "seconds": round(float(self.data.time - t0), 1),
           "shortM": round(math.hypot(wx - px, wy - py), 3)}
    bodies = self._bodies() if why == "peer" else []
    if bodies:
      # the ONE body the cause is about: the nearest to the goal where that
      # is nearer than the nearest to the robot -- where it was, and
      # whether it lies down (#365: never waited for, "lying" in the words)
      here = min(bodies, key=lambda b: math.hypot(b.x - px, b.y - py))
      there = min(bodies, key=lambda b: math.hypot(b.x - wx, b.y - wy))
      d_here = math.hypot(here.x - px, here.y - py)
      d_there = math.hypot(there.x - wx, there.y - wy)
      at_goal = d_there < d_here
      b = there if at_goal else here
      rec.update(peerAt="goal" if at_goal else "here",
                 peerM=round(d_there if at_goal else d_here, 3),
                 peerXY=(round(float(b.x), 3), round(float(b.y), 3)),
                 peerDown=bool(b.down))
    self.last_drive = rec
    return not why

  def _at_stand_in(self) -> bool:
    """Did the last plan aim at a stand-in for the goal, and is the robot
    at its end -- every metre of floor it could plan over, driven?"""
    return (self._stand_in is not None
            and math.hypot(self._stand_in[0] - self.pose[0],
                           self._stand_in[1] - self.pose[1]) < STAND_IN_REACHED_M)

  def face(self, heading: float) -> bool:
    return self.run(self.face_routine(heading))

  def face_routine(self, heading: float) -> Routine:
    """Turn in place to `heading`. False if the budget ran out first
    (issue #108) -- a robot that cannot turn must not be a robot that never
    gets back to the arbitration loop."""
    _, squared = yield from square_up_routine(
      lambda: wrap_angle(heading - self.pose[2]),
      lambda w: self._nav_routine(0.0, w),
      lambda: self._drive_routine(0.5, 0.0, 0.0),
      lambda: float(self.data.time), tol=self.FACING_TOLERANCE, tries=1,
      done_within=float("inf"), gain=2.5, limit=1.0)
    return squared

  # ---- the other robots ----------------------------------------------------

  def _cells(self, b: KeepClear) -> int:
    """The planner's disc round another robot, in cells: wider for one lying
    down (issue #365)."""
    return self.DOWN_ROBOT_CELLS if b.down else self.OTHER_ROBOT_CELLS

  def _bodies(self) -> list[KeepClear]:
    """Every other robot as a `KeepClear`: what `others` answers, a robot
    standing where the callable says no more than `(x, y)`."""
    return [KeepClear(*where()) for where in self.others]

  def _other_in_the_way(self, wx: float, wy: float) -> bool:
    """Is another robot within reach of this one, or of its goal, that
    waiting could move? Not one lying down (issue #365): it stays where it
    is until the world stands it up, and a drive that waited on it stood
    beside it for its whole timeout -- it plans round the body instead, or
    ends as any drive with nowhere to go does."""
    px, py, _ = self.pose
    for b in self._bodies():
      if not b.down and (math.hypot(b.x - px, b.y - py) < OTHER_NEAR_M
                         or math.hypot(b.x - wx, b.y - wy) < OTHER_NEAR_M):
        return True
    return False

  def peer_ahead(self, points) -> float | None:
    """The nearest point of ANOTHER ROBOT's body in the corridor this one
    is about to drive through, or None (issue #328).

    Points are the depth camera's peer channel, in the robot frame: x
    ahead of the axle, y to its left. The test is the robot's own
    footprint swept forward, not a cone -- a cone is what the LIDAR's front
    stop has, and it is why a peer 0.25 m across the bow put ZERO rays in
    it while its chassis was still wide enough to clip (measured, issue
    #328). Height is not tested: a robot is solid all the way up, and the
    only points here are a robot's.
    """
    if points is None or len(points) == 0:
      return None
    x, y = points[:, 0], points[:, 1]
    ahead = ((x > 0.0) & (x <= self.PEER_STOP_AHEAD_M)
             & (np.abs(y) <= self.PEER_STOP_HALF_M))
    return float(x[ahead].min()) if ahead.any() else None

  def watch_for_peers(self, points) -> float | None:
    """One depth frame's peer channel, recorded (issue #328): how far off
    the nearest body in the corridor was, and when it was seen.

    The seam only SEES. Whether a sighting is worth stopping for depends
    on how far this robot still has to drive, which is the drive's
    business and nobody else's -- the same division the front stop makes,
    where the scan arms a clock and `drive_to_routine` decides what to do
    about it.
    """
    near = self.peer_ahead(points)
    if near is not None:
      self.peer_seen_m, self.peer_seen_t = near, float(self.data.time)
    return near

  def peer_sighting(self) -> float | None:
    """The last peer sighting if it is still fresh (`PEER_HOLD_S`), else
    None -- a sighting that has aged out is where the robot USED to be."""
    if (self.peer_seen_m is None
        or self.data.time - self.peer_seen_t > PEER_HOLD_S):
      return None
    return self.peer_seen_m

  def peer_on_the_goal(self, wx: float, wy: float) -> float | None:
    """How far off the nearest robot standing ON this goal is, or None.

    ARITHMETIC, NOT A GUESS (issue #313). `_mask_others` takes a disc of
    `OTHER_ROBOT_CELLS` (0.60 m) out of the traversable mask around every
    other robot's reported pose, so when the goal is inside one the nearest
    cell A* may plan to is `radius - distance` away from it -- and a drive
    that stagnates there is only called arrived inside `CLOSE_ENOUGH_M`.
    A peer nearer the goal than the difference (0.45 m) therefore makes
    arrival IMPOSSIBLE, however many attempts are spent on it.

    That is the whole of #313: a robot parked at one bay's standoff is
    0.26 m from its neighbour's, the best reachable point is 0.35 m short,
    and three pick attempts out of three failed at "no route" -- while the
    same robot 0.56 m away leaves 0.06 m and three out of three land. It
    is not contention, contact or the planner: nothing the swap does can
    reach a goal the map has been told to keep it out of.

    The distance is to the REPORTED pose (a network fact, drifting
    0.24-0.55 m on the deployed pair), which is also what the mask uses --
    so this answers the question the planner actually asked. A robot lying
    down is measured to its body with its own wider disc (0.55 m; issue
    #365), for the same reason.
    """
    res = self.grid.resolution
    near = None
    for b in self._bodies():
      d = math.hypot(b.x - wx, b.y - wy)
      if d < self._cells(b) * res - CLOSE_ENOUGH_M and (near is None or d < near):
        near = d
    return near

  def reachable(self, points) -> list[bool]:
    """Which world points this robot could plan to right now (issue #346):
    traversable after the other robots are masked out, and in the SAME
    4-connected component as its own start cell -- `_plan_to`'s own two
    tests, asked of candidates rather than of one goal."""
    trav = traversable_mask(self._planning_grid(), self.INFLATION_CELLS)
    self._mask_others(trav)
    start = nearest_traversable(
      trav, self.grid.world_to_cell(self.pose[0], self.pose[1]))
    if start is None:
      return [False] * len(points)
    labels, _ = ndimage.label(trav)
    own = labels[start[1], start[0]]
    rows, cols = trav.shape
    out = []
    for wx, wy in points:
      cx, cy = self.grid.world_to_cell(wx, wy)
      out.append(0 <= cx < cols and 0 <= cy < rows
                 and bool(trav[cy, cx]) and labels[cy, cx] == own)
    return out

  def _mask_others(self, trav) -> None:
    """Take every other robot's footprint out of the traversable mask,
    inflated as the map's obstacles are (issue #167). This is where the
    other robot SAYS it is now -- a network fact, refreshed every replan --
    or, while it lies on the floor, where its body is (issue #365).
    Its body is in no scan of this robot's (`Lidar.scan_split`: the map
    never sees another robot, and since issue #316 the front-stop reflex
    always does), so the mask is the only thing routing around it.
    ⚠ A goal INSIDE one of these discs cannot be reached at all --
    `peer_on_the_goal` is that arithmetic, and callers ask it before
    spending another attempt on a drive that has nowhere to arrive."""
    rows, cols = trav.shape
    for b in self._bodies():
      r = self._cells(b)
      cx, cy = self.grid.world_to_cell(b.x, b.y)
      x0, x1 = max(cx - r, 0), min(cx + r + 1, cols)
      y0, y1 = max(cy - r, 0), min(cy + r + 1, rows)
      if x0 >= x1 or y0 >= y1:
        continue
      ys, xs = np.ogrid[y0:y1, x0:x1]
      trav[y0:y1, x0:x1] &= (xs - cx) ** 2 + (ys - cy) ** 2 > r * r
