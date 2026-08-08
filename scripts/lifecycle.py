"""Full lifecycle: PluggyBot explores, docks when hungry, charges, repeats.

A mission-level state machine on top of the navigation stack, closed into
the loop that names the project (milestone 7):

  EXPLORE --- battery low, or map done --> GO_CHARGE --> FACE_OUTLET --> DOCK
     ^   frontier-drive the map while         A* to the     square up     servo,
     |   the detector logs landmarks          standoff       (0.5 deg)    insert
     |                                                                      |
     +--- recharged, map unfinished --- CHARGE <--- charge contact ---------+
                                          |  sit + charge, then undock
                                          +--- map complete --> DONE

The battery drains against real actuator effort (pluggybot.power) and
charges on the electrical contact criterion -- it never knows what it is
plugged into, which is exactly the seam milestone 8's hub docks into.

Usage:
  MUJOCO_GL=egl uv run python scripts/lifecycle.py --headless
  uv run python scripts/lifecycle.py                       # watch live
  uv run python scripts/lifecycle.py --headless --battery-wh 0.5   # hungrier
"""

import argparse
import math
import time
from typing import Literal

import mujoco
import mujoco.viewer
import numpy as np
from PIL import Image

from pluggybot.behavior.navigation import (
  BACKOFF_TIME,
  FRONT_STOP_RANGE,
  K_HEADING,
  MAP_SAVE_PERIOD,
  REPLAN_PERIOD,
  SCAN_EVERY,
  STRIKES_TO_FINISH,
  W_MAX,
  W_SPIN,
  WAYPOINT_RADIUS,
  drive_toward,
  path_to_waypoints,
  plan,
  render_map,
)
from pluggybot.control import wheel_targets, slew, wrap_angle
from pluggybot.docking.contact import charging_contact, feelers_touching
from pluggybot.mapping.astar import astar
from pluggybot.mapping.frontier import traversable_mask
from pluggybot.mapping.landmarks import LandmarkStore, wall_normal
from pluggybot.mapping.occupancy_grid import OccupancyGrid
from pluggybot.odometry.dead_reckoning import DeadReckoner
from pluggybot.perception.outlet_spotter import OutletSpotter, latest_weights
from pluggybot.perception.scanner import Scanner
from pluggybot.power import DEMO_CAPACITY_WH, Battery
from pluggybot.viz import ViewDashboard

State = Literal["EXPLORE", "GO_CHARGE", "FACE_OUTLET", "DOCK", "CHARGE", "DONE"]

SPOT_PERIOD = 0.5        # sim seconds between YOLO looks (outlets don't move)
BACKOFF_SPEED = -0.15    # m/s straight reverse when the safety reflex trips

# ---- battery thresholds (milestone 7) ---------------------------------------
LOW_BATTERY_WH = 0.40    # go charge below this ABSOLUTE reserve. Absolute, not
                         # a fraction: the mission's cost is fixed by the room,
                         # not the pack -- a 35%-of-capacity reserve starved a
                         # 0.55 Wh cell mid-insertion (dead at 81.5 s, four
                         # seconds short of contact). And it must cover more
                         # than one clean run home: a failed dock attempt plus
                         # the redrive to a fallback outlet measured ~0.4 Wh
                         # worst case (0.25 died at 141 s mid-redrive).
CHARGED = 0.90           # stop charging here (LiPo longevity; also the demo
                         # doesn't need the taper phase a real charger has)
CHARGE_LOST_TIMEOUT = 5.0  # s without charge contact mid-charge = unseated
UNDOCK_RETRACT_TIME = 2.0  # s pulling the arm in before reversing
UNDOCK_REVERSE_TIME = 4.0  # s at -0.06 m/s: ~0.25 m off the wall

STANDOFF_DISTANCE = 0.6  # m out from the outlet: where docking will take over
STANDOFF_RETRY_STEP = 0.15   # m further out when no route to the standoff exists
ARRIVE_RADIUS = 0.12     # m: close enough to the standoff point to stop driving
CHARGE_STRIKES = 6       # pathless replans before giving up on this outlet
# Docking tolerates +/-3 deg of yaw (schuko_spike.py), so settling at 2 deg
# spent the whole budget on the controller alone. 0.5 deg leaves the margin
# for odometry drift and the docking controller's own error.
FACING_TOLERANCE = math.radians(0.5)

# ---- DOCK state (measured numbers; see SimNotes "Docking lessons") ----------
DOCK_TIMESTEP = 0.001    # mm-scale contacts: the spike's stability floor
DOCK_DROOP_COMP = 0.0078  # m: plug axis sags below the lift target under
                          # gravity (lift servo + RCC springs). Calibrated in
                          # sim; on hardware this is a calibration constant.
DOCK_CREEP = 0.3         # wheel rad/s while servoing in (~13 mm/s)
DOCK_PRESS = 0.1         # wheel rad/s pressing feelers on the wall (~2.2 N).
                         # The press must EXCEED the arm cap or the arm pushes
                         # the base backward and pays out slack forever (a
                         # seated plug read as a timeout). Raising the press
                         # instead of lowering the arm was tried and measured
                         # worse: wheel torque reaction pitches the chassis
                         # nose-down, sagging the plug ~1.6x the calibration.
DOCK_ARM_FORCE = 2.5     # N: arm cap ~ the base's measured push budget.
                         # Deliberately ABOVE the 2.2 N wheel press: the extra
                         # authority lets the funnel finish correcting the last
                         # few mm (1.8 N measured too weak to ride the chamfer).
                         # The side effect -- a seated plug slowly pushes the
                         # base off the wall -- is itself the seat detector.
DOCK_SERVO_GAIN = 3.0    # rad/s per m of lateral error seen by dock_eye
DOCK_SEAT_EXT = 0.065    # m: a stall deeper than this is seated, shallower is a jam
DOCK_SEAT_EXT_MAX = 0.15  # m: a stall beyond this reached into AIR (a mis-aimed
                          # arm found nothing to touch) -- a failure, not a seat.
                          # The first false "PLUGGED IN" stalled at 198 mm, arm
                          # fully extended over the top of the housing.
# Hand-eye calibration (measured, perfectly-aligned robot vs servo output):
#   range   lateral bias   vertical bias   -> the box clips on the frame
#   0.50 m     +0.9 mm        +1.1 mm         bottom below ~0.32 m (camera
#   0.32 m     +0.9 mm        +2.1 mm         rides 0.064 above the plug), so
#   0.26 m     +1.0 mm        +9.1 mm         its center biases UP while the
#   0.19 m     +0.4 mm       +22.8 mm         lateral center stays honest.
# Each axis therefore freezes at its own range floor:
DOCK_ZSERVO_FREEZE = 0.32   # vertical trim: frozen below this range
DOCK_YSERVO_FREEZE = 0.17   # lateral steer: usable nearly to contact
DOCK_STALL_TIME = 0.8    # s without arm progress = stalled (seated or jammed)
DOCK_VERIFY_TIME = 1.5   # s of servoing with no detection = phantom target
DOCK_MAX_ATTEMPTS = 2


class Lifecycle:
  """Mission state machine over the navigation stack.

  Two levels of state, deliberately kept apart:

    self.state -- the MISSION phase (EXPLORE / GO_CHARGE / FACE_OUTLET / DONE).
                  One method per phase. A phase method returns the (v, w) drive
                  command for this step, and hands off by assigning self.state.
    self.mode  -- the MANEUVER within a phase ("spin" / "drive"). "Turn in
                  place until done" is not a mission phase, it's how a phase
                  gets its job done.

  Why a class rather than nested functions: the phases share a dozen mutable
  values (waypoints, timers, strike counts, the map). A nested function can
  READ an enclosing local but cannot REBIND one without declaring `nonlocal`
  for every single name -- and `mode = "spin"` inside a nested function
  silently creates a new local instead, which is the bug that bites everyone
  once. Passing them all as arguments and returning them as tuples works but
  reads terribly. Instance attributes are the same sharing, spelled clearly.
  """

  def __init__(self, headless: bool, max_sim_time: float,
               weights: str | None, explore_budget: float | None = None,
               views: bool = False, battery_wh: float = DEMO_CAPACITY_WH) -> None:
    self.model = mujoco.MjModel.from_xml_path("models/room_1.xml")
    self.data = mujoco.MjData(self.model)
    self.headless = headless
    self.max_sim_time = max_sim_time
    # The timer stand-in survives as an override (tests use it; a demo can
    # force an early charge with it). None = the battery is the trigger.
    self.explore_budget = math.inf if explore_budget is None else explore_budget
    self.battery = Battery(self.model, capacity_wh=battery_wh)
    self.cruise_timestep = self.model.opt.timestep   # restored after undocking

    # -- subsystems
    self.reckoner = DeadReckoner(wheel_radius=0.045, track_width=0.21)
    self.scanner = Scanner(self.model)
    self.grid = OccupancyGrid(x_min=-3, y_min=-3, x_max=7, y_max=7, resolution=0.05)
    self.landmarks = LandmarkStore()
    self.spotter = OutletSpotter(self.model, weights) if weights else None
    # Second spotter on the carriage camera: the terminal visual servo's eye.
    self.dock_spotter = (OutletSpotter(self.model, weights, camera_name="dock_eye")
                         if weights else None)
    # Camera dashboard (issue #1): stereo pair + map + dock camera -> views.png,
    # refreshed alongside map.png. Flag-gated: three extra renders per save.
    self.dashboard = ViewDashboard(self.model) if views else None

    # -- model handles, looked up once
    m = self.model
    self.left_act = m.actuator("left_motor").id
    self.right_act = m.actuator("right_motor").id
    self.left_adr = m.joint("left_wheel_joint").qposadr[0]
    self.right_adr = m.joint("right_wheel_joint").qposadr[0]
    self.gyro = m.sensor("imu_gyro")
    self.chassis_gid = m.geom("chassis").id

    # -- mission state
    self.state: State = "EXPLORE"
    self.target = None                 # the Landmark being driven to
    self.explore_done_reason = "running"
    self.standoff_distance = STANDOFF_DISTANCE
    self.standoff_dir: tuple[float, float] | None = None
    self.charge_strikes = 0
    self.final_heading_error: float | None = None

    # -- CHARGE state
    self.charge_stage = "charging"     # charging -> retract -> backoff
    self.last_charge_contact = 0.0
    self.charging_now = False

    # -- DOCK state
    self.dock_stage = "align"          # align -> servo -> insert (-> retract)
    self.stage_t0 = 0.0
    self.dock_attempts = 0
    self.set_aside: set[int] = set()   # landmarks benched after jams (kept in map)
    self.last_seen_socket = 0.0
    self.next_dock_look = 0.0
    self.dock_w = 0.0
    self.dock_lift_trim = 0.0
    self.dock_z_fixed = False
    self.last_ext = 0.0
    self.last_ext_t = 0.0
    self.docked = False
    self.charged_once = False

    # -- shared navigation state
    self.waypoints: list[tuple[float, float]] = []
    self.blacklist: set[tuple[int, int]] = set()
    self.backoff_until = 0.0

    # -- EXPLORE-only state
    self.mode = "spin"                 # open with a 360 look-around to seed the map
    self.spin_remaining = 2 * math.pi
    self.just_spun = False
    self.next_replan = 0.0
    self.strikes = 0

    # -- bookkeeping
    self.next_spot = 0.0
    self.step_count = 0
    self.collision_steps = 0
    self.last_save = 0.0
    self.last_report = 0.0

  # ---- shared plumbing -----------------------------------------------------

  @property
  def pose(self) -> tuple[float, float, float]:
    """Dead-reckoned (x, y, theta) of the axle midpoint."""
    return (self.reckoner.x, self.reckoner.y, self.reckoner.theta)

  def perceive(self) -> None:
    """Senses, on their own cadences -- shared by every phase, because what
    the robot can see doesn't depend on what it's trying to do."""
    d, m = self.data, self.model
    self.reckoner.update(
      d.qpos[self.left_adr], d.qpos[self.right_adr],
      gyro_yaw_rate=d.sensordata[self.gyro.adr[0] + 2], dt=m.opt.timestep,
    )
    # Battery: drains against actual actuator effort every step; charges
    # whenever the charge circuit reports contact -- the battery neither
    # knows nor cares WHAT it is plugged into (milestone 8's hub reuses this).
    self.charging_now = self.state == "CHARGE" and self.charging_contact()
    self.battery.update(d, m.opt.timestep, charging=self.charging_now)

    if self.step_count % SCAN_EVERY == 0:
      angles, ranges = self.scanner.scan(d)
      self.grid.update(self.pose, angles, ranges, self.scanner.max_range)
      # Safety reflex: anything dead ahead that planning didn't expect.
      # Fires in EVERY phase and maneuver, including spins -- explore.py only
      # armed it while following waypoints, and measurement showed its
      # look-around spins passing within 0.257 m of the L-box, 7 mm outside
      # this threshold. It never collided by luck, not by design.
      # Not re-armed mid-backoff, so a backoff is a bounded 0.8 s pulse
      # rather than an open-ended reverse into whatever is behind.
      if d.time >= self.backoff_until and self.state != "DOCK":
        front = ranges[np.abs(angles) < 0.35]
        if front.min() < FRONT_STOP_RANGE:
          self.backoff_until = d.time + BACKOFF_TIME
          self.waypoints = []          # forces the phase to replan afterwards

    # No map-spotter sightings while docking/charging: the head camera is
    # then closer than its calibrated regime, and a long jam racks up dozens
    # of biased sightings that DRAG the landmark into the wall -- measured:
    # outlet C's estimate ended 8 cm inside the wall slab after failed dock
    # attempts, wall_normal() over an in-wall point returned a -67 deg
    # standoff, and every later approach was worse (a feedback loop). The
    # dock eye has its own close-range pipeline; exploration owns this one.
    if (self.spotter is not None and d.time >= self.next_spot
        and self.state not in ("DOCK", "CHARGE")):
      self.next_spot = d.time + SPOT_PERIOD
      px, py, _ = self.pose
      for x, y, z, _conf in self.spotter.spot(d, self.pose):
        self.landmarks.add_sighting(x, y, z, seen_from=(px, py))

  def actuate(self, v: float, w: float) -> None:
    """(v, w) body command -> slew-limited wheel velocity targets."""
    tl, tr = wheel_targets(v, w)
    ts = self.model.opt.timestep
    self.data.ctrl[self.left_act] = slew(self.data.ctrl[self.left_act], tl, ts)
    self.data.ctrl[self.right_act] = slew(self.data.ctrl[self.right_act], tr, ts)

  def follow_waypoints(self) -> tuple[float, float]:
    """Drive at the next waypoint, dropping any already reached. Returns
    (0, 0) once the list empties -- refilling it is the phase's job."""
    while self.waypoints:
      wx, wy = self.waypoints[0]
      if math.hypot(wx - self.pose[0], wy - self.pose[1]) < WAYPOINT_RADIUS:
        self.waypoints.pop(0)
        continue
      return drive_toward(self.pose, (wx, wy))
    return 0.0, 0.0

  def start_spin(self) -> None:
    self.mode = "spin"
    self.spin_remaining = 2 * math.pi

  PLUG_LATERAL = 0.05    # m: the plug rides this far right of the centerline

  def standoff_of(self, lm) -> tuple[float, float, float]:
    """A landmark's docking hand-off pose, using the map-derived wall normal.

    The normal is cached in self.standoff_dir for the current target: it costs
    ~150 grid lookups and is wanted on every step of GO_CHARGE, and the wall
    it describes does not move.
    """
    if lm is self.target and self.standoff_dir is not None:
      sx, sy, hd = lm.standoff(self.standoff_distance, direction=self.standoff_dir)
    else:
      d = wall_normal(self.grid, lm.x, lm.y,
                      fallback=(lm.seen_from_x - lm.x, lm.seen_from_y - lm.y))
      sx, sy, hd = lm.standoff(self.standoff_distance, direction=d)
    # Shift the AXLE target left so the PLUG (not the robot centerline) lands
    # on the socket axis: docking measured a structural 5 cm lateral miss
    # without this, forcing the visual servo to burn its whole authority.
    return sx - self.PLUG_LATERAL * math.sin(hd), \
           sy + self.PLUG_LATERAL * math.cos(hd), hd

  def plan_to(self, wx: float, wy: float) -> bool:
    """A* from the robot's cell to a world point; fills self.waypoints.

    Returns False when no route exists -- the caller decides whether that
    means "wait for a better map" or "give up". Same planner explore() uses
    via navigation.plan(); the only difference is a fixed goal instead of a
    frontier search.
    """
    trav = traversable_mask(self.grid.grid)
    rows, cols = trav.shape
    rix, riy = self.grid.world_to_cell(self.pose[0], self.pose[1])
    start = (min(max(rix, 0), cols - 1), min(max(riy, 0), rows - 1))
    if not trav[start[1], start[0]]:
      # The robot itself can stand inside the inflation ring -- guaranteed
      # right after docking, when it is pressed against a wall. A* would fail
      # from a non-traversable start, so plan from the nearest free cell and
      # let drive_toward cover the first few centimeters.
      best, best_d = None, 1e9
      for dy in range(-10, 11):
        for dx in range(-10, 11):
          x2, y2 = start[0] + dx, start[1] + dy
          if 0 <= x2 < cols and 0 <= y2 < rows and trav[y2, x2]:
            if dx * dx + dy * dy < best_d:
              best, best_d = (x2, y2), dx * dx + dy * dy
      if best is None:
        return False
      start = best
    goal = self.grid.world_to_cell(wx, wy)
    if not (0 <= goal[0] < cols and 0 <= goal[1] < rows):
      return False
    path = astar(trav, start, goal)     # None if the goal cell isn't traversable
    if path is None:
      return False
    self.waypoints = path_to_waypoints(self.grid, path)
    return True

  # ---- mission phases ------------------------------------------------------
  # Each returns the (v, w) command for this step and may reassign self.state.

  def explore(self) -> tuple[float, float]:
    """Frontier-drive the map while the detector logs outlet landmarks."""
    d = self.data
    if self.battery.energy_wh < LOW_BATTERY_WH or d.time >= self.explore_budget:
      return self.leave_explore("battery-low")

    # -- decide: replan when the plan is stale or spent (a spin runs to completion)
    if self.mode == "drive" and (d.time >= self.next_replan or not self.waypoints):
      self.next_replan = d.time + REPLAN_PERIOD
      path, status = plan(self.grid, self.pose, self.blacklist)
      if status == "ok":
        self.waypoints = path_to_waypoints(self.grid, path)
        self.strikes = 0
        self.just_spun = False
      else:
        self.waypoints = []
        if not self.just_spun:
          self.start_spin()            # look around before concluding anything
          self.just_spun = True
        else:
          # spinning here didn't help: either truly done, or count a strike
          self.strikes += 1
          if status == "no-frontiers" or self.strikes >= STRIKES_TO_FINISH:
            return self.leave_explore(status)

    # -- act
    if self.mode == "spin":
      self.spin_remaining -= W_SPIN * self.model.opt.timestep
      if self.spin_remaining > 0:
        return 0.0, W_SPIN
      self.mode = "drive"
      self.waypoints = []
      self.next_replan = d.time        # replan immediately on the fresh map
      return 0.0, 0.0
    return self.follow_waypoints()

  def leave_explore(self, reason: str) -> tuple[float, float]:
    """Exit EXPLORE: head for the nearest known outlet, or stop if none."""
    self.explore_done_reason = reason
    self.target = self.landmarks.nearest_confirmed(self.pose[0], self.pose[1])
    self.state = "GO_CHARGE" if self.target is not None else "DONE"
    self.waypoints = []
    if self.target is not None:
      # Read the wall's facing off the finished map, once, and reuse it.
      self.standoff_dir = wall_normal(
        self.grid, self.target.x, self.target.y,
        fallback=(self.target.seen_from_x - self.target.x,
                  self.target.seen_from_y - self.target.y))
    print(f"t={self.data.time:6.1f}s  EXPLORE -> {self.state}  ({reason}; "
          f"{len(self.landmarks.confirmed())} outlet(s) remembered)")
    return 0.0, 0.0

  def go_charge(self) -> tuple[float, float]:
    """Drive to the standoff point in front of the target outlet.

    Same decide/act split as explore(); the only difference is that the goal
    is a fixed point rather than whichever frontier is nearest.
    """
    d = self.data
    gx, gy, _ = self.standoff_of(self.target)

    # Arrival is checked BEFORE replanning: standing on the goal makes A*
    # return a one-cell path, which follow_waypoints immediately empties --
    # replanning first would spin between "arrived" and "replan" forever.
    if not self.waypoints and math.hypot(gx - self.pose[0], gy - self.pose[1]) < ARRIVE_RADIUS:
      print(f"t={d.time:6.1f}s  GO_CHARGE -> FACE_OUTLET  (at standoff, "
            f"{self.standoff_distance:.2f} m out)")
      self.state = "FACE_OUTLET"
      return 0.0, 0.0

    if d.time >= self.next_replan or not self.waypoints:
      self.next_replan = d.time + REPLAN_PERIOD
      if self.plan_to(gx, gy):
        self.charge_strikes = 0
      else:
        # Usually the standoff cell sits inside the wall's inflation ring:
        # the landmark is a few cm off, or the outlet is in a tight corner.
        # Backing the goal away from the wall is what makes it reachable.
        self.charge_strikes += 1
        self.standoff_distance += STANDOFF_RETRY_STEP
        if self.charge_strikes >= CHARGE_STRIKES:
          print(f"t={d.time:6.1f}s  GO_CHARGE -> DONE  (no route to any "
                f"standoff pose after {CHARGE_STRIKES} tries)")
          self.state = "DONE"
        return 0.0, 0.0

    return self.follow_waypoints()

  def face_outlet(self) -> tuple[float, float]:
    """Pivot in place until squared up with the outlet, then stop.

    The end of the mission as far as this milestone goes: the robot is parked
    at the standoff pose, facing the socket. The docking controller
    (milestone 6) takes over from exactly here, and the spike measured its
    yaw budget at +/-3 deg -- so this is the step whose accuracy matters.
    """
    _, _, want = self.standoff_of(self.target)
    err = wrap_angle(want - self.pose[2])
    if abs(err) > FACING_TOLERANCE:
      return 0.0, max(-W_MAX, min(W_MAX, K_HEADING * err))
    self.final_heading_error = err
    print(f"t={self.data.time:6.1f}s  FACE_OUTLET -> DOCK  (believed heading "
          f"error {math.degrees(err):+.2f} deg)")
    self.state = "DOCK"
    self.dock_stage = "align"
    self.dock_z_fixed = False
    self.dock_lift_trim = 0.0
    self.stage_t0 = self.data.time
    # mm-scale plug/socket contacts need the finer step (spike lesson);
    # switching mid-run is safe and keeps exploration at the cheap 2 ms.
    self.model.opt.timestep = DOCK_TIMESTEP
    self.model.actuator_forcerange[self.model.actuator("arm").id] =         [-DOCK_ARM_FORCE, DOCK_ARM_FORCE]
    return 0.0, 0.0

  def reject_target(self, why: str, forget: bool = False) -> tuple[float, float]:
    """Give up on this landmark and pick another.

    forget=True erases it -- for phantoms only (the last defense against a
    systematic detector false positive, see SimNotes). A genuine outlet that
    merely JAMMED is kept in the map but benched for this mission: watched
    failure mode was the robot jamming twice at a TRUE outlet, deleting it,
    and permanently forgetting a real charging spot."""
    print(f"t={self.data.time:6.1f}s  DOCK rejects target at "
          f"({self.target.x:+.2f}, {self.target.y:+.2f}): {why}")
    if forget and self.target in self.landmarks.landmarks:
      self.landmarks.landmarks.remove(self.target)
    self.set_aside.add(id(self.target))
    self.data.ctrl[self.model.actuator("arm").id] = 0.0
    self.data.ctrl[self.model.actuator("lift").id] = 0.0
    # Rejecting re-enters GO_CHARGE: restore cruise physics (DOCK's fine
    # timestep would otherwise tax the whole cross-map redrive, and
    # face_outlet re-applies it on the next approach anyway).
    self.model.opt.timestep = self.cruise_timestep
    self.model.actuator_forcerange[self.model.actuator("arm").id] = [-50, 50]
    self.standoff_distance = STANDOFF_DISTANCE
    self.dock_attempts = 0
    candidates = [lm for lm in self.landmarks.confirmed()
                  if id(lm) not in self.set_aside]
    self.target = min(candidates, key=lambda lm: math.hypot(
      lm.x - self.pose[0], lm.y - self.pose[1])) if candidates else None
    self.state = "GO_CHARGE" if self.target is not None else "DONE"
    if self.target is not None:
      self.standoff_dir = wall_normal(
        self.grid, self.target.x, self.target.y,
        fallback=(self.target.seen_from_x - self.target.x,
                  self.target.seen_from_y - self.target.y))
    return 0.0, 0.0

  def charging_contact(self) -> bool:
    """The sim analog of the charging circuit's voltage detector — shared
    with the RL docking env; see pluggybot.docking.contact for the history."""
    return charging_contact(self.model, self.data)

  def feelers_touching(self) -> int:
    return feelers_touching(self.model, self.data)

  def dock(self) -> tuple[float, float]:
    """Terminal docking: align lift -> visually servo in -> insert.

    Wheels do the pushing (velocity ctrl, force-limited by kv), the arm does
    the last centimeters under a 2.5 N cap (the measured push budget), and
    the feelers square the yaw mechanically when both touch the wall. All
    numbers here were measured by probes before this controller existed.
    """
    d, m = self.data, self.model
    lift, arm = m.actuator("lift").id, m.actuator("arm").id

    if self.dock_stage == "align":
      d.ctrl[lift] = (self.target.z - 0.145) + DOCK_DROOP_COMP
      if d.time - self.stage_t0 > 1.5:
        self.dock_stage = "servo"
        self.stage_t0 = d.time
        self.last_seen_socket = d.time
        self.dock_w = 0.0
      return 0.0, 0.0

    if self.dock_stage == "servo":
      # Creep in while steering the plug line onto the socket with dock_eye.
      # The camera sits ON the plug axis (same y), so a detection's lateral
      # offset IS the plug's miss distance: e = u_c * range / f. The detector
      # runs at ~4 Hz (a YOLO call costs ~10 physics steps of wall time at
      # the 1 ms docking timestep); the steering command holds in between.
      w = self.dock_w
      if self.dock_spotter is not None and d.time >= self.next_dock_look:
        self.next_dock_look = d.time + 0.25
        r = self.dock_spotter.renderer
        r.update_scene(d, camera="dock_eye")
        rgb = r.render()
        r.enable_depth_rendering()
        r.update_scene(d, camera="dock_eye")
        depth = r.render()
        r.disable_depth_rendering()
        res = self.dock_spotter.detector.predict(
          np.ascontiguousarray(rgb[:, :, ::-1]), conf=0.5, verbose=False)[0]
        if len(res.boxes):
          self.last_seen_socket = d.time
          u, v = float(res.boxes.xywh[0][0]), float(res.boxes.xywh[0][1])
          iu, iv = min(int(u), 639), min(int(v), 359)
          rng = float(np.median(depth[max(0, iv-1):iv+2, max(0, iu-1):iu+2]))
          f = 180 / math.tan(math.radians(41.0) / 2)
          # Two-axis servo, each axis trusted only in its calibrated range
          # (see the table above). Lateral: the camera shares the plug's
          # y-axis, so the horizontal image offset IS the plug's miss ->
          # steer the base. Vertical: the camera rides 0.064 m above the
          # plug axis (mount + measured RCC droop), so the socket should
          # image that far below center; the residual is a plug height
          # error the wheels cannot fix -> trim the lift.
          if rng >= DOCK_YSERVO_FREEZE:
            e = (u - 320 + 0.5) * rng / f
            w = max(-0.4, min(0.4, -DOCK_SERVO_GAIN * e))
            self.dock_w = w
          if rng >= DOCK_ZSERVO_FREEZE:
            err_z = -(v - 180 + 0.5) * rng / f + 0.064
            self.dock_lift_trim = max(-0.01, min(0.01,
              self.dock_lift_trim + 0.4 * err_z))
            d.ctrl[lift] = (self.target.z - 0.145) + DOCK_DROOP_COMP + self.dock_lift_trim
        elif (d.time - self.stage_t0 < 5.0
              and d.time - self.last_seen_socket > DOCK_VERIFY_TIME):
          # Absent detection early in the approach = phantom target; absent
          # close to the wall = normal occlusion, keep going on frozen cmds.
          return self.reject_target("no outlet visible up close (phantom?)", forget=True)
      touching = self.feelers_touching()
      if touching == 2:
        self.dock_stage = "insert"
        self.stage_t0 = d.time
        self.last_ext = 0.0
        self.last_ext_t = d.time
        # Press-sag feed-forward: pressing the feelers pitches the robot and
        # the plug sags MORE the higher the lift (moment arm). Calibrated:
        # -2.5 mm at z=0.26, -16.7 mm at z=0.38 -> ~118 mm per meter of lift.
        press_sag = 0.0025 + 0.118 * max(0.0, self.target.z - 0.26)
        d.ctrl[lift] = ((self.target.z - 0.145) + DOCK_DROOP_COMP
                        + self.dock_lift_trim + press_sag)
        d.ctrl[arm] = 0.20
        # World-frame reference for plug advance: odometry projection along
        # the heading plus arm extension. Advance is what discriminates a
        # true seat from a jam -- both push the base off the wall.
        h = self.pose[2]
        self.insert_ref = (self.pose[0] * math.cos(h) + self.pose[1] * math.sin(h))
        return 0.0, 0.0
      if touching == 1 and d.time - self.stage_t0 > 4.0:
        # one feeler down for a while: pivot gently to land the other
        w = 0.25 if d.time % 2.0 < 1.0 else -0.25
      if d.time - self.stage_t0 > 30.0:
        return self.reject_target("could not square up on the wall")
      # Approach at 0.06 m/s across the open ~0.5 m of standoff, dropping to
      # feeler-creep speed once anything touches (probe-tuned: first contact
      # at 60 mm/s just compresses solref and settles; no bounce).
      v = DOCK_CREEP * 0.045 if touching else 0.06
      return v, w

    if self.dock_stage == "insert":
      ext = d.qpos[m.joint("arm_joint").qposadr[0]]
      # Seat verdict = the charging circuit reports contact (see
      # charging_contact). Every clever inference tried before this --
      # extension windows, base-release, odometry advance -- was defeated by
      # some jam mode or frame bug; the electrical criterion is the one the
      # physical robot will use anyway.
      if self.charging_contact():
        self.docked = True
        # Deliberately NOT freezing the arm at ext: with the position servo
        # holding zero error the wheel press routes through the stiffer
        # prong-wall path and the pin floats off the floor within a step --
        # measured as an instant loss of charge contact, every time. The
        # capped arm press IS what keeps the pin on the contacts.
        print(f"t={d.time:6.1f}s  DOCK -> CHARGE  PLUGGED IN "
              f"(charge contact, ext {ext * 1000:.1f} mm, "
              f"battery {self.battery.fraction:.0%})")
        self.state = "CHARGE"
        self.charge_stage = "charging"
        self.last_charge_contact = d.time
        self.stage_t0 = d.time
        return 0.0, 0.0
      if ext > self.last_ext + 0.0005:
        self.last_ext = ext
        self.last_ext_t = d.time
      stalled = d.time - self.last_ext_t > DOCK_STALL_TIME
      released = (self.feelers_touching() == 0 and d.time - self.stage_t0 > 1.0)
      if stalled or released or ext > DOCK_SEAT_EXT_MAX:
        # no charge and no progress: jammed (or reached into air) -- retry
        self.dock_attempts += 1
        d.ctrl[arm] = 0.0
        self.dock_lift_trim = 0.0      # a bad trim may be WHY it missed
        d.ctrl[lift] = (self.target.z - 0.145) + DOCK_DROOP_COMP
        if self.dock_attempts >= DOCK_MAX_ATTEMPTS:
          return self.reject_target(f"no charge contact after "
                                    f"{DOCK_MAX_ATTEMPTS} attempts")
        print(f"t={d.time:6.1f}s  DOCK no contact (ext {ext * 1000:.0f} mm) -- retrying")
        self.dock_stage = "retract"
        self.stage_t0 = d.time
        return 0.0, 0.0
      return DOCK_PRESS * 0.045, 0.0   # gentle press while the arm works

    # retract: pull the arm in, back away, then servo back in
    if d.time - self.stage_t0 < 2.0:
      return -0.06, 0.0
    self.dock_stage = "servo"
    self.stage_t0 = d.time
    self.last_seen_socket = d.time
    return 0.0, 0.0

  def charge(self) -> tuple[float, float]:
    """Plugged in: sit still and take on energy, then undock and move on.

    Undocking is the docking film played backwards -- retract the arm, back
    straight off the wall -- and where the mission goes next depends on WHY
    it came here: a battery-low interrupt resumes exploring (the loop that
    names the project); a finished map means this was the final top-off, and
    the mission ends plugged into nothing but its own satisfaction.
    """
    d, m = self.data, self.model

    if self.charge_stage == "charging":
      if self.charging_now:
        self.last_charge_contact = d.time
      elif d.time - self.last_charge_contact > CHARGE_LOST_TIMEOUT:
        # The seated plug crept out (the arm cap's known side effect is the
        # base sliding back). Not a new mission phase: it is a dock problem,
        # so hand it back to the dock servo to re-seat.
        print(f"t={d.time:6.1f}s  CHARGE lost contact -- re-docking")
        self.state = "DOCK"
        self.dock_stage = "servo"
        self.stage_t0 = d.time
        self.last_seen_socket = d.time
        return 0.0, 0.0
      if self.battery.fraction >= CHARGED:
        print(f"t={d.time:6.1f}s  CHARGE done ({self.battery.fraction:.0%}) -- undocking")
        self.charged_once = True
        self.charge_stage = "retract"
        self.stage_t0 = d.time
        d.ctrl[m.actuator("arm").id] = 0.0
        return 0.0, 0.0
      # Charging keeps the FULL insert-stage press regime: arm at its full
      # target under the 2.5 N cap, wheels gently pressing. Both halves were
      # measured necessary, one mission-death each: wheels-at-zero let the
      # RCC relax and the pin float off the floor (14 dock->lost cycles,
      # starved at 1%); arm-frozen-at-ext routed the wheel press through the
      # stiffer prong-wall path instead of the pin (contact lost within a
      # step of every re-seat, same starvation). A real Schuko's spring
      # contacts grip the pins; a springless sim socket needs sustained
      # press as the analog.
      d.ctrl[m.actuator("arm").id] = 0.20
      return DOCK_PRESS * 0.045, 0.0

    if self.charge_stage == "retract":
      if d.time - self.stage_t0 > UNDOCK_RETRACT_TIME:
        self.charge_stage = "backoff"
        self.stage_t0 = d.time
      return 0.0, 0.0

    # backoff: straight reverse off the wall, then decide where life goes
    if d.time - self.stage_t0 < UNDOCK_REVERSE_TIME:
      return -0.06, 0.0
    if self.explore_done_reason == "battery-low":
      # The map was not finished -- back to work. Restore cruise physics
      # (the fine timestep and the 2.5 N arm cap were DOCK's settings).
      m.opt.timestep = self.cruise_timestep
      m.actuator_forcerange[m.actuator("arm").id] = [-50, 50]
      self.explore_budget = math.inf     # the timer stand-in fires once
      self.state = "EXPLORE"
      self.start_spin()                  # relocalize the frontier picture
      self.waypoints = []
      self.next_replan = d.time
      self.strikes = 0
      self.just_spun = False
      print(f"t={d.time:6.1f}s  CHARGE -> EXPLORE  (recharged, map unfinished)")
    else:
      self.state = "DONE"
      print(f"t={d.time:6.1f}s  CHARGE -> DONE  (recharged; map complete)")
    return 0.0, 0.0

  # ---- the loop ------------------------------------------------------------

  def run(self) -> None:
    phases = {"EXPLORE": self.explore, "GO_CHARGE": self.go_charge,
              "FACE_OUTLET": self.face_outlet, "DOCK": self.dock,
              "CHARGE": self.charge}
    viewer = None if self.headless else mujoco.viewer.launch_passive(self.model, self.data)
    try:
      while ((viewer is None or viewer.is_running())
             and self.data.time < self.max_sim_time
             and self.state != "DONE"):
        wall_start = time.time()

        self.perceive()
        if self.battery.empty:
          # An honest failure mode: a robot that budgets badly dies mid-room.
          print(f"t={self.data.time:6.1f}s  BATTERY DEAD in {self.state} -- "
                "mission over")
          self.explore_done_reason = "battery-dead"
          self.state = "DONE"
          break
        if self.data.time < self.backoff_until:
          v, w = BACKOFF_SPEED, 0.0    # reflex pre-empts whatever the mission wanted
        else:
          v, w = phases[self.state]()
        self.actuate(v, w)

        mujoco.mj_step(self.model, self.data)
        self.step_count += 1
        for i in range(self.data.ncon):
          c = self.data.contact[i]
          if self.chassis_gid in (c.geom1, c.geom2):
            self.collision_steps += 1
            break

        if viewer is not None:
          viewer.sync()
          leftover = self.model.opt.timestep - (time.time() - wall_start)
          if leftover > 0:
            time.sleep(leftover)
        self.report()
    finally:
      if viewer is not None:
        viewer.close()
    self.summarize()

  def report(self) -> None:
    """Periodic map dump and telemetry line."""
    t = self.data.time
    if t - self.last_save >= MAP_SAVE_PERIOD:
      self.last_save = t
      map_img = render_map(self.grid, self.pose, self.waypoints,
                           self.landmarks.landmarks)
      Image.fromarray(map_img).save("map.png")
      if self.dashboard is not None:
        self.dashboard.save(self.data, map_img)
    if self.headless and t - self.last_report >= 30.0:
      self.last_report = t
      known = int(np.count_nonzero(np.abs(self.grid.grid) > 0.5))
      print(f"t={t:6.1f}s  {self.state:<11s} mode={self.mode:<5s} "
            f"known cells={known}  outlets={len(self.landmarks.confirmed())}  "
            f"battery={self.battery.fraction:.0%} ({self.battery.last_power_w:+.1f} W)")

  def true_axle_pose(self) -> tuple[float, float, float]:
    """Ground-truth (x, y, yaw) of the axle midpoint, from qpos rather than
    odometry. qpos tracks the body origin 8 cm ahead of the axle."""
    x, y = float(self.data.qpos[0]), float(self.data.qpos[1])
    w, qx, qy, qz = (float(v) for v in self.data.qpos[3:7])
    yaw = math.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return x - 0.08 * math.cos(yaw), y - 0.08 * math.sin(yaw), yaw

  def dock_report(self) -> None:
    """Score the parked pose against ground truth, in the terms the docking
    spike measured: how far off the socket axis, and how badly rotated.

    Everything upstream (detection, projection, odometry) is believed data;
    this is the only place the true answer is consulted.
    """
    if self.target is None:
      return
    if self.docked:
      face = self.data.site_xpos[self.model.site("plug_face").id]
      best = min(("socket_a", "socket_b", "socket_c"), key=lambda n: sum(
        (self.model.body(n).pos[k] - face[k]) ** 2 for k in range(3)))
      bid = self.model.body(best).id
      sp = self.data.xpos[bid]
      mat = self.data.xmat[bid]
      nx, ny = float(mat[0]), float(mat[3])       # socket +x = outward normal
      depth = -((face[0] - sp[0]) * nx + (face[1] - sp[1]) * ny)
      lat = abs(-(face[0] - sp[0]) * ny + (face[1] - sp[1]) * nx)
      if depth > -0.05:
        # Only meaningful while still plugged in -- after a successful charge
        # the robot has undocked and the face is wherever life took it.
        print(f"\nDOCK truth vs {best}: face {depth * 1000:+.1f} mm into the recess "
              f"(floor at +17), lateral {lat * 1000:.1f} mm, height err "
              f"{(face[2] - sp[2]) * 1000:+.1f} mm")
    names = ("outlet_a", "outlet_b", "outlet_c")
    name = min(names, key=lambda n: math.hypot(
      self.model.body(n).pos[0] - self.target.x,
      self.model.body(n).pos[1] - self.target.y))
    bid = self.model.body(name).id
    ox, oy = float(self.data.xpos[bid][0]), float(self.data.xpos[bid][1])
    # The outlet body's local +x is its outward normal (see room_1.xml).
    nx, ny = float(self.data.xmat[bid][0]), float(self.data.xmat[bid][3])

    rx, ry, ryaw = self.true_axle_pose()
    dx, dy = rx - ox, ry - oy
    along = dx * nx + dy * ny            # true standoff distance from the wall
    lateral = -dx * ny + dy * nx         # offset along the wall (the miss)
    yaw_err = wrap_angle(math.atan2(-ny, -nx) - ryaw)

    print(f"\ndocking hand-off pose (truth, vs {name}):")
    print(f"  standoff from wall : {along * 100:6.1f} cm")
    print(f"  lateral offset     : {lateral * 100:+6.1f} cm   (spike budget at "
          f"contact: +/-0.3 cm)")
    print(f"  yaw error          : {math.degrees(yaw_err):+6.2f} deg  (spike budget: "
          f"+/-3 deg)")
    # Attribute that yaw error to its three independent sources, so a bad
    # number points at the subsystem to fix instead of at "something drifted".
    if self.final_heading_error is not None:
      _, _, want = self.standoff_of(self.target)
      odom_drift = wrap_angle(self.reckoner.theta - ryaw)
      dir_err = wrap_angle(want - math.atan2(-ny, -nx))
      print("  yaw error breakdown:")
      print(f"    controller settle  : {math.degrees(self.final_heading_error):+6.2f} deg"
            "  (robot vs its own target heading)")
      print(f"    odometry drift     : {math.degrees(odom_drift):+6.2f} deg"
            "  (believed heading vs true)")
      print(f"    standoff direction : {math.degrees(dir_err):+6.2f} deg"
            "  (seen-from estimate vs true wall normal)")

  def summarize(self) -> None:
    map_img = render_map(self.grid, self.pose, [], self.landmarks.landmarks)
    Image.fromarray(map_img).save("map.png")
    if self.dashboard is not None:
      self.dashboard.save(self.data, map_img)
    known = int(np.count_nonzero(np.abs(self.grid.grid) > 0.5))
    print(f"\nended after {self.data.time:.1f} sim-seconds in state {self.state}"
          f" (explore: {self.explore_done_reason})")
    print(f"battery: {self.battery.fraction:.0%} of {self.battery.capacity_wh:.2f} Wh"
          f"  (charged this mission: {self.charged_once})")
    print(f"known cells: {known}   frontiers blacklisted: {len(self.blacklist)}")
    print(f"chassis-contact steps (should be 0): {self.collision_steps}")
    # Every landmark, not just the confirmed ones: the tentative ones are on
    # map.png in olive, and they are usually the detector's false positives
    # being filtered by the sighting threshold -- worth seeing.
    confirmed = set(id(lm) for lm in self.landmarks.confirmed())
    for lm in self.landmarks.landmarks:
      if id(lm) in confirmed:
        sx, sy, sh = self.standoff_of(lm)
        print(f"  outlet    ({lm.x:+.2f}, {lm.y:+.2f}, z={lm.z:.2f}) seen x{lm.n_sightings:<3d}"
              f" standoff ({sx:+.2f}, {sy:+.2f}) hdg {math.degrees(sh):+.0f} deg")
      else:
        print(f"  tentative ({lm.x:+.2f}, {lm.y:+.2f}, z={lm.z:.2f}) seen x{lm.n_sightings:<3d}"
              f" not trusted (needs 3 sightings)")
    self.dock_report()
    print("final map saved to map.png")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--headless", action="store_true", help="no viewer, run at full speed")
  parser.add_argument("--max-sim-time", type=float, default=600.0, help="sim-seconds budget")
  parser.add_argument("--explore-budget", type=float, default=None,
                      help="optional timer override; default: the battery decides")
  parser.add_argument("--battery-wh", type=float, default=DEMO_CAPACITY_WH,
                      help="battery capacity (demo cell by default; 55.5 = the "
                           "real 3S 5Ah pack, which takes hours of sim to drain)")
  parser.add_argument("--weights", default=None, help="YOLO weights; default: newest under runs/")
  parser.add_argument("--views", action="store_true",
                      help="also save views.png: stereo pair + map + dock camera")
  args = parser.parse_args()
  Lifecycle(
    headless=args.headless, max_sim_time=args.max_sim_time,
    weights=args.weights or latest_weights(), explore_budget=args.explore_budget,
    views=args.views, battery_wh=args.battery_wh,
  ).run()
