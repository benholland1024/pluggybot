"""The wheeled rover as a `Body` (issue #380; `docs/Rover.md` is the rover).

Nothing here decides anything: every member hands the call to the mission
stack the rover always was -- `HubMission`, its `HubSwap`, the coupling's
electrical criteria -- by name at CALL time, so a test that stubs a rover
routine (`life.body.mission.drive_to_routine = ...`) stubs what the body
runs, and the day flown through the interface is the day flown before it
(`scripts/determinism_spike.py --compare`). Deleted with the rover (#376,
stage C).
"""

from pluggybot.behavior.navigation import plan
from pluggybot.body import TOPPLE_HOLD_S, Body
from pluggybot.mission.mission import (
  CHARGE_APPROACH_MAX, CHARGE_CREEP, CHARGE_PRESS, MAP_TILT_RAD, UNDOCK_REVERSE,
  HubMission, bay_standoff, charge_standoff, charge_trace, gave_up, swap_trace,
)
from pluggybot.rack.coupling import module_power_contact, rack_charge_contact
from pluggybot.robot import FIRST, RobotHandle
from pluggybot.telemetry.protocol import ROBOT_ROOT, robot_roots
from pluggybot.tick import Routine

#: How long a charge contact that dropped is pressed for again, s.
REDOCK_S = 1.0


class RoverBody(Body):
  """The rover: `HubMission` and everything it owns, as a `Body`."""

  STILL = (0.0, 0.0)
  level_tilt_rad = MAP_TILT_RAD
  #: The rover has no posture to rest in, and a fall is for somebody else
  #: to right (`body.TOPPLE_HOLD_S` is its whole budget).
  resting = False
  posture = "standing"
  rights_itself = False
  stuck_after_s = TOPPLE_HOLD_S

  def __init__(self, model, data, viewer=None, realtime: bool = True,
               rack=None, grid_bounds=(-3, -3, 7, 7),
               handle: RobotHandle = FIRST) -> None:
    self.mission = HubMission(model, data, viewer=viewer, realtime=realtime,
                              rack=rack, grid_bounds=grid_bounds, handle=handle)

  # ---- which robot, and the seam -------------------------------------------

  handle = property(lambda self: self.mission.handle)
  model = property(lambda self: self.mission.model)
  data = property(lambda self: self.mission.data)
  step_hooks = property(lambda self: self.mission.step_hooks)
  stepper = property(lambda self: self.mission.swap)

  def run(self, routine: Routine, name: str = ""):
    return self.mission.run(routine, name)

  def rebind(self, model, data) -> None:
    self.mission.rebind(model, data)

  def kept_state(self) -> tuple[dict, dict]:
    return self.mission.kept_state()

  def restore_kept(self, state: dict, arrays: dict) -> bool:
    return self.mission.restore_kept(state, arrays)

  def close(self) -> None:
    self.mission.close()

  # ---- going places --------------------------------------------------------

  def start_at(self, x, y, yaw) -> None:
    self.mission.start_at(x, y, yaw)

  def go_to_routine(self, x, y, timeout=None, stop=None) -> Routine:
    # ...forwarding only what was given: `drive_to_routine`'s own default
    # is the rover's, and a stub standing in for it may not take one
    kw = {k: v for k, v in (("timeout", timeout), ("stop", stop)) if v is not None}
    return self.mission.drive_to_routine(x, y, **kw)

  last_drive = property(lambda self: self.mission.last_drive)

  def gave_up(self, record, peer="the other robot") -> str:
    return gave_up(record, peer)

  def face_routine(self, heading) -> Routine:
    return self.mission.face_routine(heading)

  def hold_routine(self, seconds) -> Routine:
    return self.mission._drive_routine(seconds, 0.0, 0.0)

  def look_around_routine(self) -> Routine:
    return self.mission._spin_routine()

  grid = property(lambda self: self.mission.grid)

  def plan_frontier(self, blacklist):
    return plan(self.mission.grid, self.mission.pose, blacklist)

  def reachable(self, points) -> list[bool]:
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
    q = self.mission.swap.root_qadr
    return self.data.qpos[q:q + 2]

  def orientation(self):
    q = self.mission.swap.root_qadr
    return self.data.qpos[q + 3:q + 7]

  def footprint_centre(self):
    return self.mission.footprint_centre()

  def as_seen(self, x, y):
    return self.mission.as_seen(x, y)

  def level(self) -> bool:
    return self.mission.level()

  # ---- the rack ------------------------------------------------------------

  rack = property(lambda self: self.mission.rack)
  rack_prior = property(lambda self: self.mission.rack_prior)
  rack_discovered = property(lambda self: self.mission.rack_discovered)

  def start_discovery(self) -> None:
    self.mission.start_discovery()

  def refresh_rack(self):
    return self.mission.refresh_rack()

  def bay_standoff(self, station_y):
    return bay_standoff(station_y, self.mission.rack)

  def fetch_tool_routine(self, station_y, module) -> Routine:
    return self.mission.swap_at_bay_routine(station_y, "pick", module=module)

  def stow_tool_routine(self, station_y, module) -> Routine:
    return self.mission.swap_at_bay_routine(station_y, "return", module=module)

  def module_state(self, module) -> dict:
    return self.mission.swap.module_state(module)

  def tool_powered(self, module) -> bool:
    return module_power_contact(self.model, self.data, module,
                                self.mission.handle.prefix)

  def seated_on(self, module):
    for root in robot_roots(self.model):
      if module_power_contact(self.model, self.data, module,
                              root[:-len(ROBOT_ROOT)]):
        return root
    return None

  def tool(self, module, **kw):
    from pluggybot.tools.drawing import PEN_MODULE, PenPlotter
    from pluggybot.tools.gripper import CLAW_MODULE, ClawTool
    if module == CLAW_MODULE:
      return ClawTool(self.model, self.data, self.mission.swap, **kw)
    if module == PEN_MODULE:
      return PenPlotter(self.model, self.data, self.mission.swap, **kw)
    return None

  swapping_at = property(lambda self: self.mission.swapping_at)
  peer_at_bay_m = property(lambda self: self.mission.peer_at_bay_m)

  @property
  def bay_wait(self):
    return self.mission.bay_wait

  @bay_wait.setter
  def bay_wait(self, routine) -> None:
    self.mission.bay_wait = routine

  def swap_trace(self) -> str:
    return swap_trace(self.mission.last_swap)

  # ---- the dock ------------------------------------------------------------

  def charge_standoff(self):
    return charge_standoff(self.mission.rack)

  def dock_routine(self) -> Routine:
    return self.mission.charge_approach_routine(CHARGE_APPROACH_MAX, CHARGE_CREEP)

  def charging(self) -> bool:
    return rack_charge_contact(self.model, self.data, self.mission.handle.prefix)

  @property
  def docked(self) -> bool:
    return self.mission.swap.pinned

  @docked.setter
  def docked(self, held: bool) -> None:
    self.mission.swap.pinned = held

  def anchor_at_dock(self) -> None:
    self.mission.anchor_at_dock()

  def dock_hold_routine(self, seconds) -> Routine:
    return self.mission._drive_routine(seconds, CHARGE_PRESS, 0.0)

  def redock_routine(self) -> Routine:
    return self.mission._drive_routine(REDOCK_S, CHARGE_CREEP, 0.0)

  def undock_routine(self) -> Routine:
    return self.mission.swap._drive_until_routine(UNDOCK_REVERSE, -0.08,
                                                  stall_stop=False)

  def charge_trace(self) -> str:
    return charge_trace(self.mission.last_charge)

  # ---- senses --------------------------------------------------------------

  def pack(self, capacity_wh, charge_scale):
    from pluggybot.power import Battery
    return Battery(self.model, capacity_wh=capacity_wh,
                   charge_scale=charge_scale, prefix=self.mission.handle.prefix)

  def depth_camera(self):
    from pluggybot.perception.depth import DepthCamera
    return DepthCamera(self.model, handle=self.mission.handle)

  def know_peer(self, root) -> None:
    self.mission.lidar.exclude_robot(root)

  def detect_tags(self) -> dict:
    return self.mission.tags.detect(self.data)

  def spot(self, tag, at_height=None):
    return self.mission.spot(tag, at_height=at_height)

  pressing = property(lambda self: self.mission.swap.pressing)
  head_camera = property(lambda self: self.handle.el("left_eye"))

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
  press_steps = property(lambda self: self.mission.swap.press_steps)

  # ---- the motor level -----------------------------------------------------

  def actuator(self, name) -> int:
    return self.model.actuator(self.mission.handle.el(name)).id

  def ramp_routine(self, act, target, speed, settle=0.0) -> Routine:
    return self.mission.swap.ramp_routine(act, target, speed, settle=settle)

  def settle_routine(self, seconds) -> Routine:
    return self.mission.swap._run_routine(seconds, 0.0)

  def velocity_routine(self, seconds, v, w) -> Routine:
    return self.mission._drive_routine(seconds, v, w)

  def travel_routine(self, distance, v) -> Routine:
    return self.mission.swap._drive_until_routine(distance, v, stall_stop=False)

  def retract_arm_routine(self) -> Routine:
    return self.mission.set_arm_routine(0.0)

  # ---- posture and rest ----------------------------------------------------

  def rest_routine(self) -> Routine:
    return
    yield  # the rover has no posture: a routine that steps nothing

  def stand_routine(self) -> Routine:
    return
    yield
