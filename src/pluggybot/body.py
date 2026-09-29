"""What the day asks of a body (issue #380).

The robot's life -- the day loop (`lifecycle.py`), its mind, its economy,
its record -- is the same whatever it lives in; the machine is not. `Body`
is everything the loop and the procedure layer (`procedure/`) may ask of
the machine, and the ONLY way they reach it: the wheeled rover implements
it (`mission/rover.py`), the quadruped will (#375), and `StubBody` below is
the smallest thing that does, for a test that needs the lifecycle's
bookkeeping and no physics. `tests/test_body.py` walks the syntax tree of
every module on this side of the seam and fails on a reach that is not a
member here. `robot.RobotHandle` still resolves a robot's names in the
world; the body sits beside it (`Body.handle`), not in place of it.

What a body does, in the issue's words: it GOES to a place and faces a
bearing; it knows WHERE it is; it takes a TOOL from a bay and hangs it
back; it DOCKS to charge and undocks; it SENSES (a map of the floor, a
depth frame, its cameras, the other robots); it has a POSTURE it can rest
in. Under those, the MOTOR LEVEL a procedure is built from (issue #166):
an actuator walked to a setpoint, the base at a velocity, a tool's driver.

⚠ THE COMMAND IS THE BODY'S (the tick contract, `tick.py`). A routine
yields one command per physics step, and only the body that yields it
reads it: `stepper.apply(command)` turns it into actuator setpoints. The
rover's is `(v, w)`, forward speed and yaw rate. A legged body's will be
its velocity (forward, sideways, yaw rate) with a posture (height, pitch,
roll), the input its walking policy tracks (#377 decides). So code on this
side of the seam never builds a command: it composes the body's routines,
and where it must hold one physics step still it yields `STILL`.

⚠ A BODY BELIEVES; THE SIM KNOWS. `pose` is the body's own estimate (dead
reckoning on the rover). `true_pose`, `root_xy`, `orientation` and
`footprint_centre` read the world: the sim's own checks (a death's record,
a failed swap's trace), and the few acts a sensor could equally make (an
IMU's attitude; a robot lying down seen as a lump in a depth image).

⚠ A ROUTINE CALL IS NOTHING UNTIL IT IS DRIVEN (`tick.py`): every
`*_routine` member is a generator, composed with `yield from`.
"""

from __future__ import annotations

import abc
import math
from typing import Any, NamedTuple

import mujoco
import numpy as np

from pluggybot import tick
from pluggybot.robot import FIRST, RobotHandle
from pluggybot.tick import Routine


#: Torso tilt from upright that counts as knocked over (issue #107), and
#: how long it has to hold: a wheel riding a threshold tips the body for a
#: moment and recovers, a robot on its side does not. 60 deg is past any
#: pose the rover's drive can right itself from, and the quadruped's
#: get-up takes over at the same angle (`legs.posture.FALL_TILT_RAD`).
TOPPLE_TILT_RAD = math.radians(60.0)
TOPPLE_HOLD_S = 2.0


class KeepClear(NamedTuple):
  """Another robot, as one of `Body.others` answers it: where to keep clear
  of, and whether it is LYING DOWN (issue #365) -- a disc round its body,
  and nothing to wait for. A bare `(x, y)` is a robot standing."""
  x: float
  y: float
  down: bool = False


class Body(abc.ABC):
  """What the day asks of a body. See the module docstring."""

  # ---- which robot, and the seam -------------------------------------------

  #: This robot's names in the world (`robot.RobotHandle`).
  handle: RobotHandle
  #: The command that holds this body where it is (the rover's `(0, 0)`).
  STILL: Any
  #: The tilt from upright (rad) under which this body counts as level --
  #: the rover's map gate (`mission.MAP_TILT_RAD`); a lean below it has no
  #: direction worth reporting.
  level_tilt_rad: float

  @property
  @abc.abstractmethod
  def model(self) -> mujoco.MjModel:
    """The world this body lives in, rebound on a recompile."""

  @property
  @abc.abstractmethod
  def data(self) -> mujoco.MjData:
    """...and its state."""

  @property
  @abc.abstractmethod
  def step_hooks(self) -> list:
    """Callables run once after every physics step, whatever is driving:
    the lifecycle's seams (power, deaths, the event map) hang here."""

  @property
  @abc.abstractmethod
  def stepper(self):
    """What steps this body's physics (`tick.run`, `tick.run_many`):
    `apply(command)` before the world steps, `after_step()` after,
    `step(command)` for one robot alone, `STILL`, `model` / `data`."""

  @abc.abstractmethod
  def run(self, routine: Routine, name: str = "") -> Any:
    """Drive a routine to completion on this body's physics, returning what
    it returned: the blocking twin of `yield from`."""

  @abc.abstractmethod
  def rebind(self, model, data) -> None:
    """Point everything at a recompiled world (issue #168): state stays,
    ids are re-resolved by name."""

  @abc.abstractmethod
  def kept_state(self) -> tuple[dict, dict]:
    """What this body believes, for a restart (issue #345): JSON and
    arrays."""

  @abc.abstractmethod
  def restore_kept(self, state: dict, arrays: dict) -> bool:
    """Put `kept_state` back; True if its map fitted this build."""

  @abc.abstractmethod
  def close(self) -> None:
    """Release what it holds (renderers); its beliefs stay readable."""

  # ---- going places --------------------------------------------------------

  @abc.abstractmethod
  def start_at(self, x: float, y: float, yaw: float) -> None:
    """Put the body upright at a pose and tell its estimate so: a mission
    start, and a stand-up (issue #143). ⚠ A step taken here is one no other
    robot's stepper or hooks see (a stand-up lands mid-loop): the rover
    settles a second, the quadruped steps nothing (issue #387)."""

  @abc.abstractmethod
  def go_to_routine(self, x: float, y: float, timeout: float = 90.0,
                    stop=None) -> Routine:
    """Go to a world point over the map, round the other robots; True on
    arrival. Why it did not arrive is `last_drive`. The rover drives it
    (`HubMission.drive_to_routine`); a legged body walks it, through floor
    it has not mapped as well (issue #381). `timeout` is its patience;
    `stop`, a callable asked every second on the way, ends it where it
    stands when it answers True (`navigator.DRIVE_STOPPED`)."""

  #: How the last `go_to_routine` ended (issue #350): `why` ("" arrived,
  #: else the body's cause), `goal`, `seconds`, `shortM`, and for a peer
  #: `peerAt` / `peerM` / `peerXY` / `peerDown`. None before any.
  last_drive: dict | None

  @abc.abstractmethod
  def gave_up(self, record: dict, peer: str = "the other robot") -> str:
    """`last_drive` as the clause a failure line ends with (issue #350);
    `peer` names the robot a `peer` cause was about."""

  @abc.abstractmethod
  def face_routine(self, heading: float) -> Routine:
    """Turn where it stands to a heading; False if its budget ran out."""

  @abc.abstractmethod
  def hold_routine(self, seconds: float) -> Routine:
    """Stand still for `seconds`, its senses running (the map, the rack
    look, the collision count): waiting, thinking, idling."""

  @abc.abstractmethod
  def look_around_routine(self) -> Routine:
    """Look all round where it stands, to seed the map and buy sight lines
    (the rover's 360 spin)."""

  #: Its map of the floor (`mapping.OccupancyGrid`): what it plans over,
  #: what exploring plans frontiers on, what a census counts off.
  grid: Any

  @abc.abstractmethod
  def plan_frontier(self, blacklist: set) -> tuple[list | None, str]:
    """The explore's next leg (`behavior.navigation.plan`): a cell path to
    the nearest frontier it can reach over ITS map, at its inflation, and a
    status. `blacklist` is the frontiers given up on this explore."""

  @abc.abstractmethod
  def reachable(self, points) -> list[bool]:
    """Which world points it could plan to right now (issue #346)."""

  @abc.abstractmethod
  def in_sight(self, x: float, y: float) -> bool:
    """Can one `go_to_routine` plan to (x, y): in its sensors' reach and on
    its map (issue #353)?"""

  # ---- where it is ---------------------------------------------------------

  @property
  @abc.abstractmethod
  def pose(self) -> tuple[float, float, float]:
    """Where it BELIEVES it is: (x, y, heading), its own estimate."""

  @abc.abstractmethod
  def pose_xy(self) -> tuple[float, float]:
    """(x, y) of `pose`: where it SAYS it is -- what another robot may be
    told."""

  @abc.abstractmethod
  def true_pose(self) -> tuple[float, float, float]:
    """Where it IS, in `pose`'s terms, off the world: the sim's own checks,
    never an act."""

  @abc.abstractmethod
  def root_xy(self):
    """Its root body's true (x, y): what distances between robots are
    measured between in the sim's own records."""

  @abc.abstractmethod
  def orientation(self):
    """Its root's attitude, (w, x, y, z) -- what an IMU reads: the tilt a
    death or a lean is judged on."""

  @abc.abstractmethod
  def footprint_centre(self) -> tuple[float, float]:
    """The middle of the floor its body covers, off the true geometry --
    read to act only for a robot lying down (issue #365)."""

  @abc.abstractmethod
  def as_seen(self, x: float, y: float) -> tuple[float, float]:
    """A true world point where its own sensors would place it: seen from
    where it stands, placed through where it believes it stands."""

  @abc.abstractmethod
  def level(self) -> bool:
    """Level enough for a reading to be a map of the room (issue #339)."""

  # ---- the rack: a tool from a bay, and back -------------------------------

  #: What it believes about the rack (`rack.localize.RackPose`).
  rack: Any
  #: ...and the COMMISSIONED pose: the dock that defines its map frame
  #: (issue #42), where `anchor_at_dock` snaps its estimate.
  rack_prior: Any
  #: Has it confirmed the rack by its tag?
  rack_discovered: bool

  @abc.abstractmethod
  def start_discovery(self) -> None:
    """Begin watching for the rack's tag through everything it does."""

  @abc.abstractmethod
  def refresh_rack(self):
    """Adopt the rack pose it has confirmed, if it has: the pose, or None."""

  @abc.abstractmethod
  def bay_standoff(self, station_y: float) -> tuple[float, float, float]:
    """Where it stands to work the bay at `station_y` -- (x, y, heading)
    off the rack it believes in."""

  @abc.abstractmethod
  def fetch_tool_routine(self, station_y: float, module: str) -> Routine:
    """Go to a bay and take `module` onto its coupling, verified and
    retried; returns the approach's answer. Whether it holds it is
    `module_state` / `tool_powered`."""

  @abc.abstractmethod
  def stow_tool_routine(self, station_y: float, module: str) -> Routine:
    """Go to a bay and hang `module` back; returns the approach's answer.
    Whether it hangs is `module_state`."""

  @abc.abstractmethod
  def module_state(self, module: str) -> dict:
    """Where a module is, off the world: `on_fork` (riding THIS body's
    coupling), `hung` (in some bay), `bay` (the nearest bay's index)."""

  @abc.abstractmethod
  def tool_powered(self, module: str) -> bool:
    """`module` seated on THIS body's coupling, and conducting."""

  @abc.abstractmethod
  def seated_on(self, module: str) -> str | None:
    """Which robot in this world has `module` seated on its coupling, by
    root, in model order -- or None. The module is the world's."""

  @abc.abstractmethod
  def tool(self, module: str, **kw):
    """The driver that works `module` from this body (the rover's claw and
    pen, `tools/gripper.py`, `tools/drawing.py`), or None for a module it
    has no driver for. A new one every call."""

  #: The bay a swap is working at, by station y, from its first drive to
  #: its verdict (issue #347); None otherwise.
  swapping_at: float | None
  #: A manoeuvre of its reach under way that no file holds (issue #405: the
  #: quadruped's swap and arm moves, a fork half under a peg): a restart's
  #: save waits it out (`continuation.Keeper.busy`). The rover's is False.
  working: bool
  #: How far off the robot was that made the last swap give its bay up
  #: (issue #313), or None.
  peer_at_bay_m: float | None
  #: WHO WAITS FOR A TAKEN BAY (issue #346): the lifecycle hangs its
  #: routine here, `(sx, sy, kind, since) -> True once free`.
  bay_wait: Any

  @abc.abstractmethod
  def swap_trace(self) -> str:
    """One line of what the last swap did, for a failed one's log."""

  # ---- the dock ------------------------------------------------------------

  @abc.abstractmethod
  def charge_standoff(self) -> tuple[float, float, float]:
    """Where it stands to begin docking -- (x, y, heading) off the rack it
    believes in."""

  @abc.abstractmethod
  def dock_routine(self) -> Routine:
    """From the charge standoff, onto the charger until its contacts
    conduct; returns the approach's answer. Success is `charging()`."""

  @abc.abstractmethod
  def charging(self) -> bool:
    """Its charge contacts conduct: the dock's ELECTRICAL criterion."""

  #: Held on the charger: nothing it does while set is counted as travel
  #: (the rover's reckoner holds its position through the press, #94).
  docked: bool

  @abc.abstractmethod
  def anchor_at_dock(self) -> None:
    """Snap its estimate to the COMMISSIONED dock pose (issue #42): the
    one pose it occupies to millimetres by construction."""

  @abc.abstractmethod
  def dock_hold_routine(self, seconds: float) -> Routine:
    """Stay on the charger for `seconds`, keeping contact."""

  @abc.abstractmethod
  def redock_routine(self) -> Routine:
    """Contact dropped: try once to make it again."""

  @abc.abstractmethod
  def undock_routine(self) -> Routine:
    """Leave the charger."""

  @abc.abstractmethod
  def charge_trace(self) -> str:
    """One line of what the last dock approach saw, for a failed one's log."""

  # ---- senses --------------------------------------------------------------

  @abc.abstractmethod
  def pack(self, capacity_wh: float, charge_scale: float):
    """Its pack (`power.Pack`): the energy book is every body's, what it
    draws each step is this body's."""

  @abc.abstractmethod
  def depth_camera(self):
    """Its near-field depth camera (`perception.depth.DepthCamera`), built,
    or None where it has none (issue #34)."""

  @abc.abstractmethod
  def know_peer(self, root: str) -> None:
    """Another robot is in the world: its body goes to the sensors' PEER
    channel -- in the drive, never in the map (issue #316)."""

  @abc.abstractmethod
  def detect_tags(self) -> dict:
    """One tag decode from its working camera: `{id: {t, yaw, center}}`."""

  @abc.abstractmethod
  def spot(self, tag: int, at_height: float | None = None) -> dict | None:
    """One tag's centre in the believed world frame off one decode, or None
    where it does not decode from here (issue #264)."""

  #: Pressed against something it is moving into (the rover's bumper).
  pressing: bool
  #: The camera its eye looks through (issue #275), by its name in the
  #: world: the one on the head, looking along the body's +x -- the rover's
  #: `left_eye`, the quadruped's `nav_eye` -- or None where it has none (the
  #: stub). The body's, not the eye's: the rover's name kept in the loop
  #: took the served process down on legs (issue #408).
  head_camera: str | None

  # ---- the others, and collisions ------------------------------------------

  #: THE OTHER ROBOTS (issue #167): callables answering each one's reported
  #: (x, y), or a `KeepClear`; set by the pair.
  others: list

  @abc.abstractmethod
  def peer_on_the_goal(self, x: float, y: float) -> float | None:
    """How far off the nearest robot standing ON this goal is -- near
    enough that arrival there is impossible -- or None (issue #313)."""

  @abc.abstractmethod
  def watch_for_peers(self, points) -> float | None:
    """One depth frame's peer channel, recorded for the drive (issue
    #328): the nearest body in the corridor ahead, or None."""

  #: This body's geoms: what another robot's sensors attribute to it.
  geom_ids: Any
  #: For the day's summary: drives that held for a peer ahead (issue #328),
  peer_holds: int
  #: ...physics steps in a contact it did not mean (a charge press is meant),
  collision_steps: int
  #: ...and physics steps pressing against something (issue #94).
  press_steps: int

  # ---- the motor level (what a procedure is built from, issue #166) -------

  @abc.abstractmethod
  def actuator(self, name: str) -> int:
    """One of this body's actuators, by its bare name: its id."""

  @abc.abstractmethod
  def ramp_routine(self, act: int, target: float, speed: float,
                   settle: float = 0.0) -> Routine:
    """Walk one actuator's setpoint to `target` at `speed`, then settle --
    the ramping rule (CLAUDE.md), inside the primitive."""

  @abc.abstractmethod
  def setpoint(self, act: int) -> float:
    """What one of its actuators is commanded to, in its axis's units: a
    position servo's `ctrl` (the rover's), or the joint angle a controller
    turns into torque (the quadruped's arm, issue #405) -- never a torque."""

  @abc.abstractmethod
  def settle_routine(self, seconds: float) -> Routine:
    """Stand still while a mechanism comes to rest: the physics and the
    step hooks, none of `hold_routine`'s senses."""

  @abc.abstractmethod
  def velocity_routine(self, seconds: float, v: float, w: float) -> Routine:
    """The base at forward speed `v` and yaw rate `w` for `seconds`, its
    senses running -- the `drive` verb, a dance move."""

  @abc.abstractmethod
  def travel_routine(self, distance: float, v: float) -> Routine:
    """`distance` of believed travel along its heading at `v` (negative
    backs up); returns why it stopped."""

  @abc.abstractmethod
  def retract_arm_routine(self) -> Routine:
    """Its reach into the driving configuration (the rover's fork arm in)."""

  # ---- posture and rest ----------------------------------------------------

  #: Lying down to rest. Always False on the rover, which has no posture.
  resting: bool
  #: Its posture, as the wire carries it (issue #387): `standing`,
  #: `lying_down`, `lying`, `standing_up` or `getting_up` -- the rover always
  #: stands. Lying down to rest is a posture, never a fall.
  posture: str
  #: Whether it gets itself up from a fall (issue #387): the quadruped's
  #: get-up policy does; the rover waits for somebody.
  rights_itself: bool
  #: How long a fall may last before it is the `stuck` death, s: the
  #: rover's `lifecycle.TOPPLE_HOLD_S`, a body that rights itself its
  #: get-up's MEASURED budget.
  stuck_after_s: float

  @abc.abstractmethod
  def rest_routine(self) -> Routine:
    """Into its resting posture; a no-op on the rover. Who calls it -- code
    or the agent -- is #377's to decide."""

  @abc.abstractmethod
  def stand_routine(self) -> Routine:
    """Out of it; a no-op on the rover."""


def members() -> frozenset[str]:
  """Every name the interface declares: what `tests/test_body.py` lets a
  module on the loop's side of the seam read off a body."""
  return frozenset({n for n in vars(Body) if not n.startswith("_")}
                   | set(Body.__annotations__))


def body_for(model, data, handle: RobotHandle = FIRST, **kw) -> Body:
  """The body this robot IS, in this world: the one choice a lifecycle
  built without one makes (`HubLifecycle(body=None)`), by what the model
  carries under this robot's names -- legs (`legs/body.py`, issue #387) or
  the rover (`mission/rover.py`). `kw` is its navigation: the viewer,
  pacing, rack prior and map bounds."""
  if is_quadruped(model, handle):
    from pluggybot.legs.body import QuadBody
    return QuadBody(model, data, handle=handle, **kw)
  from pluggybot.mission.rover import RoverBody
  return RoverBody(model, data, handle=handle, **kw)


def is_quadruped(model, handle: RobotHandle = FIRST) -> bool:
  """Does this robot walk? Its first leg's hip, by name."""
  return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                           handle.el("FL_hip_abd")) >= 0


# ---- the stub: a body with no physics ---------------------------------------

#: The stub's world: a floor and nothing else. It compiles in a millisecond,
#: steps in microseconds, and carries no robot to be deleted with the rover.
STUB_WORLD = """<mujoco model="stub">
  <worldbody><geom name="floor" type="plane" size="20 20 0.1"/></worldbody>
</mujoco>"""


class _StubStepper:
  """Steps the stub's world: no actuators, one `mj_step`, the hooks."""

  STILL = ()

  def __init__(self, body: "StubBody") -> None:
    self.body = body

  @property
  def model(self):
    return self.body.model

  @property
  def data(self):
    return self.body.data

  def apply(self, command) -> None:
    pass

  def after_step(self) -> None:
    for hook in self.body.step_hooks:
      hook()

  def step(self, command) -> None:
    self.apply(command)
    mujoco.mj_step(self.body.model, self.body.data)
    self.after_step()


class StubBody(Body):
  """A body that does nothing physical (issue #380). Every manoeuvre
  arrives at once, as asked; every sensor answers what the test set:
  `holding` is the module on its coupling, `on_charger` what `charging`
  says, `attitude` its IMU. Time passes only where a routine waits
  (`hold_routine` and its kin step the stub's world), so the lifecycle's
  seams tick as they do on a real body. What a test that needs the
  lifecycle's BOOKKEEPING -- the mind, the economy, the record -- builds it
  on, instead of the rover: `HubLifecycle(*StubBody.world(),
  body=StubBody(...))`."""

  STILL = ()
  level_tilt_rad = math.radians(1.5)
  posture = "standing"
  rights_itself = False
  stuck_after_s = 2.0
  head_camera = None

  @staticmethod
  def world():
    """A fresh (model, data) of `STUB_WORLD`."""
    model = mujoco.MjModel.from_xml_string(STUB_WORLD)
    return model, mujoco.MjData(model)

  def __init__(self, model=None, data=None, handle: RobotHandle = FIRST,
               pose: tuple[float, float, float] = (0.0, 0.0, 0.0),
               rack=None, grid_bounds=(-3.0, -3.0, 7.0, 7.0),
               bays: dict[str, int] | None = None, draw_w: float = 0.0) -> None:
    from pluggybot.mapping.occupancy_grid import OccupancyGrid
    from pluggybot.procedure.steps import TOOL_BAYS
    from pluggybot.rack.localize import RackPose
    if model is None:
      model, data = self.world()
    self._model, self._data = model, data
    self.handle = handle
    self._hooks: list = []
    self._stepper = _StubStepper(self)
    self.x, self.y, self.theta = (float(v) for v in pose)
    gx0, gy0, gx1, gy1 = grid_bounds
    self.grid = OccupancyGrid(x_min=gx0, y_min=gy0, x_max=gx1, y_max=gy1,
                              resolution=0.05)
    self.rack = self.rack_prior = rack or RackPose.prior()
    self.rack_discovered = False
    self.bays = dict(TOOL_BAYS if bays is None else bays)
    self.draw_w = float(draw_w)
    #: what the test set: the module on its coupling, the charger's
    #: contact, the root's attitude
    self.holding: str | None = None
    self.on_charger = False
    self.attitude = (1.0, 0.0, 0.0, 0.0)
    #: every place it was sent, in order
    self.went: list[tuple[float, float]] = []
    self.last_drive = None
    self.swapping_at = self.peer_at_bay_m = None
    self.working = False
    self.bay_wait = None
    self.docked = self.pressing = self.resting = False
    self.others = []
    self.geom_ids = np.zeros(0, dtype=np.int32)
    self.peer_holds = self.collision_steps = self.press_steps = 0

  model = property(lambda self: self._model)
  data = property(lambda self: self._data)
  step_hooks = property(lambda self: self._hooks)
  stepper = property(lambda self: self._stepper)

  def run(self, routine, name=""):
    return tick.run(self._stepper, routine, name)

  def rebind(self, model, data) -> None:
    self._model, self._data = model, data

  def kept_state(self):
    return {"pose": [self.x, self.y, self.theta], "holding": self.holding}, {}

  def restore_kept(self, state, arrays) -> bool:
    self.x, self.y, self.theta = state["pose"]
    self.holding = state.get("holding")
    return False

  def close(self) -> None:
    pass

  def _wait(self, seconds: float) -> Routine:
    for _ in range(round(seconds / self._model.opt.timestep)):
      yield self.STILL

  def start_at(self, x, y, yaw) -> None:
    self.x, self.y, self.theta = float(x), float(y), float(yaw)

  def go_to_routine(self, x, y, timeout=90.0, stop=None):
    self.went.append((float(x), float(y)))
    self.x, self.y = float(x), float(y)
    self.last_drive = {"why": "", "goal": (float(x), float(y)),
                       "seconds": 0.0, "shortM": 0.0}
    return True
    yield  # a routine that steps nothing

  def gave_up(self, record, peer="the other robot") -> str:
    return f"the drive gave up ({record.get('why')})"

  def face_routine(self, heading):
    self.theta = float(heading)
    return True
    yield

  def hold_routine(self, seconds):
    yield from self._wait(seconds)

  def look_around_routine(self):
    return
    yield

  def plan_frontier(self, blacklist):
    from pluggybot.behavior.navigation import plan
    return plan(self.grid, self.pose, blacklist)

  def reachable(self, points):
    return [True] * len(points)

  def in_sight(self, x, y) -> bool:
    return True

  @property
  def pose(self):
    return self.x, self.y, self.theta

  def pose_xy(self):
    return self.x, self.y

  def true_pose(self):
    return self.pose

  def root_xy(self):
    return np.array([self.x, self.y])

  def orientation(self):
    return self.attitude

  def footprint_centre(self):
    return self.x, self.y

  def as_seen(self, x, y):
    return x, y

  def level(self) -> bool:
    w, x, y, z = self.attitude
    return 1.0 - 2.0 * (x * x + y * y) >= math.cos(self.level_tilt_rad)

  def start_discovery(self) -> None:
    pass

  def refresh_rack(self):
    return None

  def bay_standoff(self, station_y):
    x, y = self.rack.to_world(0.5, station_y)
    return x, y, self.rack.heading

  def fetch_tool_routine(self, station_y, module):
    self.holding = module
    return "arrived"
    yield

  def stow_tool_routine(self, station_y, module):
    if self.holding == module:
      self.holding = None
    return "arrived"
    yield

  def module_state(self, module) -> dict:
    held = module == self.holding
    return {"on_fork": held, "hung": not held, "bay": self.bays.get(module, 0)}

  def tool_powered(self, module) -> bool:
    return module is not None and module == self.holding

  def seated_on(self, module):
    return self.handle.root if self.tool_powered(module) else None

  def tool(self, module, **kw):
    return None

  def swap_trace(self) -> str:
    return "a stub body swaps nothing"

  def charge_standoff(self):
    x, y = self.rack.to_world(0.5, 0.0)
    return x, y, self.rack.heading

  def dock_routine(self):
    self.on_charger = True
    return "stopped"
    yield

  def charging(self) -> bool:
    return self.on_charger

  def anchor_at_dock(self) -> None:
    pass

  def dock_hold_routine(self, seconds):
    yield from self._wait(seconds)

  def redock_routine(self):
    return
    yield

  def undock_routine(self):
    self.on_charger = False
    return "arrived"
    yield

  def charge_trace(self) -> str:
    return "a stub body docks at once"

  def pack(self, capacity_wh, charge_scale):
    from pluggybot.power import Pack
    return Pack(capacity_wh=capacity_wh, charge_scale=charge_scale,
                draw_w=self.draw_w)

  def depth_camera(self):
    return None

  def know_peer(self, root) -> None:
    pass

  def detect_tags(self) -> dict:
    return {}

  def spot(self, tag, at_height=None):
    return None

  def peer_on_the_goal(self, x, y):
    return None

  def watch_for_peers(self, points):
    return None

  def actuator(self, name) -> int:
    raise KeyError(f"a stub body has no actuator {name!r}")

  def ramp_routine(self, act, target, speed, settle=0.0):
    yield from self._wait(settle)

  def setpoint(self, act) -> float:
    raise KeyError(f"a stub body has no actuator {act!r}")

  def settle_routine(self, seconds):
    yield from self._wait(seconds)

  def velocity_routine(self, seconds, v, w):
    yield from self._wait(seconds)

  def travel_routine(self, distance, v):
    return "arrived"
    yield

  def retract_arm_routine(self):
    return
    yield

  def rest_routine(self):
    self.resting = True
    return
    yield

  def stand_routine(self):
    self.resting = False
    return
    yield
