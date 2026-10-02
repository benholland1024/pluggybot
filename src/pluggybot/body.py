"""What the day asks of a body (issue #380).

The robot's life -- the day loop (`lifecycle.py`), its mind, its economy,
its record -- is the same whatever it lives in; the machine is not. `Body`
is everything the loop and the procedure layer (`procedure/`) may ask of
the machine, and the ONLY way they reach it: the quadruped implements it
(`legs/body.py`, #387), and `StubBody` below is the smallest thing that
does, for a test that needs the lifecycle's bookkeeping and no physics. (The
wheeled rover implemented it first, #380, and was deleted in #376;
`rover-final` has it.) `tests/test_body.py` walks the syntax tree of
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
quadruped's is its velocity, `(vx, vy, w)`, the input its walking policy
tracks. So code on this side of the seam never builds a command: it
composes the body's routines, and where it must hold one physics step still
it yields `STILL`.

⚠ A BODY BELIEVES; THE SIM KNOWS. `pose` is the body's own estimate (leg
odometry and the scan matcher on the quadruped). `true_pose`, `root_xy`, `orientation` and
`footprint_centre` read the world: the sim's own checks (a death's record,
a failed swap's trace), and the few acts a sensor could equally make (an
IMU's attitude; a robot lying down seen as a lump in a depth image).

⚠ A ROUTINE CALL IS NOTHING UNTIL IT IS DRIVEN (`tick.py`): every
`*_routine` member is a generator, composed with `yield from`.
"""

from __future__ import annotations

import abc
import math
from dataclasses import dataclass
from typing import Any, NamedTuple

import mujoco
import numpy as np

from pluggybot import tick
from pluggybot.robot import FIRST, RobotHandle
from pluggybot.tick import Routine


#: Torso tilt from upright that counts as knocked over (issue #107), and
#: how long it has to hold: a body tipped for a moment recovers, a robot on
#: its side does not. The quadruped's get-up takes over at the same angle
#: (`legs.posture.FALL_TILT_RAD`), and its own hold is `stuck_after_s`.
TOPPLE_TILT_RAD = math.radians(60.0)
TOPPLE_HOLD_S = 2.0


@dataclass(frozen=True)
class RackPose:
  """The frame a body's charge logic works round (`Body.rack`): an origin
  in the world and its outward normal, `yaw` -- on the quadruped its dock,
  origin at the board, +x out into the room (`QuadMission._dock_as_rack`)."""

  x: float
  y: float
  yaw: float

  def to_world(self, lx: float, ly: float) -> tuple[float, float]:
    c, s = math.cos(self.yaw), math.sin(self.yaw)
    return self.x + lx * c - ly * s, self.y + lx * s + ly * c

  @property
  def heading(self) -> float:
    """The heading a robot faces the frame with (into its outward normal)."""
    return math.atan2(-math.sin(self.yaw), -math.cos(self.yaw))

  def error_against(self, other: "RackPose") -> tuple[float, float]:
    """(position error in m, yaw error in rad) -- for reports and tests."""
    return (math.hypot(self.x - other.x, self.y - other.y),
            abs(math.atan2(math.sin(self.yaw - other.yaw),
                           math.cos(self.yaw - other.yaw))))


class KeepClear(NamedTuple):
  """Another robot, as one of `Body.others` answers it: where to keep clear
  of, and whether it is LYING DOWN (issue #365) -- a disc round its body,
  and nothing to wait for. A bare `(x, y)` is a robot standing. Where the
  pair says so, also WHO it is and whether it is RESTING (issue #415): a
  robot lying down to rest can be asked to make way (`Body.ask_way`)."""
  x: float
  y: float
  down: bool = False
  root: str = ""
  resting: bool = False


class Body(abc.ABC):
  """What the day asks of a body. See the module docstring."""

  # ---- which robot, and the seam -------------------------------------------

  #: This robot's names in the world (`robot.RobotHandle`).
  handle: RobotHandle
  #: The command that holds this body where it is.
  STILL: Any
  #: The tilt from upright (rad) under which this body counts as level --
  #: its map gate; a lean below it has no direction worth reporting.
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
    robot's stepper or hooks see (a stand-up lands mid-loop), so the
    quadruped steps nothing (issue #387)."""

  @abc.abstractmethod
  def go_to_routine(self, x: float, y: float, timeout: float = 90.0,
                    stop=None) -> Routine:
    """Go to a world point over the map, round the other robots; True on
    arrival. Why it did not arrive is `last_drive`. The quadruped walks it,
    through floor it has not mapped as well (issue #381). `timeout` is its
    patience;
    `stop`, a callable asked every second on the way, ends it where it
    stands when it answers True (`navigator.DRIVE_STOPPED`)."""

  #: How the last `go_to_routine` ended (issue #350): `why` ("" arrived,
  #: else the body's cause), `goal`, `seconds`, `shortM`, and for a peer
  #: `peerAt` / `peerM` / `peerXY` / `peerDown` (`peerRests` if it lay down
  #: to rest); `askedWay`, the robots asked to make way that said yes
  #: (issue #415). None before any.
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
    """Look all round where it stands, to seed the map and buy sight lines."""

  #: Its map of the floor (`mapping.OccupancyGrid`): what it plans over,
  #: what exploring plans frontiers on, what a census counts off.
  grid: Any
  #: THE PLACES IT HAS FOUND (issue #419, `mapping.places.Places`): each
  #: task area's tag where it saw it in its own map, kept with the map and
  #: forgotten at a true death -- or None for a body that keeps none.
  places: Any

  @abc.abstractmethod
  def find_tag_routine(self, tag: int, near: tuple[float, float] | None,
                       patience: float, stop=None) -> Routine:
    """Find the task area `tag` marks (issue #419): where it remembers it,
    confirmed by the tag, or else searched for round `near` -- a job's
    address, in its map; None, round where it stands -- until the tag is in
    view, `patience` s run out or `stop` (the robot's own interrupt, asked
    as a walk asks it) says so. Returns its record: `found`, `why`,
    `seconds`, and `at` once found."""

  @abc.abstractmethod
  def press_plate_routine(self, tag: int, patience: float, stop=None) -> Routine:
    """Walk onto the plate `tag` marks and back off it, the last step
    measured off the tag, within `patience` s (issue #419); `stop` as
    `find_tag_routine`'s. Returns its record: `pressed` (a foot on the
    pad) and `why`."""

  @abc.abstractmethod
  def draw_routine(self, board: str, program, patience: float, stop=None,
                   on_stroke=None) -> Routine:
    """Draw `program` (a `strokes.StrokeProgram`) on the whiteboard
    `board`, found by its tags, the pen on the fork (issue #406): walk to
    it, take its stance, find its face and draw, within `patience` s;
    `stop` as `find_tag_routine`'s, asked between strokes too, and
    `on_stroke(i, points, program)` hears each stroke's ink. Returns its
    record: `drew` and `why`, and the figure's stats where it drew."""

  @abc.abstractmethod
  def hide_routine(self, away_from: tuple[float, float], reach_m: float,
                   clear_of_m: float, patience: float, stop=None) -> Routine:
    """Hide from a robot counting at `away_from` (issue #404, hide and
    seek): pick a spot on the floor it has mapped, no further than
    `reach_m` to walk, more than `clear_of_m` from `away_from` and out of
    its sight where the map allows, and walk there within `patience` s;
    `stop` as `find_tag_routine`'s. Returns its record: `hid` (arrived),
    `at`, `why`, `seconds`."""

  @abc.abstractmethod
  def seek_routine(self, base: tuple[float, float], cover_m: float,
                   patience: float, stop=None) -> Routine:
    """Search its own map outward from `base`, where it counted (issue
    #404), walking until every floor it knows within reach has been within
    `cover_m` of it in plain sight, `patience` s are up, or `stop` says so
    -- never told where the other robot is. Returns its record: `why`,
    `seconds`, how many places it walked to."""

  @abc.abstractmethod
  def forget_world(self) -> None:
    """A true death (issue #419): its map and its places cleared -- the new
    robot knows where its dock is and nothing else."""

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

  #: What it believes about the frame its charge logic works round
  #: (`RackPose`): on the quadruped, its dock.
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

  #: The bay a swap is working at, by station y, from its first drive to
  #: its verdict (issue #347); None otherwise.
  swapping_at: float | None
  #: A manoeuvre of its reach under way that no file holds (issue #405: the
  #: quadruped's swap and arm moves, a fork half under a peg): a restart's
  #: save waits it out (`continuation.Keeper.busy`).
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

  #: Held on the charger: nothing it does while set is counted as travel.
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

  #: Pressed against something it is moving into (its bumper).
  pressing: bool
  #: The camera its eye looks through (issue #275), by its name in the
  #: world: the one on the head, looking along the body's +x -- the
  #: quadruped's `nav_eye` -- or None where it has none (the stub). The
  #: body's, not the eye's: another body's name kept in the loop took the
  #: served process down on legs (issue #408).
  head_camera: str | None

  # ---- the others, and collisions ------------------------------------------

  #: THE OTHER ROBOTS (issue #167): callables answering each one's reported
  #: (x, y), or a `KeepClear`; set by the pair.
  others: list
  #: How this body asks another robot to MAKE WAY (issue #415): a callable
  #: `(root, route) -> bool`, True while the robot named is stepping off
  #: `route` (this body's way, a list of (x, y) from where it stands); set
  #: by the pair, None for a robot alone.
  ask_way: Any
  #: The robot this body is stepping aside for, by its root, or None.
  making_way: str | None
  #: How many times it has stepped aside (issue #415) ...
  asides: int
  #: ...and the last time's record: for whom, from where to where, how
  #: long it took, and why it ended.
  last_aside: dict | None

  @abc.abstractmethod
  def make_way(self, route: list, by: str) -> bool:
    """Another robot (`by`, its root) asks this body to step off `route`,
    its way, which this body cuts (issue #415). Lying down to rest and free
    to -- not docked, not mid-move, no walk of its own under way -- it
    stands and walks aside beneath whatever it is holding, a rule in code
    like the rest reflex, the mind not asked. True while it makes way."""

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
    position servo's `ctrl`, or the joint angle a controller turns into
    torque (the quadruped's arm, issue #405) -- never a torque."""

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
    """Its reach into the driving configuration (the quadruped's arm folded
    to its stow)."""

  # ---- posture and rest ----------------------------------------------------

  #: Lying down to rest.
  resting: bool
  #: Its posture, as the wire carries it (issue #387): `standing`,
  #: `lying_down`, `lying`, `standing_up` or `getting_up`. Lying down to
  #: rest is a posture, never a fall.
  posture: str
  #: Whether it gets itself up from a fall (issue #387): the quadruped's
  #: get-up policy does; a body that cannot waits for somebody.
  rights_itself: bool
  #: How long a fall may last before it is the `stuck` death, s: a body that
  #: rights itself, its get-up's MEASURED budget; one that cannot,
  #: `TOPPLE_HOLD_S`.
  stuck_after_s: float

  @abc.abstractmethod
  def rest_routine(self) -> Routine:
    """Into its resting posture. Code calls it, by reflex (#387)."""

  @abc.abstractmethod
  def stand_routine(self) -> Routine:
    """Out of it."""


def members() -> frozenset[str]:
  """Every name the interface declares: what `tests/test_body.py` lets a
  module on the loop's side of the seam read off a body."""
  return frozenset({n for n in vars(Body) if not n.startswith("_")}
                   | set(Body.__annotations__))


def body_for(model, data, handle: RobotHandle = FIRST, **kw) -> Body:
  """The body this robot IS, in this world: the one choice a lifecycle
  built without one makes (`HubLifecycle(body=None)`) -- legs
  (`legs/body.py`, issue #387), found by this robot's names; a world with
  no robot of ours under them is refused. `kw` is its navigation: the
  viewer, pacing, rack prior and map bounds."""
  if not is_quadruped(model, handle):
    raise ValueError(f"no body under {handle.root!r} in this world: the "
                     "quadruped is the one body (the rover left in #376)")
  from pluggybot.legs.body import QuadBody
  return QuadBody(model, data, handle=handle, **kw)


def is_quadruped(model, handle: RobotHandle = FIRST) -> bool:
  """Does this robot walk? Its first leg's hip, by name."""
  return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                           handle.el("FL_hip_abd")) >= 0


# ---- the stub: a body with no physics ---------------------------------------

#: The stub's world: a floor and nothing else. It compiles in a millisecond,
#: steps in microseconds, and carries no robot to be deleted with a body.
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
  on, instead of the quadruped: `HubLifecycle(*StubBody.world(),
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
               bays: dict[str, int] | None = None, draw_w: float = 0.0,
               charge_w: float | None = None) -> None:
    from pluggybot.mapping.occupancy_grid import OccupancyGrid
    from pluggybot.procedure.steps import TOOL_BAYS
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
    self.rack = self.rack_prior = rack or RackPose(0.0, 0.0, 0.0)
    self.rack_discovered = False
    self.bays = dict(TOOL_BAYS if bays is None else bays)
    self.draw_w = float(draw_w)
    #: the charger's net rate into its pack, W; None is `power.CHARGE_W`
    self.charge_w = charge_w
    #: what the test set: the module on its coupling, the charger's
    #: contact, the root's attitude
    self.holding: str | None = None
    self.on_charger = False
    self.attitude = (1.0, 0.0, 0.0, 0.0)
    #: every place it was sent, in order
    self.went: list[tuple[float, float]] = []
    #: the places it knows (a test sees them in, `places.see`), and the tags
    #: a find or a press was asked for, in order, and the true deaths
    from pluggybot.mapping.places import Places
    from pluggybot.rack.tags import PLATE_TAG_IDS
    self.places = Places(ids=PLATE_TAG_IDS)
    self.found: list[int] = []
    self.pressed: list[int] = []
    #: ...each drawing (board, program name) it was asked for, and whether
    #: it draws (a test sets it): the program's own strokes as its ink
    self.drew: list[tuple[str, str]] = []
    self.draws = False
    #: ...and each hide (from where, its reach, how far clear) and each
    #: search (from where, its reach) it was asked for (issue #404)
    self.hid_from: list[tuple] = []
    self.sought: list[tuple] = []
    self.forgot = 0
    self.last_drive = None
    self.swapping_at = self.peer_at_bay_m = None
    self.working = False
    self.bay_wait = None
    self.docked = self.pressing = self.resting = False
    self.others = []
    self.ask_way = None
    #: whether it makes way when asked (a test sets it), and every ask:
    #: (route, by)
    self.makes_way = False
    self.way_asked: list[tuple[list, str]] = []
    self.making_way = None
    self.asides, self.last_aside = 0, None
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
    return {"pose": [self.x, self.y, self.theta], "holding": self.holding,
            "places": self.places.kept_state()}, {}

  def restore_kept(self, state, arrays) -> bool:
    self.x, self.y, self.theta = state["pose"]
    self.holding = state.get("holding")
    self.places.restore_kept(state.get("places"))
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

  def find_tag_routine(self, tag, near, patience, stop=None):
    """Found at once where the test put the place in, and never otherwise:
    the stub searches nothing."""
    self.found.append(int(tag))
    p = self.places.get(tag)
    return {"tag": int(tag), "found": p is not None,
            "why": "found" if p is not None else "not found", "seconds": 0.0,
            **({"at": [p.x, p.y]} if p is not None else {})}
    yield

  def press_plate_routine(self, tag, patience, stop=None):
    self.pressed.append(int(tag))
    known = self.places.get(tag) is not None
    return {"tag": int(tag), "pressed": known,
            "why": "pressed" if known else "not found", "attempts": []}
    yield

  def draw_routine(self, board, program, patience, stop=None, on_stroke=None):
    """Drawn at once where the test said it draws, the figure exactly:
    the stub has no pen and no board."""
    self.drew.append((board, getattr(program, "name", "")))
    if not self.draws:
      return {"board": board, "drew": False, "why": "not found"}
    lines = [list(s) for s in program.strokes]
    for i, line in enumerate(lines):
      if on_stroke is not None:
        on_stroke(i, line, program.name)
    return {"board": board, "drew": True, "why": "drew", "strokes": len(lines),
            "strokes_drawn": len(lines), "inked_fraction": 1.0,
            "travel_ink_fraction": 0.0, "shape_rms_mm": 0.0, "form_rms_mm": 0.0}
    yield

  def hide_routine(self, away_from, reach_m, clear_of_m, patience, stop=None):
    """Hidden at once, `clear_of_m` past `away_from` on the line through
    where it stands (or along +x from there): the stub maps nothing."""
    self.hid_from.append((tuple(away_from), float(reach_m), float(clear_of_m)))
    dx, dy = self.x - away_from[0], self.y - away_from[1]
    d = math.hypot(dx, dy)
    ux, uy = (dx / d, dy / d) if d > 0.0 else (1.0, 0.0)
    r = max(d, clear_of_m + 0.5)
    at = (away_from[0] + ux * r, away_from[1] + uy * r)
    self.x, self.y = at
    return {"hid": True, "at": [round(at[0], 3), round(at[1], 3)], "why": "hid",
            "seconds": 0.0}
    yield

  def seek_routine(self, base, cover_m, patience, stop=None):
    """Seeks by standing still until `stop` says so or `patience` runs out,
    a second at a time: the stub has no floor to walk."""
    self.sought.append((tuple(base), float(cover_m)))
    t0 = float(self._data.time)
    while float(self._data.time) - t0 < patience:
      if stop is not None and stop():
        return {"why": "stopped", "seconds": round(float(self._data.time) - t0, 1),
                "targets": 0}
      yield from self._wait(min(1.0, patience - (float(self._data.time) - t0)))
    return {"why": "out of time", "seconds": round(float(self._data.time) - t0, 1),
            "targets": 0}

  def forget_world(self) -> None:
    self.grid.grid[...] = 0.0
    self.places.forget()
    self.forgot += 1

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
    pack = Pack(capacity_wh=capacity_wh, charge_scale=charge_scale,
                draw_w=self.draw_w)
    if self.charge_w is not None:
      pack.charge_w = float(self.charge_w)
    return pack

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

  def make_way(self, route, by) -> bool:
    self.way_asked.append((list(route), by))
    return self.makes_way

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
