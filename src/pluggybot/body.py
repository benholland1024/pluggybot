"""What the day asks of a body (issue #380).

The robot's life -- the day loop (`lifecycle.py`), its mind, its economy,
its record -- is the same whatever it lives in; the machine is not. `Body`
is everything the loop and the procedure layer (`procedure/`) may ask of
the machine, and the ONLY way they reach it: the humanoid implements it
(`humanoid/body.py`, #507), and `StubBody` below is the smallest thing that
does, for a test that needs the lifecycle's bookkeeping and no physics. (The
wheeled rover implemented it first, #380, and the quadruped after it, #387;
#376 and #559 deleted them, and `rover-final` and `quadruped-final` have
them.) `tests/test_body.py` walks the syntax tree of
every module on this side of the seam and fails on a reach that is not a
member here. `robot.RobotHandle` still resolves a robot's names in the
world; the body sits beside it (`Body.handle`), not in place of it.

What a body does, in the issue's words: it GOES to a place and faces a
bearing; it knows WHERE it is; it DOCKS to charge and undocks; it SENSES
(a map of the floor, a depth frame, its cameras, the other robots); it has
a POSTURE it can rest in; and its HANDS reach, grip, carry and pull. Under
those, the MOTOR LEVEL a procedure is built from (issue #166): an actuator
walked to a setpoint.

⚠ THE COMMAND IS THE BODY'S (the tick contract, `tick.py`). A routine
yields one command per physics step, and only the body that yields it
reads it: `stepper.apply(command)` turns it into actuator setpoints. The
humanoid's carries its velocity, the input its walking policy tracks. So code on this side of the seam never builds a command: it
composes the body's routines, and where it must hold one physics step still
it yields `STILL`.

⚠ A BODY BELIEVES; THE SIM KNOWS. `pose` is the body's own estimate (leg
odometry and the scan matcher). `true_pose`, `root_xy`, `orientation` and
`footprint_centre` read the world: the sim's own checks (a death's record,
a drift row), and the few acts a sensor could equally make (an IMU's
attitude; a robot fallen seen as a lump in a depth image).

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
#: its side does not. A body's own hold is `stuck_after_s` (the humanoid's
#: get-up budget).
TOPPLE_TILT_RAD = math.radians(60.0)
TOPPLE_HOLD_S = 2.0


@dataclass(frozen=True)
class RackPose:
  """The frame a body's charge logic works round (`Body.rack`): an origin
  in the world and its outward normal, `yaw` -- the humanoid's charging
  station, +x out into the room (`humanoid.body`, off `activity.station`)."""

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
  of, and whether it is DOWN (issue #365: fallen) -- a disc round its
  body, and nothing to wait for. A bare `(x, y)` is a robot standing. Where
  the pair says so, also WHO it is and whether it is RESTING (issue #415):
  a robot resting can be asked to make way (`Body.ask_way`). And whether it
  is DEAD (issue #455): it lies where it died until the world stands it
  up, so it is neither waited for nor asked."""
  x: float
  y: float
  down: bool = False
  root: str = ""
  resting: bool = False
  dead: bool = False


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
    robot's stepper or hooks see (a stand-up lands mid-loop), so a body
    steps nothing here (issue #387)."""

  @abc.abstractmethod
  def go_to_routine(self, x: float, y: float, timeout: float = 90.0,
                    stop=None) -> Routine:
    """Go to a world point over the map, round the other robots; True on
    arrival. Why it did not arrive is `last_drive`. It walks through floor
    it has not mapped as well (issue #381). `timeout` is its
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
    read to act only for a robot lying still, fallen, resting or dead, or
    stepping aside for the one that reads it (issues #365, #455)."""

  @abc.abstractmethod
  def as_seen(self, x: float, y: float) -> tuple[float, float]:
    """A true world point where its own sensors would place it: seen from
    where it stands, placed through where it believes it stands."""

  @abc.abstractmethod
  def from_seen(self, x: float, y: float) -> tuple[float, float]:
    """`as_seen` undone: a point in its own map back to the true world
    point (issue #455). How a way it plans reaches another robot, told
    relative to that robot's body as this one sees it."""

  @abc.abstractmethod
  def level(self) -> bool:
    """Level enough for a reading to be a map of the room (issue #339)."""

  #: WHAT MOVED THE BELIEF OUTSIDE A MATCH, OR LOST IT (issue #476), the
  #: newest few, as `Navigator.belief_event` records them: their `truth` is
  #: the sim's own check, for the observatory alone.
  belief_events: Any
  #: Its scan matcher's verdicts by kind (`ScanMatcher.counts`), or {}.
  match_counts: dict

  @abc.abstractmethod
  def lost_routine(self, fixture: str, stop=None) -> Routine:
    """`fixture` (its charging station) is not in sight where the belief
    puts it in plain sight (issue #476): unless another robot can hide it,
    the robot is lost, drops its map and looks round for the fixture until
    it puts it right or `stop` says so. True once it has."""

  # ---- the charger's frame ------------------------------------------------

  #: What it believes about the frame its charge logic works round
  #: (`RackPose`): its charging station. (Named `rack` since the rover,
  #: whose charger was on its tool rack.)
  rack: Any
  #: ...and the COMMISSIONED pose: the station that defines its map frame
  #: (issue #42), where `anchor_at_dock` snaps its estimate.
  rack_prior: Any
  #: Has it confirmed the station by its tags?
  rack_discovered: bool

  @abc.abstractmethod
  def start_discovery(self) -> None:
    """Begin watching for the station's tags through everything it does."""

  @abc.abstractmethod
  def refresh_rack(self):
    """Adopt the station pose it has confirmed, if it has: the pose, or None."""

  #: A manoeuvre of its reach under way that no file holds (issue #405: a
  #: hand out): a restart's save waits it out (`continuation.Keeper.busy`).
  working: bool

  # ---- the dock ------------------------------------------------------------

  @abc.abstractmethod
  def charge_standoff(self) -> tuple[float, float, float]:
    """Where it stands to begin docking -- (x, y, heading) off the rack it
    believes in."""

  @abc.abstractmethod
  def charge_seat(self) -> tuple[float, float]:
    """Where it stands to charge, (x, y): the far end of the walk in from
    `charge_standoff`. Another robot there holds the charge bay as much as
    one on the standoff (issue #455)."""

  @abc.abstractmethod
  def dock_routine(self) -> Routine:
    """From the charge standoff, onto the charger until its contacts
    conduct; returns the approach's answer. Success is `charging()`."""

  @abc.abstractmethod
  def charging(self) -> bool:
    """Its pack takes the charger's current: the ELECTRICAL criterion."""

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

  @abc.abstractmethod
  def watch(self, tag: int | None) -> None:
    """Its head's colour camera kept reading `tag` while it goes on -- its
    hands moving, its feet stepping, standing still -- through the camera
    `spot` reads (#554's stage 2, `perception.watch.TagWatch`): a picture
    each `WATCH_HZ`th of a second, each the robot's `LATENCY_S` after it
    was taken (`sighting`). Another tag takes its place, what was read of
    the last forgotten; None stops it."""

  @abc.abstractmethod
  def sighting(self) -> dict | None:
    """The watched tag as the newest reading the robot has of it (#554):
    {tag, seen -- whether its picture shows the tag --, at -- when that
    picture was taken, sim s --, last -- where the newest picture that
    showed it put it: x, y, z, the believed world frame, range, normal,
    edge (as `spot`'s) and at -- or None}; None before a watch's first
    reading."""

  #: The camera its eye looks through (issue #275), by its name in the
  #: world: the one on the head, looking along the body's +x -- or None
  #: where it has none (the stub). The body's, not the eye's: another
  #: body's name kept in the loop took the served process down (issue
  #: #408).
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
  #: Why it last said no to making way (issue #455): "charging" (plugged in
  #: at its station, #559), "working", "walking" (a walk of its own),
  #: "getting up", "standing", "no spot" -- or "".
  way_refused: str
  #: How many times it has stepped aside (issue #415) ...
  asides: int
  #: ...and the last time's record: for whom, from where to where, how
  #: long it took, and why it ended.
  last_aside: dict | None

  @abc.abstractmethod
  def make_way(self, route: list, by: str) -> bool:
    """Another robot (`by`, its root) asks this body to step off `route`,
    its way, which this body cuts (issue #415). Resting and free to -- not
    plugged in, not getting up, no hand out, no walk of its own under way --
    it walks aside beneath whatever it is holding, a rule in code, the mind
    not asked. True while it makes way."""

  @abc.abstractmethod
  def ask_off(self, x: float, y: float, asked: dict, beyond=()) -> bool:
    """Ask a robot resting ON (x, y) -- the charge bay this body waits
    for -- to step off the way there from where this body stands, and on
    past it through `beyond` (issue #455): a drive's ask (`ask_way`), for a
    wait that walks nowhere. `asked` is the waiter's, kept across its
    wait: who said yes, and when first. True while one is making way."""

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
  #: ...and physics steps in a contact it did not mean (a charge press is
  #: meant).
  collision_steps: int

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
    torque (issue #405) -- never a torque."""

  # ---- hands (issue #507: the humanoid's) ----------------------------------

  @abc.abstractmethod
  def reach_routine(self, side: str, pos, rot=None, stiffness: float = 1.0,
                    patience: float = 20.0, path=None, turns=None) -> Routine:
    """A hand (`side`, "left" or "right") to a grip pose in the believed
    world frame -- `pos`, m, and `rot`, world from grip (its fingers along
    its -z, its jaws closing along its x), or None for any -- its arm's
    commands carrying `stiffness`, a share of its own gains: a compliant
    pull is a low one. The posture is the body's to choose. True once it
    holds there; False if no posture it holds puts the pose in reach, the
    hand stopped short on its way, a fall let go of it (#520), or
    `patience` ran out. With `path` (and `turns`, the grip's turn at each of
    its points, world from grip) it reaches in a posture from which the
    hand then FOLLOWS that path (`hand_follows`) -- or, where that posture
    makes no approach, in any, the grip as asked (#493: a knob gripped in a
    crouch whose lid's arc only a bow followed). The humanoid's (#507); a
    body with no such hand refuses it (`NotImplementedError`)."""

  @abc.abstractmethod
  def release_routine(self, side: str) -> Routine:
    """A hand back out along its way in, and its arm home: returns "home"
    once its drivers read it there (#539), else why not -- "fell" (a fall
    let go of it) or "held" (not home in the body's time: something holds
    it)."""

  @abc.abstractmethod
  def grip_routine(self, side: str, closed: bool, force: float | None = None) -> Routine:
    """A hand's gripper shut -- squeezing with `force`, N a pad, or its
    sheet's most where none is given (#525) -- or opened, standing while it
    moves; shut, it returns whether it holds something (`gripped`)."""

  @abc.abstractmethod
  def gripped(self, side: str) -> bool:
    """Whether a hand holds something, by its gripper's own reading -- its
    jaws stopped short of closed, squeezing -- never by what the world
    says it touches (#493: a quadruped's `held()` once read the world)."""

  @abc.abstractmethod
  def grip_gap(self, side: str) -> float:
    """How far apart a hand's jaws are, m, off its gripper's own encoder
    (#525): what it holds, how wide."""

  @abc.abstractmethod
  def hand_force(self, side: str) -> tuple[float, float, float]:
    """The force a hand pushes with, N, in the believed world frame, read
    off its motor currents: the humanoid's, standing, off its whole body's
    for a hand at work (#547), else its arm's."""

  @abc.abstractmethod
  def hand_follows(self, side: str, poses, rot=None) -> bool:
    """Whether a hand's grip could FOLLOW `poses` (the believed world frame)
    in turn at the turn `rot` -- one turn, or a list of one a pose -- from
    one posture it holds, its arm moving on from each to the next -- asked
    of its model of itself (#493)."""

  @abc.abstractmethod
  def hand_reaches(self, side: str, pos, rot=None) -> bool:
    """Whether `reach_routine` would plan a hand's grip to `pos` (the
    believed world frame) at the turn `rot`, or any -- asked of its model of
    itself, nothing moved: what a reach out of reach says of the other hand
    (#493)."""

  @abc.abstractmethod
  def step_routine(self, ahead: float, left: float = 0.0, turn: float = 0.0,
                   patience: float = 15.0) -> Routine:
    """THE STEP (#525): its hands home and empty, the body turns `turn`
    rad on the spot (left, +) and walks `ahead` m and `left` m in the frame
    it then stands in (negative backs up, or goes right), and stands still
    -- as the body walks: the humanoid's least step is a stride. Returns how it ended (`why`:
    "stepped", "holding" -- a hand holds something, and nothing moves --
    "blocked", "fell", "out of time"), in how long, and where its stance
    went by its own reckoning, in the stance frame it stood in before
    (`dx`, `dy`, m; `turn`, rad): what a place read before the step is
    turned into."""

  @abc.abstractmethod
  def gaze_routine(self, down: float) -> Routine:
    """Its head's cameras turned `down` rad below their gaze standing --
    0 back to it -- as the body can: the humanoid, its head fixed on its
    trunk, bows (#493: a knob at the bottom of its colour camera's view
    from a step nearer). Returns once still there."""

  @abc.abstractmethod
  def depth_cloud(self, width: int, height: int, min_z: float, frames: int = 1,
                  apart: float = 0.1) -> Routine:
    """Standing still, `frames` frames of its depth camera at `width` x
    `height` pixels and `min_z` (m, the part's nearest at that stream),
    `apart` s apart: the points it saw, (k, 3), the believed world frame,
    through its camera's pose in its model of itself (#493: the humanoid's
    D435). Its own body never in them."""

  @abc.abstractmethod
  def hand_at(self, side: str) -> tuple[float, float, float]:
    """Where a hand's grip is, m, in the believed world frame: its arm's
    encoders and its IMU through its model of itself (#493)."""

  @abc.abstractmethod
  def hand_sensed(self, side: str) -> tuple[float, float, float]:
    """`hand_at` as the hand's control loop last read it, at its last
    command (#554): what a move's stop condition reads, at the encoders'
    own rate, drawing nothing more on their noise -- read afresh between
    commands, a condition that never came true changed the flight."""

  @abc.abstractmethod
  def hand_watched(self, side: str) -> tuple[float, float, float]:
    """`hand_at` for a WATCHER (#530: a person's page shows it): the same
    encoders and IMU through a model of itself of the watcher's own, its
    noise a stream of the watcher's own, so watching a flight never moves
    it -- a probe that read `hand_at` drew on the flight's stream and
    changed what the flight did (#525)."""

  @abc.abstractmethod
  def depth_watched(self, width: int, height: int) -> np.ndarray:
    """What its depth camera sees now, for a WATCHER (#530): a `height` x
    `width` frame of axial depth, m, NaN where the part returns nothing --
    on a camera and a noise stream of the watcher's own, as `hand_watched`,
    so watching never moves the flight. Its own body is never in it."""

  @abc.abstractmethod
  def move_hand_routine(self, side: str, pos, stiffness: float = 1.0,
                        speed: float = 0.06, patience: float = 20.0,
                        path=None, turns=None) -> Routine:
    """A hand that is out -- reached, and holding there or moved on --
    carried along a straight line to `pos`, the believed world frame, m, at
    `speed`, m/s, the turn it was asked to hold kept, at `stiffness` --
    through `path` first, where given: points walked in turn as one
    motion, from one posture that follows them all (`hand_follows`),
    turning to `turns`' at each (one a point of `path` and one at `pos`,
    world from grip, None keeping the one before) where they are given. The
    posture is the body's to choose (#493). Returns how it ended (`why`:
    "arrived", "held back" -- its command there, the hand held short of it
    -- "out of reach", "not out", "until" -- `stop_hand` stopped it --
    "fell", "feet moved", "out of time"), in
    how long, how far the hand went (`moved`, m) and where it ended up from
    where it began (`dx`, `dy`, `dz`, believed world frame, m)."""

  @abc.abstractmethod
  def hand_why(self, side: str) -> str:
    """Why a hand's last `reach_routine` ended short of holding there: "out
    of reach" (no posture it holds puts the pose in reach), "stopped short"
    (it stopped on its way: pushing on something, or out of reach as the
    body truly stood), "until" (`stop_hand` stopped it), "fell", "out of
    time" -- or "" where it held (#493)."""

  @abc.abstractmethod
  def stop_hand(self, side: str) -> bool:
    """A hand's move stopped now and held where it is, as its own stops
    hold it (#554: a procedure's stop condition, `steps._guarded_routine`):
    a reach's arm where it is commanded, on its way out or in; a carry's or
    a pull's hand where it is -- each move then ending as "until"
    (`hand_why`, the routine's own `why`). True where a move was under way;
    False, nothing changed, where none was."""

  @abc.abstractmethod
  def pull_routine(self, side: str, direction, limit_n: float, distance_m: float,
                   speed: float = 0.06, patience: float = 30.0, give: bool = True) -> Routine:
    """THE PULL, a FORCE MOVE (#525): a hand that is out pulled -- or
    pushed -- along `direction` (the believed world frame) with at most
    `limit_n`, N, as the hand reads it, and at most `speed`, m/s; across
    that way, `give`, the hand yields to what pushes it (no faster than its
    speed allows) and its turn gives to what it holds, but about the jaws'
    own line -- or, not `give`, it holds its line and its turn: hybrid force
    and position. Until it has gone
    `distance_m` along its way, it is held fast (pushing with its whole
    force and not moving), its grip lets go, it can reach no further, its
    feet move, or `patience` runs out; then it holds where the hand is.
    Returns how it ended (`why`: "distance", "held fast", "let go", "out of
    reach", "not out", "until" -- `stop_hand` stopped it -- "fell", "feet
    moved", "out of time"), in how long,
    how far the hand went (`moved`) and along its way (`along`), where it
    ended up from where it began (believed world frame, m), the way the hand
    was going at its end (`direction`, a unit vector: what it held allowed)
    and the hardest it pushed along its way (`force`, N)."""

  @property
  @abc.abstractmethod
  def stance(self) -> tuple[float, float, float]:
    """Where it believes its STANCE is, (x, y, heading): the frame its
    hands' work is put in -- the humanoid's between its soles (#507), a
    body with no hand its own pose -- what a procedure's hand verbs say
    their places in (#493)."""

  # ---- posture and rest ----------------------------------------------------

  #: Resting: the humanoid rests standing (`humanoid.way`).
  resting: bool
  #: Its posture, as the wire carries it (issue #387): `standing` or
  #: `getting_up`. Resting is never a fall.
  posture: str
  #: Whether it gets itself up from a fall (issue #387): the humanoid's
  #: get-up policy does; a body that cannot waits for somebody.
  rights_itself: bool
  #: The falls its own fall watch has taken, counted up: what a fall IS is
  #: the body's to say (#493's grader asks) -- the humanoid's trunk 35 deg
  #: past the posture its legs hold, by its IMU, or its pelvis sunk
  #: (`humanoid.getup`), each the moment its get-up takes over.
  falls: int
  #: How long a fall may last before it is the `stuck` death, s: a body that
  #: rights itself, its get-up's MEASURED budget; one that cannot,
  #: `TOPPLE_HOLD_S`.
  stuck_after_s: float
  #: The trunk's pitch forward that the posture its legs hold puts it at,
  #: rad: what a topple is read past (`TOPPLE_TILT_RAD`, the death seam's).
  #: 0 for a body that stands only upright; the humanoid's bow
  #: (`Reach.held_pitch`), whose deepest, 63 deg, is past the seam's 60
  #: (#559).
  held_pitch: float

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
  built without one makes (`HubLifecycle(body=None)`) -- the humanoid
  (`humanoid/body.py`, #507), found by this robot's names; a world with no
  robot of ours under them is refused. `kw` is its navigation: the map
  bounds, its policies, the viewer and pacing."""
  if is_humanoid(model, handle):
    from pluggybot.humanoid.body import HumanoidBody
    return HumanoidBody(model, data, handle=handle,
                        **{k: v for k, v in kw.items()
                           if k in ("grid_bounds", "policy", "walk", "viewer", "realtime")})
  raise ValueError(f"no body under {handle.root!r} in this world: the humanoid "
                   "is the body (the rover left in #376, the quadruped in #559; "
                   "`rover-final` and `quadruped-final` have them)")


def is_humanoid(model, handle: RobotHandle = FIRST) -> bool:
  """Does this robot walk on two legs (#507)? Its left hip's pitch, by
  name."""
  return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                           handle.el("left_hip_pitch")) >= 0


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
  `on_charger` what `charging` says, `attitude` its IMU, `hands` and
  `grips` its hands. Time passes only where a routine waits
  (`hold_routine` and its kin step the stub's world), so the lifecycle's
  seams tick as they do on a real body. What a test that needs the
  lifecycle's BOOKKEEPING -- the mind, the economy, the record -- builds it
  on, instead of the humanoid: `HubLifecycle(*StubBody.world(),
  body=StubBody(...))`."""

  STILL = ()
  level_tilt_rad = math.radians(1.5)
  posture = "standing"
  rights_itself = False
  falls = 0
  stuck_after_s = 2.0
  held_pitch = 0.0
  head_camera = None

  @staticmethod
  def world():
    """A fresh (model, data) of `STUB_WORLD`."""
    model = mujoco.MjModel.from_xml_string(STUB_WORLD)
    return model, mujoco.MjData(model)

  def __init__(self, model=None, data=None, handle: RobotHandle = FIRST,
               pose: tuple[float, float, float] = (0.0, 0.0, 0.0),
               rack=None, grid_bounds=(-3.0, -3.0, 7.0, 7.0),
               draw_w: float = 0.0, charge_w: float | None = None) -> None:
    from pluggybot.mapping.occupancy_grid import OccupancyGrid
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
    self.draw_w = float(draw_w)
    #: the charger's net rate into its pack, W; None is `power.CHARGE_W`
    self.charge_w = charge_w
    #: what the test set: the charger's contact, the root's attitude
    self.on_charger = False
    self.attitude = (1.0, 0.0, 0.0, 0.0)
    #: every place it was sent, in order
    self.went: list[tuple[float, float]] = []
    #: the places it knows (a test sees them in, `places.see`), and the tags
    #: a find was asked for, in order
    from pluggybot.home.areas import area_ids
    from pluggybot.mapping.places import Places
    from pluggybot.perception.tags import BOARD_TAG_IDS, PLATE_TAG_IDS
    self.places = Places(ids=(*PLATE_TAG_IDS,
                              *(t for ids in BOARD_TAG_IDS.values() for t in ids),
                              *area_ids()))
    self.found: list[int] = []
    #: ...where it sent each hand, and which grips hold (#507): a hand
    #: shut where a test sent it holds, as the stub's every walk arrives.
    self.hands: dict[str, tuple] = {}
    self.grips: dict[str, bool] = {}
    #: ...the force each was last shut with (#525), how wide what a shut
    #: hand holds is, m, and which grippers are shut (each starts so)
    self.squeezes: dict[str, float | None] = {}
    self.held_gap = 0.026
    self.shut: dict[str, bool] = {}
    #: ...and the hands a test says reach nothing (#493): a reach with one
    #: is out of reach -- or, `unturned`, nothing turned as asked -- and the
    #: turns a carried hand was asked to take, (side, verb, turn) in turn,
    #: each carry's path, (side, path, pos, turns), and each pull asked
    #: (#525), (side, direction, force, distance, give)
    self.out_of_reach: set[str] = set()
    self.unturned: set[str] = set()
    self.turned: list[tuple] = []
    self.carried_paths: list[tuple] = []
    self.pulled: list[tuple] = []
    #: ...and each step asked (#525), (ahead, left, turn)
    self.stepped: list[tuple] = []
    #: ...and each tag it was asked to watch (#554), None a stop, and what
    #: a test says its newest reading is (`sighting`'s shape)
    self.watched: list = []
    self.sight: dict | None = None
    self._why: dict[str, str] = {}
    self.forgot = 0
    #: ...what moved its belief (a test puts records in), the fixtures it
    #: was lost at, and whether a search finds one (a test sets it)
    self.belief_events: list[dict] = []
    self.match_counts: dict = {}
    self.lost_at: list[str] = []
    self.finds_fixture = False
    self.last_drive = None
    self.working = False
    self.docked = self.resting = False
    self.others = []
    self.ask_way = None
    #: whether it makes way when asked (a test sets it), and every ask:
    #: (route, by)
    self.makes_way = False
    self.way_asked: list[tuple[list, str]] = []
    #: ...and every seat it asked a robot resting on to step off: (x, y)
    self.asked_off: list[tuple[float, float]] = []
    self.making_way = None
    self.way_refused = ""
    self.asides, self.last_aside = 0, None
    self.geom_ids = np.zeros(0, dtype=np.int32)
    self.peer_holds = self.collision_steps = 0

  model = property(lambda self: self._model)
  data = property(lambda self: self._data)
  step_hooks = property(lambda self: self._hooks)
  stepper = property(lambda self: self._stepper)

  def run(self, routine, name=""):
    return tick.run(self._stepper, routine, name)

  def kept_state(self):
    return {"pose": [self.x, self.y, self.theta],
            "places": self.places.kept_state()}, {}

  def restore_kept(self, state, arrays) -> bool:
    self.x, self.y, self.theta = state["pose"]
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

  def forget_world(self) -> None:
    self.grid.grid[...] = 0.0
    self.places.forget()
    self.forgot += 1

  def lost_routine(self, fixture, stop=None):
    """Lost at once, its map forgotten, and found or not as the test set:
    the stub sees nothing to search for."""
    self.lost_at.append(fixture)
    self.forget_world()
    return self.finds_fixture
    yield

  def plan_frontier(self, blacklist):
    from pluggybot.behavior.navigation import plan
    return plan(self.grid, self.pose, blacklist)

  def reachable(self, points):
    return [True] * len(points)

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

  def from_seen(self, x, y):
    return x, y

  def level(self) -> bool:
    w, x, y, z = self.attitude
    return 1.0 - 2.0 * (x * x + y * y) >= math.cos(self.level_tilt_rad)

  def start_discovery(self) -> None:
    pass

  def refresh_rack(self):
    return None

  def charge_standoff(self):
    x, y = self.rack.to_world(0.5, 0.0)
    return x, y, self.rack.heading

  def charge_seat(self):
    return self.rack.to_world(0.0, 0.0)

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

  def watch(self, tag):
    """Noted (`watched`); what it reads is what a test put in `sight`."""
    self.watched.append(None if tag is None else int(tag))
    if tag is None:
      self.sight = None

  def sighting(self):
    return self.sight

  def peer_on_the_goal(self, x, y):
    return None

  def watch_for_peers(self, points):
    return None

  def make_way(self, route, by) -> bool:
    self.way_asked.append((list(route), by))
    self.way_refused = "" if self.makes_way else "no spot"
    return self.makes_way

  def ask_off(self, x, y, asked, beyond=()) -> bool:
    self.asked_off.append((x, y))
    return False

  def actuator(self, name) -> int:
    raise KeyError(f"a stub body has no actuator {name!r}")

  def ramp_routine(self, act, target, speed, settle=0.0):
    yield from self._wait(settle)

  def setpoint(self, act) -> float:
    raise KeyError(f"a stub body has no actuator {act!r}")

  def reach_routine(self, side, pos, rot=None, stiffness=1.0, patience=20.0, path=None,
                    turns=None):
    if side in self.out_of_reach or (rot is not None and side in self.unturned):
      self._why[side] = "out of reach"
      return False
    self._why[side] = ""
    self.hands[side] = tuple(pos)
    return True
    yield

  def release_routine(self, side):
    self.hands.pop(side, None)
    self.grips.pop(side, None)
    return "home"
    yield

  def grip_routine(self, side, closed, force=None):
    self.grips[side] = bool(closed) and side in self.hands
    self.shut[side] = bool(closed)
    self.squeezes[side] = force
    return self.grips[side] if closed else True
    yield

  def gripped(self, side) -> bool:
    return self.grips.get(side, False)

  def grip_gap(self, side) -> float:
    """Shut on what a test put in it (`held_gap`), shut on nothing -- as
    each gripper starts -- or open, the Dex1-1's stroke."""
    if self.grips.get(side):
      return self.held_gap
    return 0.0 if self.shut.get(side, True) else 0.09

  def hand_force(self, side):
    return (0.0, 0.0, 0.0)

  def hand_follows(self, side, poses, rot=None) -> bool:
    """Anywhere: it has no body to be out of reach of."""
    return True

  def hand_reaches(self, side, pos, rot=None) -> bool:
    """Anywhere, but with a hand a test put `out_of_reach` -- or, turned,
    `unturned`."""
    return side not in self.out_of_reach and (rot is None or side not in self.unturned)

  def gaze_routine(self, down):
    """No head to turn: at once."""
    return
    yield

  def step_routine(self, ahead, left=0.0, turn=0.0, patience=15.0):
    """The step at once, as asked -- its hands home, turned, then walked --
    unless a hand holds something; its asks kept in `stepped`."""
    self.stepped.append((float(ahead), float(left), float(turn)))
    if any(self.grips.get(s) for s in self.hands):
      return {"why": "holding", "seconds": 0.0, "dx": 0.0, "dy": 0.0, "turn": 0.0}
    # ...its hands home first, as a walk's are
    self.hands.clear()
    c, s_ = math.cos(turn), math.sin(turn)
    dx, dy = c * ahead - s_ * left, s_ * ahead + c * left
    cw, sw = math.cos(self.theta), math.sin(self.theta)
    self.x += cw * dx - sw * dy
    self.y += sw * dx + cw * dy
    self.theta += float(turn)
    return {"why": "stepped", "seconds": 0.0, "dx": dx, "dy": dy, "turn": float(turn)}
    yield

  def depth_cloud(self, width, height, min_z, frames=1, apart=0.1):
    """No camera: nothing seen, after the time looking takes."""
    yield from self._wait(frames * apart)
    return np.zeros((0, 3))

  def hand_at(self, side):
    """Where it sent the hand; hanging at its side, where it sent none."""
    x, y, _ = self.pose
    return tuple(self.hands.get(side, (x, y, 0.8)))

  def hand_sensed(self, side):
    """`hand_at`: a stub's hand is where it is, every command."""
    return self.hand_at(side)

  def hand_watched(self, side):
    """`hand_at`: a stub draws no noise to keep apart."""
    return self.hand_at(side)

  def depth_watched(self, width, height):
    """No camera: nothing returned anywhere."""
    return np.full((int(height), int(width)), np.nan)

  def _carried(self, side, to) -> dict:
    a = np.asarray(self.hand_at(side), dtype=float)
    self.hands[side] = tuple(float(v) for v in to)
    dx, dy, dz = (float(v) for v in np.asarray(to, dtype=float) - a)
    return {"seconds": 0.0, "moved": float(np.linalg.norm([dx, dy, dz])),
            "dx": dx, "dy": dy, "dz": dz}

  def move_hand_routine(self, side, pos, stiffness=1.0, speed=0.06, patience=20.0,
                        path=None, turns=None):
    """The hand there at once, as asked (a hand that is out)."""
    self.carried_paths.append((side, list(path or []), tuple(pos),
                               None if turns is None else list(turns)))
    if turns is not None:
      self.turned.append((side, "move_hand", list(turns)[-1]))
    if side not in self.hands:
      return {"why": "not out", "seconds": 0.0, "moved": 0.0, "dx": 0.0, "dy": 0.0, "dz": 0.0}
    return {"why": "arrived", **self._carried(side, pos)}
    yield

  def hand_why(self, side) -> str:
    """Never short -- every reach holds at once -- but with a hand a test
    put `out_of_reach`."""
    return self._why.get(side, "")

  def stop_hand(self, side) -> bool:
    """Its moves are done at once: none is under way to stop."""
    return False

  def pull_routine(self, side, direction, limit_n, distance_m, speed=0.06, patience=30.0,
                   give=True):
    """The hand `distance_m` along `direction` at once: nothing holds it."""
    self.pulled.append((side, tuple(direction), float(limit_n), float(distance_m), bool(give)))
    if side not in self.hands:
      return {"why": "not out", "seconds": 0.0, "moved": 0.0, "along": 0.0, "dx": 0.0,
              "dy": 0.0, "dz": 0.0, "direction": list(direction), "force": 0.0}
    d = np.asarray(direction, dtype=float)
    d = d / max(float(np.linalg.norm(d)), 1e-9)
    to = np.asarray(self.hand_at(side), dtype=float) + float(distance_m) * d
    return {"why": "distance", **self._carried(side, to), "along": float(distance_m),
            "direction": [float(v) for v in d], "force": 0.0}
    yield

  @property
  def stance(self):
    return self.pose

  def stand_routine(self):
    self.resting = False
    return
    yield
