"""The step vocabulary: an errand as validated steps over the guarded
primitives (issue #58, rung one of agent-written code).

A PROGRAM is data -- a name, a budget, and per ROLE an ordered list of
steps, each a verb off a closed list with typed, bounded arguments. It is
validated whole before a single step runs, and then ticked from the physics
loop like every other routine (pluggybot/tick.py), one verdict per step.
What makes it safe for an agent to write is what it cannot say: every verb
is an existing primitive with the ramping, the reach's stops and the
noslip policy INSIDE it, and nothing here touches `data.ctrl` (the fence in
tests/test_procedure.py enumerates the modules that may).

The verbs: the walks and a look, and the hands' (`HANDS_VERBS`, #525;
docs/Hands.md has each one's meaning, report and failure):

  drive_to(x, y)       walk to a world point            `Body.go_to_routine`
  face(heading)        turn in place                    `Body.face_routine`
  find(tag, x, y)      a task area found by its tag     `Body.find_tag_routine`
  wait(seconds)        stand still
  move(axis, target)   one axis to a setpoint           `procedure/axes.py`
  spot, measure, watch, reach, via, move_hand, pull, close_hand,
  open_hand, withdraw, step     the hands and the feet  `Body`'s hand members

(The quadruped's -- its tools, the plates, the board, the claw, the survey
and the game's roles -- went with it in #559.)

Every verb reaches the body through `Body` (issue #380, `body.py`) and
nothing else. Each returns a verdict dict with `ok`, measured off the world
(the arrival, a grip its gripper reads, a hand at its goal), never off the
command. The runner stops at the first failed step: later steps
assume earlier ones. The result is honest about how far it got.

Three rules the shape carries for the rungs above it:

  ROLES. A program is `{role: steps}` and today every program has one role,
  `robot`: a second key is a second robot's part, not a rebuild -- so the
  slot validates any number of roles and the runner refuses more than one,
  with the reason.

  BUDGETS ARE CAPPED BY CODE. A program declares its own sim-time budget and
  the vocabulary caps it (`MAX_BUDGET_S`) and the step count (`MAX_STEPS`);
  a program past either does not validate. The runner checks the budget at
  step boundaries, which are the safe points; a walk is a safe point at
  every step (issue #381), and asks `life.interrupted` as it goes (#116).

  ABORT ENDS IT WHERE IT IS. Whatever ends a program early -- a failed
  step, the budget, an interrupt -- the verdict is taken of what it did,
  as it stands.

What a program may say is what a work order may say (TaskPattern.md §2): a
tool by name, a surveyed place by coordinate, a board by id. A program that
drives to where a movable object IS has been handed a sensor reading over the
wire, and the ladder applies to it as to any task.
"""

import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable

from pluggybot.humanoid.body import (STEP_DY_LEFT_AFTER_M, STEP_DY_SHORT_M, STEP_FRONT_HALF_M,
                                     STEP_FRONT_M, STEP_KEEP_DOWN_M, STEP_KEEP_M, STEP_LEAST_M,
                                     STEP_SHORT_M)
from pluggybot.humanoid.model import (CHOSEN, COLOR_FOVY_DEG, color_camera_ahead, color_camera_z,
                                      stand_height)
from pluggybot.perception.box import DEPTH_MIN_Z as MEASURE_NEAREST_M, FLOOR_M as MEASURE_FLOOR_M
from pluggybot.perception.depth import MAX_Z as MEASURE_REACH_M
from pluggybot.perception.tags import BLOCK_TAG_SIZE, PICTURE_H, PICTURE_W
from pluggybot.perception.watch import LATENCY_S as WATCH_LATENCY_S, WATCH_HZ
from pluggybot.tick import Routine

#: The one role every program has.
DEFAULT_ROLE = "robot"
#: Caps, by code, on what a program may declare.
MAX_STEPS = 24
MAX_BUDGET_S = 1800.0
DEFAULT_BUDGET_S = 600.0
MAX_WAIT_S = 60.0
#: Per-step drive timeout: the native errand's carry drive uses 60 s. It is
#: `drive_to`'s patience when the program names none (issue #381) ...
DRIVE_TIMEOUT_S = 60.0
#: ...and the most it may name, s: a fresh robot's first walk to the loop's
#: far west corner took 246 s, finding the house's walls on the way (44 m of
#: route, 95 walked), and its second 117 s (`scripts/unknown_spike.py` at `quadruped-final`).
MAX_PATIENCE_S = 600.0


class Refused(ValueError):
  """A program that does not validate. `reasons` is every rule it broke,
  so an author fixes them all at once rather than one per refusal."""

  def __init__(self, reasons: list[str]) -> None:
    super().__init__("; ".join(reasons))
    self.reasons = list(reasons)


@dataclass(frozen=True)
class WorldFacts:
  """What validation needs to know about the world a program will run in:
  the box the map covers, the body's axes, senses and verbs."""

  bounds: tuple[float, float, float, float]    # x_min, y_min, x_max, y_max
  #: the motor-and-sensor level (procedure/axes.py, issue #166): what
  #: `move` may move and `read` may read, presence checked at run time
  axes: tuple[str, ...] = ()
  sensors: tuple[str, ...] = ()
  #: the verbs this world's BODY can run (issue #387), or None for every one
  verbs: tuple[str, ...] | None = None
  #: the task areas' tags a `find` may name (issue #419)
  places: tuple[int, ...] = ()
  #: the hands a hand's verb may name (issue #493: the humanoid's)
  hands: tuple[str, ...] = ()
  #: whether a verb may stand alone as an `if`'s test, its failing then
  #: read rather than ending the procedure (`lang`; Ben, 2026-10-19, for
  #: the practice: a move held back ended a lid's lowering half way, the
  #: jaws still on its knob). Off for the served world until it is turned
  #: on with its rule's words.
  verb_tests: bool = False
  #: whether an expression may call sin, cos, atan2 and sqrt (`lang.MATH`;
  #: #493's practice: an arc needs them), off for the served world as above
  math: bool = False
  #: whether a hand's move -- `reach`, `move_hand`, `pull` -- may stop on a
  #: reading (`until`, `above`, `below`; #554: a procedure reacts only
  #: between verbs, and a lid fell past its handle's stop inside one), off
  #: for the served world as above
  guarded: bool = False
  #: whether a source may hold several defs, a def take arguments and call
  #: another (`lang`, #560: Ben, 2026-11-04, modular and reusable
  #: subroutines -- three pieces of a table set as one def ran 5,458
  #: characters)
  calls: bool = False


@dataclass(frozen=True)
class Step:
  verb: str
  args: dict = field(default_factory=dict)

  def as_dict(self) -> dict:
    return {"verb": self.verb, **({"args": dict(self.args)} if self.args else {})}

  @classmethod
  def from_dict(cls, spec: dict) -> "Step":
    return cls(verb=str(spec.get("verb", "")), args=dict(spec.get("args") or {}))

  def describe(self) -> str:
    inner = ", ".join(f"{k}={v!r}" if isinstance(v, str) else f"{k}={v}"
                      for k, v in self.args.items())
    return f"{self.verb}({inner})"


@dataclass(frozen=True)
class Program:
  name: str
  roles: dict = field(default_factory=dict)      # role -> tuple[Step, ...]
  budget_s: float = DEFAULT_BUDGET_S

  @classmethod
  def single(cls, name: str, steps, budget_s: float = DEFAULT_BUDGET_S,
             role: str = DEFAULT_ROLE) -> "Program":
    """The default shape: one role."""
    return cls(name=name, roles={role: tuple(steps)}, budget_s=budget_s)

  def steps(self, role: str = DEFAULT_ROLE) -> tuple:
    return tuple(self.roles.get(role, ()))

  def first(self, verb: str, arg: str, role: str = DEFAULT_ROLE):
    """The first literal `arg` a `verb` step names, or None -- what the
    errand plumbing reads for its bookkeeping (the first tool fetched, the
    board drawn on)."""
    for step in self.steps(role):
      if step.verb == verb and arg in step.args:
        return step.args[arg]
    return None

  def as_dict(self) -> dict:
    return {"name": self.name, "budgetS": self.budget_s,
            "roles": {role: [s.as_dict() for s in steps]
                      for role, steps in self.roles.items()}}

  @classmethod
  def from_dict(cls, spec: dict) -> "Program":
    roles = spec.get("roles")
    if roles is None and "steps" in spec:
      # The single-role shorthand a person types; stored in the full shape.
      roles = {DEFAULT_ROLE: spec["steps"]}
    return cls(name=str(spec.get("name", "")),
               roles={str(r): tuple(Step.from_dict(s) for s in (steps or ()))
                      for r, steps in (roles or {}).items()},
               budget_s=float(spec.get("budgetS", DEFAULT_BUDGET_S)))

  def to_json(self) -> str:
    return json.dumps(self.as_dict(), sort_keys=True)

  @classmethod
  def from_json(cls, text: str) -> "Program":
    return cls.from_dict(json.loads(text))


# ---- the verbs ---------------------------------------------------------------


@dataclass(frozen=True)
class Arg:
  """One typed argument. `choices` is a list on `WorldFacts` (the world
  decides) a name must be one of -- or, for a number, a tag must be on
  (`places`, `plates`, `cubes`; #463) -- `lo`/`hi` bound a number, `bounds`
  says a pair is checked against the world's box. One with a `default` may
  be left out, and so may one that is `optional`: the verb does without it
  (#554, a move's stop condition)."""

  kind: str                    # "float" | "str"
  lo: float | None = None
  hi: float | None = None
  choices: str | None = None
  default: float | None = None
  optional: bool = False


@dataclass(frozen=True)
class Verb:
  name: str
  args: dict                   # arg name -> Arg
  run: Callable[..., Routine]  # (life, args) -> Routine returning a verdict
  doc: str


def _go(life, x: float, y: float, timeout: float, stop) -> Routine:
  """`Body.go_to_routine`, handing it `stop` only when there is one."""
  if stop is None:
    return life.body.go_to_routine(x, y, timeout=timeout)
  return life.body.go_to_routine(x, y, timeout=timeout, stop=stop)


def _stopped(life) -> bool:
  """Did the body's last walk end on its `stop` (issue #381)?"""
  from pluggybot.navigator import DRIVE_STOPPED
  rec = getattr(life.body, "last_drive", None)
  return bool(rec) and rec.get("why") == DRIVE_STOPPED


def _interrupt(life):
  """A walk's `stop` inside a job: the robot's own hazard row, resolved
  where the walk stands (`HubLifecycle.interrupted`, issue #116) -- a walk
  is a safe point at every step. None where there is nothing to ask."""
  return getattr(life, "interrupted", None)


def _patience(life, args: dict) -> float:
  """How long this walk may take: the program's `patience`, or
  `DRIVE_TIMEOUT_S` -- and never past the program's own budget
  (`life.step_until`, `run_verb`)."""
  s = float(args.get("patience", DRIVE_TIMEOUT_S))
  until = getattr(life, "step_until", None)
  if until is not None:
    s = min(s, max(0.0, float(until) - float(life.data.time)))
  return s


def _carried(life) -> dict:
  """What a walk's report says of a hand it carried (#682; Ben, 2026-11-12:
  the walk takes the carry pose itself and says so): `carried`, each hand
  held at the side a walk carries it at -- its gripper said it held
  something as the walk began -- and, of those, `dropped`, each whose
  gripper says it holds nothing now. None carried, nothing said."""
  hands = [s for s in ("left", "right") if life.body.carrying(s)]
  if not hands:
    return {}
  dropped = [s for s in hands if not life.body.gripped(s)]
  return {"carried": hands, **({"dropped": dropped} if dropped else {})}


def _walk_stopped(life, x: float, y: float) -> dict:
  """A walk the robot's own interrupt ended (issue #381): an abort, never
  a failure -- the runners stop the program as `interrupted`."""
  px, py, _ = life.body.pose
  short = round(math.hypot(x - px, y - py), 3)
  return {"ok": False, "stopped": "interrupted", "shortM": short,
          "reason": f"stopped on the way to ({x:g}, {y:g}) by its own interrupt, "
                    f"{short:.1f} m short, at ({px:.1f}, {py:.1f})"}


def _drive_to(life, args: dict) -> Routine:
  """A walk to a world point over the map, and over floor not yet on it
  (issue #381, `Navigator.OPTIMISTIC`): no route is written for it. ⚠ Not
  with a hand out that could not come home (#678, `_arm_out`)."""
  x, y = float(args["x"]), float(args["y"])
  refused = _arm_out(life, "walk")
  if refused is not None:
    return refused
  arrived = yield from _go(life, x, y, _patience(life, args), _interrupt(life))
  px, py, _ = life.body.pose
  short = round(math.hypot(x - px, y - py), 3)
  if arrived:
    return {"ok": True, "shortM": short, **_carried(life)}
  if _stopped(life):
    return {**_walk_stopped(life, x, y), **_carried(life)}
  # WHY it gave up, not only how far short (issue #350): "stopped 9.1 m
  # short of (22, 3)" was a route the planner could not make, and Rowan
  # read it as the pack -- then told Luca, and both declined the lab.
  why = life.drive_why(x, y)
  return {"ok": False, "shortM": short, "why": why, **_carried(life),
          "reason": f"did not arrive at ({x:g}, {y:g}): {why}, at ({px:.1f}, {py:.1f})"}


def _face(life, args: dict) -> Routine:
  refused = _arm_out(life, "walk")
  if refused is not None:
    return refused
  squared = yield from life.body.face_routine(float(args["heading"]))
  return {"ok": bool(squared), **_carried(life),
          **({} if squared else {"reason": (
            f"did not square up to heading {float(args['heading']):.2f} rad "
            "within its time")})}


def _wait(life, args: dict) -> Routine:
  yield from life.body.hold_routine(float(args["seconds"]))
  return {"ok": True}


#: How long a `find` searches when it is given no patience, s: a fresh
#: quadruped found the lab's feed plate from the facility's address in
#: 62-272 s (#419: eight address errors round the house, and the address
#: itself from three starts).
FIND_PATIENCE_S = 300.0
def _find(life, args: dict) -> Routine:
  """Find the task area carrying a tag (issue #419): where the robot saw it
  last, confirmed by its tag, or else searched for round (x, y) -- a job's
  address, in its map -- until the tag is in view. ok when it was seen;
  where it is rides the verdict (`at`), and the robot remembers it."""
  tag = int(args["tag"])
  x, y = float(args["x"]), float(args["y"])
  refused = _arm_out(life, "walk")
  if refused is not None:
    forget_readings(life, "find")
    return {**refused, "tag": tag}
  patience = _patience(life, {"patience": args.get("patience", FIND_PATIENCE_S)})
  rec = yield from life.body.find_tag_routine(tag, near=(x, y), patience=patience,
                                              stop=_interrupt(life))
  why = rec.get("why", "")
  # ...where it found it, read as `find.*` (#559): in its map, as it
  # remembers the place -- its face's way the best it knows (`Places.facing`);
  # ok only with that place kept, as its text says (#686's review: ok
  # beside a reading that said it failed)
  place = (life.body.places.get(tag) if rec.get("found") and life.body.places is not None
           else None)
  out = {"ok": place is not None, "tag": tag, "why": why,
         "seconds": rec.get("seconds"), "remembered": bool(rec.get("remembered")),
         **({"at": list(rec["at"])} if rec.get("at") else {})}
  if place is not None:
    _reports(life)["find"] = {"x": float(place.x), "y": float(place.y),
                              "facing": float(life.body.places.facing(tag)[0])}
  else:
    forget_readings(life, "find")
  if not out["ok"]:
    searched = (f"did not find tag {tag}"
                + (" where it was last seen, nor" if rec.get("remembered") else "")
                + f" round ({x:g}, {y:g}) in {float(rec.get('seconds') or 0):.0f} s")
    looks = f" ({int(rec.get('arounds') or 0)} looks round)"
    walk = rec.get("walk") or {}
    # ...and one whose walks took it nowhere to look from says the last
    # walk's words (#439): "having looked from every place near there", in
    # 23 s beside the other robot, was read as the board's tags moved
    walk_why = (f": {life.drive_why(*walk['goal'], record=walk)}"
                if walk.get("why") and walk.get("goal") else "")
    out["reason"] = (
      f"stopped looking for tag {tag} by its own interrupt" if why == "interrupted"
      else f"tag {tag} marks no place in this world" if why == "not a place"
      else f"{searched}, having looked from every place near there{looks}" if why == "not found"
      else f"{searched}, and no walk took it anywhere else to look from{looks}{walk_why}"
      if why == "gave up"
      else f"{searched} and ran out of time{looks}" if why == "out of time"
      else f"saw tag {tag} but does not remember where it is" if rec.get("found")
      else f"did not find tag {tag}: {why or 'no reason given'}")
  if rec.get("trail"):
    out["trace"] = f"find tag {tag}: {find_trace(rec)}"
  return {**out, **_carried(life)}


def _walk_trace(w: dict) -> str:
  """A walk's record (`Navigator.last_drive`) as a trace says it: how it
  ended, how far short and after how long, and the robot it was about."""
  if "why" not in w:
    return "?"
  peer = "".join(f" {k}={w[k]}" for k in ("peerAt", "peerM", "peerDown", "peerRests",
                                           "peerDead", "askedWay") if k in w)
  how = "arrived" if not w["why"] else f"{w['why']} {float(w.get('shortM') or 0):.2f} m short"
  return f"{how} after {float(w.get('seconds') or 0):.0f} s{peer}"


#: What a find's trace calls each walk's goal (`PlaceWalk.find_routine`).
FIND_WALKS = {"guess": "where it should be", "address": "the address", "viewpoint": "viewpoint"}


def find_trace(rec: dict) -> str:
  """A find's trail as one line of evidence (issue #439), the log's alone:
  each walk -- where to, and how it ended -- and each look round, a run of
  three or more viewpoints whose walks ended alike as one; then how the
  find ended and where its belief stood against the truth. A failed find
  said "ran out of time (1 looks round)", and nothing in the log told a long
  walk from the house from a robot lost in the lab."""
  parts: list[str] = []
  trail = list(rec.get("trail") or ())
  i = 0
  while i < len(trail):
    e = trail[i]
    if "lookedAt" in e:
      parts.append(f"looked round at ({e['lookedAt'][0]:.2f}, {e['lookedAt'][1]:.2f})")
      i += 1
      continue
    j = i
    while (e.get("to") == "viewpoint" and j + 1 < len(trail)
           and trail[j + 1].get("to") == "viewpoint" and trail[j + 1].get("why") == e.get("why")):
      j += 1
    if j - i >= 2:
      longest = max(float(w.get("seconds") or 0) for w in trail[i:j + 1])
      parts.append(f"{j - i + 1} viewpoints, each {e.get('why', '?') or 'arrived'} "
                   f"after {longest:.0f} s at most")
    else:
      for w in trail[i:j + 1]:
        gx, gy = w.get("goal") or (math.nan, math.nan)
        parts.append(f"to {FIND_WALKS.get(w.get('to'), w.get('to'))} ({gx:.2f}, {gy:.2f}) "
                     f"{_walk_trace(w)}")
    i = j + 1
  dx, dy, dyaw = rec.get("err") or (math.nan,) * 3
  parts.append(f"-> {rec.get('why', '?')} after {float(rec.get('seconds') or 0):.0f} s, "
               f"belief off {dx:+.0f},{dy:+.0f} mm {dyaw:+.1f} deg")
  return ", ".join(parts)


def _move(life, args: dict) -> Routine:
  """One axis to a setpoint, ramped (procedure/axes.py). The axis has to be
  present and the target inside its envelope, or the step fails and says
  which."""
  from pluggybot.procedure import axes
  axis = axes.AXES.get(args["axis"])
  if axis is None:
    return {"ok": False, "reason": f"no axis {args['axis']!r}"}
  target = float(args["target"])
  if not axis.lo <= target <= axis.hi:
    return {"ok": False, "reason": f"{axis.name} target {target} is outside "
                                    f"{axis.lo}..{axis.hi} {axis.unit}".rstrip()}
  moved = yield from axis.run(life, target)
  if moved is False:        # a joint that ran out of time (the quadruped's arm)
    return {"ok": False, "reason": f"{axis.name} did not reach {target} "
                                    f"{axis.unit} in time".rstrip()}
  return {"ok": True, "axis": axis.name, "target": target}


# ---- the hands (issue #493: the humanoid's) ---------------------------------
#
# Generic for any handle and any drawer, and none knows the chest (Ben's call
# 2 on #493): `close_hand` only shuts a gripper and says whether it holds
# something, a hand is moved by verbs of its own, and `spot` says where a tag
# is. WHERE to put a hand, from which way and with what, is the procedure's.
# Every place is the body's STANCE frame (`axes.in_stance`): from the floor
# between its feet, x ahead, y to its left, z up, m. A verb's report is
# read as sensors (`life.reports`, `axes.reported`).

#: How long a `spot` stands still before it looks, s: a look taken moving
#: smears its tag.
SPOT_STILL_S = 0.3
#: ...and the bows it looks from, rad, in turn until it sees its tag: as it
#: stands, then lower (#525: the humanoid's head is fixed on its trunk, and
#: standing, a cube on its counter is under its colour camera's view; bowed
#: 0.25 and 0.45 saw the cube and the mat beside it in every set-out).
SPOT_GAZES = (0.0, 0.25, 0.45)
#: What `spot`'s and `watch`'s camera takes in, deg, as their texts say (#660):
#: the colour imager's picture standing, pitched down with the depth camera,
#: its top and its bottom below level; and how far to either side of where
#: the head faces it reaches at its top and at its bottom, the picture being
#: tilted (a bearing is wider low down: the side angle in the camera's own
#: frame, 34.3, is only the top edge's). Off the model, never typed (a test
#: reads them against `color_eye`); flown standing 18.7 and 60.7, the stand
#: holding the head 0.3 deg back, and 34.3 in the camera's own frame.
SPOT_TOP_DEG = CHOSEN.depth_pitch_deg - COLOR_FOVY_DEG / 2
SPOT_BOTTOM_DEG = CHOSEN.depth_pitch_deg + COLOR_FOVY_DEG / 2


def _spot_side_deg(up: int) -> float:
  """How far to either side of where the head faces `spot`'s picture
  reaches, deg, at its top (`up` 1) or its bottom (-1): a corner of the
  picture turned down by the camera's pitch."""
  tall = math.tan(math.radians(COLOR_FOVY_DEG / 2))
  pitch = math.radians(CHOSEN.depth_pitch_deg)
  return math.degrees(math.atan(tall * PICTURE_W / PICTURE_H
                                / (math.cos(pitch) + up * tall * math.sin(pitch))))


SPOT_SIDE_TOP_DEG, SPOT_SIDE_BOTTOM_DEG = _spot_side_deg(1), _spot_side_deg(-1)
#: ...the bottom's, in the text, said in tens: ⚠ a verb's text may not carry
#: the number a block's tag is known by, and 52.7 would read 53
#: (`test_no_hand_verb_knows_the_chest`).
SPOT_SIDE_BOTTOM_TENS = 10 * math.floor(SPOT_SIDE_BOTTOM_DEG / 10)
#: ...the picture's top over the floor IN THE STANCE FRAME, m, where the
#: robot reads every place: at x = 0 (the camera's height carried back to the
#: soles' middle, 4.5 cm behind it), falling this much a metre of x; and how
#: far ahead a small tag level with the shoulders may stand, the whole of it
#: in the picture, m. ⚠ Told from the camera, the 120 mm wall tags' 0.13 m put
#: a robot that measured it from its stance's origin inside the camera's 0.10
#: m near plane (#660's review: flown, those tags read with the camera 11-13 cm
#: off a bare wall, at 8.6 not, at 16 not): the text says "nearer still".
SPOT_TOP_DROP = math.tan(math.radians(SPOT_TOP_DEG))
SPOT_TOP_AT_FEET_M = color_camera_z() + SPOT_TOP_DROP * color_camera_ahead()
SPOT_SHOULDER_IN_M = (SPOT_TOP_AT_FEET_M - stand_height() - CHOSEN.shoulder_z
                      - BLOCK_TAG_SIZE / 2) / SPOT_TOP_DROP
#: How far a small block's tag reads, m of range, what is in the picture
#: (#660; flown on `home_humanoid`, osmesa, two noise draws): the tower's
#: 26 mm blocks on the floor, the picture's middle, read reliably out to
#: 1.5-1.6 m of range, thinning past that (one read at 1.74 m, none beyond).
#: A prop's knob on a counter leaves the
#: picture's top first: 1.1-1.4 m of range, 1.15-1.4 m off the handle.
SPOT_BLOCK_READS_M = 1.5
#: How long a `reach` waits for its hand to get there, s, and the most a
#: `move_hand` or a `pull` may.
REACH_PATIENCE_S, HAND_PATIENCE_S = 20.0, 30.0
#: The stance frame's box a hand's place is checked against, m (x, y, z).
HAND_BOX = ((-0.5, 1.0), (-0.8, 0.8), (0.0, 2.0))
#: A hand's speed when carried or pulled, m/s: the reach's own
#: (`humanoid.reach.HOLD_SPEED` is the default; `HAND_SPEED` the most).
HAND_V = (0.01, 0.06, 0.2)
#: The most a `pull` asks, N: what the standing body holds at a hand (Ben,
#: 2026-10-25, #544). Held 3 s at a counter's height it stood to 30 N
#: pulling back and 20 pushing away, falling at 40 and 30; it feels no pull
#: until the body moves (SimNotes, "Holding the pull verb's 60 N at a
#: hand"). Its text tells the robot pushing holds about two thirds of it.
PULL_MOST_N = 20.0


def _reports(life) -> dict:
  if getattr(life, "reports", None) is None:
    life.reports = {}
  return life.reports


def _grip_turn(life, args: dict):
  """A reach's grip turn, the believed world frame, off its fingers' way and
  its jaws' (stance frame): None for any, where both are left 0; or why
  not."""
  from pluggybot.procedure import axes
  a = [float(args.get(k, 0.0)) for k in ("fx", "fy", "fz")]
  c = [float(args.get(k, 0.0)) for k in ("jx", "jy", "jz")]
  na, nc = math.sqrt(sum(v * v for v in a)), math.sqrt(sum(v * v for v in c))
  if na == 0.0 and nc == 0.0:
    return None, ""
  if na == 0.0 or nc == 0.0:
    return None, "give both ways -- the fingers' (fx, fy, fz) and the jaws' (jx, jy, jz) -- or neither"
  a = [v / na for v in a]
  dot = sum(x * y for x, y in zip(a, c))
  x = [cv - dot * av for cv, av in zip(c, a)]
  nx = math.sqrt(sum(v * v for v in x))
  if nx < 0.1 * nc:
    return None, "the jaws must close across the fingers' way, not along it"
  x = [v / nx for v in x]
  z = [-v for v in a]
  y = [z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]]
  cols = [axes.from_stance(life, v, point=False) for v in (x, y, z)]
  return [[cols[j][i] for j in range(3)] for i in range(3)], ""


def _hand_place(life, args: dict) -> tuple[float, float, float]:
  from pluggybot.procedure import axes
  return axes.from_stance(life, (float(args["x"]), float(args["y"]), float(args["z"])))


def _spot(life, args: dict) -> Routine:
  """Look for a tag in the head's colour camera, standing still -- as it
  stands, then bowing lower until it is seen (`SPOT_GAZES`), standing up
  again after (#525): where its face is -- the one seen most square-on --,
  which way it looks and which way its edges run, in `spot.*`. ok when it
  is seen."""
  from pluggybot.procedure import axes
  tag = args["tag"]
  if float(tag) != int(tag):
    forget_readings(life, "spot")
    return {"ok": False, "reason": f"a tag is a whole number, not {tag}"}
  got, bowed, seen = None, False, None
  for down in SPOT_GAZES:
    if down:
      yield from life.body.gaze_routine(down)
      bowed = True
    yield from life.body.hold_routine(SPOT_STILL_S)
    got = life.body.spot(int(tag))
    if got is not None:
      # in the stance frame as it is seen: the bow moves no foot
      seen = (axes.in_stance(life, (got["x"], got["y"], got["z"])),
              axes.in_stance(life, got["normal"], point=False),
              axes.in_stance(life, got["edge"], point=False))
      break
  if bowed:
    yield from life.body.gaze_routine(0.0)
  if seen is None:
    forget_readings(life, "spot")
    # what the bows could not change, the picture's top (#660), and short:
    # History's line says each false test's reason inside its 400 characters
    return {"ok": False, "tag": int(tag),
            "reason": (f"tag {int(tag)} is not in your colour camera's picture from where you "
                       "stand, or too near or too far to read: it shows nothing higher than "
                       f"{SPOT_TOP_DEG:.0f} degrees below level")}
  (x, y, z), (nx, ny, nz), (ex, ey, ez) = seen
  _reports(life)["spot"] = {"x": x, "y": y, "z": z, "nx": nx, "ny": ny, "nz": nz,
                            "ex": ex, "ey": ey, "ez": ez, "range": float(got["range"])}
  return {"ok": True, "tag": int(tag), "at": [round(v, 4) for v in (x, y, z)]}


#: `watch` waits this long at most for its first reading, s, looking every
#: `WATCH_POLL_S`: the camera's next frame, its latency (`perception.watch`)
#: and room.
WATCH_FIRST_S, WATCH_POLL_S = 0.5, 0.01


def _watch(life, args: dict) -> Routine:
  """THE WATCH (#554's stage 2; docs/Hands.md): the head's colour camera
  kept reading one tag while the procedure goes on (`Body.watch`) -- until
  another `watch`, or the procedure's end (`lang.run_procedure_routine`) --
  what it reads riding `watch.*`. It waits for its first reading and is ok
  when that picture shows the tag; where it does not, it watches on all the
  same -- which a procedure sees only where the verb is an `if`'s test, a
  failed verb ending it otherwise, and its watch with it. One that fails
  before it begins ends the watch before it (`forget_readings`)."""
  from pluggybot.procedure import axes
  tag = args["tag"]
  if float(tag) != int(tag):
    forget_readings(life, "watch")
    return {"ok": False, "reason": f"a tag is a whole number, not {tag}"}
  life.body.watch(int(tag))
  waited = 0.0
  while life.body.sighting() is None and waited < WATCH_FIRST_S:
    yield from life.body.hold_routine(WATCH_POLL_S)
    waited += WATCH_POLL_S
  got = life.body.sighting()
  if got is None:
    # ...no reading by the end of its wait: it watches on, and until one
    # comes, reading it says this call failed
    _unread(life, "watch")
  if got is None or not got["seen"]:
    return {"ok": False, "tag": int(tag),
            "reason": f"tag {int(tag)} is not in view of your colour camera now"}
  last = got["last"]
  at = axes.in_stance(life, (last["x"], last["y"], last["z"]))
  return {"ok": True, "tag": int(tag), "at": [round(v, 4) for v in at]}


#: `measure` (issue #493, an arm of its own -- sight beside feel): how far
#: round the point named its flat top is looked for, m, and its text says
#: so (#630). Under 0.35 m the far edge of a 0.26 m deep top was clipped;
#: at 0.35 and 0.45 the four set-outs read their tops whole.
MEASURE_NEAR_M = 0.45
#: ...and an end read off points whose median stands within this of the
#: window's rim, m, is the window's, never the top's: no reading (#630:
#: from (0.9, 0) a counter's near end read the rim, 0.453-0.456 m, however
#: the robot stepped). At the drawer's set-outs 0 and 3, the 24 ends the
#: window cut stood 0.9-2.0 cm inside it, the 28 it held whole 3.9 cm and
#: more; one it holds only in part falls either side, and 8 of 42 went
#: with the window's, 4 of them read true (SimNotes, "Where `measure`
#: looks").
MEASURE_RIM_M = 0.03
#: Its ends: the stance frame's axis each lies along, whether it is that
#: axis's far end, and which way the top runs on past it.
MEASURE_ENDS = (("near", 0, False, "toward you"), ("far", 0, True, "away from you"),
                ("left", 1, True, "to your left"), ("right", 1, False, "to your right"))
#: ...and an end counts only where the camera saw past it (#636,
#: `box.past_end`): told by the walks' most common word, the cautious one
#: first in a tie -- `cut` (the picture ended first: the drawer's set-outs'
#: near ends cut by its bottom edge read 99-100 % of it, the counter's left
#: end at the picture's corner, a true one, 29-34 %), `runs` (its own top's
#: pixels go on past where it reads, the look stopping short of it),
#: `taller` (the end of the top's free surface, reading so, Ben, 2026-11-08;
#: the counter behind the cabinet 100 %), `lower` (the top stopped there).
MEASURE_WORDS = ("cut", "runs", "taller", "lower")


def _measure(life, args: dict) -> Routine:
  """The flat top of what stands at (x, y), off the head's depth camera at
  the part's own 848 x 480 (`Body.depth_pictures`): its height and where
  it ends -- nearest and farthest ahead, leftmost and rightmost -- in the
  stance frame, in `object.*`. ok when a flat top is seen. ⚠ OUR CODE'S
  PERCEPTION, a stated prior (`perception.box`): the highest flat face
  among the points round (x, y), and its edges. ⚠ AN END COUNTS ONLY WHERE
  THE CAMERA SAW PAST IT (#636): where the top runs on past the window
  (#630) or past where it read, off the picture, or out of the camera's
  range, that end is no reading, and reading it says why
  (`axes.measured`); so is every end after a look that saw no top. Where
  something taller stands at an end, it reads as where the top's free
  surface ends, and `object.<end>_taller` says so."""
  import numpy as np
  from pluggybot.perception import box
  from pluggybot.procedure.axes import UNREAD
  x, y = float(args["x"]), float(args["y"])
  pictures = yield from life.body.depth_pictures(box.DEPTH_W, box.DEPTH_H, box.DEPTH_MIN_Z,
                                                 frames=box.FRAMES, apart=box.APART_S)
  look = box.Look.of(pictures)
  bx, by, yaw = life.body.stance
  c, s_ = math.cos(yaw), math.sin(yaw)
  pts = look.points
  dx, dy = pts[:, 0] - bx, pts[:, 1] - by
  st = np.column_stack([c * dx + s_ * dy, -s_ * dx + c * dy, pts[:, 2]])
  which = np.flatnonzero(np.hypot(st[:, 0] - x, st[:, 1] - y) < MEASURE_NEAR_M)
  near = st[which]
  top = box.top_height(near) if len(near) else None
  flat = box.on_top(near, top) if top is not None else np.zeros(len(near), dtype=bool)
  on, on_which = near[flat], which[flat]
  read = {key: box.edge_read(on[:, axis], far=far) for key, axis, far, _ in MEASURE_ENDS}
  if top is None or None in read.values():
    said = f"your last `measure` saw no flat top round ({x:g}, {y:g}) from where you stood"
    _reports(life)["measure"] = {UNREAD: dict.fromkeys(("top", *read), said)}
    return {"ok": False, "reason": f"no flat top seen round ({x:g}, {y:g}) from where you stand"}
  got, past, unseen, taller = {"top": top}, {}, {}, []
  for key, axis, far, way in MEASURE_ENDS:
    end, strip = read[key]
    rim = MEASURE_NEAR_M - float(np.median(np.hypot(on[strip, 0] - x, on[strip, 1] - y)))
    if rim < MEASURE_RIM_M:
      past[key] = (f"your last `measure` looked within {MEASURE_NEAR_M:g} m of ({x:g}, {y:g}), "
                   f"and the top it saw runs on past that {way}")
      continue
    # the end's outward way and where it reads along it, the believed
    # world frame, as its points are
    along = np.array([c, s_, 0.0] if axis == 0 else [-s_, c, 0.0])
    way_out = along if far else -along
    at = np.array([bx, by, 0.0]) @ way_out + (end if far else -end)
    saw = box.past_end(look, on_which[strip], way_out, top, at)
    word = max(MEASURE_WORDS, key=saw.__getitem__)
    if saw["reach"]:
      unseen[key] = (f"your last `measure` saw the top run on past your depth camera's "
                     f"{pictures[0].reach:g} m reach {way}")
    elif saw["near"]:
      unseen[key] = (f"your last `measure` saw the top run on nearer than your depth camera "
                     f"reads, {pictures[0].near:g} m, {way}")
    elif word == "cut":
      unseen[key] = f"your last `measure` saw the top run off your depth camera's picture {way}"
    elif word == "runs":
      past[key] = f"your last `measure` saw the top run on past where it read its end {way}"
    else:
      got[key] = end
      if word == "taller":
        taller.append(key)
  _reports(life)["measure"] = {**got, UNREAD: {**past, **unseen}, "taller": taller}
  return {"ok": True, **{k: round(v, 3) for k, v in got.items()},
          **({"runsOn": list(past)} if past else {}),
          **({"unseen": list(unseen)} if unseen else {}),
          **({"taller": taller} if taller else {})}


#: What a hand's verb FELT (issue #493, opening by feel): the robot's own
#: senses of that hand -- where its grip is, the force at it, whether it
#: holds -- read this often while the verb runs, sim s, and at its end; at
#: most `FELT_MAX` of them kept a verb, evenly. Shown to the practice's
#: mind under each verb: told only how a verb ended, a mind shut its jaws
#: on nothing three tries running, and never learned which pull dropped a
#: lid.
FELT_EVERY_S, FELT_MAX = 0.25, 8


def _sensed(life, hand: str, t0: float) -> dict:
  from pluggybot.procedure import axes
  at = axes.in_stance(life, life.body.hand_at(hand))
  force = axes.in_stance(life, life.body.hand_force(hand), point=False)
  return {"t": round(float(life.data.time) - t0, 2), "at": [round(float(v), 3) for v in at],
          "force": [round(float(v), 1) for v in force], "held": bool(life.body.gripped(hand))}


#: A hand's move that stopped on its reading (#554, `until`): why its body
#: says it ended (`Body.stop_hand`), and what its verdict says it stopped on.
UNTIL = "until"


def _guard(args: dict) -> tuple[str, str, float] | None:
  """A hand's move's stop condition, off its arguments: (the sensor,
  "above" or "below", the number), or None where it has none."""
  if not args.get("until"):
    return None
  side = "above" if "above" in args else "below"
  return str(args["until"]), side, float(args[side])


def _guarded_routine(life, hand: str, guard, routine: Routine) -> Routine:
  """`routine` driven as `yield from` would, its STOP CONDITION read as the
  hand moves (#554): `guard`'s sensor read once the routine has begun --
  a hand's force from `Sensor.from_s` into it, as a move's start reads the
  move (#631) -- and then once each `Sensor.every_s` of its -- never
  oftener, as its part reads it -- and the first reading past its number
  stops `hand`'s move where it is (`Body.stop_hand`), once a move is under
  way to stop. A reading the sensor does not have yet (a watched tag not
  yet seen) is no reading past it. Returns (what it returned, and the
  reading that stopped it and how long into the verb, s -- or, where the
  sensor never had a reading, why -- or None)."""
  if guard is None:
    return (yield from routine), None
  from pluggybot.procedure import axes
  name, side, number = guard
  sensor = axes.SENSORS[name]
  reading = sensor.live or sensor.read
  # physics steps a reading, rounded UP: never oftener than its part reads
  step = float(life.model.opt.timestep)
  every = max(1, math.ceil(sensor.every_s / step - 1e-6))
  first = math.ceil(sensor.from_s / step - 1e-6)
  t0, n, stopped, unread = float(life.data.time), 0, None, ""
  heard = False
  sent, exc = None, None
  while True:
    try:
      cmd = routine.throw(exc) if exc is not None else routine.send(sent)
    except StopIteration as stop:
      return stop.value, (stopped if stopped is not None or heard or not unread
                          else {"unread": unread})
    exc = None
    if stopped is None and n >= first and (n - first) % every == 0:
      try:
        read = float(reading(life))
      except axes.NoReading as no:
        unread = str(no)
      else:
        heard = True
        if (read > number if side == "above" else read < number) and life.body.stop_hand(hand):
          stopped = {"read": read, "s": float(life.data.time) - t0}
    n += 1
    try:
      sent = yield cmd
    except GeneratorExit:
      routine.close()
      raise
    except BaseException as e:  # noqa: BLE001 -- thrown into the routine, as `yield from` would
      exc, sent = e, None


def _until_said(guard, stopped) -> str:
  """What a move that stopped on its reading says of it: `left.z read
  1.052 (above 1.05) 1.73 s in` -- the reading to as many figures as tell
  it from the number (to four, `time read 1.15 (above 1.15)` was said) --
  or, where it never had a reading, why: `watch.z above 1.05 never had a
  reading: tag 53 has not been seen since `watch` began`."""
  name, side, number = guard
  if stopped is None:
    return f"{name} {side} {number:.4g}"
  if "unread" in stopped:
    why = stopped["unread"].split(": ", 1)[-1]
    return f"{name} {side} {number:.4g} never had a reading: {why}"
  read = stopped["read"]
  digits = next((n for n in range(4, 17) if f"{read:.{n}g}" != f"{number:.{n}g}"), 17)
  return f"{name} read {read:.{digits}g} ({side} {number:.4g}) {stopped['s']:.2f} s in"


def _said(guard, stopped, why) -> dict:
  """A move's verdict's word on its stop condition: what it read, where
  the move stopped on it -- never where something else ended it in the
  same step (#554's review: a fall the step after the stop said both) --
  or why it never had a reading."""
  if stopped is None or ("unread" not in stopped and why != UNTIL):
    return {}
  return {"until": _until_said(guard, stopped)}


def _stopped_reading(life, verb: str, stopped: bool) -> None:
  """`<verb>.stopped` (#554): 1 after a move its stop condition stopped, 0
  after one that ended otherwise."""
  _reports(life).setdefault("stopped", {})[verb] = 1.0 if stopped else 0.0


def _unread(life, verb: str) -> None:
  """`verb`'s report, after a call of it that failed: no reading, and
  reading it says so (`axes.unread`), never that it has not run (#686)."""
  from pluggybot.procedure import axes
  _reports(life)[verb] = {axes.UNREAD: f"your last `{verb}` failed"}


def forget_readings(life, verb: str) -> None:
  """A call that failed before it could report -- refused before it ran
  (`lang`, a computed argument), closed by a fall or a death (`run_verb`),
  or a `spot`, a `find` or a `pull` that read false: what its verb last
  reported would read as this call's, so it goes (`_unread`) -- a retry
  computes just such an argument, as a distance left of what `pull.moved`
  read. A `watch`'s readings are the running watch's, so that watch ends
  -- here, where a refused call, which never reaches `_watch`, comes too.
  Its stop condition's reading is a move's that ended otherwise (#650's
  review)."""
  _unread(life, verb)
  if verb == "watch":
    life.body.watch(None)
  reports = _reports(life)
  if verb in reports.get("stopped", {}):
    _stopped_reading(life, verb, False)


def end_run(life, watching: bool) -> None:
  """A procedure's end (`lang.run_procedure_routine`): the words its failed
  calls left (`_unread`) end with it, so the next one's read of a verb it
  never called says it has not run (#686's review: a served robot keeps its
  reports from one run to the next) -- what a verb last reported stays, as
  ever -- and, `watching`, its watch ends with it (#554)."""
  from pluggybot.procedure import axes
  reports = _reports(life)
  for verb in [v for v, rep in reports.items() if isinstance((rep or {}).get(axes.UNREAD), str)]:
    del reports[verb]
  if watching:
    life.body.watch(None)


def _thinned(felt: list, n: int = FELT_MAX) -> list:
  """At most `n` of `felt`, evenly, its first and last kept."""
  if len(felt) <= n:
    return felt
  keep = sorted({round(i * (len(felt) - 1) / (n - 1)) for i in range(n)})
  return [felt[i] for i in keep]


def _felt_routine(life, hand: str, routine: Routine) -> Routine:
  """`routine` driven as `yield from` would, `hand`'s own senses read as it
  begins, every `FELT_EVERY_S` and as it ends: returns (what it returned,
  what was felt, `_thinned`)."""
  t0 = float(life.data.time)
  felt = [_sensed(life, hand, t0)]
  due = t0 + FELT_EVERY_S
  sent, exc = None, None
  while True:
    try:
      cmd = routine.throw(exc) if exc is not None else routine.send(sent)
    except StopIteration as stop:
      felt.append(_sensed(life, hand, t0))
      return stop.value, _thinned(felt)
    exc = None
    if float(life.data.time) >= due:
      felt.append(_sensed(life, hand, t0))
      due = float(life.data.time) + FELT_EVERY_S
    try:
      sent = yield cmd
    except GeneratorExit:
      routine.close()
      raise
    except BaseException as e:  # noqa: BLE001 -- thrown into the routine, as `yield from` would
      exc, sent = e, None


#: What a `reach` that did not get there says, by the body's why.
REACH_WHY = {"out of reach": "no posture you hold puts your {hand} hand there",
             "stopped short": "your {hand} hand stopped short on its way: out of reach from "
                              "how you truly stand, or pushing on something",
             "fell": "you fell on the way there, and let go",
             "out of time": "your {hand} hand did not get there in time"}


#: ...and one that kept off another robot's body (#678; `Body.hand_kept`,
#: `Body.arm_kept`), by what it refused: where that robot stands, from the
#: robot's own middle and said of its own body, what two minds work it out
#: from -- the other hand, a step away, a `tell` (Ben, 2026-11-12).
ARM_KEPT_OFF = {
  "reach": "your {hand} arm's way there would meet another robot, its middle {m:.2f} m from "
           "yours {where}: no way your arm takes there keeps clear of its body, so your arm "
           "did not move",
  "back": "your {hand} arm could get there, but its way home from there would meet another "
          "robot, its middle {m:.2f} m from yours {where}: no way home keeps clear of its "
          "body, so your arm did not move",
  "withdraw": "your {hand} arm's way home would meet another robot, its middle {m:.2f} m from "
              "yours {where}: no way home keeps clear of its body, so your hand stays where it "
              "is, its gripper as it was",
  "step": "your {hand} arm is out, and its way home would meet another robot, its middle "
          "{m:.2f} m from yours {where}: a step brings your arms home first, so you did not "
          "step",
  "walk": "your {hand} arm is out, and its way home would meet another robot, its middle "
          "{m:.2f} m from yours {where}: a walk brings your arms home first, so you did not "
          "walk",
  # ...and one a walk would carry (#682): its way to your side
  "step carried": "your {hand} hand holds something, and its way to your side, where a step "
                  "carries it, would meet another robot, its middle {m:.2f} m from yours "
                  "{where}: so you did not step",
  "walk carried": "your {hand} hand holds something, and its way to your side, where a walk "
                  "carries it, would meet another robot, its middle {m:.2f} m from yours "
                  "{where}: so you did not walk"}
#: ...and, after a reach's, what the other hand's ways out and home would do
#: (`Reach.clear_way` judges both: "way there" alone, beside a doc saying
#: both, left 3 of 12 asked unsure a left reach would go)
OTHER_HAND_KEPT = {True: "; your {other} hand's ways there and home keep clear of it",
                   False: "; nor does your {other} hand's",
                   None: "; your {other} hand does not reach there"}


def _where(kept: dict) -> str:
  """Where another robot stands, as the robot says it of its own body: from
  its stance's middle, a quarter of the way round each."""
  a = math.degrees(math.atan2(kept["y"], kept["x"]))
  return ("ahead of you" if abs(a) <= 45 else "on your left" if 45 < a <= 135
          else "on your right" if -135 <= a < -45 else "behind you")


def _arm_out(life, walk: str) -> dict | None:
  """A walk refused before it moves (#678): a hand out whose every way home
  would meet another robot's body (`Body.arm_kept`) -- a walk brings the
  arms home first. Its verdict, by the verb (`ARM_KEPT_OFF`), else None."""
  kept = life.body.arm_kept()
  if kept is None:
    return None
  said = ARM_KEPT_OFF[f"{walk} carried" if kept.get("carried") else walk]
  return {"ok": False, "why": "arm out", "hand": kept["hand"],
          "reason": said.format(where=_where(kept), **kept)}


def _reach(life, args: dict) -> Routine:
  """A hand's grip point to (x, y, z), its fingers along (fx, fy, fz) and
  its jaws closing along (jx, jy, jz) -- or at whatever turn the arm finds,
  both left 0 -- out short of it, then in along the fingers' way. ok once
  it holds there, or where its stop condition stopped it (#554)."""
  hand = args["hand"]
  _stopped_reading(life, "reach", False)
  rot, bad = _grip_turn(life, args)
  if bad:
    return {"ok": False, "reason": bad}
  place = _hand_place(life, args)
  guard = _guard(args)
  (ok, stopped), felt = yield from _felt_routine(life, hand, _guarded_routine(
    life, hand, guard, life.body.reach_routine(
      hand, place, rot, stiffness=float(args.get("stiffness", _STIFF.default)),
      patience=REACH_PATIENCE_S, refuse=True)))
  why = "" if ok else life.body.hand_why(hand)
  _stopped_reading(life, "reach", why == UNTIL)
  said = _said(guard, stopped, why)
  if ok:
    return {"ok": True, "hand": hand, **said, "felt": felt}
  if why == UNTIL:
    return {"ok": True, "hand": hand, "why": why, **said, "felt": felt}
  reason = REACH_WHY.get(why, "your {hand} hand did not get there").format(hand=hand)
  other = {"left": "right", "right": "left"}.get(hand)
  kept = life.body.hand_kept(hand) if why == "kept off" else None
  if kept is not None:
    reason = (ARM_KEPT_OFF["back" if kept.get("back") else "reach"].format(
      hand=hand, where=_where(kept), **{k: v for k, v in kept.items() if k != "hand"})
      + OTHER_HAND_KEPT[kept.get("other")].format(other=other))
  if why == "out of reach" and rot is not None and life.body.hand_reaches(hand, place):
    # ...turned as asked; and whether the TURN is what is out of reach,
    # off the same plan (#493: a mind asked both hands for a turn neither
    # makes there, five tries running, told only that neither could)
    reason += (" turned as asked; turned another way it can (leave fx..jz at 0 and your "
               "arm finds a turn)")
  if why == "out of reach" and other is not None:
    # what its body's own model says of the other hand (#493: 18 of 27
    # first tries took the right hand to a knob on the robot's left) --
    # turned another way too: "nor could your left hand, as asked" was read
    # as out of reach standing, and an arm stepped beside the box 59 tries
    if life.body.hand_reaches(other, place, rot):
      reason += f"; your {other} hand could reach it as asked"
    elif rot is not None and life.body.hand_reaches(other, place):
      reason += (f"; nor could your {other} hand as asked, though turned another way it can"
                 + ("" if "your arm finds a turn" in reason
                    else " (leave fx..jz at 0 and your arm finds a turn)"))
    else:
      reason += f"; nor could your {other} hand, as asked"
  return {"ok": False, "hand": hand, "why": why, "reason": reason, **said, "felt": felt}


#: What a `move_hand` that did not arrive says, by its report's why.
MOVE_WHY = {"out of reach": "your {hand} hand could reach no further that way from any "
                            "posture you hold",
            "held back": "your {hand} hand was held back short of it: what it holds or "
                         "pushes on stopped it",
            "feet moved": "your feet moved under the force, so it stopped there",
            "not out": "your {hand} hand is not out: `reach` with it first",
            "fell": "you fell on the way, and let go",
            "out of time": "your {hand} hand did not get there in time"}


def _hand_report(life, hand: str, rep: dict) -> dict:
  from pluggybot.procedure import axes
  out = {k: rep[k] for k in ("why", "seconds", "moved") if k in rep}
  if "dx" in rep:
    out["by"] = [round(v, 4) for v in axes.in_stance(life, (rep["dx"], rep["dy"], rep["dz"]),
                                                      point=False)]
  return out


def _move_hand(life, args: dict) -> Routine:
  """A hand that is out carried to (x, y, z) at `speed`, m/s, at
  `stiffness`, its turn kept -- or turned to the one given as `reach`
  takes it -- along a straight line, or through the points `via` gave it
  first, as ONE motion from one posture that follows them all, each point
  turning it as it asks (#493, merged #525); the path is then empty. ok
  when it arrives, or where its stop condition stopped it (#554)."""
  hand = args["hand"]
  _stopped_reading(life, "move_hand", False)
  rot, bad = _grip_turn(life, args)
  if bad:
    return {"ok": False, "hand": hand, "reason": bad}
  points = _paths(life).pop(hand, [])
  end = _hand_place(life, args)
  places = [p for p, _ in points]
  # a point with no turn keeps the one before -- the hand's own, first
  turns = None
  if rot is not None or any(r is not None for _, r in points):
    turns = [r for _, r in points] + [rot]
  speed = float(args.get("speed", _SPEED.default))
  length = sum(math.dist(a, b) for a, b in zip(places + [end], (places + [end])[1:]))
  guard = _guard(args)
  (rep, stopped), felt = yield from _felt_routine(life, hand, _guarded_routine(
    life, hand, guard, life.body.move_hand_routine(
      hand, end, stiffness=float(args.get("stiffness", _STIFF.default)), speed=speed,
      patience=HAND_PATIENCE_S + (2.0 * length / speed if places else 0.0),
      path=places or None, turns=turns)))
  _stopped_reading(life, "move_hand", rep.get("why") == UNTIL)
  out = {"hand": hand, **_hand_report(life, hand, rep), "felt": felt,
         **({"points": len(points)} if points else {})}
  said = _said(guard, stopped, rep.get("why"))
  if rep.get("why") in (UNTIL, "arrived"):
    return {"ok": True, **out, **said}
  return {"ok": False, **out, **said,
          "reason": MOVE_WHY.get(rep.get("why"), "your {hand} hand did not get there").format(hand=hand)}


def _pull(life, args: dict) -> Routine:
  """THE PULL, a FORCE MOVE (issue #525; docs/Hands.md): a hand that is out
  pulled -- or pushed -- along (dx, dy, dz), of any length, with at most
  `force` N as the hand feels it, at most `speed` m/s; across that way it
  goes where what it holds pushes it, and its turn gives to what it holds
  but about the jaws' own line -- or, `give` 0, it holds its line and its
  turn. Until it has gone
  `distance` m along its way, it is held fast (pushing
  with its whole force and not moving), its grip lets go, or it can reach
  no further -- or its stop condition stops it (#554) -- then it holds
  there. How far it went, the hardest it pushed, the way it was going at its
  end and whether it ended held fast ride `pull.*`. ok unless the hand was
  not out, the way is all 0 or it fell -- or `lang` refused an argument
  first -- and its text says so (#650: a served robot is shown no step's
  `why`, only what it reads)."""
  from pluggybot.procedure import axes
  hand = args["hand"]
  _stopped_reading(life, "pull", False)
  d = (float(args["dx"]), float(args["dy"]), float(args["dz"]))
  if math.sqrt(sum(v * v for v in d)) == 0.0:
    forget_readings(life, "pull")
    return {"ok": False, "reason": "give the pull a way: (dx, dy, dz) are all 0"}
  guard = _guard(args)
  (rep, stopped), felt = yield from _felt_routine(life, hand, _guarded_routine(
    life, hand, guard, life.body.pull_routine(
      hand, axes.from_stance(life, d, point=False), float(args["force"]),
      float(args["distance"]), speed=float(args.get("speed", _SPEED.default)),
      patience=HAND_PATIENCE_S, give=float(args.get("give", _GIVE.default)) >= 0.5)))
  _stopped_reading(life, "pull", rep.get("why") == UNTIL)
  out = {"hand": hand, **_hand_report(life, hand, rep), "force": rep.get("force"),
         "felt": felt, **_said(guard, stopped, rep.get("why"))}
  if rep.get("why") in ("not out", "fell"):
    # ...and no reading: the last pull's would read as this one's
    forget_readings(life, "pull")
    return {"ok": False, **out,
            "reason": MOVE_WHY.get(rep["why"], "the pull did not begin").format(hand=hand)}
  way = axes.in_stance(life, rep.get("direction") or (0.0, 0.0, 0.0), point=False)
  _reports(life)["pull"] = {"moved": float(rep.get("moved", 0.0)),
                            "force": float(rep.get("force") or 0.0),
                            "held": 1.0 if rep.get("why") == "held fast" else 0.0,
                            "dx": way[0], "dy": way[1], "dz": way[2]}
  return {"ok": True, **out}


#: Why a step stopped short, as its record says it (`Body.step_routine`).
STEP_WHY = {"too short": "your step was too short to walk",
            "turned short": "your turn stopped short of where you asked",
            "blocked": "your step got nowhere: something stands in your way",
            "fell": "you fell",
            "out of time": "your step ran out of time"}
#: ...and one blocked where it kept off another robot (issue #670;
#: `HumanoidBody._kept_off`), by how: its middle kept off that robot's, or
#: that robot's body in its way.
STEP_KEPT_OFF = {
  "keep": "your step stopped short of another robot: a step keeps your middle {keepM:.2f} m "
          "from its middle, and within that goes only further from it ({m:.2f} m now)",
  "body": "your step stopped short of another robot: its body was {m:.2f} m from your "
          "middle in the way you were stepping"}


def _step(life, args: dict) -> Routine:
  """THE STEP (issue #525; docs/Hands.md): its hands home -- a hand that
  holds something carried at its side (#682) -- the
  body turns `turn` rad on the spot and walks `ahead` m and `left` m in the
  frame it then stands in, and stands -- its feet where the hands' work is. Where its stance
  went, in the stance frame it stood in before, rides `step.*` whatever
  stopped it. ok when it stepped."""
  refused = _arm_out(life, "step")
  if refused is not None:
    # ...a hand out that could not come home: nothing moves (#678)
    _reports(life)["step"] = {"dx": 0.0, "dy": 0.0, "turn": 0.0}
    return {**refused, "seconds": 0.0, "dx": 0.0, "dy": 0.0, "turn": 0.0}
  rep = yield from life.body.step_routine(float(args["ahead"]), float(args.get("left", 0.0)),
                                          float(args.get("turn", 0.0)))
  _reports(life)["step"] = {k: float(rep.get(k, 0.0)) for k in ("dx", "dy", "turn")}
  out = {"why": rep["why"], "seconds": rep.get("seconds"),
         **{k: round(float(rep.get(k, 0.0)), 4) for k in ("dx", "dy", "turn")},
         **_carried(life)}
  if rep["why"] != "stepped":
    kept = rep.get("keptOff")
    reason = (STEP_KEPT_OFF[kept["by"]].format(**kept) if kept
              else STEP_WHY.get(rep["why"], "your step stopped short"))
    return {"ok": False, **out, "reason": reason}
  return {"ok": True, **out}


def _close_hand(life, args: dict) -> Routine:
  hand = args["hand"]
  held, felt = yield from _felt_routine(life, hand, life.body.grip_routine(
    hand, True, float(args.get("force", _GRIP.default))))
  return {"ok": bool(held), "hand": hand, "felt": felt,
          **({} if held else {"reason": f"your {hand} gripper closed on nothing it holds"})}


def _open_hand(life, args: dict) -> Routine:
  hand = args["hand"]
  _, felt = yield from _felt_routine(life, hand, life.body.grip_routine(hand, False))
  return {"ok": True, "hand": hand, "felt": felt}


#: What a `withdraw` whose arm did not get home says, by the body's why.
WITHDRAW_WHY = {"fell": "you fell on the way home",
                "held": "your {hand} arm did not get home: something holds it there"}


def _withdraw(life, args: dict) -> Routine:
  """ok once the arm is home, by its drivers (`Body.release_routine`), its
  gripper opened first (#669)."""
  hand = args["hand"]
  why, felt = yield from _felt_routine(life, hand, life.body.release_routine(hand, refuse=True))
  if why == "home":
    return {"ok": True, "hand": hand, "felt": felt}
  kept = life.body.hand_kept(hand) if why == "kept off" else None
  reason = (ARM_KEPT_OFF["withdraw"].format(hand=hand, where=_where(kept), **kept) if kept
            else WITHDRAW_WHY.get(why, "your {hand} arm did not get home").format(hand=hand))
  return {"ok": False, "hand": hand, "why": why, "felt": felt, "reason": reason}


#: A hand's path, the most points (`via`): a 64-degree arc in 4-degree
#: steps up and back is 32.
MAX_VIA = 40


def _paths(life) -> dict:
  """The points each hand's path holds, `via`'s, until its next `move_hand`
  carries it through them."""
  if getattr(life, "paths", None) is None:
    life.paths = {}
  return life.paths


def _via(life, args: dict) -> Routine:
  """A point (x, y, z) a hand's next `move_hand` goes through, and the
  grip's turn there where given (as `reach` takes it): nothing moves yet
  (issue #493, Ben, 2026-10-19: separate moves each took a posture of
  their own, and carried the reference's own arc on 2 chests of 8)."""
  hand = args["hand"]
  rot, bad = _grip_turn(life, args)
  if bad:
    return {"ok": False, "hand": hand, "reason": bad}
  points = _paths(life).setdefault(hand, [])
  if len(points) >= MAX_VIA:
    return {"ok": False, "hand": hand,
            "reason": f"a path holds {MAX_VIA} points at most: `move_hand` along it first"}
  points.append((_hand_place(life, args), rot))
  return {"ok": True, "hand": hand, "points": len(points)}
  yield


#: A TAG IS NAMED BY ITS NUMBER, and every verb that takes one says so (#640:
#: unsaid, 11 of 11 first table answers wrote `spot("93")`), its example a
#: call that compiles, by a tag no world carries (0-19 were the rover's), so
#: it names no job's -- and `find` shows none: its tag must be a task area's.
BY_NUMBER = "named by its number, not in quotes"
TAG_EXAMPLE = 12


_HAND = Arg("str", choices="hands")
_HAND_AT = {"x": Arg("float", *HAND_BOX[0]), "y": Arg("float", *HAND_BOX[1]),
            "z": Arg("float", *HAND_BOX[2])}
_STIFF = Arg("float", lo=0.05, hi=1.0, default=1.0)
#: A grip's turn, as `reach` takes it and a carried hand's may change to:
#: the fingers' way and the jaws', any length (`_grip_turn` scales them),
#: all left 0 for the turn the arm finds or keeps.
_TURN = {k: Arg("float", default=0.0) for k in ("fx", "fy", "fz", "jx", "jy", "jz")}
_SPEED = Arg("float", lo=HAND_V[0], hi=HAND_V[2], default=HAND_V[1])
#: Whether a pull gives across its way (1) or holds its line (0) (#525).
_GIVE = Arg("float", lo=0.0, hi=1.0, default=1.0)
#: How hard a gripper squeezes, N a pad (#525): the Dex1-1's adjustable 5-120.
_GRIP = Arg("float", lo=5.0, hi=120.0, default=120.0)
VERBS: dict[str, Verb] = {
  # `patience` (issue #381): the walk's budget, the robot's to set
  "drive_to": Verb("drive_to", {"x": Arg("float"), "y": Arg("float"),
                                "patience": Arg("float", lo=0.0, hi=MAX_PATIENCE_S,
                                                default=DRIVE_TIMEOUT_S)},
                   _drive_to, "walk to a world point over the map, and over floor not "
                   "yet on it; ok on arrival, and it gives up after "
                   f"`patience` seconds (at most {MAX_PATIENCE_S:.0f})"),
  "face": Verb("face", {"heading": Arg("float", lo=-math.pi, hi=math.pi)},
               _face, "turn in place; ok when squared within the budget"),
  # PLACES (issue #419): a task area found by its tag and remembered -- the
  # robot's own knowledge, never a coordinate handed over
  "find": Verb("find", {"tag": Arg("float", choices="places"), "x": Arg("float"),
                        "y": Arg("float"),
                        "patience": Arg("float", lo=0.0, hi=MAX_PATIENCE_S,
                                        default=FIND_PATIENCE_S)},
               _find, "find the task area carrying this tag: where you last saw "
               "it, else by searching round (x, y) -- a job's address -- until "
               "the tag is in view of your navigation camera; ok when seen, and you "
               "remember where it is. "
               f"It gives up after `patience` seconds (at most {MAX_PATIENCE_S:.0f}). "
               f"A tag is {BY_NUMBER}"),
  "wait": Verb("wait", {"seconds": Arg("float", lo=0.0, hi=MAX_WAIT_S)}, _wait,
               "stand still"),
  # The motor level (issue #166): what every verb above is built from.
  "move": Verb("move", {"axis": Arg("str", choices="axes"), "target": Arg("float")},
               _move, "walk one axis to a setpoint, ramped, inside its envelope"),
}


#: The hands' verbs (issues #493, #525; docs/Hands.md), the humanoid's: a
#: table of their OWN, which the language reads only where a world's facts
#: name them (`language`) -- ⚠ a verb's name is a name no local may take,
#: and in `VERBS` these would have refused a served robot's kept procedure
#: that names a local `reach` or `pull`. Two kinds of hand move: a POSITION
#: move puts the hand where it is told (`reach`, `move_hand`, `withdraw`); a
#: FORCE move pushes with a force and lets what the hand holds take it
#: (`pull`). The feet's own move is `step`, its hands home and empty.
HANDS_VERBS: dict[str, Verb] = {
  "measure": Verb("measure", {"x": Arg("float", lo=0.0, hi=3.0), "y": Arg("float", lo=-2.0, hi=2.0)},
                  _measure, "look with your depth camera at what stands within "
                  f"{MEASURE_NEAR_M:g} m of (x, y): the height of its flat top and where that "
                  "top ends -- nearest and farthest ahead, leftmost and rightmost -- ride "
                  f"`object.*`; ok when a flat top is seen, {MEASURE_FLOOR_M:g} m or more over "
                  "the floor. An end reads only where your camera saw past it: where the top "
                  f"runs on past those {MEASURE_NEAR_M:g} m, off your camera's picture or out "
                  f"of its range ({MEASURE_NEAREST_M:g} to {MEASURE_REACH_M:g} m), its end that way has "
                  "no reading, and reading it says why. Where something taller stands "
                  "just past an end, that end reads where the free top stops, and "
                  "`object.near_taller` (`far`, `left`, `right`) reads 1"),
  "spot": Verb("spot", {"tag": Arg("float", lo=0.0, hi=586.0)}, _spot,
               "stand still and look for this tag in your colour camera -- as you stand, then "
               "bowing lower until you see it -- and stand up again; ok when you see it. "
               "That camera looks down, lower than your navigation camera: its picture "
               f"runs from {SPOT_TOP_DEG:.0f} to {SPOT_BOTTOM_DEG:.0f} degrees below level, "
               f"out to {SPOT_SIDE_TOP_DEG:.0f} degrees either side of where your head faces "
               f"at its top and more than {SPOT_SIDE_BOTTOM_TENS} at its bottom, and bowing only "
               "takes it lower. With your trunk square to your feet, a thing at x in your "
               "stance frame is in the picture only if it stands lower than "
               f"({SPOT_TOP_AT_FEET_M:.2f} - {SPOT_TOP_DROP:.2f} x) m over the floor, and a tag "
               "reads only with the whole of it inside the picture: the lower a tag is, the "
               "further off it reads, a small one level with your shoulders only within "
               f"about {SPOT_SHOULDER_IN_M:.1f} m ahead of you, a larger one nearer still, and "
               "nothing as high as your camera ever. Inside the picture, a small block's tag "
               f"reads from up to about {SPOT_BLOCK_READS_M:g} m away. Where "
               "its face is, which way the face looks and which way its edges run ride "
               "`spot.*`. A small block carries its tag on every face: the one read is the one "
               "you see most square-on, `spot.nx`..`nz` say which, so from elsewhere the same "
               "tag can read another face, its middle up to the block's width from this one's. "
               f"A tag is {BY_NUMBER}: `spot({TAG_EXAMPLE})` looks for tag {TAG_EXAMPLE}"),
  "watch": Verb("watch", {"tag": Arg("float", lo=0.0, hi=586.0)}, _watch,
                "keep your colour camera reading this tag while your procedure goes on -- your "
                f"hands moving, your feet stepping, waiting -- {WATCH_HZ:g} times a second, each "
                f"reading yours {WATCH_LATENCY_S:g} s after its picture was taken; it neither "
                "bows nor turns to look, and it ends with your procedure or your next `watch`. "
                "It reads through `spot`'s camera, the top of whose picture is "
                f"{SPOT_TOP_DEG:.0f} degrees below level as you stand and lower as you bow: what "
                "is above that, it cannot see. "
                "It waits for its first reading; ok when that shows the tag (as an `if`'s test "
                "that is not, it watches on). What it reads rides `watch.*`: of a small block, "
                "as for `spot`, the face that picture shows most square-on, so which face can "
                f"change as you move. A tag is {BY_NUMBER}: `watch({TAG_EXAMPLE})` keeps reading "
                f"tag {TAG_EXAMPLE}"),
  "reach": Verb("reach", {"hand": _HAND, **_HAND_AT, **_TURN, "stiffness": _STIFF},
                _reach, "send a hand's grip -- the point between its jaws -- to (x, y, z), its "
                "fingers pointing along (fx, fy, fz) and its jaws closing along (jx, jy, jz) -- "
                "either way, a two-jaw grip -- or at whatever turn your arm finds where those "
                "are all left 0. It goes out to a point short of it, then in along the "
                "fingers; you take whatever posture reaches. stiffness 1 is your arm's full "
                "stiffness, less is softer. ok once it holds there -- or as near as it came: "
                "what it touches on its way in can hold it a few cm short, and your hand's "
                "place (`left.x` and the rest) says where it is. Its way out, and its way "
                "home from there, keep clear of the body of another robot standing near you: "
                "where no way does, your arm does not move, and it reads false saying where "
                "that robot stands"),
  "via": Verb("via", {"hand": _HAND, **_HAND_AT, **_TURN}, _via,
              "add the point (x, y, z) to your hand's path, the grip turned there to "
              "(fx, fy, fz) and (jx, jy, jz) where given as for `reach`; nothing moves "
              "until your next `move_hand`, which goes through the path's points on its way"),
  "move_hand": Verb("move_hand", {"hand": _HAND, **_HAND_AT, "speed": _SPEED,
                                  "stiffness": _STIFF, **_TURN},
                    _move_hand, "carry a hand that is out (reached) to (x, y, z) at `speed` "
                    "m/s along a straight line -- or through the points `via` added first, in "
                    "turn, as one motion from one posture that follows them all, turning as "
                    "each asks (a point with no turn keeps the one before); the path is then "
                    "empty -- its turn kept, or turned to (fx, fy, fz) and (jx, jy, jz) given "
                    "as for `reach`; ok when it arrives"),
  # ⚠ A WAY OF ANY LENGTH (`_pull` scales it to one): bounded to 1 a
  # component, two of #493's tries ended at a computed 1.03 and 1.001
  "pull": Verb("pull", {"hand": _HAND, "dx": Arg("float"), "dy": Arg("float"),
                        "dz": Arg("float"),
                        "force": Arg("float", lo=1.0, hi=PULL_MOST_N),
                        "distance": Arg("float", lo=0.01, hi=0.6), "speed": _SPEED,
                        "give": _GIVE},
               _pull, "pull a hand that is out -- or push it -- along the way (dx, dy, dz), of "
               "any length, with at most `force` N, at most `speed` m/s: across that way "
               "your hand goes where what it holds pushes it, and its turn gives to what it "
               "holds, but about the line its jaws close along (what it holds turns about "
               "that line between the jaws) -- or, `give` 0, it holds its line and its "
               "turn. It stops once it has "
               "gone `distance` m along the way, it is held fast "
               "(pushing with its whole force and not moving), its grip lets go, or it can "
               "reach no further; then it holds there. Standing, your body holds a pull of "
               f"{PULL_MOST_N:.0f} N at a hand, and pushing away from you around two thirds of "
               "that. How far it went, the hardest it pushed, whether it ended held fast and "
               "the way it was going at its end ride `pull.*`, unless it reads false; ok unless "
               "an argument was refused, your hand was not out, (dx, dy, dz) are all 0 or you "
               "fell: whatever else stopped it, held fast against something too, it is still "
               "ok, and `pull.moved` says how far it went"),
  "step": Verb("step", {"ahead": Arg("float", lo=-0.5, hi=0.5),
                        "left": Arg("float", lo=-0.5, hi=0.5, default=0.0),
                        "turn": Arg("float", lo=-3.2, hi=3.2, default=0.0)}, _step,
               "turn `turn` rad on the spot (left, +), then walk `ahead` m and `left` m in "
               "the frame you then stand in (negative: back, right), and stand still -- your "
               "hands home first, but a hand that holds something carried at your side, as "
               "every walk carries it (`carried` in its report). Your feet walk in strides, each "
               "carrying your stance about 5 cm: a walk asked under "
               f"{100 * STEP_SHORT_M:.0f} cm may take none, and a walk may stop a few cm off "
               f"what you asked. A walk asked at least {100 * STEP_LEAST_M:.0f} cm that your own "
               "legs count as under it reads false and says your step was too short to walk. "
               "A walk keeps off another robot as your walks over the map do. It stops "
               f"before it would take your middle within {STEP_KEEP_M:g} m of that robot's "
               f"middle ({STEP_KEEP_DOWN_M:g} m of one that has fallen) -- standing within that "
               "already, a step whose walk would take you nearer neither turns nor walks -- "
               f"and once that robot's body is within {STEP_FRONT_M:g} m of your middle in "
               f"the way you step and {STEP_FRONT_HALF_M:g} m either side of it. Either "
               "reads false and says so. "
               "A turn lands within a degree or two of what you asked -- one that does not "
               "get there reads false and says so -- and shuffles your feet a few cm that "
               f"`step.*` does not count. `step.dy` counts a sidestep "
               f"{100 * STEP_DY_SHORT_M[0]:.0f}-{100 * STEP_DY_SHORT_M[1]:.0f} cm short of "
               "how far your feet went, and one going left after a step ahead up to "
               f"{100 * STEP_DY_LEFT_AFTER_M:.0f} cm or more short. Your stance frame "
               "goes with your feet: where they went in the frame you stood in before rides "
               "`step.*` (`step.dx` ahead, `step.dy` left, m, and `step.turn`, rad), so a "
               "place (x, y) you read before is now (c (x - dx) + s (y - dy), c (y - dy) - "
               "s (x - dx)), c and s the cosine and sine of its turn; ok when it stepped"),
  "close_hand": Verb("close_hand", {"hand": _HAND, "force": _GRIP}, _close_hand,
                     "shut a hand's gripper where it is, squeezing with at most `force` N a "
                     "jaw -- firm holds fast; light lets what it holds turn between the jaws; "
                     "ok when it says it holds something"),
  "open_hand": Verb("open_hand", {"hand": _HAND}, _open_hand, "open a hand's gripper"),
  "withdraw": Verb("withdraw", {"hand": _HAND}, _withdraw,
                   "open a hand's gripper, letting go of whatever it holds, then bring the "
                   "hand back out along the way it went in, and its arm home -- straight "
                   "there, or out to your side first and down at your side where that keeps "
                   "the arm higher in front of you than straight would; ok once your arm is "
                   "home. Its way home keeps clear of the body of another robot standing near "
                   "you: where no way does, your gripper stays as it is and your hand where it "
                   "is, and it reads false saying where that robot stands"),
}
#: The hands' vocabulary (#525; docs/Hands.md): what the humanoid's
#: procedures call, the checks' and the practice's alike...
HAND_VERBS = ("spot", "reach", "via", "move_hand", "pull", "close_hand", "open_hand",
              "withdraw", "step")
#: ...and `measure`, the depth camera's, but where an arm is about its
#: absence (#493: depth is feedback, never a shortcut).
MEASURE_VERBS = ("measure",)
#: ...and `watch`, a tag kept in sight while it moves (#554's stage 2), but
#: where an arm is about its absence.
WATCH_VERBS = ("watch",)

#: A HAND'S MOVE THAT STOPS ON A READING (#554): `reach`, `move_hand` and
#: `pull` as `HANDS_VERBS` has them, and one stop condition after their own
#: arguments -- a sensor `read` names (`until`, a reading that changes as
#: the hand moves: `axes.Sensor.every_s`), and the number it stops past,
#: above or below (`_guarded_routine`) -- where a world's facts say so
#: (`WorldFacts.guarded`). Neither knows a job: the robot turns what it
#: wants into a reading and a number itself (Ben, 2026-10-27).
GUARD_ARGS = {"until": Arg("str", optional=True), "above": Arg("float", optional=True),
              "below": Arg("float", optional=True)}
GUARD_DOC = (". Given `until` (a sensor) and `above` or `below` (a number), it stops sooner "
             "-- where the hand is, as soon as that sensor reads past that number -- and is "
             "ok then too")
GUARDED_VERBS: dict[str, Verb] = {
  n: Verb(n, {**HANDS_VERBS[n].args, **GUARD_ARGS}, HANDS_VERBS[n].run,
          HANDS_VERBS[n].doc + GUARD_DOC)
  for n in ("reach", "move_hand", "pull")}


def language(facts: "WorldFacts | None") -> dict[str, Verb]:
  """The verbs a procedure in this world may call: `VERBS` -- or, where its
  facts name the hands' (`HANDS_VERBS`), exactly the verbs they name, so
  an unknown verb's refusal lists what the body has, a hand's move with its
  stop condition where they allow one (`GUARDED_VERBS`)."""
  named = () if facts is None or facts.verbs is None else facts.verbs
  if not any(n in HANDS_VERBS for n in named):
    return VERBS
  table = {**VERBS, **HANDS_VERBS, **(GUARDED_VERBS if facts.guarded else {})}
  return {n: table[n] for n in named}


def hand_facts(measure: bool = True, step: bool = True, guarded: bool = True,
               watching: bool = True, calls: bool = True) -> "WorldFacts":
  """What a procedure on the humanoid's hands is validated against (issues
  #493, #525): standing, a `step` at a time, and its arms' joints, the hands' vocabulary
  (`HAND_VERBS`) -- with the depth camera's `measure`, the feet's `step`,
  a hand's move's stop condition, a tag watched while it moves (#554) and
  defs that call defs (#560) unless an arm is about the absence of one --
  their sensors and the clock; a verb may be an `if`'s test, and an
  expression may call the maths (`lang.MATH`)."""
  from pluggybot.procedure import axes
  hands = tuple(v for v in HAND_VERBS if step or v != "step")
  if watching:
    # ...beside `spot`, the other verb its camera reads a tag by
    at = hands.index("spot") + 1
    hands = hands[:at] + WATCH_VERBS + hands[at:]
  return WorldFacts(bounds=(-1e3, -1e3, 1e3, 1e3), axes=axes.BODY_AXES["humanoid"],
                    sensors=(axes.HAND_SENSORS + (axes.WATCH_SENSORS if watching else ())
                             + axes.PULL_SENSORS
                             + (axes.STEP_SENSORS if step else ())
                             + (axes.MEASURE_SENSORS if measure else ())
                             + (axes.STOPPED_SENSORS if guarded else ())),
                    verbs=("wait", "move") + hands + (MEASURE_VERBS if measure else ()),
                    hands=axes.HANDS, verb_tests=True, math=True, guarded=guarded,
                    calls=calls)


#: What a SERVED humanoid walks with beside its hands (#559, stage 3): a
#: walk to a point over its map and a turn where it stands -- the walk
#: takes its arms home first (`HumanoidMission.to_walk_routine`).
SERVED_WALK_VERBS = ("drive_to", "face")


def served_hand_facts(bounds=(-1e3, -1e3, 1e3, 1e3), places: tuple = ()) -> "WorldFacts":
  """What a procedure a served humanoid writes is validated against (#559,
  stage 3): the practice's hands, every one of its switches on and their
  sensors (`hand_facts`), and its walks over its map (`SERVED_WALK_VERBS`),
  its pack, its wallet -- its LIDAR's front stop is no part a procedure
  reads. Where its world has task areas (`places`, their tags), it finds
  them, `find`, and reads where it found one (`axes.FIND_SENSORS`)."""
  from dataclasses import replace
  from pluggybot.procedure import axes
  hands = hand_facts()
  body = tuple(s for s in axes.BODY_SENSORS if s not in hands.sensors)
  places = tuple(int(t) for t in places)
  return replace(hands, bounds=tuple(float(v) for v in bounds),
                 verbs=SERVED_WALK_VERBS + (PLACE_VERBS if places else ()) + hands.verbs,
                 sensors=body + (axes.FIND_SENSORS if places else ()) + hands.sensors,
                 places=places)


#: What a verb a fall ended says where it has no words of its own for one
#: (#671): the hands' moves have theirs (`REACH_WHY`'s and the rest).
FELL = "you fell"


def dead(life) -> bool:
  """Is `life`'s robot dead (a lifecycle's `dead`, #107)? A life with none --
  a practice's, a session's -- never is."""
  return getattr(life, "dead", None) is not None


def run_verb(life, verb: Verb, args: dict, where: dict | None = None,
             until: float | None = None) -> Routine:
  """One verb, as both runners call it (this module's and the language's).
  While it runs, `life.step_now` says which step this is (`where`: the
  procedure, the count, a line), for a death to name (issue #362), and
  `life.step_until` when the program's budget ends (`until`, sim s), past
  which no walk inside it waits (`_patience`).

  ⚠ A VERB THE ROBOT'S OWN INTERRUPT ENDED IS STOPPED, NOT FAILED (issue
  #381): a walk inside it asks the interrupt as it goes (`_interrupt`),
  and when the answer latched an abort during this verb, a verb that did
  not succeed says `stopped: interrupted` -- both runners stop the program
  there as an abort, never at a failed step.

  ⚠ A FALL OR A DEATH ENDS IT AT ONCE (#671; `_until_down_routine`;
  Overseer.md §2b): a verb its body fell in fails and says so (`fell`, in
  its own words where it ended on the fall, else `FELL`), and one its robot
  died in is `stopped: died`; both runners end the run there, an `if`'s
  test or not. ⚠ A hand a death stopped is halted where it is
  (`Body.stop_hand`: a reach, a carry or a pull, never an arm on its way
  home): closed, a pull went on pulling for a dead robot. Out, it holds a
  save off until the stand-up, as a robot dead from a fall does
  (`continuation.Keeper.busy`)."""
  before = getattr(life, "step_now", None)
  before_until = getattr(life, "step_until", None)
  aborting = bool(getattr(life, "aborting", False))
  life.step_now = {**(where or {}), "verb": verb.name, "args": dict(args)}
  life.step_until = until
  try:
    verdict, ended = yield from _until_down_routine(life, _verb_routine(life, verb, args))
    if verdict is None:
      # ...closed before it said anything: what it last reported would read
      # as this call's
      forget_readings(life, verb.name)
    if ended == "died":
      if args.get("hand") in ("left", "right"):
        life.body.stop_hand(args["hand"])
      return {"ok": False, "stopped": "died"}
    said = verdict or {}
    if ended == "fell" or said.get("why") == "fell":
      own = said.get("reason") if said.get("why") == "fell" else None
      return {**said, "ok": False, "fell": True, "reason": own or FELL}
    if (not verdict.get("ok") and not aborting
        and getattr(life, "aborting", False)):
      verdict = {**verdict, "stopped": "interrupted"}
    return verdict
  finally:
    life.step_now = before
    life.step_until = before_until


def _verb_routine(life, verb: Verb, args: dict) -> Routine:
  """`verb` run, its plug out by hand first where it moves the body (#654)."""
  unplug = getattr(life, "unplug_routine", None)
  if unplug is not None and verb.name not in STILL_VERBS:
    yield from unplug()
  return (yield from verb.run(life, args))


def _until_down_routine(life, routine: Routine) -> Routine:
  """`routine` driven as `yield from` would, and CLOSED where its body falls
  (`Body.falls`, its own fall watch's count) or its robot dies (a
  lifecycle's `dead`; a life with none never does). Returns (what it
  returned, None where it was closed; and "died", "fell" or "").

  ⚠ A DEATH IS READ THE STEP IT LANDS, BEFORE THE ROUTINE RUNS AGAIN (as
  `HubLifecycle._until_stood_up_routine` reads a stand-up), and a robot
  dead as the verb is called begins none of it: no line of the verb runs
  on a dead robot. ⚠ A FALL IS READ ONCE THE ROUTINE HAS RUN ON FROM IT: a
  hand's move ends on the fall itself and says so in its own words ("fell"
  too where it returned so), and anything that does not is closed at its
  next command."""
  if dead(life):
    routine.close()
    return None, "died"
  body = life.body
  falls = body.falls
  sent, exc = None, None
  while True:
    try:
      cmd = routine.throw(exc) if exc is not None else routine.send(sent)
    except StopIteration as stop:
      return stop.value, ("fell" if body.falls != falls else "")
    exc = None
    if body.falls != falls:
      routine.close()
      return None, "fell"
    try:
      sent = yield cmd
    except GeneratorExit:
      routine.close()
      raise
    except BaseException as e:  # noqa: BLE001 -- thrown into the routine, as `yield from` would
      exc, sent = e, None
      continue
    if dead(life):
      routine.close()
      return None, "died"


#: What a body whose world has task areas it finds by their tags runs
#: (issue #419).
PLACE_VERBS = ("find",)
#: The verbs that leave a plug in its port (#654): the pack goes on filling
#: under a look or a reading. Any other verb has it out by hand first, a
#: new one too: 8 s lost beats a walk breaking the catch. A bow to look
#: lower is one of these: to its deepest, 63 deg, the plug stays in.
STILL_VERBS = ("wait", "watch", "measure", "spot", "via", "close_hand", "open_hand")


def describe_vocabulary(verbs: tuple | None = None, guarded: bool = False) -> list[dict]:
  """The verbs as data -- what a prompt or a validator's error names --
  every one, or those a body can run (`verbs`, the hands' among them, a
  hand's move with its stop condition where `guarded`). An argument that
  may be left out rides `defaults` with what it is when it is, or
  `optional` where the verb does without it."""
  table = {**VERBS, **HANDS_VERBS, **(GUARDED_VERBS if guarded else {})}
  keep = VERBS if verbs is None else {n: table[n] for n in verbs}
  out = []
  for v in keep.values():
    entry = {"verb": v.name, "args": {k: a.kind for k, a in v.args.items()},
             "doc": v.doc}
    defaults = {k: a.default for k, a in v.args.items() if a.default is not None}
    if defaults:
      entry["defaults"] = defaults
    optional = [k for k, a in v.args.items() if a.optional]
    if optional:
      entry["optional"] = optional
    out.append(entry)
  return out


def signature(entry: dict) -> str:
  """A described verb as a call: `drive_to(x, y, patience=60)`, those it
  does without in brackets at its end: `pull(..., give=1[, until, above,
  below])`."""
  given, optional = entry.get("defaults", {}), entry.get("optional", ())
  return entry["verb"] + "(" + ", ".join(
    f"{k}={given[k]:g}" if k in given else k for k in entry["args"] if k not in optional) + (
    f"[, {', '.join(optional)}]" if optional else "") + ")"


# ---- validation: total, before a single step runs ----------------------------


def check_arg(verb: Verb, name: str, value, facts: WorldFacts) -> list[str]:
  """One argument against its `Arg`: type, range, and the world's choices.
  Shared by a program's validator (every argument is literal) and the
  language's (a computed argument is checked when it is computed)."""
  arg = verb.args[name]
  if arg.kind == "float":
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
        or not math.isfinite(float(value)):
      return [f"{name} must be a finite number, not {value!r}"]
    bad = []
    if arg.lo is not None and float(value) < arg.lo:
      bad.append(f"{name}={value} is below {arg.lo}")
    if arg.hi is not None and float(value) > arg.hi:
      bad.append(f"{name}={value} is above {arg.hi}")
    if arg.choices and value not in getattr(facts, arg.choices):
      bad.append(_not_on(name, value, arg.choices, facts))
    return bad
  if not isinstance(value, str):
    return [f"{name} must be a name, not {value!r}"]
  allowed = getattr(facts, arg.choices) if arg.choices else None
  if allowed is not None and value not in allowed:
    return [f"{name}={value!r} is not one of "
            f"{', '.join(allowed) or 'nothing this world has'}"]
  return []


def _not_on(name: str, value: float, choices: str, facts: WorldFacts) -> str:
  """A tag on none of its list: `tag 20 names no place this world has
  (places: 35 to 46)`, and a tag `Arg` carries no bound, which refused it a
  second time (#463)."""
  return (f"{name} {number_text(value)} names no {choices[:-1]} this world has "
          f"({choices}: {_tags(getattr(facts, choices))})")


def number_text(v: float) -> str:
  """A number exactly as it is: every digit of 23.0000001, which `:g`
  quoted as `23` beside a list holding 23, and a whole float without its
  `.0` -- a tag, and the locals History reads (`procedure_outcome`)."""
  return str(v) if isinstance(v, int) else repr(float(v)).removesuffix(".0")


def _tags(tags: tuple[int, ...]) -> str:
  """A world's tags, a run of three or more as `lo to hi`: a refused define
  is one History line, cut at 400 characters, and the twelve places listed
  in full twice cut #463's own refused rewrite short (401). Never `lo..hi`,
  which reads as leaving `hi` out. In order: the grown world's places
  (#560) came after the counter's in the world's own order."""
  runs: list[list[int]] = []
  for t in sorted(tags):
    if runs and t == runs[-1][-1] + 1:
      runs[-1].append(t)
    else:
      runs.append([t])
  return ", ".join(f"{r[0]} to {r[-1]}" if len(r) >= 3 else ", ".join(map(str, r))
                   for r in runs) or "none"


def _check_guard(args: dict, given: set, facts: WorldFacts) -> list[str]:
  """A hand's move's stop condition (#554): a sensor `until` names, one
  that changes as the hand moves, and the number it stops past -- `above`
  or `below`, never both, never one without the other. `given` is every
  argument the call names, a computed one too."""
  named = set(given) & set(GUARD_ARGS)
  sides = named & {"above", "below"}
  if not named:
    return []
  bad = []
  if "until" not in named:
    bad.append('above and below are the number a move stops past: name the sensor it '
               'reads with until="<sensor>"')
  elif not sides:
    bad.append("until names the sensor a move stops on: give the number it stops past as "
               "above=N or below=N")
  if len(sides) == 2:
    bad.append("a move stops past its number on one side: above or below, not both")
  name = args.get("until")
  if isinstance(name, str):
    bad += guard_sensor(name, facts)
  return bad


def guard_sensor(name: str, facts: WorldFacts) -> list[str]:
  """Whether a move may stop on the sensor `until` names: one this body
  has, that changes as the hand moves -- a literal's rule, and a name an
  argument carries into a call (`lang._named`, #560)."""
  from pluggybot.procedure import axes
  live = ", ".join(s for s in facts.sensors if axes.SENSORS[s].every_s is not None)
  if name not in facts.sensors:
    return [f"until={name!r} is no sensor you have (a move may stop on {live})"]
  if axes.SENSORS[name].every_s is None:
    return [f"until={name!r} is what a verb last reported, which no move changes: a "
            f"move stops on a sensor that changes as your hand moves ({live})"]
  return []


def check_step(verb: Verb, args: dict, facts: WorldFacts,
               partial: bool = False, given=None) -> list[str]:
  """Every argument of one step. `partial` skips the missing-argument rule,
  for a language checking only the literal half of a call; `given` is
  every argument that call names, computed or not (a stop condition's
  rule reads which, #554), where it is not `args`' own.

  ⚠ A REASON NEVER NAMES ITS CALL: whoever knows where the step is names
  it, once (`validate`, `lang.validate`, History's `procedure_outcome`:
  `line 3, wait: seconds=99.0 is above 60.0`). "tag 20 is no place",
  refusing a `find`, was read as a refusal of `place` (#463), and a reason
  that named its call too was said `pick: pick(tag=99) ...` at run time."""
  if facts is not None and facts.verbs is not None and verb.name not in facts.verbs:
    return [f"not on this body yet (it has: {', '.join(facts.verbs)})"]
  bad = []
  extra = set(args) - set(verb.args)
  missing = {k for k, a in verb.args.items() if a.default is None and not a.optional} - set(args)
  if extra:
    bad.append(f"takes no {', '.join(sorted(extra))}")
  if missing and not partial:
    bad.append(f"needs {', '.join(sorted(missing))}")
  for name in verb.args:
    if name in args:
      bad += check_arg(verb, name, args[name], facts)
  if verb.name in ("drive_to", "find"):
    # Each axis on its own, in one wording, whether the other is written,
    # computed or refused (#463); one `check_arg` refused is not refused again
    x0, y0, x1, y1 = facts.bounds
    for k, lo, hi in (("x", x0, x1), ("y", y0, y1)):
      v = args.get(k)
      if isinstance(v, (int, float)) and not isinstance(v, bool) \
          and math.isfinite(v) and not lo <= v <= hi:
        bad.append(f"{k}={float(v)} is outside the map's [{lo}, {hi}]")
  if verb.name == "move" and isinstance(args.get("axis"), str) \
      and isinstance(args.get("target"), (int, float)) \
      and not isinstance(args.get("target"), bool):
    from pluggybot.procedure import axes
    axis = axes.AXES.get(args["axis"])
    if axis is not None and not axis.lo <= float(args["target"]) <= axis.hi:
      bad.append(f"{axis.name} target {args['target']} is outside "
                 f"{axis.lo}..{axis.hi} {axis.unit}".rstrip())
  if "until" in verb.args:
    bad += _check_guard(args, set(args) if given is None else set(given), facts)
  return bad


def validate(program: Program, facts: WorldFacts) -> list[str]:
  """Every rule a program can break, as readable reasons. Empty means valid.

  Total on purpose: an author sees all of it at once, and nothing partial can
  run because `run_program_routine` calls this first and refuses on any."""
  bad: list[str] = []
  if not program.name or not isinstance(program.name, str):
    bad.append("a program needs a name")
  if not program.roles:
    bad.append("a program needs at least one role")
  if not (0.0 < float(program.budget_s) <= MAX_BUDGET_S):
    bad.append(f"budgetS {program.budget_s} is outside (0, {MAX_BUDGET_S:.0f}]")
  for role, steps in program.roles.items():
    if not steps:
      bad.append(f"role {role!r} has no steps")
    if len(steps) > MAX_STEPS:
      bad.append(f"role {role!r} has {len(steps)} steps; the cap is {MAX_STEPS}")
    for i, step in enumerate(steps):
      where = f"{role}[{i}]"
      verb = VERBS.get(step.verb)
      if verb is None:
        bad.append(f"{where}: unknown verb {step.verb!r} "
                   f"(have: {', '.join(VERBS)})")
        continue
      bad += [f"{where}, {step.verb}: {r}" for r in check_step(verb, step.args, facts)]
  return bad


def compile_program(program: Program, facts: WorldFacts) -> Program:
  """Validate or raise `Refused` with every reason."""
  reasons = validate(program, facts)
  if reasons:
    raise Refused(reasons)
  return program


# ---- running: one verdict per step, ticked from the loop ---------------------


def run_program_routine(life, program: Program, facts: WorldFacts,
                        role: str | None = None) -> Routine:
  """Run ONE ROLE of a validated program; the result is honest about how
  far it got. Refuses (raises `Refused`) before the first step on anything
  validation catches. A program with several roles needs the role named --
  by the robot that claimed it (issue #167) -- and a single-role program
  runs its one role unnamed."""
  reasons = validate(program, facts)
  if role is None:
    if len(program.roles) > 1:
      reasons.append(f"{len(program.roles)} roles: this robot must be told "
                     "which one it plays")
    role = next(iter(program.roles), DEFAULT_ROLE)
  if role not in program.roles:
    reasons.append(f"no role {role!r} in this program")
  if reasons:
    raise Refused(reasons)
  steps = program.steps(role)
  result: dict[str, Any] = {"program": program.name, "role": role,
                            "total": len(steps), "completed": 0,
                            "steps": [], "ok": False}
  t0 = float(life.data.time)
  for i, step in enumerate(steps):
    # Safe points: between steps, and inside a walk (issue #381, `run_verb`).
    # A hazard row (issue #116) or the budget stops the program there,
    # never inside anything else; a death, inside anything (#671) -- the
    # interrupt's ask too, which steps the world, and a dead robot is never
    # asked one
    if i and not dead(life) and life.interrupted() and not dead(life):
      result["stopped"] = "interrupted"
      break
    if dead(life):
      result["stopped"] = "died"
      break
    if float(life.data.time) - t0 > program.budget_s:
      result["stopped"] = "budget"
      break
    life._say(f"PROCEDURE {program.name} {i + 1}/{len(steps)}: {step.describe()}")
    verdict = yield from run_verb(life, VERBS[step.verb], step.args,
                                  {"procedure": program.name, "n": i + 1,
                                   "of": len(steps)},
                                 until=t0 + program.budget_s)
    entry = {"i": i, "verb": step.verb, **verdict}
    result["steps"].append(entry)
    if verdict.get("stopped") in ("interrupted", "died"):
      result["stopped"] = verdict["stopped"]
      break
    if not verdict.get("ok"):
      result["failedAt"] = i
      life._say(f"PROCEDURE {program.name} failed at {i + 1}/{len(steps)} "
                f"{step.describe()}" + (f" -- {verdict['reason']}"
                                         if verdict.get("reason") else ""),
                detail=verdict.get("trace", ""))
      break
    result["completed"] = i + 1
  else:
    result["ok"] = True
  result["seconds"] = round(float(life.data.time) - t0, 2)
  return result
