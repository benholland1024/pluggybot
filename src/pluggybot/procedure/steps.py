"""The step vocabulary: an errand as validated steps over the guarded
primitives (issue #58, rung one of agent-written code).

A PROGRAM is data -- a name, a budget, and per ROLE an ordered list of
steps, each a verb off a closed list with typed, bounded arguments. It is
validated whole before a single step runs, and then ticked from the physics
loop like every other routine (pluggybot/tick.py), one verdict per step.
What makes it safe for an agent to write is what it cannot say: every verb
is an existing primitive with the ramping, the coupling envelope and the
noslip policy INSIDE it, and nothing here touches `data.ctrl` (the fence in
tests/test_procedure.py enumerates the modules that may).

The verbs, in the words the issue used:

  fetch(tool)          pick a module off its bay        `Body.fetch_tool_routine`
  stow()               hang the carried module back     `Body.stow_tool_routine`
  drive_to(x, y)       walk to a world point            `Body.go_to_routine`
  face(heading)        turn in place                    `Body.face_routine`
  find(tag, x, y)      a task area found by its tag     `Body.find_tag_routine`
  press(tag)           onto the plate its sign marks    `Body.press_plate_routine`
  draw(board, figure)  a figure on a found whiteboard   `Body.draw_routine`
  pick(tag)            a cube into the claw             `Body.pick_cube_routine`
  place(tag)           the held cube onto another       `Body.place_cube_routine`
  put()                the held cube onto the floor     `Body.put_cube_routine`
  grip() / release()   the claw's jaws shut / open      `Body.claw_routine`
  survey(tag)          count what stands in an area     `Body.survey_routine`
  wait(seconds)        stand still
  move(axis, target)   one axis to a setpoint           `procedure/axes.py`
  drive(v, w, seconds) the base at a velocity           `Body.velocity_routine`

...and a two-role game's, in its program alone (`GAME_VERBS`, issue #404):

  hide()               the hider's role                 `Body.hide_routine`
  seek()               the seeker's role                `Body.seek_routine`

Every verb reaches the body through `Body` (issue #380, `body.py`) and
nothing else. Each returns a verdict dict with `ok`, measured off the world
(the module seated and powered, the arrival, a foot on the plate), never
off the command. The runner stops at the first failed step: later steps
assume earlier ones. The result is honest about how far it got.

Three rules the shape carries for the rungs above it:

  ROLES. A program is `{role: steps}` and today every program has one role,
  `robot`. M12's two-role errand (a hider and a seeker, one verdict) is a
  second key, not a rebuild -- so the slot validates any number of roles and
  the runner refuses more than one, with the reason.

  BUDGETS ARE CAPPED BY CODE. A program declares its own sim-time budget and
  the vocabulary caps it (`MAX_BUDGET_S`) and the step count (`MAX_STEPS`);
  a program past either does not validate. The runner checks the budget at
  step boundaries, which are the safe points; a walk is a safe point at
  every step (issue #381), and asks `life.interrupted` as it goes (#116).

  ABORT MEANS STOW. Whatever ends a program early -- a failed step, the
  budget, an interrupt -- the errand around it hangs any carried module back
  before the verdict. A program cannot leave a tool on the floor by
  stopping.

What a program may say is what a work order may say (TaskPattern.md §2): a
tool by name, a surveyed place by coordinate, a board by id. A program that
drives to where a movable object IS has been handed a sensor reading over the
wire, and the ladder applies to it as to any task.
"""

import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable

from pluggybot.legs.rack import TOOL_BAYS as RACK_TOOL_BAYS
from pluggybot.rack.coupling import STATION_YS
from pluggybot.tick import Routine
from pluggybot.tools.strokes import PROGRAMS

#: Which bay each tool hangs in, by index into `STATION_YS` (bay <-> tag
#: pairing is by that index): the quadruped's rack's (#405). A COPY: the
#: rack generator lays its pegs from the original.
TOOL_BAYS = dict(RACK_TOOL_BAYS)
#: The one role every program has today; M12 adds the second.
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
#: route, 95 walked), and its second 117 s (`scripts/unknown_spike.py`).
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
  the boards it has, the tools on its rack, and the box the map covers."""

  boards: tuple[str, ...]
  tools: tuple[str, ...]
  bounds: tuple[float, float, float, float]    # x_min, y_min, x_max, y_max
  figures: tuple[str, ...]
  #: the motor-and-sensor level (procedure/axes.py, issue #166): what
  #: `move` may move and `read` may read, presence checked at run time
  axes: tuple[str, ...] = ()
  sensors: tuple[str, ...] = ()
  #: the verbs this world's BODY can run (issue #387), or None for every
  #: one: a body with no arm has no tool, no claw and no pen
  verbs: tuple[str, ...] | None = None
  #: the task areas' tags a `find` may name (issue #419), and the plates'
  #: among them a `press` may
  places: tuple[int, ...] = ()
  plates: tuple[int, ...] = ()
  #: the cubes' tags the claw's verbs may name (issue #407)
  cubes: tuple[int, ...] = ()
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
  be left out."""

  kind: str                    # "float" | "str"
  lo: float | None = None
  hi: float | None = None
  choices: str | None = None
  default: float | None = None


@dataclass(frozen=True)
class Verb:
  name: str
  args: dict                   # arg name -> Arg
  run: Callable[..., Routine]  # (life, args) -> Routine returning a verdict
  doc: str
  #: moves the base, so the fork goes into its carrying pose first (`run_verb`)
  drives: bool = False


def _rack(life) -> dict[str, int]:
  """Which module hangs where: the lifecycle's inventory, or the rack's own
  where there is none."""
  return getattr(life, "rack_inventory", None) or TOOL_BAYS


def _tool_station(life, tool: str) -> float:
  return STATION_YS[_rack(life)[tool]]


def _seated(life, tool: str) -> bool:
  return bool(life.body.module_state(tool)["on_fork"]) and \
    life.body.tool_powered(tool)


def _fetch(life, args: dict) -> Routine:
  tool = args["tool"]
  station = _tool_station(life, tool)
  # ⚠ THE FORK FIRST (issue #264). A fork that already holds the tool has
  # it; one that holds ANOTHER was driven into a bay loaded, and the
  # deployed robots wrote `clean_fork` / `stow_recover` procedures around
  # the wreck. Neither drives anywhere.
  held = _carried(life)
  if held == tool:
    life.module = tool
    ok = _seated(life, tool)
    return {"ok": ok, "tool": tool, "why": "held", "powered": ok,
            **({} if ok else {"reason": f"{tool} is on the fork but not "
                                        "seated; stow it and fetch it again"})}
  if held is not None:
    return {"ok": False, "tool": tool, "why": "loaded",
            "reason": f"the fork already holds {held}; stow it first"}
  life.module = tool
  why = yield from life.body.fetch_tool_routine(station, tool)
  ok = _seated(life, tool)
  life.swaps_done += 1
  verdict = {"ok": ok, "tool": tool, "why": why, "powered": ok}
  if not ok:
    # The sentence the errand's History line says (`pick_failure`): a
    # procedure cut short at its first line told the robot nothing, and the
    # deployed robots re-ran the same failing fetch five times over.
    said = getattr(life, "pick_failure", None)
    verdict["reason"] = (f"could not pick up {tool}"
                         + (f": {said(tool, station, why)}" if said else ""))
    _trace(life, verdict, f"fetch {tool}")
  return verdict


def _trace(life, verdict: dict, what: str) -> None:
  """A failed swap's trace (issue #264, `Body.swap_trace`) on the step's
  verdict, which the runner narrates as `detail` -- the log's, never the
  status line or History, which are the robot's."""
  verdict["trace"] = f"{what}: {life.body.swap_trace()}"


def _carried(life) -> str | None:
  """Which module is on the fork right now, off the coupling itself."""
  for tool in _rack(life):
    try:
      if life.body.module_state(tool)["on_fork"]:
        return tool
    except (KeyError, ValueError):
      continue
  return None


def carry_configuration_routine(life, tool: str) -> Routine:
  """The tool as a fetch left it, before any RETURN (issue #264): a cube in
  the claw set down on the floor first (#407: a claw hung back holding one
  is a cube hung on the rack), then the arm at its carrying pose, where a
  procedure may have moved it -- a stow computes its approach from the pose
  it starts at. Returns what it set down, and what it let go of where it
  stood because no put could (no room, out of reach): a claw hung back
  holding a cube is worse."""
  set_down = dropped = None
  held = life.body.held_cube() if tool == "module_claw" else None
  if held is not None:
    rec = yield from life.body.put_cube_routine(PUT_PATIENCE_S)
    if rec.get("put"):
      set_down = held
    elif life.body.held_cube() is not None:
      yield from life.body.claw_routine(closed=False)
      dropped = held if life.body.held_cube() is None else None
  yield from life.body.retract_arm_routine()
  return {"setDown": set_down, "dropped": dropped}


#: A setpoint this close to its travel value is left alone, so a verb that
#: finds the tool already posed costs no physics step.
POSE_TOL = 1e-3


def travel_pose(life, tool: str | None) -> list[tuple[int, float, float]]:
  """The CARRYING pose as `(actuator, setpoint, speed)`, in the order to
  move them (issue #347): the arm to its carry pose over the nose (#405),
  the shoulder first; an empty fork folds the arm to its stow, the shoulder
  before the elbow, so the forearm comes in over the body rather than under
  it. It sets nothing down."""
  from pluggybot.legs.arm import CARRY_Q
  from pluggybot.procedure import axes
  body = life.body
  pose = axes._ARM.stow if tool is None else CARRY_Q
  try:
    return [(body.actuator(axes.AXES[j].actuator), target, axes.AXES[j].speed)
            for j, target in zip(axes.ARM_JOINTS, pose)]
  except KeyError:
    return []


def travel_pose_routine(life) -> Routine:
  """Whatever is on the fork into its carrying pose, ramped, before a verb
  drives (issue #347); only what a procedure moved is moved back. A
  procedure leaves the axes anywhere, and until this only a RETURN
  restored them."""
  moved = False
  for act, target, speed in travel_pose(life, _carried(life)):
    if abs(life.body.setpoint(act) - target) > POSE_TOL:
      yield from life.body.ramp_routine(act, target, speed)
      moved = True
  if moved:
    yield from life.body.settle_routine(0.5)


def _stow(life, args: dict) -> Routine:
  tool = _carried(life)
  if tool is None:
    return {"ok": False, "reason": "nothing on the fork to stow"}
  yield from carry_configuration_routine(life, tool)
  station = _tool_station(life, tool)
  why = yield from life.body.stow_tool_routine(station, tool)
  st = life.body.module_state(tool)
  hung = bool(st["hung"])
  life.swaps_done += 1
  verdict = {"ok": hung, "tool": tool, "why": why,
             **({} if hung else {"reason": (
               f"could not hang {tool} back on its bay; "
               + ("it is still on the fork" if st["on_fork"] else
                  "it is neither on the fork nor on its bay"))})}
  if not hung:
    # ...and who held the bay, where the other robot did (#418), as a
    # fetch's reason says it (`HubLifecycle.held_for`)
    asked = getattr(life, "peer_at_the_bay", None)
    blocked = asked(station) if why == "blocked" and asked is not None else None
    if blocked is not None:
      verdict["reason"] += f": {life.held_for(blocked)}"
    elif why == "no rack":        # ...got there, and saw none (issue #476)
      verdict["reason"] += (": the rack was not in sight from where I believed "
                            "its bay's approach began")
    _trace(life, verdict, f"stow {tool}")
  return verdict


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
  (issue #381, `Navigator.OPTIMISTIC`): no route is written for it."""
  x, y = float(args["x"]), float(args["y"])
  arrived = yield from _go(life, x, y, _patience(life, args), _interrupt(life))
  px, py, _ = life.body.pose
  short = round(math.hypot(x - px, y - py), 3)
  if arrived:
    return {"ok": True, "shortM": short}
  if _stopped(life):
    return _walk_stopped(life, x, y)
  # WHY it gave up, not only how far short (issue #350): "stopped 9.1 m
  # short of (22, 3)" was a route the planner could not make, and Rowan
  # read it as the pack -- then told Luca, and both declined the lab.
  why = life.drive_why(x, y)
  return {"ok": False, "shortM": short, "why": why,
          "reason": f"did not arrive at ({x:g}, {y:g}): {why}, at ({px:.1f}, {py:.1f})"}


def _face(life, args: dict) -> Routine:
  squared = yield from life.body.face_routine(float(args["heading"]))
  return {"ok": bool(squared),
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
#: How long a `press` may take, s, where the program's budget does not say
#: less: flown, 12-26 s from where the find left it (#419, 22 presses), and
#: 21-44 s past the other robot lying on or by its standoff, asked off it
#: (#439, `scripts/press_spike.py`, 30 presses). The walk there is handed
#: all of it but `FINAL_S`, one walk in's worth -- every live failure was
#: that walk -- and a second walk in runs on what is left (`PRESS_TRIES`).
PRESS_PATIENCE_S = 120.0


def _find(life, args: dict) -> Routine:
  """Find the task area carrying a tag (issue #419): where the robot saw it
  last, confirmed by its tag, or else searched for round (x, y) -- a job's
  address, in its map -- until the tag is in view. ok when it was seen;
  where it is rides the verdict (`at`), and the robot remembers it."""
  tag = int(args["tag"])
  x, y = float(args["x"]), float(args["y"])
  patience = _patience(life, {"patience": args.get("patience", FIND_PATIENCE_S)})
  rec = yield from life.body.find_tag_routine(tag, near=(x, y), patience=patience,
                                              stop=_interrupt(life))
  why = rec.get("why", "")
  out = {"ok": bool(rec.get("found")), "tag": tag, "why": why,
         "seconds": rec.get("seconds"), "remembered": bool(rec.get("remembered")),
         **({"at": list(rec["at"])} if rec.get("at") else {})}
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
      else f"did not find tag {tag}: {why or 'no reason given'}")
  if rec.get("trail"):
    out["trace"] = f"find tag {tag}: {find_trace(rec)}"
  return out


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


#: What a `press` that did not press says, by its body's why (issue #419).
PRESS_WHY = {
  "not found": "tag {tag} is a plate it has not found: `find` it first",
  "not a plate": "tag {tag} marks no plate",
  "gave up": "did not get in front of tag {tag}'s plate",
  "lost": "tag {tag} was not in view from in front of its plate",
  "not pressed": "walked onto tag {tag}'s plate and no foot was on it",
  "out of time": "ran out of time before stepping onto tag {tag}'s plate",
  "interrupted": "stopped on the way to tag {tag}'s plate by its own interrupt",
  "in the way": "did not walk onto tag {tag}'s plate",
}
#: ...and how its look round for a sign not in view ended (issue #439).
PRESS_LOOKED = {"all round": ", nor all round it",
                "cut short": ", and its time ran out looking round"}


def _across_words(life, a: dict) -> str:
  """Who lay across a press's walk in and would not come off it (issue
  #455), as its reason says it."""
  other = life._peer(a["root"]) if hasattr(life, "_peer") else None
  who = other.robot_name if other is not None else "another robot"
  where = f"across the way onto it, {float(a['m']):.1f} m off"
  if a.get("dead"):
    return f"{who} lay dead {where}"
  if a.get("down"):
    return f"{who} lay fallen {where}"
  if a.get("rests"):
    return f"{who} lay {where}, and did not step off it"
  return f"{who} stood {where}, and did not move off it"


def _press(life, args: dict) -> Routine:
  """Walk onto the plate a tag marks and back off it (issue #419): a plate
  the robot has found (`find`), the last step measured off its sign. ok
  when one of its feet was on the pad.

  ⚠ A FAILED PRESS SAYS WHICH PART FAILED (issue #439): the try that
  failed, a walk that gave up in #350's words, and every try in the log's
  `trace`. Each failed live press read "ran out of time", and the robot,
  told only that the plate was never pressed, took its sensor for faulty.
  ⚠ ...AND A PRESS THAT PRESSED LEAVES ITS TRIES THERE TOO: a paid feed
  whose press took 67 s and stepped on the shock plate 22 times on the way
  left no record of how (8a61ada), nor would a press saved by its second
  walk in."""
  tag = int(args["tag"])
  rec = yield from life.body.press_plate_routine(
    tag, patience=_patience(life, {"patience": PRESS_PATIENCE_S}), stop=_interrupt(life))
  why = rec.get("why", "")
  out = {"ok": bool(rec.get("pressed")), "tag": tag, "why": why}
  if not out["ok"]:
    att = (rec.get("attempts") or [{}])[-1]
    walk = att.get("walk")
    cause = (f": {life.drive_why(*walk['goal'], record=walk)}"
             if walk and why in ("gave up", "out of time")
             else PRESS_LOOKED.get(att.get("looked"), "") if why == "lost"
             else f": {_across_words(life, att['across'])}" if why == "in the way" and att.get("across")
             else "")
    out["reason"] = PRESS_WHY.get(why, why or "did not press").format(tag=tag) + cause
  if rec.get("attempts"):
    out["trace"] = f"press tag {tag}: {press_trace(rec)}"
  return out


def press_trace(rec: dict) -> str:
  """A press's tries as one line of evidence (issue #439), the log's
  alone: each one's standoff, how long its walk there took or the walk that
  did not arrive, the look round, where it stopped against the press pose,
  and where it ended -- the belief, and its error against the truth --
  then the time too short for the next."""
  parts = []
  for i, a in enumerate(rec.get("attempts") or (), 1):
    sx, sy = a.get("standoff") or (math.nan, math.nan)
    bits = [f"#{i} standoff ({sx:.2f}, {sy:.2f})"]
    w = a.get("walk")
    if w:
      bits.append(f"walk {_walk_trace(w)}")
    elif a.get("walkS") is not None:
      bits.append(f"walked {float(a['walkS']):.0f} s")
    if "reaim" in a:
      bits.append(f"re-aimed by {a['reaim']}")
    if "looked" in a:
      bits.append(f"looked round {a['looked']}")
    if "across" in a:
      bits.append(f"walk in across {a['across']}")
    if "walkIn" in a:
      bits.append(f"walk in {a['walkIn']}, stopped {a.get('stop')} off the press pose")
    if "at" in a:
      dx, dy, dyaw = a.get("err") or (math.nan,) * 3
      bits.append(f"at ({a['at'][0]:.2f}, {a['at'][1]:.2f}) after {a.get('s', 0):.0f} s, "
                  f"belief off {dx:+.0f},{dy:+.0f} mm {dyaw:+.1f} deg")
    parts.append(", ".join(bits) + f" -> {a.get('why', '?')}")
  if "leftS" in rec:
    parts.append(f"no time for #{len(parts) + 1}: {rec['leftS']:.1f} s left")
  return "; ".join(parts)


#: How long a `draw` may take, s, where the program's budget does not say
#: less: flown from the board's look point, a house took 106-125 s.
DRAW_PATIENCE_S = 420.0
#: What a `draw` that drew nothing says, by its body's why (issue #406).
DRAW_WHY = {
  "no pen": "the pen is not on the fork: `fetch` module_pen first",
  "not found": "{board} is a board it has not found: `find` one of its tags first",
  "gave up": "did not get in front of {board}",
  "lost": "{board}'s two tags were not in one look from in front of it",
  "never touched": "lay down in front of {board} and the pen never found its face",
  "out of time": "ran out of time before drawing on {board}",
  "interrupted": "stopped drawing on {board} by its own interrupt",
  "fell": "fell over on its way to or at {board}, and the fall threw the pen",
  "not drawn": "drew nothing on {board}",
}
#: The figures a `draw` may name: the pen's menu (`tools.strokes`) without
#: the two that write text -- the validator's list (`lifecycle.world_facts`).
DRAW_FIGURES = tuple(n for n in PROGRAMS if n not in ("text", "answer"))
#: ...and what of its record rides the verdict as `used`: what the ink's
#: evaluator reads (`scoring.sample_draw`).
DRAW_USED = ("strokes", "strokes_drawn", "inked_fraction", "travel_ink_fraction",
             "shape_rms_mm", "form_rms_mm", "offset_mm", "stopped")


def _draw(life, args: dict) -> Routine:
  """Draw a figure on a whiteboard the robot has found (issue #406): the
  pen on its fork, the board by its two tags (`find` one first). The body
  walks to it, lies down, finds its face with the pen and draws, and the
  board records the ink (`HubLifecycle.ink_hook`). ok when a stroke
  landed; a walk that gave up says why in #350's words."""
  from pluggybot.lifecycle import figure_program
  board, figure = args["board"], args["figure"]
  out = {"board": board, "figure": figure}
  if _carried(life) != "module_pen":
    return {**out, "ok": False, "why": "no pen",
            "reason": DRAW_WHY["no pen"].format(board=board)}
  program = figure_program(life.world, board, figure)
  ink = getattr(life, "ink_hook", None)
  rec = yield from life.body.draw_routine(
    board, program, patience=_patience(life, {"patience": DRAW_PATIENCE_S}),
    stop=_interrupt(life), on_stroke=ink(board, program.name) if ink is not None else None)
  why = rec.get("why", "")
  out.update(ok=bool(rec.get("drew")), why=why, strokes=rec.get("strokes"),
             strokesDrawn=rec.get("strokes_drawn"),
             used={k: rec[k] for k in DRAW_USED if rec.get(k) is not None})
  if why == "interrupted":
    out["stopped"] = "interrupted"
  if not out["ok"]:
    walk = rec.get("walk")
    cause = (f": {life.drive_why(*walk['goal'], record=walk)}"
             if walk and why == "gave up" and walk.get("goal") else "")
    out["reason"] = DRAW_WHY.get(why, why or "drew nothing").format(board=board) + cause
  return out


# ---- the claw (issue #407) ---------------------------------------------------

#: How long a cube verb may take, s, where the program's budget does not say
#: less: flown from where the robot stood a few metres off, a pick took
#: 36-41 s and a place 37-61; a cube not in view is searched for in front
#: of its area's tags, a walk or two more.
CUBE_PATIENCE_S = 240.0
#: ...and `put`, which walks nowhere, s.
PUT_PATIENCE_S = 60.0
#: What a cube verb that did not do it says, by its body's why.
CUBE_WHY = {
  "no claw": "the claw is not on the fork: `fetch` module_claw first",
  "holding": "the claw already holds a cube: `place` it or `put` it down first",
  "nothing held": "nothing is in the claw: `pick` a cube first",
  "that cube is in the claw": "cube {tag} is the one in the claw",
  "not found": "did not find cube {tag}: not from where it stood, not where it "
               "saw it last, and not in front of its area's tags (`find` one of "
               "them first)",
  "never lined up": "walked in to cube {tag} three times and never lay down with "
                    "it in the claw's reach",
  "missed": "lay down at cube {tag}, closed the jaws on it and they did not hold it",
  "the wrong cube": "lay down at cube {tag}, and the jaws closed on another cube",
  "not on it": "let go over cube {tag}, and the cube it held does not rest on it",
  "not down": "let go over the floor, and the cube does not rest on it",
  "no room": "lay down to put the cube down, and there was a cube where it could",
  "out of reach": "lay down at cube {tag} and could not reach over it",
  "fell": "fell over on the way to or at cube {tag}, and the fall threw the claw",
  "out of time": "ran out of time before reaching cube {tag}",
  "interrupted": "stopped on the way to cube {tag} by its own interrupt",
}
#: ...and what a `put` that did not says: it walks to no cube.
PUT_WHY = {**CUBE_WHY,
           "out of reach": "lay down to put cube {tag} down and could not reach the floor",
           "not down": "let go of cube {tag} over the floor, and it does not rest on it",
           "fell": "fell over putting cube {tag} down, and the fall threw the claw"}


def _claw_first(life) -> Routine:
  """The claw onto an EMPTY fork, for `pick` (issue #353, `rover-final`): a
  pick with nothing on the fork fetches the claw first, exactly as `fetch`
  takes it, and one holding another tool is refused, never driven into a
  bay loaded. None once the claw is on; else the failed verdict."""
  held = _carried(life)
  if held == "module_claw":
    return None
  if held is not None:
    return {"ok": False, "reason": f"the fork holds {held}, not the claw; stow it first"}
  if "module_claw" not in _rack(life):
    return {"ok": False, "reason": "the fork is empty, and this rack has no claw"}
  got = yield from _fetch(life, {"tool": "module_claw"})
  if not got["ok"]:
    return {**got, "reason": f"the fork was empty, and fetching the claw failed: "
                             f"{got.get('reason', '')}"}
  return None


def _cube_verdict(rec: dict, key: str, tag: int | None, words: dict = CUBE_WHY) -> dict:
  why = rec.get("why", "")
  out = {"ok": bool(rec.get(key)), "why": why,
         **({"tag": tag} if tag is not None else {}),
         **({"seconds": rec["seconds"]} if "seconds" in rec else {})}
  if why == "interrupted":
    out["stopped"] = "interrupted"
  if not out["ok"]:
    out["reason"] = words.get(why, why or "did not").format(tag=tag)
    if rec.get("tries"):
      out["trace"] = f"{rec.get('op')} {tag}: " + "; ".join(
        f"#{i} walk in {a.get('walkIn')}, lying at {a.get('at')} -> {a.get('why', 'in reach')}"
        for i, a in enumerate(rec["tries"], 1))
  return out


def _pick(life, args: dict) -> Routine:
  """Take the cube carrying a tag in the claw (issue #407): found by its tag
  -- from where the robot stands, where it saw it last, or in front of its
  area's tags -- walked in to and lain down at, then closed on. ok when the
  jaws hold it, off the world. An empty fork fetches the claw first."""
  tag = int(args["tag"])
  refused = yield from _claw_first(life)
  if refused is not None:
    return {**refused, "tag": tag}
  rec = yield from life.body.pick_cube_routine(
    tag, patience=_patience(life, {"patience": CUBE_PATIENCE_S}), stop=_interrupt(life))
  return _cube_verdict(rec, "picked", tag)


def _place(life, args: dict) -> Routine:
  """Set the cube in the claw down on top of the cube carrying a tag (issue
  #407), and let go. ok when it rests on it -- one edge above it and its
  middle within half an edge, `challenge.stack`'s "rests on" -- off the
  world."""
  tag = int(args["tag"])
  if _carried(life) != "module_claw":
    return {"ok": False, "tag": tag, "reason": CUBE_WHY["no claw"]}
  rec = yield from life.body.place_cube_routine(
    tag, patience=_patience(life, {"patience": CUBE_PATIENCE_S}), stop=_interrupt(life))
  return _cube_verdict(rec, "placed", tag)


def _put(life, args: dict) -> Routine:
  """Set the cube in the claw down on the floor in front of the robot (issue
  #407): ok when it rests there and the jaws are open."""
  if _carried(life) != "module_claw":
    return {"ok": False, "reason": CUBE_WHY["no claw"]}
  rec = yield from life.body.put_cube_routine(
    _patience(life, {"patience": PUT_PATIENCE_S}))
  return _cube_verdict(rec, "put", rec.get("held"), PUT_WHY)


def _grip(life, args: dict) -> Routine:
  if _carried(life) != "module_claw":
    return {"ok": False, "reason": CUBE_WHY["no claw"]}
  held = yield from life.body.claw_routine(closed=True)
  return {"ok": bool(held), **({} if held else {"reason": "the jaws shut on nothing"})}


def _release(life, args: dict) -> Routine:
  if _carried(life) != "module_claw":
    return {"ok": False, "reason": CUBE_WHY["no claw"]}
  free = yield from life.body.claw_routine(closed=False)
  return {"ok": bool(free), **({} if free else {"reason": "something is still in the jaws"})}


# ---- the census (issue #407) ------------------------------------------------

#: How long a `survey` may take, s, where the program's budget does not say
#: less: the garden, from its tag, walked round until 90 % of its floor was
#: seen (SimNotes, "The census on legs").
SURVEY_PATIENCE_S = 600.0
#: What a `survey` that did not survey says, by its body's why.
SURVEY_WHY = {
  "not found": "tag {tag} marks a place it has not found: `find` it first",
  "no area": "the floor in front of tag {tag} is no area its walls close",
  "out of time": "ran out of time surveying round tag {tag}",
  "interrupted": "stopped surveying round tag {tag} by its own interrupt",
}


def _survey(life, args: dict) -> Routine:
  """Survey the area a tag marks and count what stands in it (issue #407,
  the census): the floor the walls enclose round the tag's front, walked
  until enough of it was seen, the depth camera's low layer counted. ok
  when it surveyed; the count and how much of the area it saw ride the
  verdict, and the count goes on the LCD's face while the LCD is on the
  fork (`HubLifecycle.screen`). The answer is never in it: the grade reads
  the world."""
  tag = int(args["tag"])
  screen = getattr(life, "screen", None)
  shows = screen is not None and _carried(life) == getattr(screen, "module", None)

  def on_count(n: int) -> None:
    if shows:
      screen.show_count(n, "plants", face="determined", hint="none")

  rec = yield from life.body.survey_routine(
    tag, patience=_patience(life, {"patience": SURVEY_PATIENCE_S}),
    stop=_interrupt(life), on_count=on_count)
  why = rec.get("why", "")
  out = {"ok": bool(rec.get("surveyed")), "tag": tag, "why": why,
         **({"count": rec["count"], "coverage": rec["coverage"]} if "count" in rec else {}),
         "shown": shows}
  if why == "interrupted":
    out["stopped"] = "interrupted"
  if not out["ok"]:
    out["reason"] = SURVEY_WHY.get(why, why or "did not survey").format(tag=tag)
  return out


#: The base's command envelope for `drive`, forward m/s and yaw rad/s.
DRIVE_V_MAX = 0.25
DRIVE_W_MAX = 1.5


def _drive(life, args: dict) -> Routine:
  """The base at the motor level: (v, w) for a bounded time, through the
  same velocity every errand's holds and creeps go through."""
  yield from life.body.velocity_routine(float(args["seconds"]),
                                        float(args["v"]), float(args["w"]))
  return {"ok": True}


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
#: How long a `reach` waits for its hand to get there, s, and the most a
#: `move_hand` or a `pull` may.
REACH_PATIENCE_S, HAND_PATIENCE_S = 20.0, 30.0
#: The stance frame's box a hand's place is checked against, m (x, y, z).
HAND_BOX = ((-0.5, 1.0), (-0.8, 0.8), (0.0, 2.0))
#: A hand's speed when carried or pulled, m/s: the reach's own
#: (`humanoid.reach.HOLD_SPEED` is the default; `HAND_SPEED` the most).
HAND_V = (0.01, 0.06, 0.2)


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
    _reports(life).pop("spot", None)
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
    _reports(life).pop("spot", None)
    return {"ok": False, "tag": int(tag),
            "reason": f"tag {int(tag)} is not in view of the head camera from where you stand"}
  (x, y, z), (nx, ny, nz), (ex, ey, ez) = seen
  _reports(life)["spot"] = {"x": x, "y": y, "z": z, "nx": nx, "ny": ny, "nz": nz,
                            "ex": ex, "ey": ey, "ez": ez, "range": float(got["range"])}
  return {"ok": True, "tag": int(tag), "at": [round(v, 4) for v in (x, y, z)]}


#: `measure` (issue #493, an arm of its own -- sight beside feel): how far
#: round the point named its flat top is looked for, m. Under 0.35 m the
#: far edge of a 0.26 m deep top was clipped; at 0.35 and 0.45 the four
#: set-outs read their tops whole.
MEASURE_NEAR_M = 0.45


def _measure(life, args: dict) -> Routine:
  """The flat top of what stands at (x, y), off the head's depth camera at
  the part's own 848 x 480 (`Body.depth_cloud`): its height and where it
  ends -- nearest and farthest ahead, leftmost and rightmost -- in the
  stance frame, in `object.*`. ok when a flat top is seen. ⚠ OUR CODE'S
  PERCEPTION, a stated prior (`perception.box`): the highest flat face
  among the points round (x, y), and its edges."""
  import numpy as np
  from pluggybot.legs import probe as lp
  from pluggybot.perception import box
  x, y = float(args["x"]), float(args["y"])
  cloud = yield from life.body.depth_cloud(lp.DEPTH_W, lp.DEPTH_H, lp.DEPTH_MIN_Z,
                                           frames=lp.FRAMES, apart=lp.APART_S)
  bx, by, yaw = life.body.stance
  c, s_ = math.cos(yaw), math.sin(yaw)
  pts = np.asarray(cloud, dtype=float).reshape(-1, 3)
  dx, dy = pts[:, 0] - bx, pts[:, 1] - by
  st = np.column_stack([c * dx + s_ * dy, -s_ * dx + c * dy, pts[:, 2]])
  near = st[np.hypot(st[:, 0] - x, st[:, 1] - y) < MEASURE_NEAR_M]
  top = box.top_height(near) if len(near) else None
  on = near[box.on_top(near, top)] if top is not None else near[:0]
  ends = ([box.edge(on[:, 0], far=False), box.edge(on[:, 0], far=True),
           box.edge(on[:, 1], far=True), box.edge(on[:, 1], far=False)]
          if len(on) else [None] * 4)
  if top is None or None in ends:
    _reports(life).pop("measure", None)
    return {"ok": False, "reason": f"no flat top seen round ({x:g}, {y:g}) from where you stand"}
  got = dict(zip(("near", "far", "left", "right"), ends), top=top)
  _reports(life)["measure"] = got
  return {"ok": True, **{k: round(v, 3) for k, v in got.items()}}


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


def _reach(life, args: dict) -> Routine:
  """A hand's grip point to (x, y, z), its fingers along (fx, fy, fz) and
  its jaws closing along (jx, jy, jz) -- or at whatever turn the arm finds,
  both left 0 -- out short of it, then in along the fingers' way. ok once
  it holds there."""
  hand = args["hand"]
  rot, bad = _grip_turn(life, args)
  if bad:
    return {"ok": False, "reason": bad}
  place = _hand_place(life, args)
  ok, felt = yield from _felt_routine(life, hand, life.body.reach_routine(
    hand, place, rot, stiffness=float(args.get("stiffness", _STIFF.default)),
    patience=REACH_PATIENCE_S))
  if ok:
    return {"ok": True, "hand": hand, "felt": felt}
  why = life.body.hand_why(hand)
  reason = REACH_WHY.get(why, "your {hand} hand did not get there").format(hand=hand)
  other = {"left": "right", "right": "left"}.get(hand)
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
  return {"ok": False, "hand": hand, "why": why, "reason": reason, "felt": felt}


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
  when it arrives."""
  hand = args["hand"]
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
  rep, felt = yield from _felt_routine(life, hand, life.body.move_hand_routine(
    hand, end, stiffness=float(args.get("stiffness", _STIFF.default)), speed=speed,
    patience=HAND_PATIENCE_S + (2.0 * length / speed if places else 0.0),
    path=places or None, turns=turns))
  out = {"hand": hand, **_hand_report(life, hand, rep), "felt": felt,
         **({"points": len(points)} if points else {})}
  if rep.get("why") == "arrived":
    return {"ok": True, **out}
  return {"ok": False, **out,
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
  no further -- then it holds there. How far it went, the hardest it pushed
  and the way it was going at its end ride `pull.*`. ok when it pulled: why
  it stopped is in the record."""
  from pluggybot.procedure import axes
  hand = args["hand"]
  d = (float(args["dx"]), float(args["dy"]), float(args["dz"]))
  if math.sqrt(sum(v * v for v in d)) == 0.0:
    _reports(life).pop("pull", None)
    return {"ok": False, "reason": "give the pull a way: (dx, dy, dz) are all 0"}
  rep, felt = yield from _felt_routine(life, hand, life.body.pull_routine(
    hand, axes.from_stance(life, d, point=False), float(args["force"]),
    float(args["distance"]), speed=float(args.get("speed", _SPEED.default)),
    patience=HAND_PATIENCE_S, give=float(args.get("give", _GIVE.default)) >= 0.5))
  out = {"hand": hand, **_hand_report(life, hand, rep), "force": rep.get("force"),
         "felt": felt}
  if rep.get("why") in ("not out", "fell"):
    # ...and no reading: the last pull's would read as this one's
    _reports(life).pop("pull", None)
    return {"ok": False, **out,
            "reason": MOVE_WHY.get(rep["why"], "the pull did not begin").format(hand=hand)}
  way = axes.in_stance(life, rep.get("direction") or (0.0, 0.0, 0.0), point=False)
  _reports(life)["pull"] = {"moved": float(rep.get("moved", 0.0)),
                            "force": float(rep.get("force") or 0.0),
                            "dx": way[0], "dy": way[1], "dz": way[2]}
  return {"ok": True, **out}


#: Why a step stopped short, as its record says it (`Body.step_routine`).
STEP_WHY = {"holding": "a step carries nothing: open your hands first",
            "blocked": "your step got nowhere: something stands in your way",
            "fell": "you fell, and got up again",
            "out of time": "your step ran out of time"}


def _step(life, args: dict) -> Routine:
  """THE STEP (issue #525; docs/Hands.md): its hands home and empty, the
  body turns `turn` rad on the spot and walks `ahead` m and `left` m in the
  frame it then stands in, and stands -- its feet where the hands' work is. Where its stance
  went, in the stance frame it stood in before, rides `step.*` whatever
  stopped it. ok when it stepped."""
  rep = yield from life.body.step_routine(float(args["ahead"]), float(args.get("left", 0.0)),
                                          float(args.get("turn", 0.0)))
  _reports(life)["step"] = {k: float(rep.get(k, 0.0)) for k in ("dx", "dy", "turn")}
  out = {"why": rep["why"], "seconds": rep.get("seconds"),
         **{k: round(float(rep.get(k, 0.0)), 4) for k in ("dx", "dy", "turn")}}
  if rep["why"] != "stepped":
    return {"ok": False, **out, "reason": STEP_WHY.get(rep["why"], "your step stopped short")}
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
  """ok once the arm is home, by its drivers (`Body.release_routine`)."""
  hand = args["hand"]
  why, felt = yield from _felt_routine(life, hand, life.body.release_routine(hand))
  if why == "home":
    return {"ok": True, "hand": hand, "felt": felt}
  return {"ok": False, "hand": hand, "why": why, "felt": felt,
          "reason": WITHDRAW_WHY.get(why, "your {hand} arm did not get home").format(hand=hand)}


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
  "fetch": Verb("fetch", {"tool": Arg("str", choices="tools")}, _fetch,
                "pick a module off its bay; ok when seated and powered", drives=True),
  "stow": Verb("stow", {}, _stow, "hang the carried module back; ok when hung",
               drives=True),
  # `patience` (issue #381): the walk's budget, the robot's to set
  "drive_to": Verb("drive_to", {"x": Arg("float"), "y": Arg("float"),
                                "patience": Arg("float", lo=0.0, hi=MAX_PATIENCE_S,
                                                default=DRIVE_TIMEOUT_S)},
                   _drive_to, "walk to a world point over the map, and over floor not "
                   "yet on it; ok on arrival, and it gives up after "
                   f"`patience` seconds (at most {MAX_PATIENCE_S:.0f})", drives=True),
  "face": Verb("face", {"heading": Arg("float", lo=-math.pi, hi=math.pi)},
               _face, "turn in place; ok when squared within the budget", drives=True),
  # PLACES (issue #419): a task area found by its tag and remembered, and a
  # plate pressed off its sign -- the robot's own knowledge, never a
  # coordinate handed over
  "find": Verb("find", {"tag": Arg("float", choices="places"), "x": Arg("float"),
                        "y": Arg("float"),
                        "patience": Arg("float", lo=0.0, hi=MAX_PATIENCE_S,
                                        default=FIND_PATIENCE_S)},
               _find, "find the task area carrying this tag: where you last saw "
               "it, else by searching round (x, y) -- a job's address -- until "
               "the tag is in view; ok when seen, and you remember where it is. "
               f"It gives up after `patience` seconds (at most {MAX_PATIENCE_S:.0f})",
               drives=True),
  "press": Verb("press", {"tag": Arg("float", choices="plates")}, _press,
                "walk onto the plate this tag's sign marks, the last step "
                "measured off the sign, and back off it; ok when a foot was on "
                "it. `find` it first", drives=True),
  # A WHITEBOARD (issue #406): a figure off the pen's menu, on a board the
  # robot has found by its tags -- never a position
  "draw": Verb("draw", {"board": Arg("str", choices="boards"),
                        "figure": Arg("str", choices="figures")}, _draw,
               "draw a figure -- " + ", ".join(f'"{f}"' for f in DRAW_FIGURES)
               + " -- on a whiteboard by its name, one you have found (`find` "
               "one of its tags first), the pen on your fork: you walk to it, "
               "lie down in front of it, find its face with the pen and draw; "
               "ok when ink landed", drives=True),
  # THE CLAW (issue #407): a cube by its tag, found and taken, set down on
  # another or on the floor -- the cube's place is the robot's to find
  "pick": Verb("pick", {"tag": Arg("float", choices="cubes")}, _pick,
               "take the cube carrying this tag in the claw: you look for it from "
               "where you stand, where you saw it last, and in front of its "
               "area's tags (`find` one of them first), walk in, lie down and "
               "close the jaws on it; ok when they hold it. An empty fork "
               "fetches the claw first", drives=True),
  "place": Verb("place", {"tag": Arg("float", choices="cubes")}, _place,
                "set the cube in the claw down on top of the cube carrying this "
                "tag, found as `pick` finds one, and let go; ok when it rests "
                "on it", drives=True),
  "put": Verb("put", {}, _put, "set the cube in the claw down on the floor in "
              "front of you, lying down to do it; ok when it rests there"),
  "grip": Verb("grip", {}, _grip, "shut the claw's jaws where it is; ok when they "
               "hold something"),
  "release": Verb("release", {}, _release, "open the claw's jaws where it is; ok "
                  "when nothing is in them -- a cube let go high falls"),
  # THE CENSUS (issue #407): what stands in an area found by its tag
  "survey": Verb("survey", {"tag": Arg("float", choices="places")}, _survey,
                 "walk round the area this tag marks -- the floor its walls and "
                 "fence enclose -- and count what stands in it, low, off the depth "
                 "camera; `find` the tag first. ok when surveyed; its count and the "
                 "share of the area it saw ride the verdict, and the count goes on "
                 "the LCD's face while the LCD is on your fork", drives=True),
  "wait": Verb("wait", {"seconds": Arg("float", lo=0.0, hi=MAX_WAIT_S)}, _wait,
               "stand still"),
  # The motor level (issue #166): what every verb above is built from.
  "move": Verb("move", {"axis": Arg("str", choices="axes"), "target": Arg("float")},
               _move, "walk one axis to a setpoint, ramped, inside its envelope"),
  "drive": Verb("drive", {"v": Arg("float", lo=-DRIVE_V_MAX, hi=DRIVE_V_MAX),
                          "w": Arg("float", lo=-DRIVE_W_MAX, hi=DRIVE_W_MAX),
                          "seconds": Arg("float", lo=0.0, hi=MAX_WAIT_S)},
                _drive, "the base at (v m/s, w rad/s) for a bounded time",
                drives=True),
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
                  _measure, "look with your depth camera at what stands at (x, y): the height of "
                  "its flat top and where that top ends -- nearest and farthest ahead, leftmost "
                  "and rightmost -- ride `object.*`; ok when a flat top is seen"),
  "spot": Verb("spot", {"tag": Arg("float", lo=0.0, hi=586.0)}, _spot,
               "stand still and look for this tag in your head camera -- as you stand, then "
               "bowing lower until you see it -- and stand up again; ok when you see it. Where "
               "its face is, which way the face looks and which way its edges run ride "
               "`spot.*`"),
  "reach": Verb("reach", {"hand": _HAND, **_HAND_AT, **_TURN, "stiffness": _STIFF},
                _reach, "send a hand's grip -- the point between its jaws -- to (x, y, z), its "
                "fingers pointing along (fx, fy, fz) and its jaws closing along (jx, jy, jz) -- "
                "either way, a two-jaw grip -- or at whatever turn your arm finds where those "
                "are all left 0. It goes out to a point short of it, then in along the "
                "fingers; you take whatever posture reaches. stiffness 1 is your arm's full "
                "stiffness, less is softer. ok once it holds there -- or as near as it came: "
                "what it touches on its way in can hold it a few cm short, and your hand's "
                "place (`left.x` and the rest) says where it is"),
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
                        "force": Arg("float", lo=1.0, hi=60.0),
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
               "reach no further; then it holds there. How far it went, the hardest it "
               "pushed and the way it was going at its end ride `pull.*`; ok when it pulled"),
  "step": Verb("step", {"ahead": Arg("float", lo=-0.5, hi=0.5),
                        "left": Arg("float", lo=-0.5, hi=0.5, default=0.0),
                        "turn": Arg("float", lo=-3.2, hi=3.2, default=0.0)}, _step,
               "turn `turn` rad on the spot (left, +), then walk `ahead` m and `left` m in "
               "the frame you then stand in (negative: back, right), and stand still -- your "
               "hands home first; a step carries nothing. A walk goes at least a stride, "
               "about 10 cm, and may stop a few cm off what you asked; a turn shuffles your "
               "feet a few cm that `step.*` does not count. Your stance frame "
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
                   "bring a hand back out along the way it went in, and its arm home -- "
                   "straight there, or out to your side first and down at your side where "
                   "that keeps the arm higher in front of you than straight would; ok once "
                   "your arm is home"),
}
#: The hands' vocabulary (#525; docs/Hands.md): what the humanoid's
#: procedures call, the checks' and the practice's alike...
HAND_VERBS = ("spot", "reach", "via", "move_hand", "pull", "close_hand", "open_hand",
              "withdraw", "step")
#: ...and `measure`, the depth camera's, but where an arm is about its
#: absence (#493: depth is feedback, never a shortcut).
MEASURE_VERBS = ("measure",)


def language(facts: "WorldFacts | None") -> dict[str, Verb]:
  """The verbs a procedure in this world may call: `VERBS` -- or, where its
  facts name the hands' (`HANDS_VERBS`), exactly the verbs they name, so
  an unknown verb's refusal lists what the body has."""
  named = () if facts is None or facts.verbs is None else facts.verbs
  if not any(n in HANDS_VERBS for n in named):
    return VERBS
  table = {**VERBS, **HANDS_VERBS}
  return {n: table[n] for n in named}


def hand_facts(measure: bool = True, step: bool = True) -> "WorldFacts":
  """What a procedure on the humanoid's hands is validated against (issues
  #493, #525): standing, a `step` at a time, and its arms' joints, the hands' vocabulary
  (`HAND_VERBS`) -- with the depth camera's `measure` and the feet's `step`
  unless an arm is about the absence of either -- their sensors and the
  clock; a verb may be an `if`'s test, and an expression may call the maths
  (`lang.MATH`)."""
  from pluggybot.procedure import axes
  return WorldFacts(boards=(), tools=(), bounds=(-1e3, -1e3, 1e3, 1e3), figures=(),
                    axes=axes.BODY_AXES["humanoid"],
                    sensors=(axes.HAND_SENSORS + axes.PULL_SENSORS
                             + (axes.STEP_SENSORS if step else ())
                             + (axes.MEASURE_SENSORS if measure else ())),
                    verbs=(("wait", "move") + tuple(v for v in HAND_VERBS if step or v != "step")
                           + (MEASURE_VERBS if measure else ())),
                    hands=axes.HANDS, verb_tests=True, math=True)


# ---- the game's verbs (issue #404) ------------------------------------------------

#: How long a role stands by at a time while its game goes on, s.
GAME_HOLD_S = 1.0
#: ...and at a time while it waits for the game to start, s: the head start
#: runs from the start, and MEASURED, a second's wait cost the hider 0.3 m
#: of its walk.
START_POLL_S = 0.1
#: How fast a hider walks to its spot, m/s: what its head start buys it in
#: metres of walk (`_hide`). MEASURED, 5.9 m in 15-21 s (11 games), the
#: drive's 0.4 m/s cruise less its turns.
HIDE_PACE_M_S = 0.3
#: ...and the least it is left to walk, m: a hider whose errand began after
#: the head start was spent still goes round a corner.
HIDE_MIN_REACH_M = 3.0
#: The referee measures torso to torso and the map a spot: a hiding spot is
#: this much farther from where the seeker counts than a find, and the
#: seeker counts floor seen only this much nearer than a find, m.
GAME_MARGIN_M = 0.3


def _game(life) -> tuple:
  """The game this robot's errand is a role in -- its referee and its task
  id -- or (None, "")."""
  game = getattr(life, "game", None)
  errand = getattr(life, "_errand_now", None)
  return game, (getattr(errand, "task_id", "") or "")


def _game_stop(life, game, task_id: str):
  """What a role's walks stop on: its game called, or the robot's own
  interrupt, asked as every walk asks it (`_interrupt`)."""
  ask = _interrupt(life)

  def stop() -> bool:
    return game.over_for(task_id) or bool(ask is not None and ask())
  return stop


def _until_over(life, stop) -> Routine:
  """Stand by until `stop` -- the game over -- or the program's budget is
  spent (`life.step_until`)."""
  until = getattr(life, "step_until", None)
  while not stop():
    if until is not None and float(life.data.time) >= until:
      return
    yield from life.body.hold_routine(GAME_HOLD_S)


def _until_started(life, game, stop) -> Routine:
  """Stand by until the game's clock starts -- both roles' errands begun
  (`HideAndSeek.begin`) -- or `stop`, or the program's budget is spent:
  True once it has started. The robot that took the first role is often
  still busy when the other takes the last, and a role played before the
  other has begun is a game against a robot doing something else."""
  until = getattr(life, "step_until", None)
  while game.started_at is None:
    if stop() or (until is not None and float(life.data.time) >= until):
      return False
    yield from life.body.hold_routine(START_POLL_S)
  return True


def _no_game(what: str) -> dict:
  return {"ok": False, "reason": f"there is no game of hide and seek to {what} in"}


def _role_ended(life, game, task_id: str, out: dict) -> dict:
  """A role's verdict: ok once its game is over, whoever won (the referee
  decides, and the pair pays); otherwise what ended it first -- the
  robot's own interrupt, which `run_verb` says as `stopped: interrupted`,
  or the program's time."""
  if game.over_for(task_id):
    return {"ok": True, **out}
  why = ("its own interrupt" if getattr(life, "aborting", False)
         else "the program's time ran out")
  return {"ok": False, **out, "reason": f"the game was still on: {why}"}


def _hide(life, args: dict) -> Routine:
  """The hider's role (issue #404): once the game is on, a spot of its own
  choosing on its own map (`Body.hide_routine`), away from where the
  seeker counts -- where the seeker SAYS it is, as a real seeker tells a
  real hider -- walked to in the head start, and then wait there for the
  game's end. ok when the game is over; the referee decides who won."""
  game, task_id = _game(life)
  if game is None or not game.playing(task_id):
    return _no_game("hide")
  stop = _game_stop(life, game, task_id)
  rec: dict = {}
  if (yield from _until_started(life, game, stop)):
    away = life.reported_xy(game.seeker_root)
    if away is None:
      return {"ok": False, "reason": "the seeker says nothing of where it is"}
    reach = max(HIDE_PACE_M_S * game.hiding_left(float(life.data.time)),
                HIDE_MIN_REACH_M)
    rec = yield from life.body.hide_routine(
      away, reach, game.find_within_m + GAME_MARGIN_M,
      _patience(life, {"patience": MAX_PATIENCE_S}), stop=stop)
    yield from _until_over(life, stop)
  return _role_ended(life, game, task_id, {
    "hid": bool(rec.get("hid")), "why": rec.get("why", ""),
    **({"at": list(rec["at"])} if rec.get("at") else {})})


def _seek(life, args: dict) -> Routine:
  """The seeker's role (issue #404): once the game is on, count where it
  stands until the head start is spent, then search its own map outward
  from there (`Body.seek_routine`) -- never told where the hider is --
  until the game is over. ok when it is; the referee decides who won."""
  game, task_id = _game(life)
  if game is None or not game.playing(task_id):
    return _no_game("seek")
  stop = _game_stop(life, game, task_id)
  rec: dict = {}
  if (yield from _until_started(life, game, stop)):
    until = getattr(life, "step_until", None)
    while not stop() and (until is None or float(life.data.time) < until):
      left = game.hiding_left(float(life.data.time))
      if left <= 0.0:
        break
      # ...a step at least: a hold of no steps stands still in no time
      yield from life.body.hold_routine(max(min(GAME_HOLD_S, left), 0.01))
    if not stop():
      rec = yield from life.body.seek_routine(
        life.body.pose_xy(), max(0.0, game.find_within_m - GAME_MARGIN_M),
        _patience(life, {"patience": MAX_BUDGET_S}), stop=stop)
    yield from _until_over(life, stop)
  return _role_ended(life, game, task_id, {"why": rec.get("why", "")})


#: A two-role game's verbs (issue #404): its roles, each a walk over the
#: robot's own map with the game's referee watching. NOT the language's:
#: no procedure the robot writes may name one (`lang` reads `VERBS`), only
#: a game's program, validated against a world's facts that list them
#: (`lifecycle.world_facts(game=True)`).
GAME_VERBS: dict[str, Verb] = {
  "hide": Verb("hide", {}, _hide, "the hider's role: a spot of its own choosing "
               "on its own map, out of the seeker's sight, walked to in the head "
               "start; ok when the game is over", drives=True),
  "seek": Verb("seek", {}, _seek, "the seeker's role: count where it stands, "
               "then search its own map outward; ok when the game is over",
               drives=True),
}
GAME_VERB_NAMES = tuple(GAME_VERBS)


def run_verb(life, verb: Verb, args: dict, where: dict | None = None,
             until: float | None = None) -> Routine:
  """One verb, as both runners call it (this module's and the language's).
  A verb that drives puts the fork into its carrying pose first (issue
  #347) -- here, once, so a verb added later cannot forget it -- and the
  verb's own use-phase sets its working pose again on arrival.

  While it runs, `life.step_now` says which step this is (`where`: the
  procedure, the count, a line), for a death to name (issue #362), and
  `life.step_until` when the program's budget ends (`until`, sim s), past
  which no walk inside it waits (`_patience`).

  ⚠ A VERB THE ROBOT'S OWN INTERRUPT ENDED IS STOPPED, NOT FAILED (issue
  #381): a walk inside it asks the interrupt as it goes (`_interrupt`),
  and when the answer latched an abort during this verb, a verb that did
  not succeed says `stopped: interrupted` -- both runners stop the program
  there as an abort, never at a failed step."""
  before = getattr(life, "step_now", None)
  before_until = getattr(life, "step_until", None)
  aborting = bool(getattr(life, "aborting", False))
  life.step_now = {**(where or {}), "verb": verb.name, "args": dict(args)}
  life.step_until = until
  try:
    if verb.drives:
      yield from travel_pose_routine(life)
    verdict = yield from verb.run(life, args)
    if (not verdict.get("ok") and not aborting
        and getattr(life, "aborting", False)):
      verdict = {**verdict, "stopped": "interrupted"}
    return verdict
  finally:
    life.step_now = before
    life.step_until = before_until


#: What a body whose arm takes no tool can run (issue #387): walking,
#: turning, standing still and the base at a velocity, and its arm's joints
#: (#405) -- nothing that needs a tool, and no `look`, whose ranges are the
#: rover's bay tags'.
BODY_VERBS = ("drive_to", "face", "wait", "drive", "move")
#: ...and one whose world has a rack at its arm's reach (issue #405).
SWAP_VERBS = ("fetch", "stow")
#: ...and one whose world has task areas it finds by their tags (issue
#: #419), and plates among them where its world has the lab they are in.
PLACE_VERBS = ("find",)
PLATE_VERBS = ("press",)
#: ...and whiteboards it finds by their tags and draws on with its rack's
#: pen (issue #406).
DRAW_VERBS = ("draw",)
#: ...and cubes it finds by their tags and its rack's claw takes (issue #407).
CLAW_VERBS = ("pick", "place", "put", "grip", "release")
#: ...and an area it surveys, found by its tag (issue #407, the census).
SURVEY_VERBS = ("survey",)


def describe_vocabulary(verbs: tuple | None = None) -> list[dict]:
  """The verbs as data -- what a prompt or a validator's error names --
  every one, or those a body can run (`verbs`, the hands' among them). An
  argument that may be left out rides `defaults` with what it is when it is."""
  keep = VERBS if verbs is None else {n: {**VERBS, **HANDS_VERBS}[n] for n in verbs}
  out = []
  for v in keep.values():
    entry = {"verb": v.name, "args": {k: a.kind for k, a in v.args.items()},
             "doc": v.doc}
    defaults = {k: a.default for k, a in v.args.items() if a.default is not None}
    if defaults:
      entry["defaults"] = defaults
    out.append(entry)
  return out


def signature(entry: dict) -> str:
  """A described verb as a call: `drive_to(x, y, patience=60)`."""
  given = entry.get("defaults", {})
  return entry["verb"] + "(" + ", ".join(
    f"{k}={given[k]:g}" if k in given else k for k in entry["args"]) + ")"


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


#: What a tag on another list is, said beside its refusal (#407: a first
#: weighing on legs asked `find` for its cube and was told only what the
#: places are).
TAG_GLOSS = {"cubes": "is a cube's, and `pick` finds a cube by its tag",
             "places": "marks a place, which `find` finds"}


def _not_on(name: str, value: float, choices: str, facts: WorldFacts) -> str:
  """A tag on none of its list: `tag 20 names no place this world has
  (places: 35 to 46); 20 is a cube's, ...`, and a tag `Arg` carries no bound,
  which refused it a second time (#463)."""
  other = "places" if choices == "cubes" else "cubes"
  gloss = (f"; {number_text(value)} {TAG_GLOSS[other]}"
           if value in getattr(facts, other) else "")
  return (f"{name} {number_text(value)} names no {choices[:-1]} this world has "
          f"({choices}: {_tags(getattr(facts, choices))}){gloss}")


def number_text(v: float) -> str:
  """A number exactly as it is: every digit of 23.0000001, which `:g`
  quoted as `23` beside a list holding 23, and a whole float without its
  `.0` -- a tag, and the locals History reads (`procedure_outcome`)."""
  return str(v) if isinstance(v, int) else repr(float(v)).removesuffix(".0")


def _tags(tags: tuple[int, ...]) -> str:
  """A world's tags, a run of three or more as `lo to hi`: a refused define
  is one History line, cut at 400 characters, and the twelve places listed
  in full twice cut #463's own refused rewrite short (401). Never `lo..hi`,
  which reads as leaving `hi` out."""
  runs: list[list[int]] = []
  for t in tags:
    if runs and t == runs[-1][-1] + 1:
      runs[-1].append(t)
    else:
      runs.append([t])
  return ", ".join(f"{r[0]} to {r[-1]}" if len(r) >= 3 else ", ".join(map(str, r))
                   for r in runs) or "none"


def check_step(verb: Verb, args: dict, facts: WorldFacts,
               partial: bool = False) -> list[str]:
  """Every argument of one step. `partial` skips the missing-argument rule,
  for a language checking only the literal half of a call.

  ⚠ A REASON NEVER NAMES ITS CALL: whoever knows where the step is names
  it, once (`validate`, `lang.validate`, History's `procedure_outcome`:
  `line 3, wait: seconds=99.0 is above 60.0`). "tag 20 is no place",
  refusing a `find`, was read as a refusal of `place` (#463), and a reason
  that named its call too was said `pick: pick(tag=99) ...` at run time."""
  if facts is not None and facts.verbs is not None and verb.name not in facts.verbs:
    return [f"not on this body yet (it has: {', '.join(facts.verbs)})"]
  bad = []
  extra = set(args) - set(verb.args)
  missing = {k for k, a in verb.args.items() if a.default is None} - set(args)
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
      if verb is None and facts is not None and step.verb in (facts.verbs or ()):
        verb = GAME_VERBS.get(step.verb)      # a game's role (issue #404)
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
    # never inside anything else.
    if i and life.interrupted():
      result["stopped"] = "interrupted"
      break
    if float(life.data.time) - t0 > program.budget_s:
      result["stopped"] = "budget"
      break
    life._say(f"PROCEDURE {program.name} {i + 1}/{len(steps)}: {step.describe()}")
    verdict = yield from run_verb(life, VERBS.get(step.verb) or GAME_VERBS[step.verb],
                                  step.args,
                                  {"procedure": program.name, "n": i + 1,
                                   "of": len(steps)},
                                  until=t0 + program.budget_s)
    entry = {"i": i, "verb": step.verb, **verdict}
    result["steps"].append(entry)
    if verdict.get("stopped") == "interrupted":
      result["stopped"] = "interrupted"
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
