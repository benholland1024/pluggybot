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
  """One typed argument. `choices` is a name on `WorldFacts` (the world
  decides), `lo`/`hi` bound a number, `bounds` says a pair is checked
  against the world's box. One with a `default` may be left out."""

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
  """Which module hangs where: the lifecycle's inventory, which the
  workshop edits (issue #168), or the rack's own where there is none."""
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
  # ...the tool's own axes at rest, the pose it hangs plumb in (#407: a
  # built tool's stow -- a hinge a procedure left swung was hung back swung,
  # off plumb, and lost)
  for act, target, speed in tool_rest(life, tool):
    if abs(life.body.setpoint(act) - target) > POSE_TOL:
      yield from life.body.ramp_routine(act, target, speed)
  yield from life.body.retract_arm_routine()
  return {"setDown": set_down, "dropped": dropped}


#: A setpoint this close to its travel value is left alone, so a verb that
#: finds the tool already posed costs no physics step.
POSE_TOL = 1e-3


def tool_rest(life, tool: str | None) -> list[tuple[int, float, float]]:
  """A tool's own axes at rest, as `(actuator, setpoint, speed)`: each at
  its joint's compiled value, the pose the tool hung in -- a built tool's
  stow, which its joint's `ref` makes `qpos0` (`workshop.build.face_xml`)."""
  from pluggybot.procedure import axes
  model, own = life.model, []
  for axis in axes.AXES.values():
    if tool is None or axis.requires != tool or not axis.actuator:
      continue
    try:
      act = model.actuator(axis.actuator)
    except KeyError:
      continue
    rest = float(model.qpos0[model.jnt_qposadr[act.trnid[0]]])
    own.append((act.id, min(max(rest, axis.lo), axis.hi), axis.speed))
  return own


def travel_pose(life, tool: str | None) -> list[tuple[int, float, float]]:
  """The CARRYING pose as `(actuator, setpoint, speed)`, in the order to
  move them (issue #347): a tool's own axes to rest (`tool_rest`), then the
  arm to its carry pose over the nose (#405), the shoulder first; an empty
  fork folds the arm to its stow, the shoulder before the elbow, so the
  forearm comes in over the body rather than under it. It sets nothing
  down."""
  from pluggybot.legs.arm import CARRY_Q
  from pluggybot.procedure import axes
  body = life.body
  pose = axes._ARM.stow if tool is None else CARRY_Q
  try:
    arm = [(body.actuator(axes.AXES[j].actuator), target, axes.AXES[j].speed)
           for j, target in zip(axes.ARM_JOINTS, pose)]
  except KeyError:
    return []
  return tool_rest(life, tool) + arm


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
#: 21-52 s past the other robot lying on or by its standoff, asked off it
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
    out["reason"] = (
      f"stopped looking for tag {tag} by its own interrupt" if why == "interrupted"
      else f"tag {tag} marks no place in this world" if why == "not a place"
      else f"{searched}, having looked from every place near there{looks}" if why == "not found"
      else f"{searched} and ran out of time{looks}" if why == "out of time"
      else f"did not find tag {tag}: {why or 'no reason given'}")
  return out


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
      peer = "".join(f" {k}={w[k]}" for k in ("peerAt", "peerM", "peerDown", "peerRests",
                                               "peerDead", "askedWay")
                     if k in w)
      bits.append(f"walk {w.get('why')} {float(w.get('shortM') or 0):.2f} m short "
                  f"after {float(w.get('seconds') or 0):.0f} s{peer}")
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
  present -- a tool's axis needs that tool on the fork -- and the target
  inside its envelope, or the step fails and says which."""
  from pluggybot.procedure import axes
  axis = axes.AXES.get(args["axis"])
  if axis is None:
    return {"ok": False, "reason": f"no axis {args['axis']!r}"}
  if axis.requires and _carried(life) != axis.requires:
    return {"ok": False, "reason": f"axis {axis.name!r} needs {axis.requires} "
                                    "on the fork"}
  target = float(args["target"])
  if not axis.lo <= target <= axis.hi:
    return {"ok": False, "reason": f"{axis.name} target {target} is outside "
                                    f"{axis.lo}..{axis.hi} {axis.unit}".rstrip()}
  moved = yield from axis.run(life, target)
  if moved is False:        # a joint that ran out of time (the quadruped's arm)
    return {"ok": False, "reason": f"{axis.name} did not reach {target} "
                                    f"{axis.unit} in time".rstrip()}
  return {"ok": True, "axis": axis.name, "target": target}


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
  "find": Verb("find", {"tag": Arg("float", lo=0, hi=999), "x": Arg("float"),
                        "y": Arg("float"),
                        "patience": Arg("float", lo=0.0, hi=MAX_PATIENCE_S,
                                        default=FIND_PATIENCE_S)},
               _find, "find the task area carrying this tag: where you last saw "
               "it, else by searching round (x, y) -- a job's address -- until "
               "the tag is in view; ok when seen, and you remember where it is. "
               f"It gives up after `patience` seconds (at most {MAX_PATIENCE_S:.0f})",
               drives=True),
  "press": Verb("press", {"tag": Arg("float", lo=0, hi=999)}, _press,
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
  "pick": Verb("pick", {"tag": Arg("float", lo=0, hi=999)}, _pick,
               "take the cube carrying this tag in the claw: you look for it from "
               "where you stand, where you saw it last, and in front of its "
               "area's tags (`find` one of them first), walk in, lie down and "
               "close the jaws on it; ok when they hold it. An empty fork "
               "fetches the claw first", drives=True),
  "place": Verb("place", {"tag": Arg("float", lo=0, hi=999)}, _place,
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
  "survey": Verb("survey", {"tag": Arg("float", lo=0, hi=999)}, _survey,
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
  every one, or those a body can run (`verbs`). An argument
  that may be left out rides `defaults` with what it is when it is."""
  keep = VERBS if verbs is None else {n: VERBS[n] for n in verbs}
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
    return bad
  if not isinstance(value, str):
    return [f"{name} must be a name, not {value!r}"]
  allowed = getattr(facts, arg.choices) if arg.choices else None
  if allowed is not None and value not in allowed:
    return [f"{name}={value!r} is not one of "
            f"{', '.join(allowed) or 'nothing this world has'}"]
  return []


def check_step(verb: Verb, args: dict, facts: WorldFacts,
               partial: bool = False) -> list[str]:
  """Every argument of one step. `partial` skips the missing-argument rule,
  for a language checking only the literal half of a call."""
  if facts is not None and facts.verbs is not None and verb.name not in facts.verbs:
    return [f"{verb.name}: not on this body yet "
            f"(it has: {', '.join(facts.verbs)})"]
  bad = []
  extra = set(args) - set(verb.args)
  missing = {k for k, a in verb.args.items() if a.default is None} - set(args)
  if extra:
    bad.append(f"{verb.name} takes no {', '.join(sorted(extra))}")
  if missing and not partial:
    bad.append(f"{verb.name} needs {', '.join(sorted(missing))}")
  for name in verb.args:
    if name in args:
      bad += check_arg(verb, name, args[name], facts)
  if verb.name in ("find", "press", "survey") and isinstance(args.get("tag"), (int, float)) \
      and not isinstance(args.get("tag"), bool):
    have = facts.places if verb.name in ("find", "survey") else facts.plates
    if int(args["tag"]) != args["tag"] or int(args["tag"]) not in have:
      what = "plate" if verb.name == "press" else "place"
      # ...and what it is, where it is a cube's: a first weighing on legs
      # asked `find` for its cube and was told only what the places are
      cube = (f"; {args['tag']:g} is a cube's, and `pick` finds a cube by its tag"
              if args["tag"] in facts.cubes else "")
      bad.append(f"tag {args['tag']:g} is no {what} this world has "
                 f"(have: {', '.join(str(t) for t in have) or 'none'}){cube}")
  if verb.name in ("pick", "place") and isinstance(args.get("tag"), (int, float)) \
      and not isinstance(args.get("tag"), bool):
    if int(args["tag"]) != args["tag"] or int(args["tag"]) not in facts.cubes:
      place = (f"; {args['tag']:g} marks a place, which `find` finds"
               if args["tag"] in facts.places else "")
      bad.append(f"tag {args['tag']:g} is no cube this world has "
                 f"(have: {', '.join(str(t) for t in facts.cubes) or 'none'}){place}")
  if verb.name in ("drive_to", "find") and all(
      isinstance(args.get(k), (int, float)) and not isinstance(args.get(k), bool)
      for k in ("x", "y")):
    x0, y0, x1, y1 = facts.bounds
    x, y = float(args["x"]), float(args["y"])
    if not (x0 <= x <= x1 and y0 <= y <= y1):
      bad.append(f"({x}, {y}) is outside the map [{x0}, {x1}] x [{y0}, {y1}]")
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
      bad += [f"{where}: {r}" for r in check_step(verb, step.args, facts)]
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
