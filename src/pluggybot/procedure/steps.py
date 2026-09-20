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

  fetch(tool)          pick a module off its bay        `swap_at_bay_routine`
  stow()               hang the carried module back     `swap_at_bay_routine`
  drive_to(x, y)       A* to a world point              `drive_to_routine`
  face(heading)        turn in place                    `face_routine`
  set_lift(height)     the mast, ramped                 `HubSwap.set_lift_routine`
  grip() / release()   the claw's jaws, ramped          `ClawTool.jaws_routine`
  pick(tag)            a tagged cube, spotted and taken  `ClawTool.drive_over_routine`
  place(tag)           the held cube onto a tagged one   `ClawTool.place_on_routine`
  draw(program, board) the pen's whole use-phase        `drawing_errand`
  look()               one tag decode, no motion        `TagSpotter.detect`
  wait(seconds)        stand still

Each returns a verdict dict with `ok`, measured off the world (the module
seated and powered, the jaws in contact, the ink in the board book), never
off the command. The runner stops at the first failed step: later steps
assume earlier ones, and running `draw` with no pen on the fork is a pen
pressed at empty air. The result is honest about how far it got.

Three rules the shape carries for the rungs above it:

  ROLES. A program is `{role: steps}` and today every program has one role,
  `robot`. M12's two-role errand (a hider and a seeker, one verdict) is a
  second key, not a rebuild -- so the slot validates any number of roles and
  the runner refuses more than one, with the reason.

  BUDGETS ARE CAPPED BY CODE. A program declares its own sim-time budget and
  the vocabulary caps it (`MAX_BUDGET_S`) and the step count (`MAX_STEPS`);
  a program past either does not validate. The runner checks the budget at
  step boundaries, which are the safe points: a stopped drive is safe, a
  stroke is not, and the draw verb hands the plotter `life.interrupted` for
  the one place mid-step matters (issue #116).

  ABORT MEANS STOW. Whatever ends a program early -- a failed step, the
  budget, an interrupt -- the errand around it hangs any carried module back
  before the verdict, exactly as a native errand does. A program cannot
  leave a tool on the floor by stopping.

What a program may say is what a work order may say (TaskPattern.md §2): a
tool by name, a surveyed place by coordinate, a board by id. A program that
drives to where a movable object IS has been handed a sensor reading over the
wire, and the ladder applies to it as to any task.
"""

import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable

from pluggybot.rack.coupling import HUB_STATION_YS, module_power_contact
from pluggybot.tick import Routine

#: Which bay each module hangs in (`HUB_STATION_YS` is indexed by bay, and
#: bay <-> tag pairing is by that index -- ToolPattern.md §6).
TOOL_BAYS = {"module_lcd": 0, "module_plug": 1, "module_pen": 2,
             "module_claw": 3, "module_seed": 4}
#: The one role every program has today; M12 adds the second.
DEFAULT_ROLE = "robot"
#: Caps, by code, on what a program may declare.
MAX_STEPS = 24
MAX_BUDGET_S = 1800.0
DEFAULT_BUDGET_S = 600.0
MAX_WAIT_S = 60.0
#: Per-step drive timeout: the native errand's carry drive uses 60 s.
DRIVE_TIMEOUT_S = 60.0
#: The lift's travel, as the plotter clips it.
LIFT_RANGE_M = (0.02, 0.30)
LIFT_SPEED = 0.05       # m/s, the lead-screw class ceiling (tools/gripper.py)


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
  against the world's box."""

  kind: str                    # "float" | "str"
  lo: float | None = None
  hi: float | None = None
  choices: str | None = None


@dataclass(frozen=True)
class Verb:
  name: str
  args: dict                   # arg name -> Arg
  run: Callable[..., Routine]  # (life, args) -> Routine returning a verdict
  doc: str


def _rack(life) -> dict[str, int]:
  """Which module hangs where: the lifecycle's inventory, which the
  workshop edits (issue #168), or the shipped five where there is none."""
  return getattr(life, "rack_inventory", None) or TOOL_BAYS


def _tool_station(life, tool: str) -> float:
  return HUB_STATION_YS[_rack(life)[tool]]


def _fetch(life, args: dict) -> Routine:
  tool = args["tool"]
  station = _tool_station(life, tool)
  life.module = tool
  why = yield from life.mission.swap_at_bay_routine(station, "pick", module=tool)
  st = life.mission.swap.module_state(tool)
  ok = bool(st["on_fork"]) and module_power_contact(
    life.model, life.data, tool, life.mission.swap.handle.prefix)
  life.swaps_done += 1
  return {"ok": ok, "tool": tool, "why": why, "powered": ok}


def _carried(life) -> str | None:
  """Which module is on the fork right now, off the coupling itself."""
  for tool in _rack(life):
    try:
      if life.mission.swap.module_state(tool)["on_fork"]:
        return tool
    except (KeyError, ValueError):
      continue
  return None


def _stow(life, args: dict) -> Routine:
  tool = _carried(life)
  if tool is None:
    return {"ok": False, "reason": "nothing on the fork to stow"}
  why = yield from life.mission.swap_at_bay_routine(_tool_station(life, tool),
                                                    "return", module=tool)
  hung = bool(life.mission.swap.module_state(tool)["hung"])
  life.swaps_done += 1
  return {"ok": hung, "tool": tool, "why": why}


def _drive_to(life, args: dict) -> Routine:
  x, y = float(args["x"]), float(args["y"])
  arrived = yield from life.mission.drive_to_routine(x, y, timeout=DRIVE_TIMEOUT_S)
  px, py, _ = life.mission.pose
  return {"ok": bool(arrived), "shortM": round(math.hypot(x - px, y - py), 3)}


def _face(life, args: dict) -> Routine:
  squared = yield from life.mission.face_routine(float(args["heading"]))
  return {"ok": bool(squared)}


def _set_lift(life, args: dict) -> Routine:
  yield from life.mission.swap.set_lift_routine(float(args["height"]),
                                                speed=LIFT_SPEED)
  return {"ok": True, "height": float(args["height"])}


def _claw(life):
  from pluggybot.tools.gripper import CLAW_MODULE, ClawTool
  if _carried(life) != CLAW_MODULE:
    return None
  return ClawTool(life.model, life.data, life.mission.swap)


def _holding_anything(claw) -> str | None:
  """Both pads touching the same geom that belongs neither to the robot nor
  to the claw module itself: `ClawTool.held`, the grip's contact criterion
  without a named target."""
  return claw.held()


#: The lifts a `pick`/`place` looks from, in the order tried. MEASURED
#: (issue #264, the home world's 20 mm block tags off the dock eye): a tag
#: that size is ~24 px wide at 0.8 m and decodes patchily -- which lift
#: sees it changes with the range by a few centimetres, and none sees the
#: floor inside ~0.65 m, where it leaves the bottom of the frame. So the
#: verb hunts, as `swap_at_bay` re-looks for a bay, and the lifts stay
#: above the pads' floor contact (0.03) with a block in the jaws.
SPOT_LIFTS = (0.06, 0.045, 0.075, 0.09, 0.105, 0.12)
#: From the grip pose the row is ~0.3 m ahead of the axle; half a metre
#: back puts it in the band the eye decodes from.
SPOT_BACK_OFF_M = 0.5
#: A decode is trusted for the approach only when the tag sits this close
#: to the camera's axis. MEASURED: a block tag seen 0.26 m off-axis at
#: 0.8 m placed its centre 25 mm long and 13 mm across (PnP on a ~24 px
#: tag); on-axis the same decode is good to a millimetre or two, which is
#: what the pick got. Off-axis, the verb first stages itself to look
#: head-on from `STAGE_M` short of the cube and spots again.
SPOT_ON_AXIS_M = 0.03
STAGE_M = 0.45
#: The objects `pick`/`place` know the shape of: the challenge blocks and
#: the bench's masses (challenge/stack.py's cube, tagged on every face), so
#: a decoded face is half an edge from the centre and the top of one is
#: half an edge above its tag.
_CUBE_TAGS: dict[int, float] = {}


def _cube_half(tag: int) -> float | None:
  if not _CUBE_TAGS:
    from pluggybot.challenge.stack import BLOCK_HALF
    from pluggybot.rack.tags import BLOCK_TAG_IDS, MASS_TAG_IDS
    for i in (*BLOCK_TAG_IDS, *MASS_TAG_IDS):
      _CUBE_TAGS[int(i)] = BLOCK_HALF
  return _CUBE_TAGS.get(int(tag))


def _spot_routine(life, tag: int) -> Routine:
  """Find one tagged cube from where the robot stands: a look at each of
  `SPOT_LIFTS` until the tag decodes, the cube's centre in the believed
  world frame off that decode (`HubMission.spot`: the tag's centre, half
  an edge further along the line of sight), or None. A sensor's answer --
  the robot has to have driven somewhere it can see the thing -- with one
  allowance: a robot that has just picked or placed stands with the grip
  point over the row, inside the eye's blind zone, so a first miss backs
  the chassis off `SPOT_BACK_OFF_M` and looks once more."""
  half = _cube_half(tag)
  if half is None:
    return None
  swap = life.mission.swap
  for attempt in range(2):
    if attempt:
      yield from swap._drive_until_routine(SPOT_BACK_OFF_M, -0.10, stall_stop=False)
      yield from swap._run_routine(0.5, 0.0)
    for lift in SPOT_LIFTS:
      yield from swap.ramp_routine(swap.lift_act, lift, LIFT_SPEED, settle=0.3)
      seen = life.mission.spot(tag)
      if seen is not None:
        x, y, z = seen["xyz"]
        tx, ty = seen["toward"]
        return {**seen, "centre": (x + half * tx, y + half * ty, z),
                "half": half, "lift": lift, "backedOff": bool(attempt)}
  return None


def _approach_routine(life, claw, tag: int, carrying: bool,
                      hang: tuple[float, float] = (0.0, 0.0)) -> Routine:
  """Spot the cube, then put the grip point over it: `ClawTool.
  drive_over_routine`, the runway-and-converge approach the pickup demo
  measured to a few millimetres. A first decode taken off-axis is only
  good enough to STAGE by (`SPOT_ON_AXIS_M`): the robot drives to look at
  the cube head-on from `STAGE_M` short of it, spots again, and approaches
  off that. Carrying, the lift goes back to carry height before every
  drive -- the looks are taken low, and at a spot lift the pads clear a
  floor block's top by ~3 mm (measured pushing the target 8 cm along the
  floor on a run-in). Returns (seen, arrived); seen is None when the tag
  was never decoded."""
  from pluggybot.tools.gripper import CARRY_LIFT
  claw.calibrate_from_body()
  seen = yield from _spot_routine(life, tag)
  if seen is None:
    return None, False
  heading = life.mission.pose[2]
  if abs(seen["lateral"]) > SPOT_ON_AXIS_M:
    # stage along the heading the procedure chose (it faced the row), so
    # the cube ends up straight ahead of the fork line -- the camera's line
    x, y, _ = seen["centre"]
    if carrying:
      yield from claw.set_lift_routine(CARRY_LIFT, settle=0.5)
    yield from claw.drive_over_routine((x - STAGE_M * math.cos(heading),
                                        y - STAGE_M * math.sin(heading)),
                                       heading, stow=not carrying)
    again = yield from _spot_routine(life, tag)
    if again is not None:
      seen = {**again, "staged": True}
  x, y, _ = seen["centre"]
  # aim the held object at the target: its hang in the chassis frame,
  # turned into the world at the approach heading, comes off the goal
  c, sn = math.cos(heading), math.sin(heading)
  x -= c * hang[0] - sn * hang[1]
  y -= sn * hang[0] + c * hang[1]
  if carrying:
    yield from claw.set_lift_routine(CARRY_LIFT, settle=0.5)
  arrived = yield from claw.drive_over_routine((x, y), heading, stow=not carrying)
  return seen, bool(arrived)


def _pick(life, args: dict) -> Routine:
  """Pick up the cube carrying a tag (issue #264): spot it, drive the grip
  point over it (`ClawTool.drive_over_routine`, the runway-and-converge
  approach the pickup demo measured to a few millimetres), close, lift.
  ok when both pads hold something afterwards -- measured, as `grip` is."""
  claw = _claw(life)
  if claw is None:
    return {"ok": False, "reason": "the claw is not on the fork"}
  held = claw.held()
  if held is not None:
    return {"ok": False, "reason": f"already holding {held}"}
  tag = int(args["tag"])
  seen, arrived = yield from _approach_routine(life, claw, tag, carrying=False)
  if seen is None:
    return {"ok": False, "tag": tag,
            "reason": f"tag {tag} is not a cube this robot can see from here"}
  picked = yield from claw.pick_up_routine()
  held = claw.held()
  return {"ok": held is not None, "tag": tag, "arrived": bool(arrived),
          "holding": held, "seenAtM": round(seen["range"], 3),
          "grippedBeforeLift": bool(picked.get("gripped_before_lift"))}


def _place(life, args: dict) -> Routine:
  """Set the held cube down on top of the cube carrying a tag: spot it,
  approach carrying (lift up, arm out), lower to its top, let go, back
  off (`ClawTool.place_on_routine`). ok is MEASURED off the world after
  the retreat: the cube that was held now rests on the target -- one
  pitch above it and within half an edge sideways (challenge/stack.py's
  own "rests on") -- and the jaws are empty. A block that fell beside
  says so."""
  claw = _claw(life)
  if claw is None:
    return {"ok": False, "reason": "the claw is not on the fork"}
  held = claw.held()
  if held is None:
    return {"ok": False, "reason": "nothing in the jaws to place"}
  tag = int(args["tag"])
  # how the held cube hangs in the jaws (`ClawTool.held_hang`): the verb
  # aims the CUBE at the target, not the grip point
  hang = claw.held_hang(held)
  seen, arrived = yield from _approach_routine(life, claw, tag, carrying=True,
                                               hang=hang[:2])
  if seen is None:
    return {"ok": False, "tag": tag,
            "reason": f"tag {tag} is not a cube this robot can see from here"}
  x, y, z = seen["centre"]
  half = seen["half"]
  # The top of the cube: which LAYER its tag sits in, off the decode's
  # height and the cube's known edge -- a reading quantised by geometry,
  # not the raw z (8 mm low off-axis), because a release aimed below the
  # surface presses the held block into it and rides the module up its fork.
  layer = max(0, int(round((z - half) / (2 * half))))
  yield from claw.place_on_routine((layer + 1) * 2 * half,
                                   bottom_below_grip=half - hang[2])
  model, data = life.model, life.data
  bid = int(model.geom_bodyid[model.geom(held).id])
  hx, hy, hz = (float(v) for v in data.xpos[bid])
  target = int(model.geom_bodyid[model.geom(_cube_geom(model, tag)).id]) \
    if _cube_geom(model, tag) else None
  if target is not None:
    tx, ty, tz = (float(v) for v in data.xpos[target])
  else:
    tx, ty, tz = x, y, z
  off = math.hypot(hx - tx, hy - ty)
  dz = hz - tz
  from pluggybot.challenge.stack import PITCH_M, PITCH_TOL_M, REST_OFFSET_M
  rests = abs(dz - PITCH_M) <= PITCH_TOL_M and off <= REST_OFFSET_M
  out = {"ok": rests and claw.held() is None, "tag": tag, "arrived": bool(arrived),
         "placed": held, "offsetMm": round(off * 1000, 1),
         "aboveMm": round(dz * 1000, 1)}
  if not rests:
    out["reason"] = (f"released, but it rests {off * 1000:.0f} mm across and "
                     f"{dz * 1000:.0f} mm up from the target, not on it")
  elif claw.held() is not None:
    out["reason"] = "the jaws did not let go"
  return out


def _cube_geom(model, tag: int) -> str | None:
  """The box geom of the cube tagged `tag`, if this world has it (the
  tower's blocks and the bench's masses carry their tag id in `tags.py`)."""
  from pluggybot.challenge.stack import BLOCKS
  from pluggybot.rack.tags import BLOCK_TAG_IDS, MASS_TAG_IDS
  names = dict(zip(BLOCK_TAG_IDS, BLOCKS))
  names.update(zip(MASS_TAG_IDS, ("mass_known", "mass_unknown")))
  body = names.get(int(tag))
  if body is None:
    return None
  try:
    return model.geom(f"{body}_box").name
  except KeyError:
    return None


def _grip(life, args: dict) -> Routine:
  claw = _claw(life)
  if claw is None:
    return {"ok": False, "reason": "the claw is not on the fork"}
  yield from claw.jaws_routine(1.0, settle=1.2)
  held = _holding_anything(claw)
  return {"ok": held is not None, "holding": held}


def _release(life, args: dict) -> Routine:
  claw = _claw(life)
  if claw is None:
    return {"ok": False, "reason": "the claw is not on the fork"}
  yield from claw.jaws_routine(0.0, settle=1.0)
  return {"ok": _holding_anything(claw) is None}


def _draw(life, args: dict) -> Routine:
  from pluggybot.lifecycle import draw_errand_for
  if _carried(life) != "module_pen":
    return {"ok": False, "reason": "the pen is not on the fork"}
  errand = draw_errand_for(life.world, life.boards, args["board"],
                           program_name=args["figure"])
  used = yield from errand.use(life)
  used = {k: v for k, v in (used or {}).items() if k != "plotter"}
  return {"ok": bool(used.get("drew")), "board": args["board"],
          "figure": args["figure"], "strokes": used.get("strokes"),
          "strokesDrawn": used.get("strokes_drawn"),
          **({"reason": used["reason"]} if used.get("reason") else {}),
          # the plotter's whole account, for the ink evaluator to read as
          # it reads a native drawing's
          "used": used}


def _look(life, args: dict) -> Routine:
  """One decode from the dock camera, no motion: the sensed result a rung-two
  program branches on through `read("look.tag")` / `look.range` /
  `look.lateral` (procedure/axes.py), which read the NEAREST decode of the
  last look. Camera-frame `t` is (lateral, vertical, forward)."""
  found = life.mission.tags.detect(life.data)
  tags = [{"id": int(tid), "lateralM": round(float(d["t"][0]), 3),
           "forwardM": round(float(d["t"][2]), 3)}
          for tid, d in sorted(found.items())]
  nearest = min(tags, key=lambda t: t["forwardM"], default=None)
  life._last_look = ({"tag": nearest["id"], "range": nearest["forwardM"],
                      "lateral": nearest["lateralM"]} if nearest else {})
  return {"ok": True, "tags": tags}
  yield  # a routine that steps nothing


def _wait(life, args: dict) -> Routine:
  yield from life.mission._drive_routine(float(args["seconds"]), 0.0, 0.0)
  return {"ok": True}


#: The base's command envelope for `drive`: the cruise the navigation law
#: uses, and the spin rate `_spin` turns at (mission/mission.py).
DRIVE_V_MAX = 0.25
DRIVE_W_MAX = 1.5


def _drive(life, args: dict) -> Routine:
  """The base at the motor level: (v, w) for a bounded time, through the
  same `_drive_routine` every errand's holds and creeps go through."""
  yield from life.mission._drive_routine(float(args["seconds"]),
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
  yield from axis.run(life, target)
  return {"ok": True, "axis": axis.name, "target": target}


VERBS: dict[str, Verb] = {
  "fetch": Verb("fetch", {"tool": Arg("str", choices="tools")}, _fetch,
                "pick a module off its bay; ok when seated and powered"),
  "stow": Verb("stow", {}, _stow, "hang the carried module back; ok when hung"),
  "drive_to": Verb("drive_to", {"x": Arg("float"), "y": Arg("float")},
                   _drive_to, "A* to a world point inside the map; ok on arrival"),
  "face": Verb("face", {"heading": Arg("float", lo=-math.pi, hi=math.pi)},
               _face, "turn in place; ok when squared within the budget"),
  "set_lift": Verb("set_lift", {"height": Arg("float", lo=LIFT_RANGE_M[0],
                                              hi=LIFT_RANGE_M[1])},
                   _set_lift, "walk the mast to a height, ramped"),
  "grip": Verb("grip", {}, _grip, "close the claw; ok when both pads hold something"),
  "release": Verb("release", {}, _release, "open the claw; ok when nothing is held"),
  # The claw's pair (issue #264), at `fetch`/`stow`'s level: a tagged cube
  # found from where the robot stands, approached closed-loop, taken or set
  # down on another. The dock eye sees a cube's 20 mm tag from roughly
  # 0.65-1.0 m and not closer, so a procedure drives within about a metre
  # and faces it first.
  "pick": Verb("pick", {"tag": Arg("float", lo=0, hi=999)}, _pick,
               "find the cube carrying this tag from here, drive over it and "
               "take it; ok when the jaws hold it"),
  "place": Verb("place", {"tag": Arg("float", lo=0, hi=999)}, _place,
                "set the held cube down on top of the cube carrying this tag "
                "and back off; ok when it rests there"),
  "draw": Verb("draw", {"figure": Arg("str", choices="figures"),
                        "board": Arg("str", choices="boards")},
               _draw, "the pen's use-phase on a board; ok when ink landed"),
  "look": Verb("look", {}, _look, "one tag decode from the dock camera, no motion"),
  "wait": Verb("wait", {"seconds": Arg("float", lo=0.0, hi=MAX_WAIT_S)}, _wait,
               "stand still"),
  # The motor level (issue #166): what every verb above is built from.
  "move": Verb("move", {"axis": Arg("str", choices="axes"), "target": Arg("float")},
               _move, "walk one axis to a setpoint, ramped, inside its envelope"),
  "drive": Verb("drive", {"v": Arg("float", lo=-DRIVE_V_MAX, hi=DRIVE_V_MAX),
                          "w": Arg("float", lo=-DRIVE_W_MAX, hi=DRIVE_W_MAX),
                          "seconds": Arg("float", lo=0.0, hi=MAX_WAIT_S)},
                _drive, "the base at (v m/s, w rad/s) for a bounded time"),
}


def describe_vocabulary() -> list[dict]:
  """The verbs as data -- what a prompt or a validator's error names."""
  return [{"verb": v.name, "args": {k: a.kind for k, a in v.args.items()},
           "doc": v.doc} for v in VERBS.values()]


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
  bad = []
  extra = set(args) - set(verb.args)
  missing = set(verb.args) - set(args)
  if extra:
    bad.append(f"{verb.name} takes no {', '.join(sorted(extra))}")
  if missing and not partial:
    bad.append(f"{verb.name} needs {', '.join(sorted(missing))}")
  for name in verb.args:
    if name in args:
      bad += check_arg(verb, name, args[name], facts)
  if verb.name == "drive_to" and all(
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
    # Safe points: between steps. A hazard row (issue #116) or the budget
    # stops the program here, never inside a step.
    if i and life.interrupted():
      result["stopped"] = "interrupted"
      break
    if float(life.data.time) - t0 > program.budget_s:
      result["stopped"] = "budget"
      break
    life._say(f"PROCEDURE {program.name} {i + 1}/{len(steps)}: {step.describe()}")
    verdict = yield from VERBS[step.verb].run(life, step.args)
    entry = {"i": i, "verb": step.verb, **verdict}
    result["steps"].append(entry)
    if not verdict.get("ok"):
      result["failedAt"] = i
      life._say(f"PROCEDURE {program.name} failed at {i + 1}/{len(steps)} "
                f"{step.describe()}" + (f" -- {verdict['reason']}"
                                         if verdict.get("reason") else ""))
      break
    result["completed"] = i + 1
  else:
    result["ok"] = True
  result["seconds"] = round(float(life.data.time) - t0, 2)
  return result
