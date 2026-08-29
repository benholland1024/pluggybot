"""Errands: a tool, a place, and something to DO there (issue #12).

The hub lifecycle used to have exactly one errand hardcoded into it -- fetch
`module_lcd`, drive to a hardcoded point, drive back -- and its `USE_TOOL`
state did no work at all. That was honest for milestone 8, whose claim was
about the swap rather than about the tool, but it means the robot's whole
life is "carry a thing somewhere and put it back".

An errand is that shape with the middle filled in:

    SWAP_PICK  ->  USE_TOOL  ->  SWAP_RETURN
    fetch `module`  drive to `use_at`,   hang it back
    from `station_y`  then run `use()`     in `station_y`

`use` is an arbitrary callable handed the live `HubLifecycle`, so the drawing
errand below and the claw/garden errands that come later are the SAME machinery
with a different middle -- and the pick/carry/stow half, which is the part that
took two issues to make repeatable, has exactly one implementation.

Two things the split buys immediately:

  - `scripts/home_draw.py` stops being a parallel mission stack. It was a
    second copy of "spawn, explore-ish, fetch, drive, draw, stow" that drifted
    from the lifecycle's copy; now it builds an errand and hands it over.
  - The overseer (issue #15) gets a menu it can choose from without knowing
    any physics: an errand is a name, a tool and a destination.

The one rule an errand's `use` must respect: LEAVE THE TOOL IN ITS CARRY
CONFIGURATION. Every stow computes its release heights from the lift it starts
at, and a tool axis parked where the last stroke left it fouls the bay's
brackets (docs/SimNotes.md, "The pen would not stow"). `PenPlotter.carry_config`
is that for the pen; a tool without a moving axis has nothing to do.
"""

import math
from dataclasses import dataclass, field
from typing import Callable

from pluggybot.control import wrap_angle
from pluggybot.economy.census import Zone, count_objects, score, survey_route, true_count
from pluggybot.rack.coupling import HUB_STATION_YS
from pluggybot.tools.drawing import Board, Envelope, PenPlotter, board_standoff
from pluggybot.tools.strokes import StrokeProgram, from_cli

LCD_BAY = HUB_STATION_YS[0]
PEN_BAY = HUB_STATION_YS[2]      # bay C, as every pen demo has used


@dataclass
class Errand:
  """One fetch-use-stow job.

  `use` is called with the running `HubLifecycle` once the robot has arrived
  at `use_at` with the tool on the fork, and whatever dict it returns is
  reported alongside the swap verdicts. Returning nothing is fine -- that is
  the milestone-8 carry errand, which proves the swap and does no work.
  """

  name: str
  module: str
  station_y: float
  use_at: tuple[float, float]
  use: Callable[["object"], dict | None] | None = None
  #: free-form, for the demo scripts and the overseer's ledger
  detail: dict = field(default_factory=dict)
  #: which EVALUATOR judges this errand and which reward-table row pays for
  #: it (issue #14) -- "draw", "census", "dance", "carry". Defaulted off the
  #: name's prefix, since every errand here is already named `task:where`,
  #: and an errand whose task has no evaluator is simply never scored
  #: (economy/scoring.py: nothing awards points without one).
  task: str = ""
  #: the TASK this errand was built to discharge (issue #21), or "" for one
  #: nobody asked for. The errand still runs the same way -- the id is what
  #: lets the finished job's verdict close the offer that produced it, so
  #: that "the robot drew a house" and "the robot did the job it took on"
  #: are the same event rather than two.
  task_id: str = ""
  #: what this errand is expected to COST, Wh, or 0 to look it up in
  #: economy/energy.json by `task` (issue #15).
  #:
  #: Set only where a better figure exists than the per-action one: a TASK
  #: carries a per-KIND estimate that knows which end of the house it is
  #: being asked about, and the far whiteboard costs more than the near one.
  #: The mission loop refuses to start an errand it cannot pay for, so this
  #: is the number standing between an overseer and a robot that dies holding
  #: the tool -- MEASURED (scripts/energy_spike.py), never guessed.
  estimate_wh: float = 0.0
  #: does the use-phase need the pre-positioning drive to have ARRIVED?
  #:
  #: True for everything that works on a fixed thing at a fixed place -- a pen
  #: at a board, a claw at a block -- and issue #23 is why the question is
  #: asked at all: a use-phase run after a drive that gave up is a pen pressing
  #: at empty air, and the mission hung there until the battery died.
  #:
  #: ⚠ FALSE FOR AN ERRAND THAT DOES ITS OWN NAVIGATION, and the census is
  #: exactly that: `use_at` is the FIRST POINT OF THE SURVEY ROUTE the use
  #: phase then drives itself, so the pre-positioning drive is redundant by
  #: construction and gating on it throws away work the robot can still do.
  #: Measured: the home mission's drive stopped 1.96 m short, and the census
  #: went on to see 100 % of the garden and count 4 of 4 plants from there.
  #: One gate for every errand cost the recorded mission its census answer,
  #: which `tests/test_telemetry.py` guards and which is half of what the
  #: showcase recording exists to show.
  needs_use_pose: bool = True

  def __post_init__(self) -> None:
    if not self.task:
      self.task = self.name.split(":", 1)[0]


def carry_errand(module: str = "module_lcd", station_y: float = LCD_BAY,
                 use_at: tuple[float, float] = (-1.2, 2.5),
                 name: str = "carry") -> Errand:
  """The milestone-8 errand: fetch a tool, take it somewhere, bring it back.

  Kept as its own thing rather than deleted, and not for nostalgia: it is the
  only errand that exercises the swap stack with NOTHING else running, so when
  a drawing errand fails this is how you find out whether the pick broke or the
  pen did.
  """
  return Errand(name=name, module=module, station_y=station_y, use_at=use_at,
                task="carry")


def drawing_errand(book, board_name: str, board: Board,
                   program: StrokeProgram | None = None,
                   program_name: str = "house",
                   size: float | None = None, text: str | None = None,
                   erase: bool = True, station_y: float = PEN_BAY,
                   module: str = "module_pen", on_drawn=None,
                   task: str = "draw") -> Errand:
  """Fetch the pen, square up to a board, erase it, draw, restore the carry pose.

  `use_at` is the plotter's own board standoff, so the mission's A* takes the
  robot to the exact pose the fine approach expects and the fine approach is
  then a settle rather than a drive.

  ERASING IS PART OF THE ERRAND, by default. A task should not have to share a
  board with whatever was there before -- and it is the sim, not the browser,
  that decides a board is blank, because the sim is what owns board state.

  The figure is sized against the pen's ENVELOPE, never against the slab
  (`Envelope.for_board`): `targets_for` CLIPS, so an oversized figure draws
  flattened against the carriage's travel limit and reports a PERFECT trace,
  because the pen went exactly where it was told to go.
  """
  envelope = Envelope.for_board(board)
  figure = program if program is not None else from_cli(program_name, size, text)
  if not figure.fits(envelope):
    figure = figure.fitted(envelope)

  def use(life) -> dict:
    plotter = PenPlotter(life.model, life.data, life.mission.swap, board=board)
    # The stroke hook is wired HERE rather than in the plotter's constructor
    # signature at the call site, because it needs the errand's program name:
    # a bare polyline list has no name, and "which programs are on this board"
    # is the board state's whole point.
    #
    # ⚠ THE ERASE RIDES THE FIRST STROKE, not the arrival (issue #30's
    # validation run). The board book is the ink's ground truth and clearing
    # it used to be unconditional -- so a robot whose drift had put the
    # believed standoff somewhere else entirely narrated "erased
    # whiteboard_a", pressed at empty air ("never reached the board"), and
    # the book ended blank as if a wipe had happened at a board nobody
    # visited. The one truthful signal that the pen is AT the slab is a
    # press that found it, and the first stroke landing is exactly that --
    # so the wipe happens then, and a draw that never touched the board
    # leaves the old ink standing, which is what physically occurred.
    erased = False

    def on_stroke(i, points, name):
      nonlocal erased
      if erase and not erased:
        book.clear(board_name, t=float(life.data.time))
        life._say(f"USE_TOOL: erased {board_name}")
        erased = True
      book.stroke(board_name, name or figure.name, points,
                  t=float(life.data.time))

    plotter.on_stroke = on_stroke
    squared = plotter.drive_to_board()
    if not squared:
      # Not even at the BELIEVED standoff -- the drive stagnated or hit the
      # reflex. Skip the press entirely, restore the carry pose, and say so:
      # the evaluator finds no new ink and fails the job honestly.
      plotter.carry_config()
      life._say(f"USE_TOOL: never squared up to {board_name} -- "
                "skipping the drawing")
      return {"squared": False, "board": board_name, "figure": figure.name,
              "error": "never squared up to the board"}
    life._say(f"USE_TOOL: drawing {figure.name} on {board_name} "
              f"({len(figure.strokes)} strokes, {figure.ink_length:.2f} m of ink)")
    result = plotter.draw_program(figure)
    # `on_drawn(life, plotter, result)` fires with the robot STILL AT THE
    # BOARD, which is the only moment a photograph of the drawing is worth
    # taking -- scripts/home_draw.py's filmstrip hangs off this, and without
    # the seam the demo would have to re-implement the errand to get it.
    if on_drawn is not None:
      on_drawn(life, plotter, result)
    # Back to the carry pose BEFORE the stow drives anywhere: lift to carry
    # height and centre the carriage, or the module fouls its bay's brackets
    # on the way in (issue #10).
    plotter.carry_config()
    rec = book[board_name]
    if not result.get("drew"):
      # The reason, out loud: "drew None/None strokes" was the validation
      # run's whole account of a pen pressing at empty air.
      life._say(f"USE_TOOL: drew nothing -- "
                f"{result.get('reason', 'no reason recorded')}")
      return {"squared": squared, "board": board_name, "figure": figure.name,
              "fill": rec.fill, "plotter": plotter, **result}
    life._say(f"USE_TOOL: drew {result.get('strokes_drawn')}/"
              f"{result.get('strokes')} strokes, "
              f"{result.get('inked_fraction', 0):.0%} inked, "
              f"{board_name} now {rec.fill:.0%} full")
    return {"squared": squared, "board": board_name, "figure": figure.name,
            "fill": rec.fill, "plotter": plotter, **result}

  return Errand(name=f"draw:{board_name}", module=module,
                station_y=station_y, use_at=board_standoff(board), use=use,
                task=task,
                detail={"board": board_name, "figure": figure.name,
                        "strokes": len(figure.strokes),
                        "ink_m": figure.ink_length})


# ---- the LCD's errands (issue #13) -----------------------------------------
# Both leave the tool exactly as they found it: the LCD has no moving axis, so
# "carry configuration" costs nothing to restore -- which is the whole reason
# the display module is the right one to hang new task SHAPES off. Whatever
# these errands get wrong, it is never the stow.

#: sim seconds per slice of a dance move. Moves are driven in slices rather
#: than in one `_drive` call so the yaw can be accumulated as it turns: a
#: full revolution starts and ends at the same heading, and a before/after
#: comparison would score a perfect 360 as no movement at all.
DANCE_SLICE = 0.25

#: (label, face, hint, v m/s, w rad/s, seconds). A routine, not a random
#: walk -- the same sequence every time, so "did it dance" is a question with
#: an answer. Faces are the point: the motion is what a two-wheel robot can
#: do, the EXPRESSION is what makes it read as dancing rather than as a fault.
DANCE_ROUTINE = (
  ("wind-up",   "curious",    "blink",  0.00, -0.9, 1.2),
  ("spin",      "surprised",  "bounce", 0.00,  1.3, 5.0),
  ("shimmy-l",  "happy",      "shake",  0.00,  1.2, 0.9),
  ("shimmy-r",  "happy",      "shake",  0.00, -1.2, 0.9),
  ("shimmy-l2", "happy",      "shake",  0.00,  1.2, 0.9),
  ("shimmy-r2", "happy",      "shake",  0.00, -1.2, 0.9),
  ("approach",  "determined", "none",   0.22,  0.0, 1.5),
  ("retreat",   "determined", "none",  -0.22,  0.0, 1.5),
  ("finale",    "happy",      "bounce", 0.00, -1.3, 5.0),
)
# ⚠ A REVERSAL COSTS TWICE THE RAMP, and the shimmy is the move that finds
# out. `control.slew` ramps the wheels at 30 rad/s^2 like a real motor
# controller, so a turn that merely STARTS from rest loses ~0.12 s to the
# ramp -- while one that reverses an existing turn must cross the whole
# +/- range, and the first half of that crossing is still turning the wrong
# way. Measured back-to-back, as the errand actually drives it:
#
#     shimmy at 1.6 rad/s for 0.5 s   ->  0.35 of the commanded arc  MISSED
#     shimmy at 1.2 rad/s for 0.9 s   ->  0.63 of the commanded arc  ok
#
# The first version scored 6/9 in a real mission, and the three that missed
# were exactly the three reversals. The fix is the CHOREOGRAPHY, not the
# tolerance: a routine that asks for motion the drivetrain cannot deliver is
# a badly written routine, and loosening the scorer until it passes would
# throw away the only thing the score measures.

#: Coverage at which a survey stops looking: the robot has seen the zone, and
#: driving the rest of the route buys nothing. MEASURED, not guessed -- the
#: garden reads 100 %% from the first vantage point, and the full four-point
#: route spent the entire demo battery re-confirming it (the mission ended
#: BATTERY DEAD with a correct answer, which is not a working robot).
SURVEY_COVERED = 0.98
#: ...but never off a single look. One vantage that happens to see everything
#: is a lucky room, not a survey, and a zone with an occluding corner would
#: report high coverage from inside the part it can see.
SURVEY_MIN_VANTAGES = 2

#: Sim seconds an errand HOLDS its result on screen before driving away.
#:
#: Not decoration -- without it the result is never seen by anyone. Python
#: between two physics steps takes ZERO sim time, so a `show_count()` followed
#: by a return lands on the screen and is overwritten by the next state's
#: automatic face before a single 20 Hz frame is built. Measured on the
#: recorded showcase mission: the census's answer appeared in exactly none of
#: 10 850 frames, while every intermediate face appeared in dozens. A result
#: that is not on the wire did not happen.
PRESENT_S = 5.0

#: A move counts as landed if it went the commanded WAY and got at least this
#: much of the commanded distance. Not tighter, on purpose: the wheels ramp
#: (control.slew) and the parking brake has a stiction deadband, so a 5 s
#: spin cannot deliver its full commanded arc and never will.
DANCE_LANDED_FRAC = 0.5


def census_errand(zone: Zone, label: str = "plants",
                  module: str = "module_lcd", station_y: float = LCD_BAY,
                  entry: tuple[float, float] | None = None,
                  prefix: str = "plant", timeout: float = 90.0,
                  name: str | None = None) -> Errand:
  """Fetch the LCD, survey a zone, count what is in it, show the number.

  The first task with HIDDEN ground truth (design doc, "Evaluation"): the
  count comes off the robot's own occupancy grid, the answer comes out of the
  model, and they are compared by code. Being lazy is a real way to fail --
  drive half the zone and the number on the screen is honestly, confidently
  wrong.

  The screen is claimed for the whole errand and updated at every vantage
  point, so a viewer watches the tally climb rather than seeing one number
  appear at the end. The lifecycle takes the screen back at the next state
  change, which is the drive home.
  """
  points = survey_route(zone, entry=entry)

  def use(life) -> dict:
    screen = getattr(life, "screen", None)
    if screen is not None:
      screen.face("curious", "blink", hold=True)
    tally: dict = {"count": 0, "objects": [], "coverage": 0.0}
    stopped, vantages = "route", 0
    for i, (wx, wy) in enumerate(points, start=1):
      arrived = life.mission.drive_to(wx, wy, timeout=timeout)
      life.mission._spin()          # a vantage point is only worth the look
      vantages = i
      tally = count_objects(life.mission.grid, zone)
      if screen is not None:
        screen.show_count(tally["count"], label, face="determined", hint="none")
      life._say(f"USE_TOOL: vantage {i}/{len(points)}"
                f"{'' if arrived else ' (short)'} -- counting "
                f"{tally['count']} {label}, {tally['coverage']:.0%} of "
                f"{zone.name} seen")
      # Two reasons to stop early, and both are the errand being a good
      # citizen of the arbitration loop rather than a script that runs to the
      # end whatever it costs. Coverage first: there is nothing left to see.
      if i >= SURVEY_MIN_VANTAGES and tally["coverage"] >= SURVEY_COVERED:
        stopped = "covered"
        break
      # And the battery, which is the honest failure: the count that gets
      # reported is the count off however much of the zone was surveyed, and
      # `coverage` is what says so. A robot that dies in the garden holding a
      # perfect answer has not done the task.
      if life.needs_charge:
        stopped = "battery"
        life._say(f"USE_TOOL: breaking off the census at {vantages} vantage(s)"
                  f" -- battery down to {life.battery.fraction:.0%}")
        break

    # The evaluator. Deliberately after the last look and deliberately not
    # shown to the robot: `verdict["truth"]` never reaches the screen.
    truth = true_count(life.model, zone, prefix=prefix)
    verdict = score(tally["count"], truth, tally["coverage"])
    if screen is not None:
      screen.show_count(tally["count"], label,
                        face="happy" if verdict["correct"] else "surprised",
                        hint="bounce" if verdict["correct"] else "none")
    life._say(f"USE_TOOL: census of {zone.name} -- reported "
              f"{verdict['counted']} {label}, truth {verdict['truth']} "
              f"({'CORRECT' if verdict['correct'] else 'wrong'}), "
              f"{verdict['coverage']:.0%} of the zone surveyed")
    # Stand still and SHOW it. See PRESENT_S: the alternative is a number
    # that exists only inside this function.
    life.mission._drive(PRESENT_S, 0.0, 0.0)
    return {"census": verdict, "zone": zone.name, "label": label,
            "objects": tally["objects"], "vantages": vantages,
            "stopped": stopped}

  # `use_at` is `points[0]` -- the first vantage of the route `use` drives
  # itself -- so this errand does not need the pre-positioning drive to have
  # arrived, and must not be skipped when it falls short. See
  # `Errand.needs_use_pose`.
  return Errand(name=name or f"census:{zone.name}", module=module,
                station_y=station_y, use_at=points[0], use=use, task="census",
                needs_use_pose=False,
                detail={"zone": zone.name, "label": label,
                        "vantages": len(points)})


def dance_errand(at: tuple[float, float], module: str = "module_lcd",
                 station_y: float = LCD_BAY,
                 routine=DANCE_ROUTINE, name: str = "dance") -> Errand:
  """Fetch the LCD, drive somewhere visible, and perform a routine with faces.

  Auto-scored on the two things that can actually go wrong: a move that did
  not happen (the wheels stalled, the robot ran into something), and a
  routine that WANDERED -- a dance that ends three metres from where it
  started has driven off, whatever its expressions said.
  """

  def use(life) -> dict:
    screen = getattr(life, "screen", None)
    mission = life.mission
    x0, y0, _ = mission.pose
    landed = []
    for step, face, hint, v, w, seconds in routine:
      if screen is not None:
        screen.face(face, hint, hold=True)
      turned = 0.0
      travelled = 0.0
      slices = max(1, round(seconds / DANCE_SLICE))
      for _ in range(slices):
        px, py, pth = mission.pose
        mission._drive(seconds / slices, v, w)
        nx, ny, nth = mission.pose
        turned += wrap_angle(nth - pth)
        travelled += math.hypot(nx - px, ny - py)
      want_turn, want_travel = w * seconds, abs(v) * seconds
      ok = True
      if want_turn:
        ok = ok and turned * want_turn > 0 \
            and abs(turned) >= DANCE_LANDED_FRAC * abs(want_turn)
      if want_travel:
        ok = ok and travelled >= DANCE_LANDED_FRAC * want_travel
      landed.append({"step": step, "landed": bool(ok),
                     "turnedRad": round(turned, 3),
                     "travelledM": round(travelled, 3)})

    drift = math.hypot(mission.pose[0] - x0, mission.pose[1] - y0)
    done = sum(1 for m in landed if m["landed"])
    if screen is not None:
      screen.face("happy" if done == len(landed) else "worried", "bounce")
    life._say(f"USE_TOOL: danced {done}/{len(landed)} moves, "
              f"{drift:.2f} m of drift")
    mission._drive(PRESENT_S, 0.0, 0.0)     # hold the bow (see PRESENT_S)
    return {"dance": {"moves": len(landed), "landed": done,
                      "driftM": round(drift, 3),
                      "complete": done == len(landed)},
            "steps": landed}

  return Errand(name=name, module=module, station_y=station_y, use_at=at,
                use=use, task="dance", detail={"moves": len(routine)})
