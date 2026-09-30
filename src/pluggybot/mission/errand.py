"""Errands: a tool, a place, and something to DO there (issue #12).

An errand is one job in the lifecycle's QUEUE: a program over the step
vocabulary (issue #58, `procedure/steps.py`) that fetches, goes, works and
stows -- the mouse's plate pressed, a procedure the robot wrote -- or, in the
shape the rover's native errands had (`rover-final`), a tool fetched from
its bay, a drive to `use_at`, a `use` callable there, and the tool hung
back. The overseer (issue #15) gets a menu it can choose from without
knowing any physics: an errand is a name, a tool and a destination.

The one rule an errand's `use` must respect: LEAVE THE TOOL IN ITS CARRY
CONFIGURATION. A tool axis parked where the work left it fouls the bay's
trays on the stow (docs/SimNotes.md, "The pen would not stow").
"""

from dataclasses import dataclass, field
from typing import Callable

from pluggybot.rack.coupling import STATION_YS


@dataclass
class Errand:
  """One fetch-use-stow job.

  `use` is called with the running `HubLifecycle` once the robot has arrived
  at `use_at` with the tool on the fork, and whatever dict it returns is
  reported alongside the swap verdicts; returning nothing is fine. A
  use-phase is a ROUTINE (pluggybot/tick.py, issue #58): a generator
  yielding one command per physics step and returning its dict, ticked from
  the lifecycle's own loop. A COMPOSED errand (`program`) replaces all of it.
  """

  name: str
  module: str
  station_y: float
  use_at: tuple[float, float]
  use: Callable[["object"], dict | None] | None = None
  #: free-form, for the demo scripts and the overseer's ledger
  detail: dict = field(default_factory=dict)
  #: which EVALUATOR judges this errand and which reward-table row pays for
  #: it (issue #14) -- "feed", "program", "care". Defaulted off the
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
  #: ⚠ FALSE FOR AN ERRAND THAT DOES ITS OWN NAVIGATION -- a program, or a
  #: survey whose `use_at` is only its route's first point: gating on the
  #: pre-positioning drive would throw away work the robot can still do
  #: (the rover's census, measured 1.96 m short and still 4 of 4 plants).
  needs_use_pose: bool = True
  #: a COMPOSED errand (issue #58, `procedure/steps.py`): the whole job --
  #: fetching and stowing included -- as validated steps ticked from the
  #: loop, instead of the fixed fetch -> `use` -> stow around a callable.
  #: `module`/`station_y` then name the FIRST tool the program fetches (or
  #: nothing), for the bookkeeping that reads them; `use` and `use_at` are
  #: unused. Abort still means stow: the loop hangs back whatever the program
  #: left on the fork.
  program: object | None = None
  #: WHICH ROLE of a multi-role program this robot plays (issue #167), or
  #: "" for a program with one.
  role: str = ""

  def __post_init__(self) -> None:
    if not self.task:
      self.task = self.name.split(":", 1)[0]


def programmed_errand(program, task: str = "program",
                      name: str | None = None, role: str = "",
                      rack: dict | None = None) -> Errand:
  """An errand whose middle AND ends are a program's steps (issue #58).

  `rack` is a lifecycle's inventory (module -> bay index into STATION_YS)
  once the workshop has hung a tool (issue #168); without it, the shipped
  five. A built tool's index is past the five, on the rail (issue #277).

  `task` names the evaluator that grades the finished job -- "program" for
  the generic per-step verdict, or an existing kind's evaluator ("draw")
  when the program discharges that kind's task, in which case the sampler
  reads the same board it reads for the native errand (`detail["board"]`).
  """
  from pluggybot.procedure.steps import TOOL_BAYS
  steps = program.steps(role) if role else program.steps()
  first_tool = (program.first("fetch", "tool", role) if role
                else program.first("fetch", "tool")) or ""
  board = program.first("draw", "board", role) if role else program.first("draw", "board")
  figure = program.first("draw", "figure", role) if role else program.first("draw", "figure")
  detail = {"program": program.name, "steps": len(steps),
            **({"role": role} if role else {})}
  if board is not None:
    detail.update({"board": board, "figure": figure})
  return Errand(name=name or f"{task}:{program.name}", module=first_tool,
                station_y=(STATION_YS[(rack or TOOL_BAYS)[first_tool]]
                           if first_tool else 0.0),
                use_at=(0.0, 0.0), use=None, task=task, program=program,
                role=role, needs_use_pose=False, detail=detail)
