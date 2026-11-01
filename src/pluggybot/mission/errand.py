"""Errands: something to DO, as one job in the lifecycle's QUEUE (issue #12).

An errand is a program over the step vocabulary (issue #58,
`procedure/steps.py`) -- a procedure the robot wrote (#166), or one a task
carries -- run by the lifecycle with one verdict per step and one verdict
for the whole. The overseer (issue #15) gets a menu it can choose from
without knowing any physics. The quadrupeds' errands were a tool fetched
from its bay, a walk, a use and the tool hung back (`quadruped-final`).
"""

from dataclasses import dataclass, field


@dataclass
class Errand:
  """One job: its program, and what judges and pays for it."""

  name: str
  #: free-form, for the demo scripts and the overseer's ledger
  detail: dict = field(default_factory=dict)
  #: which EVALUATOR judges this errand and which reward-table row pays for
  #: it (issue #14) -- "program". Defaulted off the name's prefix, since
  #: every errand here is already named `task:where`, and an errand whose
  #: task has no evaluator is simply never scored (economy/scoring.py:
  #: nothing awards points without one).
  task: str = ""
  #: the TASK this errand was built to discharge (issue #21), or "" for one
  #: nobody asked for. The errand still runs the same way -- the id is what
  #: lets the finished job's verdict close the offer that produced it.
  task_id: str = ""
  #: what this errand is expected to COST, Wh, or 0 to look it up in
  #: economy/energy.json by `task` (issue #15). Set only where a better
  #: figure exists than the per-action one: a TASK carries a per-KIND
  #: estimate. The mission loop refuses to start an errand it cannot pay
  #: for -- MEASURED (scripts/energy_spike.py), never guessed.
  estimate_wh: float = 0.0
  #: the program over the step vocabulary (issue #58, `procedure/steps.py`)
  #: or the procedure (#166, `procedure/lang.py`) that is the whole job,
  #: validated and ticked from the loop.
  program: object | None = None

  def __post_init__(self) -> None:
    if not self.task:
      self.task = self.name.split(":", 1)[0]


def programmed_errand(program, task: str = "program",
                      name: str | None = None) -> Errand:
  """An errand that is a program's steps (issue #58). `task` names the
  evaluator that grades the finished job -- "program" for the generic
  per-step verdict, or an existing kind's evaluator when the program
  discharges that kind's task."""
  steps = program.steps()
  return Errand(name=name or f"{task}:{program.name}", task=task, program=program,
                detail={"program": program.name, "steps": len(steps)})
