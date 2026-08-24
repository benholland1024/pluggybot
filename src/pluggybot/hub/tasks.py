"""Tasks: jobs the world OFFERS, and what happens to them (issue #21).

The third pattern in this repo, and the one that gives the robot a reason.
The other two describe what the robot can DO; this one describes what it has
been ASKED to do:

  an ERRAND     is a tool, a place and a use-phase (hub/errand.py). Machinery.
  an ACTIVITY   is a mechanism watching contacts and owning discrete world
                state (docs/ActivityPattern.md). Scenery that reacts.
  a TASK        is a JOB OFFER: a description, a target, a reward, a deadline
                and -- once it is over -- a verdict. It comes from outside the
                robot, it can be declined by being left alone, and it can
                lapse without anyone doing anything.

An errand is HOW a task gets done; the task is WHY. One task resolves to one
errand today, and the split is what lets a visitor ask for something without
knowing that a whiteboard means "fetch the pen from bay C".

Four rules, and the first two are the ones worth breaking a build over.

  A TASK NEVER CARRIES ITS OWN PAYOUT. `Task.task` names a row of
  hub/rewards.json and an evaluator in hub/scoring.py; what the job is WORTH
  is looked up from that table every time it is asked for, and is never a
  field anybody can set. This is issue #14's rule arriving from a new
  direction: a visitor-created task that could name its own price would be a
  stranger on the internet moving a balance, and an LLM-proposed one would be
  the model paying itself. `Task.create` refuses a task with no evaluator, so
  the unscoreable task simply cannot be constructed.

  THE WIRE MAY CARRY ANYTHING A NETWORK COULD CARRY. IT MAY NOT CARRY
  ANYTHING A SENSOR WOULD HAVE TO DISCOVER (docs/TaskPattern.md, issue #24).
  A description is a work order and real robots receive those over WiFi; a
  surveyed board id is infrastructure, like the charging rack. The ANSWER to
  a task is neither. `Task.secret` is the slot that holds one: it is never in
  `as_dict`, never in `snapshot`, never in `as_context`, so it reaches the
  wire and the model through no path at all. `whiteboard_answer` (issue #22)
  is what fills it -- a question with a checkable answer -- and it is where
  the one exception lives: `as_state` writes it to the state FILE, because an
  offer that came back from a restart with no right answer behind it could
  never be graded. The file is not the wire; it sits in /var/lib/pluggybot
  beside the reward table, which also decides what things are worth.

  EXPIRY IS AN OUTCOME, NOT A DELETION. A task that lapses ends in `expired`
  and stays visible as such, because "nobody got round to it" is a true and
  interesting thing about a robot's day, and a marker that silently vanishes
  from the website reads as a bug.

  THE BOARD IS BOUNDED. Offers are capped and resolved tasks age out oldest
  first, on the same argument as the visitor inbox: an unbounded backlog a
  robot can never work through is a memory leak with a public endpoint on the
  end of it.

Timing policy -- how often tasks appear, how long an offer stands, the
per-target cooldown -- is deliberately NOT here. That is issue #23, and it is
configuration rather than constants. What this module owns is what a task IS,
what may happen to it, and how it survives a restart.
"""

import json
import os
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from pluggybot.hub.inbox import clean
from pluggybot.hub.questions import clean_answer
from pluggybot.hub.scoring import EVALUATORS, RewardTable, Verdict, default_table
from pluggybot.telemetry.protocol import ROBOT_ROOT, TASK_SOURCES, TASK_STATES

STATE_VERSION = 1

#: Longest visitor-facing description kept. A job offer is a sentence, and
#: this is `hub/inbox.py`'s cap for the same reason -- a task can be created
#: by a stranger (issue #23), so its text is untrusted on exactly the terms
#: a suggestion is.
MAX_DESCRIPTION = 280
#: Tasks the board holds at all. Resolved ones age out oldest-first; an OPEN
#: task is never dropped to make room, because dropping a job the robot might
#: still do is a different event from that job lapsing, and only one of the
#: two has an honest name on the wire.
MAX_TASKS = 40
#: ...and how many may stand OFFERED at once. A cap, not a cadence: issue #23
#: owns how fast they arrive.
MAX_OFFERED = 6

#: The states a task moves through, in order. `offered` is a job nobody has
#: taken; `claimed` is one a robot has accepted but not started (it is queued
#: behind whatever the robot is doing); `active` is one being worked on right
#: now. The three terminal states differ in a way the site draws differently:
#: `done` was finished and judged good, `failed` was finished and judged bad,
#: `expired` was never attempted at all.
OPEN_STATES = ("offered", "claimed", "active")
TERMINAL_STATES = ("done", "failed", "expired")


def _now() -> str:
  return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# ---- the kinds (data) --------------------------------------------------------
# A KIND is what a task is about; `task` is which evaluator judges it and
# which reward-table row pays for it. The two are separate because several
# kinds can share one evaluator -- "draw a house on whiteboard_a" and (issue
# #22) "draw the answer to 2 + 3 on whiteboard_a" are both scored on ink, and
# only the second one needs an answer to be right.
#
# `estimateWh` is what makes a task refusable on ENERGY grounds (see
# `Task.claimable`). It is a declared per-kind figure, not a model -- M10's
# per-errand energy work is what replaces it -- but the figures below are
# MEASURED, off the committed recordings, from the battery at SWAP_PICK to
# the battery at the end of SWAP_RETURN:
#
#     room_hub  carry (module_lcd, across the room)      0.487 - 0.570 Wh
#     home      draw  (pen, erase, house, stow)          0.929 Wh
#     home      census (LCD, survey the garden, stow)    0.866 Wh
#
# ⚠ READ THOSE AGAINST THE PACK. room_hub's demo cell is 0.700 Wh and home's
# is 1.100 Wh, and a charge stops at 90 %. So ONE ERRAND IS ROUGHLY ONE FULL
# PACK, in both worlds, and there is no margin to be had -- which is why a
# guess here was not good enough. The first version of this table guessed
# 0.35 Wh for a drawing (measured: 0.929) and the home fixture recorded a
# robot that claimed the job at 88 % and died mid-stroke with nothing inked
# and the pen still on the fork. That failure is the entire argument for the
# gate, and it was the gate's own numbers that let it through.
#
# Rounded UP to the nearest 0.01 Wh and no further: inflating them for safety
# would make every job permanently unclaimable, because the headroom being
# padded against does not exist.


@dataclass(frozen=True)
class TaskKind:
  """One kind of job the world knows how to offer."""

  name: str
  #: the hub/scoring.py evaluator and hub/rewards.json row
  task: str
  #: what `target` names: a board, a zone, or a tool module
  target_kind: str
  #: sentence template; `{target}` and any `params` key may appear in it
  template: str
  estimate_wh: float = 0.2
  #: this job has a RIGHT ANSWER, and whoever takes it has to supply one
  #: (issue #22). It is the difference between a job a body can do and a job
  #: that needs a mind: `TaskBoard.claim` refuses one without an answer, so
  #: the scripted rotation leaves a question standing and it lapses honestly
  #: rather than being attempted by something that cannot think.
  needs_answer: bool = False

  def describe(self, target: str, params: dict) -> str:
    try:
      return self.template.format(target=target, **params)
    except KeyError:
      # A kind whose template names a parameter this task does not carry.
      # Degraded rather than raised on: a description is what a person reads,
      # and a task with an awkward sentence is better than no task.
      return self.template.replace("{target}", target)


KINDS: dict[str, TaskKind] = {
  "draw_figure": TaskKind(
    "draw_figure", task="draw", target_kind="board",
    template="Draw a {program} on {target}.", estimate_wh=0.93),
  "rate_artwork": TaskKind(
    "rate_artwork", task="artwork", target_kind="board",
    template="Draw a {program} on {target} for people to rate.",
    estimate_wh=0.93),
  "count_plants": TaskKind(
    "count_plants", task="census", target_kind="zone",
    template="Survey {target} and put the number of plants on your face.",
    estimate_wh=0.87),
  "whiteboard_answer": TaskKind(
    "whiteboard_answer", task="answer", target_kind="board",
    # ⚠ NO PRICE IN THE SENTENCE. The issue sketched "Worth 2 PluggyPoints.
    # Draw the answer to..." and the number is deliberately not here: a
    # description is written once and frozen, `reward` is looked up from
    # hub/rewards.json on every read, and a job that quoted its own price in
    # prose would go stale the first time the table was re-tuned -- with the
    # stale figure being the half a person reads. The wire carries both, side
    # by side, and only one of them is derived.
    template="Draw the answer to this question on {target}: {question}",
    estimate_wh=0.93, needs_answer=True),
  "fetch_module": TaskKind(
    "fetch_module", task="carry", target_kind="module",
    template="Fetch {target}, carry it across the room and hang it back up.",
    estimate_wh=0.57),
}


def kind_names() -> list[str]:
  """The vocabulary this producer can offer, for the telemetry header.

  A two-repo contract on the same terms as FACE_STATES and VISUAL_HINTS:
  ADDING a kind is additive (the website draws a generic marker for one it
  does not know -- rooftop-media-2026 #77), renaming one breaks both repos.
  """
  return list(KINDS)


# ---- one task ----------------------------------------------------------------


@dataclass(frozen=True)
class Task:
  """One job offer. Immutable; the board replaces it on every transition.

  Frozen on purpose. A task moves through states in one place (`TaskBoard`),
  which is also the place that emits the event and saves the file, so there
  is no way to change a task's state without the wire and the disk hearing
  about it.
  """

  id: str
  kind: str
  target: str
  description: str
  #: which EVALUATOR judges it and which reward-table row pays -- derived from
  #: the kind, never supplied, and never a points figure
  task: str
  params: dict = field(default_factory=dict)
  #: SIM seconds after which an untaken offer lapses. None means it stands
  #: until the board is full. Absolute, not a duration: the board is asked
  #: "is it past this yet" by a loop that already knows the sim clock.
  deadline: float | None = None
  estimate_wh: float = 0.2
  source: str = "system"
  state: str = "offered"
  created_t: float = 0.0
  claimed_t: float | None = None
  resolved_t: float | None = None
  claimed_by: str = ""
  #: the evaluator's verdict, as `Verdict.as_dict()` -- already redacted
  verdict: dict | None = None
  points: int = 0
  #: WHAT THE ROBOT SAID THE ANSWER IS (issue #22), set when the job is
  #: claimed and never afterwards. It comes from whatever is deciding for the
  #: robot; code never computes it, and the errand that goes and draws it
  #: never sees `secret`. That split is what makes "did it get the question
  #: right" a question with an honest answer: the commitment is frozen before
  #: a wheel turns, so nothing downstream can revise it once the ink is down.
  answer: str = ""
  #: NEVER PUBLISHED. See the module docstring: the answer to a task is not
  #: sensor data, but it is not a work order either, and there is no path
  #: from here to the wire or to the model's context.
  secret: dict = field(default_factory=dict, repr=False)

  @classmethod
  def create(cls, kind: str, target: str, task_id: str,
             params: dict | None = None, description: str = "",
             deadline: float | None = None, source: str = "system",
             t: float = 0.0, secret: dict | None = None) -> "Task":
    """Build an offered task, or raise.

    Refuses three things, and each refusal is a rule from the module
    docstring made structural rather than promised: an unknown kind, a source
    outside the vocabulary, and -- the important one -- a kind whose `task`
    has no evaluator in hub/scoring.py. There is no way to construct a task
    that could be finished and then have to be paid for by guesswork.
    """
    spec = KINDS.get(kind)
    if spec is None:
      raise ValueError(f"unknown task kind {kind!r} "
                       f"(have: {', '.join(sorted(KINDS))})")
    if source not in TASK_SOURCES:
      raise ValueError(f"unknown task source {source!r} "
                       f"(have: {', '.join(TASK_SOURCES)})")
    if spec.task not in EVALUATORS:
      raise ValueError(
        f"task kind {kind!r} is scored as {spec.task!r}, which has no "
        "evaluator in hub/scoring.py -- nothing may be offered that cannot "
        "then be judged by code (issue #14)")
    if not target:
      raise ValueError(f"task kind {kind!r} needs a {spec.target_kind} target")
    params = {str(k): v for k, v in (params or {}).items()}
    text = clean(description, MAX_DESCRIPTION) or spec.describe(target, params)
    return cls(id=task_id, kind=kind, target=target,
               description=clean(text, MAX_DESCRIPTION), task=spec.task,
               params=params, deadline=deadline,
               estimate_wh=spec.estimate_wh, source=source, state="offered",
               created_t=round(float(t), 3), secret=dict(secret or {}))

  # ---- what it is worth (looked up, never stored) --------------------------

  @property
  def target_kind(self) -> str:
    return KINDS[self.kind].target_kind

  @property
  def needs_answer(self) -> bool:
    return KINDS[self.kind].needs_answer if self.kind in KINDS else False

  @property
  def open(self) -> bool:
    return self.state in OPEN_STATES

  def reward(self, table: RewardTable | None = None) -> dict:
    """What the table says this pays, right now.

    Derived on every read rather than stored, which is the whole of the
    first rule: re-tuning hub/rewards.json re-prices every offered task, and
    nothing that can write a Task can write a payout.
    """
    row = (table if table is not None else default_table())[self.task]
    return {"task": row.task, "tier": row.tier, "base": row.base,
            "bonus": row.bonus}

  def overdue(self, now: float) -> bool:
    return self.deadline is not None and float(now) >= self.deadline

  def claimable(self, now: float, pack_wh: float | None = None) -> bool:
    """May a robot take this on, right now?

    Two gates. It has to be on offer and not already lapsed -- and it has to
    FIT: a task whose estimated cost exceeds THE REMAINING PACK is not
    claimable, it is a way to die holding a tool (issue #21).

    ⚠ `pack_wh` is the WHOLE remaining charge, not the part above the
    reserve, and that distinction is the difference between a working gate
    and a gate that refuses everything forever. The reserve is a RETURN-TRIP
    margin: an errand is allowed to spend into it, which is exactly how the
    mission loop has always worked (`needs_charge` is checked BETWEEN
    errands, never during one). Measured, the energy above the reserve is
    0.28 Wh in room_hub and 0.44 Wh in home, while the cheapest real errand
    costs 0.487 Wh -- so comparing against that would make every job in every
    world permanently unclaimable, which reads exactly like a task system
    that does not work.

    ⚠ `estimate_wh` is a per-kind figure measured off the recordings, not a
    model of THIS errand from HERE. M10's per-errand energy work replaces it.
    """
    if self.state != "offered" or self.overdue(now):
      return False
    if pack_wh is None:
      return True
    return float(pack_wh) >= self.estimate_wh

  # ---- serialisation -------------------------------------------------------

  def as_dict(self) -> dict:
    """The full public task, as the wire and the state file carry it.

    `secret` is absent, and its absence is the point -- see the module
    docstring.
    """
    out = {
      "id": self.id, "kind": self.kind, "task": self.task,
      "target": self.target, "targetKind": self.target_kind,
      "description": self.description, "params": dict(self.params),
      "state": self.state, "source": self.source,
      "deadline": self.deadline, "estimateWh": round(self.estimate_wh, 4),
      "createdT": self.created_t, "claimedT": self.claimed_t,
      "resolvedT": self.resolved_t, "claimedBy": self.claimed_by,
      "points": self.points,
    }
    if self.verdict is not None:
      out["verdict"] = dict(self.verdict)
    return out

  def as_state(self) -> dict:
    """The task as the STATE FILE carries it: `as_dict` plus the two halves
    that never go on the wire.

    The file is not the wire, and this is the line that says so. A question's
    answer lives in /var/lib/pluggybot beside the board book and the points
    ledger -- the same trust domain as hub/rewards.json, which also decides
    what things are worth and is also not published. Leaving it out instead
    would mean a restart brought back an offer that could never be graded:
    a job standing on the board with no right answer behind it, which is a
    worse kind of secret-keeping than writing it down.
    """
    return {**self.as_dict(), "answer": self.answer, "secret": dict(self.secret)}

  def snapshot(self, table: RewardTable | None = None) -> dict:
    """The task as a telemetry frame carries it: `as_dict` plus the payout
    the table currently says it is worth."""
    return {**self.as_dict(), "reward": self.reward(table)}

  def as_context(self, now: float, pack_wh: float | None = None,
                 table: RewardTable | None = None) -> dict:
    """How the overseer is shown an offer (issue #15).

    Compact: the model is choosing whether to take a job, so it gets what the
    job is, what it pays, whether it can be taken and how long it has. It
    does not get `secret`, and it does not get the verdict of a task it has
    not done.
    """
    reward = self.reward(table)
    return {"id": self.id, "kind": self.kind, "target": self.target,
            "description": self.description, "state": self.state,
            "from": self.source, "pays": reward["base"] + reward["bonus"],
            "tier": reward["tier"], "estimateWh": round(self.estimate_wh, 3),
            # ...and whether taking it means answering something. The model
            # is told, rather than left to infer it from the sentence: an
            # `answer` it forgets to fill in is a claim that gets refused,
            # and a refusal costs a whole decision.
            "needsAnswer": self.needs_answer,
            "claimable": self.claimable(now, pack_wh),
            "expiresInS": (None if self.deadline is None
                           else round(self.deadline - float(now), 1))}

  @classmethod
  def from_json(cls, spec: dict) -> "Task":
    kind = str(spec["kind"])
    known = KINDS.get(kind)
    return cls(
      id=str(spec["id"]), kind=kind, target=str(spec.get("target", "")),
      description=str(spec.get("description", "")),
      # Re-derived from the kind rather than read back, so a hand-edited
      # state file cannot re-point a task at a better-paying evaluator.
      task=known.task if known is not None else str(spec.get("task", "")),
      params=dict(spec.get("params") or {}),
      deadline=(None if spec.get("deadline") is None
                else float(spec["deadline"])),
      estimate_wh=(known.estimate_wh if known is not None
                   else float(spec.get("estimateWh") or 0.2)),
      source=str(spec.get("source", "system")),
      state=str(spec.get("state", "offered")),
      created_t=float(spec.get("createdT") or 0.0),
      claimed_t=(None if spec.get("claimedT") is None
                 else float(spec["claimedT"])),
      resolved_t=(None if spec.get("resolvedT") is None
                  else float(spec["resolvedT"])),
      claimed_by=str(spec.get("claimedBy", "")),
      verdict=(dict(spec["verdict"]) if spec.get("verdict") else None),
      points=int(spec.get("points") or 0),
      answer=str(spec.get("answer", "")),
      # Absent from anything that came off the wire, and that is fine: only
      # `as_state` writes it, and only the state file is ever read back.
      secret=dict(spec.get("secret") or {}),
    )


# ---- the board ---------------------------------------------------------------


class TaskBoard:
  """Every task this world knows about, and the only thing that moves one.

  The telemetry surface is the duck type an `ActivitySet`, a `BoardBook`, a
  `ScreenSet` and a `Ledger` present (`names` + `snapshot()`) -- but see
  `FrameBuilder`: the `tasks` block is shipped WHOLE rather than per-key
  diffed, because a task can cease to exist and a per-key diff has no way to
  say "gone".

  `on_event` receives complete protocol messages (`task_offered`,
  `task_claimed`, `task_resolved`) as they happen; wire the publisher and the
  recorder into it, exactly as for the boards and the ledger.
  """

  def __init__(self, path: str | os.PathLike | None = None,
               table: RewardTable | None = None,
               max_tasks: int = MAX_TASKS, max_offered: int = MAX_OFFERED,
               clock: Callable[[], str] = _now) -> None:
    self.table = table if table is not None else default_table()
    self.path = Path(path) if path is not None else None
    self.clock = clock
    self.max_tasks = max_tasks
    self.max_offered = max_offered
    self.on_event: list[Callable[[dict], None]] = []
    self.tasks: dict[str, Task] = {}
    self.seq = 0
    self.dropped = 0
    if self.path is not None and self.path.exists():
      self.load()

  # ---- reading -------------------------------------------------------------

  def __len__(self) -> int:
    return len(self.tasks)

  def __contains__(self, task_id: str) -> bool:
    return task_id in self.tasks

  def __getitem__(self, task_id: str) -> Task:
    return self.tasks[task_id]

  def get(self, task_id: str) -> Task | None:
    return self.tasks.get(task_id)

  @property
  def names(self) -> list[str]:
    return list(self.tasks)

  @property
  def kinds(self) -> list[str]:
    """The vocabulary this board can offer, for the telemetry header.

    Read off the board rather than imported by the frame builder, so
    `telemetry` never has to import `hub` -- the layering only runs one way
    (protocol.py says so about the vocabularies themselves).
    """
    return kind_names()

  def open_tasks(self) -> list[Task]:
    return [t for t in self.tasks.values() if t.open]

  def offered(self) -> list[Task]:
    return [t for t in self.tasks.values() if t.state == "offered"]

  def claimable(self, now: float, pack_wh: float | None = None) -> list[Task]:
    """Offered, unexpired and affordable, oldest first.

    Oldest first, and deterministically so: a scripted policy has to be
    reproducible or a mission test that exercises it is a different test
    every run.
    """
    ready = [t for t in self.offered() if t.claimable(now, pack_wh)]
    return sorted(ready, key=lambda t: (t.created_t, t.id))

  def snapshot(self) -> dict:
    return {tid: task.snapshot(self.table) for tid, task in self.tasks.items()}

  def context(self, now: float, pack_wh: float | None = None,
              limit: int = 5) -> list[dict]:
    """The offers the overseer is shown, oldest first and capped.

    Capped for the reason the visitor queue is: the robot takes at most one
    per turn, and a wall of offers is input tokens spent on jobs it will not
    reach.
    """
    return [t.as_context(now, pack_wh, self.table)
            for t in self.claimable(now, pack_wh)[:limit]]

  def stats(self) -> dict:
    counts = {state: 0 for state in TASK_STATES}
    for task in self.tasks.values():
      counts[task.state] = counts.get(task.state, 0) + 1
    return {"total": self.seq, "held": len(self.tasks), "dropped": self.dropped,
            **counts}

  # ---- offering ------------------------------------------------------------

  def next_id(self) -> str:
    self.seq += 1
    return f"t_{self.seq:04d}"

  def offer(self, kind: str, target: str, params: dict | None = None,
            description: str = "", ttl: float | None = None,
            source: str = "system", t: float = 0.0,
            secret: dict | None = None) -> Task | None:
    """Put a job into the world. `None` if there is no room for it.

    `ttl` is sim SECONDS the offer stands for, turned into an absolute
    deadline here so nothing downstream has to know when it was made. How
    long a ttl should be, and how often to call this at all, is issue #23.
    """
    if len(self.offered()) >= self.max_offered:
      return None
    task = Task.create(kind=kind, target=target, task_id=self.next_id(),
                       params=params, description=description,
                       deadline=None if ttl is None else round(float(t) + ttl, 3),
                       source=source, t=t, secret=secret)
    self.tasks[task.id] = task
    self._trim()
    self._emit({"type": "task_offered", "t": round(float(t), 3),
                "task": task.snapshot(self.table)})
    return task

  # ---- transitions ---------------------------------------------------------

  def claim(self, task_id: str, robot: str = ROBOT_ROOT, t: float = 0.0,
            pack_wh: float | None = None, answer: str = "") -> Task | None:
    """Take a job on. `None` if it is gone, taken, lapsed or unaffordable.

    None rather than an exception: the caller is a mission loop acting on an
    LLM's answer or on a race with the expiry sweep, and "that one is not
    available" is an ordinary outcome of asking, not a fault.

    ⚠ A job that NEEDS AN ANSWER cannot be claimed without one (issue #22),
    and the refusal is here rather than in the caller because it is a fact
    about the job. `answer` is sanitised on the way in: it is the one string
    a model chooses that ends up drawn on a wall, so it is capped and
    filtered to a fixed alphabet exactly as a visitor's message is, and an
    answer that survives as "" is no answer.
    """
    task = self.tasks.get(task_id)
    if task is None or not task.claimable(t, pack_wh):
      return None
    said = clean_answer(answer) if task.needs_answer else ""
    if task.needs_answer and not said:
      return None
    return self._move(replace(task, state="claimed", claimed_by=robot,
                              claimed_t=round(float(t), 3), answer=said),
                      "task_claimed", t)

  def start(self, task_id: str, t: float = 0.0) -> Task | None:
    """Mark a claimed task as being worked on right now."""
    task = self.tasks.get(task_id)
    if task is None or task.state != "claimed":
      return None
    return self._move(replace(task, state="active"), "task_claimed", t)

  def resolve(self, task_id: str, verdict: Verdict, t: float = 0.0) -> Task | None:
    """Close a task with an EVALUATOR's verdict, and nothing else.

    The type check is the same lock `Ledger.award` carries, for the same
    reason and against the same attack: a task that could be closed with a
    verdict-shaped dict is a task that can declare itself done. The points
    recorded here are the verdict's, which the ledger has already re-derived
    from the reward table -- this is a copy for display, and the balance it
    came from is the ledger's.
    """
    if not isinstance(verdict, Verdict):
      raise TypeError(
        f"resolve() takes a scoring.Verdict, got {type(verdict).__name__} -- "
        "a task is closed by a deterministic evaluator, never by whatever "
        "ran it reporting on itself (issue #14)")
    task = self.tasks.get(task_id)
    if task is None or not task.open:
      return None
    return self._move(replace(task, state="done" if verdict.ok else "failed",
                              verdict=verdict.as_dict(), points=verdict.points,
                              resolved_t=round(float(t), 3)),
                      "task_resolved", t)

  def expire_due(self, t: float) -> list[Task]:
    """Lapse every OFFERED task past its deadline. Returns what lapsed.

    Only offered ones. A task the robot has already taken is its problem
    now: abandoning a job mid-errand would leave a module on the fork, and
    the deadline is about how long an OFFER stands, not about interrupting
    work in progress.
    """
    gone = []
    for task in list(self.tasks.values()):
      if task.state == "offered" and task.overdue(t):
        moved = self._move(replace(task, state="expired",
                                   resolved_t=round(float(t), 3)),
                           "task_resolved", t)
        if moved is not None:
          gone.append(moved)
    return gone

  def _move(self, task: Task, kind: str, t: float) -> Task:
    self.tasks[task.id] = task
    msg = {"type": kind, "t": round(float(t), 3), "id": task.id,
           "state": task.state, "robot": task.claimed_by or ROBOT_ROOT}
    if kind == "task_resolved":
      msg.update({"points": task.points, "verdict": task.verdict,
                  "task": task.snapshot(self.table)})
    self._emit(msg)
    return task

  # ---- bookkeeping ---------------------------------------------------------

  def _trim(self) -> None:
    """Age out resolved tasks, oldest first. Open ones are never dropped.

    A dropped task is not an expired one and does not get an event: expiry
    is something that HAPPENED to a job in the world, while this is the
    board forgetting a job that is already over. Counted, so a truncated
    board says so.
    """
    while len(self.tasks) > self.max_tasks:
      closed = [t for t in self.tasks.values() if not t.open]
      if not closed:
        return
      oldest = min(closed, key=lambda t: (t.resolved_t or 0.0, t.id))
      del self.tasks[oldest.id]
      self.dropped += 1

  def _emit(self, msg: dict) -> None:
    for hook in self.on_event:
      hook(dict(msg))
    # Saved on every transition, like the boards and the ledger and for the
    # same reason: an offer that survives only a CLEAN shutdown does not
    # survive the thing restarts are usually about.
    self.save()

  # ---- persistence ---------------------------------------------------------

  def save(self, path: str | os.PathLike | None = None) -> Path | None:
    target = Path(path) if path is not None else self.path
    if target is None:
      return None
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(
      {"version": STATE_VERSION, "seq": self.seq, "dropped": self.dropped,
       "tasks": [t.as_state() for t in self.tasks.values()]}, indent=1) + "\n")
    # Rename over the target: a crash mid-write leaves the previous board
    # intact rather than a truncated file that loads as an empty world.
    os.replace(tmp, target)
    return target

  def load(self, path: str | os.PathLike | None = None) -> "TaskBoard":
    """Restore the board. Older state versions load, newer ones refuse --
    the same asymmetry as the boards and the ledger, and for the same
    reason: /var/lib/pluggybot outlives the image.

    ⚠ A task that was ACTIVE when the process died comes back `failed`, not
    `active`. The robot that was doing it no longer exists, and a job nobody
    is working on but which reads as in-progress is a marker that never
    resolves. `expired` would be a lie in the other direction -- the offer
    was taken.
    """
    target = Path(path) if path is not None else self.path
    if target is None or not target.exists():
      return self
    doc = json.loads(target.read_text())
    version = int(doc.get("version", 0))
    if not 1 <= version <= STATE_VERSION:
      raise ValueError(
        f"{target}: task state version {doc.get('version')!r}, expected "
        f"1..{STATE_VERSION} -- delete the file to start with no tasks")
    self.seq = int(doc.get("seq", 0))
    self.dropped = int(doc.get("dropped", 0))
    self.tasks = {}
    for spec in doc.get("tasks", ()):
      try:
        task = Task.from_json(spec)
      except (KeyError, ValueError):
        # A task written by a build that knew a kind this one does not.
        # Skipped rather than fatal: the file is world state shared across an
        # upgrade, and one unreadable row must not cost the whole board.
        self.dropped += 1
        continue
      if task.state == "active":
        task = replace(task, state="failed",
                       verdict={"task": task.task, "ok": False, "points": 0,
                                "reason": "interrupted by a restart"})
      self.tasks[task.id] = task
    self.seq = max(self.seq, len(self.tasks))
    return self
