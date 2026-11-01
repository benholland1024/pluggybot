"""The attempt protocol and the experience record (issue #493, stage 1): how
a skill is tried, graded by the world and written down -- the harness #465's
demos fly their baselines and their flights through (Evaluation.md §4).
Built as a KIT another builder could run: body-agnostic through `Body`,
each task data beside the code that grades it, and nothing here reading a
body's internals.

  Task        what is attempted, as data: the grader that judges it
              (`challenge/lid.py`), the chest (`chest.draw(k)`), where it
              stands (`mount`) and which body tries it
  set_out     the task's declared start, BUILT AFRESH: the world, the body
              where it starts, the chest shut and caught, its catch applied
              every step
  attempt     ONE TRY from the set-out: the skill run inside its budget, the
              call, the hold, one verdict through `scoring.evaluate` -- and
              the record's row
  held_out    chests nobody has flown, drawn by a seed fixed after the build
              is frozen (`seed_of`)

THE PROTOCOL:
  - an attempt is one try from the declared start; the skill's own retries
    inside it (the reference opener's grips, `opener.GRIPS`) are not
    attempts;
  - between attempts the WORLD resets the chest and the robot to their
    set-out -- each attempt is a fresh build of it -- and every reset is help
    the world gave, counted on its row (`help.reset`);
  - the call is wherever the skill's routine ended, or its budget did
    (`cut`); the hold follows, the body held still;
  - what the skill says of itself is its own test (`own`), beside the
    verdict and never read by it;
  - an attempt that RAISES, or whose world MuJoCo found unstable and reset,
    is no attempt: it raises, and the record says what (`error`), never a
    verdict.
  ⚠ AN ATTEMPT IS ITS SET-OUT EXACTLY, sensor noise and all: code that
  learns nothing between attempts gets the same verdict at every one, so a
  reference flies one.

THE RECORD, one row per attempt: the task, the attempt's number, the
process that produced the skill and the skill itself, the experience it took
(sim-seconds, procedure steps, LLM tokens, imagination CPU-seconds), the help
it had, its own test, and the verdict -- its secret metrics apart, under
`secret`. Evaluation.md §3's per-task record is read off the rows
(`qualities.per_task_record`). ⚠ Absent is None, never 0: a code skill's
tokens are a measured 0, a procedure's steps it never counted are None.

A skill is any `skill(body) -> routine` that drives the body through its
`Body` members and returns its own word (`{"ok": ..., "why": ...}`). It is
handed the BODY, never the world: the chest's truth stays the harness's.
`info` is read after the attempt, so what the caller counts during it (a
procedure's steps, the tokens its mind spent) reaches the row:

    row = attempt(Task(chest=k, body="humanoid", mount="counter"), skill, n,
                  info={"process": "try", "skill": "mine", "tokens": 1234})

What it flatters is said with every reading (`FLATTERS`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import time

import mujoco
import numpy as np

from pluggybot.activity import chest as ch
from pluggybot.challenge import door, drawer, lid, place
from pluggybot.economy import scoring

#: The set-outs: which body before which mount, each the world its
#: placements were measured for -- the humanoid at a counter
#: (`evaluation.counter`, #507). Another pair is refused: a placement is
#: measured for the body that reaches from it (the quadruped's before a
#: chest on the floor went with it, #559).
SET_OUTS = (("humanoid", "counter"),)
#: MuJoCo's counters of a world it found unstable and reset: a 10 g part on
#: a spring of 40 N*m/rad was reset 124 times in one rollout, every reading
#: finite (the review of #473).
UNSTABLE = (mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL,
            mujoco.mjtWarning.mjWARN_BADQACC)
#: The graders a task may name: the lid's, and the hands' capability
#: checks' (#525; docs/Hands.md), each watched every step and measured at
#: the call in its own way (`WATCHED`). A check is the humanoid's at its
#: counter; its set-out k is `chest` (the lid's name for it).
GRADERS = ("lid", "drawer", "door", "place")
WATCHED = {"lid": (lid.Watch, lid.measure), "drawer": (drawer.Watch, drawer.measure),
           "door": (door.Watch, door.measure), "place": (place.Watch, place.measure)}
CHECKS = tuple(g for g in GRADERS if g != "lid")
#: Every set-out ends standing still this long, s: the catch pulls only
#: once a step has run (`activity.chest.Chest.sense`), and the humanoid's
#: set-out steps nothing before it.
SETTLE_S = 0.5
#: An attempt's sim-time budget, s: the job's own time limit, where the
#: harness makes the call at the latest (`lid.TIME_LIMIT_S`)...
BUDGET_S = lid.TIME_LIMIT_S
#: ...each set-out's own (`Task.limit_s`): the task's data beside the code
#: that grades it, handed to the grader as `limitS`. The humanoid's is three
#: times its reference's slowest pass (Ben, 2026-10-17: `evaluation.opener`,
#: 17.7-22.5 s over its 32 public chests at 0f98e24), set before #493's
#: stage 3 flew.
REFERENCE_S = 22.54
LIMIT_S = {("humanoid", "counter"): 3 * REFERENCE_S}
#: ...and a job's own where it has one: the door's three times OUR door
#: procedure's slowest pass (#535: `evaluation.hands_check.DOOR`, 10.8-35.8
#: s over its 32 public set-outs at 70bf773, 20 passing), set before any
#: mind flew it, as the lid's was.
DOOR_REFERENCE_S = 35.81
JOB_LIMIT_S = {"door": 3 * DOOR_REFERENCE_S}
#: ...a WATCHED tag's pictures are sharp (#554's stage 2). MEASURED: in our
#: lid procedure the knob's tag crossed the picture at 270-440 px/s in a
#: reach or a lift (the 90th percentile at three set-outs), 2.7-4.4 px in a
#: 10 ms exposure on a tag 33 px across at 0.6 m, a cell of it 4 px.
WATCH_FLATTERS = ("with `watch`, a tag is decoded off sharp renders: moving, it crossed the "
                  "picture at up to 270-440 px/s, a smear of 3-4 px in a 10 ms exposure "
                  "that a real camera's picture has and the render does not")
#: What every result here relies on that a real world would not give it
#: (#493; Evaluation.md §5, "Graded in MuJoCo"): said with every reading.
FLATTERS = (
  "the knob carries a tag: finding it is reading a fiducial",
  "the chests are drawn to suit the probe: each shuts harder than its spring, friction "
  "and damping hold it over the probe's sweeps, and its catch stops at 4 N",
  "one geometry: the chests differ only in their hidden dynamics",
  "the humanoid is stood at its counter: walking to the chest is not the task",
  "the humanoid's mind is told the knob's size and its tag: what it would see",
  "with `measure`, a flat top's height and edges are our code's reading of its depth camera",
  WATCH_FLATTERS,
  "resets are free and exact, and nothing breaks",
  "MuJoCo against MuJoCo: an upper bound on hardware",
)
#: ...and what a reading of the cupboard's door flatters (#535).
DOOR_FLATTERS = (
  "the lever's tip carries a tag: finding it is reading a fiducial",
  "one cupboard: the doors differ only in where they stand and their hidden dynamics",
  "a freed door is kicked a few degrees ajar by its latch's spring (`activity.door.KICK_RAD`)",
  "the humanoid is stood at its counter: walking to the cupboard is not the task",
  "the humanoid's mind is told the tip's size and its tag: what it would see",
  "with `measure`, a flat top's height and edges are our code's reading of its depth camera",
  WATCH_FLATTERS,
  "resets are free and exact, and nothing breaks",
  "MuJoCo against MuJoCo: an upper bound on hardware",
)


def flatters(grader: str = "lid") -> tuple[str, ...]:
  """What a reading of `grader`'s job flatters: the door's own, else the
  lid's (`FLATTERS`)."""
  return DOOR_FLATTERS if grader == "door" else FLATTERS


#: The chests flown in public before any held-out draw: #466's batches flew
#: set-outs 0-127 on the floor (#479, #480) and #507's demo table 0-11 at
#: the counter. Chest k is the same lid on either mount (`chest.draw(k)`).
PUBLIC = range(0, 128)
#: ...and held-out chests are drawn from here up, far past anything flown,
#: so no batch run to tune a build lands on one by accident.
HELD_OUT_FROM = 1_000_000
HELD_OUT_SPAN = 1_000_000_000


#: A commit is named by this many hex digits, whatever `git` abbreviates to.
COMMIT_DIGITS = 7


def commit_of(sha: str) -> str:
  """A commit's name for the record and the seed: its first
  `COMMIT_DIGITS` hex digits, however long the sha it was handed."""
  name = (sha or "").strip().lower()
  if len(name) < COMMIT_DIGITS or any(c not in "0123456789abcdef" for c in name):
    raise ValueError(f"no commit in {sha!r}: a frozen build is named by its sha")
  return name[:COMMIT_DIGITS]


def seed_of(commit: str) -> int:
  """The held-out seed of a FROZEN build: off its commit (`commit_of`), so
  it cannot be known before the build exists, and nobody chooses it. ⚠ A
  builder could still commit until a seed looked kind; the record names
  the commit, and the result says it is weaker than a sealed pool."""
  name = commit_of(commit)
  return int(hashlib.sha256(f"pluggybot held-out {name}".encode()).hexdigest()[:8], 16)


def held_out(seed: int, n: int) -> list[int]:
  """`n` chests nobody has flown, by `seed`: distinct, from `HELD_OUT_FROM`
  up, in the order drawn."""
  rng = np.random.default_rng(int(seed))
  out: list[int] = []
  while len(out) < n:
    k = HELD_OUT_FROM + int(rng.integers(HELD_OUT_SPAN))
    if k not in out:
      out.append(k)
  return out


@dataclass(frozen=True)
class Task:
  """One task, as data (the kit's unit): `chest` names the chest and its
  set-out (`chest.draw(k)`), `body` what tries it, `mount` where the chest
  stands and `grader` the challenge that judges it."""

  chest: int
  body: str = "humanoid"
  mount: str = "counter"
  grader: str = "lid"

  def __post_init__(self) -> None:
    if int(self.chest) < 0:
      raise ValueError(f"a chest is a whole number from 0, not {self.chest}")
    if (self.body, self.mount) not in SET_OUTS:
      raise ValueError(f"no set-out for the {self.body} at a chest on the {self.mount}: "
                       f"there are {', '.join(f'{b} at the {m}' for b, m in SET_OUTS)}")
    if self.grader not in GRADERS:
      raise ValueError(f"no grader {self.grader!r} for an attempt: {', '.join(GRADERS)}")
    if self.grader in CHECKS and (self.body, self.mount) != ("humanoid", "counter"):
      raise ValueError(f"the {self.grader} is the humanoid's at its counter, not the "
                       f"{self.body}'s at the {self.mount}")

  @property
  def limit_s(self) -> float:
    """The job's time limit at this set-out, s: its own where it has one
    (`JOB_LIMIT_S`), else its set-out's (`LIMIT_S`)."""
    return JOB_LIMIT_S.get(self.grader, LIMIT_S[(self.body, self.mount)])

  @property
  def held_out(self) -> bool:
    return self.chest >= HELD_OUT_FROM

  def as_dict(self) -> dict:
    return asdict(self)

  @classmethod
  def from_dict(cls, d: dict) -> "Task":
    return cls(**{k: d[k] for k in ("chest", "body", "mount", "grader") if k in d})


@dataclass
class World:
  """A task's set-out, built: the model, its data, the body, the chest's
  activity (`activity.chest.Chest`: its catch applied every step; `lid` is
  the truth), and where the body started (x, y, heading)."""

  task: Task
  model: object
  data: object
  body: object
  chest: ch.Chest | None
  start: tuple
  #: a check's own activity (#525: the drawer's), where the job has one
  activity: object = None

  def close(self) -> None:
    self.body.close()


def set_out(task: Task, policy=None) -> World:
  """The task's declared start, built afresh: the chest shut where its
  set-out stands it and the body where it starts. `policy` is the
  humanoid's standing policy, where a caller has one."""
  if task.grader in CHECKS:
    return _check_set_out(task, policy)
  from pluggybot.evaluation import counter as ct
  so = ct.set_out(task.chest)
  model, data, body = ct.world(so, policy)
  chest = ch.Chest(model, data, so.lid)
  body.step_hooks.append(lambda: chest.sense(model, data))
  body.run(body.hold_routine(SETTLE_S))
  return World(task=task, model=model, data=data, body=body, chest=chest,
               start=tuple(so.start))


def _check_set_out(task: Task, policy=None) -> World:
  """A capability check's set-out (#525): its prop on the counter
  (`evaluation.counter`), the humanoid at its start, its activity sensing
  every step, standing still `SETTLE_S`."""
  from pluggybot.evaluation import counter as ct
  if task.grader == "drawer":
    from pluggybot.activity import drawer as dr
    so = ct.drawer_set_out(task.chest)
    model, data, body = ct.world(so, policy)
    act = dr.Drawer(model, data, so.runners)
  elif task.grader == "door":
    from pluggybot.activity import door as dd
    so = ct.door_set_out(task.chest)
    model, data, body = ct.world(so, policy)
    act = dd.Door(model, data, so.hinge)
  elif task.grader == "place":
    from pluggybot.activity import cube as cb
    so = ct.cube_set_out(task.chest)
    model, data, body = ct.world(so, policy)
    act = cb.Cube(model, data, so.load, ct.COUNTER_H + ct.BOARD_H)
  body.step_hooks.append(lambda: act.sense(model, data))
  body.run(body.hold_routine(SETTLE_S))
  return World(task=task, model=model, data=data, body=body, chest=None,
               start=tuple(so.start), activity=act)


def own_word(said) -> dict:
  """What a skill said of itself, off what its routine returned: `ok` and
  `why` as a dict's keys or an object's attributes (the probe's `Probed`),
  or nothing said (`ok` None)."""
  if said is None:
    return {"ok": None, "why": ""}
  get = said.get if isinstance(said, dict) else (lambda k, d=None: getattr(said, k, d))
  ok = get("ok", None)
  return {"ok": None if ok is None else bool(ok), "why": str(get("why", "") or "")}


def _within(data, routine, until: float, cut: list):
  """`routine` driven until it returns or the sim clock reaches `until`,
  then closed (its `finally` runs) and `cut` marked: what it returned, or
  None. An exception from the step is thrown into it, as `tick.run` does."""
  try:
    command = next(routine)
    while True:
      if float(data.time) >= until:
        routine.close()
        cut.append(True)
        return None
      try:
        yield command
      except BaseException as e:  # noqa: BLE001 -- thrown into the skill
        command = routine.throw(e)
        continue
      command = routine.send(None)
  except StopIteration as done:
    return done.value


class Unstable(RuntimeError):
  """An attempt whose world MuJoCo found unstable and reset: its readings
  are not the attempt's, and it has no verdict."""


def attempt_in(world: World, skill, n: int = 1, *, budget_s: float | None = None,
               hold_s: float = lid.HOLD_S, info: dict | None = None,
               sealed: list | None = None) -> dict:
  """One attempt in a world already set out (`attempt` builds it): the
  skill (`skill(body)` -> a routine, returning its own word) run inside
  `budget_s` while the grader watches every step, the call, the hold, and
  the verdict -- the record's row. `info` is read once the hold is over;
  the sealed verdict itself (`scoring.Verdict`) is appended to `sealed`,
  what a library admits a procedure on (`Library.admit`). Raises what the
  skill raised, or `Unstable`."""
  info = info if info is not None else {}
  task, body, data = world.task, world.body, world.data
  budget_s = task.limit_s if budget_s is None else float(budget_s)
  warned = {w: int(data.warning[w].number) for w in UNSTABLE}
  watching, snapshot = WATCHED[task.grader]
  watch = watching(world.model, data, body)
  body.step_hooks.append(watch)
  t0, wall = float(data.time), time.time()
  cut: list = []
  try:
    said = body.run(_within(data, skill(body), t0 + budget_s, cut))
    t_call = float(data.time)
    watch.hold()
    at_call = snapshot(world.model, data)
    body.run(body.hold_routine(hold_s))
  finally:
    body.step_hooks.remove(watch)
  reset = [w.name for w in UNSTABLE if data.warning[w].number > warned[w]]
  if reset:
    raise Unstable(f"MuJoCo found the world unstable and reset it ({', '.join(reset)})")
  measured = scoring.SAMPLERS[task.grader](
    world, None, said, {"at_call": at_call, "watch": watch, "started": t0,
                        "catchN": world.chest.lid.catch_n if world.chest is not None else None,
                        "limitS": budget_s})
  # A cut attempt that failed says so first (`evaluate(failed=)`): what it
  # left half done follows.
  verdict = scoring.evaluate(task.grader, measured,
                             failed=f"out of time ({budget_s:.0f} s)" if cut else "")
  if sealed is not None:
    sealed.append(verdict)
  return {
    "task": task.as_dict(), "attempt": int(n), "heldOut": task.held_out,
    "process": info.get("process"), "skill": info.get("skill"),
    "simS": round(t_call - t0, 3), "cut": bool(cut),
    "holdS": round(float(data.time) - t_call, 3),
    "steps": info.get("steps"), "tokens": info.get("tokens"),
    "imaginationCpuS": info.get("imaginationCpuS"),
    "help": {"reset": n > 1, "demonstrations": int(info.get("demonstrations", 0)),
             "interventions": int(info.get("interventions", 0))},
    "own": own_word(said),
    "verdict": {"ok": verdict.ok, "reason": verdict.reason,
                "metrics": verdict.public_metrics()},
    "secret": {k: verdict.metrics.get(k) for k in verdict.secret},
    "wallS": round(time.time() - wall, 1),
    "run": {k: info[k] for k in ("commit", "dirty", "seed") if k in info} | {
      "budgetS": budget_s, "holdS": hold_s},
  }


def attempt(task: Task, skill, n: int = 1, *, budget_s: float | None = None,
            hold_s: float = lid.HOLD_S, info: dict | None = None, build=set_out,
            sealed: list | None = None) -> dict:
  """One attempt (the protocol): the task's set-out built afresh -- the
  world's reset, counted before every attempt but the first -- the skill
  tried from it, graded, and the world closed: the record's row. `build`
  is how a set-out is made (a test's own world)."""
  world = build(task)
  try:
    return attempt_in(world, skill, n, budget_s=budget_s, hold_s=hold_s, info=info,
                      sealed=sealed)
  finally:
    world.close()
