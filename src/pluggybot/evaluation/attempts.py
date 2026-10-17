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
              where it starts with the tool its set-out gives it, the chest
              shut and caught, its catch applied every step
  attempt     ONE TRY from the set-out: the skill run inside its budget, the
              call, the hold, one verdict through `scoring.evaluate` -- and
              the record's row
  held_out    chests nobody has flown, drawn by a seed fixed after the build
              is frozen (`seed_of`)

THE PROTOCOL:
  - an attempt is one try from the declared start; the skill's own retries
    inside it (the probe's walk-ins, `legs.probe.TRIES`) are not attempts;
  - between attempts the WORLD resets the chest and the robot to their
    set-out -- each attempt is a fresh build of it -- and every reset is help
    the world gave, counted on its row (`help.reset`);
  - the call is wherever the skill's routine ended, or its budget did
    (`cut`); the hold follows, the body held still;
  - what the skill says of itself is its own test (`own`), beside the
    verdict and never read by it.
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

A skill is any `skill(world) -> routine` that drives `world.body` through
its `Body` members and returns its own word (`{"ok": ..., "why": ...}`):

    row = attempt(Task(chest=k, body="humanoid", mount="counter"), skill, n,
                  info={"process": "try", "skill": "mine", "tokens": 1234})

What it flatters is said with every result (#493, Evaluation.md §5 "Graded
in MuJoCo"): the knob's tag is a fiducial; the chests are drawn to suit the
probe; one geometry; the quadruped's claw is put on by the set-out and the
humanoid is stood at its counter; MuJoCo against MuJoCo.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import time

import mujoco
import numpy as np

from pluggybot.activity import chest as ch
from pluggybot.body import body_for
from pluggybot.challenge import lid
from pluggybot.economy import scoring

#: The set-outs: which body before which mount, each the world its
#: placements were measured for -- the quadruped before a chest on the floor
#: (`evaluation.probe`, #466's stage 2) and the humanoid at a counter
#: (`evaluation.counter`, #507). Another pair is refused: a placement is
#: measured for the body that reaches from it.
SET_OUTS = (("quadruped", "floor"), ("humanoid", "counter"))
#: The tool each body's set-out gives it: the quadruped's claw, put on by
#: the set-out (`Body.place_tool_routine`) where a real run fetches it from
#: the rack; the humanoid's hands are its own.
TOOLS = {"quadruped": "module_claw", "humanoid": None}
#: The graders a task may name: each has its own way of being watched.
GRADERS = ("lid",)
#: Every set-out ends standing still this long, s: the catch pulls only
#: once a step has run (`activity.chest.Chest.sense`), and the humanoid's
#: set-out steps nothing before it.
SETTLE_S = 0.5
#: An attempt's sim-time budget, s: the probe's own patience
#: (`legs.probe.PATIENCE_S`), which its 128 set-outs flew inside at 85-90 s.
BUDGET_S = 300.0

#: The chests flown in public before any held-out draw: #466's batches flew
#: set-outs 0-127 on the floor (#479, #480) and #507's demo table 0-11 at
#: the counter. Chest k is the same lid on either mount (`chest.draw(k)`).
PUBLIC = range(0, 128)
#: ...and held-out chests are drawn from here up, far past anything flown,
#: so no batch run to tune a build lands on one by accident.
HELD_OUT_FROM = 1_000_000
HELD_OUT_SPAN = 1_000_000_000


def seed_of(commit: str) -> int:
  """The held-out seed of a FROZEN build: off its commit, so it cannot be
  known before the build exists, and nobody chooses it. ⚠ A builder could
  still commit until a seed looked kind; the record names the commit, and
  the result says it is weaker than a sealed pool."""
  if not commit or commit == "unknown":
    raise ValueError("a held-out seed is a frozen build's: no commit, no seed")
  return int(hashlib.sha256(f"pluggybot held-out {commit}".encode()).hexdigest()[:8], 16)


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
  body: str = "quadruped"
  mount: str = "floor"
  grader: str = "lid"

  def __post_init__(self) -> None:
    if int(self.chest) < 0:
      raise ValueError(f"a chest is a whole number from 0, not {self.chest}")
    if (self.body, self.mount) not in SET_OUTS:
      raise ValueError(f"no set-out for the {self.body} at a chest on the {self.mount}: "
                       f"there are {', '.join(f'{b} at the {m}' for b, m in SET_OUTS)}")
    if self.grader not in GRADERS:
      raise ValueError(f"no grader {self.grader!r} for an attempt: {', '.join(GRADERS)}")

  @property
  def tool(self) -> str | None:
    """The tool the set-out gives the body, or None."""
    return TOOLS[self.body]

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
  chest: ch.Chest
  start: tuple

  def close(self) -> None:
    self.body.close()


def set_out(task: Task, policy=None) -> World:
  """The task's declared start, built afresh: the chest shut where its
  set-out stands it, the body where it starts and with its set-out's tool.
  `policy` is the humanoid's standing policy, where a caller has one."""
  if task.body == "quadruped":
    from pluggybot.evaluation import probe as ep
    from pluggybot.home import world as home
    so = ep.set_out(task.chest)
    model = ep.spec_of(so).compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    body = body_for(model, data, realtime=False, grid_bounds=home.GRID_BOUNDS)
    body.start_at(*so.start)
  else:
    from pluggybot.evaluation import counter as ct
    so = ct.set_out(task.chest)
    model, data, body = ct.world(so, policy)
  chest = ch.Chest(model, data, so.lid)
  body.step_hooks.append(lambda: chest.sense(model, data))
  if task.tool is not None:
    body.run(body.place_tool_routine(task.tool))
    if not body.tool_powered(task.tool):
      body.close()
      raise RuntimeError(f"the set-out could not seat {task.tool}")
  body.run(body.hold_routine(SETTLE_S))
  return World(task=task, model=model, data=data, body=body, chest=chest,
               start=tuple(so.start))


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


def attempt_in(world: World, skill, n: int = 1, *, budget_s: float = BUDGET_S,
               hold_s: float = lid.HOLD_S, info: dict | None = None) -> dict:
  """One attempt in a world already set out (`attempt` builds it): the
  skill (`skill(world)` -> a routine, returning its own word) run inside
  `budget_s` while the grader watches every step, the call, the hold, and
  the verdict -- the record's row."""
  info = dict(info or {})
  task, body, data = world.task, world.body, world.data
  watch = lid.Watch(world.model, data, body)
  body.step_hooks.append(watch)
  t0, wall = float(data.time), time.time()
  cut: list = []
  try:
    said = body.run(_within(data, skill(world), t0 + budget_s, cut))
    t_call = float(data.time)
    watch.hold()
    at_call = lid.measure(world.model, data)
    body.run(body.hold_routine(hold_s))
  finally:
    body.step_hooks.remove(watch)
  measured = scoring.SAMPLERS[task.grader](
    world, None, said, {"at_call": at_call, "watch": watch, "tool": task.tool,
                        "catchN": world.chest.lid.catch_n})
  verdict = scoring.evaluate(task.grader, measured)
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
    "run": {k: info[k] for k in ("commit", "seed") if k in info} | {
      "budgetS": budget_s, "holdS": hold_s},
  }


def attempt(task: Task, skill, n: int = 1, *, budget_s: float = BUDGET_S,
            hold_s: float = lid.HOLD_S, info: dict | None = None, build=set_out) -> dict:
  """One attempt (the protocol): the task's set-out built afresh -- the
  world's reset, counted before every attempt but the first -- the skill
  tried from it, graded, and the world closed: the record's row. `build`
  is how a set-out is made (a test's own world)."""
  world = build(task)
  try:
    return attempt_in(world, skill, n, budget_s=budget_s, hold_s=hold_s, info=info)
  finally:
    world.close()
