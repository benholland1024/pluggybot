"""The physics seam: a ROUTINE yields one drive command per physics step
(issue #58, the tick refactor).

Every manoeuvre in this repo used to be a `while` loop that stepped the
physics itself -- the primitive owned the clock, and nothing
above its frame ran until it returned. That is why an errand could not be
composed, why #116 had to hand-place its abort points inside those loops,
and why a second robot had nowhere to be ticked. A routine turns the loop
inside out without rewriting it: where the loop stepped the physics it now
YIELDS the command it would have stepped with, and the driver above does the
stepping. The code reads as the same loop, and the order of every side
effect around the step is unchanged -- which is what makes the scripted day
the same trajectory before and after (`scripts/determinism_spike.py
--compare`).

  Command   the BODY's, and read by nothing else (issue #380, `body.py`):
            the quadruped's is (vx, vy, w), the velocity its walking policy
            tracks.
  Routine   a generator of Commands whose RETURN value is the manoeuvre's
            result (`go_to_routine` returns whether it arrived). Composed
            with `yield from`, exactly as the blocking calls were composed
            with `return`.
  Step      one routine, ticked from outside: `tick()` hands back the next
            command or None when the routine has returned.
  Stepper   what drives one body's physics: `apply(command)` writes its
            setpoints before the world steps, `after_step()` is its
            bookkeeping after, `step(command)` is the three for a robot
            alone, `STILL` the command that holds it where it is, and
            `model` / `data` the world it steps (`Body.stepper`); and,
            where it has one, `held(steps)`, told how many steps another
            robot's blocking run held it (`_Loop.run`).
  run       the blocking driver, for scripts, tests and every caller that
            wants the old shape: step until the routine returns.

Two rules the seam carries:

  ⚠ A ROUTINE CALL IS NOTHING UNTIL IT IS DRIVEN. `self.drive_to_routine(x, y)`
  builds a generator and moves nothing; without `yield from` (or `run`) it is
  a truthy object that silently did not happen. `tests/test_tick.py` walks
  the syntax tree for exactly that mistake.

  ⚠ AN EXCEPTION FROM THE STEP IS THROWN INTO THE ROUTINE, not raised past
  it. `stop_when` ends a run by raising `MissionAborted` from a step hook,
  and the manoeuvre in flight may hold a `finally` that restores the physics
  timestep or clears the press flag. Raised in the driver those blocks would
  wait for garbage collection; thrown in, they run where they always ran.

  ⚠ A BLOCKING `run` INSIDE `run_many` STEPS EVERY ROBOT OF THE LOOP (#690):
  the others held through their waits, their routines where they were; what
  that run raises that is not the routine's own is raised by `run_many`, as
  it raises its own (`_Raised`). None is begun in `src/` since #692 (the
  interrupt's ask yields): the rule is for the next.
"""

from typing import Any, Generator

#: A body's own command (`body.py`): the rover's is (v, w).
Command = tuple
Routine = Generator[Command, None, Any]


class MissionAborted(RuntimeError):
  """Stop the run: raised from a step hook (`HubLifecycle.stop_when`, a
  closed viewer) and thrown into every routine in flight."""


def hold(seconds: float, timestep: float, v: float = 0.0,
         w: float = 0.0) -> Routine:
  """Drive a constant command for `seconds` of sim time, `_drive`'s shape."""
  for _ in range(round(seconds / timestep)):
    yield v, w


def once(v: float = 0.0, w: float = 0.0) -> Routine:
  """One physics step at a command."""
  yield v, w


def result(value: Any) -> Routine:
  """A routine that steps nothing and returns `value` -- what a test stubs a
  drive with (`life.body.go_to_routine = lambda *a, **kw:
  tick.result(True)`)."""
  return value
  yield  # unreachable; it is what makes this a generator


class Step:
  """One routine, driven one physics step at a time.

  `tick()` returns the command for the NEXT step, or None once the routine
  has returned -- `result` then holds what it returned and `done` is True.
  `tick(exc)` throws `exc` into the routine instead of resuming it, which is
  how a driver reports that the step it was handed raised.
  """

  def __init__(self, routine: Routine, name: str = "") -> None:
    self.routine = routine
    self.name = name
    self.done = False
    self.result: Any = None

  def tick(self, exc: BaseException | None = None) -> Command | None:
    if self.done:
      return None
    try:
      if exc is not None:
        return self.routine.throw(exc)
      return self.routine.send(None)
    except StopIteration as stop:
      self.done, self.result = True, stop.value
      return None


def run_many(pairs, name: str = "", step=None, waits=None) -> list:
  """Drive several robots' routines from ONE physics loop (issue #167).

  `pairs` is `[(stepper, routine), ...]`, one per robot. Each step: every
  robot's command is applied (`apply`), the world steps ONCE, every robot's
  bookkeeping runs (`after_step`) -- the three things `step` does for one
  robot, in the same order, so a robot alone here is the robot alone
  there. A robot whose routine has returned waits for the others: driven by
  its WAIT where `waits` gives it one -- a callable handing back a routine,
  begun again whenever it returns -- else held at `STILL`. The loop ends
  when every robot's routine has returned, and a wait is closed where it
  stands. ⚠ Held at `STILL`, a robot gets its stepper's bookkeeping and
  nothing its routines do between steps: a humanoid's scans are matched
  there, and one held so for most of an hour lost its heading (#593).

  An exception from the step -- a hook's `MissionAborted` -- is thrown into
  EVERY live routine, so each one's cleanup runs (the swap timestep, the
  press flag), and then re-raised: one robot's stop is the day's stop. A
  routine's own exception, a wait's too, is raised as it is: thrown into
  the others, a waiting robot's sensing error read as their failure.
  Returns each routine's result, in order.
  """
  steps = [Step(r, f"{name}:{i}") for i, (_, r) in enumerate(pairs)]
  steppers = [sw for sw, _ in pairs]
  waits = list(waits) if waits is not None else [None] * len(steps)
  waiting: list[Step | None] = [None] * len(steps)
  if step is None:
    import mujoco

    def step():
      mujoco.mj_step(steppers[0].model, steppers[0].data)

  loop = _Loop(steppers, step, waits, name)
  _LOOPS.append(loop)
  cmds: list[Command | None] = [None] * len(steps)
  try:
    for i in range(len(steps)):
      cmds[i] = _ticked(steps, cmds, i)
    while any(c is not None for c in cmds):
      loop.release()                        # ...its own routine drives each robot again
      now = [c if c is not None else _held(steppers, waits, waiting, i, name, "wait")
             for i, c in enumerate(cmds)]
      exc = None
      try:
        _step_world(steppers, now, step)
      except BaseException as e:  # noqa: BLE001 -- re-raised inside every routine
        exc = e
      if exc is not None:
        _thrown(steps, cmds, exc)
        continue
      for i in range(len(steps)):
        if cmds[i] is not None:
          cmds[i] = _ticked(steps, cmds, i)
  except _Raised as r:                      # a held robot's own, from wherever it came
    raise r.exc from None
  finally:
    _LOOPS.remove(loop)
    loop.release()
    for w in waiting:
      if w is not None:
        w.routine.close()
  return [st.result for st in steps]


def _step_world(steppers: list, now: list, step) -> None:
  """One step of the world: every robot's command applied in the loop's
  order, the world stepped once, every robot booked."""
  for sw, c in zip(steppers, now):
    sw.apply(c)
  step()
  for sw in steppers:
    sw.after_step()


def _thrown(steps: list, cmds: list, exc: BaseException,
            first: BaseException | None = None) -> None:
  """`exc` thrown into every live routine (`cmds[i]` not None), each one's
  next command kept; then the first any of them raised -- or `first` --
  raised: one robot's stop is the day's stop."""
  for i, st in enumerate(steps):
    if cmds[i] is None:
      continue
    try:
      cmds[i] = st.tick(exc)
    except _Raised as r:
      first, cmds[i] = first or r.exc, None
    except BaseException as e:  # noqa: BLE001
      first, cmds[i] = first or e, None
  if first is not None:
    raise first


def _ticked(steps: list, cmds: list, i: int) -> Command | None:
  """Routine `i`'s next command -- and what a blocking run inside it
  carried out (`_Raised`) raised as `run_many` raises its own: a held
  robot's wait's as it is, the step's thrown into every other live routine
  first."""
  try:
    return steps[i].tick()
  except _Raised as r:
    cmds[i] = None
    if r.step:
      _thrown(steps, cmds, r.exc, first=r.exc)
    raise r.exc from None


def _held(steppers, waits, holding, i: int, name: str, kind: str) -> Command:
  """Robot `i`'s command while its own routine does not drive it: its WAIT's
  -- begun again whenever it returns, kept in `holding[i]` -- or `STILL`
  where it has none."""
  if waits[i] is None:
    return steppers[i].STILL
  cmd = holding[i].tick() if holding[i] is not None else None
  if cmd is None:
    holding[i] = Step(waits[i](), f"{name}:{i}:{kind}")
    cmd = holding[i].tick()
  return steppers[i].STILL if cmd is None else cmd


#: Every `run_many` driving now, innermost last (#690): where a blocking
#: `run` begun inside one of its routines steps the world.
_LOOPS: list["_Loop"] = []


class _Raised(BaseException):
  """What a blocking run inside a loop raised that is not its routine's own
  -- a held robot's wait's (`step` False) or the step's -- carried past the
  routine's callers, whose `except Exception` would book another robot's
  error as its own, to `run_many`, which raises it there."""

  def __init__(self, exc: BaseException, step: bool) -> None:
    super().__init__(exc)
    self.exc, self.step = exc, step


class _Loop:
  """A `run_many` in flight: its steppers, its world's step, its robots'
  waits -- and the holds a blocking run inside it holds the other robots
  through, kept from one such run to the next (an interrupt's ask is one a
  think slice) until the loop's own next step."""

  def __init__(self, steppers: list, step, waits: list, name: str) -> None:
    self.steppers, self.step, self.waits, self.name = steppers, step, waits, name
    self.holding: list[Step | None] = [None] * len(steppers)

  def release(self) -> None:
    """The holds closed where they stand: each robot's own routine, or its
    wait, drives it again."""
    for i, h in enumerate(self.holding):
      if h is not None:
        h.routine.close()
        self.holding[i] = None

  def run(self, stepper, routine: Routine, name: str) -> Any:
    """`routine` driven to its end for `stepper`'s robot, every step the
    loop's (`_step_world`; #690; SimNotes, "One robot's interrupt asked, the
    other robot held"). ⚠ EACH OTHER ROBOT IS HELD THROUGH ITS WAIT, its own
    routine never advanced, so no second blocking run begins inside the
    first -- and told how many steps it was held (its stepper's `held`,
    where it has one). The step's exception is thrown into `routine`, then
    carried past its callers to `run_many` (`_Raised`), which throws it into
    every other live routine and raises it: one each routine might have
    taken ends the day here, where `run_many`'s own step lets it go on. A
    held robot's wait's own is carried too, and raised as it is."""
    me = next(i for i, sw in enumerate(self.steppers) if sw is stepper)
    step = Step(routine, name)
    stepped = 0
    try:
      cmd = step.tick()
      while cmd is not None:
        try:
          now = [cmd if i == me else
                 _held(self.steppers, self.waits, self.holding, i, self.name, "held")
                 for i in range(len(self.steppers))]
        except BaseException as e:  # noqa: BLE001 -- a held robot's own: the loop's
          raise _Raised(e, step=False) from e
        try:
          _step_world(self.steppers, now, self.step)
        except BaseException as e:  # noqa: BLE001 -- thrown in, then the loop's
          try:
            step.tick(e)
          except BaseException:  # noqa: BLE001 -- as it came out of the routine
            pass
          raise _Raised(e, step=True) from e
        stepped += 1
        cmd = step.tick()
    finally:
      step.routine.close()
      for i, sw in enumerate(self.steppers):
        told = getattr(sw, "held", None)
        if i != me and stepped and told is not None:
          told(stepped)
    return step.result


def run(stepper, routine: Routine, name: str = "") -> Any:
  """Drive a routine to completion, stepping `stepper` with every command,
  and return what it returned. The blocking twin of `yield from` -- and,
  begun inside a `run_many` that drives `stepper` with other robots, that
  loop's steps (`_Loop.run`)."""
  loop = next((lp for lp in reversed(_LOOPS)
               if len(lp.steppers) > 1 and any(sw is stepper for sw in lp.steppers)), None)
  if loop is not None:
    return loop.run(stepper, routine, name)
  step = Step(routine, name)
  cmd = step.tick()
  while cmd is not None:
    try:
      stepper.step(cmd)
    except BaseException as e:  # noqa: BLE001 -- re-raised inside the routine
      cmd = step.tick(e)
    else:
      cmd = step.tick()
  return step.result
