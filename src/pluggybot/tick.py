"""The physics seam: a ROUTINE yields one drive command per physics step
(issue #58, the tick refactor).

Every manoeuvre in this repo used to be a `while` loop that called
`HubSwap._step_once` itself -- the primitive owned the clock, and nothing
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
            the rover's is (v, w), forward speed m/s and yaw rate rad/s,
            which its stepper turns into wheel setpoints
            (`control.wheel_targets`) and `HubSwap._before_step` ramps.
  Routine   a generator of Commands whose RETURN value is the manoeuvre's
            result (`drive_to` returns whether it arrived, `pick` returns why
            it stopped). Composed with `yield from`, exactly as the blocking
            calls were composed with `return`.
  Step      one routine, ticked from outside: `tick()` hands back the next
            command or None when the routine has returned.
  Stepper   what drives one body's physics: `apply(command)` writes its
            setpoints before the world steps, `after_step()` is its
            bookkeeping after, `step(command)` is the three for a robot
            alone, `STILL` the command that holds it where it is, and
            `model` / `data` the world it steps (`Body.stepper`; the
            rover's is `rack/swap.HubSwap`).
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


def run_many(pairs, name: str = "", step=None) -> list:
  """Drive several robots' routines from ONE physics loop (issue #167).

  `pairs` is `[(stepper, routine), ...]`, one per robot. Each step: every
  robot's command is applied (`apply`), the world steps ONCE, every robot's
  bookkeeping runs (`after_step`) -- the three things `step` does for one
  robot, in the same order, so a robot alone here is the robot alone
  there. A robot whose routine has returned holds `STILL` and waits for
  the others; the loop ends when all have returned.

  An exception from the step -- a hook's `MissionAborted` -- is thrown into
  EVERY live routine, so each one's cleanup runs (the swap timestep, the
  press flag), and then re-raised: one robot's stop is the day's stop.
  Returns each routine's result, in order.
  """
  steps = [Step(r, f"{name}:{i}") for i, (_, r) in enumerate(pairs)]
  steppers = [sw for sw, _ in pairs]
  if step is None:
    import mujoco

    def step():
      # ⚠ READ OFF THE STEPPER EVERY STEP, never captured once (issue
      # #315). A tool built mid-run recompiles the world, and
      # `spec.recompile` returns NEW MjModel / MjData objects: a closure
      # that bound them at loop start would go on stepping the world the
      # robots left, while every rebound holder read the new one. The
      # stepper is rebound, so it always knows which world this is.
      mujoco.mj_step(steppers[0].model, steppers[0].data)
  cmds = [st.tick() for st in steps]
  while any(c is not None for c in cmds):
    exc = None
    try:
      for sw, c in zip(steppers, cmds):
        sw.apply(c if c is not None else sw.STILL)
      step()
      for sw in steppers:
        sw.after_step()
    except BaseException as e:  # noqa: BLE001 -- re-raised inside every routine
      exc = e
    if exc is not None:
      first = None
      for i, st in enumerate(steps):
        if cmds[i] is None:
          continue
        try:
          cmds[i] = st.tick(exc)
        except BaseException as e:  # noqa: BLE001
          first = first or e
          cmds[i] = None
      if first is not None:
        raise first
      continue
    cmds = [st.tick() if c is not None else None for st, c in zip(steps, cmds)]
  return [st.result for st in steps]


def run(stepper, routine: Routine, name: str = "") -> Any:
  """Drive a routine to completion, stepping `stepper` with every command,
  and return what it returned. The blocking twin of `yield from`."""
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
