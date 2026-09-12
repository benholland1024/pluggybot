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

  Command   (v, w): forward speed m/s and yaw rate rad/s, the body command
            every controller here already produces; `control.wheel_targets`
            turns it into wheel setpoints and `_step_once` ramps those.
  Routine   a generator of Commands whose RETURN value is the manoeuvre's
            result (`drive_to` returns whether it arrived, `pick` returns why
            it stopped). Composed with `yield from`, exactly as the blocking
            calls were composed with `return`.
  Step      one routine, ticked from outside: `tick()` hands back the next
            command or None when the routine has returned.
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

from pluggybot.control import wheel_targets

Command = tuple[float, float]
Routine = Generator[Command, None, Any]


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
  drive with (`life.mission.drive_to_routine = lambda *a, **kw:
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


def run(swap, routine: Routine, name: str = "") -> Any:
  """Drive a routine to completion, stepping `swap` with every command, and
  return what it returned. The blocking twin of `yield from`."""
  step = Step(routine, name)
  cmd = step.tick()
  while cmd is not None:
    try:
      swap._step_once(*wheel_targets(*cmd))
    except BaseException as e:  # noqa: BLE001 -- re-raised inside the routine
      cmd = step.tick(e)
    else:
      cmd = step.tick()
  return step.result
