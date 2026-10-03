"""A tool's own servos, walked (issue #407): the claw's slide and jaws, and
any axis a built tool registers (`workshop/build.py`, `procedure/axes.py`)
-- the one path a module's position servo is commanded through on legs.

Every setpoint is RAMPED, never written across a gap (CLAUDE.md): a stiff
servo handed a step delivers an impulse, and on the rover that threw a
module off the fork and batted a block out of the jaws. The body holds
still while one moves; the arm is held by its own driver.
"""

from __future__ import annotations

from pluggybot.tick import Routine


def ramp_routine(mission, act: int, target: float, speed: float,
                 settle: float = 0.0) -> Routine:
  """One of a tool's position servos (`act`, the world's actuator id) to
  `target`, clipped to its range, at `speed` units a second, then `settle`
  s still; returns where it was sent. `mission` is the body's
  (`legs.body.QuadMission`): its still steps keep the posture machine, the
  arm and the senses running."""
  return (yield from ramp_many_routine(mission, (act,), target, speed, settle))


def ramp_many_routine(mission, acts, target: float, speed: float,
                      settle: float = 0.0) -> Routine:
  """Several servos commanded as one (the claw's two jaws, one servo
  through a rack and pinion): each walked from where it is to `target`."""
  m, d = mission.model, mission.data
  acts = tuple(int(a) for a in acts)
  lo, hi = (float(v) for v in m.actuator_ctrlrange[acts[0]])
  target = min(max(float(target), lo), hi)
  start = [float(d.ctrl[a]) for a in acts]
  gap = max(abs(target - s) for s in start)
  n = max(1, int(gap / max(float(speed), 1e-6) / float(m.opt.timestep)))
  for k in range(1, n + 1):
    for a, s in zip(acts, start):
      d.ctrl[a] = s + (target - s) * k / n
    yield from mission._twist_routine(0.0, 0.0, 0.0)
  if settle > 0.0:
    yield from mission._drive_routine(settle, 0.0, 0.0)
  return target
