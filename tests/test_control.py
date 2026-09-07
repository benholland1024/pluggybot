"""`control.square_up` -- the ONE bounded squaring-up loop (issue #108).

The pen, the claw, the dispenser and the mission each carried their own
`while |heading error| > tol` with no bound of any kind. A robot that
cannot turn -- ridden up onto a board mount, wheels half off the floor --
sat in the pen's copy for 2000+ sim-seconds, drained the pack to 0 % at
stall current and kept going past the day's budget, because both mission
end conditions are checked between errands. These tests drive the shared
loop with a kinematic stub, so nothing here needs MuJoCo.
"""

import math

import pytest

from pluggybot.control import FACE_BUDGET_S, square_up, wrap_angle

DT = 0.002


class Body:
  """A yaw that integrates the commanded rate -- or refuses to move."""

  def __init__(self, theta: float, stuck: bool = False, lag: float = 0.0):
    self.theta, self.stuck, self.lag = theta, stuck, lag
    self.t = 0.0
    self.steps = 0
    self.w = 0.0

  def step(self, w: float) -> None:
    # a first-order lag on the rate stands in for `slew`: the body keeps
    # turning for a moment after the command drops, which is the overshoot
    # the settle-and-recheck exists for
    self.w = w if not self.lag else self.w + (w - self.w) * DT / self.lag
    if not self.stuck:
      self.theta += self.w * DT
    self.t += DT
    self.steps += 1

  def settle(self) -> None:
    for _ in range(int(0.6 / DT)):
      self.step(0.0)

  def face(self, heading: float, **kw):
    return square_up(lambda: wrap_angle(heading - self.theta), self.step,
                     self.settle, lambda: self.t, **kw)


def test_a_healthy_face_squares_up_well_inside_the_budget():
  body = Body(theta=math.radians(-90.0), lag=0.05)
  err, squared = body.face(0.0, tol=0.004)
  assert squared and abs(err) <= 0.004 * 4
  assert body.t < FACE_BUDGET_S / 3, f"took {body.t:.1f} s from a right angle"


def test_a_body_that_cannot_turn_is_given_up_on_at_the_budget():
  """THE issue-108 regression: without the bound this never returns. The
  stub raises after twice the budget so a regression fails instead of
  hanging the suite."""
  body = Body(theta=math.radians(-90.0), stuck=True)
  limit = int(2 * FACE_BUDGET_S / DT)
  real_step = body.step

  def guarded_step(w: float) -> None:
    if body.steps > limit:
      raise RuntimeError("square_up is unbounded again (issue #108)")
    real_step(w)

  body.step = guarded_step
  err, squared = body.face(0.0, tol=0.004)
  assert not squared, "a wedged body reported itself squared up"
  assert abs(err) == pytest.approx(math.pi / 2, abs=1e-6)
  assert FACE_BUDGET_S <= body.t < FACE_BUDGET_S + 1.0, \
    f"gave up at {body.t:.1f} s, budget is {FACE_BUDGET_S}"


def test_the_budget_is_the_callers_clock_not_a_step_count():
  body = Body(theta=1.0, stuck=True)
  _, squared = body.face(0.0, tol=0.004, budget_s=2.0)
  assert not squared and body.t == pytest.approx(2.0, abs=DT * 2)


def test_tries_is_still_a_bound_on_the_recheck():
  """A body whose settle always drifts it just outside `done_within` runs
  out of tries, not of time, and says so -- the old loops' `tries` kept
  its meaning."""
  body = Body(theta=0.5)
  drift = [0]

  def drifting_settle():
    drift[0] += 1
    body.theta += 0.1                    # every settle knocks it off again
    body.t += 0.6

  err, squared = square_up(lambda: wrap_angle(0.0 - body.theta), body.step,
                           drifting_settle, lambda: body.t, tol=0.004,
                           tries=3)
  assert drift[0] == 3 and not squared and abs(err) > 0.004 * 4
