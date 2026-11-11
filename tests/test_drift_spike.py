"""A drift flight whose robot never gets up ends `stuck`, never runs on for
good (`scripts/humanoid_drift_spike.py`, `stuck_guard`).

The loop's get-up has no budget of its own (`HumanoidMission._getup_routine`):
served, a fall it cannot get up from is the lifecycle's `stuck` death, whose
stand-up closes the routine; the spike's lifecycle is not mortal, and a
robot that fell round the garden lamp's pole got up for good -- 57 and 72
minutes at full CPU on a pod, with nothing on its stdout (SimNotes, "A
flight that never got up").

Pinned without a flight (docs/Testing.md): the guard's rule by direct calls,
and its wiring through `fly` on a body that steps nothing but a clock -- its
steps capped, so a flight that never ends fails here and never hangs.
"""

import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from pluggybot import tick

ROOT = Path(__file__).resolve().parents[1]
#: Where the robot lies (the garden, where 13h's did), and the fake's clock.
LIES = (5.4, 2.06, 0.0)
DT = 0.01


def _spike():
  sys.path.insert(0, str(ROOT / "scripts"))
  import humanoid_drift_spike as hds
  return hds


class Body:
  """What a flight asks of its body: a clock stepped by `tick.run`, every
  step its hooks; from `fall_t` on its get-up has the legs and never stands
  (`posture`). At most `most` steps: past them, the flight never ended."""

  def __init__(self, fall_t: float = 5.0, most: int = 20_000):
    self.data = NS(time=0.0)
    self.fall_t, self.most, self.steps = fall_t, most, 0
    self.step_hooks: list = []
    self.posture = "standing"
    self.falls, self.match_counts, self.belief_events = 0, {}, []
    self.closed = False

  def step(self, cmd) -> None:
    self.steps += 1
    assert self.steps <= self.most, "the flight never ended"
    self.data.time += DT
    if self.data.time >= self.fall_t and self.posture == "standing":
      self.posture, self.falls = "getting_up", self.falls + 1
    for hook in list(self.step_hooks):
      hook()

  def run(self, routine):
    return tick.run(self, routine)

  def true_pose(self):
    return LIES

  def start_at(self, *pose) -> None:
    pass

  def start_discovery(self) -> None:
    pass

  def look_around_routine(self):
    yield (0.0, 0.0, 0.0)

  def close(self) -> None:
    self.closed = True


class Life:
  """The lifecycle's explore: a walk that never returns on its own -- its
  get-up holds it, as `_getup_routine` held 13h's."""

  def __init__(self, body: Body):
    self.body, self.data = body, body.data

  def explore(self, budget, mark_done):
    def walking():
      while True:
        yield (0.3, 0.0, 0.0)
    return self.body.run(walking())


class Tracker:
  """`humanoid_drift_spike.Tracker`'s face, with nothing patched."""

  def __init__(self, body):
    self.phase, self.rows, self.residual, self.due = "start", [], [0.0, 0.0], 0.0

  def off(self):
    return [0.0, 0.0]

  def zone_table(self):
    return []


def test_a_get_up_not_stood_in_its_wait_ends_the_flight_and_one_stood_in_time_does_not():
  hds = _spike()
  body = Body()
  hook = hds.stuck_guard(body, wait_s=40.0)

  def at(t: float, posture: str) -> None:
    body.data.time, body.posture = t, posture
    hook()

  at(10.0, "standing")
  at(11.0, "getting_up")                    # fell
  at(50.9, "getting_up")                    # 39.9 s: still waited for
  at(51.0, "standing")                      # stood in time: forgotten
  at(52.0, "getting_up")                    # fell again
  at(91.9, "getting_up")
  with pytest.raises(hds.Stuck) as stuck:
    at(92.0, "getting_up")
  assert stuck.value.args[0] == {"fellT": 52.0, "t": 92.0, "zone": "garden", "at": [5.4, 2.06]}
  # ...raised on the seam, so `tick.run` throws it into the routine in flight
  assert issubclass(hds.Stuck, tick.MissionAborted)


def test_a_flight_whose_robot_never_gets_up_ends_stuck_with_what_it_flew(monkeypatch):
  hds = _spike()
  body = Body(fall_t=5.0)
  monkeypatch.setattr(hds, "_life", lambda: (Life(body), body, None))
  monkeypatch.setattr(hds, "draws", lambda body, seed: {})
  monkeypatch.setattr(hds, "Tracker", Tracker)
  out = hds.fly("13", "hall", explore_s=240.0, trips=2, rest_s=60.0)
  stuck = out["stuck"]
  assert stuck["phase"] == "explore" and stuck["zone"] == "garden", stuck
  assert stuck["fellT"] == pytest.approx(5.0, abs=2 * DT)
  assert stuck["t"] - stuck["fellT"] == pytest.approx(hds.GETUP_WAIT_S, abs=2 * DT)
  # ...nothing it never reached is said, and what it flew is kept
  assert "explored" not in out and "docked" not in out and out["walks"] == []
  assert out["falls"] == 1 and out["simS"] == pytest.approx(stuck["t"], abs=0.1)
  assert body.closed
