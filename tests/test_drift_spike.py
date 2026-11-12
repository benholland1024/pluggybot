"""A drift flight whose robot never gets up ends `stuck`, never runs on for
good (`scripts/humanoid_drift_spike.py`): its lifecycle is not mortal, and
where nothing dies a get-up that would have died `stuck` ends the run
(`lifecycle.Stuck`, #684; SimNotes, "A flight that never got up"). The
rule is pinned in `tests/test_reset_robot.py`; here, that the flight keeps
what it flew and says where it lies.

Pinned without a flight (docs/Testing.md): `fly` on the stub body
(`tests/test_body.py`), its get-up never standing. The stub passes time
only where a routine holds (the rest at the storeroom), and a flight that
never ended would end there anyway, without its `stuck`.
"""

import sys
from pathlib import Path

import pytest

from pluggybot.body import StubBody
from pluggybot.humanoid.body import HumanoidBody
from test_body import stub_life  # noqa: I001 -- tests/ is on sys.path

ROOT = Path(__file__).resolve().parents[1]


def _spike():
  sys.path.insert(0, str(ROOT / "scripts"))
  import humanoid_drift_spike as hds
  return hds


class Tracker:
  """`humanoid_drift_spike.Tracker`'s face, with nothing patched."""

  def __init__(self, body):
    self.phase, self.rows, self.residual, self.due = "start", [], [0.0, 0.0], 0.0

  def off(self):
    return [0.0, 0.0]

  def zone_table(self):
    return []


def test_a_flight_whose_robot_never_gets_up_ends_stuck_with_what_it_flew(monkeypatch):
  hds = _spike()
  body = StubBody()
  body.rights_itself, body.stuck_after_s = True, HumanoidBody.GETUP_BUDGET_S
  life = stub_life(body=body)
  life.max_sim_time = life.explore_deadline = 1e9     # as `station_spike._life` sets them
  monkeypatch.setattr(hds, "_life", lambda: (life, body, None))
  monkeypatch.setattr(hds, "draws", lambda body, seed: {})
  monkeypatch.setattr(hds, "Tracker", Tracker)
  body.posture = "getting_up"
  out = hds.fly("13", "hall", explore_s=240.0, trips=2, rest_s=60.0)
  stuck = out["stuck"]
  # ...in the first rest, where the stub first holds, its get-up's budget in
  assert stuck["phase"] == "rest1", stuck
  assert stuck["t"] - stuck["fellT"] == pytest.approx(body.stuck_after_s, abs=0.2)
  assert stuck["why"].startswith("fell and could not get up")
  assert stuck["at"] == [round(body.x, 2), round(body.y, 2)]
  assert stuck["zone"] == hds.zone_of(body.x, body.y)
  # ...what it flew is kept, and nothing it never reached is said
  assert [w["walk"] for w in out["walks"]] == ["out1"] and "docked" not in out
  assert out["simS"] == pytest.approx(stuck["t"], abs=0.1)
