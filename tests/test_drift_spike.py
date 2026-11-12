"""A drift flight whose robot never gets up ends `stuck`, never runs on for
good (`scripts/humanoid_drift_spike.py`): its lifecycle is not mortal, and
where nothing dies a get-up that would have died `stuck` ends the run
(`lifecycle.Stuck`, #684; SimNotes, "A flight that never got up"). The
rule is pinned in `tests/test_reset_robot.py`; here, that the flight keeps
what it flew and says where it lies.

Pinned without a flight (docs/Testing.md): `fly` on the stub body
(`tests/test_body.py`), its get-up never standing. The stub passes time
only where a routine holds (the rest at the storeroom), and a flight that
never ended would end there anyway, without its `stuck`. And that the
instruments count a doubted fit as the fit it was (#681).
"""

import sys
from pathlib import Path
from types import SimpleNamespace as NS

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


def test_a_doubted_fit_is_a_scan_its_robot_matched():
  # #681: a fit the walls only partly agree with is "doubted" and taken as it
  # came, as "ok" was (`scan_match.FIT_TAKEN`): `--done` counts it among the
  # scans a waiting robot matched, or one on a sidewalk reads as matching
  # none, #593's sign of a robot lost while it waits.
  hds = _spike()
  assert hds.scans_taken({"ok": 3, "doubted": 2, "found": 1, "slid": 4}) == 5


def test_the_zone_table_counts_a_doubted_fit_as_the_fit_it_was(capsys):
  # ...and so does the table by zone every flight prints: counted by the
  # name "ok", a sidewalk whose held fits were half doubted read "ok 52 held
  # 100", the doubted half in no column (#681's review).
  hds = _spike()
  zone = {"zone": "sidewalk_east", "s": 60.0, "walkedM": 30.0, "scans": 100,
          "returnsPerScan": 300.0, "legsM": [0.0, 0.0], "headingM": [0.0, 0.0],
          "matcherM": [0.0, 0.0], "verdicts": {"ok": 52, "doubted": 48, "held": 100}}
  hds.zone_report([{"zones": [zone]}])
  line = next(ln for ln in capsys.readouterr().out.splitlines() if "sidewalk_east" in ln)
  assert line.split("|")[3].split()[:2] == ["100", "100"], line


def test_done_counts_the_doubted_fits_of_a_robot_waiting(monkeypatch):
  # `--done`'s matchedWaiting through `done()` itself, its pair a stub whose
  # waiting robot's every fit is doubted: read by the name "ok" it matched
  # none, #593's sign of a robot lost while it waits (#681's review: the
  # helper alone was pinned, never its one line of wiring).
  from pluggybot import pair
  hds = _spike()

  class Life:
    def __init__(self):
      self.data, self.state, self.expects_work = NS(time=0.0), "DONE", False
      self.body = NS(step_hooks=[], match_counts={}, pose=(0.0, 0.0, 0.0),
                     true_pose=lambda: (0.0, 0.0, 0.0))

  lives = [Life(), Life()]

  def run_pair(ls, max_sim_time, explore_budget):
    second = ls[1]
    for k in range(1, 61):
      second.data.time = float(k)
      second.body.match_counts = {"doubted": 10 * k}
      for hook in list(second.body.step_hooks):
        hook()
    return [{"sim_time": 60.0}, {"sim_time": 60.0}]

  monkeypatch.setattr(pair, "build_pair", lambda *a, **kw: lives)
  monkeypatch.setattr(pair, "run_pair", run_pair)
  out = hds.done(seconds=60.0)
  assert out["matchedWaiting"] == 600 - 10, out["matchedWaiting"]
