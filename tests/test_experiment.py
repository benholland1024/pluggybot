"""The measurement harness (issue #106): records, the rollup, and the seam
they read from.

Everything here runs without a sim except the one slow test at the bottom,
which flies the scripted arm for half a sim-minute through the real script.
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from pluggybot.evaluation import notes as nt
from pluggybot.evaluation import record as rec
from pluggybot.evaluation import rollup as ru
from pluggybot.evaluation.run import arm_flags
from pluggybot.lifecycle import TOP_UP_BELOW, board_book
from pluggybot.mind import overseer as overseer_mod
from pluggybot.mind.overseer import Menu, Overseer

from test_overseer import FakeClient, full  # noqa: I001 -- tests/ is on sys.path

REPO = Path(__file__).parent.parent
RESULTS = REPO / "results"


def _state(fraction: float, offers=(), affordable=None, possible=None) -> dict:
  return {"simTimeS": 100.0,
          "battery": {"fraction": fraction, "wh": 8.0 * fraction,
                      "spendableWh": max(0.0, 8.0 * fraction - 0.9)},
          "offeredTasks": list(offers),
          "affordableActions": list(affordable or ["draw", "charge"]),
          "possibleActions": list(possible or ["draw", "charge"]),
          "tasksThisMission": [], "decisions": 0}


def _config(**kw) -> dict:
  return {"world": "home", "arm": "guarded", "pack": "hosting",
          "model": "fake/model", "seed": 0, "maxSimS": 3600.0, **kw}


def _say(t, fraction, state, msg) -> dict:
  return {"kind": "say", "t": t, "fraction": fraction, "wh": 8 * fraction,
          "state": state, "msg": msg}


def _decision(t, fraction, action, source="llm", **kw) -> dict:
  row = dict(t=t, fraction=fraction, wh=8 * fraction,
             spendableWh=max(0.0, 8 * fraction - 0.9),
             action=action, source=source, task="", board="", program="",
             zone="", reason="", wallS=2.0, error="", learn=False,
             forget=False, note=False, escalate=False, offers=[],
             affordable=["draw", "charge"], possible=["draw", "charge"],
             hunger=None)
  row.update(kw)
  return rec.decision_row(**row)


def _result(**kw) -> dict:
  return {"aborted": False, "stranded": False, "battery": 0.6,
          "sim_time": 3650.0, "charge_cycles": 2, "earned": 100, "points": 71,
          "metabolism": {"consumed": 29, "spilled": 0, "state": "satisfied"},
          "overseer": {"backend": "huggingface", "constrained": True,
                       "budgetLeft": 40, "usd": 0.001, "priced": True,
                       "inputTokens": 1, "outputTokens": 1},
          "thought_stats": {"refusals": [], "chars": {}},
          "errands": [], "task_stats": {"done": 3, "failed": 1, "expired": 2},
          **kw}


# ---- the seam ----------------------------------------------------------------


def test_the_decision_hook_carries_the_context_and_the_clock():
  """Pass 1a had to monkeypatch three methods to see what the model was
  shown. `on_decision` is that seam, and it carries the state VERBATIM,
  the wall time, and the vendor's own error text -- the one thing
  `usage.errors` forgets after five entries. Fails without the hook:
  `Overseer` has no `on_decision`."""
  menu = Menu.for_world("home", board_book("home"))
  boss = Overseer(menu, client=FakeClient(full(action="draw", board="whiteboard_a",
                                               program="house", reason="ok"),
                                          "this is not json"))
  seen = []
  boss.on_decision.append(seen.append)
  state = _state(0.5)
  boss.decide(state)
  assert seen[0]["state"]["battery"]["fraction"] == 0.5
  assert seen[0]["decision"].action == "draw" and seen[0]["error"] == ""
  assert seen[0]["wallS"] >= 0.0
  boss.decide(_state(0.4))
  assert seen[1]["decision"].scripted, "a garbled answer falls back"
  assert seen[1]["error"].startswith("call:"), seen[1]["error"]
  # ...and the scripted path reports through the same seam, with its state.
  boss.decide_scripted(_state(0.3), "scripted-mode")
  assert seen[2]["state"]["battery"]["fraction"] == 0.3
  assert seen[2]["decision"].source == "fallback:scripted-mode"


# ---- the record --------------------------------------------------------------


def test_a_record_classifies_every_charge_by_cause_and_splits_voluntary():
  """Three causes, not two: the baseline's eleven gate deferrals would have
  been booked as `forced` by the provisional schema. And a chosen `charge`
  at 88 % is refused by the loop, so `chosen` and `honoured` differ."""
  events = [
    _decision(100, 0.60, "charge"),                      # honoured
    _say(100, 0.60, "DECIDE", "DECIDE charge: topping up early"),
    _say(101, 0.60, "GO_CHARGE", "GO_CHARGE -> CHARGE (pins connected)"),
    _say(600, 0.90, "CHARGE", "CHARGE complete (90%) -- backing off"),
    _decision(700, 0.88, "charge"),                      # refused, >= 0.75
    _say(700, 0.88, "DECIDE", "DECIDE charge: again"),
    _say(700, 0.88, "DECIDE", "DECIDE: already at 88%, not worth a trip"),
    _decision(1200, 0.20, "draw"),
    _say(1200, 0.20, "DECIDE", "DECIDE draw (house on whiteboard_b): ok"),
    _say(1201, 0.20, "DECIDE", "DEFER draw:whiteboard_b: draw needs 1.99 Wh"),
    _say(1202, 0.20, "GO_CHARGE", "GO_CHARGE -> CHARGE (pins connected)"),
    _say(1800, 0.90, "CHARGE", "CHARGE complete (90%) -- backing off"),
    _say(2400, 0.08, "EXPLORE", "EXPLORE -> GO_CHARGE (battery low)"),
    _say(2401, 0.08, "GO_CHARGE", "GO_CHARGE -> CHARGE (pins connected)"),
    _decision(3000, 0.5, "take_task", source="fallback:timeout", task="t_1"),
    _decision(3100, 0.5, "draw", source="fallback:garbled",
              error="call: ValueError: no"),
  ]
  r = rec.validate(rec.build_record(_config(), _result(), events, 500.0,
                                    datetime.now(timezone.utc),
                                    hashes=rec.data_hashes("home"), commit="abc"))
  ch = r["charging"]
  assert (ch["forced"], ch["deferred"], ch["docked"]) == (1, 1, 3)
  assert ch["voluntary"]["chosen"] == 2 and ch["voluntary"]["honoured"] == 1
  assert ch["voluntary"]["honouredFrac"] == [0.60]
  assert [c["cause"] for c in ch["entries"]] == ["voluntary", "deferred", "forced"]
  assert r["end"] == "day over" and r["survival"]["deaths"] == {"flat": 0, "stuck": 0}
  assert r["survival"]["minFraction"] == 0.08
  m = r["mind"]
  assert (m["decisions"], m["llmCalls"], m["fallbacks"]) == (5, 3, 2)
  assert m["fallbackReasons"] == {"fallback:timeout": 1, "fallback:garbled": 1}
  assert m["errors"] == ["call: ValueError: no"], "the vendor's words survive"
  assert "escalations" not in m, "absent, not zero, with no escalation model"
  assert r["economy"]["identityHolds"] is True
  assert TOP_UP_BELOW == 0.75, "the honoured split assumes the loop's gate"


def test_anticipation_needs_the_offers_the_model_was_shown():
  """Both definitions the baseline tried, computable only because each
  decision row keeps the board as it stood."""
  offers = [{"id": "t_9", "kind": "count_plants", "estimateWh": 1.18,
             "claimable": False, "expiresInS": 300.0}]
  events = [
    _decision(100, 0.20, "charge", offers=offers,
              affordable=["charge"], possible=["draw", "charge"]),
    _say(100, 0.20, "DECIDE", "DECIDE charge: a big job is coming"),
    _say(101, 0.20, "GO_CHARGE", "GO_CHARGE -> CHARGE (pins connected)"),
  ]
  r = rec.build_record(_config(), _result(), events, 1.0,
                       datetime.now(timezone.utc), hashes=rec.data_hashes("home"),
                       commit="abc")
  assert r["charging"]["anticipation"] == {"offer": 1, "action": 1}
  assert r["decisionRows"][0]["unfundableOffers"] == ["t_9"]
  assert r["decisionRows"][0]["notAffordable"] == ["draw"]


def test_a_killed_run_is_a_record_that_says_so_and_is_no_death():
  """The baseline's wedged day (#108) never ended. Killed on wall clock, it
  is recorded from the rows it flushed -- and it is NOT a `stuck` death,
  because the harness cannot tell a wedge from a slow box."""
  events = [_decision(100, 0.5, "draw"),
            _say(2259, 0.57, "USE_TOOL", "USE_TOOL: arrived"),
            _say(4327, 0.0, "USE_TOOL", "TASK t_0016 expired: whatever")]
  r = rec.validate(rec.build_record(_config(), None, events, 9000.0,
                                    datetime.now(timezone.utc),
                                    hashes=rec.data_hashes("home"), commit="abc"))
  assert r["end"] == "killed" and r["simSeconds"] == 4327
  # The rows show the pack at zero at t=4327, so THAT is known: a flat death
  # with a survival span. What is not known is whether it was stuck first.
  assert r["survival"]["survivalS"] == [4327]
  assert r["survival"]["deaths"] == {"flat": 1, "stuck": 0}
  assert r["survival"]["batteryEnd"] == 0.0
  # ...and without a zero in the rows, nothing at all is claimed.
  r2 = rec.validate(rec.build_record(_config(), None, events[:2], 9000.0,
                                     datetime.now(timezone.utc),
                                     hashes=rec.data_hashes("home"), commit="abc"))
  assert r2["survival"]["survivalS"] == [] and r2["survival"]["deaths"] == \
    {"flat": 0, "stuck": 0}
  doc = ru.rollup([r])
  s = doc["series"][0]
  assert s["killed"] == 1 and s["survival"]["n"] == 0


def test_a_record_keeps_the_id_its_file_was_named_by():
  """The parent names the file before the child starts; the child must not
  re-derive the id off its own clock. Found in the first committed set:
  ten records whose `runId` disagreed with their file name by seconds."""
  cfg = _config(runId="2026-09-07T00-00-00Z_home_guarded_hosting_x_s0",
                startedAt="2026-09-07T00:00:00+00:00")
  r = rec.build_record(cfg, _result(), [], 1.0,
                       datetime(2026, 9, 7, 0, 0, 7, tzinfo=timezone.utc),
                       hashes=rec.data_hashes("home"), commit="abc")
  assert r["runId"] == cfg["runId"] and r["startedAt"] == cfg["startedAt"]


def test_a_pack_that_reached_zero_mid_day_is_a_flat_death():
  """§3: `flat` is the pack reaching zero -- not the run ending on it. The
  motors do not stop at 0 Wh, so a day can hit zero inside an errand, dock
  on nothing and end "day over"; the first committed set had one."""
  events = [_say(900, 0.30, "USE_TOOL", "USE_TOOL: arrived"),
            _say(1400, 0.0, "USE_TOOL", "USE_TOOL: never got there"),
            _say(1500, 0.0, "GO_CHARGE", "GO_CHARGE -> CHARGE (pins connected)"),
            _say(2100, 0.9, "CHARGE", "CHARGE complete (90%) -- backing off")]
  r = rec.build_record(_config(), _result(battery=0.55), events, 1.0,
                       datetime.now(timezone.utc), hashes=rec.data_hashes("home"),
                       commit="abc")
  assert r["end"] == "day over"
  assert r["survival"]["deaths"] == {"flat": 1, "stuck": 0}
  assert r["survival"]["survivalS"] == [1400] and r["survival"]["flatAtS"] == 1400
  assert r["charging"]["entries"][0]["cause"] == "forced"


def test_a_stranded_day_is_a_stuck_death():
  """§3: `stuck` is "knocked over, wedged, or unable to reach the rack"; a
  `stranded` end is the third of those. The first committed set had one:
  a dropped module, then "no route to the charge bay" at 24 %."""
  r = rec.build_record(_config(), _result(stranded=True, battery=0.24,
                                          sim_time=1083.0),
                       [_say(1083, 0.24, "GO_CHARGE", "GO_CHARGE: no route")],
                       1.0, datetime.now(timezone.utc),
                       hashes=rec.data_hashes("home"), commit="abc")
  assert r["end"] == "stranded"
  assert r["survival"]["deaths"] == {"flat": 0, "stuck": 1}
  assert r["survival"]["survivalS"] == [1083.0]


def test_end_causes_are_read_off_the_result():
  assert rec.end_cause({"aborted": False, "stranded": True, "battery": 0.2,
                        "sim_time": 100}, 3600) == "stranded"
  assert rec.end_cause({"aborted": False, "stranded": False, "battery": 0.0,
                        "sim_time": 100}, 3600) == "flat"
  assert rec.end_cause({"aborted": False, "stranded": False, "battery": 0.5,
                        "sim_time": 3650}, 3600) == "day over"
  assert rec.end_cause({"aborted": False, "stranded": False, "battery": 0.5,
                        "sim_time": 400}, 3600) == "complete"
  assert rec.end_cause(None, 3600) == "killed"


def test_validation_names_what_is_wrong():
  r = rec.build_record(_config(), _result(), [], 1.0, datetime.now(timezone.utc),
                       hashes=rec.data_hashes("home"), commit="abc")
  assert rec.problems(r) == []
  bad = {**r, "end": "vanished", "dataHashes": {**r["dataHashes"], "energy": "x"}}
  found = rec.problems(bad)
  assert any("end 'vanished'" in p for p in found)
  assert any("dataHashes['energy']" in p for p in found)
  with pytest.raises(ValueError, match="vanished"):
    rec.validate(bad)


def test_the_world_is_part_of_the_regime(tmp_path, monkeypatch):
  """Issue #110 changed one attribute of the robot model and every scripted
  day after it was a different trajectory, with the five data files
  untouched. So the world -- its XML, its includes, its assets -- hashes
  into the regime, and two worlds never share one."""
  home, hub = rec.world_hash("home"), rec.world_hash("room_hub")
  assert home != hub and len(home) == 64
  assert rec.data_hashes("home")["world"] == home
  # an included file counts: copy the world tree, touch the robot model
  import shutil
  tree = tmp_path / "models"
  shutil.copytree(REPO / "models", tree)
  monkeypatch.setattr(rec, "REPO", tmp_path)
  assert rec.world_hash("home") == home, "the copy hashes the same"
  robot = tree / "pluggybot_fork.xml"
  robot.write_text(robot.read_text().replace('offsamples="0"', 'offsamples="4"'))
  assert rec.world_hash("home") != home, "an included file's edit is invisible"


def test_data_hashes_follow_the_env_override(tmp_path, monkeypatch):
  """A record hashes the file the sim would READ, not the shipped one:
  `$PLUGGY_ENERGY` re-points the sim, so it re-points the hash."""
  before = rec.data_hashes("home")
  other = tmp_path / "energy.json"
  other.write_text('{"version": 1, "default": {"errandWh": {}}, "worlds": {}}')
  monkeypatch.setenv(rec.DATA_FILES["energy"][1], str(other))
  after = rec.data_hashes("home")
  assert after["energy"] != before["energy"]
  assert {k: v for k, v in after.items() if k != "energy"} == \
    {k: v for k, v in before.items() if k != "energy"}


# ---- the rollup --------------------------------------------------------------


def _record(seed: int, hashes: dict, **result_kw) -> dict:
  return rec.build_record(_config(seed=seed), _result(**result_kw),
                          [_decision(100, 0.5, "draw")], 10.0,
                          datetime(2026, 9, 7, 0, 0, seed, tzinfo=timezone.utc),
                          hashes=hashes, commit="abc")


def test_the_rollup_refuses_to_aggregate_across_a_data_file_edit():
  """Two runs, one name, two `energy.json`s: not a series. The refusal names
  both regimes so whoever reads it can see which run to move."""
  a = rec.data_hashes("home")
  b = {**a, "energy": "f" * 64}
  with pytest.raises(ru.MixedRegime) as err:
    ru.rollup([_record(0, a), _record(1, b)])
  assert "2 regimes (data files + world)" in str(err.value)
  assert a["energy"][:12] in str(err.value) and "fff" in str(err.value)
  # ...and a different WORLD is a different regime too (issue #110).
  with pytest.raises(ru.MixedRegime):
    ru.rollup([_record(0, a), _record(1, {**a, "world": "0" * 64})])
  # ...while two runs on ONE regime are a series with a distribution.
  doc = ru.rollup([_record(0, a, battery=0.6), _record(1, a, battery=0.3)],
                  current=lambda world: a)
  s = doc["series"][0]
  assert s["n"] == 2 and s["current"] is True
  assert s["survival"]["minFraction"]["values"] == [0.5, 0.5]
  assert s["charging"]["voluntaryChosen"] == {"n": 2, "min": 0, "median": 0,
                                              "max": 0, "values": [0, 0]}
  assert ru.rollup([_record(0, a)],
                   current=lambda world: b)["series"][0]["current"] is False


def test_the_rollup_keeps_an_intervened_run_out_of_survival():
  a = rec.data_hashes("home")
  r = _record(0, a)
  r["interventions"] = [{"t": 10, "what": "battery set to 100 %"}]
  s = ru.rollup([r, _record(1, a)])["series"][0]
  assert s["withInterventions"] == 1 and s["survival"]["n"] == 1


def test_the_autonomous_arm_is_refused_until_it_exists():
  # `standing_orders` is stated rather than defaulted (issue #125): whose
  # the fallback is is part of what an arm MEANS, and `guarded` is the
  # control -- tests/test_standing_orders.py pins the other half.
  assert arm_flags("scripted") == {"overseer": False,
                                   "standing_orders": False}
  assert arm_flags("guarded") == {"overseer": True, "standing_orders": False}
  with pytest.raises(NotImplementedError, match="Evaluation.md"):
    arm_flags("autonomous")


def test_the_wall_limit_has_a_floor_for_short_days():
  """A run's start-up cost does not scale with the day: 3 x 30 s killed the
  slow test below under the full suite's load and called it `killed`."""
  sys.path.insert(0, str(REPO / "scripts"))
  from experiment import WALL_FLOOR_S, WALL_PER_SIM_S, wall_limit_for
  assert wall_limit_for(30.0) == WALL_FLOOR_S
  assert wall_limit_for(3600.0) == WALL_PER_SIM_S * 3600.0
  assert wall_limit_for(3600.0, 50.0) == 50.0


# ---- the committed results ---------------------------------------------------


def test_the_archive_is_kept_out_of_the_rollup():
  """`results/archive/` holds runs from a previous regime (the pre-#110
  world) and is deliberately not a series: the loader reads one level."""
  archived = sorted((RESULTS / "archive").glob("*.json"))
  assert archived, "the pre-fix set should be archived, not deleted"
  live = {r["runId"] for r in ru.load_records(RESULTS)}
  assert not any(p.stem in live for p in archived)
  # ...and an archived record is visibly from before the world hash existed
  assert "world" not in json.loads(archived[0].read_text())["dataHashes"]


def test_committed_results_are_valid_and_the_rollup_is_current():
  """Results are vendored the way `protocol/` fixtures are (Evaluation.md
  §4): every record validates, and `results/rollup.json` is exactly what
  the records roll up to against TODAY's data files. Editing a data file
  flips `current` and fails this, which is the cheapest place to learn the
  committed numbers now describe a previous regime."""
  records = ru.load_records(RESULTS)
  assert records, "the first committed result set is missing (issue #106)"
  for path, r in zip(sorted(p for p in RESULTS.glob("*.json")
                            if p.name not in ru.NOT_RECORDS), records):
    assert rec.problems(r) == [], r["runId"]
    assert path.stem == r["runId"], f"{path.name} carries runId {r['runId']}"
  committed = json.loads((RESULTS / ru.ROLLUP_NAME).read_text())
  fresh = ru.rollup(records, current=rec.data_hashes)
  assert committed == fresh, \
    "stale rollup: uv run python scripts/experiment.py --rollup"


def test_the_write_ups_cover_every_committed_series_and_nothing_else():
  """Evaluation.md §8's rule, with teeth (rooftop-media-2026 #187). A result
  set lands with what it MEANT or it is a set nobody can read, including us
  in three months -- and the website's data page renders this file rather
  than inventing an interpretation at render time.

  Both directions, because each fails silently on its own: a new series
  flown with no entry ships numbers with no reading, and an entry left
  behind by a re-flown series describes runs that are no longer there."""
  doc = nt.load(RESULTS)
  committed = json.loads((RESULTS / ru.ROLLUP_NAME).read_text())
  ids = [nt.series_id(s["world"], s["arm"], s["pack"], s["model"],
                      s.get("label", ""))
         for s in committed["series"]]
  assert nt.problems(doc, ids) == []


def test_the_write_ups_are_not_read_as_runs():
  """`results/` is globbed for records, so a sidecar that is not named as
  one is loaded as a malformed run. Fails without `NOT_RECORDS`: the rollup
  reports eleven runs and refuses the eleventh as invalid."""
  assert nt.NOTES_NAME in ru.NOT_RECORDS
  runs = ru.load_records(RESULTS)
  assert all("runId" in r for r in runs)
  assert len(runs) == json.loads((RESULTS / ru.ROLLUP_NAME).read_text())["runs"]


def test_a_write_up_names_its_limits_or_it_is_not_one():
  """The four fields are §3's baseline sections reduced, and `notShown` is
  the one a writer skips -- so an entry without it is refused rather than
  rendered as a set with no caveats. Same for a series named by a typo,
  which otherwise explains nothing while looking complete."""
  entry = {"series": "home/scripted/hosting/none", "title": "t",
           "date": "2026-09-07", "ran": "r", "found": ["f"],
           "changed": "", "notShown": "n"}
  assert nt.problems({"schema": 1, "entries": [entry]},
                     ["home/scripted/hosting/none"]) == [], \
    "an empty `changed` is a claim (nothing moved); an absent key is not"
  blank = nt.problems({"schema": 1, "entries": [{**entry, "notShown": ""}]}, None)
  assert any("notShown" in p for p in blank), blank
  gone = nt.problems({"schema": 1, "entries": [entry]}, ["home/guarded/hosting/x"])
  assert len(gone) == 2 and any("no write-up" in p for p in gone), gone
  assert any("not in the rollup" in p for p in gone), gone


@pytest.mark.slow
def test_the_script_writes_a_record_and_a_rollup(tmp_path):
  """The acceptance line, at the smallest size that flies: one scripted run
  of room_hub for 30 sim-seconds through the real script and child process.
  Slow because it is a whole-mission run, and short because its claim -- a
  valid record and a rollup appear -- is settled the moment the child exits.
  """
  env = {**os.environ, "MUJOCO_GL": os.environ.get("MUJOCO_GL", "egl")}
  proc = subprocess.run(
    [sys.executable, "scripts/experiment.py", "--arm", "scripted", "--world",
     "room_hub", "--pack", "demo", "-n", "1", "--max-sim-time", "30",
     "--errand", "none", "--no-tasks", "--no-metabolism",
     "--results", str(tmp_path)],
    cwd=REPO, env=env, capture_output=True, text=True, timeout=600)
  assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
  records = ru.load_records(tmp_path)
  assert len(records) == 1
  r = records[0]
  assert rec.problems(r) == [] and r["arm"] == "scripted"
  assert r["end"] in ("day over", "complete") and r["simSeconds"] >= 30
  assert r["mind"]["decisions"] == 0, "a scripted day asks no mind"
  doc = json.loads((tmp_path / ru.ROLLUP_NAME).read_text())
  assert doc["series"][0]["n"] == 1 and doc["series"][0]["current"] is True


# ---- the deadline, and the runs the box decided (issue #117) ------------------


def _fallback_record(seed: int, hashes: dict, rate: float, **cfg) -> dict:
  """A record whose fallback rate is `rate`, built from real decision rows so
  the number is derived rather than asserted into place."""
  n = 8
  rows = [_decision(10.0 * i, 0.5, "draw",
                    source="fallback:timeout" if i < round(rate * n) else "llm")
          for i in range(n)]
  return rec.build_record(_config(seed=seed, **{"deadlineS": 8.0, **cfg}),
                          _result(),
                          rows, 10.0,
                          datetime(2026, 9, 7, 0, 0, seed, tzinfo=timezone.utc),
                          hashes=hashes, commit="abc")


def test_a_run_the_box_decided_is_not_a_survival_data_point():
  """Issue #117. Every fallback is the scripted rotation deciding, and the
  rotation never charges -- so a guarded day flown a third by the rotation is
  a third a scripted day wearing the guarded name, and averaging its survival
  into the series reports the box.

  The three exclusions are one shape: an admin, the wall clock, and now the
  load. None of them deletes anything -- the run is still in the series, it
  still validates, and the reason travels with it."""
  a = rec.data_hashes("home")
  ok, over = _fallback_record(0, a, 0.125), _fallback_record(1, a, 0.5)
  assert ok["mind"]["fallbackRate"] == 0.125
  assert over["mind"]["fallbackRate"] == 0.5
  s = ru.rollup([ok, over])["series"][0]
  assert s["fallbackLimit"] == ru.FALLBACK_LIMIT["guarded"] == 0.25
  assert s["n"] == 2, "the excluded run is still IN the series"
  assert s["survival"]["n"] == 1 and s["overFallback"] == 1
  assert rec.problems(over) == [], "an excluded run still validates"
  [gone] = s["survival"]["excluded"]
  assert gone["runId"] == over["runId"] and "0.5" in gone["why"]
  assert "guarded" in gone["why"], "the reason names the arm's own threshold"
  # ...and the premise the threshold is judged against: on the arm where the
  # rotation IS the mind there is no limit and nothing is disqualified.
  scripted = ru.rollup([_fallback_record(2, a, 0.5, arm="scripted",
                                         model=None)])["series"][0]
  assert scripted["fallbackLimit"] is None
  assert scripted["survival"]["n"] == 1 and scripted["survival"]["excluded"] == []


def test_autonomous_is_the_strict_arm():
  """The thresholds are a judgement call and differ BY ARM on purpose: a
  fallback dilutes a `guarded` day, whose rails still charge the robot, and
  can end an `autonomous` one, where nothing else will. A single number for
  both would have to be one or the other."""
  assert ru.FALLBACK_LIMIT["autonomous"] < ru.FALLBACK_LIMIT["guarded"]
  assert set(ru.FALLBACK_LIMIT) == set(rec.ARMS)


def test_a_quiet_series_and_a_loaded_one_are_not_one_average():
  """`--label` (issue #117). Two runs of the same configuration on different
  machines are two results, and the pair is the point -- without the label in
  the series key the rollup pools them and reports a box that never existed.
  """
  a = rec.data_hashes("home")
  loaded = _fallback_record(0, a, 0.5)
  quiet = _fallback_record(0, a, 0.0, label="quiet")
  doc = ru.rollup([loaded, quiet])
  assert doc["runs"] == 2 and len(doc["series"]) == 2
  by_label = {s["label"]: s for s in doc["series"]}
  assert by_label[""]["n"] == 1 and by_label["quiet"]["n"] == 1
  assert "quiet" in quiet["runId"], "the label is legible in the file name"
  assert quiet["label"] == "quiet" and loaded["label"] == ""
  # ...and the write-ups follow the same key, so the quiet set lands with its
  # own reading rather than being explained by the loaded set's.
  assert nt.series_id("home", "guarded", "hosting", "m", "quiet") == \
      "home/guarded/hosting/m/quiet"
  assert nt.series_id("home", "guarded", "hosting", "m") == \
      "home/guarded/hosting/m", "an unlabelled series keeps the id it had"


def test_a_series_cannot_span_two_deadlines():
  """The deadline is part of the regime and no data-file hash catches it: it
  caps how much of a day the model decided at all, so an 8 s series pooled
  with a 20 s one averages two experiments. This is what raising
  `CALL_TIMEOUT_S` would otherwise have done silently to the committed set.
  """
  a = rec.data_hashes("home")
  with pytest.raises(ru.MixedRegime, match="deadline"):
    ru.rollup([_fallback_record(0, a, 0.0),
               _fallback_record(1, a, 0.0, deadlineS=20.0)])


def test_the_deadline_is_one_number_for_the_experiment_and_the_world(monkeypatch):
  """One source of truth (issue #117). The measurement and the deployed world
  are held to the same deadline ON PURPOSE: if the experiment ran at one and
  the website at another, the results would describe a robot nobody can
  watch. Nothing may override it per-world -- there is no flag and no env
  var, and the harness records the same constant it flew under."""
  from test_webserver import _serve_wiring

  monkeypatch.delenv(overseer_mod.BACKEND_ENV, raising=False)
  monkeypatch.delenv(overseer_mod.MODEL_ENV, raising=False)
  monkeypatch.setenv(overseer_mod.MODEL_ENV, "Qwen/Qwen3-4B-Instruct-2507")
  life, _, _ = _serve_wiring(monkeypatch, ["--world", "home", "--free-run",
                                           "--overseer"])
  served = life.init_kwargs["overseer"]
  assert served.backend == "huggingface"
  assert served.timeout_s == overseer_mod.CALL_TIMEOUT_S
  # ...and the harness writes down the number it actually flew under, off
  # the same constant rather than a copy of it.
  cfg = _config()
  record = rec.build_record({**cfg, "deadlineS": overseer_mod.CALL_TIMEOUT_S},
                            _result(), [], 1.0,
                            datetime(2026, 9, 7, tzinfo=timezone.utc),
                            hashes=rec.data_hashes("home"), commit="abc")
  assert record["config"]["deadlineS"] == overseer_mod.CALL_TIMEOUT_S
  assert served.timeout_s == record["config"]["deadlineS"]


def test_a_fallback_rate_is_a_function_of_the_latency_and_the_deadline():
  """Why this issue costs ten minutes and not five hours: the timeout share
  is arithmetic over a measured distribution, so the deadline is chosen from
  a curve the probe can produce with no physics at all."""
  sys.path.insert(0, str(REPO / "scripts"))
  from overseer_probe import CANDIDATE_DEADLINES, timeout_share

  latencies = [2.0, 2.5, 3.0, 9.0, 30.0]
  assert timeout_share(latencies, 8.0) == 0.4
  assert timeout_share(latencies, 45.0) == 0.0
  assert timeout_share([], 8.0) == 0.0
  # Monotonic in the deadline, which is the property that makes "what cap
  # gives under 5 %" a lookup rather than a search.
  shares = [timeout_share(latencies, d) for d in CANDIDATE_DEADLINES]
  assert shares == sorted(shares, reverse=True)


def test_a_distribution_is_read_at_its_tail_by_an_order_statistic():
  """`p95` here is the 95th value of a hundred, never an interpolation
  between two of them: at n=50 an interpolated tail is a number no call took,
  and the tail is what a deadline is chosen from."""
  vals = list(range(1, 101))
  assert rec.percentile(vals, 95) == 95 and rec.percentile(vals, 50) == 50
  assert rec.percentile(vals, 100) == 100 and rec.percentile([7], 95) == 7
  d = rec.dist(vals, rec.LATENCY_PERCENTILES)
  assert (d["p90"], d["p95"], d["max"]) == (90, 95, 100)
  # ...and percentiles are reported where the TAIL is the point and nowhere
  # else: a p95 over five days' charge counts is the maximum in disguise.
  assert "p95" not in rec.dist([1, 2, 3])


#: The quiet-box call-latency distribution the deadline was chosen from
#: (issue #117, 2026-09-07; `overseer_probe.py --calls 50`, Qwen3-4B on the
#: HF router). Here so a later change to `CALL_TIMEOUT_S` has to say which
#: of the two it disagrees with -- the measurement, or the margin.
MEASURED_P95_S = 6.59
MEASURED_MAX_S = 7.38


def test_the_deadline_clears_the_measured_tail_and_stays_under_a_model_load():
  """Issue #117: 8 s was never argued for and sat on the distribution -- the
  slowest of fifty calls used 92 % of it, which is why a box with a VM on it
  moved the same arm from 0 % to 19-47 % fallback.

  Both ceilings are load-bearing. `llm.LOCAL_TIMEOUT_S` is 45 s because a
  COLD LOCAL MODEL takes 27.3 s to load, a reason that does not apply to an
  API -- so an API deadline over that figure would quietly delete the
  distinction. And an escalation must stay the more patient of the two,
  because it buys a bigger model answering more tokens."""
  from pluggybot.mind import llm

  assert overseer_mod.CALL_TIMEOUT_S >= 3 * MEASURED_P95_S, \
      "the deadline has to clear the measured tail with margin, not sit on it"
  assert overseer_mod.CALL_TIMEOUT_S > 2 * MEASURED_MAX_S
  assert MEASURED_MAX_S < 8.0, \
      "the premise: at the OLD deadline nothing timed out on a quiet box, " \
      "so the committed 19-47 % was the box and not the model"
  # The deadline now covers a cold local model load on its own, so the local
  # constant is a FLOOR -- and the ordering it protects is the one asserted:
  # the slow path is never the impatient one.
  assert llm.default_timeout("local", overseer_mod.CALL_TIMEOUT_S) >= \
      overseer_mod.CALL_TIMEOUT_S
  assert overseer_mod.ESCALATE_TIMEOUT_S > overseer_mod.CALL_TIMEOUT_S, \
      "an escalation buys a bigger model answering more tokens"
