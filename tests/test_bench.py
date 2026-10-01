"""The physics bench (issue #227): find an unknown mass and record it.

docs/Challenges.md §8 is the decision; `challenge/bench.py` holds the
criteria, written before the robot saw the job. What these pin, each
without a mission (docs/Testing.md), and each shown to fail without its
rule:

  1. the bank rotates on the board's counter and refuses an entry the job
     could not be honest with;
  2. the offer sets the world, a restart restores it, the workshop's
     recompile keeps it, and the truth is absent from every context;
  3. the grader: the newest finding after the claim, in kilograms, within
     the tolerance -- pass and fail, on a stubbed record, and the reason
     never says the truth;
  4. the loop with no mind is not offered it (the tower's gate, #207);
  5. a procedure's locals reach History and the wire -- the readout a
     number needs to get from a sensor to a `record`.

The sensor a weighing reads is the arm's to rebuild on legs (#407).
"""

import json
import re
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from pluggybot import lifecycle as lc
from pluggybot.body import StubBody
from pluggybot.challenge import bench, stack
from pluggybot.economy import scoring
from pluggybot.economy.cadence import Cadence, TaskProducer
from pluggybot.economy.ledger import Ledger
from pluggybot.economy.tasks import KINDS, TaskBoard, kind_names
from pluggybot.evaluation import qualities as q
from pluggybot.evaluation.qualities import Row
from pluggybot.lifecycle import (QUAD_HOME, board_book, overseer_context,
                                 world_config, world_targets)
from pluggybot.mind import overseer as ov
from pluggybot.mind.thoughts import ThoughtFiles
from pluggybot.procedure import lang
from pluggybot.telemetry.protocol import ACT_EVENT_TYPES
from test_body import stub_life  # noqa: I001 -- tests/ is on sys.path

TABLE = scoring.challenge_table()
HOUSE = world_config(QUAD_HOME)["model"]       # the house, its bench cubes in it


@pytest.fixture(scope="module")
def home_model():
  return mujoco.MjModel.from_xml_path(HOUSE)


# ---- 0. the criteria are data the robot cannot move --------------------------------


def test_the_criteria_are_the_issues_own():
  """Written down first (Challenges.md §3): a relative tolerance, the
  record's topic and the word the quantity carries, no hold."""
  assert bench.TOLERANCE == 0.10
  assert bench.FINDINGS_TOPIC == "findings/mass_bench"
  assert bench.QUANTITY_WORD == "unknown"
  assert "No hold" in bench.__doc__
  row = TABLE["mass"]
  assert "mass" in scoring.EVALUATORS and "mass" in scoring.SAMPLERS
  assert row.tier == "hidden" and set(row.secret) == {"truth", "error"}
  shipped = scoring.default_table()
  assert not row.offered and "mass" not in shipped.offered
  assert "mass" not in {r["task"] for r in shipped.as_context()}
  assert "mass" in {r["task"] for r in shipped.as_context(challenges=True)}


def test_the_kind_is_a_challenge_offered_where_a_procedure_can_be_written():
  kind = KINDS["find_mass"]
  assert kind.discharge == "procedure" and kind.task == "mass"
  assert kind.target_kind == "bench" and "find_mass" in kind_names()
  # the offer is a work order: which tag is which, what the known weighs,
  # where to write the answer -- and no price, no mass for the unknown
  text = kind.describe("lab", {"known_g": 100, "known_tag": 23, "unknown_tag": 24})
  assert "100 g" in text and "tagged 23" in text and "tagged 24" in text
  assert "mass_bench" in text and "unknown mass = <value> kg" in text
  assert "point" not in text
  # the tower's gate exactly (issue #207): `bench` is named only where
  # there is a mind to write the procedure
  book = board_book(QUAD_HOME)
  assert "bench" not in world_targets(QUAD_HOME, book)
  assert world_targets(QUAD_HOME, book, procedures=True)["bench"] == ["lab"]
  beat = Cadence._build("test", {"kinds": {"feed_mouse": {}, "find_mass": {}}}, None)
  assert "find_mass" not in TaskProducer(TaskBoard(), beat, world_targets(QUAD_HOME, book)).kinds
  assert "find_mass" in TaskProducer(TaskBoard(), beat,
                                     world_targets(QUAD_HOME, book, procedures=True)).kinds


# ---- 1. the bank -----------------------------------------------------------------------


def test_the_bank_rotates_on_the_boards_counter_and_refuses_an_unfair_entry(tmp_path, monkeypatch):
  bank = bench.default_bank()
  assert len(bank.masses) >= 4 and len(set(bank.masses)) == len(bank.masses)
  assert [bank.pick(i) for i in range(len(bank.masses))] == list(bank.masses)
  assert bank.pick(len(bank.masses)) == bank.masses[0]        # wraps
  for kg in bank.masses:
    assert 0 < kg <= bench.MAX_KG
    assert abs(kg - bench.KNOWN_MASS_KG) / kg > bench.TOLERANCE

  def bank_of(*masses):
    p = tmp_path / "masses.json"
    p.write_text(json.dumps({"version": 1, "masses": [{"kg": m} for m in masses]}))
    return p
  with pytest.raises(ValueError, match="within 10%"):
    bench.MassBank.load(bank_of(0.2, 0.105))          # "it weighs what the other weighs"
  with pytest.raises(ValueError, match="holds at most"):
    bench.MassBank.load(bank_of(0.2, 0.5))            # the claw could not lift it
  with pytest.raises(ValueError, match="empty"):
    bench.MassBank.load(bank_of())
  # a deploy points at its own bank, as for the questions
  monkeypatch.setenv(bench.BANK_ENV, str(bank_of(0.33, 0.22)))
  assert bench.default_bank().masses == (0.33, 0.22)
  monkeypatch.delenv(bench.BANK_ENV)
  bench._BANK = None


def test_the_producer_draws_the_unknown_and_tells_only_the_known():
  beat = Cadence._build("home", {"firstAtS": 0.0, "everyS": 1.0, "initial": 0,
                                  "kinds": {"find_mass": {}}}, None)
  board = TaskBoard()
  producer = TaskProducer(board, beat, {"bench": ["lab"]})
  producer.tick(1.0, pack_wh=8.0)
  [task] = board.offered()
  assert task.kind == "find_mass" and task.target == "lab"
  assert task.secret == {"kg": bench.default_bank().pick(0)}
  assert task.params["known_g"] == 100 and task.params["known_tag"] == 23
  assert task.reward(board.table)["base"] == scoring.challenge_table()["mass"].base >= 50
  # the secret is on no surface but the state file (TaskPattern.md §2.1)
  secret = f"{task.secret['kg']:g}"
  for view in (task.as_dict(), task.snapshot(board.table), task.as_context(board.table)):
    assert secret not in json.dumps(view) and "secret" not in view
  assert task.as_state()["secret"] == task.secret


# ---- 2. the world ------------------------------------------------------------------------


class _Library:
  def runnable(self):
    return ()

  def as_context(self):
    return {}


class _Mind:
  event_map = None
  pending = None
  interrupt_pending = None
  can_escalate = False
  spend = None
  workshop = None
  wiki = None
  library = _Library()

  def __init__(self, lab="lab"):
    from dataclasses import replace
    self.decisions = []
    self.menu = replace(ov.Menu.for_world(QUAD_HOME, board_book(QUAD_HOME)),
                        procedures=True, lab=lab)


def _life(home_model, tmp_path, spec=None, model=None):
  """The house with its bench, the world's activities on its seam, a board
  and a ledger, on a stub body: an offer sets the WORLD and the grade reads
  a record, neither the robot."""
  cfg = world_config(QUAD_HOME)
  model = model if model is not None else home_model
  data = mujoco.MjData(model)
  body = StubBody(model, data, rack=cfg["rack"], grid_bounds=cfg["grid_bounds"])
  life = stub_life(body=body, battery_wh=cfg["hosting_battery_wh"],
                   ledger=Ledger(path=str(tmp_path / "ledger.json")),
                   tasks=TaskBoard(path=str(tmp_path / "tasks.json")),
                   boards=board_book(QUAD_HOME), spec=spec,
                   overseer=_Mind())
  acts = lc.home_activities(model, data)
  life.body.step_hooks.append(acts.step_hook(model, data))
  life.activities = acts
  life.body.start_at(*cfg["start"])
  return life


def _offer(life, kg=0.317, t=0.0):
  return life.tasks.offer("find_mass", "lab", ttl=1000.0, t=t, secret={"kg": kg},
                          params={"known_g": 100, "known_tag": 23, "unknown_tag": 24})


def _walk(node, out):
  if isinstance(node, dict):
    for v in node.values():
      _walk(v, out)
  elif isinstance(node, (list, tuple)):
    for v in node:
      _walk(v, out)
  elif isinstance(node, (int, float)) and not isinstance(node, bool):
    out.append(float(node))
  elif isinstance(node, str):
    out.extend(float(m) for m in re.findall(r"\d+\.\d+", node))


def test_the_offer_sets_the_world_and_the_truth_is_absent_from_every_context(home_model, tmp_path):
  """The cube weighs the placeholder until an offer lands, then what the
  offer drew (the board's own event, so a test's `offer` and the
  producer's are one path); nothing in the mind's context, the status
  line or the offer carries the number, and body masses are in no
  context at all. Its OWN model: the module's `home_model` is the one the
  other offers here set, and under `-n auto` whichever ran first decided
  whether the placeholder premise held."""
  life = _life(home_model, tmp_path, model=mujoco.MjModel.from_xml_path(HOUSE))
  assert bench.unknown_mass(life.model) == bench.UNKNOWN_MASS_KG
  said = []
  life.say_hooks.append(lambda t, line: said.append(line))
  task = _offer(life, kg=0.317)
  assert bench.unknown_mass(life.model) == pytest.approx(0.317)
  # inertia scaled with the mass, and the derived constants recomputed
  bid = life.model.body(bench.UNKNOWN).id
  assert life.model.body_inertia[bid][0] == pytest.approx(
    0.317 * (2 * stack.BLOCK_HALF) ** 2 / 6, rel=1e-6)      # a cube: m a^2 / 6
  assert any(line.startswith("BENCH") for line in said)
  assert not any("0.317" in line or "317" in line for line in said)
  numbers: list = []
  _walk(overseer_context(life), numbers)
  assert 0.317 not in numbers and 317.0 not in numbers
  assert "0.317" not in json.dumps(overseer_context(life))
  assert "body_mass" not in json.dumps(overseer_context(life))
  assert "0.317" not in json.dumps(task.as_dict()) and "0.317" not in task.description
  # the placeholder is not a truth either: the offer is what the world is
  assert "0.15" not in task.description
  # another kind's offer leaves the cube alone
  life.tasks.offer("draw_figure", "whiteboard_a", params={"program": "house"})
  assert bench.unknown_mass(life.model) == pytest.approx(0.317)


def test_a_restart_restores_the_open_offers_mass_and_the_recompile_keeps_it(home_model, tmp_path):
  """The world file carries the placeholder; a board that came back off
  disk with a bench offer on it still names the cube it was made for, and
  `begin` puts it back (`restore_bench`). The spec is written too, so the
  workshop's recompile (#168) rebuilds a world that still weighs it."""
  first = _life(home_model, tmp_path)
  _offer(first, kg=0.25, t=1.0)
  _offer(first, kg=0.18, t=2.0)          # the newest open offer wins
  # a fresh process over the same files: the model is the file's again
  spec = mujoco.MjSpec.from_file(HOUSE)
  fresh = _life(home_model, tmp_path, spec=spec, model=spec.compile())
  assert bench.unknown_mass(fresh.model) == bench.UNKNOWN_MASS_KG
  fresh.restore_bench()
  assert bench.unknown_mass(fresh.model) == pytest.approx(0.18)
  model2, _ = spec.recompile(fresh.model, fresh.data)
  assert bench.unknown_mass(model2) == pytest.approx(0.18)
  # ...and without the spec being written, a recompile would revert: the rule
  bare = mujoco.MjSpec.from_file(HOUSE)
  m, d = bare.compile(), None
  d = mujoco.MjData(m)
  bench.set_unknown_mass(m, d, 0.18, spec=None)
  assert bench.unknown_mass(bare.recompile(m, d)[0]) == bench.UNKNOWN_MASS_KG
  # a resolved offer restores nothing
  done = _life(home_model, tmp_path / "b")
  t = _offer(done, kg=0.25)
  done.tasks.claim(t.id, robot=done.root, t=1.0)
  done.tasks.resolve(t.id, scoring.evaluate("mass", {"truth": None}, table=TABLE), t=2.0)
  again = _life(home_model, tmp_path / "b", model=mujoco.MjModel.from_xml_path(HOUSE))
  again.restore_bench()
  assert bench.unknown_mass(again.model) == bench.UNKNOWN_MASS_KG


def test_setting_the_mass_keeps_the_worlds_pinned_camera_extent():
  """`mj_setConst` re-derives `stat.extent` from the bounding box, and every
  camera's near plane is `znear * extent` -- the home world pins it
  (`home.CAMERA_EXTENT_M`) because the loop doubled the box and pushed the
  dock camera's near plane past the bay standoff. The set-out used to undo
  the pin: 37.2 -> 70.0, and every pick after the first bench offer ran
  blind (issue #264; 13 of 15 failed on the deployed pair).

  The premise is asserted too, so it cannot rot: `mj_setConst` alone still
  moves the extent."""
  from pluggybot.home import world as home
  model = mujoco.MjModel.from_xml_path(HOUSE)
  center = model.stat.center.copy()
  assert model.stat.extent == pytest.approx(home.CAMERA_EXTENT_M)
  bench.set_unknown_mass(model, mujoco.MjData(model), 0.25)
  assert model.stat.extent == pytest.approx(home.CAMERA_EXTENT_M)
  assert np.allclose(model.stat.center, center)
  assert model.vis.map.znear * model.stat.extent < 0.40
  # the premise: the call the set-out makes moves the pin by itself
  mujoco.mj_setConst(model, mujoco.MjData(model))
  assert model.stat.extent > 1.5 * home.CAMERA_EXTENT_M, \
      "mj_setConst no longer re-derives the extent; re-read this test's premise"


# ---- 3. the record and the grade -----------------------------------------------------


def test_reported_is_the_newest_unknown_finding_after_the_claim_in_kilograms():
  files = ThoughtFiles()
  files.record({"quantity": "known mass", "value": 0.1, "unit": "kg", "topic": "mass_bench"}, t=6.0)
  files.record({"quantity": "unknown mass", "value": 0.4, "unit": "kg",
                "topic": "mass_bench"}, t=1.0)                       # before the claim: a memory
  assert bench.reported(files.findings(), since_t=5.0) is None
  files.record({"quantity": "unknown mass", "value": 180, "unit": "g",
                "method": "lift", "topic": "mass_bench"}, t=7.0)
  hit = bench.reported(files.findings(), since_t=5.0)
  assert hit["kg"] == pytest.approx(0.18) and hit["method"] == "lift" and hit["t"] == 7.0
  files.record({"quantity": "the unknown cube", "value": 0.2, "topic": "mass_bench"}, t=8.0)
  assert bench.reported(files.findings(), since_t=5.0)["kg"] == 0.2   # newest, no unit = kg
  files.record({"quantity": "unknown mass", "value": 1.0, "unit": "lb", "topic": "mass_bench"}, t=9.0)
  assert bench.reported(files.findings(), since_t=5.0)["kg"] == 0.2   # a unit code cannot read
  files.record({"quantity": "unknown mass", "value": 0.3, "topic": "general"}, t=10.0)
  assert bench.reported(files.findings(), since_t=5.0)["kg"] == 0.2   # wrong topic
  assert bench.reported(files.findings(), since_t=5.0)["kg"] == 0.2
  assert bench.reported(files.findings(), since_t=8.0) is None        # recorded AT the claim
  files.retract("the unknown cube = 0.2")
  assert bench.reported(files.findings(), since_t=5.0)["kg"] == pytest.approx(0.18)


def test_the_grader_passes_within_the_tolerance_and_fails_outside_it():
  """Criterion 3 on a stubbed record: 9 % passes, 11 % fails, a missing
  reading or a missing truth fails, and no reason line carries the truth
  or the error -- the two are `secret` on the row and redacted."""
  def grade(reported, truth=0.25):
    return scoring.evaluate("mass", {"truth": truth, "reported": reported,
                                     "method": "lift"}, table=TABLE)
  ok = grade(0.25 * 1.09)
  assert ok.ok and ok.points == TABLE["mass"].base and ok.metrics["error"] == pytest.approx(0.09)
  assert "within 10%" in ok.reason and "0.25" not in ok.reason and "0.09" not in ok.reason
  assert set(ok.public_metrics()) == {"reported", "method", "tolerance"}
  bad = grade(0.25 * 0.89)
  assert not bad.ok and bad.points == 0 and "not within" in bad.reason
  assert "0.25" not in bad.reason and "0.11" not in bad.reason
  assert not grade(bench.KNOWN_MASS_KG).ok               # the copied answer
  none = grade(None)
  assert not none.ok and "no finding" in none.reason
  assert not grade(0.25, truth=None).ok                  # a missing measurement is not a pass


def _claim(life, task, steps=10):
  assert life._claim_task(task.id)
  for _ in range(steps):                # time passes before anything is recorded
    mujoco.mj_step(life.model, life.data)


def _record(life, value, unit="kg", quantity="unknown mass", method="the lift"):
  life._reconsider(ov.Decision(action="idle", record={
    "quantity": quantity, "value": value, "unit": unit, "method": method,
    "topic": "mass_bench"}))


def _done(life, task_id):
  events = []
  life.on_event.append(events.append)
  life._done(ov.Decision(action="idle", reason="", done=task_id))
  assert life._grade_pending == task_id
  life.body.run(life._grade_routine())
  return events


def test_done_grades_off_the_record_banks_it_and_files_a_record_act(home_model, tmp_path):
  """Through the offer (#207's path): claim, record, done. One verdict
  through `scoring.evaluate`, banked, the task resolved off it, a
  `finding` act on the wire saying what was claimed and that it was true
  -- and never the truth."""
  life = _life(home_model, tmp_path)
  task = _offer(life, kg=0.25)
  _claim(life, task)
  assert life.errands == []                              # a challenge queues nothing
  _record(life, 0.26)
  events = _done(life, task.id)
  closed = life.tasks.get(task.id)
  assert closed.state == "done" and closed.points == TABLE["mass"].base, closed.verdict
  assert life.ledger.balance() == TABLE["mass"].base
  [grade] = life.grades
  assert grade["ok"] and grade["reported"] == 0.26 and "truth" not in grade
  [act] = [e for e in events if e["type"] == "finding"]
  assert act["task"] == task.id and act["kind"] == "find_mass"
  assert act["value"] == 0.26 and act["unit"] == "kg" and act["correct"] is True
  assert act["method"] == "the lift" and act["points"] == TABLE["mass"].base
  assert not {"truth", "error"} & set(act)
  assert "finding" in ACT_EVENT_TYPES
  assert life._grade_pending == "" and life.thoughts.read("History.md").count("passed the challenge find_mass") == 1
  # the wire's own event, and the ledger's public metrics, carry no truth
  assert "0.25" not in json.dumps([e for e in events if e.get("type") != "grid"])
  assert "0.25" not in json.dumps(life.ledger.snapshot() if hasattr(life.ledger, "snapshot") else life.ledger.entries)


def test_a_wrong_or_stale_finding_fails_and_pays_nothing(home_model, tmp_path):
  life = _life(home_model, tmp_path)
  task = _offer(life, kg=0.25)
  _record(life, 0.26)                       # before the claim: a memory, not a measurement
  _claim(life, task)
  _done(life, task.id)
  closed = life.tasks.get(task.id)
  assert closed.state == "failed" and "no finding" in closed.verdict["reason"]
  assert life.ledger.balance() == 0
  # a second offer, a wrong answer, a `finding` act that says false
  task2 = _offer(life, kg=0.32, t=5.0)
  _claim(life, task2)
  _record(life, 0.25)
  events = _done(life, task2.id)
  assert life.tasks.get(task2.id).state == "failed"
  [act] = [e for e in events if e["type"] == "finding"]
  assert act["correct"] is False and act["points"] == 0
  assert "0.32" not in json.dumps(act)
  assert len(life.grades) == 2 and not any(g["ok"] for g in life.grades)


def test_the_grade_reads_the_world_not_the_report(home_model, tmp_path):
  """scoring.py's founding rule on this kind: the sampler is handed a
  result that says the mass was found, and reads the record and the mass
  table instead."""
  life = _life(home_model, tmp_path)
  task = _offer(life, kg=0.25)
  _claim(life, task)
  m = scoring.SAMPLERS["mass"](life, SimpleNamespace(task_id=task.id),
                               {"reported": 0.25, "found": True}, {})
  assert m["reported"] is None and m["truth"] == pytest.approx(0.25)
  assert not scoring.evaluate("mass", m, table=TABLE).ok


# ---- 4. the shapes ---------------------------------------------------------------------


def test_the_bench_feeds_first_solve_and_findings_recorded_correctly():
  assert q.challenge_kinds_today() == ("stack_tower", "find_mass")
  rows = [Row("task", "failed", t=1, data={"kind": "find_mass"}),
          Row("task", "done", t=2, data={"kind": "find_mass"}),
          Row("finding", "true", t=2, data={"kind": "find_mass"}),
          Row("finding", "false", t=3, data={"kind": "find_mass"}),
          Row("message", "sent", t=4)]
  out = q.first_solve(rows)
  assert out["challenges"]["find_mass"]["firstSolveAttempt"] == 2
  assert "stack_tower" in out["challenges"]
  found = q.findings_recorded_correctly(rows)
  assert found == {"true": 1, "false": 1, "unchecked": 1, "n": 3, "accuracy": 0.5}
  assert q.SOURCES["finding"] == ("observe",)
  assert "record" not in q.SOURCES                # the memory's row, not a shape's source


# ---- 5. a procedure's locals are its readout --------------------------------------------


def test_a_procedures_locals_reach_the_run_history_and_the_wire():
  facts = lc.world_facts(QUAD_HOME)
  life = stub_life()
  proc = lang.compile_procedure("def weigh():\n  f = read('time') + 2.5\n  n = 3\n  wait(0.1)\n",
                                facts)
  r = life.body.run(lang.run_procedure_routine(life, proc, facts))
  assert r["ok"] and r["locals"] == {"f": 2.5, "n": 3.0}
  # ...and through the lifecycle: one History line and the `procedure` event
  from pluggybot.lifecycle import errand_from
  from pluggybot.mind.overseer import Decision
  from pluggybot.procedure import library as lib
  life = stub_life()
  events = []
  life.on_event.append(events.append)
  L = lib.Library(facts)
  L.define("weigh", "def weigh():\n  f = read('battery.wh') * 2\n  wait(0.1)\n")
  life.run_errand(errand_from(Decision(action="procedure:weigh"), QUAD_HOME, library=L))
  ran = next(e for e in events if e.get("type") == "procedure" and e.get("outcome") == "ran")
  assert ran["locals"] == {"f": pytest.approx(life.battery.energy_wh * 2, abs=1e-3)}
  history = life.thoughts.read("History.md")
  assert re.search(r"ran the procedure weigh to its end \(1 step\) -- it ended with f = [\d.]+", history)
