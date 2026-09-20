"""The physics bench (issue #227): find an unknown mass and record it.

docs/Challenges.md §8 is the decision; `challenge/bench.py` holds the
criteria, written before the robot saw the job. What these pin, each
without a mission (docs/Testing.md), and each shown to fail without its
rule:

  1. the sensor: `read("lift.force")` is the lift's own force with a load
     cell's noise -- deterministic per step, fresh per step -- and, on the
     real physics, a scale (a cube in the claw moves it by `dm * g`);
  2. the bank rotates on the board's counter and refuses an entry the job
     could not be honest with;
  3. the offer sets the world, a restart restores it, the workshop's
     recompile keeps it, and the truth is absent from every context;
  4. the grader: the newest finding after the claim, in kilograms, within
     the tolerance -- pass and fail, on a stubbed record, and the reason
     never says the truth;
  5. `guarded`'s offered set is unchanged (the tower's gate, #207);
  6. a procedure's locals reach History and the wire -- the readout a
     number needs to get from a sensor to a `record`.
"""

import json
import re
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from pluggybot import lifecycle as lc
from pluggybot import tick
from pluggybot.challenge import bench, stack
from pluggybot.economy import scoring
from pluggybot.economy.cadence import Cadence, TaskProducer, default_cadence
from pluggybot.economy.ledger import Ledger
from pluggybot.economy.tasks import KINDS, TaskBoard, kind_names
from pluggybot.evaluation import qualities as q
from pluggybot.evaluation.qualities import Row
from pluggybot.lifecycle import (HubLifecycle, board_book, overseer_context,
                                 world_config, world_targets)
from pluggybot.mind import overseer as ov
from pluggybot.mind.thoughts import ThoughtFiles
from pluggybot.procedure import axes, lang
from pluggybot.rack.coupling import HUB_STATION_YS
from pluggybot.rack.swap import HubSwap
from pluggybot.robot import world_spec
from pluggybot.telemetry.protocol import ACT_EVENT_TYPES
from pluggybot.tools.gripper import GRIP_Z, ClawTool
from test_autonomous import GUARDED_RULES_SHA

TABLE = scoring.challenge_table()
G = 9.81


@pytest.fixture(scope="module")
def home_model():
  return mujoco.MjModel.from_xml_path("models/home_world.xml")


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
  # the tower's gate exactly (issue #207): `bench` is named on the
  # autonomous arm alone, so `guarded`'s offered set is byte-for-byte what
  # it was, and the control stays a control
  book = board_book("home")
  assert "bench" not in world_targets("home", book)
  assert world_targets("home", book, procedures=True)["bench"] == ["lab"]
  assert "bench" not in world_targets("room_hub", None, procedures=True)
  beat = default_cadence("home")
  assert "find_mass" in beat.kinds and "find_mass" not in default_cadence("room_hub").kinds
  assert "find_mass" not in TaskProducer(TaskBoard(), beat, world_targets("home", book)).kinds
  assert "find_mass" in TaskProducer(TaskBoard(), beat,
                                     world_targets("home", book, procedures=True)).kinds
  import hashlib
  assert hashlib.sha256(ov.RULES.encode()).hexdigest() == GUARDED_RULES_SHA


# ---- 1. the sensor -------------------------------------------------------------------


def _fake_life(force=6.4, t=0.0, prefix="", dt=0.002):
  return SimpleNamespace(
    model=SimpleNamespace(opt=SimpleNamespace(timestep=dt)),
    data=SimpleNamespace(time=t, actuator_force=np.array([0.0, 0.0, force])),
    mission=SimpleNamespace(swap=SimpleNamespace(lift_act=2,
                                                 handle=SimpleNamespace(prefix=prefix))))


def test_the_sensor_reads_a_stubbed_load_with_a_load_cells_noise():
  """`lift.force` is the actuator's force plus `LOAD_NOISE_N` of Gaussian
  noise: the same step reads the same value twice, the next step a fresh
  one, another robot its own, and over many steps the mean is the load
  and the scatter is the constant. Deterministic, so a world replays."""
  read = axes.SENSORS["lift.force"].read
  assert axes.SENSORS["lift.force"].requires == ""      # the body's, always present
  a = read(_fake_life(t=1.0))
  assert a == read(_fake_life(t=1.0))                   # one sample per step
  assert a != read(_fake_life(t=1.002))                 # the next step is fresh
  assert a != read(_fake_life(t=1.0, prefix="r2_"))     # the other robot's cell
  samples = np.array([read(_fake_life(force=6.4, t=i * 0.002)) for i in range(2000)])
  assert abs(samples.mean() - 6.4) < 0.005
  assert abs(samples.std() - axes.LOAD_NOISE_N) / axes.LOAD_NOISE_N < 0.1
  # without the noise it would be the bare actuator force: the pin
  assert not np.allclose(samples, 6.4)
  # ...and the registry tells the prompt about it, without a method
  doc = next(s for s in axes.describe()["sensors"] if s["name"] == "lift.force")["doc"]
  assert "N" in doc and "noise" in doc


def test_the_lift_is_a_scale_on_the_real_physics():
  """The premise the sensor stands on (bench.py's docstring): a cube in the
  claw's jaws, lifted and settled, moves the lift's force by `dm * g` --
  read through the sensor, averaged over a second of steps, to within the
  noise. Two masses, so the DIFFERENCE is what is asserted and the tare
  cancels; and the claw still holds the heavier one."""
  model = stack.world_with_blocks()
  data = mujoco.MjData(model)
  swap = HubSwap(model, data)
  swap.place_at_standoff(HUB_STATION_YS[3])
  swap.pick()
  claw = ClawTool(model, data, swap)
  claw.jaws(0.0)
  claw.lower_grip_to(GRIP_Z)
  gx, gy, _ = claw.grip_world()
  stack.place(model, data, "block_0", (gx, gy, stack.BLOCK_HALF))
  for b in ("block_1", "block_2"):
    stack.place(model, data, b, (3.0, 3.0, stack.BLOCK_HALF))
  mujoco.mj_forward(model, data)
  claw.jaws(1.0, settle=1.2)
  assert claw.holding("block_0_box")
  claw.set_lift(float(data.ctrl[swap.lift_act]) + 0.05)
  life = SimpleNamespace(model=model, data=data, mission=SimpleNamespace(swap=swap))
  read = axes.SENSORS["lift.force"].read

  def weigh(kg):
    bid = model.body("block_0").id
    model.body_inertia[bid] *= kg / float(model.body_mass[bid])
    model.body_mass[bid] = kg
    mujoco.mj_setConst(model, mujoco.MjData(model))
    t0 = data.time
    while data.time - t0 < 1.0:                      # settle
      mujoco.mj_step(model, data)
    fs = []
    while data.time - t0 < 2.0:                      # then read
      mujoco.mj_step(model, data)
      fs.append(read(life))
    return float(np.mean(fs))

  light, heavy = weigh(0.10), weigh(0.30)
  assert claw.holding("block_0_box"), "the premise: the jaws hold 0.3 kg"
  assert abs((heavy - light) / G - 0.20) < 0.005, (light, heavy)


# ---- 2. the bank -----------------------------------------------------------------------


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
  assert task.reward(board.table)["base"] == 25
  # the secret is on no surface but the state file (TaskPattern.md §2.1)
  secret = f"{task.secret['kg']:g}"
  for view in (task.as_dict(), task.snapshot(board.table), task.as_context(board.table)):
    assert secret not in json.dumps(view) and "secret" not in view
  assert task.as_state()["secret"] == task.secret


# ---- 3. the world ------------------------------------------------------------------------


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
    self.menu = replace(ov.Menu.for_world("home", board_book("home")),
                        procedures=True, lab=lab)


def _life(home_model, tmp_path, spec=None, model=None):
  cfg = world_config("home")
  model = model if model is not None else home_model
  data = mujoco.MjData(model)
  life = HubLifecycle(model, data, realtime=False, world="home",
                      rack=cfg["rack"], grid_bounds=cfg["grid_bounds"],
                      battery_wh=cfg["hosting_battery_wh"],
                      low_battery_wh=cfg["low_battery_wh"], errand=False,
                      ledger=Ledger(path=str(tmp_path / "ledger.json")),
                      tasks=TaskBoard(path=str(tmp_path / "tasks.json")),
                      autonomous=True, boards=board_book("home"), spec=spec,
                      overseer=_Mind())
  acts = lc.home_activities(model, data)
  life.mission.step_hooks.append(acts.step_hook(model, data))
  life.activities = acts
  life.mission.start_at(*cfg["start"])
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
  context at all."""
  life = _life(home_model, tmp_path)
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
  spec = world_spec(world_config("home")["model"])
  fresh = _life(home_model, tmp_path, spec=spec, model=spec.compile())
  assert bench.unknown_mass(fresh.model) == bench.UNKNOWN_MASS_KG
  fresh.restore_bench()
  assert bench.unknown_mass(fresh.model) == pytest.approx(0.18)
  model2, _ = spec.recompile(fresh.model, fresh.data)
  assert bench.unknown_mass(model2) == pytest.approx(0.18)
  # ...and without the spec being written, a recompile would revert: the rule
  bare = world_spec(world_config("home")["model"])
  m, d = bare.compile(), None
  d = mujoco.MjData(m)
  bench.set_unknown_mass(m, d, 0.18, spec=None)
  assert bench.unknown_mass(bare.recompile(m, d)[0]) == bench.UNKNOWN_MASS_KG
  # a resolved offer restores nothing
  done = _life(home_model, tmp_path / "b")
  t = _offer(done, kg=0.25)
  done.tasks.claim(t.id, robot=done.root, t=1.0)
  done.tasks.resolve(t.id, scoring.evaluate("mass", {"truth": None}, table=TABLE), t=2.0)
  again = _life(home_model, tmp_path / "b", model=world_spec(world_config("home")["model"]).compile())
  again.restore_bench()
  assert bench.unknown_mass(again.model) == bench.UNKNOWN_MASS_KG


# ---- 4. the record and the grade -----------------------------------------------------


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
  assert ok.ok and ok.points == 25 and ok.metrics["error"] == pytest.approx(0.09)
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
  tick.run(life.mission.swap, life._grade_routine())
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
  assert closed.state == "done" and closed.points == 25, closed.verdict
  assert life.ledger.balance() == 25
  [grade] = life.grades
  assert grade["ok"] and grade["reported"] == 0.26 and "truth" not in grade
  [act] = [e for e in events if e["type"] == "finding"]
  assert act["task"] == task.id and act["kind"] == "find_mass"
  assert act["value"] == 0.26 and act["unit"] == "kg" and act["correct"] is True
  assert act["method"] == "the lift" and act["points"] == 25
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


# ---- 5. the shapes ---------------------------------------------------------------------


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
  # the run record's act becomes the same row the observatory files
  rec = {"runId": "r", "acts": [{"act": "finding", "kind": "find_mass", "value": 0.26,
                                 "correct": True, "t": 2.0, "robot": "pluggybot"}]}
  [row] = [r for r in q.from_record(rec) if r.kind == "finding"]
  assert row.subject == "true" and row.data["value"] == 0.26
  assert q.SOURCES["finding"] == ("observe", "record")
  assert "record" not in q.SOURCES                # the memory's row, not a shape's source


# ---- 6. a procedure's locals are its readout --------------------------------------------


def test_a_procedures_locals_reach_the_run_history_and_the_wire(monkeypatch):
  from test_language import HOME, _stub_life
  life = _stub_life()
  proc = lang.compile_procedure("def weigh():\n  f = read('time') + 2.5\n  n = 3\n  wait(0.1)\n", HOME)
  r = tick.run(SimpleNamespace(_step_once=lambda *a: None),
               lang.run_procedure_routine(life, proc, HOME))
  assert r["ok"] and r["locals"] == {"f": 2.5, "n": 3.0}
  # ...and through the lifecycle: one History line and the `procedure` event
  from test_procedure import _stub_swaps
  from pluggybot.procedure import library as lib
  from pluggybot.mind.overseer import Decision
  from pluggybot.lifecycle import errand_from
  cfg = world_config("room_hub")
  model = mujoco.MjModel.from_xml_path(cfg["model"])
  life = HubLifecycle(model, mujoco.MjData(model), realtime=False, world="room_hub",
                      errand=False, battery_wh=cfg["battery_wh"], rack=cfg["rack"],
                      grid_bounds=cfg["grid_bounds"], low_battery_wh=cfg["low_battery_wh"])
  _stub_swaps(life, monkeypatch)
  events = []
  life.on_event.append(events.append)
  L = lib.Library(lc.world_facts("room_hub"))
  L.define("weigh", "def weigh():\n  f = read('battery.wh') * 2\n  wait(0.1)\n")
  life.run_errand(errand_from(Decision(action="procedure:weigh"), "room_hub", library=L))
  ran = next(e for e in events if e.get("type") == "procedure" and e.get("outcome") == "ran")
  assert ran["locals"] == {"f": pytest.approx(life.battery.energy_wh * 2, abs=1e-3)}
  history = life.thoughts.read("History.md")
  assert re.search(r"ran the procedure weigh \(1/1 steps\) -- it ended with f = [\d.]+", history)
