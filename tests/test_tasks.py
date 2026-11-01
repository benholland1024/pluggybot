"""Guards for the task model (issue #21).

The rules this file exists to hold down, in the order they matter:

  1. NOTHING PRICES ITS OWN WORK. A task names an evaluator and a reward-table
     row; what it PAYS is looked up from economy/rewards.json, and there is no
     field on a task that anybody can set to change it. This is issue #14's
     rule arriving from the direction a visitor and an LLM can both reach.
  2. A TASK IS CLOSED BY AN EVALUATOR. `resolve` takes a `scoring.Verdict` and
     nothing that merely looks like one.
  3. NOTHING SECRET REACHES THE WIRE OR THE MODEL.
  4. CHARGE PRIORITY IS NEVER OVERRIDDEN BY A CLAIMABLE TASK.

Everything else here -- persistence, expiry, the bounded board -- is
bookkeeping around those four.
"""

import dataclasses
import json

import pytest

from pluggybot import lifecycle as lc
from pluggybot.challenge import drawer
from pluggybot.economy import scoring
from pluggybot.mind.overseer import Menu
from pluggybot.economy.tasks import (
  KINDS, MAX_DESCRIPTION, MAX_OFFERED, Task, TaskBoard, kind_names,
)
from pluggybot.telemetry.protocol import TASK_SOURCES, TASK_STATES

TABLE = scoring.default_table()

#: A drawer pulled out and let go, as its grader measures it (#525), and
#: one left shut.
PULLED = {"mostM": 0.18, "fastestMS": 0.1, "tiltDeg": 2.0, "falls": 0,
          "outAtCallM": 0.17, "outM": 0.17, "holdS": drawer.HOLD_S, "elapsedS": 20.0,
          "limitS": 960.0, "touchedDuringHold": []}
SHUT = {**PULLED, "mostM": 0.0, "outAtCallM": 0.0, "outM": 0.0}
#: A chest opened and set down, its true catch beside: secret on its row.
LID_SET_DOWN = {"topDeg": 63.8, "downRadS": 0.51, "tiltDeg": 2.8, "falls": 0,
                "touchedDuringHold": [], "angleAtCallDeg": -0.2, "angleDeg": -0.2,
                "caught": True, "holdS": 5.0, "elapsedS": 40.0, "catchN": 3.77}


def board(**kw) -> TaskBoard:
  return TaskBoard(clock=lambda: "2026-08-23T00:00:00", **kw)


def offered(b: TaskBoard, **kw) -> Task:
  spec = {"kind": "open_drawer", "target": "drawer", "params": {"minutes": 16},
          "t": 0.0}
  spec.update(kw)
  task = b.offer(**spec)
  assert task is not None
  return task


def errand_kind(monkeypatch, **kw) -> str:
  """A job discharged by an ERRAND -- the procedure it carries (#58),
  graded by the drawer's row -- as no kind the humanoids are offered is
  (their hand jobs are procedures they write): what the board and the loop
  still do with one."""
  monkeypatch.setitem(KINDS, "errand_job", dataclasses.replace(
    KINDS["open_drawer"], name="errand_job", discharge="errand", draws=False, **kw))
  return "errand_job"


def waits(seconds: float = 1.0) -> dict:
  """A procedure of one wait, as a task carries it."""
  from pluggybot.procedure.steps import Program, Step
  return Program.single("job", [Step("wait", {"seconds": seconds})]).as_dict()


# ---- the hard rules ----------------------------------------------------------


def test_a_task_cannot_carry_its_own_payout():
  """The whole reason a task names an evaluator instead of a number.

  A visitor creates tasks (issue #23) and an overseer will propose them; if
  the payout travelled with the task, either could set it. It is looked up
  from the table on every read instead, so re-pricing a job is a JSON edit
  and pricing one is impossible.
  """
  b = board()
  task = offered(b)
  assert task.reward(TABLE) == {"task": "open_drawer", "tier": "auto",
                                "base": TABLE["open_drawer"].base,
                                "bonus": TABLE["open_drawer"].bonus}
  # There is no points-in field to pass, and the one that exists is an
  # OUTPUT: it is zero until an evaluator closes the task.
  assert task.points == 0
  assert "reward" not in dataclasses.asdict(task)
  fields = {f.name for f in dataclasses.fields(Task)}
  assert "base" not in fields and "bonus" not in fields


def test_repricing_the_table_reprices_every_offered_task(tmp_path):
  """`reward` is derived, not stored -- and this is how you can tell. (A
  hand job's row is in the challenge set, so that is the file edited.)"""
  spec = json.loads(scoring.CHALLENGES_PATH.read_text())
  spec["tasks"]["open_drawer"]["base"] = 999
  path = tmp_path / "challenges.json"
  path.write_text(json.dumps(spec))
  cheap = board()
  task = offered(cheap)
  assert task.reward(TABLE)["base"] != 999
  assert task.reward(scoring.RewardTable.load(challenges=path))["base"] == 999


def test_a_task_that_cannot_be_judged_cannot_be_created(monkeypatch):
  """The unscoreable task is not refused at payout time -- it cannot exist.

  Guarded by removing the evaluator rather than by inventing a kind, because
  the failure this prevents is a real one: somebody adds a kind, forgets the
  evaluator, and the robot does the job and is paid by whatever the table
  happened to say.
  """
  monkeypatch.setitem(KINDS, "orphan", dataclasses.replace(
    KINDS["open_drawer"], name="orphan", task="no_such_evaluator"))
  with pytest.raises(ValueError, match="no evaluator"):
    Task.create("orphan", "drawer", task_id="t_0001")


def test_only_an_evaluator_can_close_a_task():
  """`resolve` is `Ledger.award`'s lock, on the other end of the same job."""
  b = board()
  task = offered(b)
  b.claim(task.id, t=1.0)
  lookalike = {"task": "open_drawer", "ok": True, "points": 20, "reason": "trust me"}
  with pytest.raises(TypeError, match="scoring.Verdict"):
    b.resolve(task.id, lookalike, t=2.0)
  # ...and the real thing works, and pays what the evaluator said.
  verdict = scoring.evaluate("open_drawer", PULLED, table=TABLE)
  closed = b.resolve(task.id, verdict, t=2.0)
  assert closed.state == "done"
  assert closed.points == verdict.points


def test_a_failed_job_is_failed_not_expired():
  """Three terminal states, and they are not interchangeable: `failed` is a
  job the robot tried and got wrong, `expired` is one nobody attempted. The
  website draws them differently and a person reads them differently."""
  b = board()
  task = offered(b)
  b.claim(task.id, t=1.0)
  b.resolve(task.id, scoring.evaluate("open_drawer", SHUT, table=TABLE), t=2.0)
  assert b[task.id].state == "failed"
  assert b[task.id].points == 0

  lapsed = offered(b, ttl=10.0)
  assert b.expire_due(50.0) == [b[lapsed.id]]
  assert b[lapsed.id].state == "expired"
  assert b[lapsed.id].verdict is None      # nobody judged it; nobody tried


def test_a_secret_reaches_neither_the_wire_nor_the_model():
  """The honesty rule, made structural rather than promised: a hand job's
  secret is the draw of its prop's hidden dynamics (#559), and a robot
  shown it would know the drawer's friction before it touched it."""
  b = board()
  task = offered(b, secret={"draw": 4817})
  assert task.secret == {"draw": 4817}
  blobs = [json.dumps(task.as_dict()), json.dumps(task.snapshot(TABLE)),
           json.dumps(task.as_context(0.0, 9.0, TABLE)),
           json.dumps(b.snapshot()), json.dumps(b.context(0.0, 9.0))]
  for blob in blobs:
    assert "4817" not in blob and '"secret"' not in blob


# ---- claimability, including the energy gate ---------------------------------


def test_a_task_that_costs_more_than_the_pack_is_not_claimable():
  """The reserve is only checked BETWEEN errands, so a job the robot cannot
  afford has to be refused before it is started, not during.

  The numbers are the ones that matter here. An `open_drawer` is priced at
  7.082 Wh -- MEASURED, economy/energy.json -- so a robot with 8 Wh to
  spend can take it and one with 7 cannot, which is the whole behaviour.

  ⚠ The board is given the world's energy table, because that is what
  production does (issue #15) and the price is per WORLD and per TARGET.
  """
  from pluggybot.economy.energy import load as load_energy
  b = board(energy=load_energy(lc.HUMANOID_HOME))
  task = offered(b)
  assert task.estimate_wh == pytest.approx(7.082)
  assert task.claimable(0.0, pack_wh=8.0)
  assert not task.claimable(0.0, pack_wh=7.0)
  assert b.claim(task.id, t=0.0, pack_wh=7.0) is None
  assert b[task.id].state == "offered"    # still on offer for a fuller robot
  assert b.claim(task.id, t=0.0, pack_wh=8.0) is not None


def test_every_kind_is_affordable_off_a_full_pack():
  """Every kind IS affordable off a full pack, or it could never be taken
  at all and offering it would be a lie about what the robot can do --
  priced PER WORLD (`TaskBoard.estimate_for`, issue #15), the kinds with no
  row of their own at the dearest one the world measured; each in every
  world that has something for it to be about (#559: the humanoids' hand
  jobs are their house's alone)."""
  from pluggybot.economy.energy import load as load_energy
  for world in (lc.HUMANOID_HOME,):
    cap = lc.world_config(world)["battery_wh"]
    board = TaskBoard(energy=load_energy(world))
    here = lc.world_targets(world, lc.board_book(world), procedures=True,
                            robots=("Pluggy", "Rowan"))
    for name, kind in KINDS.items():
      if kind.target_kind not in here:
        continue
      priced = board.estimate_for(name)
      if priced is None:
        priced = kind.estimate_wh
      assert priced <= cap * 0.9, f"{world}/{name}: {priced} Wh"


def test_a_lapsed_offer_cannot_be_claimed_and_a_taken_one_cannot_be_retaken():
  b = board()
  task = offered(b, ttl=10.0)
  assert b.claim(task.id, t=50.0) is None       # past its deadline
  fresh = offered(b, ttl=None)
  assert b.claim(fresh.id, t=1.0) is not None
  assert b.claim(fresh.id, t=2.0) is None       # somebody already has it


def test_expiry_only_touches_offers_never_work_in_progress():
  """A deadline is how long an OFFER stands. A job lapsing while a robot is
  at work on it would take the job out from under the work."""
  b = board()
  task = offered(b, ttl=10.0)
  b.claim(task.id, t=1.0)
  b.start(task.id, t=2.0)
  assert b[task.id].state == "active"
  assert b.expire_due(999.0) == []
  assert b[task.id].state == "active"


def test_claimable_is_oldest_first_and_deterministic():
  """A scripted policy that is not reproducible makes every mission test a
  different test (`Math.random`-shaped bugs, twice paid for in this repo)."""
  b = board()
  a = offered(b, t=5.0)
  c = offered(b, kind="open_door", target="door", t=1.0)
  assert [t.id for t in b.claimable(10.0, 9.0)] == [c.id, a.id]


# ---- the board is bounded ----------------------------------------------------


def test_the_offer_queue_is_capped():
  b = board()
  made = [b.offer("open_drawer", "drawer") for _ in range(MAX_OFFERED + 3)]
  assert made.count(None) == 3
  assert len(b.offered()) == MAX_OFFERED


def test_resolved_tasks_age_out_but_open_ones_never_do():
  """Dropping a job the robot might still do is a different event from that
  job lapsing, and only one of the two has an honest name on the wire."""
  b = board(max_tasks=3, max_offered=99)
  keep = offered(b, t=0.0)
  for i in range(6):
    task = offered(b, t=float(i))
    b.claim(task.id, t=float(i))
    b.resolve(task.id, scoring.evaluate("open_drawer", PULLED, table=TABLE),
              t=float(i))
  assert keep.id in b                     # never dropped: still on offer
  assert len(b) <= 3
  assert b.dropped >= 3
  assert b.stats()["dropped"] == b.dropped


# ---- persistence -------------------------------------------------------------


def test_a_task_outlives_a_restart(tmp_path):
  path = tmp_path / "tasks.json"
  b = board(path=path)
  standing = offered(b, ttl=100.0)
  finished = offered(b, kind="open_door", target="door", t=1.0)
  b.claim(finished.id, t=1.0)
  b.resolve(finished.id, scoring.evaluate("open_drawer", PULLED, table=TABLE),
            t=2.0)

  back = board(path=path)
  assert back[standing.id].state == "offered"
  # the deadline is what was LEFT at the last save (t=2.0), from the new
  # mission's 0 -- a deadline in the old clock could outlive every later
  # mission (the served world, 2026-09-22)
  assert back[standing.id].deadline == standing.deadline - 2.0
  assert back[standing.id].description == standing.description
  assert back[finished.id].state == "done"
  assert back[finished.id].points > 0
  # The counter, not the length: an id is never reused after a trim.
  assert back.next_id() != standing.id


def _act_under_way(b: TaskBoard, t: float = 1.0) -> Task:
  """A job whose claim IS the act (#228), on disk as taken and under way:
  it happened or it did not, and a restart cannot tell which."""
  act = b.offer("take_points", "r2_pluggybot", params={"amount": 25}, t=0.0)
  b.claim(act.id, robot="pluggybot", t=t)
  b.start(act.id, t=t + 1.0)
  return act


def test_a_job_under_way_at_a_restart_is_still_its_robots(tmp_path, monkeypatch):
  """The robot that took it is still that robot (issue #345): a procedure
  job `active` on disk comes back active and its robot's (the procedure is
  its to run and say done), and an errand job comes back CLAIMED by it --
  its errand is queued again and marks it active when it starts
  (`HubLifecycle._resume_jobs`). Before #345 each came back `failed`, a job
  lost to the restart rather than to the work. An ACT still fails: it
  happened or it did not."""
  path = tmp_path / "tasks.json"
  b = board(path=path)
  task = offered(b)
  b.claim(task.id, robot="r2_pluggybot", t=1.0)
  b.start(task.id, t=2.0)
  errand = offered(b, kind=errand_kind(monkeypatch), params={"procedure": waits()})
  b.claim(errand.id, robot="r2_pluggybot", t=1.0)
  b.start(errand.id, t=2.0)
  act = _act_under_way(b)
  back = board(path=path)
  assert (back[task.id].state, back[task.id].claimed_by) == ("active", "r2_pluggybot")
  assert (back[errand.id].state, back[errand.id].claimed_by) == ("claimed", "r2_pluggybot")
  assert back[act.id].state == "failed"


def test_an_offer_keeps_the_life_it_had_left_across_a_restart(tmp_path):
  """A deadline is absolute sim time and every mission starts at 0. Measured
  on the deployed world (2026-09-22): an offer made at 3125 s of a 3600 s
  mission stood ~4700 s, and one made at 3606 s could never lapse -- and it
  held its target booked. On load the deadline is what was left at the last
  save, from 0; an older file with no `simTime` loads as it was."""
  path = tmp_path / "tasks.json"
  b = board(path=path)
  late = offered(b, ttl=720.0, t=3125.0)                     # deadline 3845
  last = offered(b, kind="open_door", target="door", ttl=720.0, t=3606.0)  # deadline 4326
  assert b.sim_t == 3606.0
  back = board(path=path)
  assert back[late.id].deadline == 239.0 and back[last.id].deadline == 720.0
  assert [t.id for t in back.expire_due(238.0)] == []
  assert [t.id for t in back.expire_due(239.0)] == [late.id]
  assert [t.id for t in back.expire_due(720.0)] == [last.id]
  # already overdue at the save: lapses at once, never a negative deadline
  b2 = board(path=tmp_path / "over.json")
  gone = offered(b2, ttl=10.0, t=0.0)
  offered(b2, kind="open_door", target="door", ttl=100.0, t=50.0)  # the save at 50
  assert board(path=tmp_path / "over.json")[gone.id].deadline == 0.0
  # an older build's file, no simTime: loaded as it was
  old = tmp_path / "old.json"
  was = offered(board(path=old), ttl=720.0, t=3125.0)
  doc = json.loads(old.read_text())
  del doc["simTime"]
  old.write_text(json.dumps(doc))
  assert board(path=old)[was.id].deadline == 3845.0


def test_a_task_interrupted_by_a_restart_is_said_on_the_wire_once(tmp_path):
  """`load` runs before any hook exists, so a job a restart failed left no
  row: on the observatory the bench's one claim read `active` for five
  hours (2026-09-22). `announce_interrupted` emits the `task_resolved`
  each would have been, after the hooks are attached, and once."""
  path = tmp_path / "tasks.json"
  b = board(path=path)
  task = _act_under_way(b)
  back = board(path=path)
  assert [t.id for t in back.interrupted] == [task.id]
  heard = []
  back.on_event.append(heard.append)
  [said] = back.announce_interrupted(t=0.5)
  assert said.id == task.id and said.state == "failed"
  [msg] = heard
  assert msg["type"] == "task_resolved" and msg["state"] == "failed"
  assert msg["id"] == task.id and msg["points"] == 0
  assert msg["verdict"]["reason"] == "interrupted by a restart"
  assert msg["task"]["kind"] == task.kind
  assert back.announce_interrupted() == [] and len(heard) == 1
  assert board(path=path).interrupted == []                 # failed on disk now


def test_a_claim_survives_a_restart_and_an_acts_does_not(tmp_path):
  """A claim is still its robot's (issue #345): the lifecycle queues its
  errand again (`_resume_jobs`), and gives it back only if that robot is
  not in the new world (`release_absent`) -- which is what closed the day
  the served pair held a claim with nobody behind it (2026-09-17). Its
  secret stays with it. An act's claim is cleared, the act not done; a
  resolved job keeps its claims."""
  path = tmp_path / "tasks.json"
  b = board(path=path)
  task = offered(b, ttl=100.0, secret={"draw": 9})
  act = b.offer("take_points", "r2_pluggybot", params={"amount": 25}, t=0.0)
  done = offered(b, kind="open_door", target="door", t=1.0)
  b.claim(task.id, robot="r2_pluggybot", t=1.0)
  b.claim(act.id, robot="pluggybot", t=1.0)
  b.claim(done.id, t=1.0)
  b.resolve(done.id, scoring.evaluate("open_drawer", PULLED, table=TABLE), t=2.0)

  back = board(path=path)
  assert (back[task.id].state, back[task.id].claimed_by) == ("claimed", "r2_pluggybot")
  assert back[task.id].secret == {"draw": 9}
  t = back[act.id]
  assert t.state == "offered" and t.claimed_by == "" and t.claimed_t is None
  assert t.claimable(50.0)
  assert back[done.id].state == "done"
  # ...and one whose robot is not in this world goes back on offer
  gone = back.release_absent({"pluggybot"})
  assert {g.id for g in gone} == {task.id}
  assert back[task.id].claimable(50.0)
  assert back[task.id].deadline == task.deadline - 2.0      # rebased, see above


def test_a_hand_edited_state_file_cannot_re_point_a_task_at_a_richer_row(tmp_path):
  """`task` is re-derived from the KIND on load. The state file is world
  state in a mounted volume, so it is not a trusted document."""
  path = tmp_path / "tasks.json"
  b = board(path=path)
  task = offered(b, kind="take_points", target="r2_pluggybot", params={"amount": 25})
  doc = json.loads(path.read_text())
  doc["tasks"][0]["task"] = "open_chest"  # a take pays less than a chest
  doc["tasks"][0]["estimateWh"] = 3.0
  path.write_text(json.dumps(doc))
  back = board(path=path)
  assert back[task.id].task == "take"
  assert back[task.id].estimate_wh == KINDS["take_points"].estimate_wh


def test_a_newer_state_version_refuses_to_load(tmp_path):
  path = tmp_path / "tasks.json"
  path.write_text(json.dumps({"version": 99, "tasks": []}))
  with pytest.raises(ValueError, match="task state version"):
    board(path=path)


def test_a_row_of_a_kind_this_build_does_not_know_is_dropped_at_load(tmp_path):
  """The deployed volume's board outlives the kinds a build knows: the
  quadrupeds' jobs went with them (#559) and their rows stay in the file.
  Kept, the next save raised `KeyError` on the row's kind -- the served
  process's first offer after the upgrade -- so a row of an unknown kind is
  dropped as the board loads, counted, and the rest of the board kept."""
  path = tmp_path / "tasks.json"
  b = board(path=path)
  task = offered(b, kind="take_points", target="r2_pluggybot", params={"amount": 25})
  doc = json.loads(path.read_text())
  old = {**doc["tasks"][0], "id": "t_old", "kind": "draw_figure", "task": "draw",
         "target": "whiteboard_a", "state": "expired"}
  doc["tasks"].append(old)
  path.write_text(json.dumps(doc))
  back = board(path=path)
  assert set(back.tasks) == {task.id} and back.dropped == doc.get("dropped", 0) + 1
  back.save()                                    # ...and the board saves again
  assert [t["kind"] for t in json.loads(path.read_text())["tasks"]] == ["take_points"]


# ---- the vocabularies are a two-repo contract --------------------------------


def test_every_kind_names_a_real_evaluator_and_a_real_reward_row():
  for name, spec in KINDS.items():
    assert spec.name == name
    assert spec.task in scoring.EVALUATORS, name
    assert spec.task in TABLE, name
    assert spec.target_kind in ("board", "zone", "module", "world",
                                "challenge", "robot", "cage", "bench",
                                "chest", "drawer", "door", "cube"), name
    # a job whose claim is the act (issue #228) moves no body and is the
    # one kind honestly priced at nothing
    assert spec.estimate_wh > 0.0 or spec.discharge == "act", name


def test_the_state_and_source_vocabularies_cover_what_the_board_produces():
  b = board()
  assert set(kind_names()) == set(KINDS)
  assert set(b.stats()) >= set(TASK_STATES)
  for source in TASK_SOURCES:
    assert offered(b, source=source).source == source
  with pytest.raises(ValueError, match="unknown task source"):
    Task.create("open_drawer", "drawer", "t_x", source="the_ceo")


def test_a_description_from_a_stranger_is_cleaned_and_capped():
  """A task can be created by a visitor (issue #23), so its text is untrusted
  on exactly the terms a visitor's message is -- and cleaned by the same
  function, so the two can never disagree."""
  task = Task.create("open_drawer", "drawer", "t_1",
                     description="pull\nthe\rdrawer\x00 " + "x" * 400)
  assert "\n" not in task.description and "\x00" not in task.description
  assert len(task.description) <= MAX_DESCRIPTION


# ---- the events ---------------------------------------------------------------


def test_the_lifecycle_of_a_task_is_on_the_wire():
  seen = []
  b = board()
  b.on_event.append(seen.append)
  task = offered(b, kind="open_chest", target="chest")
  b.claim(task.id, t=1.0)
  b.start(task.id, t=2.0)
  b.resolve(task.id, scoring.evaluate("open_chest", LID_SET_DOWN, table=TABLE), t=3.0)
  assert [m["type"] for m in seen] == [
    "task_offered", "task_claimed", "task_claimed", "task_resolved"]
  assert [m.get("state") for m in seen[1:]] == ["claimed", "active", "done"]
  assert seen[0]["task"]["reward"]["base"] == TABLE["open_chest"].base
  assert seen[-1]["points"] == b[task.id].points > 0
  # The verdict on the wire is the REDACTED one -- `public_metrics`, the same
  # object the ledger publishes: the chest's catch is secret on its row.
  assert "catchN" not in json.dumps(seen[-1])


# ---- the overseer's end -------------------------------------------------------


def test_taking_a_task_that_is_not_on_offer_is_a_malformed_answer():
  """Unlike `respond_to`, which is dropped: there the action survives without
  it, and here the id IS the action."""
  menu = Menu(boards=("whiteboard_a",))
  raw = {"action": "take_task", "reason": "", "board": "", "program": "",
         "zone": "", "note": "", "respond_to": "", "outcome": "", "reply": "",
         "task": "t_9999"}
  with pytest.raises(ValueError, match="not on offer"):
    menu.validate(raw, offered=("t_0001",))
  ok = menu.validate(raw, offered=("t_9999",))
  assert ok.action == "take_task" and ok.task == "t_9999"


def test_the_schema_does_not_change_when_the_board_does():
  """The task id is a free string, not an enum, for the reason `respond_to`
  is: the offers change every call, and a schema that changes every call
  misses the server-side compilation cache and buys nothing."""
  menu = Menu(boards=("whiteboard_a",))
  assert json.dumps(menu.schema()) == json.dumps(menu.schema())
  assert menu.schema()["properties"]["task"] == {"type": "string"}
  assert "take_task" in menu.schema()["properties"]["action"]["enum"]


def test_the_overseer_sees_what_a_job_pays_but_not_what_it_is_worth_deciding():
  """`as_context` answers the affordability question in code. "Can I pay for
  this" is arithmetic with a right answer, and nothing is gained by asking a
  language model to do it."""
  b = board()
  task = offered(b, ttl=100.0)
  ctx = b.context(now=10.0, pack_wh=0.2)
  assert ctx == []                        # unaffordable ones are not shown
  ctx = b.context(now=10.0, pack_wh=20.0)
  assert ctx[0]["id"] == task.id
  assert ctx[0]["claimable"] is True
  assert ctx[0]["pays"] == TABLE["open_drawer"].base + TABLE["open_drawer"].bonus
  assert ctx[0]["expiresInS"] == pytest.approx(90.0)


# ---- the errand it turns into -------------------------------------------------


def test_a_claimed_task_builds_an_errand_that_carries_its_id(monkeypatch):
  """One evaluation, two consumers: the verdict that pays for the finished
  errand is the verdict that closes the offer."""
  b = board()
  task = offered(b, kind=errand_kind(monkeypatch), params={"procedure": waits()})
  errand = lc.errand_for_task(task, lc.HUMANOID_HOME)
  assert errand is not None
  assert errand.task_id == task.id
  assert errand.task == KINDS[task.kind].task == "open_drawer"


def test_a_task_this_world_cannot_build_is_left_alone_not_failed(monkeypatch):
  """None rather than an exception: a task is untrusted input in the way an
  overseer decision is, and the loop's answer to "I cannot do that" is to
  leave the offer standing, so it lapses honestly."""
  b = board()
  kind = errand_kind(monkeypatch)
  task = offered(b, kind=kind)                           # no procedure to run
  assert lc.errand_for_task(task, lc.HUMANOID_HOME) is None
  assert b[task.id].state == "offered"
  # ...and one whose procedure names a verb this body has not (the
  # quadruped's fetch) -- while the same job with one it has is built
  from pluggybot.procedure.steps import Program, Step
  fetches = Program.single("job", [Step("fetch", {"tool": "module_pen"})]).as_dict()
  assert lc.errand_for_task(offered(b, kind=kind, params={"procedure": fetches}),
                            lc.HUMANOID_HOME) is None
  assert lc.errand_for_task(offered(b, kind=kind, params={"procedure": waits()}),
                            lc.HUMANOID_HOME) is not None
  # ...and a job whose procedure the robot writes builds none at all
  assert lc.errand_for_task(offered(b, params={"procedure": waits()}),
                            lc.HUMANOID_HOME) is None


# ---- the loop: charge priority, and the end-to-end flight ---------------------


def test_charge_priority_is_never_overridden_by_a_claimable_task(monkeypatch):
  """THE acceptance test for issue #21, and the same claim issue #15 pinned
  from the other side: the branch ORDER in `HubLifecycle.run()`.

  A robot that starts BELOW its reserve and a job it could otherwise take. It
  must charge first: `needs_charge` is checked before the task branch is ever
  reached, and a job offer is errand-tier work, never an override. A robot
  that took the job would run flat in the middle of it, because the reserve
  is only re-checked BETWEEN errands.

  ⚠ Two things had to be got right for this test to be able to FAIL, and both
  are worth knowing about the design.

  It asserts WHEN THE CLAIM HAPPENED, not when the errand ran. Claiming only
  queues an errand, and the errand queue already sits below `needs_charge` --
  so an inverted branch order still charges before it drives anywhere, and a
  test watching the errand passes either way. Measured: with the task
  branch moved above the charge branch, watching the errand still passed.

  And it uses a ZERO-COST job, which no real kind is, carrying the
  procedure that discharges it (the one errand a job builds on a body with
  no tool errand). The energy gate makes
  every ordinary task unclaimable exactly when charging is due -- `usable_wh`
  is energy ABOVE the reserve, and `needs_charge` is energy BELOW it, so the
  two conditions are the same one and the gate silently covers for the branch
  order. That is a good property and not the property under test; the cost is
  zeroed here so the ordering is the only thing left holding the line. The
  gate has its own test, one screen up.

  Shown to fail without the fix: move the `elif self._claim_next_task()`
  branch above `if self.needs_charge` in `HubLifecycle.run()`, and the job is
  taken on at t~0 with the pack below its reserve.
  """
  from test_body import stub_life
  b = board()
  task = offered(b, kind=errand_kind(monkeypatch, estimate_wh=0.0),
                 params={"procedure": waits()})
  assert task.claimable(0.0, pack_wh=0.0), "the energy gate still covers"
  life = stub_life(tasks=b)
  # Below the reserve at t=0: the cheapest state that puts the two branches
  # in direct conflict -- the robot needs to charge AND there is work waiting.
  life.battery.energy_wh = life.low_battery_wh * 0.6
  # The first moment the pack is legal again. Only charging can raise it, so
  # this is "the charge worked" expressed as a time the claim can be compared
  # against -- rather than as a state name, which is what made the first
  # version of this test unfalsifiable.
  legal_at: list[float] = []
  life.body.step_hooks.append(
    lambda: legal_at.append(float(life.data.time))
    if not legal_at and life.battery.energy_wh >= life.low_battery_wh else None)

  # ⚠ STOP ON THE CLAIM, NOT THE BUDGET (issue #54). This test asserts WHEN
  # the job was taken on, never that the errand behind it ran -- so the fetch
  # that follows the claim is pure cost. Measured 292.9 s.
  #
  # The predicate is the SUCCESS condition. With the branches inverted the
  # job is claimed at t~0 below the reserve, `legal_at` is still empty, the
  # hook does not fire, and the run goes the full 900 s and fails on the
  # ordering exactly as before.
  life.stop_when(lambda: bool(legal_at) and life.charge_cycles >= 1
                 and b[task.id].claimed_t is not None)
  r = life.run(lc.world_config(lc.HUMANOID_HOME)["start"], max_sim_time=900.0,
               explore_budget=10.0)

  assert r["charge_cycles"] >= 1, "a waiting job bricked the robot"
  assert legal_at, "the robot never got back above its reserve"
  # The claim must have happened, or the ordering was never put to the test.
  assert task.id in r["tasks_claimed"], "the job was never taken at all"
  claimed_t = b[task.id].claimed_t
  assert claimed_t is not None
  assert claimed_t >= legal_at[0], (
    f"the job was taken on at t={claimed_t:.1f} s, while the pack was still "
    f"below its reserve (legal again at t={legal_at[0]:.1f} s)")
  # ...and the job really was buildable throughout, so the ordering was
  # tested rather than dodged by an offer nothing could have accepted.
  assert lc.errand_for_task(task, lc.HUMANOID_HOME) is not None
