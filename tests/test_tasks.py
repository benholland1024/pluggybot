"""Guards for the task model (issue #21).

The rules this file exists to hold down, in the order they matter:

  1. NOTHING PRICES ITS OWN WORK. A task names an evaluator and a reward-table
     row; what it PAYS is looked up from hub/rewards.json, and there is no
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

from pluggybot.hub import lifecycle as lc
from pluggybot.hub import scoring
from pluggybot.hub.overseer import Menu, scripted
from pluggybot.hub.tasks import (
  KINDS, MAX_OFFERED, Task, TaskBoard, kind_names,
)
from pluggybot.telemetry.protocol import TASK_SOURCES, TASK_STATES

TABLE = scoring.default_table()

GOOD_DRAWING = {"board": "whiteboard_a", "strokes": 6, "strokesInked": 6,
                "formMm": 0.6, "inkedFraction": 0.99,
                "travelInkFraction": 0.0, "fill": 0.3}
NO_DRAWING = {"board": "whiteboard_a", "strokes": 6, "strokesInked": 0,
              "fill": 0.0}


def board(**kw) -> TaskBoard:
  return TaskBoard(clock=lambda: "2026-08-23T00:00:00", **kw)


def offered(b: TaskBoard, **kw) -> Task:
  spec = {"kind": "draw_figure", "target": "whiteboard_a",
          "params": {"program": "house"}, "t": 0.0}
  spec.update(kw)
  task = b.offer(**spec)
  assert task is not None
  return task


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
  assert task.reward(TABLE) == {"task": "draw", "tier": "auto",
                                "base": TABLE["draw"].base,
                                "bonus": TABLE["draw"].bonus}
  # There is no points-in field to pass, and the one that exists is an
  # OUTPUT: it is zero until an evaluator closes the task.
  assert task.points == 0
  assert "reward" not in dataclasses.asdict(task)
  fields = {f.name for f in dataclasses.fields(Task)}
  assert "base" not in fields and "bonus" not in fields


def test_repricing_the_table_reprices_every_offered_task(tmp_path):
  """`reward` is derived, not stored -- and this is how you can tell."""
  spec = json.loads(scoring.TABLE_PATH.read_text())
  spec["tasks"]["draw"]["base"] = 999
  path = tmp_path / "rewards.json"
  path.write_text(json.dumps(spec))
  cheap = board()
  task = offered(cheap)
  assert task.reward(TABLE)["base"] != 999
  assert task.reward(scoring.RewardTable.load(path))["base"] == 999


def test_a_task_that_cannot_be_judged_cannot_be_created(monkeypatch):
  """The unscoreable task is not refused at payout time -- it cannot exist.

  Guarded by removing the evaluator rather than by inventing a kind, because
  the failure this prevents is a real one: somebody adds a kind, forgets the
  evaluator, and the robot does the job and is paid by whatever the table
  happened to say.
  """
  monkeypatch.setitem(KINDS, "orphan", dataclasses.replace(
    KINDS["draw_figure"], name="orphan", task="no_such_evaluator"))
  with pytest.raises(ValueError, match="no evaluator"):
    Task.create("orphan", "whiteboard_a", task_id="t_0001")


def test_only_an_evaluator_can_close_a_task():
  """`resolve` is `Ledger.award`'s lock, on the other end of the same job."""
  b = board()
  task = offered(b)
  b.claim(task.id, t=1.0)
  lookalike = {"task": "draw", "ok": True, "points": 20, "reason": "trust me"}
  with pytest.raises(TypeError, match="scoring.Verdict"):
    b.resolve(task.id, lookalike, t=2.0)
  # ...and the real thing works, and pays what the evaluator said.
  verdict = scoring.evaluate("draw", GOOD_DRAWING, table=TABLE)
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
  b.resolve(task.id, scoring.evaluate("draw", NO_DRAWING, table=TABLE), t=2.0)
  assert b[task.id].state == "failed"
  assert b[task.id].points == 0

  lapsed = offered(b, ttl=10.0)
  assert b.expire_due(50.0) == [b[lapsed.id]]
  assert b[lapsed.id].state == "expired"
  assert b[lapsed.id].verdict is None      # nobody judged it; nobody tried


def test_a_secret_reaches_neither_the_wire_nor_the_model():
  """The honesty rule, made structural rather than promised.

  Nothing fills `secret` yet (issue #22 is the first kind that needs one), so
  this test is what stops the slot quietly becoming publishable before it is
  ever used.
  """
  b = board()
  task = offered(b, secret={"answer": "5"})
  assert task.secret == {"answer": "5"}
  blobs = [json.dumps(task.as_dict()), json.dumps(task.snapshot(TABLE)),
           json.dumps(task.as_context(0.0, 9.0, TABLE)),
           json.dumps(b.snapshot()), json.dumps(b.context(0.0, 9.0))]
  for blob in blobs:
    assert "answer" not in blob and '"5"' not in blob


# ---- claimability, including the energy gate ---------------------------------


def test_a_task_that_costs_more_than_the_pack_is_not_claimable():
  """The reserve is only checked BETWEEN errands, so a job the robot cannot
  afford has to be refused before it is started, not during.

  The numbers are the ones that matter here. `draw_figure` is estimated at
  0.93 Wh, MEASURED off the committed home recording, against a 1.100 Wh cell
  that charges to 90 % -- so a freshly-charged robot can just take it and a
  half-empty one cannot, which is the whole behaviour. The first version of
  this table guessed 0.35 Wh and the fixture recorded a robot dying mid-
  stroke with the pen still on the fork.
  """
  b = board()
  task = offered(b)                       # draw_figure: 0.93 Wh, measured
  assert task.claimable(0.0, pack_wh=0.99)      # home, just after a charge
  assert not task.claimable(0.0, pack_wh=0.55)  # home, at half
  assert b.claim(task.id, t=0.0, pack_wh=0.55) is None
  assert b[task.id].state == "offered"    # still on offer for a fuller robot
  assert b.claim(task.id, t=0.0, pack_wh=0.99) is not None


def test_the_energy_gate_is_measured_against_the_whole_pack():
  """⚠ Not against the energy ABOVE the reserve, and the difference is the
  difference between a gate and a wall.

  An errand is allowed to spend into the reserve -- that is what the reserve
  IS, a return-trip margin -- and measured off the recordings one errand
  costs roughly one full pack in both worlds, while the energy above the
  reserve is 0.28 Wh (room_hub) and 0.44 Wh (home). Comparing against that
  would refuse every job in every world forever, which reads from outside
  exactly like a task system nobody wired up.
  """
  headroom = {"room_hub": 0.700 * 0.9 - 0.350, "home": 1.100 * 0.9 - 0.550}
  cheapest = min(k.estimate_wh for k in KINDS.values())
  for world, above_reserve in headroom.items():
    assert above_reserve < cheapest, (
      f"{world}: {above_reserve:.2f} Wh above the reserve would now afford "
      f"the cheapest job ({cheapest} Wh) -- re-read this test, the premise "
      "it pins has changed")
  # ...and every kind IS affordable off a full pack, or it could never be
  # taken at all and offering it would be a lie about what the robot can do.
  for world, cap in (("room_hub", 0.700), ("home", 1.100)):
    for name, kind in KINDS.items():
      if world == "room_hub" and kind.target_kind in ("board", "zone"):
        continue                          # room_hub has neither
      assert kind.estimate_wh <= cap * 0.9, f"{world}/{name}"


def test_a_lapsed_offer_cannot_be_claimed_and_a_taken_one_cannot_be_retaken():
  b = board()
  task = offered(b, ttl=10.0)
  assert b.claim(task.id, t=50.0) is None       # past its deadline
  fresh = offered(b, ttl=None)
  assert b.claim(fresh.id, t=1.0) is not None
  assert b.claim(fresh.id, t=2.0) is None       # somebody already has it


def test_expiry_only_touches_offers_never_work_in_progress():
  """A deadline is how long an OFFER stands. Abandoning a claimed job
  mid-errand would leave a module on the fork, which is the one failure the
  swap stack spent two issues learning to avoid."""
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
  c = offered(b, target="whiteboard_b", t=1.0)
  assert [t.id for t in b.claimable(10.0, 9.0)] == [c.id, a.id]


# ---- the board is bounded ----------------------------------------------------


def test_the_offer_queue_is_capped():
  b = board()
  made = [b.offer("draw_figure", "whiteboard_a") for _ in range(MAX_OFFERED + 3)]
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
    b.resolve(task.id, scoring.evaluate("draw", GOOD_DRAWING, table=TABLE),
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
  finished = offered(b, target="whiteboard_b", t=1.0)
  b.claim(finished.id, t=1.0)
  b.resolve(finished.id, scoring.evaluate("draw", GOOD_DRAWING, table=TABLE),
            t=2.0)

  back = board(path=path)
  assert back[standing.id].state == "offered"
  assert back[standing.id].deadline == standing.deadline
  assert back[standing.id].description == standing.description
  assert back[finished.id].state == "done"
  assert back[finished.id].points > 0
  # The counter, not the length: an id is never reused after a trim.
  assert back.next_id() != standing.id


def test_a_task_interrupted_by_a_restart_comes_back_failed(tmp_path):
  """`active` on disk means the robot doing it no longer exists. Coming back
  `active` would be a marker that never resolves; coming back `expired` would
  claim nobody took the offer, and somebody did."""
  path = tmp_path / "tasks.json"
  b = board(path=path)
  task = offered(b)
  b.claim(task.id, t=1.0)
  b.start(task.id, t=2.0)
  assert board(path=path)[task.id].state == "failed"


def test_a_hand_edited_state_file_cannot_re_point_a_task_at_a_richer_row(tmp_path):
  """`task` is re-derived from the KIND on load. The state file is world
  state in a mounted volume, so it is not a trusted document."""
  path = tmp_path / "tasks.json"
  b = board(path=path)
  task = offered(b, kind="fetch_module", target="module_lcd")
  doc = json.loads(path.read_text())
  doc["tasks"][0]["task"] = "draw"        # carry pays less than draw
  doc["tasks"][0]["estimateWh"] = 0.0
  path.write_text(json.dumps(doc))
  back = board(path=path)
  assert back[task.id].task == "carry"
  assert back[task.id].estimate_wh == KINDS["fetch_module"].estimate_wh


def test_a_newer_state_version_refuses_to_load(tmp_path):
  path = tmp_path / "tasks.json"
  path.write_text(json.dumps({"version": 99, "tasks": []}))
  with pytest.raises(ValueError, match="task state version"):
    board(path=path)


# ---- the vocabularies are a two-repo contract --------------------------------


def test_every_kind_names_a_real_evaluator_and_a_real_reward_row():
  for name, spec in KINDS.items():
    assert spec.name == name
    assert spec.task in scoring.EVALUATORS, name
    assert spec.task in TABLE, name
    assert spec.target_kind in ("board", "zone", "module"), name
    assert spec.estimate_wh > 0.0, name


def test_the_state_and_source_vocabularies_cover_what_the_board_produces():
  b = board()
  assert set(kind_names()) == set(KINDS)
  assert set(b.stats()) >= set(TASK_STATES)
  for source in TASK_SOURCES:
    assert offered(b, source=source).source == source
  with pytest.raises(ValueError, match="unknown task source"):
    Task.create("draw_figure", "whiteboard_a", "t_x", source="the_ceo")


def test_a_description_from_a_stranger_is_cleaned_and_capped():
  """A task can be created by a visitor (issue #23), so its text is untrusted
  on exactly the terms a suggestion is -- and cleaned by the same function,
  so the two can never disagree."""
  task = Task.create("draw_figure", "whiteboard_a", "t_1",
                     description="draw\na\rhouse\x00 " + "x" * 400)
  assert "\n" not in task.description and "\x00" not in task.description
  assert len(task.description) <= 280


# ---- the events ---------------------------------------------------------------


def test_the_lifecycle_of_a_task_is_on_the_wire():
  seen = []
  b = board()
  b.on_event.append(seen.append)
  task = offered(b)
  b.claim(task.id, t=1.0)
  b.start(task.id, t=2.0)
  b.resolve(task.id, scoring.evaluate("draw", GOOD_DRAWING, table=TABLE), t=3.0)
  assert [m["type"] for m in seen] == [
    "task_offered", "task_claimed", "task_claimed", "task_resolved"]
  assert [m.get("state") for m in seen[1:]] == ["claimed", "active", "done"]
  assert seen[0]["task"]["reward"]["base"] == TABLE["draw"].base
  assert seen[-1]["points"] == b[task.id].points
  # The verdict on the wire is the REDACTED one -- `public_metrics`, the same
  # object the ledger publishes.
  assert "truth" not in json.dumps(seen[-1])


# ---- the overseer's end -------------------------------------------------------


def test_the_scripted_policy_takes_a_job_before_inventing_one():
  """The fallback has to do a real day's work with the API down, and "somebody
  asked for this" outranks the rotation. Note what it does NOT do: it takes
  the OLDEST claimable offer, not the best-paying one, because a scripted
  policy that optimised the reward table would be a second scorer."""
  menu = Menu(boards=("whiteboard_a",), programs=("house",))
  offers = [{"id": "t_0002", "claimable": True, "pays": 2},
            {"id": "t_0003", "claimable": True, "pays": 99}]
  decision = scripted(menu, {"offeredTasks": offers}, "timeout")
  assert decision.action == "take_task"
  assert decision.task == "t_0002"
  assert decision.source == "fallback:timeout"
  # ...and with nothing claimable it is the rotation exactly as before.
  dull = [{"id": "t_0004", "claimable": False}]
  assert scripted(menu, {"offeredTasks": dull}, "timeout").action != "take_task"


def test_taking_a_task_that_is_not_on_offer_is_a_malformed_answer():
  """Unlike `respond_to`, which is dropped: there the action survives without
  it, and here the id IS the action."""
  menu = Menu(boards=("whiteboard_a",), programs=("house",))
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
  menu = Menu(boards=("whiteboard_a",), programs=("house",))
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
  ctx = b.context(now=10.0, pack_wh=2.0)
  assert ctx[0]["id"] == task.id
  assert ctx[0]["claimable"] is True
  assert ctx[0]["pays"] == TABLE["draw"].base + TABLE["draw"].bonus
  assert ctx[0]["expiresInS"] == pytest.approx(90.0)


# ---- the errand it turns into -------------------------------------------------


def test_a_claimed_task_builds_an_errand_that_carries_its_id():
  """One evaluation, two consumers: the verdict that pays for the finished
  errand is the verdict that closes the offer."""
  book = lc.board_book("home")
  b = board()
  task = offered(b, target=next(iter(book.names)))
  errand = lc.errand_for_task(task, "home", book)
  assert errand is not None
  assert errand.task_id == task.id
  assert errand.task == KINDS[task.kind].task


def test_a_task_this_world_cannot_build_is_left_alone_not_failed():
  """None rather than an exception: a task is untrusted input in the way an
  overseer decision is, and the loop's answer to "I cannot do that" is to
  leave the offer standing, so it lapses honestly."""
  b = board()
  task = offered(b, kind="count_plants", target="garden")
  assert lc.errand_for_task(task, "room_hub", None) is None
  assert b[task.id].state == "offered"


# ---- the loop: charge priority, and the end-to-end flight ---------------------


def _lifecycle(world: str, **kw):
  """The bits of the mission stack a loop-order test needs. Deliberately the
  same shape as `test_overseer.py`'s, because it is testing the same thing at
  the same seam -- which branch of `run()` wins."""
  import mujoco
  cfg = lc.world_config(world)
  model = mujoco.MjModel.from_xml_path(cfg["model"])
  data = mujoco.MjData(model)
  return lc.HubLifecycle(model, data, realtime=False, world=world,
                         battery_wh=cfg["battery_wh"], rack=cfg["rack"],
                         grid_bounds=cfg["grid_bounds"],
                         low_battery_wh=cfg["low_battery_wh"], **kw)


@pytest.mark.slow
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
  test watching the swap states passes either way. Measured: with the task
  branch moved above the charge branch, watching `SWAP_PICK` still passed.

  And it uses a ZERO-COST job, which no real kind is. The energy gate makes
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
  monkeypatch.setitem(KINDS, "fetch_module",
                      dataclasses.replace(KINDS["fetch_module"],
                                          estimate_wh=0.0))
  b = board()
  task = offered(b, kind="fetch_module", target="module_lcd")
  assert task.claimable(0.0, pack_wh=0.0), "the energy gate still covers"
  life = _lifecycle("room_hub", tasks=b, errand=False)
  # Below the reserve at t=0: the cheapest state that puts the two branches
  # in direct conflict -- the robot needs to charge AND there is work waiting.
  life.battery.energy_wh = life.low_battery_wh * 0.6
  # The first moment the pack is legal again. Only charging can raise it, so
  # this is "the charge worked" expressed as a time the claim can be compared
  # against -- rather than as a state name, which is what made the first
  # version of this test unfalsifiable.
  legal_at: list[float] = []
  life.mission.step_hooks.append(
    lambda: legal_at.append(float(life.data.time))
    if not legal_at and life.battery.energy_wh >= life.low_battery_wh else None)

  r = life.run(lc.world_config("room_hub")["start"], max_sim_time=200.0,
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
  assert lc.errand_for_task(task, "room_hub", None) is not None
