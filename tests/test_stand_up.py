"""A stand-up ends the errand it lands in, and the map hears it (issue #348).

Rowan, 2026-09-23 (#339): knocked over mid-pick, an unbounded loop inside
the errand outlived the death, the timer stood the robot up MID-ERRAND, and
the loop drove it into a wall until flat -- every life, thirteen of them.
#341 bounded that loop; this ends ANY routine a stand-up lands in, so the
next unbounded loop costs one life rather than every one after it. Ben's
decision (2026-09-24): the robot comes back out of errand mode, and
`stood_up` is an event its own map can act on. And since #572 it comes back
on no map of the house: what it mapped and found goes, as a true death's.

Every rule here is pinned without flying a death, on the stub body
(`tests/test_body.py`): a stubbed errand that never returns, a pack set to
zero, and a three-second timer.
"""

import ast
from pathlib import Path

import pytest

from pluggybot import lifecycle as lc
from pluggybot import tick
from pluggybot.body import StubBody
from pluggybot.economy.tasks import TaskBoard
from pluggybot.lifecycle import (DEATH_ENDED, HUMANOID_HOME, MAP_GONE, STOOD_UP, HubLifecycle,
                                 board_book, world_config)
from pluggybot.mind import events as ev
from pluggybot.mind import overseer as ov
from pluggybot.mind.inbox import Inbox
from pluggybot.mind.overseer import Menu, Overseer
from pluggybot.mind.thoughts import HISTORY
from pluggybot.mission.errand import programmed_errand
from pluggybot.perception.tags import PLATE_TAG_IDS
from pluggybot.procedure.steps import Program, Step
from pluggybot.robot import SECOND

from test_body import stub_life  # noqa: I001 -- tests/ is on sys.path
from test_overseer import FakeClient


#: test_auto_restart.py's timer and for its reason (issue #158): every claim
#: here is about ORDER -- closed at the stand-up, not after it -- and none
#: depends on the delay's size.
TIMER_S = 3.0
PAST_S = TIMER_S + 1.0


@pytest.fixture(scope="module")
def menu():
  return Menu.for_world(HUMANOID_HOME, board_book(HUMANOID_HOME))


def _life(start=None, body=None, **kw) -> HubLifecycle:
  """A mortal robot on the short timer, standing at its start pose."""
  kw.setdefault("mortal", True)
  kw.setdefault("restart_after_s", TIMER_S)
  life = stub_life(body=body, **kw)
  start = start or world_config(HUMANOID_HOME)["start"]
  life.body.start_at(*start)
  life.home_pose = tuple(start)
  life.survival_since = float(life.data.time)
  return life


def _hold(life, seconds: float) -> None:
  life.body.run(life.body.hold_routine(seconds))


def _lap():
  """A procedure's errand, as the robot's own `procedure:<name>` builds it
  (`lifecycle.errand_from`): no job behind it."""
  return programmed_errand(Program.single("lap", [Step("wait", {"seconds": 1.0})]),
                           task="program", name="procedure")


def _drawer_job(board, life):
  """A hand job (#559), claimed: the robot writes and runs the procedure
  that does it, and says `done`."""
  task = board.offer("open_drawer", "drawer", t=0.0)
  board.claim(task.id, robot=life.root, t=0.0)
  return task


def _mapped(menu, *rows) -> Overseer:
  """A mind whose list is `rows`; nothing here asks it anything."""
  return Overseer(menu, origin="unseeded", client=FakeClient(),
                  event_map=ev.EventMap(tuple(rows)))


def _forever(life, errand, closed: list, kill: bool = True):
  """An errand that never returns -- #339's `refine_standoff`, minus the
  wall -- in errand mode as `run_errand_routine` puts a robot there, and
  (by default) dying in it. `closed` gets the sim time its `finally` ran."""
  life._in_errand = True
  life._errand_now = errand
  life._errand_name = errand.name
  life.state = "PROCEDURE"
  if kill:
    life.battery.energy_wh = 0.0
  try:
    while True:
      yield life.body.STILL
  finally:
    closed.append(float(life.data.time))


def _history(life) -> list[str]:
  return list(life.thoughts.volatile()[HISTORY])


# ---- the errand is closed, and only this robot's --------------------------


def test_a_stand_up_closes_the_errand_it_lands_in_and_no_other_robots_routine():
  """The issue's core, on the pair's one physics loop: robot 1's errand is
  closed the step its stand-up ends -- its `finally` runs, it is out of
  errand mode -- and robot 2's routine runs every step it asked for. An
  exception through `tick.run_many` would have been thrown into both."""
  cfg = world_config(HUMANOID_HOME)
  model, data = StubBody.world()
  a = _life(body=StubBody(model, data))
  b = _life(start=cfg["start2"], body=StubBody(model, data, handle=SECOND),
            robot_name="Rowan")
  a.peers, b.peers = [b], [a]
  closed: list[float] = []
  steps = round((TIMER_S + 2.0) / a.model.opt.timestep)

  def steady():
    for _ in range(steps):
      yield b.body.STILL
    return steps

  errand = _lap()
  out = tick.run_many([
    (a.body.stepper, a._until_stood_up_routine(_forever(a, errand, closed))),
    (b.body.stepper, steady())])

  assert [r["auto"] for r in a.resets] == [True], "the timer stood robot 1 up"
  assert out[0] is STOOD_UP
  #  ...closed the step the stand-up ended, which is when the survival clock
  #  restarts, not at some later look
  assert closed == [pytest.approx(a.survival_since, abs=2 * a.model.opt.timestep)]
  assert not a._in_errand and a._errand_now is None and a._errand_name == ""
  assert a.state == "EXPLORE", "out of errand mode"
  assert a.errand_results[-1]["error"] == DEATH_ENDED
  #  robot 2: every step it asked for, and returned rather than closed
  assert out[1] == steps
  assert b.resets == [] and b.dead is None


def test_the_day_loop_ends_an_errand_that_never_returns_at_the_stand_up():
  """The WIRING, on the real day loop: its spin stubbed, its one errand one
  that never returns and dies in it, and an explore budget of nothing so
  the day ends once the loop comes back round. Shown to fail with the
  errand branch's `_until_stood_up_routine` taken out: the errand is resumed
  after the stand-up (and would have driven on for ever)."""
  life = _life()
  try:
    closed: list[float] = []
    resumed: list[float] = []

    def errand_routine(errand):
      life._in_errand, life._errand_now = True, errand
      life.state = "PROCEDURE"
      life.battery.energy_wh = 0.0
      try:
        while True:
          yield life.body.STILL
          if life.resets:
            #  still driven after the stand-up: say so and stop, so the test
            #  FAILS here rather than hanging
            resumed.append(float(life.data.time))
            return {"errand": errand.name}
      finally:
        closed.append(float(life.data.time))

    life.run_errand_routine = errand_routine
    errand = _lap()
    life.errands = [errand]
    day = life.begin(world_config(HUMANOID_HOME)["start"], max_sim_time=60.0,
                     explore_budget=0.0)
    life.body.run(day)

    assert resumed == [], "the errand ran on after the stand-up"
    assert len(closed) == 1 and life.resets and life.resets[0]["auto"]
    assert life.errands == [] and life.state == "DONE"
    assert life.dead is None
    assert any(f"dying cut short {errand.name}" in ln for ln in _history(life))
  finally:
    life.body.close()


#: Every routine the loop drives that moves the body for long, and so can
#: still be running when the timer fires five minutes after a death.
LONG = {"run_errand_routine", "explore_routine", "_after_decision_routine",
        "go_charge_routine", "charge_routine"}


def test_every_long_manoeuvre_the_loop_drives_runs_under_the_rule():
  """The rest of the wiring, off the syntax tree: in the day loop, the
  charge trip and the decision branch, every call to a routine in `LONG` is
  the argument of `_until_stood_up_routine`. The flown test above pins the
  errand branch; this is the same one line for each of the others, and a
  flight per line would be the dear way to say it. Shown to fail with any
  one of them unwrapped."""
  tree = ast.parse(Path(lc.__file__).read_text())
  cls = next(n for n in tree.body
             if isinstance(n, ast.ClassDef) and n.name == "HubLifecycle")
  loops = {"_day_routine", "_charge_trip_routine", "_decide_routine",
           "_arbitrate_routine"}
  bare, seen = [], set()
  for fn in cls.body:
    if not (isinstance(fn, ast.FunctionDef) and fn.name in loops):
      continue
    for parent in ast.walk(fn):
      for child in ast.iter_child_nodes(parent):
        child._parent = parent  # type: ignore[attr-defined]
    for node in ast.walk(fn):
      if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
          and node.func.attr in LONG):
        seen.add(node.func.attr)
        up = node._parent  # type: ignore[attr-defined]
        if not (isinstance(up, ast.Call) and isinstance(up.func, ast.Attribute)
                and up.func.attr == "_until_stood_up_routine"):
          bare.append(f"{fn.name}:{node.lineno} {node.func.attr}")
  assert bare == [], "driven without the stand-up's rule"
  assert seen == LONG, "each is called where this looks"


def test_a_charge_trip_a_stand_up_ends_is_not_a_failed_dock():
  """`STOOD_UP` is falsy, so "did it get to the charger" reads no -- and a
  no there is `_strand`, a `stuck` death. A robot just stood up with a full
  pack must not be killed for a trip it never finished. Shown to fail
  without the `is STOOD_UP` check in `_charge_trip_routine`."""
  life = _life()
  try:
    closed: list[float] = []

    def go_charge():
      life.battery.energy_wh = 0.0
      try:
        while True:
          yield life.body.STILL
      finally:
        closed.append(float(life.data.time))

    life.go_charge_routine = go_charge
    life.body.run(life._charge_trip_routine())
    assert len(closed) == 1 and len(life.resets) == 1
    assert life.dead is None and not life.stranded, \
        "a stand-up is not a failed dock"
    assert [d["cause"] for d in life.deaths] == ["flat"]
  finally:
    life.body.close()


def test_a_stand_up_inside_a_blocking_stretch_puts_the_state_back():
  """A stretch the routine steps itself (`_ask_interrupt` blocks while its
  call flies) can hold the stand-up, and the routine then runs on to its
  next yield -- setting its own state -- before the wrapper can look. The
  state goes back to the stand-up's. Shown to fail without the reset."""
  life = _life()
  try:
    life.battery.energy_wh = 0.0
    _hold(life, 0.5)
    assert life.dead is not None

    def blocking():
      _hold(life, PAST_S)                         # the timer fires in here
      life.state = "PROCEDURE"                    # ...and the errand runs on
      yield life.body.STILL
      raise AssertionError("resumed after the stand-up")

    out = life.body.run(life._until_stood_up_routine(blocking()))
    assert out is STOOD_UP and life.resets
    assert life.state == "EXPLORE"
  finally:
    life.body.close()


def test_closing_is_safe_because_no_routine_yields_in_a_finally():
  """`_until_stood_up_routine` CLOSES the routine, which throws
  `GeneratorExit` at its innermost yield: a `finally` (or a handler
  catching that) which yielded again would make `close()` raise, and the
  stand-up would crash the day it was meant to save. None does; this keeps
  it so."""
  offenders = []
  for path in sorted(Path(lc.__file__).parent.rglob("*.py")):
    tree = ast.parse(path.read_text())
    for fn in ast.walk(tree):
      if not isinstance(fn, ast.FunctionDef):
        continue
      for node in ast.walk(fn):
        if not isinstance(node, ast.Try):
          continue
        blocks = [node.finalbody] + [
          h.body for h in node.handlers
          if h.type is None or any(n in ast.unparse(h.type)
                                   for n in ("BaseException", "GeneratorExit"))]
        if any(isinstance(n, (ast.Yield, ast.YieldFrom))
               for block in blocks for stmt in block for n in ast.walk(stmt)):
          offenders.append(f"{path.name}:{node.lineno} {fn.name}")
  assert offenders == []


# ---- the job, and what the robot and its map are told ---------------------


def test_a_hand_job_stays_the_robots_and_the_map_hears_the_stand_up_and_the_failure(menu):
  """A procedure's errand a stand-up ends is failed for the map -- it hears
  `stood_up` (who: the timer) beside `task_failed` -- and History says what
  dying cut short; the hand job the robot holds is NOT failed: its procedure
  is the robot's to run again, as across a restart (#345). (A job with an
  errand behind it was failed there; the quadruped's went with it, #559.)"""
  board = TaskBoard()
  said: list[dict] = []
  board.on_event.append(said.append)
  boss = _mapped(menu, ev.Row("stood_up", "idle", kind="timer"))
  life = _life(tasks=board, overseer=boss)
  try:
    task = _drawer_job(board, life)
    errand = _lap()
    errand.task_id = task.id                    # its procedure, run for the job
    life._next_events_check = float("inf")      # the map reads it below, once

    out = life.body.run(
      life._until_stood_up_routine(_forever(life, errand, [])))

    assert out is STOOD_UP
    assert board[task.id].state == "claimed" and board[task.id].claimed_by == life.root
    assert not [m for m in said if m["type"] == "task_resolved"]
    assert ("stood_up", "timer") in life._occurred
    assert ("task_failed", errand.name) in life._occurred
    life._next_events_check = 0.0
    life._events_step()
    assert life.queued_row == ev.Row("stood_up", "idle", kind="timer")
    cut = [ln for ln in _history(life) if f"dying cut short {errand.name}" in ln]
    assert cut and "is failed" not in cut[0]
  finally:
    life.body.close()


def test_a_voided_procedure_run_closes_as_aborted_with_no_count():
  """A run the site saw `validated` says how it ended: `aborted`, stopped
  `stood_up`. Its step count died with the runner, so it carries NONE --
  a consumer shows a missing count as unknown, and a zero would be a lie.
  Shown to fail without the emit in `_void_errand`."""
  from pluggybot.mission.errand import programmed_errand
  from pluggybot.procedure.steps import Program, Step
  from pluggybot.telemetry.protocol import PROCEDURE_OUTCOMES
  life = _life()
  try:
    seen: list[dict] = []
    life.on_event.append(seen.append)
    errand = programmed_errand(Program.single("lap", [Step("wait", {"seconds": 1.0})]))
    voided = life._until_stood_up_routine(_forever(life, errand, []))
    assert life.body.run(voided) is STOOD_UP
    [end] = [e for e in seen if e["type"] == "procedure"]
    assert (end["name"], end["outcome"], end["stopped"]) == ("lap", "aborted", "stood_up")
    assert end["outcome"] in PROCEDURE_OUTCOMES
    assert "completed" not in end and "total" not in end
  finally:
    life.body.close()


@pytest.mark.parametrize("who", ["timer", "admin"])
def test_who_stood_the_robot_up_is_the_kind_a_row_narrows_on(menu, who):
  """`stood_up` carries WHO as its kind: the world's timer, or an admin's
  `reset_robot` -- the `reset` event's `auto`, as a word. A row on the other
  one does not fire."""
  rows = (ev.Row("stood_up", "explore", kind="admin"),
          ev.Row("stood_up", "idle", kind="timer"))
  life = _life(inbox=Inbox(), overseer=_mapped(menu, *rows))
  try:
    life.battery.energy_wh = 0.0
    _hold(life, 0.5)
    assert life.dead is not None
    if who == "admin":
      life.inbox.offer({"type": "reset_robot", "id": "rr_348", "from": "ben"})
      life._visitor_step()
    else:
      _hold(life, PAST_S)
    assert life.dead is None and life.resets[-1]["auto"] is (who == "timer")
    life._next_events_check = 0.0
    life._events_step()
    assert life.queued_row == rows[0 if who == "admin" else 1]
  finally:
    life.body.close()


def test_a_stand_up_drops_what_the_map_queued_while_the_robot_lay_dead(menu):
  """A row that fired on the body that died -- a `battery_below` crossed on
  the way to zero -- is about a robot that is no longer there, and the slot
  it held failed the `stood_up` row `busy`. Shown to fail without the
  stand-up clearing `queued_row`."""
  rows = (ev.Row("battery_below", "charge", value=0.2), ev.Row("stood_up", ev.ASK))
  life = _life(overseer=_mapped(menu, *rows))
  try:
    life.battery.energy_wh = 0.0
    _hold(life, 0.5 + ev.MIN_PERIOD_S)
    assert life.dead is not None and life.queued_row == rows[0]
    _hold(life, PAST_S)
    assert life.dead is None
    life._next_events_check = 0.0
    life._events_step()
    assert life.queued_row == rows[1]
  finally:
    life.body.close()


def test_a_queued_row_that_is_still_news_waits_its_turn(menu):
  """Only a `battery_below` row is about what the stand-up undid (the pack
  it refills). A row queued for anything else -- here a completion, on a
  living robot an admin puts back -- is still true and is kept."""
  row = ev.Row("task_complete", ev.ASK)
  life = _life(inbox=Inbox(), overseer=_mapped(menu, row))
  try:
    life.queued_row = row
    life.inbox.offer({"type": "reset_robot", "id": "rr_q", "from": "ben"})
    life._visitor_step()
    assert life.resets and life.resets[-1]["auto"] is False
    assert life.queued_row == row
  finally:
    life.body.close()


def test_after_a_true_death_the_errand_is_not_the_new_robots_to_remember(menu):
  """The robot a true death starts inherits one line: that it is not the
  first (`_true_death`). The errand the old one died in still ends, but the
  new robot's History and fresh map hear nothing of it. Shown to fail
  without the check in `_void_errand`."""
  life = _life(overseer=_mapped(menu, ev.Row("task_failed", ev.ASK)))
  try:
    errand = _lap()
    life._in_errand, life._errand_now = True, errand
    life.deaths.append({"t": 1.0, "cause": "flat", "hearts": 0})
    life._void_errand()
    assert not life._in_errand and life._errand_now is None
    assert life.errand_results[-1]["error"] == DEATH_ENDED
    assert not any("dying cut short" in ln for ln in _history(life))
    assert ("task_failed", errand.name) not in life._occurred
  finally:
    life.body.close()


# ---- the vocabulary ----------------------------------------------------------


def test_stood_up_is_an_event_a_row_narrows_by_who(menu):
  """An eleventh event type, filtered like `task_complete` but on its own
  vocabulary: a token from another event's is refused, never dropped (a
  row that READS narrow must BEHAVE narrow). News that can wait, never an
  interrupt -- the errand it would interrupt has already ended."""
  assert "stood_up" in ev.EVENT_TYPES and "stood_up" in ev.DISCRETE_EVENTS
  assert "stood_up" not in ev.INTERRUPTING_EVENTS
  assert ev.kind_vocabulary("stood_up", menu) == ev.STOOD_UP_KINDS == ("timer", "admin")
  assert set(ev.STOOD_UP_KINDS) <= set(ev.kind_tokens(menu)), "the schema's enum"
  row = ev.row({"event": "stood_up", "action": ev.ASK, "kind": "admin",
                "value": 0.5}, menu)
  assert row == ev.Row("stood_up", ev.ASK, None, "admin")
  with pytest.raises(ValueError, match="unknown kind"):
    ev.row({"event": "stood_up", "action": ev.ASK, "kind": "offers"}, menu)


def test_the_rule_names_stood_up_and_who_and_shows_no_rule_for_it():
  """`EVENT_MAP_RULE` lists the event with its two kinds, in `MORTAL_RULE`'s
  words ("stood back up"), and demonstrates nothing: no worked row reacts
  to a stand-up, because what to do then is the answer the map is read for."""
  rule = ov.EVENT_MAP_RULE
  line = " ".join(rule.split("  stood_up", 1)[1].split("\n\n", 1)[0].split())
  assert "stood back up" in line and "ended there" in line
  assert "`timer`" in line and "`admin`" in line
  assert not any("stood_up" in ln for ln in rule.splitlines() if "->" in ln)
  assert "stood back up" in ov.MORTAL_RULE


# ---- on no map ---------------------------------------------------------------


@pytest.mark.parametrize("who", ["timer", "admin"])
def test_a_stand_up_forgets_the_map_and_the_places_found_in_it(who):
  """#572, Ben's call: a stand-up puts the belief right at the start, and a
  map laid round a wrong one pulls it back (the humanoids were 1-5 m off
  within an hour live, on maps their scans agreed with). So the robot comes
  back on no map -- its grid, its places and the explore that laid them
  gone, as a true death's -- after a death on the timer and from an admin's
  hand on a living robot alike, and History says so. Shown to fail without
  `_stand_up`'s `_forget_world`."""
  life = _life(inbox=Inbox())
  try:
    life.body.grid.grid[3:6, 3:6] = 2.0
    life.body.places.see(PLATE_TAG_IDS[0], 25.0, 4.83, 0.0, 0.0)
    life.floor_explored, life.blacklist = True, {(4, 4)}
    assert len(life.body.places) == 1
    if who == "timer":
      life.battery.energy_wh = 0.0
      _hold(life, 0.5)
      assert life.dead is not None
      _hold(life, PAST_S)
    else:
      life.inbox.offer({"type": "reset_robot", "id": "rr_572", "from": "ben"})
      life._visitor_step()
    assert life.dead is None and life.resets[-1]["auto"] is (who == "timer")
    assert life.body.forgot == 1 and not life.body.grid.grid.any()
    assert len(life.body.places) == 0
    assert not life.floor_explored and life.blacklist == set()
    said = "stood back up on my own" if who == "timer" else "reset by ben while still awake"
    assert any(said in line and MAP_GONE in line for line in _history(life))
  finally:
    life.body.close()


# ---- never onto another robot ------------------------------------------------


def test_a_stand_up_never_lands_on_another_robot():
  """#387: a stand-up is a warp, and a warp INTO another robot's body is an
  overlap the solver throws both out of. It takes the first commissioned
  start no robot's body is on (`up_pose`) -- the other's, with two -- and
  waits only while every one is taken. Stub bodies: the rule is where it
  puts the robot."""
  a_body, b_body = StubBody(), StubBody()
  a = stub_life(body=a_body, mortal=True, restart_after_s=TIMER_S)
  b = stub_life(body=b_body, robot_name="Rowan")
  a.peers, b.peers = [b], [a]
  a.home_pose, b.home_pose = (1.0, 1.0, 0.0), (4.0, 1.0, 0.0)
  a_body.x, a_body.y = 2.5, 3.0
  b_body.x, b_body.y = 1.3, 1.1                   # Rowan on the dead robot's start
  a.battery.energy_wh = 0.0
  a_body.run(a_body.hold_routine(PAST_S))
  assert a.dead is None, "stood up on time"
  assert (a_body.x, a_body.y) == (4.0, 1.0), "...on the start that was clear"
  assert any("Rowan's start pose" in line for line in a.log)
  # ...and with EVERY start taken, it waits, then goes home once it is clear
  b.home_pose = (1.5, 1.0, 0.0)
  a.battery.energy_wh = 0.0
  a_body.run(a_body.hold_routine(PAST_S))
  assert a.dead is not None, "no clear start: still down past the timer"
  b_body.x, b_body.y = 6.0, 6.0
  a_body.run(a_body.hold_routine(0.5))
  assert a.dead is None and (a_body.x, a_body.y) == (1.0, 1.0)
