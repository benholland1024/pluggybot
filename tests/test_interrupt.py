"""The mid-errand interrupt (issue #116): the first interruptibility the
mission loop has ever had.

An errand was uninterruptible until this. `run_errand` checked nothing, so a
decision taken at 15 % was IRREVOCABLE and self-preservation could only be
measured at errand boundaries -- there was no moment at which the robot could
notice it had got it wrong. That is the poor instrument this replaces.

The POLICY half is #127's: a `battery_below` / `points_below` row of the
agent's own event map, validated by the same function as everything else.
What is here is the MECHANICS, which is what the issue's own last comment
calls its substance:

  `test_an_aborted_procedure_hangs_its_tool_back_even_with_the_endpoint_down`
      ⚠ ABORT MEANS STOW, NEVER DROP -- the one rule that cannot be got
      wrong -- with a dead client, because a row naming an action asks
      nobody and so still works when the endpoint is down
  `test_an_interrupt_nobody_answers_aborts_rather_than_carries_on`
      the one place in this design where failing SAFE is right
  `test_a_world_with_no_map_cannot_be_interrupted_at_all`
      no map, no hazard row, so nothing reaches the robot mid-job

The loop's half is a day on the stub body (`tests/test_body.py`); that the
quadruped's stow hangs a tool on its bay is the rack's own flight
(`tests/test_quad_rack.py`).

Nothing here touches the network: the client is the injected seam, as in
tests/test_overseer.py, whose fakes these reuse.
"""

import pytest

from pluggybot.evaluation.arms import arm_flags
from pluggybot.lifecycle import QUAD_HOME, THINK_SLICE_S, board_book, world_config, world_facts
from pluggybot.mind import events as ev
from pluggybot.mind.overseer import Menu, Overseer
from pluggybot.mission.errand import programmed_errand
from pluggybot.procedure import lang

from test_body import stub_life  # noqa: I001 -- tests/ is on sys.path
from test_overseer import FakeClient


@pytest.fixture(scope="module")
def menu():
  return Menu.for_world(QUAD_HOME, board_book(QUAD_HOME))


def _state(fraction: float) -> dict:
  """A synthetic decision state: what the interrupt's question is shown."""
  return {"simTimeS": 100.0,
          "battery": {"fraction": fraction, "wh": 8.0 * fraction,
                      "spendableWh": max(0.0, 8.0 * fraction - 0.9)},
          "offeredTasks": [], "affordableActions": ["explore", "charge"],
          "possibleActions": ["explore", "charge"],
          "tasksThisMission": [], "decisions": 0}


def make(menu, *answers, origin="seeded", **kw) -> Overseer:
  kw.setdefault("client", FakeClient(*answers))
  return Overseer(menu, origin=origin, **kw)


# ---- which rows may interrupt at all ----------------------------------------


def test_only_the_two_hazards_interrupt_and_the_rest_wait():
  """⚠ NOT EVERY ROW INTERRUPTS, and the line is a property of the EVENT
  rather than a per-row flag nobody asked for.

  A pack falling through a threshold and a wallet falling through one are
  states that get worse while the errand finishes, and finishing the errand
  is what makes them worse. A message arriving, a clock ticking round, a
  completion, a pack coming back UP are news that can wait -- and a map whose
  `every 60` row aborted every drawing would be a configuration language that
  punishes its user for a row that reads harmless."""
  assert ev.INTERRUPTING_EVENTS == ("battery_below", "points_below")
  assert set(ev.INTERRUPTING_EVENTS) <= set(ev.EVENT_TYPES)
  for quiet in ("every", "message_received", "task_complete", "task_failed",
                "battery_above", "nothing_to_do", "decision_failed"):
    assert quiet not in ev.INTERRUPTING_EVENTS


def test_the_agent_is_told_which_rows_can_reach_it_mid_job(menu):
  """A row that can abort the robot's work and is described as if it cannot
  is a rule the code contradicts -- which is what M14 found in the charging
  rule, and the reason the prompt is part of the arm rather than a later
  refinement (docs/Evaluation.md §2)."""
  text = make(menu).system[0]["text"]
  assert "MID-JOB" in text
  assert "battery_below" in text and "points_below" in text
  assert "Everything else waits until you are done." in text
  # ...and that stopping is not free, which is the whole of what makes the
  # question answerable.
  assert "Stopping is never free" in text
  assert "nobody can be reached in time you stow and go" in text


def test_the_interrupt_asks_a_binary_and_not_an_action(menu):
  """⚠ "Carry on with what you are doing" is not something the menu can
  express: the menu names things to START and the robot is already half-way
  through one. So the interrupt asks a strictly SMALLER question than a
  decision -- one boolean and a sentence -- and nothing in it can name a
  board, a task or an action, so the surface the fixed menu defends does not
  grow."""
  schema = make(menu).interrupt_schema()
  assert set(schema["properties"]) == {"continue_errand", "reason"}
  assert schema["properties"]["continue_errand"]["type"] == "boolean"
  assert schema["additionalProperties"] is False
  for named in ("action", "task", "board", "program", "zone"):
    assert named not in schema["properties"]


# ---- the answer, and every way it can fail ----------------------------------


def _answer(boss, state=None):
  boss.start_interrupt(state or _state(0.1), "feed:lab",
                       "your pack is at 10%")
  while boss.interrupt_pending:
    pass
  return boss.interrupt_result()


def test_a_model_that_says_carry_on_is_believed(menu):
  boss = make(menu, {"continue_errand": True, "reason": "nearly finished"})
  answer = _answer(boss)
  assert answer["continue"] is True and answer["source"] == "llm"
  assert answer["why"] == "nearly finished"


@pytest.mark.parametrize("client,why", [
  (FakeClient(TimeoutError("too slow")), "timeout"),
  (FakeClient(RuntimeError("connection reset")), "offline"),
  (FakeClient("not json at all"), "garbled"),
])
def test_an_interrupt_nobody_answers_aborts_rather_than_carries_on(menu,
                                                                   client,
                                                                   why):
  """⚠ THE ONE PLACE IN THIS DESIGN WHERE FAILING SAFE IS RIGHT, and it is
  the opposite of `mind/mode.py`'s rule one module over: an unreadable
  operator mode means `llm` rather than `paused`, because a stuck world looks
  broken to everybody. Here the alternative is a robot that keeps driving
  because nobody answered -- and the interrupt fires precisely when the pack
  is low, which is when the fallback rate has always been worst."""
  boss = make(menu, client=client)
  answer = _answer(boss)
  assert answer["continue"] is False
  assert answer["source"] == f"fallback:{why}"
  assert boss.stats()["interrupts"]["aborted"] == 1


def test_a_spent_budget_aborts_rather_than_waving_it_through(menu):
  """An interrupt is an UNSCHEDULED call: it lands on top of whatever the
  hour's decisions have already cost. A world that answered "carry on"
  because it could not afford to ask would be exactly the robot that keeps
  driving because nobody replied."""
  boss = make(menu, {"continue_errand": True, "reason": "sure"},
              calls_per_hour=0)
  answer = _answer(boss)
  assert answer["continue"] is False and answer["source"] == "fallback:budget"


def test_an_interrupt_does_not_disturb_a_decision_in_flight(menu):
  """⚠ ITS OWN SLOT, and the reason is that the two genuinely overlap: the
  errand being interrupted was queued by a decision, and a slow one may still
  be out there. Sharing `_slot` would have whichever landed second silently
  discard the other."""
  boss = make(menu, {"continue_errand": True, "reason": "ok"})
  boss.start(_state(0.9))
  answer = _answer(boss)
  assert answer["continue"] is True
  decision = boss.result(_state(0.9))
  assert decision.action and boss.decisions[-1] is decision


def test_the_question_names_the_errand_and_what_stopping_costs(menu):
  """A robot asked "carry on?" without being told it is HOLDING something
  reads the question as free. `abort` is not "stop", it is "drive back and
  hang the thing up"."""
  client = FakeClient({"continue_errand": False, "reason": "too low"})
  boss = make(menu, client=client)
  _answer(boss)
  turn = client.calls[-1]["messages"][0]["content"]
  assert "feed:lab" in turn
  assert "put the tool back" in turn and "not free" in turn
  # ...and it rides the SAME cached prefix, byte for byte: a second question,
  # not a second mind.
  assert client.calls[-1]["system"] == boss.system


# ---- the mechanics, through the loop -----------------------------------------


#: The threshold the rows below fire on. The stub's pack is full until the
#: first fetch, which drops it under this: the row fires mid-errand, never
#: before one started (a row fired between errands is QUEUED, not an
#: interrupt -- correct, and not what these tests are about).
MID_ERRAND = 0.7

#: A job that fetches a tool, works with it for a while, and stows it.
JOB = "def job():\n  fetch(\"module_lcd\")\n  wait(2)\n  wait(2)\n  stow()\n"


def _job():
  return programmed_errand(lang.compile_procedure(JOB, world_facts(QUAD_HOME)),
                           task="program", name="procedure")


def _hazard(row, client=None, errands=()):
  """A stub robot whose map is `row`, and whose pack drops under
  `MID_ERRAND` during its first fetch -- held a think slice, so the seam that
  reads the map sees the crossing mid-errand (a stub fetch takes no time)."""
  boss = make(Menu.for_world(QUAD_HOME, None),
              **({"client": client} if client is not None else {}))
  life = stub_life(overseer=boss, errands=list(errands))
  boss.event_map = ev.EventMap((row,))
  real, dropped = life.body.fetch_tool_routine, []

  def fetch(*a, **kw):
    got = yield from real(*a, **kw)
    if not dropped:
      life.battery.energy_wh = life.battery.capacity_wh * (MID_ERRAND - 0.05)
      dropped.append(float(life.data.time))
      yield from life.body.hold_routine(THINK_SLICE_S)
    return got
  life.body.fetch_tool_routine = fetch
  return life


def test_an_aborted_procedure_hangs_its_tool_back_even_with_the_endpoint_down():
  """⚠ THE RULE THAT CANNOT BE GOT WRONG: an errand abandoned with a module
  on the fork leaves it wherever it stopped, and a module dropped in the
  rack's approach lane is what stranded a run once. The aborted job ends
  with its tool HUNG, exactly as a finished one leaves it.

  ⚠ WITH A CLIENT THAT RAISES ON EVERY CALL, which is why a row naming an
  action beats a fixed interrupt: the agent pre-committed, so code carries
  the instruction out and asks nobody -- it keeps working when the endpoint
  is DOWN, which is exactly when a low-battery interrupt is worth having.
  The interrupt still fires, still aborts, and its source is the ROW."""
  row = ev.Row(event="battery_below", action="charge", value=MID_ERRAND)
  life = _hazard(row, client=FakeClient(RuntimeError("down")))
  said: list[str] = []
  life.say_hooks.append(lambda t, msg: said.append(msg))
  out = life.body.run(life.run_errand_routine(_job()))
  [entry] = life.interrupts
  assert entry["outcome"] == "aborted"
  assert entry["asked"] is False, "a row naming an action makes no call"
  assert entry["source"] == "event:battery_below"
  assert out["procedure"]["stopped"] == "interrupted"
  assert out["procedure"]["completed"] == 1, "the fetch, and nothing after it"
  assert out["picked"] and out["stowed"], "abort means STOW, never drop"
  assert life.body.holding is None
  # ⚠ AND IT IS NOT AN ERROR. The job did not fail, it was stopped on
  # purpose -- folding the two would put an act of caution in `whFailed`.
  assert not out.get("error")
  # ...nor is it narrated as a failed walk: a reader who cannot tell a
  # choice from a fault reads one of them as the navigation being broken.
  assert not any("never got there" in line for line in said), \
      "an abort must not narrate as a failed walk"
  assert any(line.startswith("INTERRUPT ") for line in said), \
      "and it says what actually happened"


def test_a_later_errand_runs_cleanly_after_an_abort():
  """The acceptance criterion's other half, as a day on the stub: three
  jobs and a hazard row that fires once, during the first fetch -- and
  every job after the abort fetches, works and hangs its tool back
  UNINTERRUPTED. The latch is the errand's, not the day's. Shown to fail
  by dropping `self._aborting = False` from `run_errand_routine`'s
  per-errand reset."""
  life = _hazard(ev.Row(event="battery_below", action="idle", value=MID_ERRAND),
                 errands=[_job() for _ in range(3)])
  life.stop_when(lambda: len(life.errand_results) >= 3)
  out = life.run(start=world_config(QUAD_HOME)["start"], max_sim_time=300.0)

  assert [i["outcome"] for i in out["interrupts"]] == ["aborted"]
  first, *later = out["errands"]
  assert first["procedure"]["stopped"] == "interrupted"
  assert first["picked"] and first["stowed"]
  assert len(later) == 2, "the day did not go on after the abort"
  for i, e in enumerate(later, start=1):
    assert not e["procedure"].get("stopped"), f"errand {i} inherited the abort"
    assert e["picked"], f"errand {i} could not fetch its module"
    assert e["stowed"], f"errand {i} could not hang its module back"
    assert not e.get("error"), f"errand {i}: {e.get('error')}"


def test_the_abort_latches_so_nobody_is_asked_twice(menu, monkeypatch):
  """"One question, one answer" -- a second interrupt inside one errand is a
  spin, and by the time it would fire the robot is already doing the thing
  the first answer asked for. `run_errand` has three safe points and a
  drawing has one per stroke, so the latch is what stands between an abort
  and a question at every one of them.

  Asserted on the latch itself rather than through a mission, because a
  mission cannot be made to reach a second safe point on request."""
  from pluggybot.lifecycle import HubLifecycle

  life = HubLifecycle.__new__(HubLifecycle)
  life._aborting = True
  life._interrupt_pending = "a row nobody must look at"
  #  Latched: it answers without resolving, so `_resolve_interrupt` -- which
  #  is what spends a call -- is never reached.
  monkeypatch.setattr(HubLifecycle, "_resolve_interrupt",
                      lambda self, row: pytest.fail("asked twice"))
  assert life.interrupted() is True
  assert life._interrupt_pending == "a row nobody must look at", \
      "a latched abort does not even consume the pending row"


def test_a_continue_consumes_the_row_rather_than_charging_afterwards(menu):
  """⚠ Leaving it queued would run the row's action the moment the errand
  ended -- the robot going to the rack anyway, five minutes after deciding
  not to, which is not what "carry on" means."""
  import inspect

  from pluggybot.lifecycle import HubLifecycle
  src = inspect.getsource(HubLifecycle._resolve_interrupt)
  assert "self.queued_row = None" in src
  assert "not self._aborting and self.queued_row is row" in src


# ---- where the resolution happens, and where it must not ---------------------


def test_the_seam_only_sets_a_flag(menu):
  """⚠ RESOLVING AN INTERRUPT MAY MEAN AN API CALL, and the seam runs BETWEEN
  PHYSICS STEPS: a call there freezes the world, and stepping the sim from
  inside a step hook re-enters it -- which #143 measured as a RecursionError
  rather than a slow leak. The hook sets a flag and nothing else."""
  import inspect

  from pluggybot.lifecycle import HubLifecycle
  src = inspect.getsource(HubLifecycle._events_step)
  assert "self._interrupt_pending = row" in src
  for forbidden in ("_ask_interrupt", "_resolve_interrupt", "_drive",
                    "start_interrupt"):
    assert forbidden not in src, f"the seam must not reach {forbidden}"


def test_the_call_steps_the_sim_rather_than_freezing_it(menu):
  """`_decide`'s rule, and it matters more here: the robot is standing still
  mid-errand and this is the one moment the stream is most worth watching."""
  import inspect

  from pluggybot.lifecycle import HubLifecycle
  src = inspect.getsource(HubLifecycle._ask_interrupt)
  assert "while self.overseer.interrupt_pending:" in src
  assert "self.body.run(self.body.hold_routine(THINK_SLICE_S))" in src


def test_interrupted_is_a_method_because_it_has_a_side_effect(menu):
  """A property that made an API call would be a side effect hiding behind an
  attribute read, in a file where `needs_charge` next door is genuinely
  free."""
  from pluggybot.lifecycle import HubLifecycle
  assert callable(HubLifecycle.interrupted)
  assert not isinstance(HubLifecycle.interrupted, property)
  assert isinstance(HubLifecycle.needs_charge, property)


# ---- no map, no interrupt ------------------------------------------------


def test_a_world_with_no_map_cannot_be_interrupted_at_all():
  """⚠ OFF WHERE THERE IS NO MAP, and by CONSTRUCTION rather than by a flag
  check: an interrupt needs a hazard row, and a hazard row needs an event
  map -- which the loop with no mind has no origin for, and a mind at
  origin `none`, the default, does not have."""
  assert "origin" not in arm_flags("scripted")
  assert arm_flags("autonomous", "A0")["origin"] == "none"
  # ...and `none` means no map, so nothing can fire mid-errand there either
  from pluggybot.mind.events import origin_map
  assert origin_map("none", None) is None
