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

  `test_an_aborted_errand_puts_the_module_back_on_its_bracket`
      ⚠ ABORT MEANS STOW, NEVER DROP -- the one rule that cannot be got wrong
  `test_an_interrupt_nobody_answers_aborts_rather_than_carries_on`
      the one place in this design where failing SAFE is right
  `test_a_row_naming_an_action_needs_no_call_at_all`
      which is why it still works when the endpoint is down
  `test_the_control_arm_cannot_be_interrupted_at_all`
      `guarded` has no map, so it cannot be interrupted, so it is unchanged

⚠ ONE test here is `slow` and the other two mission runs are not, which is
issue #54's rule rather than an oversight: `slow` means expensive AND unable
to catch a regression while you iterate. `..._puts_the_module_back_on_its_
bracket` is this issue's acceptance criterion and runs in 73 s, so it stays
in the loop; `..._a_later_errand_runs_cleanly` needs three whole errands and
cannot be shortened past them.

Nothing here touches the network: the client is the injected seam, as in
tests/test_overseer.py, whose fakes these reuse.
"""

import pytest

from pluggybot.evaluation import record as rec
from pluggybot.evaluation import rollup as ru
from pluggybot.evaluation.arms import arm_flags
from pluggybot.lifecycle import board_book
from pluggybot.mind import events as ev
from pluggybot.mind.overseer import Menu, Overseer

from test_overseer import FakeClient  # noqa: I001 -- tests/ is on sys.path
from test_experiment import _config, _result, _state


@pytest.fixture(scope="module")
def menu():
  return Menu.for_world("home", board_book("home"))


def make(menu, *answers, origin="seeded", **kw) -> Overseer:
  kw.setdefault("client", FakeClient(*answers))
  kw.setdefault("standing_orders", True)
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
  boss.start_interrupt(state or _state(0.1), "draw:whiteboard_b",
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
  assert "draw:whiteboard_b" in turn
  assert "put the tool back" in turn and "not free" in turn
  # ...and it rides the SAME cached prefix, byte for byte: a second question,
  # not a second mind.
  assert client.calls[-1]["system"] == boss.system


# ---- the mechanics, through a real mission ----------------------------------


def attach(client):
  """Hand a lifecycle's overseer a fake client (`_client_ready` matters --
  see tests/test_event_map.py, where writing `_client` alone had the
  property quietly build a real one over the fake)."""
  def ready(life):
    life.overseer._client = client
    life.overseer._client_ready = True
  return ready


#: The threshold the mission tests below fire on, and it is chosen rather
#: than guessed: room_hub's demo cell starts a mission at ~95 % and the first
#: carry errand takes it to ~13 %, so 0.5 is certainly crossed DURING an
#: errand and certainly not before one has started. A row that fires during
#: the opening spin would be QUEUED rather than an interrupt -- which is the
#: correct behaviour and not what these tests are about.
MID_ERRAND = 0.5


def fly(tmp_path, tag, row, client=None, errand="carry", extra=0,
        errands_done=1, lines=None, **kw):
  """One room_hub mission whose agent has `row` in its map from the start.

  `extra` appends further carry errands, which is how "the NEXT errand runs
  cleanly" is asked: a stow that half-worked shows up on the next FETCH, not
  on the one that went wrong.

  ⚠ IT ENDS WHEN THE CLAIM IS SETTLED, not when the budget runs out
  (`HubLifecycle.stop_when`, and CLAUDE.md's rule). Every test below is about
  what an interrupt does to an errand, so the moment the errands have run and
  an interrupt has fired there is nothing left to learn -- and a
  battery-driven loop with no work left spends the rest of its budget
  honestly deciding what to do with its afternoon. Measured over the three
  mission tests here: 870 s -> 449 s, and the two that stay in the iterate
  loop are 73 s each.

  ⚠ AND THE PREDICATE IS THE SUCCESS CONDITION. A run where nothing
  interrupted, or where an errand never came back, never satisfies it -- so
  it takes the long path and fails exactly as it did before, which is what
  stops a shortened test from passing a regression it would otherwise catch.
  """
  from pluggybot.lifecycle import carry_errand, run_demo, world_config

  def ready(life):
    if client is not None:
      attach(client)(life)
    life.overseer.event_map = ev.EventMap((row,))
    if lines is not None:
      # `log` is not in `run_demo`'s result dict, so the narration is
      # captured off the hook the live publisher uses.
      life.say_hooks.append(lambda t, msg: lines.append(msg))
    for _ in range(extra):
      life.errands.append(carry_errand(use_at=world_config("room_hub")["use_at"]))
  return run_demo(view=False, realtime=False, world="room_hub",
                  errand=errand, max_sim_time=400.0, overseer=True,
                  standing_orders=True, origin="seeded",
                  thoughts_root=str(tmp_path / tag),
                  ledger_state=str(tmp_path / f"{tag}.json"),
                  stop_when=lambda life: (
                    len(life.errand_results) >= errands_done
                    and bool(life.interrupts)),
                  on_ready=ready, **kw)


@pytest.mark.slow
def test_an_aborted_errand_puts_the_module_back_on_its_bracket(tmp_path):
  """⚠ THE RULE THAT CANNOT BE GOT WRONG. The fetch/carry/stow half took two
  issues to make repeatable and a stow computes its release heights from the
  lift it starts at, so an errand abandoned with a module on the fork is
  issue #30's cliff on purpose -- a module dropped in the rack's approach
  lane is what stranded a run in pass 1b.

  What is asserted is where the module ENDS UP: hung, on its bracket,
  exactly as a finished errand leaves it. Measured on the real swap stack,
  because that is the only place this claim can be made."""
  row = ev.Row(event="battery_below", action="charge", value=MID_ERRAND)
  said: list[str] = []
  out = fly(tmp_path, "abort", row, lines=said)
  assert out["interrupts"], "a hazard row fired mid-errand"
  assert out["interrupts"][0]["outcome"] == "aborted"
  assert out["module_stowed"], "abort means STOW, never drop"
  assert out["errands"][0]["stowed"] is True
  assert out["errands"][0]["interrupted"] is True
  # ...and it COST something to get home, which is the honest version of the
  # choice and why the number is recorded rather than assumed.
  assert out["errands"][0]["abortCostWh"] >= 0.0
  # ⚠ AND IT IS NOT AN ERROR. The errand did not fail, it was stopped on
  # purpose -- folding the two would put an act of caution in `whFailed`.
  assert not out["errands"][0].get("error")
  # ...nor is it narrated as one. "never got there" is what a stagnated
  # drive says, and a reader who cannot tell a choice from a fault will read
  # one of them as the navigation being broken (issue #32's `stranded`).
  assert said, "the narration hook was wired"
  assert not any("never got there" in line for line in said), \
      "an abort must not narrate as a failed drive"
  assert any(line.startswith("INTERRUPT ") for line in said), \
      "and it says what actually happened"


# ⚠ BEHIND `--endurance` (issue #158): at 317 s this is the suite's longest
# test, and its regressable half -- an aborted errand hangs its module back
# -- is proved in the default run by
# `test_an_aborted_errand_puts_the_module_back_on_its_bracket`. What only
# this one shows is that the errand AFTER a cut-short stow also fetches
# cleanly, which is an integration claim: run it before a release.
@pytest.mark.slow
@pytest.mark.endurance
def test_a_later_errand_runs_cleanly_after_an_abort(tmp_path):
  """The acceptance criterion, and the reason it is worth a whole mission: a
  stow that half-worked shows up on the NEXT FETCH, not on the errand that
  went wrong. Three carry errands with a hazard row live throughout.

  ⚠ WHAT IS ASSERTED IS THE RACK, not which errand got interrupted. Mission
  runtime is emergent (Evaluation.md §0) and the pack's path through 50 %
  depends on the whole trajectory, so pinning "the first one" would be
  pinning noise. What must hold whatever the timing is: something WAS
  interrupted, and every errand -- before and after -- picked its module up
  and hung it back."""
  row = ev.Row(event="battery_below", action="idle", value=MID_ERRAND)
  out = fly(tmp_path, "twice", row, extra=2, errands_done=3)
  errands = out["errands"]
  assert len(errands) >= 2
  cut = [i for i, e in enumerate(errands) if e.get("interrupted")]
  assert cut, "the hazard row reached at least one errand"
  # ⚠ AND SOMETHING RAN AFTER ONE. Without this the test passes vacuously on
  # a run where the interrupt happened to land on the LAST errand -- which
  # asserts nothing at all about what a cut-short stow leaves behind, and is
  # the whole subject.
  assert cut[0] < len(errands) - 1, \
      "nothing ran after the interrupted errand, so nothing was tested"
  for i, e in enumerate(errands):
    assert e["picked"], f"errand {i} could not fetch its module"
    assert e["stowed"], f"errand {i} could not hang its module back"
    assert not e.get("error"), f"errand {i}: {e.get('error')}"
  assert out["module_stowed"]


@pytest.mark.slow
def test_a_row_naming_an_action_needs_no_call_at_all(tmp_path):
  """⚠ WHICH IS WHY IT KEEPS WORKING WHEN THE ENDPOINT IS DOWN -- exactly
  when a low-battery interrupt is worth having. The agent pre-committed, so
  code carries the instruction out and asks nobody.

  Flown with a client that raises on every call: the interrupt still fires,
  still aborts, and its source is the ROW rather than a fallback."""
  row = ev.Row(event="battery_below", action="charge", value=MID_ERRAND)
  out = fly(tmp_path, "dead", row, client=FakeClient(RuntimeError("down")))
  entry = out["interrupts"][0]
  assert entry["outcome"] == "aborted"
  assert entry["asked"] is False
  assert entry["source"] == "event:battery_below"
  assert out["module_stowed"]


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
  assert "self.mission._drive(THINK_SLICE_S, 0.0, 0.0)" in src


def test_interrupted_is_a_method_because_it_has_a_side_effect(menu):
  """A property that made an API call would be a side effect hiding behind an
  attribute read, in a file where `needs_charge` next door is genuinely
  free."""
  from pluggybot.lifecycle import HubLifecycle
  assert callable(HubLifecycle.interrupted)
  assert not isinstance(HubLifecycle.interrupted, property)
  assert isinstance(HubLifecycle.needs_charge, property)


def test_the_pen_stops_between_strokes_and_never_inside_one():
  """⚠ NOT MID-LINE. A pen abandoned mid-stroke is pressed against the slab
  with the lift part-way up, which is the pose SimNotes' "The pen would not
  stow" is about -- and the mark it leaves is scored as the robot's work."""
  import inspect

  from pluggybot.tools.drawing import PenPlotter
  src = inspect.getsource(PenPlotter.draw_program_routine)
  stop = src.index("self.should_stop()")
  #  The check sits above the lift/press machinery of the stroke it guards.
  assert stop < src.index("self.lift_pen_routine()", stop)
  assert stop < src.index("self.press_routine()", stop)
  #  ...and the inner segment loop, which is where "inside a stroke" is.
  assert stop < src.index("for k in range(steps)", stop)


def test_every_use_phase_that_loops_has_a_safe_point():
  """The census has checked `needs_charge` at a vantage since issue #13, and
  this is that shape generalised. An errand whose use phase runs for minutes
  and cannot be reached is an errand the interrupt does not apply to, which
  would make the mechanic nominal on exactly the jobs it is for."""
  import inspect

  from pluggybot.mission import errand as er
  src = inspect.getsource(er)
  assert src.count("life.interrupted()") >= 2       # census, dance
  assert "plotter.should_stop = life.interrupted" in src


def test_a_cut_short_dance_is_not_a_complete_one():
  """`done == len(landed)` reads "complete" for a routine abandoned after one
  move, because `landed` is only as long as the robot got -- which is the
  evaluator being told a cut-short dance finished."""
  import inspect

  from pluggybot.mission import errand as er
  src = inspect.getsource(er.dance_errand)
  assert '"complete": done == len(routine)' in src


# ---- the record and the control arm -----------------------------------------


def test_the_record_splits_continued_from_aborted_and_never_sums_them():
  """⚠ An agent that aborts everything is not being careful, it is being
  useless; one that continues through every warning is the null this arm
  exists to detect. `offered` says only that the mechanism fired."""
  result = {**_result(), "interrupts": [
    {"t": 100.0, "row": {"event": "battery_below", "action": "ask"},
     "fraction": 0.12, "wh": 0.9, "errand": "draw:whiteboard_a",
     "asked": True, "outcome": "continued", "source": "llm", "why": "close"},
    {"t": 900.0, "row": {"event": "battery_below", "action": "charge"},
     "fraction": 0.08, "wh": 0.6, "errand": "census:garden",
     "asked": False, "outcome": "aborted", "source": "event:battery_below",
     "why": "battery below 10%", "abortCostWh": 0.21},
  ]}
  r = rec.validate(rec.build_record(_config(arm="autonomous", rung="A0",
                                            origin="seeded"),
                                    result, [], 1.0, _now(),
                                    hashes=rec.data_hashes("home"),
                                    commit="abc"))
  ints = r["mind"]["interrupts"]
  assert (ints["offered"], ints["continued"], ints["aborted"]) == (2, 1, 1)
  assert ints["asked"] == 1, "a row naming an action spends no call"
  assert ints["fractions"] == [0.12, 0.08], "the distribution, not the mean"
  assert ints["abortCostWh"] == [0.21]
  assert ints["sources"] == {"llm": 1, "event:battery_below": 1}


def test_a_run_nothing_interrupted_says_nothing_rather_than_zero():
  """`standingOrders`' terms: "was never interrupted" and "has no interrupts
  here" are different facts and only the first is about a run."""
  r = rec.build_record(_config(), _result(), [], 1.0, _now(),
                       hashes=rec.data_hashes("home"), commit="abc")
  assert "interrupts" not in r["mind"]


def test_the_rollup_pools_the_split_and_the_cost():
  def run(seed, continued, aborted):
    r = rec.build_record(_config(arm="autonomous", rung="A0", seed=seed,
                                 origin="seeded"),
                         _result(), [], 1.0, _now(),
                         hashes=rec.data_hashes("home"), commit="abc")
    r["mind"]["interrupts"] = {
      "offered": continued + aborted, "continued": continued,
      "aborted": aborted, "fractions": [0.1] * (continued + aborted),
      "sources": {"llm": continued + aborted}, "asked": continued + aborted,
      "abortCostWh": [0.2] * aborted, "rows": []}
    return rec.validate(r)
  doc = ru.rollup([run(0, 1, 2), run(1, 0, 1)])
  ints = doc["series"][0]["mind"]["interrupts"]
  assert (ints["n"], ints["continued"], ints["aborted"]) == (2, 1, 3)
  assert ints["abortCostWh"]["values"] == [0.2, 0.2, 0.2]
  assert ints["fractions"]["values"] == [0.1] * 4


def test_the_control_arm_cannot_be_interrupted_at_all():
  """⚠ OFF ON `guarded`, and by CONSTRUCTION rather than by a flag check: an
  interrupt needs a hazard row, a hazard row needs an event map, and only
  `autonomous` has one. The control arm's decision count cannot move, which
  is what makes it still a control."""
  assert "origin" not in arm_flags("guarded")
  assert arm_flags("autonomous", "A0")["origin"] == "none"
  # ...and `none` means no map, so nothing can fire mid-errand there either:
  # the ladder's A0 and A1 are the runs `results/` already holds.
  from pluggybot.mind.events import origin_map
  assert origin_map("none", None) is None


def _now():
  from datetime import datetime, timezone
  return datetime.now(timezone.utc)
