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

  `test_an_aborted_procedure_stops_at_a_safe_point_with_the_endpoint_down`
      ⚠ AN ABORT STOPS THE JOB AT A SAFE POINT, and is no error -- with a
      dead client, because a row naming an action asks nobody and so still
      works when the endpoint is down
  `test_an_interrupt_nobody_answers_aborts_rather_than_carries_on`
      the one place in this design where failing SAFE is right
  `test_a_world_with_no_map_cannot_be_interrupted_at_all`
      no map, no hazard row, so nothing reaches the robot mid-job

The loop's half is a day on the stub body (`tests/test_body.py`).

Nothing here touches the network: the client is the injected seam, as in
tests/test_overseer.py, whose fakes these reuse.
"""

import math

import pytest

from pluggybot import tick
from pluggybot.navigator import DRIVE_STOPPED
from pluggybot.evaluation.arms import arm_flags
from pluggybot.lifecycle import HUMANOID_HOME, THINK_SLICE_S, board_book, world_config, world_facts
from pluggybot.mind import events as ev
from pluggybot.mind.overseer import Menu, Overseer
from pluggybot.mission.errand import programmed_errand
from pluggybot.procedure import lang
from pluggybot.procedure import steps as st
from pluggybot.procedure.steps import DRIVE_TIMEOUT_S

from test_body import stub_life  # noqa: I001 -- tests/ is on sys.path
from test_overseer import FakeClient


@pytest.fixture(scope="module")
def menu():
  return Menu.for_world(HUMANOID_HOME, board_book(HUMANOID_HOME))


def _state(fraction: float) -> dict:
  """A synthetic decision state: what the interrupt's question is shown."""
  return {"simTimeS": 100.0,
          "battery": {"fraction": fraction, "wh": 8.0 * fraction,
                      "spendableWh": max(0.0, 8.0 * fraction - 0.9)},
          "offeredTasks": [],
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
  assert "nobody can be reached in time you stop and go" in text


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
  boss.start_interrupt(state or _state(0.1), "procedure",
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
  """A robot asked "carry on?" without being told what stopping costs reads
  the question as free: the work done so far is scored as it stands."""
  client = FakeClient({"continue_errand": False, "reason": "too low"})
  boss = make(menu, client=client)
  _answer(boss)
  turn = client.calls[-1]["messages"][0]["content"]
  assert "procedure" in turn
  assert "not free" in turn and "scored as it stands" in turn
  # ...and it rides the SAME cached prefix, byte for byte: a second question,
  # not a second mind.
  assert client.calls[-1]["system"] == boss.system


# ---- the mechanics, through the loop -----------------------------------------


#: The threshold the rows below fire on. The stub's pack is full until the
#: first walk, which drops it under this: the row fires mid-errand, never
#: before one started (a row fired between errands is QUEUED, not an
#: interrupt -- correct, and not what these tests are about).
MID_ERRAND = 0.7

#: A job that walks somewhere and works there for a while.
JOB = "def job():\n  drive_to(2.0, 1.0)\n  wait(2)\n  wait(2)\n"


def _job():
  return programmed_errand(lang.compile_procedure(JOB, world_facts(HUMANOID_HOME)),
                           task="program", name="procedure")


def _hazard(row, client=None, errands=()):
  """A stub robot whose map is `row`, and whose pack drops under
  `MID_ERRAND` during its first walk -- held a think slice, so the seam that
  reads the map sees the crossing mid-errand (a stub walk takes no time)."""
  boss = make(Menu.for_world(HUMANOID_HOME, None),
              **({"client": client} if client is not None else {}))
  life = stub_life(overseer=boss, errands=list(errands))
  boss.event_map = ev.EventMap((row,))
  real, dropped = life.body.go_to_routine, []

  def walk(*a, **kw):
    got = yield from real(*a, **kw)
    if not dropped:
      life.battery.energy_wh = life.battery.capacity_wh * (MID_ERRAND - 0.05)
      dropped.append(float(life.data.time))
      yield from life.body.hold_routine(THINK_SLICE_S)
    return got
  life.body.go_to_routine = walk
  return life


def test_an_aborted_procedure_stops_at_a_safe_point_with_the_endpoint_down():
  """⚠ AN ABORT STOPS THE JOB WHERE IT IS SAFE TO: at a verb's end, or a
  walk's second -- and it is not an error, nor a failed walk (the
  quadruped's version of this rule hung its tool back; the humanoid
  carries none, #559).

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
  assert out["procedure"]["completed"] == 1, "the walk, and nothing after it"
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
  jobs and a hazard row that fires once, during the first walk -- and
  every job after the abort walks and works to its end UNINTERRUPTED. The
  latch is the errand's, not the day's. Shown to fail by dropping
  `self._aborting = False` from `run_errand_routine`'s per-errand reset."""
  life = _hazard(ev.Row(event="battery_below", action="idle", value=MID_ERRAND),
                 errands=[_job() for _ in range(3)])
  life.stop_when(lambda: len(life.errand_results) >= 3)
  out = life.run(start=world_config(HUMANOID_HOME)["start"], max_sim_time=300.0)

  assert [i["outcome"] for i in out["interrupts"]] == ["aborted"]
  first, *later = out["errands"]
  assert first["procedure"]["stopped"] == "interrupted"
  assert len(later) == 2, "the day did not go on after the abort"
  for i, e in enumerate(later, start=1):
    assert not e["procedure"].get("stopped"), f"errand {i} inherited the abort"
    assert e["procedure"]["completed"] == e["procedure"]["total"] == 3, e["procedure"]
    assert not e.get("error"), f"errand {i}: {e.get('error')}"


def test_the_abort_latches_so_nobody_is_asked_twice(menu, monkeypatch):
  """"One question, one answer" -- a second interrupt inside one errand is a
  spin, and by the time it would fire the robot is already doing the thing
  the first answer asked for. `run_errand` has three safe points and a
  drawing has one per stroke, so the latch is what stands between an abort
  and a question at every one of them -- and a walk's `stop` reads it too
  (`interrupt_due`, #692), so a walk after an abort ends where it stands.

  Asserted on the latch itself rather than through a mission, because a
  mission cannot be made to reach a second safe point on request. Shown to
  fail with the latch's early answer taken out: asked twice."""
  from pluggybot.lifecycle import HubLifecycle

  life = HubLifecycle.__new__(HubLifecycle)
  life._aborting = True
  life._interrupt_pending = "a row nobody must look at"
  #  Latched: it answers without resolving, so `_resolve_interrupt_routine`
  #  -- which is what spends a call -- is never reached, and nothing steps.
  monkeypatch.setattr(HubLifecycle, "_resolve_interrupt_routine",
                      lambda self, row: pytest.fail("asked twice"))
  assert tick.run(None, life.interrupted_routine()) is True
  assert life._interrupt_pending == "a row nobody must look at", \
      "a latched abort does not even consume the pending row"
  assert life.interrupt_due() is True


def test_a_continue_consumes_the_row_rather_than_charging_afterwards(menu):
  """⚠ Leaving it queued would run the row's action the moment the errand
  ended -- the robot going to the rack anyway, five minutes after deciding
  not to, which is not what "carry on" means."""
  import inspect

  from pluggybot.lifecycle import HubLifecycle
  src = inspect.getsource(HubLifecycle._resolve_interrupt_routine)
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
  """`_decide_routine`'s rule, and it matters more here: the robot is
  standing still mid-errand and this is the one moment the stream is most
  worth watching -- and its think slices are YIELDED as that rule's are
  (#692): waited blocking (`body.run`), the pair's other robot paused for
  the whole ask (#690)."""
  import inspect

  from pluggybot.lifecycle import HubLifecycle
  src = inspect.getsource(HubLifecycle._ask_interrupt_routine)
  assert "while self.overseer.interrupt_pending:" in src
  assert "yield from self.body.hold_routine(THINK_SLICE_S)" in src
  assert ".run(" not in src


def test_the_interrupt_is_asked_by_a_routine_and_read_by_a_plain_method(menu):
  """Resolving has a side effect -- a call, the world stepped while it
  flies -- so it is a ROUTINE (#692), never a property: one would be a side
  effect hiding behind an attribute read, in a file where `needs_charge`
  next door is genuinely free. And it has no blocking twin: handed to a
  walk as its `stop`, `interrupted()` asked inside the walk, blocking. What
  a walk is handed reads, and resolves nothing (`interrupt_due`)."""
  import inspect

  from pluggybot.lifecycle import HubLifecycle
  assert inspect.isgeneratorfunction(HubLifecycle.interrupted_routine)
  assert not hasattr(HubLifecycle, "interrupted")
  assert callable(HubLifecycle.interrupt_due)
  assert not isinstance(inspect.getattr_static(HubLifecycle, "interrupt_due"), property)
  assert isinstance(HubLifecycle.needs_charge, property)


# ---- the ask is a routine, and what it stopped goes on (issue #692) ------------


class _Mind:
  """The interrupt's call, in flight `slices` think slices and answering
  `carry_on`: what `HubLifecycle._ask_interrupt_routine` reads of a mind --
  and when each ask began and was answered (`asked`)."""

  event_map = None

  def __init__(self, data, slices: int = 6, carry_on: bool = True) -> None:
    self.data, self.slices, self.carry_on = data, slices, carry_on
    self.asked: list[list[float]] = []
    self.left = 0

  def start_interrupt(self, state, errand, why) -> None:
    self.asked.append([float(self.data.time)])
    self.left = self.slices

  @property
  def interrupt_pending(self) -> bool:
    self.left -= 1
    return self.left >= 0

  def interrupt_result(self) -> dict:
    self.asked[-1].append(float(self.data.time))
    return {"continue": self.carry_on, "why": "", "source": "llm"}


def _flag_at(life, t: float) -> None:
  """The seam's flag, set once at sim time `t`: a hazard `ask` row fired
  mid-errand (`_events_step`)."""
  row, done = ev.Row(event="battery_below", action=ev.ASK, value=0.5), []

  def hook():
    if not done and float(life.data.time) >= t - 1e-9:
      done.append(t)
      life._interrupt_pending = row
  life.body.step_hooks.append(hook)


def _paced_walk(life, speed: float = 1.0, held_s: float = 0.0) -> list[dict]:
  """`life`'s walk, a stub's that takes time: straight at its goal at
  `speed` m/s, its `stop` asked every `STOP_EVERY_S` as the Navigator asks
  it -- True ends it where it stands, `DRIVE_STOPPED` -- and out of
  patience at `timeout`; the first held `held_s` as it begins, which its
  clock counts out as a walk's counts a hold (`Navigator.held_s`). Each
  walk's start, patience and ending, in order."""
  from pluggybot import navigator as nav
  body, walks = life.body, []

  def go(x, y, timeout=90.0, stop=None):
    walks.append({"t": float(life.data.time), "timeout": timeout, "from": (body.x, body.y)})
    for _ in range(round((0.0 if len(walks) > 1 else held_s) / life.model.opt.timestep)):
      yield body.STILL
    t0 = float(life.data.time)
    next_stop = t0 + nav.STOP_EVERY_S
    while True:
      d, why = math.hypot(x - body.x, y - body.y), None
      if d < 1e-9:
        why = ""
      elif float(life.data.time) - t0 >= timeout - 1e-9:
        why = "timeout"
      elif stop is not None and float(life.data.time) >= next_stop - 1e-9:
        next_stop += nav.STOP_EVERY_S
        if stop():
          why = nav.DRIVE_STOPPED
      if why is not None:
        walks[-1].update(why=why, ended=float(life.data.time))
        body.last_drive = {"why": why, "goal": (x, y), "shortM": round(d, 3),
                           "seconds": round(float(life.data.time) - t0, 1)}
        return not why
      k = min(1.0, speed * life.model.opt.timestep / d)
      body.x, body.y = body.x + k * (x - body.x), body.y + k * (y - body.y)
      yield body.STILL
  body.go_to_routine = go
  return walks


def _asked_life(monkeypatch, carry_on: bool):
  """A stub robot stepped 0.05 s a step, its mind answering its interrupt
  `carry_on` after six think slices (0.6 s)."""
  from pluggybot import lifecycle
  monkeypatch.setattr(lifecycle, "overseer_context", lambda life: {})
  life = stub_life()
  life.model.opt.timestep = 0.05
  life.overseer = _Mind(life.data, carry_on=carry_on)
  return life


def _ran(life, source: str) -> dict:
  facts = world_facts(HUMANOID_HOME)
  return life.body.run(lang.run_procedure_routine(
    life, lang.compile_procedure(source, facts), facts))


@pytest.mark.parametrize("carry_on", [True, False])
def test_a_walk_its_interrupt_stops_asks_where_it_stands_and_walks_on_with_the_patience_left(
    monkeypatch, carry_on):
  """#692: a walk's `stop` only READS that an interrupt is due
  (`interrupt_due`): the walk ends where it stands, `DRIVE_STOPPED`, and its
  verb asks (`interrupted_routine`, its think slices yielded). Carried on,
  it walks on to its goal from there, a fresh walk given the patience it
  had left -- the ask counting none of it -- and the run goes on; told to
  stop, the verb is `stopped: interrupted` where it stood, never a failure.
  Shown to fail with the ask made inside the walk's `stop`, blocking, as it
  was: one walk, asked from inside, never a second."""
  life = _asked_life(monkeypatch, carry_on)
  walks = _paced_walk(life, held_s=0.5)         # held 0.5 s, then 3 m at 1 m/s
  _flag_at(life, 1.7)                           # ...its stop read at 1.5 s and 2.5 s
  r = _ran(life, "def job():\n  drive_to(3.0, 0.0)\n  wait(1)\n")
  [(t0, t1)] = life.overseer.asked
  assert (t0, t1) == pytest.approx((2.5, 3.1)), "asked as its walk's stop read it, 6 slices"
  assert walks[0]["why"] == DRIVE_STOPPED and walks[0]["ended"] == pytest.approx(2.5)
  if carry_on:
    assert len(walks) == 2, walks
    assert walks[1]["t"] == pytest.approx(3.1) and walks[1]["from"] == pytest.approx((2.0, 0.0))
    assert walks[1]["timeout"] == pytest.approx(DRIVE_TIMEOUT_S - 2.0), \
        "the patience its own clock left: the hold and the ask none of it"
    assert walks[1]["why"] == "" and life.body.pose[:2] == pytest.approx((3.0, 0.0))
    assert r["ok"] and r["completed"] == 2, r
    assert [i["outcome"] for i in life.interrupts] == ["continued"]
  else:
    assert len(walks) == 1, walks
    assert r["stopped"] == "interrupted" and r["completed"] == 0 and "failedAt" not in r, r
    assert r["steps"][0]["stopped"] == "interrupted"
    assert r["steps"][0]["shortM"] == pytest.approx(1.0)
    assert [i["outcome"] for i in life.interrupts] == ["aborted"]


def test_a_death_in_a_walks_ask_ends_the_run_at_once_its_walk_never_resumed(monkeypatch):
  """#671 inside #692: a walk's ask is inside its verb, which a death ends
  the step it lands -- the ask with it, never answered, the walk never
  walked on by a dead robot, the run `stopped: died` and its next verb never
  begun. Shown to fail with the ask made inside the walk's `stop`, blocking:
  it ran on to its answer, the dead robot's walk carrying on."""
  life = _asked_life(monkeypatch, carry_on=True)
  walks = _paced_walk(life)
  _flag_at(life, 1.5)

  def dies():
    if life.dead is None and float(life.data.time) >= 2.3 - 1e-9:
      life.dead = {"cause": "flat"}           # ...as its call flew
  life.body.step_hooks.append(dies)
  r = _ran(life, "def job():\n  drive_to(3.0, 0.0)\n  wait(1)\n")
  assert r["stopped"] == "died" and r["completed"] == 0, r
  assert r["steps"][0]["stopped"] == "died"
  [ask] = life.overseer.asked
  assert ask == pytest.approx([2.0]), "asked, and never answered"
  assert len(walks) == 1 and life.interrupts == []


def _paced_find(life, seconds: float) -> list[dict]:
  """`life`'s find, a stub's that takes time: its place seen once it has
  searched `seconds` in all, its `stop` asked every second as a find's
  walks ask it -- True ends it `interrupted` where it stands -- and out of
  time at `patience`. Each search's start, patience and record."""
  body, finds, searched = life.body, [], [0.0]

  def find(tag, near, patience, stop=None):
    t0 = float(life.data.time)
    rec = {"tag": tag, "remembered": False, "guesses": 0, "viewpoints": 1, "arounds": 1,
           "trail": [{"lookedAt": [round(t0, 2), 0.0]}]}
    finds.append({"t": t0, "patience": patience, "rec": rec})
    next_stop = t0 + 1.0
    while True:
      why = None
      if searched[0] >= seconds - 1e-9:
        why = "found"
        body.places.see(tag, near[0], near[1], float(life.data.time), view=0.0,
                        tag_facing=0.0)
      elif float(life.data.time) - t0 >= patience - 1e-9:
        why = "out of time"
      elif stop is not None and float(life.data.time) >= next_stop - 1e-9:
        next_stop += 1.0
        if stop():
          why = "interrupted"
      if why is not None:
        rec.update(found=why == "found", why=why,
                   seconds=round(float(life.data.time) - t0, 1),
                   **({"at": list(near)} if why == "found" else {}))
        return rec
      searched[0] += life.model.opt.timestep
      yield body.STILL
  body.find_tag_routine = find
  return finds


@pytest.mark.parametrize("carry_on", [True, False])
def test_a_find_its_interrupt_stops_asks_and_searches_on_with_the_patience_left(
    monkeypatch, carry_on):
  """...and a `find`, whose walks and looks read the same `stop`: ended
  where it stands, its verb asks. Carried on, it searches again from there
  with the patience it had left, and its record is ONE find -- the seconds
  it searched and its looks summed, every look on its trail, the ask none
  of them; told to stop, `stopped: interrupted`. Shown to fail with the ask
  made inside the find's `stop`, blocking: one search, never a second."""
  life = _asked_life(monkeypatch, carry_on)
  finds = _paced_find(life, seconds=3.0)
  _flag_at(life, 1.5)
  r = _ran(life, "def job():\n  find(68, 1.0, 2.0)\n  wait(1)\n")
  [(t0, t1)] = life.overseer.asked
  assert (t0, t1) == pytest.approx((2.0, 2.6))
  assert finds[0]["rec"]["why"] == "interrupted"
  if carry_on:
    assert len(finds) == 2, finds
    assert finds[1]["t"] == pytest.approx(2.6)
    assert finds[1]["patience"] == pytest.approx(st.FIND_PATIENCE_S - 2.0)
    assert r["ok"] and r["completed"] == 2, r
    found = r["steps"][0]
    assert found["ok"] and found["seconds"] == pytest.approx(3.0), found
    assert found["trace"].count("looked round") == 2, found["trace"]
  else:
    assert len(finds) == 1, finds
    assert r["stopped"] == "interrupted" and r["completed"] == 0, r
    assert r["steps"][0]["stopped"] == "interrupted"
    assert "by its own interrupt" in r["steps"][0]["reason"]


def test_a_walk_whose_ask_fails_is_stopped_where_it_stands():
  """Every failure aborts (`interrupt_result`), through the routine too: a
  row fired on the seam mid-walk, the walk ended where it stood, and the
  call that could not be made -- the endpoint down -- answered stop. The
  run is `stopped: interrupted`, never an error. Shown to fail with a
  failed call read as carry on: the walk went on to its goal."""
  boss = make(Menu.for_world(HUMANOID_HOME, None), client=FakeClient(RuntimeError("down")))
  life = stub_life(overseer=boss)
  boss.event_map = ev.EventMap((ev.Row(event="battery_below", action=ev.ASK,
                                       value=MID_ERRAND),))
  life.model.opt.timestep = 0.05
  walks = _paced_walk(life)

  def drained():
    if float(life.data.time) >= 0.5 - 1e-9:
      life.battery.energy_wh = min(life.battery.energy_wh,
                                   life.battery.capacity_wh * (MID_ERRAND - 0.05))
  life.body.step_hooks.append(drained)
  facts = world_facts(HUMANOID_HOME)
  errand = programmed_errand(lang.compile_procedure(
    "def job():\n  drive_to(3.0, 0.0)\n  wait(1)\n", facts), task="program", name="procedure")
  out = life.body.run(life.run_errand_routine(errand))
  [entry] = life.interrupts
  assert entry["asked"] and entry["outcome"] == "aborted", entry
  assert entry["source"] == "fallback:offline"
  assert [w["why"] for w in walks] == [DRIVE_STOPPED], walks
  assert out["procedure"]["stopped"] == "interrupted" and out["procedure"]["completed"] == 0
  assert not out.get("error")


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
