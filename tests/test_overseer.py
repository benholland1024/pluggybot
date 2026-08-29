"""The LLM overseer (issue #15) -- and mostly, what it CANNOT do.

Nothing here touches the network. The Anthropic client is an injected seam and
every case below hands in a fake that is slow, that raises, that lies, or that
answers perfectly -- which is also how the acceptance criterion "kill the API
and the robot keeps working" is checked without unplugging anything.

The load-bearing test is `test_charge_priority_survives_an_overseer_that_never_
charges`: it flies a real mission with an overseer that answers `idle` to every
question and asserts the robot still charges. Every other guarantee in this
module is a convenience next to that one -- an LLM that can decline to charge
is an LLM that bricks the world overnight.
"""

import json
import math
import threading
import time
from collections import Counter

import mujoco
import pytest

from pluggybot.mind import overseer as ov
from pluggybot.mind.journal import MAX_NOTE_CHARS, Journal, read_goals
from pluggybot.lifecycle import (
  HubLifecycle, board_book, errand_from, world_config, zone_centre,
)
from pluggybot.mind.overseer import Decision, Menu, Overseer, scripted
from pluggybot.economy.scoring import default_table


# ---- fakes -------------------------------------------------------------------


class FakeUsage:
  def __init__(self, **kw):
    self.input_tokens = kw.get("input_tokens", 1200)
    self.output_tokens = kw.get("output_tokens", 60)
    self.cache_read_input_tokens = kw.get("cache_read_input_tokens", 0)
    self.cache_creation_input_tokens = kw.get("cache_creation_input_tokens", 0)


class FakeBlock:
  type = "text"

  def __init__(self, text):
    self.text = text


class FakeResponse:
  def __init__(self, payload, **usage):
    text = payload if isinstance(payload, str) else json.dumps(payload)
    self.content = [FakeBlock(text)]
    self.usage = FakeUsage(**usage)


class FakeClient:
  """`.messages.create(**kw)` -> whatever the script says, in order.

  `answers` may hold dicts (a decision), strings (raw text), or exceptions
  (raised). The last entry repeats forever, so a test does not have to know
  how many times the overseer will ask.
  """

  def __init__(self, *answers, delay: float = 0.0, usage: dict | None = None):
    self.answers = list(answers) or [{"action": "idle", "reason": "ok"}]
    self.delay = delay
    self.usage = usage or {}
    self.calls: list[dict] = []
    self.messages = self

  def create(self, **kwargs):
    self.calls.append(kwargs)
    if self.delay:
      time.sleep(self.delay)
    answer = self.answers[min(len(self.calls) - 1, len(self.answers) - 1)]
    if isinstance(answer, Exception):
      raise answer
    return FakeResponse(answer, **self.usage)


#: Contention knobs for the back-to-back race test below. Sized by measuring
#: the defect: 4 burners x 40 decisions sent 1 request out of 40, while 8 x 100
#: sent 2 out of 100 and cost the suite 44 s. Smaller AND more sensitive.
BURNERS = 4
DECISIONS = 40


def full(**kw) -> dict:
  """A schema-complete answer; the server guarantees these fields."""
  return {"action": "idle", "reason": "because", "board": "", "program": "",
          "zone": "", "note": "", **kw}


@pytest.fixture(scope="module")
def book():
  return board_book("home")


@pytest.fixture(scope="module")
def menu(book):
  return Menu.for_world("home", book)


def make(menu, *answers, **kw) -> Overseer:
  kw.setdefault("client", FakeClient(*answers))
  return Overseer(menu, **kw)


# ---- the menu is the world, not a wish list ----------------------------------


def test_the_menu_offers_only_what_the_world_can_do(book):
  home = Menu.for_world("home", book)
  assert set(home.boards) == {"whiteboard_a", "whiteboard_b"}
  assert home.census_zone == "garden"
  assert {"draw", "census", "explore"} <= set(home.available())
  # room_hub has no whiteboards, nothing countable and one undivided room.
  # Offering `draw` there is how a decision loop finds a dead end by driving
  # into it, so it is simply not on the menu.
  hub = Menu.for_world("room_hub", None)
  assert "draw" not in hub.available()
  assert "census" not in hub.available()
  assert "explore" not in hub.available()
  assert "carry" in hub.available()


def test_text_is_not_an_offerable_figure(menu):
  """Hershey lettering takes arbitrary caller text, which is the surface
  issue #16 is about. It comes back when visitor text has somewhere safe to
  land -- until then the model cannot ask the robot to write words."""
  assert "text" not in menu.programs
  assert "house" in menu.programs


def test_the_schema_constrains_every_parameter_to_the_menu(menu):
  schema = menu.schema()
  assert schema["additionalProperties"] is False
  assert schema["properties"]["action"]["enum"] == list(menu.available())
  assert set(schema["properties"]["board"]["enum"]) == {*menu.boards, ""}
  assert set(schema["properties"]["zone"]["enum"]) == {*menu.zones, ""}


@pytest.mark.parametrize("raw, why", [
  ({"action": "hack_the_ledger"}, "an action off the vocabulary"),
  ({"action": "draw", "board": "the_ceiling"}, "a board that is not there"),
  ({"action": "draw", "program": "a_portrait_of_ben"}, "an unknown figure"),
  ({"action": "explore", "zone": "the_moon"}, "an unknown zone"),
  ({"action": ""}, "no action at all"),
])
def test_an_answer_off_the_menu_is_refused(menu, raw, why):
  with pytest.raises(ValueError):
    menu.validate(raw)


def test_a_drawing_without_a_choice_still_draws(menu):
  """A `draw` with the parameters left blank picks the first board and figure
  rather than failing: the model committed to the task, and refusing over an
  unfilled optional would send a perfectly good decision to the fallback."""
  d = menu.validate({"action": "draw", "reason": "the wall is bare"})
  assert d.board == menu.boards[0]
  assert d.program == menu.programs[0]


# ---- the failure paths, which are the ones that must never surprise ----------


def test_a_good_answer_is_used_verbatim(menu):
  boss = make(menu, full(action="draw", board="whiteboard_b", program="tree",
                         reason="whiteboard_b is empty"))
  d = boss.decide({})
  assert (d.action, d.board, d.program) == ("draw", "whiteboard_b", "tree")
  assert d.source == "llm" and not d.scripted
  assert boss.usage.llm_calls == 1 and boss.usage.fallbacks == 0


@pytest.mark.parametrize("answer, expect", [
  (RuntimeError("connection reset"), "fallback:RuntimeError"),
  ("I would love to draw a house!", "fallback:ValueError"),
  ({"action": "delete_the_ledger", "reason": "..."}, "fallback:ValueError"),
])
def test_a_broken_answer_falls_back_to_the_scripted_policy(menu, answer,
                                                           expect):
  boss = make(menu, answer)
  d = boss.decide({"tasksThisMission": [], "decisions": 0})
  assert d.source.startswith(expect)
  # ...and the fallback is a real day's work, not a shrug: the whole point of
  # the acceptance criterion is that the robot keeps DOING things.
  assert d.action in ("draw", "census", "dance", "carry", "explore")


def test_a_slow_call_is_abandoned_rather_than_waited_on(menu):
  """The physics-never-blocks guarantee, from the overseer's side: `pending`
  goes false by the deadline even though the worker thread is still alive, so
  the caller's `while pending: step_the_sim()` loop is released by the CLOCK
  and not by the API."""
  boss = Overseer(menu, client=FakeClient(full(), delay=5.0), timeout_s=0.05)
  t0 = time.monotonic()
  boss.start({})
  assert time.monotonic() - t0 < 0.05, "start() must not block"
  while boss.pending:
    time.sleep(0.005)
  assert time.monotonic() - t0 < 4.0, "the deadline, not the call, releases us"
  assert boss.result({}).source == "fallback:timeout"


def test_the_call_budget_is_hard(menu):
  client = FakeClient(full(action="dance", reason="."))
  boss = Overseer(menu, client=client, calls_per_hour=2)
  sources = [boss.decide({"decisions": i}).source for i in range(4)]
  assert sources[:2] == ["llm", "llm"]
  assert sources[2:] == ["fallback:budget", "fallback:budget"]
  # The budget is not advice: the client was asked exactly twice.
  assert len(client.calls) == 2
  assert boss.budget_left() == 0


def test_back_to_back_decisions_all_reach_the_model(menu):
  """Publishing the answer and releasing the in-flight flag must be ONE
  critical section.

  `result()` returns the moment `_slot` is set, so anything between setting it
  and clearing `_in_flight` is a window where the caller has its answer and
  the next `start()` still believes a call is running -- and silently refuses
  to make one. First version had the two split by a `_meter()` call and a lock
  re-acquisition, which looked harmless.

  ⚠ THE BURNERS ARE THE TEST. Serially on an idle machine this passes with the
  bug in place; the window only opens under GIL contention. Measured with the
  defect reintroduced, at this exact size: **1 of 40** decisions reached the
  model and 39 came back scripted. That is why it escaped a serial run and
  only surfaced in the full parallel suite.
  """
  stop = threading.Event()

  def burn():
    x = 0
    while not stop.is_set():
      x = (x + 1) % 1000003

  for _ in range(BURNERS):
    threading.Thread(target=burn, daemon=True).start()
  try:
    client = FakeClient(full(action="dance", reason="."))
    boss = Overseer(menu, client=client, calls_per_hour=1000)
    sources = [boss.decide({"decisions": i}).source for i in range(DECISIONS)]
  finally:
    stop.set()

  assert len(client.calls) == DECISIONS, \
    f"only {len(client.calls)} of {DECISIONS} decisions reached the model"
  assert set(sources) == {"llm"}, Counter(sources)


def test_the_budget_is_a_rolling_hour(menu):
  now = [0.0]
  boss = Overseer(menu, client=FakeClient(full()), calls_per_hour=1,
                  clock=lambda: now[0])
  assert boss.decide({}).source == "llm"
  assert boss.decide({}).source == "fallback:budget"
  now[0] = 3601.0
  assert boss.decide({}).source == "llm"


def test_a_dead_endpoint_backs_off_instead_of_hammering(menu):
  """Kill the API and the robot keeps working -- but it must also stop
  asking. A missing key does not fail at client construction (measured: it
  raises on the first REQUEST), so without this a keyless deploy burns its
  whole hourly budget on calls that cannot succeed."""
  client = FakeClient(RuntimeError("no route to host"))
  boss = Overseer(menu, client=client, calls_per_hour=60)
  for _ in range(ov.MAX_CONSECUTIVE_ERRORS):
    assert boss.decide({}).source.startswith("fallback:RuntimeError")
  assert len(client.calls) == ov.MAX_CONSECUTIVE_ERRORS
  assert boss.decide({}).source == "fallback:cooloff"
  assert len(client.calls) == ov.MAX_CONSECUTIVE_ERRORS, "still calling out"
  assert boss.stats()["cooloffS"] > 0


def test_a_recovered_endpoint_is_used_again(menu):
  now = [0.0]
  client = FakeClient(*([RuntimeError("blip")] * ov.MAX_CONSECUTIVE_ERRORS),
                      full(action="dance", reason="."))
  boss = Overseer(menu, client=client, clock=lambda: now[0])
  for _ in range(ov.MAX_CONSECUTIVE_ERRORS):
    boss.decide({})
  assert boss.decide({}).source == "fallback:cooloff"
  now[0] += ov.COOLOFF_BASE_S + 1.0
  assert boss.decide({}).source == "llm"


def test_it_cannot_idle_its_life_away(menu):
  """`idle` and `journal` cost nothing and do nothing. Two in a row is a
  pause; a third would be a robot narrating a life it is not living, so the
  scripted policy takes the turn."""
  boss = make(menu, full(action="journal", note="thinking about it"))
  sources = [boss.decide({"decisions": i, "tasksThisMission": []}).source
             for i in range(ov.MAX_IDLE_RUN + 1)]
  assert sources[:ov.MAX_IDLE_RUN] == ["llm"] * ov.MAX_IDLE_RUN
  assert sources[-1] == "fallback:idle-run"
  assert boss.decisions[-1].action not in ov.IDLE_ACTIONS


# ---- the prompt --------------------------------------------------------------


def test_the_stable_prefix_is_byte_identical_across_calls(menu):
  """The prompt-cache prerequisite, and the cheapest possible guard against
  the classic silent invalidator. If someone puts a timestamp, a battery
  reading or a note into the system prompt, `cache_read_input_tokens` goes to
  zero and NOTHING ELSE BREAKS -- the bill just quietly grows."""
  a, b = Overseer(menu, client=FakeClient()), Overseer(menu,
                                                       client=FakeClient())
  assert a.system == b.system
  assert a.system[0]["cache_control"] == {"type": "ephemeral"}
  text = a.system[0]["text"]
  # Quoted JSON keys, not bare words: the RULES prose talks ABOUT the battery
  # and the journal, which is stable text and entirely fine. What must never
  # appear is a key from the volatile turn, or today's date.
  for key in ('"simTimeS"', '"reserveWh"', '"recentTasks"',
              '"tasksThisMission"', '"visitorSuggestions"', '"thoughts"'):
    assert key not in text, f"{key} belongs in the user turn"
  assert time.strftime("%Y") not in text, "a timestamp in the cached prefix"
  # ...and the thought files (issue #38), which is the same trap with a new
  # door: the two a HUMAN writes are constants within a run and belong here,
  # while the two that change during one would invalidate this prefix on
  # every self-edit. Their CONTENT, not their names -- the rules prose names
  # all four, which is stable text and exactly right, the same distinction
  # the comment above draws. tests/test_thoughts.py holds the rest.
  boss = Overseer(menu, client=FakeClient())
  boss.thoughts.learn("this is a thing I worked out", t=1.0)
  boss.thoughts.record("this is a thing that happened", t=2.0)
  assert boss.system[0]["text"] == text, "a self-edit moved the cached prefix"
  assert "this is a thing I worked out" not in text
  assert "this is a thing that happened" not in text


def test_the_prompt_never_carries_a_hidden_answer(menu):
  """The census knows how many plants are really in the garden. The overseer
  must not, or the task whose whole point is going and counting arrives
  pre-solved in its own context."""
  text = json.dumps(default_table().as_context())
  assert "secret" not in text
  assert "truth" not in text
  # ...and the same rule downstream: a banked census verdict is redacted by
  # `Verdict.public_metrics`, which is what `context_for` replays.
  assert "truth" not in json.dumps(
    [r for r in default_table().as_context() if r["task"] == "census"])


def test_the_context_is_the_live_lifecycle_and_carries_no_truth(menu):
  """`context_for` reads the running robot rather than a parallel tally, so
  what the overseer is told cannot drift from what the robot is."""
  life = _lifecycle("room_hub")
  life.verdicts.append({"task": "census", "ok": False, "points": 0,
                        "reason": "reported 3 in garden (wrong)",
                        "metrics": {"counted": 3, "coverage": 0.4}})
  state = ov.context_for(life, Journal())
  assert state["points"] == 0
  assert state["battery"]["fraction"] == pytest.approx(1.0, abs=0.01)
  assert state["tasksThisMission"] == ["census"]
  assert state["visitorMessages"] == []         # nobody has said anything
  assert "truth" not in json.dumps(state)


# ---- the scripted policy is a real policy ------------------------------------


def test_the_fallback_rotates_rather_than_repeating(menu):
  seen = []
  for i in range(4):
    d = scripted(menu, {"tasksThisMission": seen, "decisions": i,
                        "mapDone": False}, "budget")
    seen.append(d.action)
  assert len(set(seen)) == len(seen), f"the fallback repeated itself: {seen}"


def test_the_fallback_is_deterministic(menu):
  state = {"tasksThisMission": ["draw"], "decisions": 3, "mapDone": False}
  runs = [scripted(menu, state, "budget").as_dict() for _ in range(5)]
  assert all(r == runs[0] for r in runs)


def test_the_fallback_still_has_something_to_do_when_everything_is_done(menu):
  d = scripted(menu, {"tasksThisMission": ["draw", "census", "dance", "carry"],
                      "decisions": 1, "mapDone": True}, "budget")
  assert d.action in menu.available()


# ---- decisions become errands, through the same builders as the presets ------


@pytest.mark.parametrize("action, module", [
  ("draw", "module_pen"), ("census", "module_lcd"), ("dance", "module_lcd"),
  ("carry", "module_lcd"),
])
def test_every_task_action_builds_a_real_errand(book, action, module):
  e = errand_from(Decision(action=action, board="whiteboard_a",
                           program="house"), "home", book)
  assert e is not None and e.module == module
  assert e.task == action


@pytest.mark.parametrize("action", ["idle", "journal", "explore", "charge"])
def test_the_non_errand_actions_build_no_errand(book, action):
  assert errand_from(Decision(action=action), "home", book) is None


def test_an_impossible_errand_is_none_rather_than_an_exception(book):
  """A decision is untrusted input in exactly the way a visitor message will
  be (issue #16). The mission loop's answer to "I cannot do that" is to ask
  again, never to end."""
  assert errand_from(Decision(action="draw", board="whiteboard_a"),
                     "room_hub", None) is None


def test_a_zone_resolves_to_somewhere_inside_it():
  x, y = zone_centre("home", "garden")
  garden = next(z for z in world_config("home")["zones"]
                if z["name"] == "garden")
  assert garden["min"][0] <= x <= garden["max"][0]
  assert garden["min"][1] <= y <= garden["max"][1]
  with pytest.raises(ValueError):
    zone_centre("home", "the_attic")


# ---- memory ------------------------------------------------------------------


def test_the_journal_persists_and_is_bounded(tmp_path):
  path = tmp_path / "journal.json"
  j = Journal(path)
  for i in range(5):
    j.note(f"note {i}", t=float(i))
  assert j.note("") is None, "an empty note is cost with no content"
  assert len(Journal(path)) == 5
  assert [n["text"] for n in Journal(path).recent(2)] == ["note 3", "note 4"]
  long = Journal(path).note("x" * (MAX_NOTE_CHARS * 3))
  assert len(long["text"]) == MAX_NOTE_CHARS


def test_the_journal_streams_every_note_as_it_happens(tmp_path):
  """The acceptance criterion "journal entries stream and appear on the
  site". The site runs on a different box and cannot read this file."""
  seen = []
  j = Journal(tmp_path / "j.json")
  j.on_event.append(seen.append)
  j.note("the pen was low on the bracket again", t=12.5)
  assert len(seen) == 1 and seen[0]["t"] == 12.5


def test_goals_are_read_and_never_written(tmp_path):
  path = tmp_path / "goals.md"
  assert read_goals(None) == read_goals(path)      # missing -> the defaults
  path.write_text("Draw a robot on every wall.\n")
  assert read_goals(path).strip() == "Draw a robot on every wall."
  Overseer(Menu(boards=("a",), programs=("house",)), goals=read_goals(path),
           client=FakeClient()).decide({})
  assert path.read_text() == "Draw a robot on every wall.\n"


# ---- cost accounting ---------------------------------------------------------


def test_what_it_cost_is_measured_and_reported(menu):
  boss = Overseer(menu, client=FakeClient(
    full(), usage={"input_tokens": 1000, "output_tokens": 100,
                   "cache_read_input_tokens": 4000}))
  boss.decide({})
  stats = boss.stats()
  assert stats["inputTokens"] == 1000 and stats["cacheReadTokens"] == 4000
  # $1/MTok in, $5/MTok out, cache reads at a tenth.
  assert stats["usd"] == pytest.approx((1000 + 400 + 500) / 1e6, rel=1e-6)
  assert stats["cacheHitRate"] == pytest.approx(0.8, abs=1e-4)
  assert stats["model"] == "claude-haiku-4-5"


def test_the_model_is_the_one_the_issue_chose():
  assert ov.MODEL == "claude-haiku-4-5"


def test_effort_is_never_sent(menu):
  """`output_config.effort` is not supported on Haiku 4.5 and returns a 400
  there -- so structured outputs go in `output_config` and nothing else
  does."""
  client = FakeClient(full())
  Overseer(menu, client=client).decide({})
  config = client.calls[0]["output_config"]
  assert set(config) == {"format"}
  assert config["format"]["type"] == "json_schema"
  assert "thinking" not in client.calls[0]


def test_the_overseer_is_off_unless_asked_for(monkeypatch, book):
  monkeypatch.delenv(ov.ENABLE_ENV, raising=False)
  assert ov.build("home", book) == (None, None)
  boss, journal = ov.build("home", book, enabled=True, client=FakeClient())
  assert boss is not None and journal is not None


# ---- the mission ------------------------------------------------------------


def _lifecycle(world: str, **kw) -> HubLifecycle:
  cfg = world_config(world)
  model = mujoco.MjModel.from_xml_path(cfg["model"])
  data = mujoco.MjData(model)
  return HubLifecycle(model, data, realtime=False, world=world,
                      battery_wh=cfg["battery_wh"], rack=cfg["rack"],
                      grid_bounds=cfg["grid_bounds"],
                      low_battery_wh=cfg["low_battery_wh"], **kw)


def test_the_arbitration_loop_is_untouched_without_an_overseer():
  """Every existing demo, mission test and recording must behave exactly as
  it did. The overseer being opt-in is what makes that true."""
  life = _lifecycle("room_hub")
  assert life.overseer is None and life.journal is None
  assert life.decisions == []
  assert "overseer" in life.__dict__


@pytest.mark.slow
def test_charge_priority_survives_an_overseer_that_never_charges():
  """THE regression test for issue #15: the branch ORDER.

  A robot that starts below its reserve, and an overseer that answers `idle`
  to every question it is ever asked. It must charge first and be asked
  second, because `needs_charge` is checked before the overseer is reached and
  there is no action in the vocabulary that suppresses it.

  Shown to fail without the fix: move the `elif self.overseer is not None`
  branch above the `if self.needs_charge` branch in `HubLifecycle.run()` and
  this reports DECIDE before GO_CHARGE and zero charge cycles.
  """
  boss = Overseer(Menu.for_world("room_hub", None),
                  client=FakeClient(full(action="idle",
                                         reason="I would rather not")))
  life = _lifecycle("room_hub", overseer=boss, journal=Journal(), errand=False)
  # Below the reserve at t=0. The cheapest state that puts the two branches in
  # direct conflict: the robot needs to charge AND is being told not to bother.
  life.battery.energy_wh = life.low_battery_wh * 0.6
  states: list[str] = []
  life.mission.step_hooks.append(
    lambda: states.append(life.state)
    if life.state != (states[-1] if states else None) else None)

  r = life.run(world_config("room_hub")["start"], max_sim_time=200.0,
               explore_budget=10.0)

  assert "GO_CHARGE" in states, "an idling overseer bricked the robot"
  assert "DECIDE" in states, "the overseer was never consulted at all"
  assert states.index("GO_CHARGE") < states.index("DECIDE"), \
    f"the LLM was asked before the robot charged: {states}"
  assert r["charge_cycles"] >= 1
  # ...and it really was the LLM being overruled, not the fallback covering
  # for it -- an overseer that never answered would prove nothing here.
  assert any(d["source"] == "llm" for d in r["decisions"])
  assert r["overseer"]["llmCalls"] >= 1


def test_a_chosen_drawing_becomes_an_errand_for_that_exact_board(book):
  """The overseer names a board and a figure, and both survive into the errand.

  Deliberately NOT a whole mission. The claim worth testing here is that a
  chosen drawing goes through the same `draw_errand_for` the preset queue does
  -- a second drawing path is exactly what issue #12 spent itself removing --
  and that is settled at the errand, not at the pen. Whether the pen then puts
  ink on a board is `test_drawing.py`'s and the full-lifecycle test's job, and
  paying three minutes of mission time to re-confirm it here would buy nothing.
  """
  a = errand_from(Decision(action="draw", board="whiteboard_b",
                           program="tree"), "home", book)
  assert a.detail == {"board": "whiteboard_b", "figure": "tree",
                      "strokes": a.detail["strokes"],
                      "ink_m": a.detail["ink_m"]}
  assert a.name == "draw:whiteboard_b" and a.task == "draw"
  # ...and a different choice really is a different errand, so a board id
  # threaded through by accident would show up here.
  b_errand = errand_from(Decision(action="draw", board="whiteboard_a",
                                  program="sun"), "home", book)
  assert b_errand.name == "draw:whiteboard_a"
  assert b_errand.detail["figure"] == "sun"
  assert b_errand.use_at != a.use_at, "both figures drove to the same board"


def test_a_note_reaches_the_journal_and_the_narration():
  """A decision's note is written once, streamed once, and readable next
  time -- the loop that makes the journal memory rather than a log."""
  boss = Overseer(Menu.for_world("room_hub", None),
                  client=FakeClient(full(action="carry", reason="tidying up",
                                         note="bay A sticks a little")))
  journal = Journal()
  streamed: list[dict] = []
  journal.on_event.append(streamed.append)
  life = _lifecycle("room_hub", overseer=boss, journal=journal, errand=False)
  said: list[str] = []
  life.say_hooks.append(lambda t, line: said.append(line))
  life.mission.start_at(*world_config("room_hub")["start"])
  try:
    life._decide()
  finally:
    life.mission.close()

  assert [n["text"] for n in journal.recent()] == ["bay A sticks a little"]
  assert len(streamed) == 1
  assert any(line.startswith("JOURNAL bay A sticks") for line in said)
  assert any("DECIDE carry: tidying up" in line for line in said)
  # The decision produced a real errand, queued for the loop rather than run
  # inline -- so if it dropped the battery, the next pass charges first.
  assert [e.task for e in life.errands] == ["carry"]


def test_a_full_battery_cannot_be_charged_for_points():
  """`charge` is a scored task and the trip to the rack costs energy, so an
  unconditional `charge` action is perpetual motion paid in points: spend
  battery driving out, earn points putting it back, repeat.

  Shown to fail without the fix: drop the `TOP_UP_BELOW` guard in
  `HubLifecycle._decide` and a robot at 100 % drives to the rack and banks a
  charge verdict for topping up what the trip itself spent.
  """
  boss = Overseer(Menu.for_world("room_hub", None),
                  client=FakeClient(full(action="charge",
                                         reason="might as well")))
  life = _lifecycle("room_hub", overseer=boss, errand=False)
  life.mission.start_at(*world_config("room_hub")["start"])
  said: list[str] = []
  life.say_hooks.append(lambda t, line: said.append(line))
  try:
    assert life.battery.fraction > 0.75
    life._decide()
  finally:
    life.mission.close()

  assert life.charge_cycles == 0, "a full robot drove to the rack for points"
  assert life.verdicts == [], "it banked a charge verdict it did not earn"
  assert any("not worth a trip" in line for line in said)
  # The forced charge is untouched -- `needs_charge` is absolute energy
  # against the worst return trip and never consults this floor.
  life.battery.energy_wh = life.low_battery_wh * 0.5
  assert life.needs_charge


def test_the_sim_keeps_running_while_the_overseer_thinks():
  """A slow API must cost the robot a pause, not the world a freeze. The
  telemetry stream is built off physics steps, so a blocking call here would
  stop every viewer's clock for the length of an HTTP request."""
  boss = Overseer(Menu.for_world("room_hub", None),
                  client=FakeClient(full(action="carry", reason="."),
                                    delay=0.6),
                  timeout_s=2.0)
  life = _lifecycle("room_hub", overseer=boss, errand=False)
  life.mission.start_at(*world_config("room_hub")["start"])
  life.max_sim_time = 60.0
  life.explore_deadline = life.data.time + 1.0
  life.blacklist, life.map_done = set(), True
  steps = []
  life.mission.step_hooks.append(lambda: steps.append(life.data.time))
  t0 = life.data.time
  life._decide()
  life.mission.close()
  assert life.data.time > t0, "the sim did not advance while it was thinking"
  assert len(steps) > 100, f"only {len(steps)} physics steps during the call"
  assert life.decisions[0]["source"] == "llm"
  assert math.isclose(life.data.time - t0, 0.6, abs_tol=0.5)
