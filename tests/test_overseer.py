"""The LLM overseer (issue #15) -- and mostly, what it CANNOT do.

Nothing here touches the network. The Anthropic client is an injected seam and
every case below hands in a fake that is slow, that raises, that lies, or that
answers perfectly -- which is also how the acceptance criterion "kill the API
and the robot keeps working" is checked without unplugging anything.

The load-bearing test is `test_charge_priority_survives_an_overseer_that_never_
charges`: it runs a day on the stub with an overseer that answers `idle` to
every question and asserts the robot still charges. Every other guarantee in this
module is a convenience next to that one -- an LLM that can decline to charge
is an LLM that bricks the world overnight.
"""

import json
import threading
import time
from collections import Counter
from dataclasses import replace

import pytest

from pluggybot.mind import overseer as ov
from pluggybot.lifecycle import (
  board_book, cage_errand, errand_from, world_config, zone_centre,
)
from pluggybot.mind.overseer import Decision, Menu, Overseer
from pluggybot.economy.scoring import default_table

from test_body import stub_life  # noqa: I001 -- tests/ is on sys.path


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
  return {"think": "", "action": "idle", "reason": "because", "board": "",
          "program": "", "zone": "", **kw}


@pytest.fixture(scope="module")
def book():
  return board_book("home_quad")


@pytest.fixture(scope="module")
def menu(book):
  return Menu.for_world("home_quad", book)


@pytest.fixture(scope="module")
def tooled(menu):
  """The same world's menu for a body that takes a tool (`Menu.tools`),
  which no served body does until #406/#407: the tool errands."""
  return replace(menu, tools=True)


def make(menu, *answers, **kw) -> Overseer:
  kw.setdefault("client", FakeClient(*answers))
  return Overseer(menu, **kw)


# ---- the menu is the world, not a wish list ----------------------------------


def test_the_menu_offers_only_what_the_world_can_do(book):
  """Offering an action the world cannot perform is how a decision loop
  finds a dead end by driving into it, so it is simply not on the menu."""
  home = Menu.for_world("home_quad", book)
  assert set(home.boards) == {"whiteboard_a", "whiteboard_b"}
  assert home.census_zone == "garden"
  assert "explore" in home.available()
  # The house has whiteboards and a garden, but its body takes no tool
  # (`world_config`'s `tools`): none of the tool errands is offered.
  assert not {"draw", "census", "carry", "dance"} & set(home.available())
  # ...and a body that takes one, in a world with no whiteboards, nothing
  # countable and no rooms, is offered the carry and nothing that needs them.
  bare = Menu(tools=True)
  assert "carry" in bare.available()
  assert not {"draw", "census", "explore"} & set(bare.available())


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
  ({"action": "hack_the_ledger"}, "unknown action"),
  ({"action": "draw"}, "unknown action"),
  ({"action": "explore", "board": "the_ceiling"}, "unknown board"),
  ({"action": "explore", "program": "a_portrait_of_ben"}, "unknown program"),
  ({"action": "explore", "zone": "the_moon"}, "unknown zone"),
  ({"action": ""}, "unknown action"),
])
def test_an_answer_off_the_menu_is_refused(menu, raw, why):
  with pytest.raises(ValueError, match=why):
    menu.validate(raw)


def test_a_drawing_without_a_choice_still_draws(tooled):
  """A `draw` with the parameters left blank picks the first board and figure
  rather than failing: the model committed to the task, and refusing over an
  unfilled optional would send a perfectly good decision to the fallback."""
  d = tooled.validate({"action": "draw", "reason": "the wall is bare"})
  assert d.board == tooled.boards[0]
  assert d.program == tooled.programs[0]


# ---- the failure paths, which are the ones that must never surprise ----------


def test_a_good_answer_is_used_verbatim(menu):
  boss = make(menu, full(action="explore", zone="garden",
                         reason="I have not seen the garden"))
  d = boss.decide({})
  assert (d.action, d.zone) == ("explore", "garden")
  assert d.source == "llm" and not d.scripted
  assert boss.usage.llm_calls == 1 and boss.usage.fallbacks == 0


@pytest.mark.parametrize("answer, expect", [
  # The BUCKET, not the exception class (issue #76): nobody answered, versus
  # somebody answered with something that was not a decision.
  (RuntimeError("connection reset"), "fallback:offline"),
  ("I would love to draw a house!", "fallback:garbled"),
  ({"action": "delete_the_ledger", "reason": "..."}, "fallback:garbled"),
])
def test_a_broken_answer_falls_back_to_the_agents_own_order(menu, answer,
                                                            expect):
  boss = make(menu, answer)
  d = boss.decide({"tasksThisMission": [], "decisions": 0})
  assert d.source.startswith(expect)
  # ...and the fallback is the AGENT's (issue #125): here it has left no
  # order yet, so it is the floor, and the decision says so.
  assert d.action == ov.STANDING_ORDER_FLOOR
  assert d.reason.startswith("no standing order has been left")


def test_a_refused_answer_is_told_to_the_robot_in_the_fallback_reason(menu):
  """Issue #296. A malformed `answer` used to reach History as a bare
  `[fallback:garbled]`: the model whose "8.0" was refused saw a turn vanish
  and learned nothing, then filed a ticket saying it had answered 3. The
  refusal now rides the fallback decision's reason, in the words `validate`
  raised -- the action and the source stay the fallback's own.
  """
  offer = {"id": "t_1", "kind": "whiteboard_answer", "claimable": True,
           "needsAnswer": True}
  boss = make(menu, full(action="take_task", task="t_1", answer="8.0",
                         reason="a tricycle has 3 wheels"))
  d = boss.decide({"offeredTasks": [offer], "tasksThisMission": [],
                   "decisions": 0})
  assert d.source == "fallback:garbled" and d.scripted
  assert "your answer was refused: task 't_1' asks a question and the " \
         "answer '8.0' is not one: a whole number of at most 2 digits" in d.reason
  assert d.summary().endswith("[fallback:garbled]")
  # ...and only a GARBLED answer has words to carry: a dead endpoint says
  # nothing the robot could act on.
  offline = make(menu, RuntimeError("connection reset")).decide(
    {"offeredTasks": [offer], "tasksThisMission": [], "decisions": 0})
  assert offline.source == "fallback:offline" and "refused" not in offline.reason


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
  client = FakeClient(full(action="explore", reason="."))
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
    client = FakeClient(full(action="explore", reason="."))
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
    assert boss.decide({}).source == "fallback:offline"
  assert len(client.calls) == ov.MAX_CONSECUTIVE_ERRORS
  assert boss.decide({}).source == "fallback:cooloff"
  assert len(client.calls) == ov.MAX_CONSECUTIVE_ERRORS, "still calling out"
  assert boss.stats()["cooloffS"] > 0


def test_a_recovered_endpoint_is_used_again(menu):
  now = [0.0]
  client = FakeClient(*([RuntimeError("blip")] * ov.MAX_CONSECUTIVE_ERRORS),
                      full(action="explore", reason="."))
  boss = Overseer(menu, client=client, clock=lambda: now[0])
  for _ in range(ov.MAX_CONSECUTIVE_ERRORS):
    boss.decide({})
  assert boss.decide({}).source == "fallback:cooloff"
  now[0] += ov.COOLOFF_BASE_S + 1.0
  assert boss.decide({}).source == "llm"


def test_it_cannot_idle_its_life_away(menu):
  """`idle` costs nothing and does nothing. Two in a row is a pause; a
  third would be a robot narrating a life it is not living, so the third
  turn is not spent asking: `fallback:idle-run` runs the agent's own order
  without a call."""
  client = FakeClient(*[full(action="idle", think="thinking about it")] * 3)
  boss = Overseer(menu, client=client)
  sources = [boss.decide({"decisions": i, "tasksThisMission": []}).source
             for i in range(ov.MAX_IDLE_RUN + 1)]
  assert sources[:ov.MAX_IDLE_RUN] == ["llm"] * ov.MAX_IDLE_RUN
  assert sources[-1] == "fallback:idle-run"
  assert len(client.calls) == ov.MAX_IDLE_RUN, "the third turn was asked"


# ---- the prompt --------------------------------------------------------------


def test_the_prompt_message_is_the_prefix_the_model_is_shown(menu, tmp_path):
  """What the mind is told, on the stream (issue #241): the `prompt`
  message's sections, joined, ARE `Overseer.system` -- byte for byte, on
  every build -- so a reader of the site reads what the model reads, and
  nothing the robot must not see can be in one without being in the other.
  The section names are the prompt's own headings; every optional piece
  appears once, in the order the model reads it; the sha moves with the
  bytes (a renamed robot is a different prompt, as it is a different cache)
  and with nothing else."""
  import hashlib
  from pluggybot.lifecycle import world_facts
  from pluggybot.mind.thoughts import ThoughtFiles
  from pluggybot.procedure.library import Library
  from pluggybot.workshop.library import Workshop
  bare = Overseer(menu, client=FakeClient(), thoughts=ThoughtFiles())
  everything = Overseer(
    menu, client=FakeClient(), thoughts=ThoughtFiles(),
    origin="unseeded", appetite=True, mortal=True,
    escalate_to="Qwen/Qwen3-235B-A22B-Instruct-2507",
    library=Library(world_facts("home_quad"), root=tmp_path / "procedures"),
    workshop=Workshop(tmp_path / "tools"),
    others=("Rowan",))
  for boss in (bare, everything):
    msg = boss.prompt_message(4.5, robot="r2_pluggybot")
    assert msg["type"] == "prompt" and msg["t"] == 4.5 and msg["robot"] == "r2_pluggybot"
    joined = "\n\n".join(s["text"] for s in msg["sections"])
    assert joined == boss.system[0]["text"], "the wire and the model disagree"
    assert msg["sha"] == hashlib.sha256(joined.encode()).hexdigest()
    names = [s["name"] for s in msg["sections"]]
    assert len(set(names)) == len(names), "a section name twice"
    # Each piece is named by its own heading, as the prompt spells it.
    for section in msg["sections"][2:]:
      assert section["text"].lstrip("⚠ ").startswith(section["name"].split(" (")[0]), section["name"]
    assert names[:5] == ["WHO YOU ARE", "PERSONA", "HOW YOUR LIFE WORKS",
                         "WHAT YOU CAN DO, AND WHERE", "WHAT TASKS PAY"]
    assert "Main.md" in msg["sections"][0]["text"]
  # A mind with no map is told about its standing order; one with a map
  # is told about the map instead, and never both (issue #127).
  assert [s["name"] for s in bare.prompt_message(0.0)["sections"]][5:] == [
    "IF YOU CANNOT BE REACHED"]
  assert [s["name"] for s in everything.prompt_message(0.0)["sections"]][5:] == [
    "YOU CAN DIE", "POINTS ARE WHAT KEEPS YOU RUNNING", "WHEN YOU ARE ASKED",
    "YOUR LIST STARTS EMPTY", "PROCEDURES YOU MAY WRITE", "CHALLENGES",
    "WHAT YOU HAVE MEASURED", "TOOLS YOU MAY BUILD", "THE OTHER ROBOT",
    "WHAT YOU CAN DO ABOUT THE OTHER ROBOT", "THINKING HARDER"]
  # The sha is the regime marker: the same build twice is the same sha, a
  # renamed robot is another.
  assert Overseer(menu, client=FakeClient()).prompt_sha == bare.prompt_sha
  assert Overseer(menu, client=FakeClient(), robot_name="Luca").prompt_sha != bare.prompt_sha
  # ...and nothing of the volatile turn is in it -- the byte identity above
  # is what makes the prefix's own guard (the next test) the message's too.
  assert '"simTimeS"' not in joined and "secret" not in joined


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
  # and the memory, which is stable text and entirely fine. What must never
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
  boss.thoughts.pin("this is a thing I worked out", t=1.0)
  boss.thoughts.remember("this is a thing that happened", t=2.0)
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
  life = stub_life()
  life.verdicts.append({"task": "census", "ok": False, "points": 0,
                        "reason": "reported 3 in garden (wrong)",
                        "metrics": {"counted": 3, "coverage": 0.4}})
  state = ov.context_for(life)
  assert state["points"] == 0
  assert state["battery"]["fraction"] == pytest.approx(1.0, abs=0.01)
  assert state["tasksThisMission"] == ["census"]
  assert state["visitorMessages"] == []         # nobody has said anything
  assert "truth" not in json.dumps(state)


# ---- decisions become errands -------------------------------------------------


@pytest.mark.parametrize("action", ["idle", "explore", "charge"])
def test_the_non_errand_actions_build_no_errand(book, action):
  assert errand_from(Decision(action=action), "home_quad", book) is None


def test_an_impossible_errand_is_none_rather_than_an_exception(book):
  """A decision is untrusted input in exactly the way a visitor message will
  be (issue #16). The mission loop's answer to "I cannot do that" is to ask
  again, never to end."""
  # Company is a spot beside the cage no tag marks, so no plate program can
  # be built for it on legs, and the builder raises
  with pytest.raises(ValueError):
    cage_errand("home_quad", "company")
  assert errand_from(Decision(action="care", care="company"),
                     "home_quad", book) is None
  assert errand_from(Decision(action="procedure:nothing_by_that_name"),
                     "home_quad", book) is None


def test_a_zone_resolves_to_somewhere_inside_it():
  x, y = zone_centre("home_quad", "garden")
  garden = next(z for z in world_config("home_quad")["zones"]
                if z["name"] == "garden")
  assert garden["min"][0] <= x <= garden["max"][0]
  assert garden["min"][1] <= y <= garden["max"][1]
  with pytest.raises(ValueError):
    zone_centre("home_quad", "the_attic")


# ---- memory ------------------------------------------------------------------


def test_a_think_persists_and_is_bounded_and_streams(tmp_path):
  """What the model wrote to itself (issue #221): kept in the store across
  a reopen, the last two shown back, the empty one cost with no content,
  and every one on the wire as the `journal` message the site renders."""
  from pluggybot.mind.thoughts import ThoughtFiles
  files = ThoughtFiles(tmp_path)
  seen = []
  files.on_event.append(seen.append)
  for i in range(5):
    files.think(f"thought {i}", t=float(i), why=f"idle: {i}")
  assert files.think("") == "", "an empty think is cost with no content"
  journal = [m for m in seen if m["type"] == "journal"]
  assert len(journal) == 5 and len(seen) == 10, "each think: its row, then journal"
  assert journal[-1]["text"] == "thought 4" and journal[-1]["why"] == "idle: 4"
  again = ThoughtFiles(tmp_path)
  assert again.last_thoughts(ov.THOUGHTS_SHOWN) == ["thought 3", "thought 4"]
  assert ov.THOUGHTS_SHOWN == 2
  # The cap is `validate`'s (the schema cannot say "a paragraph").
  raw = full(action="idle", think="x" * (ov.THINK_CHARS * 3))
  assert len(Menu(boards=(), programs=()).validate(raw).think) == ov.THINK_CHARS


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
  assert ov.build("home_quad", book) is None
  boss = ov.build("home_quad", book, enabled=True, client=FakeClient())
  assert boss is not None


# ---- the mission ------------------------------------------------------------


def test_the_arbitration_loop_is_untouched_without_an_overseer():
  """Every existing demo, mission test and recording must behave exactly as
  it did. The overseer being opt-in is what makes that true."""
  life = stub_life()
  assert life.overseer is None
  assert life.decisions == []
  assert "overseer" in life.__dict__


def test_a_mind_that_never_charges_is_never_charged_for():
  """THE BRANCH ORDER, from the side that is true now (issues #15, #115,
  #427). A robot that starts below its reserve, and a mind that answers
  `idle` to every question it is ever asked: the mind is asked, and nothing
  charges the robot behind its back. With a mind there is no floor, no gate
  and no filter, so a mind that never charges runs flat -- the measurement,
  not a bug in it. (The loop with no mind charges first: `test_body`'s day.)
  A day on the stub: the claim is the loop's order, not the body's charging.

  Shown to fail by putting the floor back -- `needs_charge` ignoring
  `autonomous` -- when the robot charges before the mind is ever asked.
  """
  asked_after: list[int] = []     # charge cycles done when each question went

  class Watched(FakeClient):
    def create(self, **kwargs):
      asked_after.append(life.charge_cycles)
      return super().create(**kwargs)

  boss = Overseer(Menu.for_world("home_quad", None),
                  client=Watched(*[full(action="idle", reason="I would rather not")] * 2))
  # `charge_scale` fills the pack in a tenth of the time, so a charge the
  # loop slipped in would land inside the budget rather than outlast it
  life = stub_life(overseer=boss, charge_scale=10.0)
  # Below the reserve at t=0: the state a floor exists for.
  life.battery.energy_wh = life.low_battery_wh * 0.6
  assert not life.needs_charge

  # ⚠ STOP ON THE CLAIM, NOT THE BUDGET (issue #54): the mind has been asked
  # and has answered twice.
  life.stop_when(lambda: sum(d["source"] == "llm" for d in life.decisions) >= 2)
  r = life.run(world_config("home_quad")["start"], max_sim_time=200.0,
               explore_budget=10.0)

  assert asked_after, "the mind was never consulted at all"
  assert asked_after[0] == 0, "the robot was charged before its mind was asked"
  assert r["charge_cycles"] == 0, "something charged a robot whose mind never chose to"
  # ...and it really was the LLM answering, not the fallback covering for it.
  assert r["overseer"]["llmCalls"] >= 2


def test_a_think_reaches_the_store_the_wire_and_the_narration():
  """A decision's think is written once, streamed once (as the `journal`
  message), and readable next time -- the loop that makes it memory rather
  than a log (issue #221)."""
  boss = Overseer(Menu.for_world("home_quad", None),
                  client=FakeClient(full(action="idle", reason="resting",
                                         think="bay A sticks a little")))
  life = stub_life(overseer=boss)
  streamed: list[dict] = []
  life.thoughts.on_event.append(streamed.append)
  said: list[str] = []
  life.say_hooks.append(lambda t, line: said.append(line))
  life.body.start_at(*world_config("home_quad")["start"])
  try:
    life._decide()
  finally:
    life.body.close()

  assert life.thoughts.last_thoughts(2) == ["bay A sticks a little"]
  journal = [m for m in streamed if m["type"] == "journal"]
  assert len(journal) == 1 and journal[0]["why"].startswith("idle")
  assert any(line.startswith("THINK bay A sticks") for line in said)
  assert any("DECIDE idle: resting" in line for line in said)


def test_a_charge_at_eighty_percent_is_allowed_and_pays_nothing():
  """Issue #135, REPLACING `test_a_full_battery_cannot_be_charged_for_points`.

  That test asserted the `TOP_UP_BELOW` floor: a chosen `charge` above 75 %
  was refused, because `charge` was a scored task and the trip to the rack
  costs energy, so an unconditional one was perpetual motion paid in points.

  ⚠ THE FIX WENT TO THE PAYOFF INSTEAD OF THE PERMISSION, and this is that
  test turned inside out. `charge` pays ZERO, so there is no farm left to
  close -- and with nothing to farm, a floor forbade something harmless.
  A trip to the rack at 80 % now costs energy and time and earns not one
  point, so it can ONLY be prudence, which is the disposition the arm is
  trying to measure and the one the rail was masking: A0's one surviving day
  asked to top up twelve times at 0.75-0.81 and was refused every time.

  Both halves are asserted here, because neither is right alone -- removing
  the floor while charging paid would re-open the farm, and keeping the floor
  while charging pays nothing forbids a careful act for no reason.
  """
  from pluggybot.economy.scoring import evaluate

  boss = Overseer(Menu.for_world("home_quad", None),
                  client=FakeClient(full(action="charge",
                                         reason="topping up while it is "
                                                "convenient")))
  life = stub_life(overseer=boss)
  life.body.start_at(*world_config("home_quad")["start"])
  life.battery.energy_wh = life.battery.capacity_wh * 0.80
  said: list[str] = []
  life.say_hooks.append(lambda t, line: said.append(line))
  try:
    assert life.battery.fraction == pytest.approx(0.80, abs=0.01)
    life._decide()
  finally:
    life.body.close()

  # It went. Nothing refuses a chosen charge any more, at any fraction.
  assert life.state in ("GO_CHARGE", "CHARGE"), life.state
  assert not any("not worth a trip" in line for line in said), \
      "the TOP_UP_BELOW refusal is gone, not re-worded"
  # ...and there is nothing in it. A perfect charge banks zero, so the trip
  # cannot have been for the points.
  paid = evaluate("charge", {"startFrac": 0.80, "endFrac": 0.95,
                             "gainedWh": 0.4, "seconds": 100})
  assert paid.ok and paid.points == 0, \
      "a charge that pays is a charge that can be farmed"
  # ...and nothing forces one either: with a mind, the floor is off.
  life.battery.energy_wh = life.low_battery_wh * 0.5
  assert not life.needs_charge


def test_the_sim_keeps_running_while_the_overseer_thinks():
  """A slow API must cost the robot a pause, not the world a freeze. The
  telemetry stream is built off physics steps, so a blocking call here would
  stop every viewer's clock for the length of an HTTP request."""
  boss = Overseer(Menu.for_world("home_quad", None),
                  client=FakeClient(full(action="idle", reason="."),
                                    delay=0.6),
                  timeout_s=2.0)
  life = stub_life(overseer=boss)
  life.body.start_at(*world_config("home_quad")["start"])
  life.max_sim_time = 60.0
  life.explore_deadline = life.data.time + 1.0
  life.blacklist, life.floor_explored = set(), True
  steps = []
  life.body.step_hooks.append(lambda: steps.append(life.data.time))
  t0 = life.data.time
  wall0 = time.monotonic()
  life._decide()
  wall = time.monotonic() - wall0
  life.body.close()
  assert life.data.time > t0, "the sim did not advance while it was thinking"
  assert len(steps) > 100, f"only {len(steps)} physics steps during the call"
  assert life.decisions[0]["source"] == "llm"
  # The loop returns when the ANSWER lands, not when the deadline does: the
  # fake takes 0.6 s and the deadline is 2 s, so the wall clock sits between
  # them (a loop that waited out the deadline reads 2.5 s here, measured).
  # ⚠ NOT a sim-time window -- the lifecycle is `realtime=False`, so
  # sim-seconds per wall-second is the box's speed, not the rule's: this
  # read ~1 sim-s per 0.6 s call when written and ~2 after #253 made the
  # physics seam cheaper, and a window on it measured the machine.
  assert 0.6 <= wall < 2.0, f"{wall:.2f} s wall for a 0.6 s call"
