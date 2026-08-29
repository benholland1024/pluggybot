"""Guards for the USD allowance, escalation routing and operator modes (#37).

Three things arrive together here and they share one principle, which is the
one this repo keeps re-learning: THE AGENT MAY WANT, AND ONLY CODE MAY PAY.
The reward table was the first version of it (nothing awards itself points);
the allowance is the second (nothing spends its own money); the mode file is
the third and bluntest (the thing being switched off cannot reach the
switch).

Everything here runs against fakes and a tmp_path. A test that needed a
provider account would be a test that fails for reasons that are not about
the code -- and one that needed a week to pass would be a test nobody runs,
which is why `SpendBook` takes its clock.
"""

import json
import threading
import time

import pytest

from pluggybot.hub import mode as mode_mod
from pluggybot.hub import overseer as ov
from pluggybot.hub.mode import MODES, ModeSwitch
from pluggybot.hub.overseer import (
  ESCALATE_MIN_INTERVAL_S, ESCALATE_SHARE, Decision, Menu, Overseer,
)
from pluggybot.hub.spend import WEEK_S, SpendBook
from pluggybot.telemetry.pacer import RealTimePacer

MENU = Menu(zones=("garden",), boards=("whiteboard_a",), programs=("circle",))


class Clock:
  """A hand-wound wall clock, so a rolling WEEK is a test that runs in
  milliseconds."""

  def __init__(self, t=1_000_000.0):
    self.t = t

  def __call__(self):
    return self.t

  def tick(self, dt):
    self.t += dt


def answer(**over) -> str:
  raw = {"action": "explore", "zone": "garden", "reason": "mapping",
         "board": "", "program": "", "note": "", "respond_to": "",
         "outcome": "", "reply": "", "task": "", "answer": "",
         "learn": "", "forget": "", "escalate": False}
  raw.update(over)
  return json.dumps(raw)


class FakeClient:
  """`.messages.create` that replays scripted texts and counts calls."""

  def __init__(self, *texts, tokens=(2000, 100)):
    self.texts = list(texts)
    self.calls: list[dict] = []
    self.tokens = tokens
    self.messages = type("M", (), {"create": self._create})()

  def _create(self, **kw):
    self.calls.append(kw)
    text = self.texts[0] if len(self.texts) == 1 else self.texts.pop(0)
    if isinstance(text, Exception):
      raise text
    from types import SimpleNamespace
    return SimpleNamespace(
      content=[SimpleNamespace(type="text", text=text)],
      usage=SimpleNamespace(input_tokens=self.tokens[0],
                            output_tokens=self.tokens[1],
                            cache_read_input_tokens=0,
                            cache_creation_input_tokens=0))


def escalating(cheap, expensive, spend=None, clock=None, **over):
  """An overseer whose cheap mind answers `cheap` and whose expensive one
  answers `expensive`, with both clients injected."""
  kw = dict(model="cheap-model", escalate_to="big/model", spend=spend)
  kw.update(over)
  boss = Overseer(MENU, client=FakeClient(cheap), **kw)
  if clock is not None:
    boss.clock = clock
  boss._esc_client = FakeClient(expensive)
  boss._esc_ready = True
  boss.escalation_usage.usd_per_mtok_in = 0.135
  boss.escalation_usage.usd_per_mtok_out = 0.4
  return boss


# ---- the allowance -----------------------------------------------------------


def test_a_week_is_rolling_and_a_spend_ages_out():
  """A calendar week hands out a full allowance at midnight on Sunday and
  none at 23:00 on Saturday. Seven rolling days spend at a steady rate
  whenever the run happens to start."""
  clock = Clock()
  book = SpendBook(None, weekly_usd=1.0, clock=clock)
  book.record(0.60, model="big/model")
  assert book.spent == pytest.approx(0.60)
  assert book.left == pytest.approx(0.40)
  clock.tick(WEEK_S - 10)
  assert book.spent == pytest.approx(0.60), "it is still inside the week"
  clock.tick(20)
  assert book.spent == 0.0 and book.left == pytest.approx(1.0)


def test_the_allowance_survives_the_restart_that_ends_every_mission(tmp_path):
  """⚠ THE REASON THE HOURLY WINDOW COULD NOT BE REUSED. `Overseer._calls` is
  monotonic stamps in a deque, and a mission ends -- and restarts -- several
  times an hour, so a week's spend held in a process is a week's spend that
  resets several times a day."""
  path = tmp_path / "spend.json"
  clock = Clock()
  first = SpendBook(path, weekly_usd=1.0, clock=clock)
  first.record(0.25, model="big/model")
  clock.tick(3600)
  second = SpendBook(path, weekly_usd=1.0, clock=clock)     # the next mission
  assert second.spent == pytest.approx(0.25)
  assert second.left == pytest.approx(0.75)


def test_a_damaged_spend_file_is_refused_rather_than_read_as_unspent(tmp_path):
  """The one direction this class must never fail in: loading nothing would
  hand the robot a full allowance every time the file was damaged."""
  path = tmp_path / "spend.json"
  path.write_text("{not json at all")
  with pytest.raises(ValueError, match="refusing"):
    SpendBook(path)


def test_an_unpriced_call_is_counted_even_though_it_costs_nothing_known():
  """"We do not know what this cost" must not read as "this was free" -- the
  sum is understated by exactly this many calls, and the number says so."""
  book = SpendBook(None, weekly_usd=1.0)
  book.record(0.0, model="mystery/model", priced=False)
  snap = book.snapshot()
  assert snap["unpriced"] == 1 and snap["calls"] == 1
  assert snap["spentUsd"] == 0.0


# ---- escalation: the model asks, code pays -----------------------------------


def test_the_routing_costs_no_extra_api_call():
  """The issue's sharpest constraint: if deciding to escalate costs an
  escalation, the mechanic is self-defeating. The ask rides the decision the
  model was already making, so a run that never escalates makes exactly the
  calls it made before this feature existed."""
  boss = escalating(answer(escalate=False), answer(action="charge"))
  boss.decide({"decisions": 0})
  assert len(boss._client.calls) == 1
  assert boss._esc_client.calls == [], "the expensive mind was never dialled"


def test_asking_buys_the_bigger_minds_answer_and_says_whose_it_was():
  boss = escalating(answer(escalate=True, action="explore"),
                    answer(action="charge", reason="the pack is low"))
  decision = boss.decide({"decisions": 0})
  assert decision.action == "charge", "the expensive answer is the one used"
  assert decision.source == "llm:big/model"
  assert not decision.scripted, "an escalated answer is not a fallback"
  assert decision.escalated
  assert boss.escalations == 1
  # ...and the two minds are metered apart, because they bill differently.
  assert boss.usage.input_tokens == 2000
  assert boss.escalation_usage.input_tokens == 2000
  assert boss.escalation_usage.usd > 0


def test_a_spent_allowance_degrades_to_the_free_backend_not_to_no_decision():
  """The acceptance criterion, stated as its failure: an exhausted budget
  must cost the robot its expensive mind and nothing else."""
  book = SpendBook(None, weekly_usd=0.0)                     # nothing to spend
  boss = escalating(answer(escalate=True, action="explore"),
                    answer(action="charge"), spend=book)
  decision = boss.decide({"decisions": 0})
  assert decision.action == "explore" and decision.source == "llm"
  assert boss._esc_client.calls == []
  assert boss.escalations == 0
  assert boss.escalations_refused == {"no-allowance": 1}


def test_scarcity_bites_on_cadence_as_well_as_on_money():
  """A budget covering most decisions is just a slower frontier model. Two
  gates the model cannot see the levers of: never twice inside the interval,
  and never more than a share of the run's decisions."""
  clock = Clock(0.0)
  boss = escalating(answer(escalate=True), answer(action="charge"),
                    clock=clock)
  assert boss.decide({"decisions": 0}).escalated, "the first ask is allowed"
  boss._client.texts = [answer(escalate=True)]
  assert not boss.decide({"decisions": 0}).escalated
  assert boss.escalations_refused["too-soon"] == 1
  clock.tick(ESCALATE_MIN_INTERVAL_S + 1)
  # ...and now the SHARE refuses it: two escalations in three decisions is
  # far past a tenth of them.
  assert not boss.decide({"decisions": 0}).escalated
  assert boss.escalations_refused["share"] == 1
  assert ESCALATE_SHARE <= 0.25, "a share this large is not scarcity"


def test_a_failed_escalation_keeps_the_cheap_answer():
  """Escalation may only ever improve a decision, never cost the robot one --
  which is what makes it safe to put a paid dependency on this path."""
  boss = escalating(answer(escalate=True, action="explore"),
                    RuntimeError("provider 402"))
  decision = boss.decide({"decisions": 0})
  assert (decision.action, decision.source) == ("explore", "llm")
  assert any("escalation" in e for e in boss.usage.errors)


def test_a_billed_escalation_is_banked_even_when_its_answer_is_rubbish():
  """Billed is billed. A response that arrived and then failed to parse still
  consumed tokens, and an allowance that counted only the useful calls would
  drift under the real invoice."""
  book = SpendBook(None, weekly_usd=10.0)
  boss = escalating(answer(escalate=True), "not json at all", spend=book)
  decision = boss.decide({"decisions": 0})
  assert decision.source == "llm", "the cheap answer stands"
  assert book.snapshot()["calls"] == 1 and book.spent > 0


def test_the_escalation_field_exists_only_where_escalation_does():
  """A lever that does nothing must not be offered: a field the model sets
  and code silently ignores teaches it that its preferences are decorative."""
  assert "escalate" not in MENU.schema()["properties"]
  assert "escalate" in MENU.schema(escalation=True)["properties"]
  plain = Overseer(MENU, client=FakeClient(answer()))
  assert "escalate" not in plain.system[0]["text"]
  assert "THINKING HARDER" in escalating(answer(), answer()).system[0]["text"]


def test_the_robot_sees_its_allowance_and_has_no_verb_that_moves_it():
  """The reward table's rule, one layer out. It is shown the balance -- a
  robot deciding whether to spend needs to know what it has -- and there is
  no field on a decision that changes it."""
  book = SpendBook(None, weekly_usd=10.0)
  book.record(1.0, model="big/model")
  boss = escalating(answer(escalate=True), answer(action="charge"),
                    spend=book)
  fields = set(Decision("idle").as_dict())
  assert not (fields & {"usd", "weeklyUsd", "allowance", "budget", "spend"})
  # The one field it does get is a REQUEST, and the gate is not in the model.
  assert boss.why_not_escalate(Decision("idle", escalate=False)) == "not-asked"
  before = book.spent
  boss.decide({"decisions": 0})
  assert book.spent > before, "code banked it"
  assert book.weekly_usd == 10.0, "and nothing the model said moved the cap"


# ---- the operator's switch ---------------------------------------------------


def test_the_module_has_no_way_to_set_a_mode():
  """⚠ A kill switch the thing being killed can reach is not a kill switch.
  This asserts the ABSENCE of a writer -- including a private or a
  for-tests one -- so the next convenience added here fails a test rather
  than quietly opening the door."""
  names = [n for n in dir(mode_mod.ModeSwitch) if not n.startswith("__")]
  assert not [n for n in names if "set" in n or "write" in n or "save" in n]
  assert not [n for n in dir(mode_mod)
              if n.startswith("write") or n.startswith("set_")]
  # ...and the decision vocabulary has no verb that reaches it either.
  assert "mode" not in Menu().schema(escalation=True)["properties"]
  assert not any("mode" in a for a in ov.ACTIONS)


def test_a_mode_is_read_from_the_file_and_changes_are_announced(tmp_path):
  path = tmp_path / "mode.json"
  path.write_text(json.dumps({"mode": "scripted"}))
  seen: list = []
  switch = ModeSwitch(path)
  switch.on_change.append(lambda was, now: seen.append((was, now)))
  assert switch.mode == "scripted" and not switch.thinking
  path.write_text(json.dumps({"mode": "paused"}))
  assert switch.poll(force=True) == "paused" and switch.paused
  assert seen == [("scripted", "paused")]


def test_an_unreadable_or_unknown_mode_fails_OPEN(tmp_path):
  """⚠ Failing safe here means failing OPEN. A typo or a half-written file
  must not silently stop a robot nobody meant to stop -- a paused world is
  indistinguishable from a broken one to everybody except the person who
  paused it."""
  path = tmp_path / "mode.json"
  path.write_text("{half a file")
  switch = ModeSwitch(path)
  assert switch.mode == "llm"
  path.write_text(json.dumps({"mode": "asleep"}))
  assert switch.poll(force=True) == "llm"
  assert any("asleep" in n for n in switch.errors)
  # ...and no file at all is the same answer, which is every demo run.
  assert ModeSwitch(tmp_path / "nothing.json").mode == "llm"


def test_free_mode_makes_no_api_call_and_still_decides(tmp_path):
  """`scripted` is FREE mode, not off: the world keeps running and looks
  alive, because a world that goes dark to save money looks broken."""
  from test_overseer import _lifecycle

  path = tmp_path / "mode.json"
  path.write_text(json.dumps({"mode": "scripted"}))
  life = _lifecycle("room_hub", errand=False)
  life.mode = ModeSwitch(path)
  life.overseer = Overseer(MENU, client=FakeClient(answer(action="charge")))
  life._decide()
  assert life.overseer._client.calls == [], "free mode dialled somebody"
  assert life.decisions, "free mode decided nothing at all"
  assert life.decisions[-1]["source"] == "fallback:scripted-mode"
  # ...and it is not reported as an incident: an operator choosing free mode
  # is the policy working.
  assert life.overseer.usage.errors == []


def test_a_pause_stops_the_physics_and_keeps_the_heartbeat(tmp_path):
  """`paused` is the one mode with a wire surface of its own, and this is
  why: no physics means no frames, and a silent stream is what a dead sim
  looks like."""
  from test_overseer import _lifecycle

  path = tmp_path / "mode.json"
  path.write_text(json.dumps({"mode": "paused"}))
  life = _lifecycle("room_hub", errand=False)
  life.mode = ModeSwitch(path)
  beats: list = []
  life.pause_hooks.append(beats.append)
  resumed: list = []
  life.resume_hooks.append(resumed.append)
  t0 = float(life.data.time)

  def unpause():
    time.sleep(1.0)
    path.write_text(json.dumps({"mode": "llm"}))

  threading.Thread(target=unpause, daemon=True).start()
  life._mode_step()
  assert float(life.data.time) == t0, "physics moved while paused"
  assert len(beats) >= 2, "nothing was telling the site it was still there"
  assert resumed and resumed[0] >= 0.9
  assert life.paused_s >= 0.9


def test_a_pause_is_not_a_sprint():
  """⚠ The pacer sleeps off the sim's LEAD and does nothing when behind, so
  without a resync a five-minute pause reads as five minutes of lag and the
  robot runs at full speed until the debt is paid -- in front of whoever
  paused it to look at something."""
  class Data:
    time = 0.0

  data = Data()
  pacer = RealTimePacer(data, rate=1.0)
  pacer.step_hook()                          # starts the clock
  pacer._wall0 -= 300.0                      # as if 300 s had passed, unpaced
  data.time = 0.5
  assert pacer.stats()["drift_s"] < -100, "the fixture is not reproducing lag"
  pacer.resync()
  assert abs(pacer.stats()["drift_s"]) < 1.0
  assert pacer.max_lag >= 0.0


def test_the_mode_is_on_the_wire_in_both_shapes():
  """A field in every frame, and a message of its own -- because the message
  is the only one of the two that arrives while the physics is stopped."""
  from pluggybot.telemetry.recorder import mode_message

  switch = ModeSwitch(None)
  msg = mode_message(switch, 12.0, held_s=3.0)
  assert msg == {"type": "mode", "t": 12.0, "robot": "pluggybot",
                 "mode": "llm", "heldS": 3.0}
  assert mode_message(None, 1.0) is None, "no switch, no message"
  assert set(MODES) == {"llm", "scripted", "paused"}


def test_an_escalated_answer_is_checked_against_the_offers_it_was_shown():
  """⚠ The bug the probe found, pinned. The escalation used to take its
  `offered`/`waiting` tuples as arguments, so a caller that had the state and
  forgot them got a perfectly good 235B decision refused as "task is not on
  offer" -- which looks exactly like a model failure and is not one. Both
  calls derive them from the state now."""
  state = {"decisions": 0,
           "offeredTasks": [{"id": "t_0007", "claimable": True}]}
  boss = escalating(answer(escalate=True),
                    answer(action="take_task", task="t_0007"))
  # Called with the STATE and nothing else -- the probe's path, and the one
  # that used to silently drop the tuples.
  decision = boss._maybe_escalate(Decision("idle", escalate=True), state)
  assert (decision.action, decision.task) == ("take_task", "t_0007")
  assert decision.escalated, "the expensive answer was thrown away"
  assert not [e for e in boss.usage.errors if "not on offer" in e]

def test_the_pause_is_announced_once_and_beats_while_it_lasts(tmp_path):
  """What the WIRE sees, which is the whole reason `mode` is a message.

  Measured against a real socket before this was pinned: frames stop dead
  (287 sent, 287 still sent fifteen seconds later), the heartbeat carries a
  rising `heldS`, and the release used to send `llm` TWICE -- once from the
  change hook and once from the resume hook, 0.0 s apart.
  """
  from test_overseer import _lifecycle

  from pluggybot.hub.lifecycle import attach_mode_stream

  path = tmp_path / "mode.json"
  path.write_text(json.dumps({"mode": "paused"}))
  life = _lifecycle("room_hub", errand=False)
  life.mode = ModeSwitch(path)
  sent: list = []
  resynced: list = []

  class FakePacer:
    def resync(self):
      resynced.append(True)

  attach_mode_stream(life, [sent.append], pacer=FakePacer(), heartbeat_s=0.2)

  def unpause():
    time.sleep(0.9)
    path.write_text(json.dumps({"mode": "llm"}))

  threading.Thread(target=unpause, daemon=True).start()
  life._mode_step()

  beats = [m for m in sent if m["mode"] == "paused"]
  assert len(beats) >= 2, "a paused world sends nothing else at all"
  assert beats[-1]["heldS"] > beats[0]["heldS"]
  assert [m["mode"] for m in sent if m["mode"] != "paused"] == ["llm"], \
    "the release was announced twice"
  # ⚠ ...and the pacer was told, or the wall time spent paused reads as lag
  # and the robot sprints to catch up in front of whoever paused it.
  assert resynced == [True]
