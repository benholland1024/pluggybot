"""A robot answers a visitor while it charges (issue #730; rooftop-media-2026
#518's sim half, Ben's calls of 2026-11-16 and 11-18).

A charge's hold ends with its plug in where a person's message came that the
agent's map asks on -- its first `message_received` row an `ask` -- and the
loop asks at once, plugged in, on #654's rules: a `charge` goes straight on,
what stands still leaves the plug in, what moves takes it out first. ⚠ A
FALLBACK PUTS THE ROBOT BACK ON ITS CHARGE (Ben's call of 11-18): its standing
order, or its `decision_failed` row, runs when the charge ends, at the pack
it would have run at without the break. ⚠ ONE CYCLE however often it is
broken: one `charge` row, one `charge_cycles` and one `task_complete charge`,
whether it is resumed or the pack fills while the robot stands still. ⚠ N =
1: a message is asked about once during an outage -- by the first call whose
context held it, whatever came of the call.

Each rule is pinned on the stub (`test_run_end`'s helpers) and was shown
failing on the tree before (loop-staging e629cddd, where a message waited
for the charge's end) and, where the last build had the defect, on that
build's last commit (5c288531): round 6's five each have a test here.
"""
import pytest

from pluggybot import lifecycle as lc
from pluggybot.mind import events as ev
from pluggybot.mind import overseer as ov
from pluggybot.mind import text as text_registry

from test_continuation import _history  # noqa: I001 -- tests/ is on sys.path
from test_overseer import CutResponse, FakeClient, full
from test_run_end import _minded, _say

#: The agent's map asks it when somebody speaks.
ASKS = ev.Row(event="message_received", action=ev.ASK)


def _answer(action: str = "charge", to: str = "m_1", **kw) -> dict:
  return full(action=action, respond_to=to, outcome="replied",
              reply="I am charging, and here to talk.", **kw)


def _charges(life) -> list[dict]:
  return [v for v in life.verdicts if v["task"] == "charge"]


def _occurred(life) -> list:
  """Every `(event, kind)` the loop tells the map of, as it happens."""
  told: list = []
  occur = life._occur

  def spy(event, kind=""):
    told.append((event, kind))
    occur(event, kind)
  life._occur = spy
  return told


def _charge(life) -> None:
  """A decided `charge`, its plug in already (#654): the branch runs the
  charge, until it ends or a message breaks its hold."""
  life.body.run(life._after_decision_routine(ov.Decision(action="charge")))


def _next_pass(life) -> None:
  """The loop's next decision pass (`_arbitrate_routine`)."""
  life.body.run(life._arbitrate_routine())


def _unplugs(life) -> list:
  """Every unplug by hand, with the pack at it, and the plug out after."""
  out: list = []

  def undock():
    out.append(round(life.battery.fraction, 3))
    life.body.on_charger = False
    return "arrived"
    yield
  life.body.undock_routine = undock
  return out


class _Asked(FakeClient):
  """Answers in order, and keeps the pack's fraction at each ask."""

  def __init__(self, life_box: list, *answers, **kw):
    super().__init__(*answers, **kw)
    self.life_box = life_box
    self.at: list[float] = []

  def create(self, **kwargs):
    self.at.append(self.life_box[0].battery.fraction)
    return super().create(**kwargs)


def _life_asked(*rows, answers=(), **kw):
  box: list = []
  client = _Asked(box, *answers, **kw)
  life = _minded(*rows, client=client)
  box.append(life)
  return life, client


# ---- answered plugged in, and one charge -------------------------------------


def test_a_message_is_answered_plugged_in_and_the_charge_carries_on_as_one():
  """#730's first box. The map's `message_received -> ask` waited behind the
  charge (43 minutes on 10-29); now the hold ends with the plug in, the
  loop asks, a `charge` goes straight on, and the charge is ONE: its row
  from where it began (the last build banked the part after the break
  alone, and the part before it was in no row), one cycle, one
  `task_complete charge`, one unplug, at full."""
  life, client = _life_asked(ASKS, answers=[_answer("charge")], delay=0.1)
  told = _occurred(life)
  unplugs = _unplugs(life)
  _say(life)
  _charge(life)
  assert life.body.on_charger and life.battery.fraction < lc.CHARGED, "the hold ran on"
  assert life.charge_cycles == 0 and not _charges(life), "nothing banked at the break"
  _next_pass(life)                          # ...the pack filling as it thinks
  assert client.at and client.at[0] < lc.CHARGED, "asked plugged in"
  assert life.queued_row is None, "the message's own ask row was this ask"
  assert [r["id"] for r in life.replies] == ["m_1"]
  assert life.charge_cycles == 1 and life.battery.fraction >= lc.CHARGED
  [row] = _charges(life)
  assert row["metrics"]["startFrac"] == pytest.approx(0.4, abs=0.002), row["metrics"]
  assert told.count(("task_complete", "charge")) == 1
  assert unplugs and min(unplugs) >= lc.CHARGED, unplugs
  assert any("a message came in" in ln for ln in _history(life))


def test_a_stand_still_answer_closes_the_charge_when_the_pack_fills():
  """Round 6's second: the model answers `idle` plugged in, and the pack
  filled while it stood: no `task_complete charge`, no `charge_cycles`, no
  row -- and a map that asks on `task_complete:charge` was never asked (an
  `unminded` death at 1800 s). The charge left open closes where the pack
  is full, standing still, with its row, its cycle and its event."""
  on_full = ev.Row(event="task_complete", kind="charge", action=ev.ASK)
  life, _ = _life_asked(ASKS, on_full, answers=[_answer("idle")])
  told = _occurred(life)
  _say(life)
  _charge(life)
  assert life.body.on_charger and life.battery.fraction < lc.CHARGED, "the hold ran on"
  _next_pass(life)
  assert life.body.on_charger, "standing still leaves the plug in"
  life.body.run(life.body.hold_routine(400.0))
  assert life.battery.fraction >= lc.CHARGED
  assert life.charge_cycles == 1 and len(_charges(life)) == 1
  assert told.count(("task_complete", "charge")) == 1
  assert life.queued_row == on_full, "the map heard the charge complete"


def test_an_answer_that_moves_takes_the_plug_out_first_and_banks_what_it_got():
  """What moves takes the plug out first (#654), and the charge it cut is
  banked as a charge cut short always is -- the run's hour, a plug that
  came out: its row, never a cycle or a `task_complete charge`."""
  life, _ = _life_asked(ASKS, answers=[_answer("explore")])
  told = _occurred(life)
  unplugs = _unplugs(life)
  order: list = []

  def explore(**kw):
    order.append(life.body.on_charger)
    return "budget"
    yield
  life.explore_routine = explore
  _say(life)
  _charge(life)
  _next_pass(life)
  assert order == [False], "explored with its plug in"
  assert len(unplugs) == 1 and unplugs[0] < lc.CHARGED
  [row] = _charges(life)
  assert row["metrics"]["startFrac"] == pytest.approx(0.4, abs=0.01)
  assert life.charge_cycles == 0 and ("task_complete", "charge") not in told


# ---- a failed call: back on its charge -----------------------------------------


@pytest.mark.parametrize("failing", [
  TimeoutError("nobody answered in time"), ConnectionError("the endpoint is down"),
  "this is not a decision", CutResponse("")],
  ids=["timeout", "offline", "garbled", "cutoff"])
def test_a_failed_call_puts_the_robot_back_on_its_charge(failing):
  """Round 6's first, the death path: a message ended a charge at 40 %, the
  call failed, and the fallback ran its order at once -- an `explore`
  unplugged the robot and walked it off part-charged, where before the
  break the same fallback ran at a full pack. The robot goes back on its
  charge, and its `decision_failed` row runs when the charge ends."""
  failed = ev.Row(event="decision_failed", action="explore")
  life, client = _life_asked(ASKS, failed, answers=[failing])
  unplugs = _unplugs(life)
  order: list = []

  def explore(**kw):
    order.append(round(life.battery.fraction, 3))
    return "budget"
    yield
  life.explore_routine = explore
  _say(life)
  _charge(life)
  _next_pass(life)
  assert client.at and client.at[0] < lc.CHARGED, "the break asked, plugged in"
  assert life.decisions[-1]["source"].startswith("fallback:"), life.decisions[-1]
  assert unplugs and min(unplugs) >= lc.CHARGED, f"unplugged below full: {unplugs}"
  assert order and order[0] >= lc.CHARGED, "its order ran before the charge ended"
  assert life.charge_cycles == 1 and len(_charges(life)) == 1
  assert len(client.calls) == 1, "asked again about the same message"


def test_a_later_fallback_while_the_charge_is_open_goes_back_on_it_too():
  """The model answered the break standing still, its plug in and its
  charge open; a later ask fails. Its order would unplug the robot the
  break left part-charged, as round 6's did: it goes back on the charge
  the break left open, and the order runs when it ends."""
  failed = ev.Row(event="decision_failed", action="explore")
  quiet = ev.Row(event="nothing_to_do", action=ev.ASK)
  life, client = _life_asked(ASKS, failed, quiet,
                             answers=[_answer("idle"), ConnectionError("down")])
  unplugs = _unplugs(life)
  order: list = []

  def explore(**kw):
    order.append(round(life.battery.fraction, 3))
    return "budget"
    yield
  life.explore_routine = explore
  _say(life)
  _charge(life)
  _next_pass(life)                          # answered: stand still, the plug in
  assert life.body.on_charger and life.battery.fraction < lc.CHARGED
  _next_pass(life)                          # asked again, and the call fails
  assert len(client.calls) == 2
  assert unplugs and min(unplugs) >= lc.CHARGED, f"unplugged below full: {unplugs}"
  assert order and order[0] >= lc.CHARGED
  assert life.charge_cycles == 1 and len(_charges(life)) == 1


def test_a_charge_holds_a_moment_before_a_message_breaks_it():
  """Round 6's fourth, its cause: carried on with a message come in while
  it thought, the charge ended before a step, at the instant of the
  decision, and every turn of a chat read "charge took no time"."""
  life, _ = _life_asked(ASKS)
  _say(life)
  life._next_events_check = 0.0
  life._events_step()                       # owed before the charge begins
  t0 = float(life.data.time)
  _charge(life)
  assert life.body.on_charger and life.battery.fraction < lc.CHARGED
  assert float(life.data.time) > t0, "broken before it held a moment"


def test_a_failed_answers_order_waits_through_the_next_messages_break():
  """Back on its charge after a failed call, the robot is broken again by
  the next person's message: the order the failure left must not run then,
  at the pack the break left (it is the next ask's to supersede), and the
  charge goes on as one."""
  failed = ev.Row(event="decision_failed", action="explore")
  life, client = _life_asked(ASKS, failed,
                             answers=[ConnectionError("down"), _answer("charge", to="m_2")])
  unplugs = _unplugs(life)
  order: list = []

  def explore(**kw):
    order.append(round(life.battery.fraction, 3))
    return "budget"
    yield
  life.explore_routine = explore
  t0 = float(life.data.time)

  def second():
    if life.data.time > t0 + 40.0 and "m_2" not in {m.id for m in life.inbox.peek()}:
      if not any(r["id"] == "m_2" for r in life.replies):
        _say(life, 2)
  life.body.step_hooks.append(second)
  _say(life)
  _charge(life)
  _next_pass(life)                          # m_1's call fails: back on the charge...
  assert life.body.on_charger and life.battery.fraction < lc.CHARGED, "...broken by m_2"
  assert not order and not unplugs, "the failed answer's order ran at the next break"
  _next_pass(life)                          # m_2's call is answered: carried on
  assert [r["id"] for r in life.replies] == ["m_2"]
  assert not order and min(unplugs) >= lc.CHARGED
  assert life.charge_cycles == 1 and len(_charges(life)) == 1


def test_a_failed_answers_order_waits_for_the_next_run_when_the_hour_cuts_the_charge():
  """The run's hour cuts the charge the robot went back on: its plug stays
  in and what it got is banked (#654), and the order is the next run's
  mind's to give again -- run past the hour, an `explore` took the plug out
  for a walk the run had no time for."""
  failed = ev.Row(event="decision_failed", action="explore")
  life, _ = _life_asked(ASKS, failed, answers=[ConnectionError("down")])
  order: list = []

  def explore(**kw):
    order.append(life.body.on_charger)
    return "time"
    yield
  life.explore_routine = explore
  _say(life)
  _charge(life)
  life.max_sim_time = float(life.data.time) + 20.0
  _next_pass(life)
  assert life.body.on_charger and not order
  [row] = _charges(life)
  assert row["metrics"]["startFrac"] == pytest.approx(0.4, abs=0.01)


def test_a_charge_closed_as_the_failure_is_said_leaves_its_order_to_run_now():
  """Saying a failed answer may step the world (`_after_decision_routine`
  grades a job first), and the charge it would go back on may close
  meanwhile -- its plug out, its pack full: nothing to go back on, and the
  order runs at once."""
  failed = ev.Row(event="decision_failed", action="explore")
  life, _ = _life_asked(ASKS, failed, answers=[ConnectionError("down")])
  order: list = []

  def explore(**kw):
    order.append(life.body.on_charger)
    return "budget"
    yield
  life.explore_routine = explore
  said = life._after_decision_routine

  def said_and_unplugged(decision, act=True):
    out = yield from said(decision, act=act)
    if not act:
      life.body.on_charger = False          # ...the plug out as it was said,
      life.charging_now = False             # read by the seam's next step
      life._open_charge_step()
    return out
  life._after_decision_routine = said_and_unplugged
  _say(life)
  _charge(life)
  _next_pass(life)
  assert order == [False] and len(_charges(life)) == 1


def test_an_outage_asks_once_per_message_and_never_loops_on_the_plug():
  """N = 1 (#730, written first): with the endpoint down, a message is asked
  about once -- the last build retried it, and a retry spent on `busy` lost
  it (round 6's third) -- and a second message is asked once too. The
  charge runs to full between them and its order runs after."""
  life, client = _life_asked(ASKS, answers=[ConnectionError("down")])
  _say(life, 1)
  _charge(life)
  _next_pass(life)                        # asked about m_1, failed: back on the charge
  assert life.charge_cycles == 1 and len(client.calls) == 1
  life.body.on_charger = True             # ...and a second charge, and a second message
  life.battery.energy_wh = life.battery.capacity_wh * 0.4
  _say(life, 2)
  _charge(life)
  _next_pass(life)
  assert len(client.calls) == 2, "one ask a message"
  assert life.charge_cycles == 2 and len(_charges(life)) == 2


def test_a_call_still_out_holds_the_break_and_loses_no_message():
  """Round 6's third: `busy` is a call that outlived its deadline still
  being read, and an ask started then makes none -- the last build spent
  its one retry on it and the message was never asked about again. A hold
  is not broken while a call is out; once it is done, it is, and the
  message is answered within the charge."""
  life, client = _life_asked(ASKS, answers=[_answer("charge")])
  life.overseer._in_flight = True         # the worker of a call that timed out
  t0 = float(life.data.time)
  read: list = []

  def done():
    if life.data.time > t0 + 30.0 and not read:
      read.append(True)
      life.overseer._in_flight = False    # ...done reading, once
  life.body.step_hooks.append(done)
  _say(life)
  _charge(life)
  assert life.body.on_charger and life.battery.fraction < lc.CHARGED, "never broken"
  assert float(life.data.time) > t0 + 30.0, "broken while the call was still out"
  _next_pass(life)
  assert not any(d["source"] == "fallback:busy" for d in life.decisions), life.decisions
  assert [r["id"] for r in life.replies] == ["m_1"]
  assert life.charge_cycles == 1 and len(_charges(life)) == 1


# ---- History, and asks that cost ----------------------------------------------


def test_a_visitor_writing_through_every_think_is_said_once():
  """Round 6's fourth: a message came during each think, the answer chose
  `charge`, the resumed charge returned before a step, and History read
  "charge took no time" beside "stopped at N% for a message" on every turn.
  The break is said once a charge, and the charge holds a moment before
  it breaks again."""
  box: list = []

  class Chatty(FakeClient):
    def create(self, **kwargs):
      n = len(self.calls) + 1
      if n < 4:                             # another person writes while it thinks
        box[0].inbox.offer({"type": "message", "id": f"m_{n + 1}", "from": "a visitor",
                            "text": "and now?"})
      return super().create(**kwargs)
  answers = [_answer("charge", to=f"m_{k}") for k in range(1, 5)]
  life = _minded(ASKS, client=Chatty(*answers, delay=0.02))
  box.append(life)
  _say(life)
  _charge(life)
  for _ in range(4):
    _next_pass(life)
  assert [r["id"] for r in life.replies] == ["m_1", "m_2", "m_3", "m_4"]
  history = _history(life)
  said = [ln for ln in history if "a message came in" in ln or "for a message" in ln]
  assert len(said) == 1, said
  assert not any("took no time" in ln for ln in history), history
  assert life.charge_cycles == 1 and len(_charges(life)) == 1


def test_a_message_the_model_was_shown_asks_no_second_time():
  """Round 6's fifth, and the tree before's: a message that reached the
  inbox just before an ask was shown in it, and the map, seeing it a tick
  later, fired `message_received` and paid for a second ask about a message
  the model had answered."""
  life, client = _life_asked(ASKS, answers=[_answer("idle")], delay=0.02)
  _say(life)                                # ...queued; the seam has not looked yet
  life.body.run(life._decide_routine())     # shown it; the seam looks as the call flies
  assert [r["id"] for r in life.replies] == ["m_1"]
  assert life.queued_row is None, "the map asked about a message the model was shown"
  _next_pass(life)
  assert len(client.calls) == 1


# ---- who is speaking, and what the map says -----------------------------------


@pytest.mark.parametrize("raw,sender", [
  ({"type": "rating", "seq": 3, "quality": 0.5}, None),
  ({"type": "set_battery", "frac": 0.5}, None),
  ({"type": "ticket_reply", "id": "op_1", "ticket": "tk_0001", "text": "seen"}, None),
  ({"type": "message", "id": "t_1", "from": "Rowan", "text": "the station is yours"}, "peer")],
  ids=["rating", "reach-in", "ticket-line", "the-other-robots-tell"])
def test_nobody_speaking_ends_no_charge(raw, sender):
  """Rounds 3 and 4: a rating, a reach-in and a ticket's line wait in the
  same queue for code, and the other robot's `tell` comes in it as a
  message -- none is a person's message, and none ends a charge's hold."""
  life, client = _life_asked(ASKS)
  life.inbox.offer(raw, sender=text_registry.PEER if sender else text_registry.VISITOR)
  _charge(life)
  assert life.charge_cycles == 1 and life.battery.fraction >= lc.CHARGED
  assert not client.calls


def test_a_map_that_asks_on_no_message_is_never_cut_short_by_one():
  """Round 3: the map is the whole of when the agent is asked, and a
  message ended the hold through a periodic ask waiting in the map's
  slot."""
  every = ev.Row(event="every", action=ev.ASK, value=900.0)
  life, client = _life_asked(every)
  life.queued_row = every
  _say(life)
  _charge(life)
  assert life.charge_cycles == 1 and life.battery.fraction >= lc.CHARGED
  assert life.queued_row == every and not client.calls


def test_the_maps_first_row_for_a_message_decides():
  """Round 4: the first row for an event is the one that fires, so
  `message_received -> idle` ahead of `message_received -> ask` never asks
  on a message, and no message ends a charge through the second."""
  life, client = _life_asked(ev.Row(event="message_received", action="idle"), ASKS)
  _say(life)
  _charge(life)
  assert life.charge_cycles == 1 and life.battery.fraction >= lc.CHARGED
  assert not client.calls
  # ...and as the map stands when the hold looks: an ask that fired, its
  # map edited since to put `idle` first, ends no hold
  life, client = _life_asked(ASKS)
  _say(life)
  life._next_events_check = 0.0
  life._events_step()                       # the ask fired as the message came
  life.overseer.event_map = ev.EventMap([ev.Row(event="message_received", action="idle"),
                                         ASKS])
  _charge(life)
  assert life.charge_cycles == 1 and not client.calls


def test_a_message_the_map_did_not_fire_for_ends_no_charge():
  """The map fires one row a tick, and where another row wins the tick a
  message came in, its `message_received` row does not fire -- nothing
  asked, nothing stamped the unminded clock -- and no charge is broken for
  it: the map is the whole of when the agent is asked."""
  first = ev.Row(event="battery_below", value=0.9, action="look")
  life, client = _life_asked(first, ASKS)
  _say(life)
  _charge(life)
  assert life.queued_row == first, "the battery's row took the tick"
  assert life.charge_cycles == 1 and not client.calls


@pytest.mark.parametrize("refusing", ["cooloff", "allowance", "budget", "scripted", "no-client"])
def test_a_message_no_mind_would_answer_waits_for_the_charge_to_end(refusing, tmp_path):
  """Rounds 2-3: cooling off, its purse or its hour's calls spent, the
  switch on `scripted`, no endpoint: nobody would answer, and the hold goes
  on (the queued ask waits for the charge's end, as before #730)."""
  import json

  from pluggybot.mind.mode import ModeSwitch
  life, client = _life_asked(ASKS)
  if refusing == "cooloff":
    life.overseer._cooloff_until = life.overseer.clock() + 1e9
  elif refusing == "allowance":
    life.overseer._allowance_ok = lambda: False
  elif refusing == "budget":
    life.overseer.budget_left = lambda: 0
  elif refusing == "no-client":
    life.overseer._client, life.overseer._client_ready = None, True
  else:
    switch = tmp_path / "mode.json"
    switch.write_text(json.dumps({"mode": "scripted"}))
    life.mode = ModeSwitch(switch)
  _say(life)
  _charge(life)
  assert life.charge_cycles == 1 and life.battery.fraction >= lc.CHARGED
  assert life.queued_row == ASKS, "the ask waits for the loop, after the charge"
  assert not client.calls


def test_the_idle_run_throttle_costs_a_message_no_answer():
  """The throttle (`MAX_IDLE_RUN`) answers one ask `idle-run` and opens
  again: a break that meets it goes back on its charge without a call, and
  the next break, a moment on, asks the model -- the message keeps its one
  ask, which only a call spends."""
  life, client = _life_asked(ASKS, answers=[_answer("charge")])
  life.overseer._idle_run = ov.MAX_IDLE_RUN
  _say(life)
  _charge(life)
  _next_pass(life)                          # throttled: back on the charge, no call
  assert not client.calls and not life.replies
  assert life.body.on_charger and life.charge_cycles == 0, "broken again, a moment on"
  _next_pass(life)                          # ...and asked
  assert [r["id"] for r in life.replies] == ["m_1"]
  assert len(client.calls) == 1
  assert life.charge_cycles == 1 and len(_charges(life)) == 1
  assert not any("no answer came" in ln for ln in _history(life)), "a refusal is no failed call"


# ---- what a death, a stand-up and the run's end do to it ----------------------


def test_a_true_death_resets_what_it_must():
  """Round 5: the new robot owes the old one's visitors no break, holds no
  charge the old one left open, and banks none of it."""
  from pluggybot.economy.ledger import Ledger
  life, client = _life_asked(ASKS, answers=[_answer("idle")])
  life.ledger = Ledger()
  _say(life)
  _charge(life)
  assert life.body.on_charger and life.battery.fraction < lc.CHARGED
  life._true_death(float(life.data.time))
  life.body.run(life.body.hold_routine(400.0))
  assert not _charges(life) and life.charge_cycles == 0
  # ...and the new robot, its own list asking on a message, charges as one:
  # never for the old robot's visitor, never carrying the old charge on
  life.overseer.event_map = ev.EventMap([ASKS])
  life.battery.energy_wh = life.battery.capacity_wh * 0.4
  _charge(life)
  assert life.charge_cycles == 1 and not client.calls, "broken for the old robot's message"
  [row] = _charges(life)
  assert row["metrics"]["seconds"] < 250.0, f"the old robot's charge carried on: {row}"


def test_a_stand_up_drops_a_charge_left_open_and_its_ask():
  """A stand-up puts the body back at its start with a full pack, and a
  charge it closes banks nothing (#348): the break's ask and the charge it
  left open go with it -- an ask after it is the loop's, whose fallback runs
  at once, and the full pack closes no charge."""
  failed = ev.Row(event="decision_failed", action="explore")
  life, _ = _life_asked(ASKS, failed, answers=[ConnectionError("down")])
  ran: list = []

  def explore(**kw):
    ran.append(life.charge_cycles)
    return "budget"
    yield
  life.explore_routine = explore
  _say(life)
  _charge(life)                             # broken, its ask owed
  life.dead = {"t": float(life.data.time), "cause": "unpaid", "why": "a test"}
  life._stand_up("timer", auto=True, was=life.dead)
  life.body.on_charger = False              # ...at its start, the plug nowhere
  life.body.run(life.body.hold_routine(5.0))
  assert not _charges(life) and life.charge_cycles == 0
  _next_pass(life)
  assert ran == [0], "its order waited on a charge the stand-up ended"


def test_the_runs_end_banks_a_charge_left_open_and_leaves_the_plug_in():
  """A charge the run's hour cuts banks what it got and leaves the plug in
  (#654); one a message left open -- the robot standing still plugged in
  as its run ends -- does the same."""
  life, _ = _life_asked(ASKS, answers=[_answer("idle")])
  _say(life)
  _charge(life)
  _next_pass(life)
  assert life.body.on_charger and life.battery.fraction < lc.CHARGED
  summary = life.run(start=(life.body.x, life.body.y, 0.0), max_sim_time=float(life.data.time))
  [row] = _charges(life)
  assert row["metrics"]["startFrac"] == pytest.approx(0.4, abs=0.01)
  assert summary["charge_cycles"] == 0 and life.body.on_charger


def test_an_answer_past_the_runs_hour_is_still_acted_on():
  """The hour may come while the break's call flies: the answer is the
  model's, and is acted on as one given on the hour always was -- its reply
  sent, and a `charge` banking what it got with the plug left in."""
  box: list = []

  class Late(FakeClient):
    def create(self, **kwargs):
      box[0].max_sim_time = float(box[0].data.time)   # the hour, mid-call
      return super().create(**kwargs)
  life = _minded(ASKS, client=Late(_answer("charge"), delay=0.02))
  box.append(life)
  _say(life)
  _charge(life)
  _next_pass(life)
  assert [r["id"] for r in life.replies] == ["m_1"]
  assert life.body.on_charger, "unplugged on the hour"
  [row] = _charges(life)
  assert row["metrics"]["startFrac"] == pytest.approx(0.4, abs=0.01)
  assert life.charge_cycles == 0

