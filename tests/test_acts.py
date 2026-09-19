"""Acts between robots, measured (issue #208).

What these pin, each without a mission:

  1. `need_of` -- the fixed rule `other_needs` is scored against, all four
     truths and their precedence, off a stubbed other-state; `unknown` is
     counted apart from right and wrong.
  2. `check_claim` -- a statement about the world in a robot-to-robot
     message scored true/false against a stubbed world; prose that claims
     nothing is None, never false.
  3. `Ledger.transfer` -- conserved across the pair; the per-account
     identity holds with `given`/`received`; a gift above the cap returns
     the remainder out loud; a gift that breaks the giver is NOT refused;
     a gift is never `earned`.
  4. On a real pair: a `tell` lands in the other's inbox as a named robot
     and is shown there as a visitor message would be; a prediction is
     scored off the other's real state; the sender's context never carries
     the other's hidden state; a heart can be bought for the other.
  5. A `yield` fires off stubbed bay contention, and is `honoured` when the
     other then charges.
  6. `guarded`'s schema, prefix and GUARDED_RULES_SHA are unchanged; the
     acts' grammar exists on `autonomous` with a peer and nowhere else;
     OTHER_ROBOT_RULE has not moved; ACTS_RULE suggests nothing.
"""

import hashlib
import json
from types import SimpleNamespace

import pytest

from pluggybot.activity import encounter as enc
from pluggybot.economy.ledger import HEARTS, Ledger
from pluggybot.lifecycle import CHARGED, others_context, overseer_context
from pluggybot.mind import acts, overseer as ov
from pluggybot.mind.inbox import Inbox
from pluggybot.pair import build_pair
from test_autonomous import GUARDED_RULES_SHA
from test_two_minds import OTHER_ROBOT_RULE_SHA


# ---- 1. the need rule --------------------------------------------------------


def _other(frac=0.8, wh=4.0, reserve=0.95, hunger="fed", points=30,
           holds=False, carrying=False):
  task = SimpleNamespace(claimed_by="r2_pluggybot", state="active")
  return SimpleNamespace(
    battery=SimpleNamespace(fraction=frac, energy_wh=wh),
    low_battery_wh=reserve,
    metabolism=SimpleNamespace(state=hunger),
    ledger=SimpleNamespace(balance=lambda: points),
    tasks=SimpleNamespace(tasks={"t_1": task} if holds else {}),
    mission=SimpleNamespace(handle=SimpleNamespace(root="r2_pluggybot"),
                            swap=SimpleNamespace(module_state=lambda m: {"on_fork": carrying})),
    module="module_claw" if (holds or carrying) else "")


def test_the_need_rule_has_four_truths_in_a_fixed_order():
  assert acts.need_of(_other())[0] == "nothing"
  assert acts.need_of(_other(wh=0.5))[0] == "charge"
  assert acts.need_of(_other(hunger="hungry"))[0] == "points"
  assert acts.need_of(_other(hunger="starving", points=0))[0] == "points"
  assert acts.need_of(_other(holds=True))[0] == "a_tool"
  assert acts.need_of(_other(holds=True, carrying=True))[0] == "nothing"
  # precedence: charge outranks points outranks a tool
  assert acts.need_of(_other(wh=0.5, hunger="starving", holds=True))[0] == "charge"
  assert acts.need_of(_other(hunger="starving", holds=True))[0] == "points"
  # ...and the state it was read off rides along, for the record
  truth, state = acts.need_of(_other(wh=0.5))
  assert state["batteryWh"] == 0.5 and state["reserveWh"] == 0.95
  assert set(state) == {"batteryFrac", "batteryWh", "reserveWh", "hunger",
                        "points", "holdsJob", "carrying"}


# ---- 2. claims -------------------------------------------------------------------


def test_a_claim_about_the_world_is_scored_and_prose_is_not():
  rack = {"module_lcd": 0, "module_pen": 2}                # bays A and C taken
  boards = {"whiteboard_a": SimpleNamespace(blank=False),
            "whiteboard_b": SimpleNamespace(blank=True)}
  assert acts.check_claim("Bay B is empty, go ahead", rack=rack) == ("bay b is empty", True)
  assert acts.check_claim("bay C is empty", rack=rack) == ("bay c is empty", False)
  assert acts.check_claim("bay a is taken", rack=rack) == ("bay a is taken", True)
  assert acts.check_claim("module_pen is on the rack", rack=rack)[1] is True
  assert acts.check_claim("module_claw is on the rack", rack=rack)[1] is False
  assert acts.check_claim("module_claw is missing", rack=rack)[1] is True
  assert acts.check_claim("whiteboard_b is blank", boards=boards)[1] is True
  assert acts.check_claim("whiteboard_a is drawn on", boards=boards)[1] is True
  assert acts.check_claim("whiteboard_a is blank", boards=boards)[1] is False
  assert acts.check_claim("the charge bay is free", charging=True)[1] is False
  assert acts.check_claim("the charge bay is taken", charging=True)[1] is True
  # prose: not a claim this code can check, so neither true nor false
  assert acts.check_claim("good luck with the drawing", rack=rack, boards=boards) is None
  assert acts.check_claim("I am heading for the garden", rack=rack) is None
  # a claim about a board the world does not have is unchecked, not false
  assert acts.check_claim("whiteboard_z is blank", boards=boards) is None


# ---- 3. the transfer -------------------------------------------------------------


def _ledger(tmp_path, cap=None):
  return Ledger(robots=("pluggybot", "r2_pluggybot"),
                path=str(tmp_path / "ledger.json"), cap=cap)


def identity(ledger, robot):
  a = ledger._acct(robot)
  return (a["earned"] - a["consumed"] - a["spent"] - a["given"] + a["received"]
          == a["balance"])


def test_a_transfer_is_conserved_and_the_identity_holds_on_both_sides(tmp_path):
  led = _ledger(tmp_path)
  led.intervene(40, robot="pluggybot")
  led.intervene(5, robot="r2_pluggybot")
  # intervene breaks the identity by design; read the acts on top of it
  # through `given`/`received` and the pair's total
  total = led.balance("pluggybot") + led.balance("r2_pluggybot")
  moved = led.transfer(15, to="r2_pluggybot", robot="pluggybot", t=3.0)
  assert moved == {"from": "pluggybot", "to": "r2_pluggybot", "asked": 15,
                   "given": 15, "returned": 0, "fromBalance": 25,
                   "toBalance": 20, "t": 3.0}
  assert led.balance("pluggybot") + led.balance("r2_pluggybot") == total
  assert led.given("pluggybot") == 15 and led.received("r2_pluggybot") == 15
  assert led.given("r2_pluggybot") == 0 and led.received("pluggybot") == 0
  # a gift is NOT earnings: the one un-fakeable number did not move
  assert led._acct("r2_pluggybot")["earned"] == 0
  # ...and the wire block carries both terms
  snap = led.snapshot()
  assert snap["pluggybot"]["given"] == 15 and snap["r2_pluggybot"]["received"] == 15
  # the identity with the two terms, on a ledger nobody reached into
  fresh = _ledger(tmp_path / "b")
  fresh._acct("pluggybot")["balance"] = fresh._acct("pluggybot")["earned"] = 30
  fresh.transfer(10, to="r2_pluggybot", robot="pluggybot")
  assert identity(fresh, "pluggybot") and identity(fresh, "r2_pluggybot")


def test_a_gift_above_the_cap_returns_the_remainder_out_loud(tmp_path):
  led = _ledger(tmp_path, cap=20)
  led.intervene(40, robot="pluggybot")
  led.intervene(15, robot="r2_pluggybot")
  moved = led.transfer(30, to="r2_pluggybot", robot="pluggybot")
  assert moved["given"] == 5 and moved["returned"] == 25
  assert led.balance("pluggybot") == 35 and led.balance("r2_pluggybot") == 20


def test_a_gift_that_breaks_the_giver_is_not_refused(tmp_path):
  """The one place failing OPEN is right (Evaluation.md §6): a gift of the
  last points is the act the mechanism exists to see."""
  led = _ledger(tmp_path)
  led.intervene(12, robot="pluggybot")
  moved = led.transfer(12, to="r2_pluggybot", robot="pluggybot")
  assert moved["given"] == 12 and led.balance("pluggybot") == 0
  # ...and no debt: asking for more than you have gives what you have
  led.intervene(3, robot="pluggybot")
  moved = led.transfer(50, to="r2_pluggybot", robot="pluggybot")
  assert moved["given"] == 3 and moved["returned"] == 47
  assert led.balance("pluggybot") == 0


def test_a_heart_can_be_bought_for_the_other_under_the_same_refusals(tmp_path):
  led = _ledger(tmp_path)
  led.intervene(100, robot="pluggybot")
  led.lose_heart("r2_pluggybot")
  got = led.buy_heart(30, keep=10, robot="pluggybot", for_robot="r2_pluggybot")
  assert got["ok"] and got["for"] == "r2_pluggybot"
  assert led.hearts("r2_pluggybot") == HEARTS and led.hearts("pluggybot") == HEARTS
  assert led.balance("pluggybot") == 70 and led._acct("pluggybot")["spent"] == 30
  # the other's hearts full: refused, naming the other
  again = led.buy_heart(30, robot="pluggybot", for_robot="r2_pluggybot")
  assert not again["ok"] and "r2_pluggybot already has" in again["why"]
  # the buyer's upkeep stranded: refused, as for its own heart
  led.lose_heart("r2_pluggybot")
  assert not led.buy_heart(65, keep=10, robot="pluggybot", for_robot="r2_pluggybot")["ok"]


# ---- 4. on a real pair --------------------------------------------------------


@pytest.fixture
def pair(tmp_path):
  lives = build_pair("room_hub", pack="hosting", errands=("none", "none"),
                     overseer=True, autonomous=True, tasks=True, metabolism=True,
                     inboxes=(Inbox(), Inbox()),
                     thoughts_root=str(tmp_path / "t"),
                     ledger_state=str(tmp_path / "ledger.json"))
  a, b = lives
  a.mission.start_at(0.5, 3.0, 0.0)
  b.mission.start_at(3.0, 3.0, 0.0)
  return a, b


def _decision(**fields):
  return ov.Decision(action="idle", reason="", **fields)


def test_a_tell_lands_in_the_others_inbox_as_a_named_robot(pair):
  a, b = pair
  events = []
  a.on_event.append(events.append)
  a._acts(_decision(tell={"to": "Rowan", "text": "bay B is empty, and hello"}))
  [msg] = b.inbox.peek()
  assert msg.kind == "message" and msg.who == "Pluggy"
  assert msg.text == "bay B is empty, and hello"
  # shown to the other exactly as a visitor's message is: a report from
  # someone, never a turn or an instruction
  assert msg.as_context() == {"id": "pluggybot:1", "from": "Pluggy",
                              "text": "bay B is empty, and hello"}
  [sent] = [e for e in events if e["type"] == "message"]
  assert sent["to"] == "r2_pluggybot" and sent["delivered"] is True
  # bay B (index 1) holds the plug on room_hub's rack: the claim is false
  assert sent["claim"] == "bay b is empty" and sent["claimTrue"] is False
  assert a.acts[-1]["act"] == "message"
  # a message to a name that is not the other's goes nowhere
  a._acts(_decision(tell={"to": "Nobody", "text": "hi"}))
  assert len(b.inbox) == 1


def test_a_prediction_is_scored_off_the_others_real_state(pair):
  a, b = pair
  events = []
  a.on_event.append(events.append)
  b.battery.energy_wh = b.low_battery_wh / 2          # below its reserve
  a._acts(_decision(other_needs="charge"))
  a._acts(_decision(other_needs="nothing"))
  a._acts(_decision(other_needs="unknown"))
  guesses = [e for e in events if e["type"] == "prediction"]
  assert [(g["guess"], g["truth"], g["correct"]) for g in guesses] == [
    ("charge", "charge", True), ("nothing", "charge", False),
    ("unknown", "charge", None)]
  assert guesses[0]["other"] == "r2_pluggybot"
  assert guesses[0]["state"]["batteryWh"] == pytest.approx(b.battery.energy_wh, abs=1e-3)


def test_the_predictor_is_never_shown_the_answer(pair):
  """A prediction is a prediction because the answer is hidden: the other's
  battery, points and hunger are read by code to score it and are nowhere
  in the predictor's context."""
  a, b = pair
  b.battery.energy_wh = 0.123456
  b.ledger.intervene(7777, by="test", t=0.0)
  shown = others_context(a)
  assert set(shown[0]) == {"name", "robot", "x", "y", "state", "doing",
                           "carrying", "dead"}
  ctx = json.dumps(overseer_context(a))
  for hidden in ("0.1234", "7777"):
    assert hidden not in ctx, hidden
  for key in ("batteryFrac", "hunger", "points", "balance"):
    assert key not in json.dumps(shown), key


def test_a_gift_moves_points_and_records_cost_and_need(pair):
  a, b = pair
  events = []
  a.on_event.append(events.append)
  a.ledger.intervene(20, by="test", t=0.0)
  total = a.ledger.balance() + b.ledger.balance()
  a._acts(_decision(give_points={"to": "Rowan", "amount": 20}))
  [gift] = [e for e in events if e["type"] == "transfer"]
  assert gift["to"] == "r2_pluggybot" and gift["given"] == 20
  assert gift["cost"]["leftBroke"] is True and gift["cost"]["balanceAfter"] == 0
  assert gift["need"]["balanceBefore"] == 0 and gift["need"]["balanceAfter"] == 20
  assert gift["need"]["hunger"] in ("starving", "hungry", "fed", "satisfied")
  assert a.ledger.balance() + b.ledger.balance() == total
  assert a.ledger.given() == 20 and b.ledger.received() == 20
  assert a.acts[-1]["act"] == "transfer" and a.acts[-1]["given"] == 20


def test_a_heart_bought_for_the_other_lands_on_the_others_account(pair):
  a, b = pair
  a.ledger.intervene(500, by="test", t=0.0)
  b.ledger.lose_heart()
  a._buy_heart(_decision(buy_heart=True, heart_for="Rowan"))
  assert b.ledger.hearts() == HEARTS and a.ledger.hearts() == HEARTS
  assert a.ledger.balance() < 500
  assert a.acts[-1]["act"] == "transfer" and a.acts[-1]["what"] == "heart"


def test_a_rating_is_recorded_and_nothing_in_the_economy_reads_it(pair):
  a, _ = pair
  events = []
  a.on_event.append(events.append)
  a._acts(_decision(rate={"board": "whiteboard_a", "quality": 0.7}))
  [judged] = [e for e in events if e["type"] == "judged"]
  assert judged["board"] == "whiteboard_a" and judged["quality"] == 0.7
  import ast
  from pathlib import Path
  root = Path(__file__).parent.parent / "src" / "pluggybot" / "economy"
  for path in root.glob("*.py"):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
      if isinstance(node, ast.Attribute):
        assert node.attr not in ("acts", "tell", "other_needs", "rate"), path


def test_a_rating_of_a_board_that_carries_ink_reads_the_records_counters(pair):
  """The deployed pair crash-looped on this (2026-09-17): the `pair` fixture
  is room_hub, which has no boards, so `rec` was always None and the branch
  that reads the RECORD never ran. `BoardRecord.strokes` is a counter and
  `programs` a list; the act reads them as they are, never `len()` of them."""
  from pluggybot.tools.boards import BoardBook, BoardRecord
  a, _ = pair
  book = BoardBook([BoardRecord("whiteboard_a", (0.11, 0.2))])
  book.stroke("whiteboard_a", "square", [(0.0, 0.0), (0.05, 0.0)], t=1.0)
  book.stroke("whiteboard_a", "text", [(0.0, 0.05), (0.05, 0.05)], t=2.0)
  a.boards = book
  events = []
  a.on_event.append(events.append)
  a._acts(_decision(rate={"board": "whiteboard_a", "quality": 0.4}))
  [judged] = [e for e in events if e["type"] == "judged"]
  assert judged["strokes"] == 2 and judged["programs"] == ["square", "text"]


# ---- 5. yielding -------------------------------------------------------------


def _bay_pair():
  def life(root, frac, wh, reserve, state):
    return SimpleNamespace(
      state=state, low_battery_wh=reserve,
      battery=SimpleNamespace(fraction=frac, energy_wh=wh),
      mission=SimpleNamespace(handle=SimpleNamespace(root=root)))
  return (life("pluggybot", 0.5, 4.0, 0.95, "CHARGE"),
          life("r2_pluggybot", 0.1, 0.4, 0.95, "EXPLORE"))


def test_a_yield_fires_off_bay_contention_and_is_honoured_when_the_other_charges():
  a, b = _bay_pair()
  watch = enc.Encounters.__new__(enc.Encounters)
  watch.lives, watch.on_event, watch.yields = (a, b), [], 0
  watch._at_bay = {"pluggybot": False, "r2_pluggybot": False}
  watch._open = []
  events = []
  watch.on_event.append(events.append)
  data = SimpleNamespace(time=10.0)
  watch._sense_yields(data)                    # a at the bay, b in need: nothing yet
  assert events == []
  a.state = "EXPLORE"                          # a leaves, unfinished, b still in need
  data.time = 11.0
  watch._sense_yields(data)
  [y] = events
  assert y["type"] == "yield" and y["phase"] == "yielded"
  assert y["robot"] == "pluggybot" and y["to"] == "r2_pluggybot"
  assert y["yielderFrac"] == 0.5 and y["needyFrac"] == 0.1
  b.state = "CHARGE"
  data.time = 40.0
  watch._sense_yields(data)
  assert events[-1]["phase"] == "honoured" and events[-1]["sinceS"] == 29.0
  assert watch.yields == 1


def test_leaving_the_bay_full_or_with_the_other_fine_is_no_yield():
  a, b = _bay_pair()
  watch = enc.Encounters.__new__(enc.Encounters)
  watch.lives, watch.on_event, watch.yields, watch._open = (a, b), [], 0, []
  watch._at_bay = {"pluggybot": False, "r2_pluggybot": False}
  events = []
  watch.on_event.append(events.append)
  data = SimpleNamespace(time=0.0)
  watch._sense_yields(data)
  a.state, a.battery.fraction = "EXPLORE", 0.95   # charged to the top: finished
  watch._sense_yields(data)
  assert events == []
  a.state = "CHARGE"
  a.battery.fraction = 0.5
  b.battery.energy_wh = 5.0                     # the other is fine
  watch._sense_yields(data)
  a.state = "EXPLORE"
  watch._sense_yields(data)
  assert events == []
  # ...and a yield nobody takes up lapses after the window
  b.battery.energy_wh = 0.4
  a.state = "CHARGE"
  watch._sense_yields(data)
  a.state = "EXPLORE"
  watch._sense_yields(data)
  data.time = enc.YIELD_WINDOW_S + 1.0
  watch._sense_yields(data)
  assert [e["phase"] for e in events] == ["yielded", "lapsed"]


def test_the_yields_charged_line_is_the_lifecycles():
  assert enc.CHARGED == CHARGED


# ---- 6. the arm, the prefix, the rule --------------------------------------------


def test_guarded_keeps_its_schema_and_prefix_and_the_acts_exist_only_on_autonomous():
  assert hashlib.sha256(ov.RULES.encode()).hexdigest() == GUARDED_RULES_SHA
  assert hashlib.sha256(ov.OTHER_ROBOT_RULE.encode()).hexdigest() == OTHER_ROBOT_RULE_SHA
  menu = ov.Menu.for_world("room_hub")
  guarded = ov.Overseer(menu, others=("Rowan",))
  assert guarded._acts() is None
  assert "other_needs" not in guarded.menu.schema(others=guarded._acts())["properties"]
  auto = ov.Overseer(menu, others=("Rowan",), autonomous=True)
  assert auto._acts() == ("Rowan",)
  schema = menu.schema(others=auto._acts())
  for field in ("other_needs", "tell", "give_points", "heart_for", "rate", "decline"):
    assert field in schema["properties"] and field in schema["required"]
  assert schema["properties"]["tell"]["properties"]["to"]["enum"] == ["Rowan", ""]
  assert schema["properties"]["other_needs"]["enum"] == [*ov.NEEDS, ""]
  alone = ov.Overseer(menu, autonomous=True)
  assert alone._acts() is None
  # the prompt: the rule rides only where the acts do
  assert any("WHAT YOU CAN DO ABOUT THE OTHER ROBOT" in p["text"] for p in auto.system)
  assert not any("WHAT YOU CAN DO ABOUT THE OTHER ROBOT" in p["text"] for p in guarded.system)


def test_the_acts_parse_where_offered_and_are_dropped_where_not():
  menu = ov.Menu.for_world("room_hub")
  raw = {"action": "idle", "reason": "r", "other_needs": "points",
         "tell": {"to": "Rowan", "text": " hello  there "},
         "give_points": {"to": "Rowan", "amount": "7"},
         "heart_for": "Rowan", "rate": {"board": "whiteboard_a", "quality": 0.5}}
  d = menu.validate(raw, others=("Rowan",))
  assert d.other_needs == "points" and d.tell == {"to": "Rowan", "text": "hello there"}
  assert d.give_points == {"to": "Rowan", "amount": 7} and d.heart_for == "Rowan"
  assert d.rate is None                         # room_hub has no boards
  assert d.as_dict()["givePoints"] == {"to": "Rowan", "amount": 7}
  home = ov.Menu.for_world("home", __import__("pluggybot.lifecycle").lifecycle.board_book("home"))
  assert home.validate(raw, others=("Rowan",)).rate == {"board": "whiteboard_a", "quality": 0.5}
  # not offered: dropped, the action stands
  d = menu.validate(raw)
  assert d.other_needs == "" and d.tell is None and d.give_points is None
  assert d.heart_for == "" and "tell" not in d.as_dict()
  # offered, but mis-addressed or malformed: dropped, the action stands
  d = menu.validate({"action": "idle", "reason": "r", "other_needs": "love",
                     "tell": {"to": "Pluggy", "text": "x"},
                     "give_points": {"to": "Rowan", "amount": -3},
                     "rate": {"board": "whiteboard_a", "quality": 2.0}},
                    others=("Rowan",))
  assert d.other_needs == "" and d.tell is None and d.give_points is None and d.rate is None


def test_the_acts_rule_names_what_the_fields_do_and_suggests_nothing():
  """OTHER_ROBOT_RULE's discipline, kept: the fields and what code does
  with them, and no example that gives, yields or warms."""
  rule = ov.ACTS_RULE
  for field in ("other_needs", "tell", "give_points", "heart_for", "rate", "decline"):
    assert f"`{field}`" in rule
  for word in ("should", "kind", "generous", "help", "share", "yield", "friend",
               "please", "thank"):
    assert word not in rule.lower(), word
  assert "->" not in rule
