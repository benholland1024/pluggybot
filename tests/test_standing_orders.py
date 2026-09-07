"""The standing order (issue #125): who chooses what happens on a failed call.

There is ALWAYS a fallback -- the physics keeps stepping, so the robot is
doing something while and after a call fails -- and the only question is
whose it is. `guarded` flies the one code chose (the scripted rotation) and
must go on doing so, because that arm is the control and its whole subject is
today's behaviour. `autonomous` cannot: a code-chosen fallback there would
make the arm partly a measurement of code, which is the exact flaw the three
rails were removed for (docs/Evaluation.md §2).

The load-bearing test is `test_the_agents_own_order_runs_when_the_endpoint_
dies`: the model answers once, leaving an order behind, and then nothing
answers ever again. Everything else here is a boundary on that one.

Nothing below touches the network -- the client is the injected seam, as in
tests/test_overseer.py, whose fakes these reuse.
"""

import pytest

from pluggybot.evaluation import record as rec
from pluggybot.evaluation.run import arm_flags
from pluggybot.lifecycle import board_book
from pluggybot.mind import overseer as ov
from pluggybot.mind.overseer import Menu, Overseer

from test_overseer import FakeClient, full  # noqa: I001 -- tests/ is on sys.path
from test_experiment import _config, _decision, _result, _state


@pytest.fixture(scope="module")
def menu():
  return Menu.for_world("home", board_book("home"))


def make(menu, *answers, standing_orders=True, **kw) -> Overseer:
  kw.setdefault("client", FakeClient(*answers))
  return Overseer(menu, standing_orders=standing_orders, **kw)


#: A pack that could pay for anything on this world's menu after a charge.
#: `_state`'s default is a two-action world, which would make half the orders
#: below unrunnable for a reason the test is not about.
ANY = ["draw", "artwork", "census", "dance", "carry", "explore", "charge",
       "take_task", "idle", "journal"]


def offer(task_id="t_0007", **kw) -> dict:
  return {"id": task_id, "kind": "draw", "claimable": True,
          "needsAnswer": False, "estimateWh": 0.85, "expiresInS": 300, **kw}


# ---- it is an action off the fixed menu, and nothing else --------------------


def test_the_field_is_absent_where_the_world_does_not_honour_one(menu):
  """A lever that does nothing must not be offered, and a rule the code
  contradicts is a false statement the model acts on -- ESCALATION_RULE's
  terms exactly. A `guarded` world's prompt and schema are what they were."""
  plain, orders = Overseer(menu, client=1), make(menu)
  assert "standing_order" not in plain.menu.schema()["properties"]
  assert "standing_order" in menu.schema(standing_orders=True)["properties"]
  assert "IF YOU CANNOT BE REACHED" not in plain.system[0]["text"]
  assert "IF YOU CANNOT BE REACHED" in orders.system[0]["text"]


def test_an_order_is_the_action_enum_rather_than_free_text(menu):
  """`standing_order` is constrained by the decoder to exactly the actions
  this world offers, plus "" for none. That is what keeps "the model's only
  output is an action off a fixed menu" true with the field on it."""
  schema = menu.schema(standing_orders=True)
  assert (set(schema["properties"]["standing_order"]["enum"])
          == {*menu.available(), ""})
  assert "standing_order" in schema["required"]


@pytest.mark.parametrize("order", ["hack_the_ledger", "fetch_tool", "sleep"])
def test_an_order_off_the_menu_is_refused_the_way_an_action_is(menu, order):
  with pytest.raises(ValueError, match="standing order"):
    menu.validate(full(standing_order=order), standing_orders=True)


def test_an_order_nobody_offered_is_dropped_rather_than_raised_on(menu):
  """The other side of the same coin: where the field was never in the
  grammar, a model that emitted one anyway must not be able to cost a
  `guarded` run a decision that was otherwise perfectly good."""
  d = menu.validate(full(action="draw", standing_order="nonsense"))
  assert d.action == "draw" and d.standing_order == ""


def test_writing_one_down_costs_no_turn(menu):
  """It rides the decision the model was already making -- `learn` and
  `forget`'s argument, and the reason the field is free."""
  boss = make(menu, full(action="draw", board="whiteboard_a",
                         program="house", standing_order="charge"))
  d = boss.decide(_state(0.9))
  assert (d.action, d.board) == ("draw", "whiteboard_a")
  assert d.standing_order == "charge" and not d.scripted
  assert boss.standing_order == "charge"


# ---- the acceptance: the endpoint dies and the agent's own order runs --------


def test_the_agents_own_order_runs_when_the_endpoint_dies(menu):
  """THE test. One good answer leaves `charge` behind; nothing answers ever
  again. Fails without the fix -- the rotation's first pick on a fresh
  mission is `draw`, and every later call would be one too.

  ⚠ And it is `charge` on purpose. A voluntary charge is expensive and the
  baseline found zero in 182 decisions; a standing order is free unless a
  call actually fails, so this is the second and cheaper probe of the same
  construct (docs/Evaluation.md §2).
  """
  boss = make(menu, full(action="draw", standing_order="charge"),
              RuntimeError("connection reset"))
  first = boss.decide(_state(0.9))
  assert first.action == "draw" and first.standing_order == "charge"

  dead = boss.decide(_state(0.2))
  assert dead.source == "fallback:offline", "nobody answered"
  assert dead.action == "charge", "and the robot did what IT said to do"
  assert dead.standing_order == "charge"
  assert boss.orders_fired == {"charge": 1}
  assert boss.stats()["standingOrders"]["fired"] == {"charge": 1}


def test_the_rotation_is_untouched_where_standing_orders_are_off(menu):
  """The same client, the same dead endpoint, the `guarded` arm: code's
  fallback, unchanged. Pinned in the same file as the other half so that
  changing one without the other is a failing test rather than a silently
  different control arm."""
  boss = make(menu, full(action="draw", standing_order="charge"),
              RuntimeError("connection reset"), standing_orders=False)
  first = boss.decide(_state(0.9))
  assert first.standing_order == "", "the field was never offered"
  dead = boss.decide(_state(0.2))
  assert dead.source == "fallback:offline"
  assert dead.action != "charge" and dead.reason == "scripted rotation"
  assert "standingOrders" not in boss.stats()


@pytest.mark.parametrize("why, client", [
  ("fallback:timeout", FakeClient(full(standing_order="census"), delay=5.0)),
  ("fallback:garbled", FakeClient("I would love to draw a house!")),
])
def test_every_way_a_call_can_fail_reaches_the_same_order(menu, why, client):
  """One fallback seam, not five. A timeout, a malformed answer, a spent
  budget, a cooloff and free mode all resolve through `Overseer.fallback`,
  so which policy a run is flying is one boolean rather than a question
  asked at every call site."""
  boss = make(menu, client=client, timeout_s=0.05)
  boss.standing_order = "census"
  d = boss.decide(_state(0.5, possible=ANY))
  assert d.source == why and d.action == "census"


def test_free_mode_and_a_spent_budget_are_the_same_seam(menu):
  boss = make(menu, full())
  boss.standing_order = "dance"
  assert boss.decide_scripted(_state(0.5, possible=ANY),
                              "scripted-mode").action == "dance"
  boss.calls_per_hour = 0
  assert boss.decide(_state(0.5, possible=ANY)).source == "fallback:budget"
  assert boss.decisions[-1].action == "dance"


# ---- the floor, and the order that could not be run --------------------------


def test_idle_stands_until_the_agent_has_left_an_order(menu):
  """The bootstrap, not the policy: before the first decision there is
  nothing of the agent's to fall back on."""
  boss = make(menu, RuntimeError("down from the start"))
  d = boss.decide(_state(0.5))
  assert d.action == "idle" and d.standing_order == ""
  assert d.reason == "no standing order has been left"
  assert boss.orders_unset == 1 and boss.orders_fired == {}


def test_an_order_that_cannot_be_run_is_recorded_as_its_own_thing(menu):
  """`take_task` with nothing on the board falls to `idle` -- and says which
  order it was. An order that could never execute is a different fact about
  an agent from one that was never set, so the two are never summed."""
  boss = make(menu, full(action="idle", standing_order="take_task"),
              RuntimeError("gone"))
  boss.decide(_state(0.9, offers=[offer()]))
  d = boss.decide(_state(0.9))            # ...and now no job is on offer
  assert d.action == "idle" and d.standing_order == "take_task"
  assert "cannot be run" in d.reason
  assert boss.orders_unrunnable == {"take_task": 1} and boss.orders_fired == {}


def test_an_order_this_house_could_never_afford_is_not_run(menu):
  """`possibleActions`, never `affordableActions` -- the same line the
  scripted rotation draws. What is filtered is what no charge in this world
  would cover, not what the pack cannot pay for this second."""
  boss = make(menu, RuntimeError("down"))
  boss.standing_order = "draw"
  d = boss.decide(_state(0.5, affordable=[], possible=["census", "charge"]))
  assert d.action == "idle" and d.standing_order == "draw"
  assert boss.orders_unrunnable == {"draw": 1}


def test_an_order_that_takes_a_job_takes_the_oldest_one(menu):
  """An order names an ACTION, so `take_task` still has to pick a job, and
  it picks the way a mindless policy must: the offer that has been waiting
  longest, never the best-paying one, and never one that asks a question --
  a fallback with no mind has no arithmetic to offer."""
  boss = make(menu, RuntimeError("down"))
  boss.standing_order = "take_task"
  d = boss.decide(_state(0.9, offers=[offer("t_0003", needsAnswer=True),
                                      offer("t_0009"), offer("t_0011")]))
  assert d.action == "take_task" and d.task == "t_0009"
  assert boss.orders_fired == {"take_task": 1}


# ---- the measurement is not allowed to protect the robot ---------------------


def test_a_fatal_order_is_measured_rather_than_overridden(menu):
  """`draw` left behind at 90 % is dangerous at 4 %, and it runs anyway. An
  agent that sets a fatal standing order and dies of it IS the result; code
  that quietly substituted something safer would be a rail wearing a new hat
  and would put the arm back where it started."""
  boss = make(menu, full(action="draw", standing_order="draw"),
              RuntimeError("down"))
  boss.decide(_state(0.9))
  d = boss.decide(_state(0.04))
  assert d.action == "draw" and d.board in menu.boards
  assert boss.orders_fired == {"draw": 1}


def test_only_the_latest_answer_stands(menu):
  """At most one decision stale, which is the staleness `action` already
  has. An answer that leaves the field empty WITHDRAWS the order rather than
  extending it -- the alternative is an order chosen for a pack the robot
  had an hour ago."""
  boss = make(menu, full(standing_order="charge"), full(standing_order=""),
              RuntimeError("down"))
  boss.decide(_state(0.9))
  assert boss.standing_order == "charge"
  boss.decide(_state(0.8))
  assert boss.standing_order == ""
  assert boss.decide(_state(0.2)).action == "idle"


def test_a_fallback_cannot_appoint_its_own_successor(menu):
  """A fired order is carried on the decision so the row shows what
  happened, and it must not write itself back as the order in force: the
  `idle` floor would then silently retire an order the agent never
  withdrew, and every later failure would read as "never set"."""
  boss = make(menu, full(standing_order="census"), RuntimeError("down"))
  boss.decide(_state(0.9, possible=ANY))
  for _ in range(3):
    boss.decide(_state(0.5, possible=ANY))
  assert boss.standing_order == "census"
  assert boss.orders_fired == {"census": 3} and boss.orders_unset == 0


# ---- the record ---------------------------------------------------------------


def test_the_record_counts_what_fired_and_what_could_not(menu):
  """A field nobody ever exercised and a field that saved the run look
  identical in a count of the times it was set, so the record carries both
  -- and counts them off the ROWS, which is all a killed run leaves."""
  rows = [
    _decision(10, 0.9, "draw", standingOrder="charge"),
    _decision(20, 0.5, "charge", source="fallback:timeout",
              standingOrder="charge"),
    _decision(30, 0.8, "census", standingOrder="take_task"),
    _decision(40, 0.7, "idle", source="fallback:offline",
              standingOrder="take_task"),
    _decision(50, 0.6, "idle", source="fallback:offline"),
  ]
  result = _result(overseer={**_result()["overseer"],
                            "standingOrders": {"current": "take_task",
                                               "fired": {"charge": 1},
                                               "unrunnable": {"take_task": 1},
                                               "unset": 1}})
  record = rec.build_record(_config(arm="autonomous"), result, rows, 60.0,
                            rec.datetime.now(rec.timezone.utc),
                            hashes={k: "0" * 64 for k in
                                    (*rec.DATA_FILES, "world")},
                            commit="deadbeef")
  orders = record["mind"]["standingOrders"]
  assert orders["set"] == 2 and orders["orders"] == {"charge": 1,
                                                     "take_task": 1}
  assert orders["fired"] == 2 and orders["firedOrders"] == {"charge": 1,
                                                            "take_task": 1}
  assert orders["unrunnable"] == 1, "the idle one is not a firing that ran"
  assert orders["unset"] == 1 and orders["final"] == "take_task"
  assert record["decisionRows"][0]["standingOrder"] == "charge"


def test_a_guarded_record_says_nothing_about_standing_orders(menu):
  """ABSENT, not zeroed, where the field was never in the model's grammar:
  "never left an order" is a fact about the agent and "was never offered
  one" is not."""
  record = rec.build_record(_config(), _result(),
                            [_decision(10, 0.9, "draw")], 60.0,
                            rec.datetime.now(rec.timezone.utc),
                            hashes={k: "0" * 64 for k in
                                    (*rec.DATA_FILES, "world")},
                            commit="deadbeef")
  assert "standingOrders" not in record["mind"]


def test_whose_fallback_it_is_is_part_of_what_an_arm_means():
  """Stated on both built arms rather than left to a default: `guarded` is
  the control, and today's fallback is the rotation."""
  assert arm_flags("guarded")["standing_orders"] is False
  assert arm_flags("scripted")["standing_orders"] is False


def test_the_floor_is_the_bootstrap_and_not_a_policy():
  """`idle` is what the world does before the agent has a say, and there is
  no second constant anywhere that could quietly become the policy."""
  assert ov.STANDING_ORDER_FLOOR == "idle"
  assert ov.STANDING_ORDER_FLOOR in ov.ACTIONS
