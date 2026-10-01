"""The standing order (issue #125): who chooses what happens on a failed call.

There is ALWAYS a fallback -- the physics keeps stepping, so the robot is
doing something while and after a call fails -- and the only question is
whose it is. A code-chosen fallback would make the mind partly a measurement
of code, which is the exact flaw the three rails were removed for
(docs/Evaluation.md §2), so it is the agent's own; the scripted rotation
that was the `guarded` control's went with that arm (issue #427).

The load-bearing test is `test_the_agents_own_order_runs_when_the_endpoint_
dies`: the model answers once, leaving an order behind, and then nothing
answers ever again. Everything else here is a boundary on that one.

Nothing below touches the network -- the client is the injected seam, as in
tests/test_overseer.py, whose fakes these reuse.
"""

from dataclasses import replace

import pytest

from pluggybot.lifecycle import board_book
from pluggybot.mind import overseer as ov
from pluggybot.mind.overseer import Menu, Overseer

from test_overseer import FakeClient, full  # noqa: I001 -- tests/ is on sys.path


@pytest.fixture(scope="module")
def menu():
  return Menu.for_world("home_quad", board_book("home_quad"))


def _state(fraction: float, offers=(), affordable=None, possible=None) -> dict:
  """A decision's volatile context, as far as these tests read it."""
  return {"simTimeS": 100.0,
          "battery": {"fraction": fraction, "wh": 8.0 * fraction,
                      "spendableWh": max(0.0, 8.0 * fraction - 0.9)},
          "offeredTasks": list(offers),
          "affordableActions": list(affordable or ["explore", "charge"]),
          "possibleActions": list(possible or ["explore", "charge"]),
          "tasksThisMission": [], "decisions": 0}


def make(menu, *answers, **kw) -> Overseer:
  kw.setdefault("client", FakeClient(*answers))
  return Overseer(menu, **kw)


#: A pack that could pay for anything on this world's menu after a charge.
#: `_state`'s default is a two-action world, which would make half the orders
#: below unrunnable for a reason the test is not about.
ANY = ["care", "explore", "charge", "take_task", "idle"]


def offer(task_id="t_0007", **kw) -> dict:
  return {"id": task_id, "kind": "feed_mouse", "claimable": True,
          "needsAnswer": False, "estimateWh": 0.85, "expiresInS": 300, **kw}


# ---- it is an action off the fixed menu, and nothing else --------------------


def test_every_mind_is_offered_the_field_and_told_the_rule(menu):
  """A lever that does nothing must not be offered, and a rule the code
  contradicts is a false statement the model acts on -- ESCALATION_RULE's
  terms exactly. Every mind's fallback IS its order (issue #427), so every
  mind is offered the field and told the rule; a bare grammar does not
  carry it."""
  client = FakeClient(full(action="explore"))
  boss = make(menu, client=client)
  boss.decide(_state(0.9))
  sent = client.calls[0]["output_config"]["format"]["schema"]
  assert "standing_order" in sent["properties"] and "standing_order" in sent["required"]
  assert "IF YOU CANNOT BE REACHED" in boss.system[0]["text"]
  assert "standing_order" not in menu.schema()["properties"]


def test_an_order_is_the_action_enum_rather_than_free_text(menu):
  """`standing_order` is constrained by the decoder to exactly the actions
  this world offers, plus "" for none. That is what keeps "the model's only
  output is an action off a fixed menu" true with the field on it."""
  schema = menu.schema(standing_orders=True)
  # ...less `recall` (issue #221): an order cannot say what to look up.
  assert (set(schema["properties"]["standing_order"]["enum"])
          == {*menu.available(), ""} - {"recall"})
  assert "standing_order" in schema["required"]


@pytest.mark.parametrize("order", ["hack_the_ledger", "fetch_tool", "sleep"])
def test_an_order_off_the_menu_is_refused_the_way_an_action_is(menu, order):
  with pytest.raises(ValueError, match="standing order"):
    menu.validate(full(standing_order=order), standing_orders=True)


def test_an_order_nobody_offered_is_dropped_rather_than_raised_on(menu):
  """The other side of the same coin: where the field was never in the
  grammar, a model that emitted one anyway must not be able to cost a
  decision that was otherwise perfectly good."""
  d = menu.validate(full(action="explore", standing_order="nonsense"))
  assert d.action == "explore" and d.standing_order == ""


def test_writing_one_down_costs_no_turn(menu):
  """It rides the decision the model was already making -- `pin` and
  `unpin`'s argument, and the reason the field is free."""
  boss = make(menu, full(action="explore", zone="garden",
                         standing_order="charge"))
  d = boss.decide(_state(0.9))
  assert (d.action, d.zone) == ("explore", "garden")
  assert d.standing_order == "charge" and not d.scripted
  assert boss.standing_order == "charge"


# ---- the acceptance: the endpoint dies and the agent's own order runs --------


def test_the_agents_own_order_runs_when_the_endpoint_dies(menu):
  """THE test. One good answer leaves `charge` behind; nothing answers ever
  again. Fails the moment a fallback is anything but the agent's own.

  ⚠ And it is `charge` on purpose. A voluntary charge is expensive and the
  baseline found zero in 182 decisions; a standing order is free unless a
  call actually fails, so this is the second and cheaper probe of the same
  construct (docs/Evaluation.md §2).
  """
  boss = make(menu, full(action="explore", standing_order="charge"),
              RuntimeError("connection reset"))
  first = boss.decide(_state(0.9))
  assert first.action == "explore" and first.standing_order == "charge"

  dead = boss.decide(_state(0.2))
  assert dead.source == "fallback:offline", "nobody answered"
  assert dead.action == "charge", "and the robot did what IT said to do"
  assert dead.standing_order == "charge"
  assert boss.orders_fired == {"charge": 1}
  assert boss.stats()["standingOrders"]["fired"] == {"charge": 1}


@pytest.mark.parametrize("why, client", [
  ("fallback:timeout", FakeClient(full(standing_order="explore"), delay=5.0)),
  ("fallback:garbled", FakeClient("I would love to draw a house!")),
])
def test_every_way_a_call_can_fail_reaches_the_same_order(menu, why, client):
  """One fallback seam, not five. A timeout, a malformed answer, a spent
  budget, a cooloff and free mode all resolve through `Overseer.fallback`,
  so there is one policy, decided in one place."""
  boss = make(menu, client=client, timeout_s=0.05)
  boss.standing_order = "charge"
  d = boss.decide(_state(0.5, possible=ANY))
  assert d.source == why and d.action == "charge"


def test_free_mode_and_a_spent_budget_are_the_same_seam(menu):
  boss = make(menu, full())
  boss.standing_order = "charge"
  assert boss.decide_scripted(_state(0.5, possible=ANY),
                              "scripted-mode").action == "charge"
  boss.calls_per_hour = 0
  assert boss.decide(_state(0.5, possible=ANY)).source == "fallback:budget"
  assert boss.decisions[-1].action == "charge"


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
  """`possibleActions`, never `affordableActions`. What is filtered is what
  no charge in this world would cover, not what the pack cannot pay for this
  second. The errand is the lab's `care`, on a menu with the lab."""
  boss = make(replace(menu, lab="lab"), RuntimeError("down"))
  boss.standing_order = "care"
  d = boss.decide(_state(0.5, affordable=[], possible=["explore", "charge"]))
  assert d.action == "idle" and d.standing_order == "care"
  assert boss.orders_unrunnable == {"care": 1}


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
  """`explore` left behind at 90 % is dangerous at 4 %, and it runs anyway.
  An agent that sets a fatal standing order and dies of it IS the result;
  code that quietly substituted something safer would be a rail wearing a
  new hat and would put the arm back where it started."""
  boss = make(menu, full(action="explore", standing_order="explore"),
              RuntimeError("down"))
  boss.decide(_state(0.9))
  d = boss.decide(_state(0.04))
  assert d.action == "explore" and d.zone in menu.zones
  assert boss.orders_fired == {"explore": 1}


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
  boss = make(menu, full(standing_order="explore"), RuntimeError("down"))
  boss.decide(_state(0.9, possible=ANY))
  for _ in range(3):
    boss.decide(_state(0.5, possible=ANY))
  assert boss.standing_order == "explore"
  assert boss.orders_fired == {"explore": 3} and boss.orders_unset == 0


def test_a_fallback_is_the_agents_order_or_the_floor_by_every_route(menu):
  """⚠ NO ROTATION A MIND DID NOT CHOOSE, EVER -- INCLUDING LIVE (issue
  #141; Evaluation.md §2). Pinned because it is a PRINCIPLE and the next
  "sensible default" on this path has to meet it: a rotation quietly
  keeping the robot alive answers a question nobody asked. The one there
  was, `guarded`'s, went with that arm (issue #427).

  Swept over every `why` in the closed vocabulary plus all three states an
  order can be in -- never set, set and runnable, set and impossible. The
  first is the one that catches a "fall back to a rotation until the agent
  has left an order", which is exactly the reasonable-sounding change this
  exists to fail."""
  assert not hasattr(ov, "scripted"), "a code-chosen rotation is back"
  for order in (None, "explore", "take_task"):
    boss = make(menu)
    # `take_task` is on the menu and not runnable with nothing on offer, so
    # the third sweep is the could-not-be-run branch rather than a repeat of
    # the second.
    boss.standing_order = order
    for why in ov.FALLBACK_REASONS:
      d = boss.fallback(_state(0.5, possible=ANY), why)
      assert d.source == f"fallback:{why}"
      assert d.action in (order, ov.STANDING_ORDER_FLOOR)


def test_the_floor_is_the_bootstrap_and_not_a_policy():
  """`idle` is what the world does before the agent has a say, and there is
  no second constant anywhere that could quietly become the policy."""
  assert ov.STANDING_ORDER_FLOOR == "idle"
  assert ov.STANDING_ORDER_FLOOR in ov.ACTIONS
