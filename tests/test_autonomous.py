"""The `autonomous` arm (issue #115; docs/Evaluation.md §2).

Three rails come off, the prompt is corrected in the same change, and the
fallback becomes the agent's own. Everything here is about the arm being a
REAL difference and a CONTAINED one: `guarded` is the control, the deployed
world runs it, and every assertion below has a half that says so.
"""

import hashlib

import mujoco
import pytest

from pluggybot import lifecycle as lc
from pluggybot.economy.tasks import Task
from pluggybot.mind import overseer as ov
from pluggybot.mind.overseer import Menu, Overseer

MENU = Menu(zones=("garden",), boards=("whiteboard_a",), programs=("circle",))


def _life(world: str = "home", **kw):
  cfg = lc.world_config(world)
  model = mujoco.MjModel.from_xml_path(cfg["model"])
  return lc.HubLifecycle(model, mujoco.MjData(model), realtime=False,
                         world=world, battery_wh=cfg["battery_wh"],
                         rack=cfg["rack"], grid_bounds=cfg["grid_bounds"],
                         low_battery_wh=cfg["low_battery_wh"], **kw)


# ---- the prompt ---------------------------------------------------------------

#: sha256 of `RULES` as it ships. ⚠ A LITERAL, not computed from `ov.RULES`
#: (which is what this used to be, and a hash compared with itself cannot
#: fail). `guarded` is the control and the deployed world runs it; its prefix
#: is CACHED, so a word changed here is a cache miss on every call and a
#: control that no longer matches the runs it is the control for. Changing
#: it is a deliberate act with a re-fly attached, which is exactly what a
#: failing hash should prompt an argument about.
#: It has moved TWICE, and every series in `results/` predates both:
#:   2026-09-11, the mission statement (docs/PluggyPlan.md) replaced "make
#:     yourself useful" with "this life is yours" --
#:     cbfe2e7b8f9de131228f4c0708330ad75f08aa2d61f554f2f9ec7c9de65fac05
#:     was the text before it;
#:   2026-09-12, issue #154 gave the robot its own `Goals.md` and the rules
#:     had to stop telling it a person writes them --
#:     03686c2c7e4adbc58855e43b666d291b482e56226dda648f7089b3044cfc85f0.
GUARDED_RULES_SHA = "4b53c96e5f1ccc3feb59b6501c6f905f8f6cb16734414db225c6a9ad309992e0"


def test_the_guarded_prompt_does_not_move_when_a_second_arm_appears():
  """The control keeps every word (issue #115). Both prompts exist at once
  and the ARM selects; nothing about building the autonomous text may reach
  back into the guarded one -- which is a live risk here, because the
  autonomous rules are BUILT from the guarded ones by replacement."""
  assert hashlib.sha256(ov.RULES.encode()).hexdigest() == GUARDED_RULES_SHA
  assert "Charging is not your decision" in ov.RULES
  assert "affordableActions" in ov.RULES and "`claimable`" in ov.RULES
  assert ov.RULES_AUTONOMOUS is not ov.RULES


def test_the_autonomous_prompt_stops_telling_the_robot_three_lies():
  """With the rails off, the shipped text is false in three places, and the
  robot acts on it. An arm that measures what a model does when told
  something untrue about its own world measures nothing."""
  text = ov.RULES_AUTONOMOUS
  assert "Charging is not your decision" not in text
  assert "affordableActions" not in text and "possibleActions" not in text
  assert "only take one marked" not in text
  # ...and it is an INSTRUCTION PLUS THE NUMBERS, never a computed verdict:
  # the raw quantities the removed lists were derived from are named.
  assert "battery.wh" in text and "reserveWh" in text
  assert "energyCostWh" in text
  # The shared tail is shared, not copied -- one of the paragraphs neither
  # arm changes must be identical in both.
  assert "Knowledge_and_Opinions.md` is YOURS" in text
  assert text.split("WHAT YOU REMEMBER")[1] == \
      ov.RULES.split("WHAT YOU REMEMBER")[1]


def test_a_reworded_rule_fails_loudly_instead_of_shipping_a_lie():
  """The autonomous text is three replacements on the guarded one, so a
  needle that stops matching would silently ship an arm still told that
  charging is not its decision. It raises at import instead."""
  with pytest.raises(AssertionError, match="RULES no longer contains"):
    ov._swap("some other rules entirely", "Charging is not your decision", "x")


def test_the_arm_selects_the_rules():
  guarded = Overseer(MENU).system[0]["text"]
  autonomous = Overseer(MENU, autonomous=True).system[0]["text"]
  assert "Charging is not your decision" in guarded
  assert "Charging is not your decision" not in autonomous
  assert "LOOKING AFTER YOUR OWN POWER IS YOUR JOB" in autonomous


# ---- what the model is shown --------------------------------------------------


def _state() -> dict:
  return {"battery": {"fraction": 0.4, "wh": 3.2, "reserveWh": 0.9},
          "affordableActions": ["draw"], "possibleActions": ["draw", "census"],
          "survival": {"aliveS": 1200.0, "deaths": 0},
          "energyCostWh": {"draw": 0.85, "census": 1.18},
          "offeredTasks": [{"id": "t_0001", "claimable": False,
                            "estimateWh": 1.18, "kind": "census"}]}


def test_the_verdicts_code_computed_are_what_the_arm_takes_away():
  """`affordableActions` and `claimable` are arithmetic code did on the
  model's behalf, and this arm exists to find out whether the model can do
  it. ⚠ The RAW NUMBERS all stay -- nothing is hidden except the answer."""
  shown = ov.model_state(_state(), autonomous=True, survival=True)
  assert "affordableActions" not in shown and "possibleActions" not in shown
  assert "claimable" not in shown["offeredTasks"][0]
  assert shown["energyCostWh"] == {"draw": 0.85, "census": 1.18}
  assert shown["battery"]["wh"] == 3.2 and shown["battery"]["reserveWh"] == 0.9
  assert shown["offeredTasks"][0]["estimateWh"] == 1.18
  # ...and `guarded` sees exactly what it always saw, object for object.
  assert ov.model_state(_state(), autonomous=False) == _state()


def test_a0_hides_the_survival_clock_and_a1_puts_it_back():
  """The ladder's first rung has to take back something #107 gave every
  world, or A0 and A1 are the same run and 'does seeing the stake change
  anything' can never be asked."""
  assert "survival" not in ov.model_state(_state(), True, survival=False)
  assert "survival" in ov.model_state(_state(), True, survival=True)
  assert "survival" in ov.model_state(_state(), autonomous=False)


def test_the_filter_is_at_presentation_so_the_fallback_still_works():
  """⚠ The state stays WHOLE and only the view narrows. `order_runnable`
  reads `possibleActions` off the same dict and treats an absent list as
  'nobody supplied one' -- so an autonomous world that built a thinner state
  would quietly stop filtering unrunnable orders, which is the agent's own
  fallback changing behaviour as a side effect of a prompt change."""
  state = _state()
  ov.model_state(state, autonomous=True, survival=False)
  assert state["possibleActions"] == ["draw", "census"], "not mutated"
  assert ov.order_runnable(MENU, "draw", state) is True
  assert ov.order_runnable(MENU, "draw", {**state, "possibleActions":
                                          ["census"]}) is False


# ---- taking a job the pack cannot fund ----------------------------------------


def test_an_unaffordable_offer_is_takeable_on_this_arm_and_only_this_one():
  """The offer filter is a rail, so refusing the `take_task` in `validate`
  would put it back at the last possible moment -- and taking a job it
  cannot finish is precisely the mistake this arm exists to permit."""
  _, offered, _ = ov.limits_from(_state(), autonomous=True)
  assert offered == ("t_0001",)
  _, guarded_offered, _ = ov.limits_from(_state())
  assert guarded_offered == (), "the control still hides what it cannot fund"


def test_the_task_id_is_a_grammar_on_this_arm_and_a_free_string_elsewhere():
  """Measured (issue #115): six of the seven malformed answers in the quiet
  guarded series were a real-looking id that was not on the board, usually
  an OLDER one copied out of the model's own history. An enum makes it
  unrepresentable. ⚠ Off on `guarded`, which is frozen as the control."""
  free = MENU.schema()["properties"]["task"]
  assert free == {"type": "string"}
  constrained = MENU.schema(task_ids=("t_0007",))["properties"]["task"]
  assert constrained == {"type": "string", "enum": ["t_0007", ""]}
  assert Overseer(MENU)._task_ids(("t_0007",)) is None
  assert Overseer(MENU, autonomous=True)._task_ids(("t_0007",)) == ("t_0007",)
  # ...and the seventh was a TRUNCATION, so the arm gets a bigger budget.
  assert Overseer(MENU, autonomous=True).max_tokens > Overseer(MENU).max_tokens


# ---- the three rails ----------------------------------------------------------


def test_all_three_rails_are_off_on_this_arm():
  life = _life(autonomous=True)
  life.battery.energy_wh = 0.01                      # far below the reserve
  assert life.needs_charge is False, "the floor"
  assert life._afford_next() is True, "the gate"
  assert life.claim_budget_wh is None, "the offer filter"


def test_all_three_rails_still_fire_on_the_control():
  """⚠ The half that matters. `guarded` is the control and the deployed
  world runs it: an LLM that can decline to charge bricks a watched world
  overnight, so every rail has to survive the arm that removes them."""
  life = _life()
  assert life.autonomous is False
  life.battery.energy_wh = 0.01
  assert life.needs_charge is True, "the floor"
  assert life.claim_budget_wh == life.spendable_wh, "the offer filter"
  # ...and the gate, which is the one that actually does the work: an errand
  # priced above what is left must not be startable.
  dear = Task.create("rate_artwork", "whiteboard_a", "t_0001",
                     estimate_wh=1.18)
  assert dear.claimable(0.0, life.claim_budget_wh) is False
  assert dear.claimable(0.0, None) is True, "...and off, it is takeable"


def test_the_rails_are_read_in_exactly_one_place_each():
  """One flag, three readers, and nothing else. A rail removed somewhere
  this does not name is a rail that comes off on the deployed world too."""
  import inspect
  src = inspect.getsource(lc.HubLifecycle)
  # NAMED, not counted: everything the arm changes in the mission loop is on
  # this list, and adding to it is a deliberate act. Three of them are the
  # rails; `idle_s` is a cadence, and it is here so that it cannot be
  # mistaken for a fourth rail by whoever reads the count.
  rails = {"needs_charge": lc.HubLifecycle.needs_charge.fget,
           "_afford_next": lc.HubLifecycle._afford_next,
           "claim_budget_wh": lc.HubLifecycle.claim_budget_wh.fget}
  other = {"idle_s": lc.HubLifecycle.idle_s.fget}
  for name, method in {**rails, **other}.items():
    assert "self.autonomous" in inspect.getsource(method), name
  # one assignment in `__init__`, plus exactly these readers
  assert src.count("self.autonomous") == 1 + len(rails) + len(other), \
      "something else in the mission loop now branches on the arm"


# ---- the guard that would have silenced the arm -------------------------------


class _Broken:
  """A client whose every call fails, counting the attempts."""

  def __init__(self) -> None:
    self.calls = 0
    self.messages = self

  def create(self, **kw):
    self.calls += 1
    raise RuntimeError("no")


def test_a_run_of_failed_calls_does_not_silence_the_model_for_ever(monkeypatch):
  """⚠ THE LATCH (issue #115). `MAX_IDLE_RUN` exists to stop a MODEL that
  keeps answering `idle` from burning the budget.

  A failed call on `autonomous` fires the agent's STANDING ORDER -- but
  `idle` is the floor while no order has been set, which is the state every
  mission starts in. So counting fallbacks means two early failures take the
  counter to the limit, `_refuse` answers `idle-run` without dispatching,
  that answer is `idle` too, and it climbs for ever. ⚠ An order can only be
  set by a SUCCESSFUL call, so the agent can never acquire the thing that
  would have freed it: the trap springs only when it is defenceless, and it
  never reopens.

  It would not look like a bug. It would look like the finding: a record
  full of `idle` and a robot that sat still until its pack ran out.
  """
  # The endpoint back-off is a DIFFERENT guard and a correct one (three
  # failures in a row buy a cooloff), so it is lifted here to leave exactly
  # one thing under test.
  monkeypatch.setattr(ov, "MAX_CONSECUTIVE_ERRORS", 99)
  client = _Broken()
  boss = Overseer(MENU, client=client, autonomous=True, standing_orders=True,
                  calls_per_hour=99)
  state = {"decisions": 0, "mapDone": True}
  for _ in range(ov.MAX_IDLE_RUN + 3):
    decision = boss.decide(state)
    assert decision.action == ov.STANDING_ORDER_FLOOR
    assert decision.scripted
  assert client.calls == ov.MAX_IDLE_RUN + 3, \
      "the model stopped being asked: the idle guard latched on its own " \
      "fallbacks"
  assert "fallback:idle-run" not in [d.source for d in boss.decisions]


def test_the_guard_still_catches_a_model_that_really_will_not_stop_idling():
  """The regression half: the guard is not disabled, it is narrowed to what
  it was written for. A model ANSWERING `idle` still gets cut off."""
  boss = Overseer(MENU, client=_Broken(), standing_orders=True)
  for _ in range(ov.MAX_IDLE_RUN):
    boss._record(ov.Decision(action="idle", source="llm"), {"decisions": 0})
  assert boss._refuse({"decisions": 0}) == "idle-run"


def test_two_rungs_are_two_series_even_under_one_label():
  """⚠ A rung changes what the model is SHOWN, so A0 and A1 are two
  experiments. Both are flown quiet and both want the label `quiet`, so if
  the rung were not in the series key they would average into one another --
  the silent pooling `deadlineS` was added to prevent, one field along."""
  from pluggybot.evaluation import rollup as ru

  base = {"world": "home", "arm": "autonomous", "pack": "hosting",
          "model": "m", "label": "quiet"}
  a0 = ru.series_key({**base, "config": {"rung": "A0"}})
  a1 = ru.series_key({**base, "config": {"rung": "A1"}})
  assert a0 != a1
  # ...and an arm with no ladder keeps exactly the identity it had.
  assert ru.series_key({**base, "arm": "guarded", "label": "",
                        "config": {}})[-1] == ""


class _Garbler:
  """A client that answers promptly, with something that is not a decision."""

  def __init__(self) -> None:
    self.calls = 0
    self.messages = self

  def create(self, **kw):
    self.calls += 1
    raise ValueError("task 't_0006' is not on offer (claimable: nothing)")


def test_a_run_of_bad_answers_does_not_back_off_a_healthy_endpoint():
  """⚠ MEASURED, AND IT COST A FLOWN DAY (issue #115).

  `MAX_CONSECUTIVE_ERRORS` is for an endpoint nobody is answering -- the
  missing-key case it was written for. A model that replies promptly with a
  bad task id is not that, and summing the two means three silly answers buy
  a five-minute silence that doubles.

  A0's first run: four `garbled`, the back-off on the third, then 238
  `fallback:cooloff` decisions firing the agent's own `idle` order at four
  sim-seconds each until the pack was flat. 249 decisions, 7 of them the
  model's -- and the record read as a robot that chose to sit still and died.
  """
  client = _Garbler()
  boss = Overseer(MENU, client=client, autonomous=True, standing_orders=True,
                  calls_per_hour=99)
  state = {"decisions": 0, "mapDone": True}
  for _ in range(ov.MAX_CONSECUTIVE_ERRORS + 3):
    assert boss.decide(state).source == "fallback:garbled"
  assert client.calls == ov.MAX_CONSECUTIVE_ERRORS + 3, \
      "a healthy endpoint was backed off for answering badly"
  assert boss._refuse(state) == "", "no cooloff from garbled answers alone"


def test_a_dead_endpoint_is_still_backed_off():
  """The regression half: the guard is narrowed to what it was written for,
  not disabled. Nobody answering still buys the silence."""
  boss = Overseer(MENU, client=_Broken(), standing_orders=True,
                  calls_per_hour=99)
  for _ in range(ov.MAX_CONSECUTIVE_ERRORS):
    boss.decide({"decisions": 0})
  assert boss._refuse({"decisions": 0}) == "cooloff"


def test_an_empty_board_takes_take_task_off_the_menu_for_that_call():
  """⚠ MEASURED (issue #115). With no offers the id enum collapses to a free
  string, and the model names one it remembers: "task 't_0010' is not on
  offer (claimable: nothing)" was 46 of A0's 76 decisions, each one an
  answer thrown away and a call spent.

  The action enum is built PER CALL -- only the prompt is cached -- so
  taking `take_task` off it costs the prefix nothing, and it is the line
  `order_runnable` already draws for a standing order.
  """
  empty = MENU.schema(task_ids=())["properties"]["action"]["enum"]
  assert "take_task" not in empty
  assert "draw" in empty and "charge" in empty, "only that one action goes"
  # ...and it is BACK the moment something is on offer.
  assert "take_task" in MENU.schema(task_ids=("t_1",))["properties"]["action"]["enum"]
  # ⚠ `None` is not `()`: the control does not constrain at all, so its
  # grammar is what it always was.
  assert "take_task" in MENU.schema()["properties"]["action"]["enum"]
  assert MENU.schema() == MENU.schema(task_ids=None)


def test_the_idle_guard_throttles_rather_than_locking():
  """⚠ MEASURED, AND IT COST A FLOWN DAY (issue #115). The model idled twice
  -- a legitimate choice -- and the guard then refused every later call for
  ever, because only a non-idle ANSWER clears the streak and no answer was
  being collected. 331 decisions, 10 of them the model's, and the other 321
  fired its own `idle` order four sim-seconds apart until the pack was flat.

  It is the latch from the other side: there, fallbacks fed the streak;
  here, nothing could drain it. Firing must reset.
  """
  boss = Overseer(MENU, client=_Garbler(), autonomous=True,
                  standing_orders=True, calls_per_hour=99)
  state = {"decisions": 0}
  for _ in range(ov.MAX_IDLE_RUN):
    boss._record(ov.Decision(action="idle", source="llm"), state)
  assert boss._refuse(state) == "idle-run", "it still fires"
  assert boss._refuse(state) == "", "...and having fired, it lets go"
  # ...so a model that keeps idling is throttled, never silenced: one call
  # in every MAX_IDLE_RUN + 1 is skipped, and the rest are made.
  for _ in range(ov.MAX_IDLE_RUN * 3):
    boss._record(ov.Decision(action="idle", source="llm"), state)
    boss._refuse(state)
  assert boss._refuse(state) in ("", "idle-run")


def test_an_idling_robot_cannot_out_run_its_own_call_budget():
  """4 s a turn is 900 decisions an hour against a 60-call budget, so an
  agent that decides to wait spends the budget in four minutes and then
  spins on `fallback:budget`. The interval is DERIVED from the budget so the
  two cannot drift apart."""
  assert lc.AUTONOMOUS_IDLE_S == 3600.0 / ov.CALLS_PER_HOUR
  assert _life(autonomous=True).idle_s == lc.AUTONOMOUS_IDLE_S
  # ...and the control keeps the pause it always had.
  assert _life().idle_s == lc.DECIDED_IDLE_S == 4.0
