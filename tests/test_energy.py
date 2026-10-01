"""Guards for the per-errand energy model (issue #15).

economy/energy.py exists to close the one way an overseer can still strand the
robot: `needs_charge` is checked BETWEEN errands and never inside one, so a
job bigger than what is left in the pack cannot be survived by any charging
policy. What these hold down:

  1. THE ARITHMETIC HAS FOUR ANSWERS, not two. "charge and try again", "this
     world can never do that" and "this cell was always too small for this"
     want different things from the mission loop, and collapsing any pair of
     them writes either a charge/defer spin or a deleted capability.
  2. THE MARGIN IS ALL-OR-NOTHING: a world whose charged pack funds its
     dearest job plus the return trip keeps the whole reserve, and one that
     cannot keeps none. A margin charged on a battery smaller than one
     errand refuses every job in that world forever.
  3. AN ERRAND THAT WILL NOT FIT IS DEFERRED, NOT STARTED. This is the
     acceptance criterion, and `test_an_overseer_that_only_ever_picks_the_
     dearest_errand_is_sent_to_charge_first` is it as a whole day.
  4. NOTHING SPINS. An errand that cannot be paid for after two charges is
     given up on, and one no pack here could cover is dropped on sight.
  5. THE MODEL IS TOLD, and told only what was measured. Costs ride the
     cached prefix; what the pack can pay for right now rides the volatile
     turn; an unmeasured action is priced for the GATE and never printed to
     the model as though somebody had measured it.
"""

import json
from dataclasses import replace
from pathlib import Path

import mujoco
import pytest

from pluggybot import lifecycle as lc
from pluggybot.economy import energy as en
from pluggybot.legs import model as legs_model
from pluggybot.legs import world as legs_world
from pluggybot.lifecycle import QUAD_HOME
from pluggybot.mind import overseer as ov
from pluggybot.mind.thoughts import GOALS, ThoughtFiles
from pluggybot.mission.errand import Errand

RESERVE = legs_world.RESERVE_WH    # the house's return-trip reserve
DEMO_WH = legs_world.DEMO_WH       # ...its demo pack, which funds it
HOSTING_WH = legs_model.PACK_WH    # what the deployment actually runs


def table(**kw) -> en.EnergyModel:
  spec = {"world": "test", "errand_wh": {"draw": 0.93, "census": 0.97,
                                         "carry": 0.69, "dance": 0.79}}
  spec.update(kw)
  return en.EnergyModel(**spec)


def feed_errand() -> Errand:
  """The one paid job on legs, as the loop builds it from an offer."""
  return lc.cage_errand(QUAD_HOME, "feed", task="feed")


def tooled(menu):
  """A menu for a body that takes a tool (`Menu.tools`), which no served
  body does until #406/#407: the rotation over the tool errands."""
  return replace(menu, tools=True)


# ---- 1. three answers, not two ----------------------------------------------


@pytest.mark.parametrize("energy_wh, charged_wh, cost_wh, want, runs", [
  # A hosting pack with plenty in it: just do the job.
  (7.0, 7.2, None, en.OK, True),
  # The same pack, nearly flat: the job fits a full charge, so charge first.
  (1.0, 7.2, None, en.CHARGE_FIRST, False),
  # A pack that funds margins and still cannot cover THIS job -- which can
  # only be a job priced above the table, i.e. a task's own per-kind
  # estimate. Never, in this world.
  (7.0, 7.2, 9.0, en.BEYOND, False),
  # A DEMO cell, which has no margin to fund: smaller than its jobs on
  # purpose, so the job is attempted and the cell is what gets named.
  (0.90, 0.90, None, en.OVERSPEND, True),
])
def test_affordability_has_a_now_a_later_a_never_and_a_demo_cell(
    energy_wh, charged_wh, cost_wh, want, runs):
  """Four answers, three loop behaviours, and a boolean cannot express any of
  it. `charge_first` returned as `beyond` refuses work a top-up would allow;
  `beyond` returned as `charge_first` is the loop charging and retrying
  forever; `overspend` returned as `beyond` deletes a job from a cell that
  was always too small for it and ran it anyway."""
  fit = table().afford("census", energy_wh=energy_wh, charged_wh=charged_wh,
                       reserve_wh=RESERVE, cost_wh=cost_wh)
  assert fit.state == want, fit.why()
  assert fit.ok is runs


def test_the_why_line_says_which_of_the_four_it_is():
  """The narration channel is what a person watching reads, and "deferred",
  "given up on" and "this cell was always too small" must not look the same
  on the wire -- the same argument `Decision.source` is built on."""
  t = table()

  def why(**kw) -> str:
    return t.afford("census", reserve_wh=RESERVE, **kw).why()

  assert "charging first" in why(energy_wh=1.0, charged_wh=7.2)
  assert "not possible in this world" in why(energy_wh=7.0, charged_wh=7.2,
                                             cost_wh=9.0)
  assert "smaller than this job" in why(energy_wh=0.9, charged_wh=0.9)
  assert "costs about" in why(energy_wh=7.0, charged_wh=7.2)


# ---- 2. the margin ----------------------------------------------------------


def test_the_demo_pack_charges_the_whole_margin():
  """The served world's demo pack funds its dearest job AND the return
  trip, so it keeps the trip in hand like the hosting one, and the
  mid-errand death is not reachable on the world the tests fly."""
  t = en.load(QUAD_HOME)
  assert t.margin_wh(DEMO_WH * lc.CHARGED, RESERVE) == RESERVE


def test_a_hosting_pack_charges_the_whole_reserve():
  """...and the same table on a battery the deployment actually runs keeps
  the return trip in hand, which is the entire point."""
  t = en.load(QUAD_HOME)
  assert t.margin_wh(HOSTING_WH * lc.CHARGED, RESERVE) == RESERVE


def test_the_margin_flips_on_the_dearest_errand_not_the_mean():
  """A world that can fund its average job and not its worst one has no
  margin to keep: the errand that strands the robot is the expensive one."""
  t = table(errand_wh={"cheap": 0.1, "dear": 2.0})
  assert t.margin_wh(2.0 + 0.5, 0.5) == 0.5
  assert t.margin_wh(2.0 + 0.4, 0.5) == 0.0       # 0.1 short of the dear one


def test_a_targets_own_row_is_priced_apart_from_the_bare_action():
  """A cost key may name a TARGET (`care:feed`), which wins over the bare
  action's row: two plates, or two whiteboards, are not the same trip.
  Padding the table is the fix not to reach for; a second measured row
  costs nothing and is true."""
  t = en.load(QUAD_HOME)
  assert t.cost("care", "feed") == t.errand_wh["care:feed"] != t.cost("care")
  # ...and an unmeasured target falls back to the bare action rather than to
  # the dearest thing in the world.
  assert t.cost("care", "nowhere") == t.cost("care")
  # The margin has to survive the DEAREST thing the robot can be asked for,
  # per-target rows included, or a hosting pack could still be caught out.
  assert t.dearest_wh() >= max(t.errand_wh.values())


def test_a_task_offer_is_priced_for_the_target_it_names():
  from pluggybot.economy.tasks import TaskBoard
  board = TaskBoard(energy=table(errand_wh={"draw": 0.93, "draw:whiteboard_b": 1.11}))
  near = board.offer("draw_figure", "whiteboard_a", params={"program": "house"})
  far = board.offer("draw_figure", "whiteboard_b", params={"program": "house"})
  assert far.estimate_wh > near.estimate_wh, (near.estimate_wh, far.estimate_wh)


def test_an_unmeasured_errand_is_priced_as_the_dearest_one():
  """Pessimistic on purpose. An unpriced errand guessed cheap is the mid-
  errand death this module exists to prevent; guessed dear it is only ever a
  charge the robot did not strictly need."""
  t = table()
  assert t.cost("something_new") == max(t.errand_wh.values())
  assert en.EnergyModel(world="bare").cost("anything") == en.FALLBACK_WH


def test_a_free_errand_is_refused_at_load(tmp_path):
  """A table edited into saying an errand costs nothing is a table saying the
  one thing it must never say."""
  bad = tmp_path / "energy.json"
  bad.write_text(json.dumps({"version": en.ENERGY_VERSION,
                             "worlds": {QUAD_HOME: {"errandWh": {"feed": 0.0}}}}))
  with pytest.raises(ValueError, match="zero or less"):
    en.load(QUAD_HOME, bad)


def test_a_version_this_build_does_not_know_is_refused(tmp_path):
  bad = tmp_path / "energy.json"
  bad.write_text(json.dumps({"version": en.ENERGY_VERSION + 7}))
  with pytest.raises(ValueError, match="energy version"):
    en.load(QUAD_HOME, bad)


def test_the_shipped_table_is_the_one_the_deploy_can_repoint(tmp_path, monkeypatch):
  """$PLUGGY_ENERGY, on the same terms as $PLUGGY_REWARDS and
  $PLUGGY_CADENCE: a mounted file, no rebuild."""
  mine = tmp_path / "energy.json"
  mine.write_text(json.dumps({"version": en.ENERGY_VERSION,
                              "worlds": {QUAD_HOME: {"errandWh": {"feed": 3.5}}}}))
  monkeypatch.setenv(en.ENERGY_ENV, str(mine))
  assert en.load(QUAD_HOME).cost("feed") == 3.5


def _shipped_worlds() -> list[str]:
  doc = json.loads((Path(__file__).parents[1]
                    / "src/pluggybot/economy/energy.json").read_text())
  return list(doc["worlds"])


def test_a_task_kind_is_never_priced_below_the_errand_that_discharges_it():
  """⚠ TWO TABLES, ONE TRUTH, AND THE DRIFT IS SILENT. `TaskKind.estimate_wh`
  is what a board with no row for a job prices it at, and it is a separate
  number from economy/energy.json. Under-priced, it is a robot that takes
  on a job it cannot finish.

  This is not hypothetical: `count_plants` was 0.87 Wh against a census that
  measured 1.12, which the `ENERGY ... economy/energy.json is low` narration
  line caught on a real unattended run. The dearest measured world is the
  bar, because the kind's figure is world-agnostic and the offer could be
  anywhere.
  """
  from pluggybot.economy.tasks import KINDS
  dearest: dict = {}
  for world in _shipped_worlds():
    for key, wh in en.load(world).errand_wh.items():
      # `care` and `care:feed` are the same JOB priced for different
      # targets, and the fallback has to cover the dearest of them: it is
      # what a world nobody has measured is charged, and there is no target
      # row there to correct it.
      action = key.split(":", 1)[0]
      dearest[action] = max(dearest.get(action, 0.0), wh)
  for name, spec in KINDS.items():
    floor = dearest.get(spec.task)
    if floor is None:
      continue                          # nothing measured for it yet
    assert spec.estimate_wh >= floor - 1e-9, (
      f"{name} is priced at {spec.estimate_wh} Wh against a measured "
      f"{floor} Wh {spec.task} -- re-run scripts/energy_spike.py")


def test_every_shipped_world_prices_every_errand_it_can_build():
  """A world that can build an errand it has no measurement for falls back to
  its dearest, which is safe but is not a measurement. This is the reminder
  to run scripts/energy_spike.py when a world learns a new trick."""
  for world in _shipped_worlds():
    book = lc.board_book(world)
    priced = set(en.load(world).errand_wh)
    for action in (*ov.ERRAND_ACTIONS, "feed", "shock"):
      try:
        lc.errands_for(action, world, book)
      except ValueError:
        continue                      # this world cannot do it at all
      assert action in priced, f"{world} can do {action} and has no cost for it"


# ---- 3 & 4. the gate in the mission loop ------------------------------------


def life_with(battery_wh: float = HOSTING_WH, errands=None, **kw) -> lc.HubLifecycle:
  """A lifecycle with nothing driving: `_afford_next` touches no physics,
  so it lives on a stub body (issue #380) and this stays a fast test."""
  from test_body import stub_life
  return stub_life(battery_wh=battery_wh, errands=errands or [], **kw)


def test_an_errand_that_will_not_fit_is_deferred_and_stays_queued():
  """THE ACCEPTANCE CRITERION, as arithmetic: an errand whose cost exceeds
  what the pack can spend is not started. Deferred, not dropped -- the job is
  still the job, and the answer is a charge."""
  life = life_with(errands=[feed_errand()])
  life.battery.energy_wh = 0.8            # feed 1.77 + the 3.7 reserve
  assert life._afford_next() is False
  assert len(life.errands) == 1, "a deferred errand must survive the charge"
  assert any("DEFER" in line for line in life.log), life.log


def test_the_same_errand_runs_once_the_pack_can_pay_for_it():
  life = life_with(errands=[feed_errand()])
  life.battery.energy_wh = life.energy.cost("feed") + RESERVE + 1.0
  assert life._afford_next() is True
  assert len(life.errands) == 1


def test_an_errand_no_charge_could_cover_is_dropped_not_deferred():
  """⚠ THE SPIN. `beyond` deferred would be: charge, still short, charge
  again, forever -- a robot that never does anything, wearing a safety
  feature's clothes. It is dropped, said out loud, and the loop moves on."""
  life = life_with(errands=[Errand(name="huge", module="module_lcd",
                                   station_y=0.0, use_at=(1.5, 1.8),
                                   task="carry", estimate_wh=HOSTING_WH)])
  assert life._afford_next() is True       # nothing left to gate
  assert life.errands == []
  assert any("SKIP" in line and "not possible in this world" in line
             for line in life.log), life.log


def test_the_demo_pack_funds_the_dearest_job_with_the_reserve_intact():
  """No shipped world offers a job bigger than its own charged pack less the
  reserve, so `overspend` is reachable only synthetically -- and it stays
  guarded there, in
  `test_affordability_has_a_now_a_later_a_never_and_a_demo_cell`, which
  builds its own too-small cell. The four answers still exist and still must
  not collapse; a future world that exercises the fourth is flown again."""
  gift = lc.cage_errand(QUAD_HOME, "toy")                  # care, the dearest row
  life = life_with(battery_wh=DEMO_WH, errands=[gift])
  assert life.energy.cost("care") == life.energy.dearest_wh()
  assert life.energy.dearest_wh() + RESERVE <= life.charged_wh, \
      "the demo pack no longer holds reserve + the dearest job"
  assert life._afford_next() is True
  assert len(life.errands) == 1
  assert not any("smaller than this job" in line for line in life.log), \
      "the demo pack went back to overspending"


def test_an_errand_deferred_too_often_is_given_up_on():
  """The other spin, and a different fault: the pack COULD hold this job and
  does not, which means charging is what is broken. Two goes, then the errand
  is dropped rather than blocking the queue behind it."""
  life = life_with(errands=[feed_errand()])
  life.battery.energy_wh = 0.8
  for _ in range(lc.MAX_ERRAND_DEFERRALS):
    assert life._afford_next() is False    # ...charge would go here
  assert life._afford_next() is True
  assert life.errands == []
  assert any("giving up on it" in line for line in life.log), life.log


def test_a_task_errand_is_priced_by_its_own_kind_not_by_the_action():
  """A task's estimate knows which target it is being asked about; a
  per-action figure cannot. The task's has to win, or a far target is
  priced as the near one."""
  life = life_with()
  errand = feed_errand()
  errand.estimate_wh = 3.0
  assert life.affords(errand).cost_wh == 3.0
  errand.estimate_wh = 0.0
  assert life.affords(errand).cost_wh == en.load(QUAD_HOME).cost("feed")


def test_the_demo_pack_keeps_the_return_trip_out_of_a_job_s_budget():
  """The demo pack funds the margin, so `spendable_wh` is the pack LESS the
  reserve -- the same shape the hosting test below asserts."""
  life = life_with(battery_wh=DEMO_WH)
  # A pack ABOVE the reserve, said relative to it: spendable energy is
  # clamped at zero under it, which is a different claim.
  life.battery.energy_wh = RESERVE + 1.0
  assert life.reserve_margin_wh == RESERVE
  assert life.spendable_wh == pytest.approx(1.0)
  # ...and what the WORLD can offer shrinks by the same reserve: `fundable_wh`
  # is a charged pack less the margin.
  assert life.fundable_wh == pytest.approx(DEMO_WH * lc.CHARGED - RESERVE)


def test_a_hosting_pack_keeps_the_return_trip_out_of_a_job_s_budget():
  life = life_with(battery_wh=HOSTING_WH)
  life.battery.energy_wh = RESERVE + 3.0
  assert life.reserve_margin_wh == RESERVE
  assert life.spendable_wh == pytest.approx(3.0)


def test_a_charge_timeout_is_sized_against_the_pack_it_has_to_fill():
  """⚠ A TIMEOUT IN SECONDS IS A TIMEOUT IN WATT-HOURS. 400 s was right for a
  0.7 Wh cell and quietly ended the deployed 8 Wh one at about two thirds,
  narrating "CHARGE complete (65 %)" every cycle."""
  demo = life_with(battery_wh=DEMO_WH)
  host = life_with(battery_wh=HOSTING_WH)
  rate = en.load(QUAD_HOME).charge_w
  assert demo.charge_timeout >= demo.charged_wh * 3600.0 / rate
  assert demo.charge_timeout >= lc.CHARGE_TIMEOUT_MIN
  # Long enough to actually put a hosting pack's worth in at the measured
  # net rate, with the slack a real press needs.
  assert host.charge_timeout >= host.charged_wh * 3600.0 / rate
  assert host.charge_timeout > demo.charge_timeout


# ---- 5. what the model is shown ---------------------------------------------


def lab_menu():
  """The house's menu with the lab's `care` on it -- the one errand a mind
  may choose on legs, and so the one the prompt prices."""
  menu = replace(ov.Menu.for_world(QUAD_HOME, lc.board_book(QUAD_HOME)), lab="lab")
  return replace(menu, costs_wh=en.load(QUAD_HOME).as_context(menu.available()))


def test_the_costs_ride_the_cached_prefix_and_not_the_turn():
  """What an errand costs is a property of the world, so it belongs in the
  stable half. Putting it in the volatile turn would invalidate the prompt
  cache on every call for a number that never changes -- the classic silent
  invalidator docs/Overseer.md §6 is about."""
  menu = lab_menu()
  prompt = ov.system_prompt(ThoughtFiles(texts={GOALS: "be useful"}), menu,
                             ov.default_table())
  text = prompt[0]["text"]
  assert "energyCostWh" in text
  assert '"care"' in text and str(menu.costs_wh["care"]) in text


def test_only_measured_costs_are_shown_to_the_model():
  """⚠ `cost()` answers for ANY name, because the gate has to price an
  unmeasured errand at something. Printing that fallback would tell the model
  that `idle` costs 1.86 Wh -- false, and exactly the confident wrong number
  the rest of this design keeps out of the prompt."""
  for menu in (lab_menu(), ov.Menu.for_world(QUAD_HOME, lc.board_book(QUAD_HOME))):
    assert set(menu.costs_wh) <= set(en.load(QUAD_HOME).errand_wh)
    for free in ("idle", "explore", "charge", "take_task"):
      assert free not in menu.costs_wh


def test_the_scripted_policy_never_rotates_onto_what_the_world_cannot_do():
  """The fallback is a real day's work, so it has to obey the same gate the
  loop does -- otherwise the API going down means the robot proposing an
  errand this world refuses, over and over, until the budget runs out."""
  menu = tooled(ov.Menu.for_world(QUAD_HOME, lc.board_book(QUAD_HOME)))
  state = {"decisions": 0, "floorExplored": True,
           "possibleActions": ["carry", "explore", "idle", "charge"]}
  for _ in range(4):
    d = ov.scripted(menu, state, "test")
    assert d.action in ("carry", "explore", "idle", "charge"), d.action
    state["decisions"] += 1


def test_the_scripted_policy_still_picks_what_a_charge_would_afford():
  """⚠ THE TIGHTER LIST WOULD STARVE IT. `affordableActions` is what the pack
  can pay for THIS SECOND, and an errand it cannot is one the loop charges for
  and then runs -- so filtering the rotation on it would put the robot on
  `explore` for the whole minute before every charge, which is not the
  fallback doing a day's work."""
  menu = tooled(ov.Menu.for_world(QUAD_HOME, lc.board_book(QUAD_HOME)))
  d = ov.scripted(menu, {"decisions": 0, "floorExplored": True,
                         "affordableActions": [],
                         "possibleActions": ["draw", "carry"]}, "test")
  assert d.action == "draw"


def test_an_empty_possible_list_filters_nothing():
  """A caller that supplies none -- a unit test, an older context dict --
  must not be read as "this robot can do nothing"."""
  menu = tooled(ov.Menu.for_world(QUAD_HOME, lc.board_book(QUAD_HOME)))
  d = ov.scripted(menu, {"decisions": 0, "floorExplored": True}, "test")
  assert d.action == "draw"


def test_the_context_carries_what_the_pack_can_actually_spend():
  """`wh` is what is in the pack; `spendableWh` is what a job may cost. They
  differ by the margin, and on a hosting pack that difference is the whole
  guard -- a model shown only `wh` would plan against energy it is not
  allowed to spend."""
  life = life_with(battery_wh=HOSTING_WH)
  life.battery.energy_wh = RESERVE + 3.0
  state = ov.context_for(life, affordable=["care"], possible=["care", "explore"])
  assert state["battery"]["wh"] == pytest.approx(RESERVE + 3.0)
  assert state["battery"]["spendableWh"] == pytest.approx(3.0)
  assert state["affordableActions"] == ["care"]
  assert state["possibleActions"] == ["care", "explore"]


def test_the_prompt_still_never_carries_a_hidden_answer():
  """The energy block is new context, and new context is a new chance to
  leak. Same claim as `test_the_prompt_never_carries_a_hidden_answer`, made
  again against the half of the prompt this issue touched."""
  text = ov.system_prompt(ThoughtFiles(texts={GOALS: "be useful"}), lab_menu(),
                          ov.default_table())[0]["text"]
  assert "truth" not in text.lower().split("energycostwh")[-1]


# ---- 3, the gate and the cap -------------------------------------------------

#: The SMALLEST pack that is in the margin regime here (the dearest errand,
#: 1.859 Wh, plus the 3.7 Wh reserve must fit a charged pack: capacity >=
#: 5.559 / 0.9 = 6.18), so these cost one short charge. The regime is what is
#: under test, not the capacity -- the hosting pack is the same arithmetic
#: with more room in it. A pack below the floor silently drops to zero
#: margin (the all-or-nothing rule), which is exactly what the first
#: assertion catches.
MARGIN_PACK_WH = 6.2


class OneNote:
  """An `anthropic` client stand-in that answers the same action forever.

  Deliberately local rather than imported from `tests/test_overseer.py`:
  `tests/` is not a package, and a cross-test import that works under one
  invocation and not another is a test that fails for a reason nobody in it
  is talking about.
  """

  class _Response:
    def __init__(self, payload):
      self.content = [type("Block", (), {"type": "text",
                                         "text": json.dumps(payload)})()]
      self.usage = type("Usage", (), {
        "input_tokens": 1200, "output_tokens": 40,
        "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0})()

  def __init__(self, action: str, reason: str, **fields) -> None:
    self.payload = {"action": action, "reason": reason, "board": "",
                    "program": "", "zone": "", "note": "", "respond_to": "",
                    "outcome": "", "reply": "", "task": "", "answer": "",
                    **fields}
    self.calls: list[dict] = []
    self.messages = self

  def create(self, **kwargs):
    self.calls.append(kwargs)
    return self._Response(self.payload)


# What regresses is a RULE -- an inequality in `_afford_next`, the loop bound
# in `charge()`, the two factors of `charge_timeout` agreeing -- pinned first,
# then a day through each on the stub (issues #158, #380). What only physics
# adds is the press's real rate and an errand's real cost: measurements,
# `scripts/energy_spike.py`'s, folded into energy.json.


def test_the_gate_refuses_an_errand_the_pack_cannot_finish_and_permits_one_it_can():
  """The rule the day below runs on: `_afford_next` is False when cost plus
  the return-trip reserve exceeds what is in the pack, and True when it
  does not. Shown to fail with `_afford_next` returning True
  unconditionally, which is the regression that day was written for.
  """
  life = life_with(battery_wh=MARGIN_PACK_WH)
  assert life.reserve_margin_wh == RESERVE, "not in the margin regime"
  cost = life.energy.cost("feed")
  assert cost + RESERVE <= life.charged_wh, "the feed must be fundable"

  # Just short of cost + reserve: the gate must send the robot to charge.
  life.errands = [feed_errand()]
  life.battery.energy_wh = cost + RESERVE - 0.01
  assert life._afford_next() is False, "started an errand it could not finish"
  assert life.errands, "a deferred errand was dropped rather than kept"
  # Just over: the same errand runs.
  life.errands = [feed_errand()]
  life.battery.energy_wh = cost + RESERVE + 0.01
  assert life._afford_next() is True, "refused an errand the pack covers"


def test_the_charge_loop_is_bounded_by_the_scaled_cap_not_the_old_constant():
  """The wiring the table-rate charge below runs through: `charge()` gives
  a cycle `self.charge_timeout` seconds, sized to THIS pack, and not the
  flat `CHARGE_TIMEOUT` that ended every deployed charge at 65 %.

  The press is FAKED -- the stub's hold advances the sim clock and steps no
  physics -- and the pack is held below full, so the loop runs to its bound
  and the bound is what is measured. Shown to fail by putting
  `CHARGE_TIMEOUT` back in place of `self.charge_timeout` in
  `HubLifecycle.charge_routine`: the elapsed time collapses to 400 s.
  """
  life = life_with(battery_wh=HOSTING_WH, charge_scale=1.0)
  assert life.charge_timeout > lc.CHARGE_TIMEOUT, "pick a bigger pack"

  def fake_press(seconds):
    life.data.time += float(seconds)
    return
    yield
  life.body.dock_hold_routine = fake_press
  life.charging_now = True                   # the pins conduct throughout
  life.battery.energy_wh = 0.9               # ...and it never fills

  t0 = life.data.time
  life.charge()
  elapsed = life.data.time - t0
  assert elapsed == pytest.approx(life.charge_timeout, abs=1.0), \
      f"the cycle ran {elapsed:.0f} s against a cap of {life.charge_timeout:.0f}"
  assert elapsed > lc.CHARGE_TIMEOUT * 1.5, \
      "the loop is still reading the flat 400 s constant"


@pytest.mark.parametrize("scale", [1.0, 2.0, 5.0])
@pytest.mark.parametrize("pack_wh", [0.7, 6.0])
def test_the_scaled_cap_still_clears_the_time_a_full_charge_takes(pack_wh,
                                                                  scale):
  """The two factors of `charge_timeout` agreeing: the cap divides by the
  scale and so does the fill time, so at every scale the cap stays
  `CHARGE_TIMEOUT_SLACK` above what a charge from empty actually needs.
  The two ways it goes wrong -- forget the SLACK and a press that drops
  contact once is cut off short of full; forget the FLOOR and a small
  pack's cap shrinks under the time it takes to seat the pins -- are each
  a mutation this catches. The rate is the SLOWEST measured press, which
  is the one the cap has to cover.
  """
  life = life_with(battery_wh=pack_wh, charge_scale=scale)
  fill_s = life.charged_wh * 3600.0 / (life.energy.charge_w * scale)
  assert life.charge_timeout >= fill_s * lc.CHARGE_TIMEOUT_SLACK * 0.999, \
      "the cap sits under the slack a real press needs"
  assert life.charge_timeout >= lc.CHARGE_TIMEOUT_MIN, \
      "the floor is gone: a small pack gets less than it takes to seat the pins"


def test_an_overseer_that_only_ever_picks_the_dearest_errand_is_sent_to_charge_first():
  """⚠ THE ACCEPTANCE CRITERION, adversarially, as a day on the stub (issue
  #380): an overseer that answers `care` -- the dearest errand a mind can
  choose on legs -- to every question, on a pack just short of its cost
  plus the return-trip reserve. The loop must refuse it IN ADVANCE, charge,
  and only then run it: not "a sensible model plans well" but "a model that
  plans badly cannot strand the robot", the shape of
  `test_charge_priority_survives_an_overseer_that_never_charges`.

  The refusal is the GATE's (the pack is above the floor), and the model is
  gated rather than replaced: the act run is the one it chose. The gate is
  rail two, so the arm is the one that keeps it, with the lab's `care` on
  its menu. Shown to fail with `_afford_next` returning True
  unconditionally: the act is set off on before any charge.
  """
  boss = ov.Overseer(lab_menu(), client=OneNote("care", "I like the mouse", care="toy"))
  life = life_with(battery_wh=MARGIN_PACK_WH, overseer=boss,
                   boards=lc.board_book(QUAD_HOME))
  cost = life.energy.cost("care")
  assert cost == life.energy.dearest_wh()
  assert life.reserve_margin_wh == RESERVE, "not in the margin regime"
  assert cost + RESERVE <= life.charged_wh
  life.battery.energy_wh = cost + RESERVE - 0.01
  assert not life.needs_charge, "the floor would refuse it; the gate is under test"
  sent: list[tuple[str, float]] = []
  for name in ("dock_routine", "find_tag_routine"):
    def spy(*a, _real=getattr(life.body, name), _name=name, **kw):
      sent.append((_name, life.battery.energy_wh))
      return (yield from _real(*a, **kw))
    setattr(life.body, name, spy)
  # Ends at the first errand's result, where the claim is decided either way
  # (charged first, or not): an ungated act decided again and again on the
  # stub runs away rather than failing. The budget is a backstop with room
  # for a late answer, which on the stub is SIM time.
  life.stop_when(lambda: len(life.errand_results) >= 1)
  r = life.run(lc.world_config(QUAD_HOME)["start"], max_sim_time=300.0)

  assert any("DEFER care" in line for line in life.log), \
      f"the act was never deferred: {life.log[-8:]}"
  assert [n for n, _ in sent][:2] == ["dock_routine", "find_tag_routine"], \
      f"the act started before a charge: {sent[:2]}"
  assert sent[1][1] >= cost + RESERVE, "set off before the pack covered it"
  assert r["errands"], r["errands"]
  assert any(d["action"] == "care" and d["source"] == "llm"
             for d in r["decisions"]), "the fallback chose it, not the model"


@pytest.mark.parametrize("scale", [1.0, 5.0])
def test_a_charge_at_the_tables_own_rate_completes_inside_its_cap(scale):
  """⚠ A TIMEOUT IN SECONDS IS A TIMEOUT IN WATT-HOURS, through `charge()`:
  a 40 Wh pack from 0.9 Wh, on a stub whose pins net exactly the table's
  `chargeW` -- the slowest press measured -- reaches CHARGED rather than
  hitting its cap partway up and narrating "CHARGE complete (65 %)", which
  is what the deployed 8 Wh sim once did. At 1x that takes longer than the
  old flat 400 s; at 5x it fits a cap five times tighter (and floored).
  Whether a real press nets the table's rate is
  `scripts/energy_spike.py`'s to measure.

  Shown to fail by putting `CHARGE_TIMEOUT` back in place of
  `self.charge_timeout` in `HubLifecycle.charge_routine`.
  """
  from pluggybot.body import STUB_WORLD, StubBody

  from test_body import stub_life
  cfg = lc.world_config(QUAD_HOME)
  # The stub's clock at 10 ms rather than 2: the press in a fifth of the
  # steps, and nothing the stub steps is physics.
  model = mujoco.MjModel.from_xml_string(
    STUB_WORLD.replace("<worldbody>", '<option timestep="0.01"/><worldbody>'))
  body = StubBody(model, mujoco.MjData(model), rack=cfg["rack"],
                  grid_bounds=cfg["grid_bounds"])
  # 40 Wh: a pack the table's press takes longer than the old 400 s to fill
  life = stub_life(body=body, battery_wh=40.0, errands=[], charge_scale=scale)
  life.battery.draw_w = life.battery.charge_w - life.energy.charge_w
  life.battery.energy_wh = 0.9
  topped: list[float] = []
  life.say_hooks.append(
    lambda _t, msg: topped.append(life.battery.fraction)
    if msg.startswith("CHARGE complete") else None)
  t0 = float(life.data.time)
  assert life.go_charge(), "never reached the dock"
  life.charge()
  took = float(life.data.time) - t0
  assert topped and topped[0] >= lc.CHARGED, \
      f"the charge stopped at {(topped or [0])[0]:.1%} -- the cap and the rate disagree"
  assert life.charge_cycles == 1 and took < life.charge_timeout
  if scale == 1.0:
    assert took > lc.CHARGE_TIMEOUT, "the premise: longer than the old flat cap"
