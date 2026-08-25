"""Guards for the per-errand energy model (issue #15).

hub/energy.py exists to close the one way an overseer can still strand the
robot: `needs_charge` is checked BETWEEN errands and never inside one, so a
job bigger than what is left in the pack cannot be survived by any charging
policy. What these hold down:

  1. THE ARITHMETIC HAS FOUR ANSWERS, not two. "charge and try again", "this
     world can never do that" and "this cell was always too small for this"
     want different things from the mission loop, and collapsing any pair of
     them writes either a charge/defer spin or a deleted capability.
  2. THE MARGIN IS ALL-OR-NOTHING, and on both demo cells it is zero -- which
     is what makes it true that every existing mission behaves exactly as it
     did. A margin charged on a battery smaller than one errand refuses every
     job in every world forever.
  3. AN ERRAND THAT WILL NOT FIT IS DEFERRED, NOT STARTED. This is the
     acceptance criterion, and `test_a_hosting_pack_charges_before_the_errand`
     is it flown on real physics.
  4. NOTHING SPINS. An errand that cannot be paid for after two charges is
     given up on, and one no pack here could cover is dropped on sight.
  5. THE MODEL IS TOLD, and told only what was measured. Costs ride the
     cached prefix; what the pack can pay for right now rides the volatile
     turn; an unmeasured action is priced for the GATE and never printed to
     the model as though somebody had measured it.
"""

import json

import mujoco
import pytest

from pluggybot.hub import energy as en
from pluggybot.hub import lifecycle as lc
from pluggybot.hub import overseer as ov
from pluggybot.hub.errand import Errand, carry_errand

HOME_RESERVE = 0.55            # home_world's return-trip reserve
HOME_DEMO_WH = 1.10            # ...and its demo cell
HOSTING_WH = 8.0               # what the deployment actually runs


def model_of(world: str = "home"):
  return mujoco.MjModel.from_xml_path(lc.world_config(world)["model"])


def table(**kw) -> en.EnergyModel:
  spec = {"world": "test", "errand_wh": {"draw": 0.93, "census": 0.97,
                                         "carry": 0.69, "dance": 0.79}}
  spec.update(kw)
  return en.EnergyModel(**spec)


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
  forever; `overspend` returned as `beyond` deletes the home census from
  every mission that has ever run one."""
  fit = table().afford("census", energy_wh=energy_wh, charged_wh=charged_wh,
                       reserve_wh=HOME_RESERVE, cost_wh=cost_wh)
  assert fit.state == want, fit.why()
  assert fit.ok is runs


def test_the_why_line_says_which_of_the_four_it_is():
  """The narration channel is what a person watching reads, and "deferred",
  "given up on" and "this cell was always too small" must not look the same
  on the wire -- the same argument `Decision.source` is built on."""
  t = table()

  def why(**kw) -> str:
    return t.afford("census", reserve_wh=HOME_RESERVE, **kw).why()

  assert "charging first" in why(energy_wh=1.0, charged_wh=7.2)
  assert "not possible in this world" in why(energy_wh=7.0, charged_wh=7.2,
                                             cost_wh=9.0)
  assert "smaller than this job" in why(energy_wh=0.9, charged_wh=0.9)
  assert "costs about" in why(energy_wh=7.0, charged_wh=7.2)


# ---- 2. the margin, and the demo cells that must not move -------------------


def test_a_demo_cell_charges_no_margin_at_all():
  """⚠ THE LOAD-BEARING COMPATIBILITY CLAIM. One errand costs roughly one
  full pack on both demo cells, so a reserve margin charged there refuses
  every job in every world forever -- the task system silently doing nothing,
  which is the failure issue #21 already paid for once. Zero here is what
  makes "every existing mission behaves exactly as it did" true."""
  for world, cell, reserve in (("home", HOME_DEMO_WH, HOME_RESERVE),
                               ("room_hub", 0.70, lc.LOW_BATTERY_WH)):
    t = en.load(world)
    assert t.margin_wh(cell * lc.CHARGED, reserve) == 0.0, world


def test_a_hosting_pack_charges_the_whole_reserve():
  """...and the same table on a battery the deployment actually runs keeps
  the return trip in hand, which is the entire point."""
  t = en.load("home")
  assert t.margin_wh(HOSTING_WH * lc.CHARGED, HOME_RESERVE) == HOME_RESERVE


def test_the_margin_flips_on_the_dearest_errand_not_the_mean():
  """A world that can fund its average job and not its worst one has no
  margin to keep: the errand that strands the robot is the expensive one."""
  t = table(errand_wh={"cheap": 0.1, "dear": 2.0})
  assert t.margin_wh(2.0 + 0.5, 0.5) == 0.5
  assert t.margin_wh(2.0 + 0.4, 0.5) == 0.0       # 0.1 short of the dear one


def test_the_far_board_is_priced_apart_from_the_near_one():
  """⚠ THE DEFECT CLAUDE.md RECORDS UNDER ISSUE #21, closed. home's two
  whiteboards are not the same job: `whiteboard_b` is 7 m away through a
  doorway and measures 1.113 Wh against `whiteboard_a`'s 0.929. One number
  for both either kills the robot on the way back from the far one -- which
  is exactly what happened, a job claimed at 88 %, drawn perfectly, dead at
  0 %% -- or prices the near one off a demo cell that draws on it every run.

  Padding the table is the fix the note there tells you not to reach for. A
  second measured row costs nothing and is true.
  """
  t = en.load("home")
  assert t.cost("draw", "whiteboard_b") > t.cost("draw", "whiteboard_a")
  # ...and an unmeasured target falls back to the bare action rather than to
  # the dearest thing in the world.
  assert t.cost("draw", "whiteboard_z") == t.cost("draw")
  # The margin has to survive the DEAREST thing the robot can be asked for,
  # per-target rows included, or a hosting pack could still be caught out.
  assert t.dearest_wh() >= t.cost("draw", "whiteboard_b")


def test_a_task_offer_is_priced_for_the_board_it_names():
  from pluggybot.hub.tasks import TaskBoard
  board = TaskBoard(energy=en.load("home"))
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
                             "worlds": {"home": {"errandWh": {"draw": 0.0}}}}))
  with pytest.raises(ValueError, match="zero or less"):
    en.load("home", bad)


def test_a_version_this_build_does_not_know_is_refused(tmp_path):
  bad = tmp_path / "energy.json"
  bad.write_text(json.dumps({"version": en.ENERGY_VERSION + 7}))
  with pytest.raises(ValueError, match="energy version"):
    en.load("home", bad)


def test_the_shipped_table_is_the_one_the_deploy_can_repoint(tmp_path, monkeypatch):
  """$PLUGGY_ENERGY, on the same terms as $PLUGGY_REWARDS and
  $PLUGGY_CADENCE: a mounted file, no rebuild."""
  mine = tmp_path / "energy.json"
  mine.write_text(json.dumps({"version": en.ENERGY_VERSION,
                              "worlds": {"home": {"errandWh": {"draw": 3.5}}}}))
  monkeypatch.setenv(en.ENERGY_ENV, str(mine))
  assert en.load("home").cost("draw") == 3.5


def test_a_task_kind_is_never_priced_below_the_errand_that_discharges_it():
  """⚠ TWO TABLES, ONE TRUTH, AND THE DRIFT IS SILENT. `TaskKind.estimate_wh`
  is what `Task.claimable` and the errand gate both trust, and it is a
  separate number from hub/energy.json because it knows which target was
  asked for. Under-priced, it is a robot that takes on a job it cannot
  finish.

  This is not hypothetical: `count_plants` was 0.87 Wh against a census that
  measures 1.12, which the new `ENERGY ... hub/energy.json is low` narration
  line caught on a real unattended run. The dearest measured world is the
  bar, because the table is world-agnostic and the offer could be anywhere.
  """
  from pluggybot.hub.tasks import KINDS
  dearest: dict = {}
  for world in ("home", "room_hub"):
    for key, wh in en.load(world).errand_wh.items():
      # `draw` and `draw:whiteboard_b` are the same JOB priced for different
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
  for world in ("home", "room_hub"):
    book = lc.board_book(world)
    priced = set(en.load(world).errand_wh)
    for action in ov.ERRAND_ACTIONS:
      try:
        lc.errands_for(action, world, book)
      except ValueError:
        continue                      # this world cannot do it at all
      assert action in priced, f"{world} can do {action} and has no cost for it"


# ---- 3 & 4. the gate in the mission loop ------------------------------------


def life_with(world: str = "home", battery_wh: float = HOSTING_WH,
              errands=None) -> lc.HubLifecycle:
  """A lifecycle with no viewer and nothing driving. `_afford_next` touches
  no physics, so this stays a fast test."""
  model = model_of(world)
  cfg = lc.world_config(world)
  return lc.HubLifecycle(model, mujoco.MjData(model), realtime=False,
                         battery_wh=battery_wh, rack=cfg["rack"],
                         grid_bounds=cfg["grid_bounds"],
                         low_battery_wh=cfg["low_battery_wh"],
                         world=world, errands=errands or [])


def test_an_errand_that_will_not_fit_is_deferred_and_stays_queued():
  """THE ACCEPTANCE CRITERION, as arithmetic: an errand whose cost exceeds
  what the pack can spend is not started. Deferred, not dropped -- the job is
  still the job, and the answer is a charge."""
  life = life_with(errands=[carry_errand(use_at=(1.5, 1.8))])
  life.battery.energy_wh = 0.8            # carry 0.689 + 0.55 margin = 1.24
  assert life._afford_next() is False
  assert len(life.errands) == 1, "a deferred errand must survive the charge"
  assert any("DEFER" in line for line in life.log), life.log


def test_the_same_errand_runs_once_the_pack_can_pay_for_it():
  life = life_with(errands=[carry_errand(use_at=(1.5, 1.8))])
  life.battery.energy_wh = 4.0
  assert life._afford_next() is True
  assert len(life.errands) == 1


def test_an_errand_no_charge_could_cover_is_dropped_not_deferred():
  """⚠ THE SPIN. `beyond` deferred would be: charge, still short, charge
  again, forever -- a robot that never does anything, wearing a safety
  feature's clothes. It is dropped, said out loud, and the loop moves on."""
  life = life_with(battery_wh=HOSTING_WH,
                   errands=[Errand(name="huge", module="module_lcd",
                                   station_y=0.0, use_at=(1.5, 1.8),
                                   task="carry", estimate_wh=99.0)])
  assert life._afford_next() is True       # nothing left to gate
  assert life.errands == []
  assert any("SKIP" in line and "not possible in this world" in line
             for line in life.log), life.log


def test_a_demo_cell_still_attempts_a_job_bigger_than_itself():
  """⚠ THE OTHER WAY THIS GUARD COULD HAVE GONE WRONG, and it nearly did.
  home's census measures 1.14 Wh against a 0.99 Wh charged demo cell, so a
  `beyond` refusal there would have deleted the census from every home
  mission -- including the committed showcase recording, in which the robot
  completes the survey, stows the LCD, and only then runs flat. A guard that
  deletes a capability the fixture proves exists is not a guard."""
  life = life_with(battery_wh=HOME_DEMO_WH,
                   errands=[Errand(name="census:garden", module="module_lcd",
                                   station_y=0.0, use_at=(1.5, 1.8),
                                   task="census")])
  assert life.energy.cost("census") > life.charged_wh, \
      "the fixture only means anything while the census outgrows the cell"
  assert life._afford_next() is True
  assert len(life.errands) == 1
  assert any("smaller than this job" in line for line in life.log), life.log


def test_an_errand_deferred_too_often_is_given_up_on():
  """The other spin, and a different fault: the pack COULD hold this job and
  does not, which means charging is what is broken. Two goes, then the errand
  is dropped rather than blocking the queue behind it."""
  life = life_with(errands=[carry_errand(use_at=(1.5, 1.8))])
  life.battery.energy_wh = 0.8
  for _ in range(lc.MAX_ERRAND_DEFERRALS):
    assert life._afford_next() is False    # ...charge would go here
  assert life._afford_next() is True
  assert life.errands == []
  assert any("giving up on it" in line for line in life.log), life.log


def test_a_task_errand_is_priced_by_its_own_kind_not_by_the_action():
  """A task's estimate knows which end of the house it is being asked about;
  a per-action figure cannot. The dearer one has to win, or the far
  whiteboard is priced as the near one -- which is exactly the 0.968-against-
  0.93 death CLAUDE.md records."""
  life = life_with()
  errand = carry_errand(use_at=(1.5, 1.8))
  errand.estimate_wh = 3.0
  assert life.affords(errand).cost_wh == 3.0
  errand.estimate_wh = 0.0
  assert life.affords(errand).cost_wh == en.load("home").cost("carry")


def test_a_demo_cell_spends_its_whole_pack_exactly_as_it_always_did():
  """The compatibility claim from the loop's side: with a zero margin,
  `spendable_wh` is the pack, which is the number every existing mission and
  both committed recordings were produced against."""
  life = life_with(battery_wh=HOME_DEMO_WH)
  life.battery.energy_wh = 0.77
  assert life.reserve_margin_wh == 0.0
  assert life.spendable_wh == pytest.approx(0.77)
  assert life.fundable_wh == pytest.approx(HOME_DEMO_WH * lc.CHARGED)


def test_a_hosting_pack_keeps_the_return_trip_out_of_a_job_s_budget():
  life = life_with(battery_wh=HOSTING_WH)
  life.battery.energy_wh = 3.0
  assert life.reserve_margin_wh == HOME_RESERVE
  assert life.spendable_wh == pytest.approx(3.0 - HOME_RESERVE)


def test_a_charge_timeout_is_sized_against_the_pack_it_has_to_fill():
  """⚠ A TIMEOUT IN SECONDS IS A TIMEOUT IN WATT-HOURS. 400 s was right for a
  0.7 Wh cell and quietly ended the deployed 8 Wh one at about two thirds,
  narrating "CHARGE complete (65 %)" every cycle."""
  demo = life_with(battery_wh=HOME_DEMO_WH)
  host = life_with(battery_wh=HOSTING_WH)
  assert demo.charge_timeout == lc.CHARGE_TIMEOUT_MIN
  # Long enough to actually put a hosting pack's worth in at the measured
  # net rate, with the slack a real press needs.
  rate = en.load("home").charge_w
  assert host.charge_timeout >= host.charged_wh * 3600.0 / rate
  assert host.charge_timeout > demo.charge_timeout


# ---- 5. what the model is shown ---------------------------------------------


def test_the_costs_ride_the_cached_prefix_and_not_the_turn():
  """What an errand costs is a property of the world, so it belongs in the
  stable half. Putting it in the volatile turn would invalidate the prompt
  cache on every call for a number that never changes -- the classic silent
  invalidator docs/Overseer.md §6 is about."""
  book = lc.board_book("home")
  menu = ov.Menu.for_world("home", book)
  prompt = ov.system_prompt("be useful", menu, ov.default_table())
  text = prompt[0]["text"]
  assert "energyCostWh" in text
  assert '"draw"' in text and str(menu.costs_wh["draw"]) in text


def test_only_measured_costs_are_shown_to_the_model():
  """⚠ `cost()` answers for ANY name, because the gate has to price an
  unmeasured errand at something. Printing that fallback would tell the model
  that `idle` costs 0.97 Wh -- false, and exactly the confident wrong number
  the rest of this design keeps out of the prompt."""
  menu = ov.Menu.for_world("home", lc.board_book("home"))
  assert set(menu.costs_wh) <= set(en.load("home").errand_wh)
  for free in ("idle", "journal", "explore", "charge", "take_task"):
    assert free not in menu.costs_wh


def test_the_scripted_policy_never_rotates_onto_what_the_world_cannot_do():
  """The fallback is a real day's work, so it has to obey the same gate the
  loop does -- otherwise the API going down means the robot proposing an
  errand this world refuses, over and over, until the budget runs out."""
  menu = ov.Menu.for_world("home", lc.board_book("home"))
  state = {"decisions": 0, "mapDone": True,
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
  menu = ov.Menu.for_world("home", lc.board_book("home"))
  d = ov.scripted(menu, {"decisions": 0, "mapDone": True,
                         "affordableActions": [],
                         "possibleActions": ["draw", "carry"]}, "test")
  assert d.action == "draw"


def test_an_empty_possible_list_filters_nothing():
  """A caller that supplies none -- a unit test, an older context dict --
  must not be read as "this robot can do nothing"."""
  menu = ov.Menu.for_world("home", lc.board_book("home"))
  d = ov.scripted(menu, {"decisions": 0, "mapDone": True}, "test")
  assert d.action == "draw"


def test_the_context_carries_what_the_pack_can_actually_spend():
  """`wh` is what is in the pack; `spendableWh` is what a job may cost. They
  differ by the margin, and on a hosting pack that difference is the whole
  guard -- a model shown only `wh` would plan against energy it is not
  allowed to spend."""
  life = life_with(battery_wh=HOSTING_WH)
  life.battery.energy_wh = 3.0
  state = ov.context_for(life, None, affordable=["carry"],
                         possible=["carry", "census"])
  assert state["battery"]["wh"] == pytest.approx(3.0)
  assert state["battery"]["spendableWh"] == pytest.approx(3.0 - HOME_RESERVE)
  assert state["affordableActions"] == ["carry"]
  assert state["possibleActions"] == ["carry", "census"]


def test_the_prompt_still_never_carries_a_hidden_answer():
  """The energy block is new context, and new context is a new chance to
  leak. Same claim as `test_the_prompt_never_carries_a_hidden_answer`, made
  again against the half of the prompt this issue touched."""
  menu = ov.Menu.for_world("home", lc.board_book("home"))
  text = ov.system_prompt("be useful", menu, ov.default_table())[0]["text"]
  assert "truth" not in text.lower().split("energycostwh")[-1]


# ---- 3, on real physics ------------------------------------------------------

#: The SMALLEST pack that is in the margin regime on `home` (the dearest
#: errand, 1.14 Wh, plus the 0.55 Wh reserve must fit a charged pack), so
#: these cost one charge cycle instead of the six a real 8 Wh pack would take.
#: The regime is what is under test, not the capacity -- `--pack hosting` is
#: the same arithmetic with more room in it.
MARGIN_PACK_WH = 2.0


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

  def __init__(self, action: str, reason: str) -> None:
    self.payload = {"action": action, "reason": reason, "board": "",
                    "program": "", "zone": "", "note": "", "respond_to": "",
                    "outcome": "", "reply": "", "task": "", "answer": ""}
    self.calls: list[dict] = []
    self.messages = self

  def create(self, **kwargs):
    self.calls.append(kwargs)
    return self._Response(self.payload)


def home_lifecycle(**kw) -> lc.HubLifecycle:
  cfg = lc.world_config("home")
  model = model_of("home")
  return lc.HubLifecycle(model, mujoco.MjData(model), realtime=False,
                         rack=cfg["rack"], grid_bounds=cfg["grid_bounds"],
                         low_battery_wh=cfg["low_battery_wh"],
                         boards=lc.board_book("home"), world="home", **kw)


@pytest.mark.slow
def test_an_overseer_that_only_ever_picks_the_dearest_errand_never_dies():
  """⚠ THE ACCEPTANCE CRITERION, flown, and the adversarial version of it.

  An overseer that answers `census` to every question it is ever asked --
  home's most expensive errand, 1.14 Wh against a 1.8 Wh charged pack. The
  robot must notice, in advance, that the second one will not fit, go and
  charge, and then do it. Without the gate `needs_charge` is false at 0.58 Wh
  (the reserve is 0.55), the errand starts anyway, and the robot ends the
  survey with less energy than it takes to get back to the rack.

  Adversarial rather than cooperative on purpose: the guarantee issue #15
  needs is not "a sensible model plans well", it is "a model that plans badly
  cannot strand the robot", which is the same shape as
  `test_charge_priority_survives_an_overseer_that_never_charges`.

  Shown to fail without the fix: make `HubLifecycle._afford_next` return True
  unconditionally and this reports SWAP_PICK before any charge, then a robot
  at 0 %% with the LCD still on the fork.
  """
  boss = ov.Overseer(ov.Menu.for_world("home", lc.board_book("home")),
                     client=OneNote("census", "I like counting"))
  life = home_lifecycle(battery_wh=MARGIN_PACK_WH, overseer=boss, errands=[])
  assert life.reserve_margin_wh == HOME_RESERVE, "not in the margin regime"
  assert life.energy.cost("census") + HOME_RESERVE <= life.charged_wh

  states: list[str] = []
  life.mission.step_hooks.append(
    lambda: states.append(life.state)
    if life.state != (states[-1] if states else None) else None)
  low = []
  life.mission.step_hooks.append(
    lambda: low.append(life.battery.fraction))

  r = life.run(lc.world_config("home")["start"], max_sim_time=900.0,
               explore_budget=15.0)

  assert any("DEFER" in line for line in life.log), \
      f"the second census was never deferred: {life.log[-12:]}"
  assert r["charge_cycles"] >= 1, "the robot never charged"
  assert r["battery"] > 0.0 and min(low) > 0.0, \
      "the robot ran flat -- the gate did not stop the errand it could not pay for"
  assert len(r["errands"]) >= 2, f"only got {len(r['errands'])} errands"
  for done in r["errands"]:
    print(f"  {done['errand']}: {done['energyWh']} Wh against an estimate "
          f"of {done['estimateWh']}")
    assert done["stowed"], f"{done['errand']} was abandoned on the fork"
    # ⚠ THE ESTIMATE MAY BE EXCEEDED, BY LESS THAN THE MARGIN. That is the
    # invariant, and asserting the stricter "never exceeded" is asserting
    # something the design does not claim: an errand's cost depends on where
    # the robot started and on how much of the map it already had, and the
    # FIRST errand of a mission plans through unknown space. What the margin
    # is FOR is absorbing exactly that -- an overrun smaller than the return
    # trip cannot strand the robot, and one larger than it is a stale table,
    # which the loop narrates as `ENERGY ... hub/energy.json is low`.
    assert done["energyWh"] <= done["estimateWh"] + HOME_RESERVE, \
        f"{done['errand']} cost {done['energyWh']} against an estimate of " \
        f"{done['estimateWh']} -- more than the margin can absorb"
  # ...and it really was the model being gated, not the fallback covering.
  assert any(d["source"] == "llm" for d in r["decisions"])
  # ⚠ CHARGE PRIORITY IS UNTOUCHED. The gate adds a reason to charge; it must
  # never have added a way not to.
  assert states.index("GO_CHARGE") < states.index("SWAP_PICK") \
      or "GO_CHARGE" in states[states.index("SWAP_PICK"):], states


@pytest.mark.slow
def test_a_charge_completes_on_a_pack_the_old_flat_timeout_could_not_fill():
  """⚠ A TIMEOUT IN SECONDS IS A TIMEOUT IN WATT-HOURS, flown.

  A pack that takes longer than the old flat 400 s to refill must still reach
  `CHARGED`, rather than hitting the cap two thirds of the way up and
  narrating "CHARGE complete (65 %%)" -- which is what the deployed 8 Wh sim
  has been doing.

  Shown to fail without the fix: put `CHARGE_TIMEOUT` back in place of
  `self.charge_timeout` in `HubLifecycle.charge` and the run ends well below
  `CHARGED`.
  """
  life = home_lifecycle(battery_wh=6.0, errands=[])
  assert life.charge_timeout > lc.CHARGE_TIMEOUT, "pick a bigger pack"
  # ⚠ READ BEFORE THE UNDOCK. `charge()` backs off the rack when it is done,
  # and that 0.30 m of reversing is real travel off the pack -- reading
  # `fraction` after the call returns measures the drive home, not the
  # charge, and lands a couple of tenths of a percent under CHARGED.
  topped: list[float] = []
  life.say_hooks.append(
    lambda _t, msg: topped.append(life.battery.fraction)
    if msg.startswith("CHARGE complete") else None)
  life.max_sim_time = 3000.0
  life.blacklist, life.map_done = set(), False
  life.explore_deadline = 1e9
  life.mission.start_at(*lc.world_config("home")["start"])
  life.mission.start_discovery()
  life.mission._spin()
  try:
    life.explore(budget=30.0, mark_done=False)
    life.battery.energy_wh = 0.9          # a long way from full
    assert life.go_charge(), "never reached the rack"
    life.charge()
  finally:
    life.mission.close()
  assert topped and topped[0] >= lc.CHARGED, \
      f"the charge stopped at {(topped or [0])[0]:.1%}"
  assert life.charge_cycles == 1
