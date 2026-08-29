"""Points as metabolism: hunger, satisfaction and a cap (issue #36).

The acceptance list, one section each:

  the clock       consumption ticks on SIM time, persists across a restart,
                  and never double-charges
  the cap         earnings past the ceiling are refused OUT LOUD
  the rhythm      work -> satisfied -> free time -> hungry again, over a
                  synthetic day (a real one is an hour of wall clock)
  zero            a starving robot charges, navigates and stows, on real
                  physics -- the criterion that is enforced by ABSENCE, and
                  so is the one that needs a whole mission to check
  the data        rate and cap are re-tunable without a code change
  the wire        the state reaches the site and the model
"""

import json

import mujoco
import pytest

from pluggybot.economy.ledger import Ledger
from pluggybot.economy.metabolism import (Appetite, Metabolism, TICK_S,
                                          METABOLISM_PATH)
from pluggybot.economy.scoring import evaluate
from pluggybot.telemetry.protocol import HUNGER_STATES, PROTOCOL_VERSION


#: A carry that actually happened, in the evaluator's own metric names.
CARRIED = {"picked": True, "stowed": True, "module": "module_lcd"}

#: The smallest world with a body the telemetry census keys on.
MINI_XML = """
<mujoco><worldbody>
  <body name="pluggybot"><freejoint/><geom size="0.1"/></body>
</worldbody></mujoco>
"""


def mini():
  model = mujoco.MjModel.from_xml_string(MINI_XML)
  return model, mujoco.MjData(model)


def appetite(**kw) -> Appetite:
  """A small, fast appetite: 360 points/hour is a point every 10 s, so a
  test's day fits in minutes of sim time rather than hours."""
  spec = {"points_per_hour": 360.0, "cap": 20, "satisfied_at": 12,
          "hungry_at": 5}
  spec.update(kw)
  return Appetite(**spec)


def fed(ledger, points: int, task: str = "carry") -> None:
  """Bank `points` by the only door there is -- a real verdict off a real
  evaluator, because `award` refuses anything else."""
  while points > 0:
    v = evaluate(task, CARRIED, table=ledger.table)
    assert v.points > 0
    ledger.award(v)
    points -= v.points


# ---- the clock ---------------------------------------------------------------


def test_hunger_is_charged_on_sim_time_not_wall_time():
  led = Ledger(cap=20)
  fed(led, 20)
  m = Metabolism(led, appetite())
  m.tick(0.0)                       # anchors, charges nothing
  assert m.points == 20
  # 100 sim seconds at 360/hour is 10 points, however long that took to run.
  m.tick(100.0)
  assert led.consumed() == 10
  assert m.points == 10


def test_a_tick_finer_than_the_interval_charges_nothing_and_loses_nothing():
  """The seam runs at ~500 Hz. Ticks under TICK_S must be free AND must not
  drop the time they saw -- an interval check that reset the anchor would
  quietly make a fast-stepping world immortal."""
  led = Ledger(cap=20)
  fed(led, 20)
  m = Metabolism(led, appetite())
  m.tick(0.0)
  for i in range(1, 1000):
    m.tick(i * TICK_S / 100.0)      # 100 sub-interval ticks per second
  assert led.consumed() == 0, "a sub-interval tick must cost nothing"
  m.tick(100.0)
  # Every one of those seconds is still charged for.
  assert led.consumed() == 10


def test_a_restart_neither_wipes_hunger_nor_back_charges_an_hour_of_it(tmp_path):
  """The issue's own wording, and the two failures are opposite signs.

  Sim time begins again at 0 on every mission, so the anchor is re-taken and
  the gap costs nothing; what survives is the BALANCE, in the ledger's file.
  """
  state = tmp_path / "ledger.json"
  led = Ledger(path=state, cap=20)
  fed(led, 20)
  m = Metabolism(led, appetite())
  m.tick(0.0)
  m.tick(90.0)                      # 9 points eaten over a mission
  assert m.points == 11

  # ...the container cycles. New process, new ledger off the same file, and
  # sim time restarts at zero.
  led2 = Ledger(path=state, cap=20)
  assert led2.balance() == 11, "hunger did not survive the restart"
  m2 = Metabolism(led2, appetite())
  m2.tick(0.0)
  assert m2.points == 11, "the restart back-charged the gap"
  # ...and the new mission's own seconds are charged normally.
  m2.tick(30.0)
  assert m2.points == 8


def test_sim_time_going_backwards_re_anchors_rather_than_crediting():
  """The other half of the restart rule, and the one a fresh `Metabolism`
  cannot reach: an appetite that OUTLIVES a sim reset sees time run
  backwards. Charging a negative interval would hand the robot a refund of
  everything it had eaten -- and the sign is the whole bug, so nothing about
  the balance looks wrong until it is enormous."""
  led = Ledger(cap=20)
  fed(led, 20)
  m = Metabolism(led, appetite())
  m.tick(0.0)
  m.tick(100.0)
  assert m.points == 10
  m.tick(0.0)                       # a new mission on the same object
  assert m.points == 10, "time going backwards paid the robot back"
  m.tick(20.0)                      # ...and the new run's seconds cost normally
  assert m.points == 8


def test_the_fraction_of_a_point_survives_a_restart(tmp_path):
  """Without the carry, a mission shorter than one point's worth of time
  costs nothing -- and a robot restarted every few minutes would never get
  hungry at all. The strongest form of the double-charge bug's mirror."""
  state = tmp_path / "ledger.json"
  led = Ledger(path=state, cap=20)
  fed(led, 20)
  # 5 s at 360/hour is half a point: nothing is eaten, but half is owed.
  m = Metabolism(led, appetite())
  m.tick(0.0)
  m.tick(5.0)
  assert led.consumed() == 0
  assert led.owed() == pytest.approx(0.5)
  # The carry is written by the next save from any source, not by a file
  # write a second -- so an award mid-mission is what puts it on disk.
  fed(led, 2)

  led2 = Ledger(path=state, cap=20)
  m2 = Metabolism(led2, appetite())
  m2.tick(0.0)
  # Another five seconds completes the point the last run started.
  m2.tick(5.0)
  assert led2.consumed() == 1


def test_hunger_stops_at_zero_and_never_goes_into_debt():
  """"Zero is narrative, never a capability lock" (issue #36), and arrears
  are the sneaky version of the lock: a robot that owed an hour of appetite
  would see its first job back pay nothing."""
  led = Ledger(cap=20)
  fed(led, 4)
  m = Metabolism(led, appetite())
  m.tick(0.0)
  m.tick(3600.0)                    # a whole hour at 360/hour
  assert m.points == 0
  assert m.state == "starving"
  fed(led, 4)
  assert m.points == 4, "a starving robot's next job must pay in full"


# ---- the cap -----------------------------------------------------------------


def test_earnings_over_the_cap_are_refused_out_loud():
  led = Ledger(cap=6)
  v = evaluate("carry", CARRIED, table=led.table)
  led.award(v)                      # carry pays 2
  led.award(v)
  led.award(v)                      # ...and the balance is now full at 6
  entry = led.award(v)
  assert led.balance() == 6
  # The reward table's answer is untouched: what the job paid and what fit
  # are different numbers, and both are on the entry.
  assert entry["points"] == 2
  assert entry["banked"] == 0 and entry["spilled"] == 2
  assert led.spilled() == 2, "the cap refused points and said nothing"
  # ...and the balance still reconciles.
  acct = led.robots["pluggybot"]
  assert acct["earned"] - acct["consumed"] - acct["spent"] == led.balance()


def test_a_partial_award_banks_what_fits():
  led = Ledger(cap=5)
  v = evaluate("carry", CARRIED, table=led.table)
  led.award(v)
  led.award(v)                      # 4 of 5
  entry = led.award(v)              # 2 offered, 1 fits
  assert entry["points"] == 2 and entry["banked"] == 1 and entry["spilled"] == 1
  assert led.balance() == 5


def test_no_cap_means_no_cap_and_no_new_fields():
  """Every mission, demo and recording before this issue. The absent fields
  are the claim: an entry written without an appetite is byte-identical to
  the ones written before the cap existed."""
  led = Ledger()
  v = evaluate("carry", CARRIED, table=led.table)
  entry = None
  for _ in range(50):
    entry = led.award(v)
  assert led.balance() == 100
  assert "banked" not in entry and "spilled" not in entry


def test_a_rating_settles_through_the_cap():
  """A deferred payout is not a way around a ceiling (issue #16 meets #36)."""
  led = Ledger(cap=4)
  v = evaluate("artwork", {"board": "whiteboard_a", "strokes": 6,
                           "strokesInked": 6, "formMm": 0.6,
                           "inkedFraction": 0.99, "travelInkFraction": 0.0,
                           "fill": 0.3}, table=led.table)
  assert v.pending
  entry = led.award(v)
  fed(led, 4)                       # fill the balance while the rating flies
  settled = led.settle(entry["seq"], 1.0)
  assert led.balance() == 4
  assert settled["points"] > 0 and settled["banked"] == 0
  assert settled["spilled"] == settled["points"]


# ---- the rhythm --------------------------------------------------------------


def test_a_day_is_work_then_satisfaction_then_free_time_then_hunger_again():
  """The mechanic's whole point, over a synthetic day.

  A real one is an hour of wall clock (`--metabolism` on a hosting pack);
  what is under test here is the STATE MACHINE, which is where the rhythm
  lives -- the physics is the same physics either way.
  """
  led = Ledger(cap=20)
  m = Metabolism(led, appetite())
  m.tick(0.0)
  assert m.state == "starving", "a fresh robot has not earned anything yet"

  # It works. Six points in, it is fed but not yet done.
  fed(led, 6)
  m.tick(TICK_S)
  assert m.state == "fed" and not m.satisfied

  # ...and at twelve it has had enough and stops.
  fed(led, 6)
  m.tick(2 * TICK_S)
  assert m.state == "satisfied" and m.satisfied

  # The free time: it eats down THROUGH the satisfied line and stays
  # satisfied, which is the hysteresis -- a robot on the line must not flip
  # state every point.
  m.tick(72.0)                      # 70 s of appetite: seven points gone
  assert 5 <= m.points < 12
  assert m.state == "satisfied", "the latch dropped at the wrong threshold"

  # ...until it crosses the HUNGRY line, which is lower, and goes back to work.
  m.tick(102.0)
  assert m.points < 5
  assert m.state == "hungry" and not m.satisfied


def test_the_state_does_not_flap_around_a_single_threshold():
  """Without hysteresis a balance sitting on the line changes state on every
  point eaten and every point earned -- forty transitions an hour, and forty
  lines in History.md saying nothing."""
  led = Ledger(cap=20)
  fed(led, 12)
  m = Metabolism(led, appetite())
  m.tick(0.0)
  assert m.state == "satisfied" and m.changed() == ""
  flips = 0
  t = 0.0
  for _ in range(20):
    # Bounce across the satisfied line: eat 10 s (1 point), earn one back.
    t += 10.0
    m.tick(t)
    fed(led, 2)
    if m.changed():
      flips += 1
  assert flips == 0, "the satisfied latch flapped inside its own band"


def test_a_transition_is_narrated_once_not_a_level_every_tick():
  led = Ledger(cap=20)
  m = Metabolism(led, appetite())
  fed(led, 12)
  m.tick(0.0)
  assert m.changed() == "satisfied"
  for i in range(1, 30):
    m.tick(i * TICK_S)
    assert m.changed() == "", "a level was reported as an event"


def test_a_restarted_robot_in_the_ambiguous_band_goes_back_to_work():
  """The latch is not persisted, and is seeded on the HUNGRY side: a robot
  that came back still coasting on a satisfaction it could no longer justify
  would idle through the first stretch of every mission, and there is no way
  to tell the two cases apart from the balance alone."""
  led = Ledger(cap=20)
  fed(led, 8)                       # between hungry_at (5) and satisfied_at (12)
  m = Metabolism(led, appetite())
  assert not m.satisfied
  assert m.state == "fed"


# ---- the data ----------------------------------------------------------------


def test_rate_and_cap_are_data_and_re_tunable_without_a_code_change(tmp_path):
  path = tmp_path / "metabolism.json"
  path.write_text(json.dumps({
    "version": 1,
    "default": {"pointsPerHour": 45.0, "cap": 90, "satisfiedAt": 45,
                "hungryAt": 20},
    "worlds": {"home": {"pointsPerHour": 12.0, "cap": 60}},
  }))
  base = Appetite.load("room_hub", path)
  assert base.points_per_hour == 45.0 and base.cap == 90
  # A world block overrides key by key, like cadence.json -- the two
  # thresholds here are not restated and come from the default block.
  home = Appetite.load("home", path)
  assert home.points_per_hour == 12.0 and home.cap == 60
  assert home.satisfied_at == 45 and home.hungry_at == 20


#: What a `home` robot actually BANKS in a sim-hour, measured over two
#: unattended runs with no overseer (issue #36's calibration comment). The
#: hosting figure is the one that matters: on the demo cell every point came
#: from CHARGING and no job was completed at all, so a rate calibrated there
#: would make charging the food -- see economy/metabolism.json's note.
MEASURED_POINTS_PER_HOUR = 102.0


def test_the_shipped_file_leaves_the_robot_half_its_day():
  """The tuning claim, against MEASURED throughput rather than a projection.

  This is the guard on a future re-tune, not on today's value: the mechanic
  is the free time, so an appetite that eats the whole income deletes it
  while still looking like it works.
  """
  a = Appetite.load("home")
  duty = a.points_per_hour / MEASURED_POINTS_PER_HOUR
  assert 0.25 <= duty <= 0.60, (
    f"an appetite of {a.points_per_hour}/h against a measured "
    f"{MEASURED_POINTS_PER_HOUR}/h income is {duty:.0%} of the robot's "
    "capacity -- issue #36 asks for roughly half, and at the top of that "
    "range there is no time for anything but earning")
  assert a.satisfied_at <= a.cap and a.hungry_at < a.satisfied_at
  # The rhythm the thresholds buy: free time falling from satisfiedAt back to
  # hungryAt, and the climb back up at whatever is left of the income.
  #
  # ⚠ IDEALISED, and the real cycle is LONGER -- measured, a cold robot took
  # 44 min to reach `satisfied` against the ~26 this arithmetic predicts,
  # because real income is lumpy (a 640 s charge, failed jobs, exploring all
  # land inside it). So this is a sanity bound on the thresholds against the
  # rate, NOT a promise that one mission shows a whole cycle -- it does not,
  # and that is why hunger persists in the ledger instead.
  free_h = (a.satisfied_at - a.hungry_at) / a.points_per_hour
  work_h = (a.satisfied_at - a.hungry_at) / (MEASURED_POINTS_PER_HOUR
                                             - a.points_per_hour)
  assert free_h >= 0.25, f"only {free_h * 60:.0f} min of free time a cycle"
  assert free_h + work_h <= 1.5, (
    f"an idealised cycle of {(free_h + work_h) * 60:.0f} min is already "
    "longer than a mission before real income's lumpiness is added")


@pytest.mark.parametrize("bad", [
  {"hungry_at": 50, "satisfied_at": 40},      # inverted hysteresis
  {"satisfied_at": 200},                      # can never be satisfied
  {"hungry_at": 0},                           # starving and hungry collapse
  {"cap": 0},
  {"points_per_hour": -1.0},
])
def test_an_incoherent_appetite_is_refused_at_load(bad):
  """A mounted file nobody validated fails INVISIBLY: every ordering below
  runs without raising anything downstream and just quietly deletes the
  mechanic."""
  with pytest.raises(ValueError):
    appetite(**bad)


def test_a_future_version_of_the_file_is_refused(tmp_path):
  path = tmp_path / "metabolism.json"
  path.write_text(json.dumps({"version": 99, "default": {}}))
  with pytest.raises(ValueError, match="metabolism version"):
    Appetite.load("home", path)


def test_the_committed_file_is_the_shipped_default():
  assert METABOLISM_PATH.exists()
  doc = json.loads(METABOLISM_PATH.read_text())
  assert doc["version"] == 1
  # The note is the tuning brief -- it carries the throughput arithmetic the
  # numbers have to be read against, and a file without it is a file whose
  # next editor re-derives it or guesses.
  assert any("job" in line for line in doc["note"])


# ---- the wire ----------------------------------------------------------------


def test_the_snapshot_is_the_vocabulary_and_says_where_it_sits():
  led = Ledger(cap=20)
  m = Metabolism(led, appetite())
  fed(led, 12)
  m.tick(0.0)
  snap = m.snapshot()
  assert snap["state"] in HUNGER_STATES
  assert snap["satisfied"] is True
  assert snap == {"state": "satisfied", "satisfied": True, "points": 12,
                  "cap": 20, "pointsPerHour": 360.0, "satisfiedAt": 12,
                  "hungryAt": 5, "consumed": 0, "spilled": 0}


def test_the_ledger_block_explains_its_own_balance():
  """earned - consumed - spent == balance has to be checkable from the wire
  alone: a site that showed the balance falling with no consumed figure
  beside it would be showing points leaking."""
  led = Ledger(cap=20)
  fed(led, 12)
  m = Metabolism(led, appetite())
  m.tick(0.0)
  m.tick(50.0)
  block = led.snapshot()["pluggybot"]
  assert block["earned"] - block["consumed"] - block["spent"] == block["balance"]
  assert block["consumed"] == 5


def test_the_frame_carries_the_block_and_the_header_the_vocabulary(tmp_path):
  from pluggybot.telemetry.recorder import FrameBuilder
  model, data = mini()
  led = Ledger(cap=20)
  m = Metabolism(led, appetite())
  builder = FrameBuilder(model, data, metabolism=m)
  header = builder.header()
  assert header["protocolVersion"] == PROTOCOL_VERSION
  assert header["hungerStates"] == list(HUNGER_STATES)
  first = builder.build()
  assert first["metabolism"]["state"] == "starving"
  # Whole-block on change, like `spend`: an unchanged appetite costs no bytes.
  data.time = 0.05
  assert "metabolism" not in builder.build()
  fed(led, 12)
  m.tick(1.0)
  data.time = 1.0
  assert builder.build()["metabolism"]["state"] == "satisfied"


def test_a_world_with_no_appetite_advertises_none(tmp_path):
  """`hungerStates` empty is the honest answer, on `taskKinds`' terms -- and
  the block is absent rather than zeroed, so a site can tell "not hungry" from
  "does not get hungry"."""
  from pluggybot.telemetry.recorder import FrameBuilder
  builder = FrameBuilder(*mini())
  assert builder.header()["hungerStates"] == []
  assert "metabolism" not in builder.build()


# ---- what the model is told --------------------------------------------------


def test_the_model_is_shown_its_hunger_and_has_no_verb_for_it():
  """The reward table's rule, one layer out (issue #36): shown, unreachable.
  A decision has no field that moves a point in either direction."""
  from pluggybot.mind.overseer import Decision
  fields = set(Decision("idle").as_dict())
  assert not fields & {"points", "eat", "cap", "satisfied", "metabolism",
                       "consume"}


def test_the_appetite_rules_ride_the_cached_prefix_and_only_when_there_is_one():
  """The prompt-cache split, on ESCALATION_RULE's terms: the RULES are a
  property of the world and go in the prefix, the numbers change per call and
  ride the user turn. A world with no appetite keeps the exact prefix it had
  before this existed."""
  from pluggybot.mind.overseer import Menu, system_prompt
  from pluggybot.economy.scoring import default_table
  from pluggybot.mind.thoughts import ThoughtFiles
  th = ThoughtFiles.open(None)
  menu, table = Menu.for_world("home"), default_table()
  off = system_prompt(th, menu, table, name="Pluggy")[0]["text"]
  on = system_prompt(th, menu, table, name="Pluggy", appetite=True)[0]["text"]
  assert "POINTS ARE FOOD" not in off
  assert "POINTS ARE FOOD" in on
  # The prefix is built once and sent verbatim, so nothing in it may vary
  # per call -- no balance, no state, no rate.
  for number in ("45", "90", "satisfied\":", "hungryAt"):
    assert number not in on.split("POINTS ARE FOOD")[1].split("\n\n")[0]


def test_the_context_carries_the_appetite_and_omits_it_when_there_is_none():
  from pluggybot.mind import overseer as ov

  class FakeLife:
    class data:
      time = 12.0
    class battery:
      fraction = 0.5
      energy_wh = 0.4
      empty = False
    low_battery_wh = 0.1
    spendable_wh = 0.3
    charging_now = False
    boards = None
    ledger = None
    verdicts: list = []
    map_done = True

  state = ov.context_for(FakeLife(), metabolism={"state": "hungry",
                                                 "points": 3})
  assert state["metabolism"] == {"state": "hungry", "points": 3}
  # Absent, NOT zeroed: "there is no hunger here" and "you are not hungry
  # right now" must not read the same to a model.
  assert "metabolism" not in ov.context_for(FakeLife())


# ---- zero is narrative, never a capability lock -------------------------------


def test_a_starving_robot_still_looks_worried_and_nothing_else():
  """The whole of what running out does to the face -- and it must not stomp
  the states that are about something more urgent than food."""
  from pluggybot.tools.screen import face_for
  assert face_for("EXPLORE", 1.0) == face_for("EXPLORE", 1.0,
                                              hunger="starving")
  assert face_for("CHARGE", 1.0) == face_for("CHARGE", 1.0, hunger="starving")
  assert face_for("GO_CHARGE", 0.1) == face_for("GO_CHARGE", 0.1,
                                                hunger="starving")
  assert face_for("IDLE", 1.0) == ("idle", "blink")
  assert face_for("IDLE", 1.0, hunger="starving") == ("worried", "shake")
  assert face_for("IDLE", 1.0, hunger="hungry") == ("idle", "blink")


def test_nothing_in_the_mission_loop_reads_a_balance():
  """The criterion is enforced by ABSENCE, so this reads the source.

  A grep is a poor test in general and the right one here: the claim is that
  no branch exists, and the mission test below cannot prove a branch is
  missing -- it can only fail to take one.
  """
  import inspect

  from pluggybot import lifecycle
  src = inspect.getsource(lifecycle.HubLifecycle)
  for method in ("needs_charge", "go_charge", "charge", "run_errand",
                 "_afford_next", "explore"):
    attr = getattr(lifecycle.HubLifecycle, method)
    body = inspect.getsource(attr.fget if isinstance(attr, property) else attr)
    for banned in ("metabolism", "balance()", "starving", "satisfied"):
      assert banned not in body, (
        f"HubLifecycle.{method} reads {banned!r} -- survival must not depend "
        "on points (issue #36)")
  # ...and the ONLY methods that touch the appetite at all are the four that
  # report it. An allowlist rather than a count, so a new reader has to be
  # named here and looked at -- which is the review this test exists to
  # force, since the whole criterion is that no branch was added.
  readers = {name for name, attr in vars(lifecycle.HubLifecycle).items()
             if callable(attr)
             and "self.metabolism" in inspect.getsource(attr)}
  assert readers == {
    "__init__",           # holds it
    "_metabolism_step",   # ticks it and narrates a transition
    "_screen_step",       # a starving robot looks worried and nothing else
    "run",                # reports it in the mission summary it returns
  }, f"the appetite reached {readers} -- is one of those a capability gate?"
  assert "metabolism" in src


@pytest.mark.slow
def test_a_starving_robot_still_charges_navigates_and_stows(tmp_path,
                                                            monkeypatch):
  """Issue #36's third acceptance criterion, on real physics.

  It has to be a WHOLE MISSION. The criterion is that no branch reads the
  balance, and the grep above is what proves the branch is absent -- but a
  grep cannot see a gate that arrived through a helper, an errand, or the
  overseer's own filtering. What it takes to be sure is a robot that is
  actually broke for the whole run doing actually everything: driving to a
  bay, picking a module up, carrying it, hanging it back on its bracket, and
  then finding the dock and charging.

  Starved by CONFIGURATION rather than by a stub, which also exercises the
  `$PLUGGY_METABOLISM` door end to end: an appetite of a point a sim-second
  eats every job's payout inside twenty seconds, so the balance is on the
  floor from the first minute to the last.
  """
  from pluggybot.lifecycle import run_demo

  data_file = tmp_path / "metabolism.json"
  data_file.write_text(json.dumps({
    "version": 1,
    "default": {"pointsPerHour": 3600.0, "cap": 90, "satisfiedAt": 45,
                "hungryAt": 20},
  }))
  monkeypatch.setenv("PLUGGY_METABOLISM", str(data_file))

  # Stop on the CLAIM, not the budget (issue #54): the claim is settled once
  # the tool has gone out and come back and the robot has charged. A run that
  # fails never satisfies it and goes the whole distance, which is what keeps
  # a shortened test able to catch the regression it exists for.
  #
  # The predicate also SAMPLES, on the physics seam, which is the only place
  # the balance can be read while the robot is mid-manoeuvre: the balance at
  # the end of the run is the balance after the charge PAID, and says nothing
  # about what the robot had while it was driving.
  seen: list[tuple[str, int]] = []
  hungers: set[str] = set()

  def settled(life):
    seen.append((life.state, life.metabolism.points))
    hungers.add(life.metabolism.state)
    return (life.swaps_done >= 2 and life.charge_cycles >= 1
            and life.mission.swap.module_state(life.module)["hung"])

  r = run_demo(world="room_hub", ledger_state=str(tmp_path / "ledger.json"),
               stop_when=settled)
  # THE PREMISE, not the claim: the robot really was broke while it worked.
  assert r["metabolism"]["consumed"] > 0, "the appetite never ticked"
  assert "starving" in hungers, "never actually starved; this proves nothing"
  # It never accumulated anything -- each job's payout was eaten inside
  # twenty sim-seconds of landing.
  assert max(p for _, p in seen) <= 5, "not actually broke; this proves nothing"
  # ⚠ MINIMUM, not the balance on ARRIVAL. A job's payout lands part-way
  # through the phase that earned it and buys the robot a few seconds of
  # solvency, so "broke for the whole phase" and "broke when it started" are
  # both claims about the payout's TIMING. What the criterion is actually
  # about is whether being broke stopped the robot doing the phase -- so:
  # it was at zero points during every phase of the mission, and did the
  # phase anyway.
  for phase in ("SWAP_PICK", "SWAP_RETURN", "GO_CHARGE", "CHARGE"):
    points = [p for state, p in seen if state == phase]
    assert points, f"the mission never reached {phase}"
    assert min(points) == 0, f"never actually broke during {phase}"
  # ...and it did every single thing anyway.
  assert r["rack_discovered"], "a broke robot could not find the rack"
  assert r["swaps_done"] == 2, "a broke robot could not run the errand"
  assert r["module_stowed"], "a broke robot could not stow the tool"
  assert r["charge_cycles"] >= 1, "a broke robot could not charge -- this is "\
                                  "the failure the mechanic must never have"
  assert r["collision_steps"] == 0
  # It was still PAID for the work; the points were simply eaten again.
  assert r["earned"] > 0 and r["verdicts"]
  assert r["metabolism"]["state"] in HUNGER_STATES
  assert all(v["ok"] for v in r["verdicts"]), r["verdicts"]
