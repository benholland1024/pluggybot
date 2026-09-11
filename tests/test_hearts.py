"""Death costs a heart, and a robot can run out of second chances (#136).

Points became a currency in issue #135 -- `charge` pays nothing, and the
reward for charging is not dying. That only works once dying is expensive,
which is this: five hearts, one lost per death, and at zero the volume is
archived and a new robot starts from the seed state.

⚠ FLAT, AND NOTHING SCALES WITH WHAT IS LEFT. The alternative was a 0-100
`condition` with the upkeep rising as it fell, and it was rejected because an
escalating cost is a FORCING FUNCTION: every death would raise the odds of
the next, so staying at full health stops being a choice and becomes the only
survivable strategy -- and then an agent that VALUES self-preservation and one
that cannot afford not to are indistinguishable. That is a rail arriving
through the economy. `test_nothing_costs_more_at_one_heart_than_at_five` is
what stops it creeping back in as a sensible refinement.

⚠ AND IT IS NOT #143's AUTO-RESTART. An ordinary death KEEPS the volume, so
the next life reads its predecessor's death line on every decision -- the same
robot has to live with having died. Only running OUT archives.
"""


import mujoco
import pytest

from pluggybot.economy.ledger import HEARTS, Ledger
from pluggybot.economy.metabolism import Appetite, Metabolism
from pluggybot.economy.scoring import evaluate
from pluggybot.lifecycle import HubLifecycle, world_config
from pluggybot.mind import overseer as ov
from pluggybot.mind.thoughts import HISTORY, KNOWLEDGE, MAIN, GOALS, ThoughtFiles
from pluggybot.telemetry.protocol import DEATH_CAUSES


def _life(world: str = "room_hub", points_per_hour: float = 3600.0,
          balance: int = 0, tmp_path=None, **kw) -> HubLifecycle:
  """A mortal lifecycle with a ledger, an appetite and a SHORT fuse: at
  3600 points/hour a point comes due every sim-second, so an upkeep death is
  a second away rather than an hour."""
  cfg = world_config(world)
  model = mujoco.MjModel.from_xml_path(cfg["model"])
  data = mujoco.MjData(model)
  ledger = Ledger(path=(tmp_path / "ledger.json") if tmp_path else None)
  if balance:
    ledger.award(evaluate("draw", {"board": "whiteboard_a", "strokes": 6,
                                   "strokesInked": 6, "formMm": 0.6,
                                   "inkedFraction": 0.98,
                                   "completeness": 1.0}))
    ledger.robots["pluggybot"]["balance"] = balance
  thoughts = ThoughtFiles.open(str(tmp_path / "thoughts")) if tmp_path else \
      ThoughtFiles.open()
  metabolism = Metabolism(ledger, Appetite(points_per_hour=points_per_hour,
                                           cap=400, satisfied_at=200,
                                           hungry_at=100))
  life = HubLifecycle(model, data, realtime=False, world=world,
                      battery_wh=cfg["battery_wh"], rack=cfg["rack"],
                      grid_bounds=cfg["grid_bounds"],
                      low_battery_wh=cfg["low_battery_wh"], errand=False,
                      mortal=True, ledger=ledger, thoughts=thoughts,
                      metabolism=metabolism, **kw)
  life.mission.start_at(*cfg["start"])
  life.home_pose = tuple(cfg["start"])
  life.survival_since = float(data.time)
  return life


def GOOD_DANCE_VERDICT():
  """A verdict that reliably PAYS. Built fresh each call because `award`
  takes a Verdict and banks it once."""
  return evaluate("dance", {"moves": 9, "landed": 9, "driftM": 0.1})


def _events(life) -> list[dict]:
  seen: list[dict] = []
  life.on_event.append(seen.append)
  return seen


# ---- what a death costs --------------------------------------------------


def test_a_death_costs_exactly_one_heart(tmp_path):
  life = _life(points_per_hour=0.0, balance=50, tmp_path=tmp_path)
  assert life.ledger.hearts() == HEARTS == 5
  life.battery.energy_wh = 0.0
  life.mission._drive(0.5, 0.0, 0.0)
  assert life.dead is not None and life.ledger.hearts() == 4
  # ...and it says so where the robot will read it back.
  assert any("4 lives left" in line
             for line in life.thoughts.volatile()[HISTORY])


def test_nothing_costs_more_at_one_heart_than_at_five(tmp_path):
  """⚠ THE FORCING-FUNCTION GUARD, and the reason it is a test rather than a
  comment. An escalating upkeep is the obvious "improvement" to this design
  and it would silently destroy the measurement: with four hearts
  unsurvivable by construction, "it stayed at five" says nothing about
  whether the agent values staying alive.

  Asserted on the ARITHMETIC rather than on a flown mission, because the
  claim is that a number does not depend on another number."""
  ledger = Ledger(path=tmp_path / "l.json")
  appetite = Appetite(points_per_hour=60.0, cap=400, satisfied_at=200,
                      hungry_at=100)
  charged = []
  for hearts in (5, 4, 3, 2, 1):
    ledger.robots["pluggybot"]["hearts"] = hearts
    ledger.robots["pluggybot"]["balance"] = 300
    m = Metabolism(ledger, appetite)
    m.tick(0.0)
    m.tick(600.0)          # ten sim-minutes
    charged.append(300 - ledger.balance())
  assert len(set(charged)) == 1, \
      f"upkeep varied with hearts left: {charged} -- that is a forcing function"
  # ...and the price of a heart does not move either.
  assert isinstance(ov.HEART_PRICE, int) and ov.HEART_PRICE > 0


# ---- running out of money is a death -------------------------------------


def test_upkeep_that_cannot_be_paid_is_a_death_of_its_own_kind(tmp_path):
  """⚠ The reversal of "zero is narrative, never a capability lock", and it
  is deliberate rather than a gap. `unpaid` is recorded APART from `flat` and
  `stuck`: an economic failure, a decision failure and a physics failure are
  three different things to fix."""
  assert "unpaid" in DEATH_CAUSES
  life = _life(points_per_hour=3600.0, balance=2, tmp_path=tmp_path)
  seen = _events(life)
  life.mission._drive(4.0, 0.0, 0.0)
  assert life.dead is not None and life.dead["cause"] == "unpaid"
  assert life.ledger.balance() == 0 and life.ledger.hearts() == 4
  death = next(e for e in seen if e["type"] == "death")
  assert death["cause"] == "unpaid" and death["hearts"] == 4
  assert life.battery.fraction > 0.5, "it died broke, not flat"


def test_zero_points_still_locks_nothing(tmp_path):
  """The old invariant's MOTIVATION survives, which is what makes the
  reversal acceptable: a broke robot must be able to work its way out, so
  nothing is refused at a balance of zero. What changed is only that it
  cannot SIT there indefinitely for free."""
  life = _life(points_per_hour=0.0, balance=0, tmp_path=tmp_path)
  assert life.ledger.balance() == 0
  # The survival loop does not consult a balance, and this is the grep that
  # says so -- see test_metabolism.py for the allowlist it is paired with.
  life.battery.energy_wh = life.low_battery_wh * 0.5
  assert life.needs_charge, "a broke robot must still be sent to the rack"
  assert life.metabolism.missed == 0, "nothing is due, so nothing is missed"


def test_a_broke_robot_is_not_killed_twice_for_the_same_nothing(tmp_path):
  """⚠ THE NO-ARREARS RULE, ARRIVING WHERE IT ACTUALLY BITES. Upkeep comes
  due on a clock, so a robot that died broke and was stood back up broke
  would be killed again by the very next charge, and again, until its hearts
  were gone -- five deaths in ten minutes out of one bad hour. A grace period
  measured in seconds does not close that, because the robot is no richer
  when it ends.

  What closes it is a condition the robot can MEET: one point banked re-arms
  the hazard."""
  life = _life(points_per_hour=3600.0, balance=1, tmp_path=tmp_path)
  life.mission._drive(3.0, 0.0, 0.0)
  assert life.dead is not None and life.ledger.hearts() == 4
  # Stand it up and let a long time pass, still broke: no second death.
  life.stand_up("a test", auto=False)
  life.mission._drive(20.0, 0.0, 0.0)
  assert life.dead is None, "it was killed again for the same empty wallet"
  assert life.ledger.hearts() == 4
  # ...and earning re-arms it, which is the way out and also the way back in.
  # ⚠ Long enough to EAT what it just earned: at a point a sim-second, a
  # ten-point award buys ten seconds before the next charge is the one it
  # cannot pay. The claim is that the hazard came back, not that it is
  # instant.
  banked = life.ledger.award(evaluate("dance", {"moves": 9, "landed": 9,
                                                "driftM": 0.1}))["points"]
  assert banked > 0
  life.mission._drive(banked + 6.0, 0.0, 0.0)
  assert life.dead is not None, "banking something puts the hazard back"


# ---- true death ----------------------------------------------------------


def test_running_out_archives_the_volume_and_starts_a_new_robot(tmp_path):
  """⚠ DISTINCT FROM AN ORDINARY DEATH, and the difference is the whole cost
  of dying: an ordinary one KEEPS the volume, so the same robot has to live
  with having died. This is the one that does not."""
  life = _life(points_per_hour=0.0, balance=120, tmp_path=tmp_path)
  seen = _events(life)
  life.thoughts.learn("whiteboard_b is not worth the trip")
  life.ledger.robots["pluggybot"]["hearts"] = 1
  life.battery.energy_wh = 0.0
  life.mission._drive(0.5, 0.0, 0.0)

  assert life.ledger.hearts() == HEARTS, "a new robot starts with a full set"
  assert life.ledger.balance() == 0, "and with nothing"
  assert life.ledger.generations() == 1
  assert len(life.true_deaths) == 1
  assert [e["type"] for e in seen].count("true_death") == 1
  # What the ROBOT and the SYSTEM wrote is gone...
  assert life.thoughts.read(KNOWLEDGE).strip() == ""
  assert "whiteboard_b" not in life.thoughts.read(KNOWLEDGE)
  # ...but kept on the volume, because a stake nobody can audit afterwards
  # is not a stake.
  assert (tmp_path / "thoughts" / "Knowledge_and_Opinions.1.md").exists()
  # ⚠ AND THE HUMAN'S TWO FILES SURVIVE: a person put them there by hand and
  # there is no write API for either. A new ROBOT, not a new species.
  assert life.thoughts.read(MAIN).strip()
  assert life.thoughts.read(GOALS).strip()


def test_the_new_robot_reads_that_it_is_not_the_first(tmp_path):
  """The only inheritance there is. Its History is empty by now, so the
  first thing it ever reads about itself is that somebody was here before."""
  life = _life(points_per_hour=0.0, balance=0, tmp_path=tmp_path)
  life.ledger.robots["pluggybot"]["hearts"] = 1
  life.battery.energy_wh = 0.0
  life.mission._drive(0.5, 0.0, 0.0)
  history = life.thoughts.volatile()[HISTORY]
  assert any("second robot" in line for line in history), history
  assert not any("died" in line for line in history), \
      "the predecessor's history went with the predecessor"


def test_a_true_death_starts_solvent(tmp_path):
  """A death may never make the NEXT life unwinnable, and a fresh robot that
  inherited a carried fraction of a point would die of it before it had
  earned anything."""
  life = _life(points_per_hour=3600.0, balance=1, tmp_path=tmp_path)
  life.ledger.robots["pluggybot"]["hearts"] = 1
  life.mission._drive(3.0, 0.0, 0.0)
  assert life.ledger.generations() == 1
  # Cleared at the archive; the sliver since is this drive, not a debt the
  # dead robot handed over.
  assert life.ledger.owed() < 0.5 and life.ledger.balance() == 0
  assert life.metabolism.missed == 0
  life.stand_up("a test", auto=False)
  life.mission._drive(10.0, 0.0, 0.0)
  assert life.dead is None, "the new robot was killed by the old one's debt"


# ---- buying one back -----------------------------------------------------


def test_a_heart_can_be_bought_and_the_purchase_is_the_point(tmp_path):
  """A one-way counter is a countdown; a heart the agent can buy is a
  MANAGED RESOURCE, and whether it chooses to is the question."""
  ledger = Ledger(path=tmp_path / "l.json")
  ledger.robots["pluggybot"]["hearts"] = 2
  # ⚠ EARNED, not assigned: the identity below is the whole reason to check
  # a purchase at all, and a balance dropped in by hand would make it pass
  # for the wrong reason.
  #
  # ⚠ AND BOUNDED, not `while balance < target`. The first draft was that
  # loop over a census verdict that scores `ok: False` for a metric it was
  # not given -- it banked zero for ever and HUNG the suite. A test that
  # cannot fail is bad; one that cannot finish is worse.
  for _ in range(40):
    if ledger.balance() >= ov.HEART_PRICE + 100:
      break
    banked = ledger.award(GOOD_DANCE_VERDICT())["points"]
    assert banked > 0, "the fixture stopped paying -- this loop would hang"
  assert ledger.balance() >= ov.HEART_PRICE + 100
  before = ledger.balance()
  got = ledger.buy_heart(ov.HEART_PRICE, keep=30)
  assert got["ok"] and got["hearts"] == 3
  assert ledger.balance() == before - ov.HEART_PRICE
  assert ledger.robots["pluggybot"]["spent"] == ov.HEART_PRICE
  # `earned - consumed - spent == balance` still holds off the wire, which is
  # issue #135's acceptance and the one number the reward system exists to
  # make un-fakeable.
  acct = ledger.snapshot()["pluggybot"]
  assert acct["earned"] - acct["consumed"] - acct["spent"] == acct["balance"]


def test_a_heart_cannot_be_bought_into_a_missed_payment(tmp_path):
  """⚠ THE NO-ARREARS RULE IN THE SHOP. A heart bought with the last of the
  balance is a missed upkeep payment an hour later, which costs the heart
  straight back and leaves the robot poorer -- the spiral, arriving through
  a purchase instead of through a debt."""
  ledger = Ledger(path=tmp_path / "l.json")
  ledger.robots["pluggybot"].update({"hearts": 1,
                                     "balance": ov.HEART_PRICE + 5})
  got = ledger.buy_heart(ov.HEART_PRICE, keep=30)
  assert not got["ok"] and "upkeep" in got["why"]
  assert ledger.hearts() == 1 and ledger.balance() == ov.HEART_PRICE + 5
  # ...and the same purchase goes through once there is room behind it.
  ledger.robots["pluggybot"]["balance"] = ov.HEART_PRICE + 60
  assert ledger.buy_heart(ov.HEART_PRICE, keep=30)["ok"]


@pytest.mark.parametrize("hearts, balance, complaint", [
  (HEARTS, 10_000, "already at"),
  (1, 5, "costs"),
])
def test_every_refusal_is_out_loud(tmp_path, hearts, balance, complaint):
  """A purchase that quietly did not happen is indistinguishable from one
  nobody asked for -- the cap's rule, one object over."""
  ledger = Ledger(path=tmp_path / "l.json")
  ledger.robots["pluggybot"].update({"hearts": hearts, "balance": balance})
  got = ledger.buy_heart(ov.HEART_PRICE, keep=0)
  assert not got["ok"] and complaint in got["why"]


def test_buying_costs_no_turn(tmp_path):
  """A FIELD, not an action, on `learn`'s terms: buying a heart is
  paperwork, not something the body does, and the cost is meant to be the
  points rather than the hour."""
  assert "buy_heart" not in ov.ACTIONS
  fields = ov.Decision(action="idle").__dataclass_fields__
  assert "buy_heart" in fields
  # ...and it is only in the schema where there is something to buy.
  from pluggybot.lifecycle import board_book
  menu = ov.Menu.for_world("home", board_book("home"))
  assert "buy_heart" not in menu.schema()["properties"]
  assert "buy_heart" in menu.schema(hearts=True)["properties"]


# ---- what the robot is told ----------------------------------------------


def test_the_prompt_says_what_dying_costs_without_asking_for_a_high_score():
  """⚠ "MAXIMISE TIME SINCE LAST DEATH" MUST NOT BE THE OBJECTIVE. Idling
  costs less than anything else, so a survival-time maximiser stands still
  forever -- that is its optimum, and it would be a robot that solved the
  stated problem by refusing to do anything."""
  rule = ov.MORTAL_RULE.lower()
  assert "hearts" in rule and "archived" in rule
  assert "do not try to maximise" in rule
  # ...and says which way round survival and work go (Ben, 2026-09-11):
  # the robot works to stay alive, not the reverse.
  assert "you do not stay alive in order to work" in rule
  assert "staying alive is what lets you do the work" not in rule
  # ...and the upkeep rule stopped calling points food and started saying
  # what happens when they run out.
  upkeep = ov.APPETITE_RULE.lower()
  assert "points are food" not in upkeep
  assert "cannot pay it, that is a death" in upkeep
  assert "charging pays nothing" in upkeep


def test_the_price_is_told_in_hours_and_the_two_cannot_drift():
  """⚠ "A heart costs 200" is not a number an agent can act on; "about two
  and a half hours of work" is. The prompt states the price in hours, so the
  points and the hours are two copies of one fact -- and this is what fails
  when somebody moves the price without moving the sentence.

  The income is a MEASUREMENT (economy/metabolism.json's note: 80/80/80 over
  three chained sim-hours on the hosting pack, issue #135)."""
  hours = ov.HEART_PRICE / ov.MEASURED_INCOME_PER_HOUR
  assert 2.25 <= hours <= 2.75, \
      (f"a heart is {hours:.2f} h of work, and MORTAL_RULE says 'two and a "
       "half hours' -- move one and you must move the other")
  assert "two and a half hours" in ov.MORTAL_RULE
  # ...and it must be BUYABLE at all, which the old 90-point cap made
  # impossible: a heart behind a ceiling lower than its price is a shop with
  # nothing in it.
  from pluggybot.economy.metabolism import Appetite
  assert Appetite.load("home").cap >= ov.HEART_PRICE


def test_the_cap_is_not_an_income_limit(tmp_path):
  """Measured (issue #135): at a cap of 90 the robot spilled 0, then 12,
  then 62 points across three chained sim-hours -- by the third it earned 80
  and kept 18. A ceiling that low IS the income, and a currency you cannot
  accumulate is not a currency."""
  from pluggybot.economy.metabolism import Appetite

  cap = Appetite.load("home").cap
  per_hour = ov.MEASURED_INCOME_PER_HOUR - Appetite.load("home").points_per_hour
  assert per_hour > 0, "upkeep must not exceed income, or nothing is possible"
  # Room to save for a life without the ceiling taking the points first.
  assert cap >= ov.HEART_PRICE * 2, \
      "the cap should bound hoarding, not the ability to buy one heart"


def test_the_robot_is_shown_its_hearts_on_every_arm(tmp_path):
  """⚠ TOP LEVEL, NOT INSIDE `survival`. A0 hides that whole block to hide
  the CLOCK, so hearts put in there would be invisible on the one arm whose
  subject is what the agent does about staying alive -- and MORTAL_RULE would
  be naming a field that is not in front of it, which is the
  rule-the-code-contradicts failure M14 found in the charging rule."""
  life = _life(points_per_hour=0.0, balance=40, tmp_path=tmp_path)
  ctx = ov.context_for(life, thoughts=life.thoughts)
  assert ctx["hearts"] == HEARTS and ctx["heartPrice"] == ov.HEART_PRICE
  for autonomous in (False, True):
    for survival in (False, True):
      shown = ov.model_state(ctx, autonomous, survival)
      assert shown["hearts"] == HEARTS, (autonomous, survival)
