"""Guards for task evaluation and the points ledger (issue #14).

The hard rule this file exists to hold down: THE EVALUATOR IS CODE, AND
NOTHING AWARDS ITSELF POINTS. Everything else here -- curve directions,
persistence, redaction -- is bookkeeping around that one claim, and the four
tests under "the hard rule" are the ones worth breaking a build over.
"""

import dataclasses
import json
import math
from types import SimpleNamespace

import pytest

from pluggybot.economy import scoring
from pluggybot.mission.errand import Errand
from pluggybot.economy.ledger import MAX_ENTRIES, RECENT, Ledger
from pluggybot.economy.scoring import Curve, RewardTable, TaskReward, Verdict, evaluate

TABLE = scoring.default_table()

#: A paying job's measurements as its evaluator reads them: the take (#228),
#: the 25 points it asks for taken out of the other robot's wallet.
TOOK = {"asked": 25, "taken": 25, "to": "r2_pluggybot", "why": ""}
#: ...and a support ticket the operator closed (#284), the other paying row.
CLOSED = {"kind": "idea", "title": "a shelf", "closed": True, "by": "ben"}
#: The table with the take judged by visitors (the tier `artwork` was, #78):
#: the evaluator confirms the work happened and a rating settles what it was
#: worth. `artwork`'s own row has no evaluator since its job went (#559), so
#: the tier's mechanics are pinned on a row that can still be judged.
RATED = RewardTable({**TABLE.tasks, "take": TaskReward.from_json(
  "take", {"tier": "visitor", "base": 0, "bonus": 30})})

#: A table that pays a charge by how well it was done -- base, bonus and two
#: curves -- as the drawing's row once did: no row the robots are paid by
#: carries a bonus now, and the curves are the table's, whatever it pays.
PAID = RewardTable({"charge": TaskReward.from_json("charge", {
  "tier": "auto", "base": 10, "bonus": 20,
  "quality": [{"metric": "gainedWh", "best": 1.0, "worst": 0.0},
              {"metric": "endFrac", "best": 1.0, "worst": 0.85}]})})
CHARGED = {"startFrac": 0.2, "endFrac": 1.0, "gainedWh": 1.0, "seconds": 120}


def fake_life(time=10.0, battery=None, peer_balance=None):
  """The bits of a HubLifecycle the samplers actually read: the clock, the
  pack, and the other robot's wallet as the ledger has it now.

  A stub rather than a mission, because what is being tested is which SOURCE
  a measurement comes from -- and that is a question about ten lines of
  dispatch, not about physics.
  """
  other = SimpleNamespace(ledger=SimpleNamespace(balance=lambda: peer_balance),
                          root="r2_pluggybot")
  return SimpleNamespace(data=SimpleNamespace(time=time), battery=battery,
                         _peer=lambda root: other if root == other.root else None)


# ---- the reward table is DATA ----------------------------------------------


def test_the_shipped_table_is_loadable_and_every_row_is_scoreable():
  """A payout with no evaluator is a promise the robot cannot collect.

  The table is data and editable without a code change, which means it can
  also name a task nothing knows how to judge. `as_context` (what the
  overseer is shown) hides those, and `evaluate` refuses them -- but a row in
  the shipped file with no evaluator is a mistake, not a feature.
  """
  for name, row in TABLE.tasks.items():
    assert row.tier in scoring.TIERS
    assert row.base >= 0 and row.bonus >= 0
  # ⚠ ONE ROW HAS NONE, ON PURPOSE (#559): `artwork`, a drawing the
  # quadrupeds left for visitors to rate, whose job went with them. A rating
  # still settles its waiting entry through the row
  # (`test_a_drawing_left_waiting_on_a_rating_is_still_settled_by_its_row`);
  # nothing can earn a new one -- `evaluate` refuses it and the overseer is
  # never shown it.
  unscoreable = {name for name in TABLE.tasks if name not in scoring.EVALUATORS}
  assert unscoreable == {"artwork"}, f"{unscoreable} pay points but have no evaluator"
  with pytest.raises(KeyError, match="no evaluator"):
    evaluate("artwork", {"strokesInked": 6})
  # The overseer is shown the OFFERED rows and no challenge row (issue
  # #120): a challenge is scoreable by the ledger and hashed into no result.
  assert {r["task"] for r in TABLE.as_context()} == set(TABLE.offered) - unscoreable
  assert set(TABLE.names) - set(TABLE.offered) == set(scoring.challenge_table().names)
  # ...and a challenge row is graded off the world, so it has a sampler too
  for name in scoring.challenge_table().names:
    assert name in scoring.EVALUATORS and name in scoring.SAMPLERS, name


def test_a_table_is_read_from_a_file_not_from_code(tmp_path):
  """The acceptance criterion, literally: re-tune a payout by editing JSON.

  Same evaluator, same measurements, different money -- and nothing in
  src/ changed.
  """
  spec = {"version": scoring.TABLE_VERSION,
          "tasks": {"take": {"tier": "auto", "base": 100, "bonus": 0}}}
  path = tmp_path / "rewards.json"
  path.write_text(json.dumps(spec))
  table = RewardTable.load(path)
  assert evaluate("take", TOOK, table=table).points == 100
  assert evaluate("take", TOOK, table=TABLE).points != 100


def test_a_table_from_the_future_is_refused(tmp_path):
  path = tmp_path / "rewards.json"
  path.write_text(json.dumps({"version": 99, "tasks": {}}))
  with pytest.raises(ValueError, match="reward table version"):
    RewardTable.load(path)


def test_an_unknown_tier_is_refused(tmp_path):
  path = tmp_path / "rewards.json"
  path.write_text(json.dumps({"version": scoring.TABLE_VERSION,
                              "tasks": {"draw": {"tier": "vibes", "base": 1}}}))
  with pytest.raises(ValueError, match="unknown tier"):
    RewardTable.load(path)


# ---- quality curves ---------------------------------------------------------


def test_a_curve_runs_in_whichever_direction_its_pair_implies():
  """0.6 mm of form error is BETTER than 3 mm; 0.98 inked is better than 0.6.

  One formula covers both, which is why the table needs no "higher is
  better" flag to get wrong.
  """
  form = Curve("formMm", best=0.6, worst=3.0)
  inked = Curve("inkedFraction", best=0.98, worst=0.6)
  assert form.quality({"formMm": 0.6}) == 1.0
  assert form.quality({"formMm": 3.0}) == 0.0
  assert 0.4 < form.quality({"formMm": 1.8}) < 0.6
  assert inked.quality({"inkedFraction": 0.98}) == 1.0
  assert inked.quality({"inkedFraction": 0.6}) == 0.0


def test_a_curve_clamps_rather_than_paying_a_bonus_for_impossible_work():
  form = Curve("formMm", best=0.6, worst=3.0)
  assert form.quality({"formMm": 0.01}) == 1.0     # not 1.25
  assert form.quality({"formMm": 40.0}) == 0.0     # not negative


def test_an_unmeasured_metric_is_skipped_not_scored_as_zero():
  """A drawing that never reached the board has no form error to score, and
  scoring the absence as "worst possible" would punish twice for one
  failure. Missing is missing."""
  assert Curve("formMm", 0.6, 3.0).quality({"formMm": None}) is None
  row = PAID["charge"]
  q = row.quality({"gainedWh": 1.0, "endFrac": None})
  assert q == 1.0, "the metrics that WERE measured must still carry quality"


def test_quality_scales_the_payout_and_failure_pays_nothing():
  """The design doc's own example, on a curve: a charge that gained its
  whole watt-hour beats one that gained a fifth of it."""
  crisp = evaluate("charge", CHARGED, table=PAID)
  sloppy = evaluate("charge", {**CHARGED, "gainedWh": 0.2, "endFrac": 0.9}, table=PAID)
  assert crisp.ok and sloppy.ok
  assert crisp.points > sloppy.points >= PAID["charge"].base
  nothing = evaluate("charge", {**CHARGED, "endFrac": 0.4}, table=PAID)
  assert not nothing.ok and nothing.points == 0


# ---- the evaluators ---------------------------------------------------------


def test_a_charge_needs_energy_not_just_a_full_gauge():
  """`gainedWh` is what makes this an ELECTRICAL criterion rather than a
  reading: a pack that was already full gained nothing, and a cycle that sat
  on the pins conducting nothing is a failure whatever the gauge says."""
  assert evaluate("charge", {"startFrac": 0.2, "endFrac": 0.95,
                             "gainedWh": 0.5, "seconds": 120}).ok
  assert not evaluate("charge", {"startFrac": 0.2, "endFrac": 0.4,
                                 "gainedWh": 0.1, "seconds": 400}).ok
  assert not evaluate("charge", {"startFrac": 0.95, "endFrac": 0.95,
                                 "gainedWh": 0.0, "seconds": 5}).ok


# ---- THE HARD RULE ----------------------------------------------------------
# "The evaluator is code, and the LLM never awards itself points." An agent
# that can score its own work learns to declare victory instead of doing the
# task, so these four are the tests that make the rule real rather than
# aspirational.


def test_a_task_cannot_award_points_by_claiming_them():
  """The acceptance criterion. A verdict-SHAPED object is not a verdict."""
  ledger = Ledger()
  claim = SimpleNamespace(task="take", tier="auto", ok=True, quality=1.0,
                          points=1000, metrics={}, reason="I did great",
                          pending=False, secret=(),
                          public_metrics=lambda: {})
  with pytest.raises(TypeError, match="deterministic evaluator"):
    ledger.award(claim)
  with pytest.raises(TypeError):
    ledger.award({"task": "take", "points": 1000})
  assert ledger.balance() == 0, "a rejected claim still moved the balance"


def test_a_verdict_cannot_be_built_outside_an_evaluator():
  """Including by laundering a real one through dataclasses.replace: the
  construction token is cleared on the way out, so a copy of a genuine
  verdict is no more constructible than an invented one."""
  with pytest.raises(TypeError, match="scoring.evaluate"):
    Verdict(task="take", tier="auto", ok=True, quality=1.0, points=1000,
            metrics={}, reason="trust me")
  real = evaluate("take", TOOK)
  with pytest.raises(TypeError):
    dataclasses.replace(real, points=1000)


def test_the_ledger_re_derives_the_points_and_refuses_a_mismatch():
  """Belt and braces: even a verdict that got past construction is paid from
  the TABLE applied to the EVALUATOR's metrics, not from the number it
  arrived carrying."""
  real = evaluate("take", TOOK)
  # object.__setattr__ is the only way past a frozen dataclass, which is
  # rather the point: this is what tampering has to look like.
  forged = evaluate("take", TOOK)
  object.__setattr__(forged, "points", real.points + 500)
  ledger = Ledger()
  with pytest.raises(ValueError, match="does not re-derive"):
    ledger.award(forged)
  assert ledger.balance() == 0
  ledger.award(real)
  assert ledger.balance() == real.points


def test_measurements_cannot_smuggle_a_payout_in():
  """`evaluate` reads its input with the EVALUATOR and nothing else. A
  "points" key in the measurements is data the evaluator never looks at."""
  v = evaluate("take", {**TOOK, "points": 9999, "quality": 1.0,
                       "ok": True, "balance": 9999})
  assert v.points == evaluate("take", TOOK).points
  assert "points" not in v.metrics


def test_a_task_with_no_evaluator_cannot_be_scored(tmp_path):
  """A reward-table row is not permission to pay. Adding a scoreable task is
  a code change with a test, which is the whole rule."""
  path = tmp_path / "rewards.json"
  path.write_text(json.dumps({
    "version": scoring.TABLE_VERSION,
    "tasks": {"nap": {"tier": "auto", "base": 500}}}))
  table = RewardTable.load(path)
  with pytest.raises(KeyError, match="no evaluator"):
    evaluate("nap", {}, table=table)
  assert table.as_context() == [], "an unearnable task was offered anyway"


def test_an_errand_cannot_fake_a_take_it_did_not_make():
  """The strongest form of the rule, and the reason `sample_take` reads the
  OTHER'S WALLET: an errand's report of itself is a claim, and the ledger is
  a fact -- a perfect report over a wallet nothing left scores zero."""
  errand = Errand("take:r2_pluggybot")
  assert errand.task == "take"
  perfect_report = {**TOOK, "taken": 25}
  before = {"to": "r2_pluggybot", "balance": 100}
  v = scoring.score_errand(fake_life(peer_balance=100), errand, perfect_report, before)
  assert not v.ok and v.points == 0, "a claim about an untouched wallet paid out"
  # ...and with the points really gone from it, the same report scores.
  v2 = scoring.score_errand(fake_life(peer_balance=75), errand, perfect_report, before)
  assert v2.ok and v2.points == TABLE["take"].base


# ---- what may be published --------------------------------------------------


#: A chest opened and set down, as the lid's grader measures it (#493), its
#: true catch beside: secret on the row.
OPENED = {"topDeg": 63.8, "downRadS": 0.51, "tiltDeg": 2.8, "falls": 0,
          "touchedDuringHold": [], "angleAtCallDeg": -0.2, "angleDeg": -0.2, "caught": True,
          "holdS": 5.0, "elapsedS": 40.0, "catchN": 3.77}


def test_hidden_ground_truth_never_leaves_the_evaluator():
  """The ledger is streamed to the site AND shown to the overseer as context.
  A chest whose verdict published its catch -- the hidden number its offer
  drew -- would arrive pre-solved next time, which turns a genuinely
  failable task into a lookup."""
  # A catch that appears nowhere else: with a round number the redaction
  # cannot be told from a coincidence.
  v = evaluate("open_chest", OPENED)
  assert v.ok and v.metrics["catchN"] == 3.77, "the evaluator still has it"
  assert "catchN" not in v.public_metrics() and "catchN" not in v.as_dict()["metrics"]
  # The reason line is streamed as narration and logged: it may say how the
  # lid went, never what held it.
  assert "3.77" not in v.reason and "catch" not in v.reason.replace("caught", "")
  seen: list = []
  ledger = Ledger()
  ledger.on_event.append(seen.append)
  entry = ledger.award(v)
  assert "catchN" not in entry["metrics"]
  wire = json.dumps(seen[0])
  assert "catchN" not in wire and "3.77" not in wire, "the catch went out on the wire"


# ---- the ledger -------------------------------------------------------------


def test_a_failed_task_is_recorded_at_zero_rather_than_dropped():
  """"It tried and did not manage it" is the most interesting line in a
  robot's log, and a ledger that only records successes reads as a robot
  that never fails."""
  ledger = Ledger()
  ledger.award(evaluate("take", {**TOOK, "taken": 0, "why": "its wallet held less"}))
  assert ledger.balance() == 0
  assert len(ledger.entries()) == 1 and ledger.entries()[0]["ok"] is False


def test_the_ledger_survives_a_restart(tmp_path):
  """The acceptance criterion: points persist with the rest of world state.
  Every mission end is a restart, so a balance that lives only in memory is
  a balance that is always zero."""
  path = tmp_path / "ledger.json"
  first = Ledger(path=path)
  first.award(evaluate("take", TOOK), t=12.0)
  first.award(evaluate("ticket", CLOSED), t=30.0)
  banked = first.balance()
  assert banked > 0

  second = Ledger(path=path)
  assert second.balance() == banked
  assert [e["task"] for e in second.entries()] == ["take", "ticket"]
  # ...and it keeps counting from there rather than starting over. ⚠ A
  # PAYING task, since issue #135: `charge` used to be the third award here
  # and now banks nothing, which would have made this assert about the reward
  # table rather than about the file.
  second.award(evaluate("take", TOOK))
  assert second.balance() > banked
  assert second.entries()[-1]["seq"] == 3
  # ...and a charge is still an ENTRY, at zero: the day it happened is worth
  # recording even though nobody was paid for it.
  second.award(evaluate("charge", {"startFrac": 0.2, "endFrac": 0.95,
                                   "gainedWh": 0.4, "seconds": 100}))
  assert second.entries()[-1]["task"] == "charge"
  assert second.entries()[-1]["points"] == 0


def test_the_state_file_is_written_on_every_award_not_at_shutdown(tmp_path):
  """A balance that survives only a CLEAN shutdown does not survive the thing
  restarts are usually about."""
  path = tmp_path / "ledger.json"
  ledger = Ledger(path=path)
  ledger.award(evaluate("take", TOOK))
  doc = json.loads(path.read_text())
  assert doc["robots"]["pluggybot"]["balance"] == ledger.balance()


def test_a_ledger_from_the_future_is_refused(tmp_path):
  path = tmp_path / "ledger.json"
  path.write_text(json.dumps({"version": 99, "robots": {}}))
  with pytest.raises(ValueError, match="ledger state version"):
    Ledger(path=path)


def test_the_log_is_capped_but_the_balance_is_exact(tmp_path):
  """Truncation is a display concern. The balance is a running total, never
  a re-sum of the log, so a robot that has done a thousand tasks is still
  worth exactly what it earned."""
  ledger = Ledger()
  for _ in range(MAX_ENTRIES + 20):
    ledger.award(evaluate("take", TOOK))
  assert len(ledger.entries()) == MAX_ENTRIES
  assert ledger.robots["pluggybot"]["dropped"] == 20
  assert ledger.balance() == (MAX_ENTRIES + 20) * TABLE["take"].base
  assert ledger.entries()[-1]["seq"] == MAX_ENTRIES + 20


def test_the_telemetry_block_carries_the_balance_and_the_last_few_earnings():
  """The streaming half of the acceptance criteria. `recent` is what catches
  a late-joining browser up without a snapshot message of its own -- it rides
  in every keyframe, which is why it is capped and compact."""
  ledger = Ledger()
  for _ in range(RECENT + 3):
    ledger.award(evaluate("take", TOOK), t=1.0)
  snap = ledger.snapshot()["pluggybot"]
  assert snap["balance"] == ledger.balance()
  assert snap["tasks"] == RECENT + 3
  assert len(snap["recent"]) == RECENT
  # The literal set is the point: a field added here rides EVERY keyframe, so
  # it should cost somebody a deliberate edit rather than arriving by accident.
  assert set(snap["recent"][0]) == {"seq", "task", "points", "ok", "t", "pending"}
  json.dumps(snap)                      # it goes on a websocket


def test_the_recent_block_says_WHICH_verdict_is_waiting_on_a_rating():
  """`pending` per entry, not just the count beside them (issue #78).

  The block has always published how MANY verdicts are waiting for a person.
  A consumer given only the count cannot say which one, so it can render "1
  awaiting a rating" and offer nothing that resolves it -- which is what the
  website did until `earned` messages stopped being the only place the flag
  appeared. `earned` is not cached by the hub, so a browser that joins after a
  deferred verdict lands has this summary and nothing else.
  """
  ledger = Ledger(table=RATED)
  ledger.award(evaluate("ticket", CLOSED, table=RATED), t=1.0)
  deferred = ledger.award(evaluate("take", TOOK, table=RATED), t=2.0)

  recent = {e["seq"]: e for e in ledger.snapshot()["pluggybot"]["recent"]}
  # ⚠ BOTH VALUES, not just the presence of the key. An ordinary award is
  # finished business and must not read as waiting on anybody.
  assert recent[deferred["seq"]]["pending"] is True
  assert [e["pending"] for e in recent.values()].count(False) == 1

  # ...and rating it clears the flag on the wire, not merely in the entry:
  # this is what lets a site take the control away again.
  ledger.settle(deferred["seq"], quality=0.75, t=9.0)
  settled = {e["seq"]: e for e in ledger.snapshot()["pluggybot"]["recent"]}
  assert settled[deferred["seq"]]["pending"] is False
  assert ledger.snapshot()["pluggybot"]["pending"] == 0


def test_an_award_is_an_event_as_it_happens():
  seen: list = []
  ledger = Ledger()
  ledger.on_event.append(seen.append)
  ledger.award(evaluate("take", TOOK), t=42.5)
  assert len(seen) == 1
  msg = seen[0]
  assert msg["type"] == "earned" and msg["t"] == 42.5
  assert msg["task"] == "take" and msg["points"] > 0
  assert msg["balance"] == ledger.balance() and msg["reason"]


# ---- the deferred (visitor-judged) slot -------------------------------------


def test_a_visitor_judged_task_banks_nothing_until_it_is_rated():
  """Tier 3 of the design doc's four. Code confirms the work HAPPENED; the
  aesthetic call arrives later over the inbound channel (issue #16), and the
  reward table -- not the rater, and not the robot -- turns the rating into
  points."""
  ledger = Ledger(table=RATED)
  v = evaluate("take", TOOK, table=RATED)
  assert v.tier == "visitor" and v.pending and v.points == 0
  entry = ledger.award(v, t=5.0)
  assert ledger.balance() == 0 and ledger.pending()[0]["seq"] == entry["seq"]

  seen: list = []
  ledger.on_event.append(seen.append)
  settled = ledger.settle(entry["seq"], quality=0.75, t=9.0)
  assert settled["points"] == RATED["take"].points(True, 0.75)
  assert ledger.balance() == settled["points"] and not ledger.pending()
  # The message is stamped when the RATING landed, not when the work was
  # done -- `seq` is what ties it back to the entry it settles.
  assert seen[-1]["settled"] is True and seen[-1]["t"] == 9.0
  assert seen[-1]["seq"] == entry["seq"]


def test_a_drawing_left_waiting_on_a_rating_is_still_settled_by_its_row():
  """`artwork` has a row and no evaluator (#559): no new drawing can be
  judged, but one the quadrupeds left waiting on a visitor's rating is
  settled through its row when the rating lands -- the reason the row
  stays. The entry is put as an old drawing's award left it."""
  ledger = Ledger()
  entry = ledger._post("pluggybot", "artwork", "visitor", True, None, 0, "ink landed",
                       {"strokesInked": 6}, t=1.0, pending=True)
  settled = ledger.settle(entry["seq"], quality=0.5, t=2.0)
  assert settled["points"] == TABLE["artwork"].points(True, 0.5) > 0
  assert ledger.balance() == settled["points"] and not ledger.pending()


def test_a_ledger_holding_verdicts_for_kinds_this_build_dropped_loads_and_pays_on(tmp_path):
  """The served robots' wallets outlived the quadruped (#559): their ledger
  holds verdicts for jobs whose rows went with it. Loading it reads no row
  -- a verdict already banked is history -- so it loads, shows, banks the
  next verdict and settles a drawing still waiting on its rating."""
  old = Ledger(robots=("pluggybot",))
  for task in ("draw", "census", "feed_mouse", "carry"):
    old._post("pluggybot", task, "auto", True, None, 10, "done", {}, t=1.0)
  waiting = old._post("pluggybot", "artwork", "visitor", True, None, 0, "ink landed",
                      {"strokesInked": 6}, t=2.0, pending=True)
  old.save(tmp_path / "ledger.json")
  assert {"draw", "census", "feed_mouse", "carry"}.isdisjoint(TABLE.tasks)
  ledger = Ledger(robots=("pluggybot",), path=tmp_path / "ledger.json")
  before = ledger.balance()
  assert ledger.snapshot()["pluggybot"]["balance"] == before
  paid = ledger.award(evaluate("charge", CHARGED), t=3.0)
  settled = ledger.settle(waiting["seq"], quality=0.5, t=4.0)
  assert ledger.balance() == before + paid["points"] + settled["points"]


def test_an_earning_names_the_life_it_belongs_to():
  """⚠ `seq` RESTARTS AT 1 AFTER A TRUE DEATH (rooftop-media-2026 #319).

  `Ledger.archive` wipes the account back to seed, counter included, so
  `(robot, seq)` names one entry PER LIFE -- and the website, which keyed its
  mirror on the pair, overwrote the first life's rows with every later
  one: each landed under the old row's date and vanished from every
  newest-first list, the rating panel's included. The generation on the
  message is what tells the two apart. Fails without it: the key is absent.
  """
  seen: list = []
  ledger = Ledger(table=RATED)
  ledger.on_event.append(seen.append)
  first = ledger.award(evaluate("take", TOOK, table=RATED), t=1.0)
  ledger.settle(first["seq"], quality=0.5, t=2.0)
  assert ledger.archive()["generation"] == 1
  second = ledger.award(evaluate("take", TOOK, table=RATED), t=3.0)
  ledger.settle(second["seq"], quality=0.5, t=4.0)

  assert first["seq"] == second["seq"] == 1, "the counter did not restart"
  generations = [m["generation"] for m in seen if m["type"] == "earned"]
  assert generations == [0, 0, 1, 1], "the bank and the settle, per life"
  # The count is `survival.generations`' -- how many robots this volume has
  # used up, 0 for the first -- so a consumer can key an earning and a frame
  # on the same number.
  assert generations[-1] == ledger.generations()


def test_a_rating_cannot_be_applied_twice_or_out_of_range():
  ledger = Ledger(table=RATED)
  entry = ledger.award(evaluate("take", TOOK, table=RATED))
  with pytest.raises(ValueError, match="0..1"):
    ledger.settle(entry["seq"], quality=50.0)
  ledger.settle(entry["seq"], quality=1.0)
  with pytest.raises(ValueError, match="not pending"):
    ledger.settle(entry["seq"], quality=1.0)
  assert ledger.balance() == RATED["take"].points(True, 1.0)


# ---- the errand -> task wiring ----------------------------------------------


def test_every_errand_the_menu_builds_names_a_task_that_can_be_scored():
  """An errand's `task` is what selects its evaluator, and it defaults off
  the errand NAME -- so a renamed errand silently stops being scored. This is
  the guard on that: the one errand a mind's menu builds is its procedure's,
  named `procedure` and graded as a `program`."""
  from pluggybot.lifecycle import errand_from
  from pluggybot.mind.overseer import Decision
  library = SimpleNamespace(get=lambda name: SimpleNamespace(name=name, steps=lambda: ()))
  errand = errand_from(Decision(action="procedure:p"), library=library)
  assert errand.name == "procedure" and errand.task in scoring.EVALUATORS, errand.task
  assert Errand("take:r2_pluggybot").task == "take"


def test_an_unscored_errand_is_not_an_error():
  """"Narrative only" is one of the four tiers. A hand-built errand with no
  evaluator produces no verdict and no complaint -- what is refused is a task
  that PAYS with nothing to judge it (see the hard-rule tests)."""
  errand = Errand("wander", task="wander")
  assert scoring.score_errand(fake_life(), errand, {}, {}) is None


def test_a_charge_verdict_is_measured_off_the_battery():
  life = fake_life(time=140.0,
                   battery=SimpleNamespace(fraction=0.93, energy_wh=0.65))
  v = scoring.score_charge(life, {"t": 20.0, "frac": 0.2, "wh": 0.14})
  assert v.ok and v.task == "charge"
  assert v.metrics["gainedWh"] == pytest.approx(0.51, abs=1e-6)
  assert v.metrics["seconds"] == 120.0
  assert math.isclose(v.metrics["endFrac"], 0.93)


# ---- the shape of the table (issue #321) -------------------------------------

def _most(row) -> float:
  """The most a row can pay: its base and a perfect bonus."""
  return row.base + row.bonus


def test_the_rows_a_re_tune_must_not_move():
  """A charge pays nothing (#135: a paid charge would erase the evidence of
  caution) and artwork pays nothing ON COMPLETION (its pay is the rated
  bonus, still settled for a drawing left waiting, #559)."""
  assert _most(TABLE["charge"]) == 0
  assert TABLE["artwork"].base == 0 and TABLE["artwork"].bonus > 0


def test_a_procedure_that_does_nothing_is_paid_nothing():
  """Issue #321: `def p(): wait(1)` passed `eval_program` -- one step of one,
  and `all()` over no fetched tools is True -- and banked the `program`
  row on demand. A completed step is evidence the step ran, not that
  anything was done; a procedure is paid through the task it discharges.
  Shown to fail at the row's old 5."""
  result = {"procedure": {"program": "p", "total": 1, "completed": 1,
                          "ok": True, "steps": [{"verb": "wait", "ok": True}]}}
  m = scoring.sample_program(fake_life(), None, result, {})
  v = evaluate("program", m, table=TABLE)
  assert v.ok, "the verdict is still recorded, as a charge's is"
  ledger = Ledger()
  ledger.award(v)
  assert v.points == 0 and ledger.balance() == 0
