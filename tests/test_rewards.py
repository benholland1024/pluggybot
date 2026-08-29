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
from pluggybot.tools.boards import BoardBook, BoardRecord
from pluggybot.mission.errand import Errand, carry_errand, dance_errand
from pluggybot.economy.ledger import MAX_ENTRIES, RECENT, Ledger
from pluggybot.economy.scoring import Curve, RewardTable, Verdict, evaluate

TABLE = scoring.default_table()

GOOD_DRAWING = {"board": "whiteboard_a", "strokes": 6, "strokesInked": 6,
                "formMm": 0.6, "inkedFraction": 0.99,
                "travelInkFraction": 0.0, "fill": 0.3}


def book(*records) -> BoardBook:
  return BoardBook(records or [BoardRecord("whiteboard_a", (0.11, 0.2))],
                   clock=lambda: "2026-08-16T00:00:00")


def fake_life(boards=None, module_hung=True, time=10.0, battery=None):
  """The bits of a HubLifecycle the samplers actually read.

  A stub rather than a mission, because what is being tested is which SOURCE
  a measurement comes from -- and that is a question about ten lines of
  dispatch, not about physics.
  """
  swap = SimpleNamespace(
    module_state=lambda name: {"hung": module_hung, "on_fork": not module_hung})
  return SimpleNamespace(boards=boards, data=SimpleNamespace(time=time),
                         mission=SimpleNamespace(swap=swap), battery=battery)


# ---- the reward table is DATA ----------------------------------------------


def test_the_shipped_table_is_loadable_and_every_row_is_scoreable():
  """A payout with no evaluator is a promise the robot cannot collect.

  The table is data and editable without a code change, which means it can
  also name a task nothing knows how to judge. `as_context` (what the
  overseer is shown) hides those, and `evaluate` refuses them -- but a row in
  the shipped file with no evaluator is a mistake, not a feature.
  """
  for name, row in TABLE.tasks.items():
    assert name in scoring.EVALUATORS, f"{name} pays points but has no evaluator"
    assert row.tier in scoring.TIERS
    assert row.base >= 0 and row.bonus >= 0
  assert {r["task"] for r in TABLE.as_context()} == set(TABLE.names)


def test_a_table_is_read_from_a_file_not_from_code(tmp_path):
  """The acceptance criterion, literally: re-tune a payout by editing JSON.

  Same evaluator, same measurements, different money -- and nothing in
  src/ changed.
  """
  spec = {"version": scoring.TABLE_VERSION,
          "tasks": {"draw": {"tier": "auto", "base": 100, "bonus": 0}}}
  path = tmp_path / "rewards.json"
  path.write_text(json.dumps(spec))
  table = RewardTable.load(path)
  assert evaluate("draw", GOOD_DRAWING, table=table).points == 100
  assert evaluate("draw", GOOD_DRAWING, table=TABLE).points != 100


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
  row = TABLE["draw"]
  q = row.quality({"completeness": 1.0, "formMm": None, "inkedFraction": None})
  assert q == 1.0, "the metrics that WERE measured must still carry quality"


def test_quality_scales_the_payout_and_failure_pays_nothing():
  """The design doc's own example: a 0.6 mm drawing beats a 3 mm one."""
  crisp = evaluate("draw", GOOD_DRAWING)
  sloppy = evaluate("draw", {**GOOD_DRAWING, "formMm": 3.0,
                             "inkedFraction": 0.6})
  assert crisp.ok and sloppy.ok
  assert crisp.points > sloppy.points >= TABLE["draw"].base
  nothing = evaluate("draw", {**GOOD_DRAWING, "strokesInked": 0})
  assert not nothing.ok and nothing.points == 0


# ---- the evaluators ---------------------------------------------------------


def test_a_drawing_is_scored_on_ink_not_on_intent():
  """Strokes COMMANDED are not strokes drawn. 6 planned and 3 inked is a
  half-finished house, and it must be paid as one."""
  full = evaluate("draw", GOOD_DRAWING)
  half = evaluate("draw", {**GOOD_DRAWING, "strokesInked": 3})
  assert half.ok, "ink landed -- it is a worse drawing, not a failed task"
  assert half.metrics["completeness"] == 0.5
  assert half.points < full.points


def test_a_pen_that_never_lifted_fails_however_good_the_shape_is():
  """Ink laid down while TRAVELLING between strokes is a line across the
  figure that nothing commanded -- a different fault from a wobbly stroke,
  and not one a good form error should be able to buy off."""
  v = evaluate("draw", {**GOOD_DRAWING, "travelInkFraction": 0.6})
  assert not v.ok and v.points == 0
  assert "did not lift" in v.reason


def test_the_census_is_pass_or_fail_against_hidden_truth():
  right = evaluate("census", {"counted": 4, "truth": 4, "coverage": 0.99,
                              "zone": "garden", "vantages": 3})
  wrong = evaluate("census", {"counted": 3, "truth": 4, "coverage": 0.99,
                              "zone": "garden", "vantages": 3})
  assert right.ok and right.points > 0
  assert not wrong.ok and wrong.points == 0


def test_a_census_that_never_happened_does_not_score_as_correct():
  """A use-phase that raised, or never ran because the tool was dropped,
  reports no count. Defaulting the missing pair to zero would make "nothing
  happened" compare EQUAL to "nothing happened" and pay full marks -- the
  exact shape of bug this module exists to prevent."""
  v = scoring.evaluate("census", {"zone": "garden"})
  assert not v.ok and v.points == 0
  assert "no count" in v.reason
  errand = Errand("census:garden", "module_lcd", 0.0, (0, 0))
  crashed = {"error": "RuntimeError: the arm jammed"}
  assert not scoring.score_errand(fake_life(), errand, crashed, {}).ok


def test_a_close_but_wrong_count_earns_nothing_even_at_full_coverage():
  """Coverage scales the payout; it never rescues a wrong answer. A robot
  that surveyed the whole garden and still miscounted did not do the task."""
  v = evaluate("census", {"counted": 5, "truth": 4, "coverage": 1.0,
                          "zone": "garden", "vantages": 4})
  assert not v.ok and v.points == 0


def test_a_dance_that_wandered_off_is_not_a_dance():
  landed = {"moves": 9, "landed": 9, "driftM": 0.1}
  assert evaluate("dance", landed).ok
  assert not evaluate("dance", {**landed, "driftM": 3.0}).ok
  assert not evaluate("dance", {**landed, "landed": 4}).ok, \
    "under half the routine happened"


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


def test_a_carry_errand_needs_both_halves():
  assert evaluate("carry", {"picked": True, "stowed": True,
                            "module": "module_lcd"}).ok
  assert not evaluate("carry", {"picked": True, "stowed": False,
                                "module": "module_lcd"}).ok


# ---- THE HARD RULE ----------------------------------------------------------
# "The evaluator is code, and the LLM never awards itself points." An agent
# that can score its own work learns to declare victory instead of doing the
# task, so these four are the tests that make the rule real rather than
# aspirational.


def test_a_task_cannot_award_points_by_claiming_them():
  """The acceptance criterion. A verdict-SHAPED object is not a verdict."""
  ledger = Ledger()
  claim = SimpleNamespace(task="draw", tier="auto", ok=True, quality=1.0,
                          points=1000, metrics={}, reason="I did great",
                          pending=False, secret=(),
                          public_metrics=lambda: {})
  with pytest.raises(TypeError, match="deterministic evaluator"):
    ledger.award(claim)
  with pytest.raises(TypeError):
    ledger.award({"task": "draw", "points": 1000})
  assert ledger.balance() == 0, "a rejected claim still moved the balance"


def test_a_verdict_cannot_be_built_outside_an_evaluator():
  """Including by laundering a real one through dataclasses.replace: the
  construction token is cleared on the way out, so a copy of a genuine
  verdict is no more constructible than an invented one."""
  with pytest.raises(TypeError, match="scoring.evaluate"):
    Verdict(task="draw", tier="auto", ok=True, quality=1.0, points=1000,
            metrics={}, reason="trust me")
  real = evaluate("draw", GOOD_DRAWING)
  with pytest.raises(TypeError):
    dataclasses.replace(real, points=1000)


def test_the_ledger_re_derives_the_points_and_refuses_a_mismatch():
  """Belt and braces: even a verdict that got past construction is paid from
  the TABLE applied to the EVALUATOR's metrics, not from the number it
  arrived carrying."""
  real = evaluate("draw", GOOD_DRAWING)
  # object.__setattr__ is the only way past a frozen dataclass, which is
  # rather the point: this is what tampering has to look like.
  forged = evaluate("draw", GOOD_DRAWING)
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
  v = evaluate("draw", {**GOOD_DRAWING, "points": 9999, "quality": 1.0,
                        "ok": True, "balance": 9999})
  assert v.points == evaluate("draw", GOOD_DRAWING).points
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


def test_an_errand_cannot_fake_a_drawing_it_did_not_draw():
  """The strongest form of the rule, and the reason `sample_draw` reads the
  BOARD BOOK: an errand's `use` is arbitrary caller-supplied code, so its
  report of itself is a claim. The board is a fact -- the pen writes it while
  tracing -- and a perfect report over a blank board scores zero.
  """
  blank = book()
  errand = Errand(name="draw:whiteboard_a", module="module_pen", station_y=0.0,
                  use_at=(0, 0), detail={"board": "whiteboard_a", "strokes": 6})
  perfect_report = {"drew": True, "strokes": 6, "strokes_drawn": 6,
                    "shape_rms_mm": 0.1, "inked_fraction": 1.0,
                    "travel_ink_fraction": 0.0}
  v = scoring.score_errand(fake_life(boards=blank), errand, perfect_report,
                           before={"board": "whiteboard_a", "strokes": 0,
                                   "clears": 0})
  assert not v.ok and v.points == 0, "a claim about a blank board paid out"
  # ...and with real ink on the board, the same report scores.
  blank.stroke("whiteboard_a", "house", [(0.0, 0.0), (0.02, 0.0), (0.02, 0.02)])
  v2 = scoring.score_errand(fake_life(boards=blank), errand, perfect_report,
                            before={"board": "whiteboard_a", "strokes": 0,
                                    "clears": 0})
  assert v2.ok and v2.points > 0


def test_a_carry_verdict_asks_the_coupling_not_the_errand():
  """`stowed` comes off `module_state` -- the coupling's own answer about
  whether the module is hanging on its bracket."""
  errand = carry_errand()
  lying = {"picked": True, "stowed": True}
  assert not scoring.score_errand(fake_life(module_hung=False), errand,
                                  lying, {}).ok
  assert scoring.score_errand(fake_life(module_hung=True), errand, lying, {}).ok


# ---- what may be published --------------------------------------------------


def test_hidden_ground_truth_never_leaves_the_evaluator():
  """The ledger is streamed to the site AND shown to the overseer as context.
  A census that published its own answer would arrive pre-solved next time,
  which turns a genuinely failable task into a lookup."""
  # Deliberately a WRONG count, so the truth is a number that appears nowhere
  # else: with counted == truth the redaction cannot be told from a
  # coincidence.
  v = evaluate("census", {"counted": 2, "truth": 7, "coverage": 0.99,
                          "zone": "garden", "vantages": 3})
  assert v.metrics["truth"] == 7, "the evaluator still needs the answer"
  assert "truth" not in v.public_metrics() and "error" not in v.public_metrics()
  assert "truth" not in v.as_dict()["metrics"]
  # The reason line is streamed as narration and logged: it may say the
  # answer was wrong, never what the right one was.
  assert "wrong" in v.reason and "7" not in v.reason
  seen: list = []
  ledger = Ledger()
  ledger.on_event.append(seen.append)
  entry = ledger.award(v)
  assert "truth" not in entry["metrics"]
  wire = json.dumps(seen[0])
  assert "truth" not in wire and '"error"' not in wire, \
    "the answer went out on the wire"


# ---- the ledger -------------------------------------------------------------


def test_a_failed_task_is_recorded_at_zero_rather_than_dropped():
  """"It tried and did not manage it" is the most interesting line in a
  robot's log, and a ledger that only records successes reads as a robot
  that never fails."""
  ledger = Ledger()
  ledger.award(evaluate("draw", {**GOOD_DRAWING, "strokesInked": 0}))
  assert ledger.balance() == 0
  assert len(ledger.entries()) == 1 and ledger.entries()[0]["ok"] is False


def test_the_ledger_survives_a_restart(tmp_path):
  """The acceptance criterion: points persist with the rest of world state.
  Every mission end is a restart, so a balance that lives only in memory is
  a balance that is always zero."""
  path = tmp_path / "ledger.json"
  first = Ledger(path=path)
  first.award(evaluate("draw", GOOD_DRAWING), t=12.0)
  first.award(evaluate("carry", {"picked": True, "stowed": True,
                                 "module": "module_lcd"}), t=30.0)
  banked = first.balance()
  assert banked > 0

  second = Ledger(path=path)
  assert second.balance() == banked
  assert [e["task"] for e in second.entries()] == ["draw", "carry"]
  # ...and it keeps counting from there rather than starting over
  second.award(evaluate("charge", {"startFrac": 0.2, "endFrac": 0.95,
                                   "gainedWh": 0.4, "seconds": 100}))
  assert second.balance() > banked
  assert second.entries()[-1]["seq"] == 3


def test_the_state_file_is_written_on_every_award_not_at_shutdown(tmp_path):
  """A balance that survives only a CLEAN shutdown does not survive the thing
  restarts are usually about."""
  path = tmp_path / "ledger.json"
  ledger = Ledger(path=path)
  ledger.award(evaluate("draw", GOOD_DRAWING))
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
    ledger.award(evaluate("carry", {"picked": True, "stowed": True,
                                    "module": "module_lcd"}))
  assert len(ledger.entries()) == MAX_ENTRIES
  assert ledger.robots["pluggybot"]["dropped"] == 20
  assert ledger.balance() == (MAX_ENTRIES + 20) * TABLE["carry"].base
  assert ledger.entries()[-1]["seq"] == MAX_ENTRIES + 20


def test_the_telemetry_block_carries_the_balance_and_the_last_few_earnings():
  """The streaming half of the acceptance criteria. `recent` is what catches
  a late-joining browser up without a snapshot message of its own -- it rides
  in every keyframe, which is why it is capped and compact."""
  ledger = Ledger()
  for _ in range(RECENT + 3):
    ledger.award(evaluate("carry", {"picked": True, "stowed": True,
                                    "module": "module_lcd"}), t=1.0)
  snap = ledger.snapshot()["pluggybot"]
  assert snap["balance"] == ledger.balance()
  assert snap["tasks"] == RECENT + 3
  assert len(snap["recent"]) == RECENT
  assert set(snap["recent"][0]) == {"seq", "task", "points", "ok", "t"}
  json.dumps(snap)                      # it goes on a websocket


def test_an_award_is_an_event_as_it_happens():
  seen: list = []
  ledger = Ledger()
  ledger.on_event.append(seen.append)
  ledger.award(evaluate("draw", GOOD_DRAWING), t=42.5)
  assert len(seen) == 1
  msg = seen[0]
  assert msg["type"] == "earned" and msg["t"] == 42.5
  assert msg["task"] == "draw" and msg["points"] > 0
  assert msg["balance"] == ledger.balance() and msg["reason"]


# ---- the deferred (visitor-judged) slot -------------------------------------


def test_a_visitor_judged_task_banks_nothing_until_it_is_rated():
  """Tier 3 of the design doc's four. Code confirms the work HAPPENED; the
  aesthetic call arrives later over the inbound channel (issue #16), and the
  reward table -- not the rater, and not the robot -- turns the rating into
  points."""
  ledger = Ledger()
  v = evaluate("artwork", GOOD_DRAWING)
  assert v.tier == "visitor" and v.pending and v.points == 0
  entry = ledger.award(v, t=5.0)
  assert ledger.balance() == 0 and ledger.pending()[0]["seq"] == entry["seq"]

  seen: list = []
  ledger.on_event.append(seen.append)
  settled = ledger.settle(entry["seq"], quality=0.75, t=9.0)
  assert settled["points"] == TABLE["artwork"].points(True, 0.75)
  assert ledger.balance() == settled["points"] and not ledger.pending()
  # The message is stamped when the RATING landed, not when the work was
  # done -- `seq` is what ties it back to the entry it settles.
  assert seen[-1]["settled"] is True and seen[-1]["t"] == 9.0
  assert seen[-1]["seq"] == entry["seq"]


def test_a_rating_cannot_be_applied_twice_or_out_of_range():
  ledger = Ledger()
  entry = ledger.award(evaluate("artwork", GOOD_DRAWING))
  with pytest.raises(ValueError, match="0..1"):
    ledger.settle(entry["seq"], quality=50.0)
  ledger.settle(entry["seq"], quality=1.0)
  with pytest.raises(ValueError, match="not pending"):
    ledger.settle(entry["seq"], quality=1.0)
  assert ledger.balance() == TABLE["artwork"].points(True, 1.0)


# ---- the errand -> task wiring ----------------------------------------------


def test_every_errand_the_menu_builds_names_a_task_that_can_be_scored():
  """An errand's `task` is what selects its evaluator, and it defaults off
  the errand NAME -- so a renamed errand silently stops being scored. This is
  the guard on that."""
  from pluggybot.lifecycle import errands_for
  for kind in ("carry", "dance"):
    for errand in errands_for(kind, "room_hub"):
      assert errand.task in scoring.EVALUATORS, \
        f"{kind}: errand {errand.name!r} has unscoreable task {errand.task!r}"
  assert dance_errand((0, 0)).task == "dance"
  assert Errand("census:garden", "module_lcd", 0.0, (0, 0)).task == "census"


def test_an_unscored_errand_is_not_an_error():
  """"Narrative only" is one of the four tiers. A hand-built errand with no
  evaluator produces no verdict and no complaint -- what is refused is a task
  that PAYS with nothing to judge it (see the hard-rule tests)."""
  errand = Errand("wander", "module_lcd", 0.0, (0, 0), task="wander")
  assert scoring.score_errand(fake_life(), errand, {}, {}) is None


def test_a_second_drawing_is_not_scored_on_the_first_ones_ink():
  """The reason `board_before` exists. Erasing is the default, not a
  guarantee: a `--no-erase` second cycle over the same board would otherwise
  inherit the previous figure's strokes and score them again."""
  b = book()
  b.stroke("whiteboard_a", "house", [(0.0, 0.0), (0.03, 0.0)])
  before = {"board": "whiteboard_a", "strokes": b["whiteboard_a"].strokes,
            "clears": 0}
  errand = Errand(name="draw:whiteboard_a", module="module_pen", station_y=0.0,
                  use_at=(0, 0), detail={"board": "whiteboard_a", "strokes": 2})
  v = scoring.score_errand(fake_life(boards=b), errand,
                           {"strokes": 2, "shape_rms_mm": 0.8}, before)
  assert not v.ok, "the previous drawing's strokes were counted as this one's"


def test_a_charge_verdict_is_measured_off_the_battery():
  life = fake_life(time=140.0,
                   battery=SimpleNamespace(fraction=0.93, energy_wh=0.65))
  v = scoring.score_charge(life, {"t": 20.0, "frac": 0.2, "wh": 0.14})
  assert v.ok and v.task == "charge"
  assert v.metrics["gainedWh"] == pytest.approx(0.51, abs=1e-6)
  assert v.metrics["seconds"] == 120.0
  assert math.isclose(v.metrics["endFrac"], 0.93)
