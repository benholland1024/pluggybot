"""`scripts/determinism_spike.py`'s restart check read without flying it
(issue #780): a restore begins the day loop again at its top, so whatever the
loop was inside -- its explore's walk, a walk off the rack, the day's start --
ends there by design (Webserver.md: the errand in flight ends), and the day
after differs from the one flown straight through. `--resume-at T --in-step`
saves only at a step with nothing in flight, and says what it passed over.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import determinism_spike as ds  # noqa: E402
from test_body import stub_life  # noqa: E402 -- tests/ is on sys.path


def _flown(life):
  """The stub's day hooked as the spike hooks its saving arm (`ds.Flight`),
  `clean` asked at every step: its spans of steps, each (first t, last t,
  what was in flight, whether it was clean)."""
  flight = ds.Flight(life)
  seen = []
  d = life.data
  life.body.last_hooks.append(lambda: seen.append(
    (round(float(d.time), 3), tuple(flight.ending()), flight.clean(float(d.time)))))
  life.run(start=(2.5, 3.0, 0.0), max_sim_time=6.0)
  spans = []
  for t, names, clean in seen:
    if spans and spans[-1][2:] == [names, clean]:
      spans[-1][1] = t
    else:
      spans.append([t, t, names, clean])
  return flight, spans


def test_a_restart_ends_what_the_day_loop_is_inside_and_the_check_saves_past_it():
  """A day on the stub: its start's look round, an explore whose one walk
  (the stub's arrives at once: here it holds 0.1 s) ends at the charging
  station, the explore's look round, then the stand-by -- a walk off the
  station, then holding. Every step but the stand-by's hold has something a
  restart ends in flight, the loop's own routine first: the save passes over
  each, and counts them."""
  life = stub_life()
  life.expects_work = True
  body = life.body
  station = body.grid.world_to_cell(0.0, 0.0)
  plans = iter([([(station[0] + 40, station[1]), station], "ok")])
  body.plan_frontier = lambda blacklist: next(plans, (None, "no-frontiers"))

  def walk(x, y, timeout=90.0, stop=None):
    yield from body.hold_routine(0.1)
    body.x, body.y = float(x), float(y)
    return True
  body.go_to_routine = walk
  body.look_around_routine = lambda: body.hold_routine(0.05)
  flight, spans = _flown(life)
  assert [(names, clean) for _, _, names, clean in spans] == [
    (("the day's start",), False),
    (("explore_routine", "walk"), False),
    (("explore_routine", "hold_routine"), False),         # ...its look round
    (("_clear_rack_routine", "walk"), False),             # ...off the station
    ((), True)], spans
  assert spans[-1][1] - spans[-1][0] > 5.0, "the stand-by holds, nothing in flight"
  passed = sum(round((b - a) / 0.002) + 1 for a, b, _, clean in spans if not clean)
  assert flight.passed == passed
  assert flight.first == {"t": spans[0][0], "inFlight": ["the day's start"]}


def test_the_check_says_where_it_saved_and_what_a_restart_would_have_ended():
  """The line the check prints (`saved_words`): where it saved, and from
  where it passed over how many steps, the first with what in flight -- or
  that it saved nowhere, and why: what was in flight, or no step the keeper
  saves in, or (at the pass) no pass."""
  first = {"t": 200.088, "inFlight": ["explore_routine", "drive_to_routine"]}
  assert ds.saved_words({"k": "saved", "t": 257.488, "passedOver": 573,
                         "firstInFlight": first}, True, 200.0) == (
    "saved inside the hooks of the step at t=257.488, the first with nothing in "
    "flight: from t=200.088 it passed over 573 steps, the first with "
    "explore_routine > drive_to_routine in flight, which a restart ends by design "
    "(the errand in flight ends, #780)")
  assert ds.saved_words({"k": "saved", "t": 300.088, "passedOver": 0,
                         "firstInFlight": None}, True, 300.0) == (
    "saved inside the hooks of the step at t=300.088")
  assert ds.saved_words({"k": "saved", "t": 257.434}, False, 60.0) == (
    "saved at the day loop's pass at t=257.434")
  assert ds.saved_words({"k": "nosave", "passedOver": 9, "firstInFlight": first},
                        True, 200.0).startswith(
    "NOT SAVED: no step from t=200.0 to the day's end had nothing in flight; from "
    "t=200.088 it passed over 9 steps, the first with explore_routine > drive_to_routine")
  assert ds.saved_words({"k": "nosave", "passedOver": 0, "firstInFlight": None}, True, 200.0) == (
    "NOT SAVED: no step from t=200.0 to the day's end was one the keeper saves in "
    "(owed, its scan due, nothing mid-move)")
  assert ds.saved_words({"k": "nosave", "passedOver": 0, "firstInFlight": None}, False, 400.0) == (
    "NOT SAVED: the day loop never passed its top from t=400.0 to the day's end")


def test_a_divergence_from_a_save_in_a_step_where_the_days_are_in_different_states_names_the_limit():
  """The check's last line (`verdict`): a save in a step lands in the
  stand-by, which a restored loop holds from a fresh 5 s, so two days in
  different states at their first difference are the loop's next branch
  taken at another instant (#780's limit, flown on the demo pack: a charge
  begun at 280.088 where the straight day began it at 282.434). In the same
  state, or from a save at the pass, a divergence is said plainly."""
  base = {"diverged": True, "lastSameT": 280.0, "firstDiffT": 280.5}
  line, code = ds.verdict({**base, "stateA": "EXPLORE", "stateB": "GO_CHARGE"}, None, True)
  assert code == 1 and line == (
    "DIVERGED: same until t=280.0, differs at 280.5 (states EXPLORE / GO_CHARGE) -- the "
    "days in different states there: from a save in the stand-by, its next branch begun "
    "at another instant, as a restored stand-by holds a fresh 5 s (#780's limit)")
  plain = "DIVERGED: same until t=280.0, differs at 280.5 (states EXPLORE / "
  assert ds.verdict({**base, "stateA": "EXPLORE", "stateB": "EXPLORE"}, None, True) == (
    plain + "EXPLORE)", 1)
  assert ds.verdict({**base, "stateA": "EXPLORE", "stateB": "GO_CHARGE"}, None, False) == (
    plain + "GO_CHARGE)", 1)
  same = {"diverged": False, "stepsCompared": 240}
  assert ds.verdict(same, None, True) == (
    "IDENTICAL after the restore over 240 state samples and every look and scan", 0)
  assert ds.verdict(same, {"kind": "lidar", "t": 300.188}, True)[1] == 1


def test_what_the_loop_is_inside_is_read_only_through_its_hand_driver():
  """`in_flight` walks into a routine's local `routine` only where the
  routine is the loop's hand driver (`HAND_DRIVEN`): a body's routine that
  keeps one of the loop's in a local of that name is not walked into. And a
  hand driver of the loop's whose routine cannot be read is in flight under
  its own name: never none."""
  class Body:
    def hold(self, routine):
      while True:
        yield ()

  class Loop:
    def explore(self):
      yield ()

    def _until_stood_up_routine(self, other):
      kept = other                        # ...in a local of another name
      while kept:
        yield ()

    def day(self, body, wrapped):
      if wrapped:
        yield from self._until_stood_up_routine(self.explore())
      yield from body.hold(self.explore())

  life, body = Loop(), Body()
  held = life.day(body, False)
  next(held)
  assert ds.in_flight(life, held) == [], "the stand-by's hold: nothing in flight"
  wrapping = life.day(body, True)
  next(wrapping)
  assert ds.in_flight(life, wrapping) == ["_until_stood_up_routine"]


def test_the_saving_arm_asks_before_it_writes():
  """The one line of wiring in the flown child (`child`'s `save_here`): a
  save inside a step asks `Flight.clean` before the world is written, and
  goes no further where it says no."""
  import ast
  import inspect
  tree = ast.parse(inspect.getsource(ds.child))
  save = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "save_here")
  calls = {ast.unparse(n.func): n.lineno for n in ast.walk(save) if isinstance(n, ast.Call)}
  assert calls["flight.clean"] < calls["continuation.write"]
  gate = next(n for n in ast.walk(save) if isinstance(n, ast.If)
              and "flight.clean" in ast.unparse(n.test))
  test = gate.test
  assert isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And) and len(test.values) == 2
  assert [ast.unparse(v) for v in test.values][0] == "in_step"
  assert isinstance(test.values[1], ast.UnaryOp) and isinstance(test.values[1].op, ast.Not)
  assert isinstance(gate.body[0], ast.Return)
