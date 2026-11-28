"""A procedure's carry asked a turn arrives only once its turn is there too
(issue #767): a `move_hand` that turned the hand with little or no move of
its place read "arrived" once its place was within `reach.ARRIVE_M`, its
command still turning, and whatever ran next kept that half-made turn. Its
command now arrives once its place AND its turn are there -- the held-back
clock starts then -- and it reads "arrived" once the hand is within
`reach.ARRIVE_ROT` of that turn too, beyond the turn its solve misses it by
and the sag it carried in, else "held back", saying which held it back; the
loop's own carries and every carry that keeps its turn are as they were. One flight on the bare floor (#767's own, about 5 s of sim), the rest
direct calls."""

import math

import mujoco
import numpy as np
import pytest

from pluggybot import lifecycle as lc
from pluggybot.body import StubBody, body_for
from pluggybot.humanoid import model as hm
from pluggybot.humanoid import reach as rc
from pluggybot.humanoid.arms import HandGoal, _rot_error
from pluggybot.procedure import lang
from pluggybot.procedure import steps as st

from test_hands import _bare, _bare_reach, _fly, _Life, _turned

#: #767's own flight: a reach, then a turn on the spot of 7.5 deg -- its
#: place read where the hand is -- then a move 3 cm back that keeps its turn.
REACH = 'reach("left", 0.35, 0.25, 1.0, 0.5, 0, -0.866, 0, -1, 0)'
ASKED = (0.414, -0.078, -0.907)
TURN = ('move_hand("left", read("left.x"), read("left.y"), read("left.z"), '
        'fx=0.414, fy=-0.078, fz=-0.907, jx=0.004, jy=-0.996, jz=0.087)')
BACK = 'move_hand("left", read("left.x") - 0.03, read("left.y"), read("left.z"))'


def _fingers(model, data, body):
  """The grip's fingers' way, world frame, off the simulator: the test's
  alone (the robot reads its own model of itself)."""
  i = model.site(body.handle.el("left_grip")).id
  return -data.site_xmat[i].reshape(3, 3)[:, 2]


def _deg(a, b) -> float:
  a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
  return math.degrees(math.acos(max(-1.0, min(1.0, float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))))))


def test_a_turn_on_the_spot_arrives_only_once_turned_and_the_next_move_keeps_all_of_it():
  """#767's flight: the turn read "arrived" 0.11 s in, its command 1.2 deg
  short of the turn asked and its fingers 4.8 deg into 7.3 -- and the move
  after it, keeping its turn, kept that command's: it ended 1.2 deg off the
  way asked. Now its command is the turn asked as it arrives, no sooner than
  its turn takes (`reach.HAND_TURN`), and the next move keeps all of it."""
  model = mujoco.MjModel.from_xml_string(hm.body_xml())
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  mujoco.mj_forward(model, data)
  body = body_for(model, data)
  life, facts = _Life(body), st.hand_facts()

  def run(call):
    proc = lang.compile_procedure("def a():\n  " + call + "\n", facts)
    return body.run(lang.run_procedure_routine(life, proc, facts))
  try:
    assert run(REACH)["ok"]
    arms = body.mission.reach
    kept = arms.cmd["left"].rot.copy()
    turned = run(TURN)
    step = turned["steps"][-1]
    assert turned["ok"] and step["why"] == "arrived", step
    # ...its command the turn asked, whole, and no sooner than that turn takes
    asked = arms.goal["left"].rot
    whole = float(np.linalg.norm(_rot_error(asked, kept)))
    assert whole > math.radians(7.0), "the premise: a turn of 7.5 deg"
    assert np.array_equal(arms.cmd["left"].rot, asked), "its command still turning"
    assert step["seconds"] >= whole / rc.HAND_TURN - 0.01, step
    # ...and the next move, keeping its turn, ends with the fingers as asked
    assert run(BACK)["ok"]
    assert _deg(_fingers(model, data, body), ASKED) < 0.8
  finally:
    body.close()


def _carrying(r, asked, turns=True, verb=True, hand=None, miss=0.0):
  """`r`'s left hand out where it is held, its command unturned and the hand
  turned `hand` as the carry begins (its command's, by default), then
  carried nowhere -- its place there at once -- turned to `asked` where
  `turns`, by a procedure's verb where `verb`; its arm's solve reaching
  whatever it is asked to within `miss` of its turn, rad."""
  h = np.array([0.35, 0.2, 0.9])
  r.cmd["left"] = HandGoal(h.copy(), np.eye(3), 1.0)
  r.goal["left"] = HandGoal(h.copy(), np.eye(3), 1.0)
  r.phase["left"] = "hold"
  r.grip_now = lambda side: (h.copy(), np.eye(3) if hand is None else hand)
  r.posture_for = lambda side, pos, rot=None, **kw: (0.0, 0.0, 0.0)
  r.path_for = lambda side, poses, rot=None, **kw: (0.0, 0.0, 0.0)
  r.self_model.solve = lambda side, pos, rot=None, iters=40: (np.zeros(len(hm.ARM)), 0.0, miss)
  assert r.carry("left", h, **({"path": [], "turns": [asked]} if turns else {}), verb=verb)
  return h


def _ticks(r, h, hand, most=400):
  """The carry's loop ticked (`Reach._held`) with the hand at its place and
  turned `hand(r)` by its senses, until it ends: how many ticks it took."""
  for k in range(1, most + 1):
    r._held("left", rc.EVERY_S, h, hand(r))
    if r.phase["left"] != "carry":
      return k
  return None


def _following(r):
  """A hand turned as its command is."""
  return r.cmd["left"].rot.copy()


@pytest.mark.parametrize("deg", [7.0, 80.0])
def test_a_turned_carry_arrives_on_the_tick_its_command_is_turned_and_not_before(deg):
  """Its hand following its command: arrived on the tick the command's turn
  gets there, not on its place's first one -- and a turn longer than
  `HELD_BACK_S` takes at `HAND_TURN` arrives too, never "held back" while
  its command still turns: the clock starts once the command is there."""
  r = _bare_reach()
  asked = _turned(deg)
  h = _carrying(r, asked)
  took = _ticks(r, h, _following)
  turning = math.ceil(math.radians(deg) / (rc.HAND_TURN * rc.EVERY_S) - 1e-9)
  assert took == turning and r.report["left"]["why"] == "arrived", (took, r.report["left"])
  if deg > 60.0:
    assert turning * rc.EVERY_S > rc.HELD_BACK_S, "the premise: a turn longer than the clock"


def test_a_turned_carry_whose_hand_is_held_off_its_turn_is_held_back_by_its_turn():
  """Its command turned but its hand held 0.15 rad off the turn asked --
  what it holds, or its arm's own reach -- it is held back `HELD_BACK_S`
  after its command got there, its report saying the turn held it and how
  far; 0.08 rad off, under the solve's own 0.1 (`arms.REACH_ROT_TOL`), it
  arrived: as the table's plate sat 0.03-0.04 rad off its turn at rest."""
  r = _bare_reach()
  asked = _turned(7.0)
  off = asked @ _turned(math.degrees(0.15)).T
  h = _carrying(r, asked)
  took = _ticks(r, h, lambda r: off)
  turning = math.ceil(math.radians(7.0) / (rc.HAND_TURN * rc.EVERY_S) - 1e-9)
  rep = r.report["left"]
  assert rep["why"] == "held back" and rep["heldBy"] == "turn", rep
  clock, n = 0.0, 0
  while clock < rc.HELD_BACK_S:
    clock, n = clock + rc.EVERY_S, n + 1
  assert took == turning - 1 + n, took
  assert rep["turnRad"] == pytest.approx(0.15, abs=1e-4)
  near = asked @ _turned(math.degrees(0.08)).T
  r = _bare_reach()
  h = _carrying(r, asked)
  assert _ticks(r, h, lambda r: near) == turning and r.report["left"]["why"] == "arrived"


def test_a_turn_its_solve_misses_arrives_once_the_hand_is_where_its_arm_turns_it():
  """A carry goes on with its solve up to `HOLD_ROT_TOL` off its turn (#493's
  lid arc at the wrist's end): its hand, where its arm turns it -- 0.15 rad
  off the turn asked -- arrives on the tick its command's turn gets there,
  never held back on the solve's own miss."""
  r = _bare_reach()
  asked = _turned(7.0)
  assert 0.15 < rc.HOLD_ROT_TOL, "the premise: the carry goes on"
  where = asked @ _turned(math.degrees(0.15)).T
  h = _carrying(r, asked, miss=0.15)
  turning = math.ceil(math.radians(7.0) / (rc.HAND_TURN * rc.EVERY_S) - 1e-9)
  assert _ticks(r, h, lambda r: where) == turning and r.report["left"]["why"] == "arrived"


def test_a_move_that_restates_its_turn_arrives_with_the_sag_it_carried_in():
  """A move given the turn it keeps -- written out again -- by a hand a load
  holds 0.12 rad off it as it begins arrives on its place as a move given
  none does, its hand still that far off; one asked a turn arrives once
  turned as far as asked, its sag kept."""
  r = _bare_reach()
  sagged = _turned(math.degrees(0.12))
  h = _carrying(r, np.eye(3), hand=sagged)
  assert _ticks(r, h, lambda r: sagged) == 1 and r.report["left"]["why"] == "arrived"
  r = _bare_reach()
  asked = _turned(7.0)
  h = _carrying(r, asked, hand=sagged)
  turning = math.ceil(math.radians(7.0) / (rc.HAND_TURN * rc.EVERY_S) - 1e-9)
  assert _ticks(r, h, lambda r: asked @ sagged) == turning and r.report["left"]["why"] == "arrived"


def test_a_stuck_turn_is_held_back_past_its_allowances():
  """Past the solve's miss and the sag carried in, a hand held off its turn
  is still held back by it: with 0.03 rad of each, one held 0.25 rad off."""
  r = _bare_reach()
  asked = _turned(7.0)
  h = _carrying(r, asked, hand=_turned(math.degrees(0.03)), miss=0.03)
  _ticks(r, h, lambda r: asked @ _turned(math.degrees(0.25)).T)
  rep = r.report["left"]
  assert rep["why"] == "held back" and rep["heldBy"] == "turn", rep
  assert rep["turnRad"] == pytest.approx(0.25, abs=1e-4)


def test_a_turned_carry_held_short_of_its_place_says_its_place():
  r = _bare_reach()
  asked = _turned(7.0)
  h = _carrying(r, asked)
  short = h + np.array([0.0, 0.0, -2.0 * rc.ARRIVE_M])
  for _ in range(400):
    r._held("left", rc.EVERY_S, short, _following(r))
    if r.phase["left"] != "carry":
      break
  assert r.report["left"]["why"] == "held back" and r.report["left"]["heldBy"] == "place"


def test_a_turned_paths_points_go_on_their_place_and_its_end_waits_for_its_turn():
  """A path's points are left on their place as before, their turns still
  coming; its end arrives on the tick its command's turn gets there, the
  last point's, and not before."""
  r = _bare_reach()
  h = np.array([0.35, 0.2, 0.9])
  r.cmd["left"] = HandGoal(h.copy(), np.eye(3), 1.0)
  r.goal["left"] = HandGoal(h.copy(), np.eye(3), 1.0)
  r.phase["left"] = "hold"
  r.grip_now = lambda side: (h.copy(), np.eye(3))
  r.path_for = lambda side, poses, rot=None, **kw: (0.0, 0.0, 0.0)
  r.self_model.solve = lambda side, pos, rot=None, iters=40: (np.zeros(len(hm.ARM)), 0.0, 0.0)
  first, last = _turned(20.0), _turned(-10.0)
  assert r.carry("left", h, path=[h.copy()], turns=[first, last], verb=True)
  r._held("left", rc.EVERY_S, h, _following(r))
  assert r.phase["left"] == "carry" and np.array_equal(r.goal["left"].rot, last), "left on its place"
  ticks = 1
  while r.phase["left"] == "carry" and ticks < 400:
    was = r.cmd["left"].rot.copy()
    r._held("left", rc.EVERY_S, h, _following(r))
    ticks += 1
  assert r.report["left"]["why"] == "arrived" and not np.array_equal(was, last), was
  assert np.array_equal(r.cmd["left"].rot, last)


def test_a_turned_carry_stopped_on_its_way_stops_where_it_is_as_before():
  """A stop condition met mid-turn (`Reach.halt`) ends it at once, held at
  the turn its command had got to, as any carry's stop."""
  r = _bare_reach()
  asked = _turned(30.0)
  h = _carrying(r, asked)
  for _ in range(5):
    r._held("left", rc.EVERY_S, h, _following(r))
  partial = r.cmd["left"].rot.copy()
  assert r.halt("left", "until") and r.report["left"]["why"] == "until"
  assert np.array_equal(r.cmd["left"].rot, partial) and r.phase["left"] == "hold"


@pytest.mark.parametrize("turns, verb", [(False, True), (True, False)])
def test_a_carry_that_keeps_its_turn_or_the_loops_own_arrives_on_its_place_as_before(turns, verb):
  """A procedure's carry asked no turn, and the loop's own asked one (a
  charge's plug carried to the port), arrive as they always did: on their
  place's first tick, whatever the hand's turn -- and say nothing of it."""
  r = _bare_reach()
  asked = _turned(7.0)
  off = _turned(math.degrees(0.3))
  h = _carrying(r, asked, turns=turns, verb=verb, hand=off)
  assert _ticks(r, h, lambda r: off) == 1 and r.report["left"]["why"] == "arrived"
  assert "turnRad" not in r.report["left"] and "heldBy" not in r.report["left"]


class _Ends(StubBody):
  """A stub whose every carry ends as a test says."""

  def __init__(self, *a, **kw):
    super().__init__(*a, **kw)
    self.ends = []

  def move_hand_routine(self, side, pos, stiffness=1.0, speed=0.06, patience=20.0, path=None,
                        turns=None, verb=False):
    return {"seconds": 1.2, "moved": 0.0, "dx": 0.0, "dy": 0.0, "dz": 0.0, **self.ends.pop(0)}
    yield


def test_a_move_held_back_by_its_turn_says_how_far_short_its_turn_was():
  """Held back by its turn, a `move_hand` reads false saying it got there
  and how far short of the turn asked it turned, in History's words too;
  held back by its place, as it always did."""
  body = _Ends(*_bare())
  body.ends = [{"why": "held back", "heldBy": "turn", "turnRad": 0.1745, "shortM": 0.004},
               {"why": "held back", "heldBy": "place", "turnRad": 0.01, "shortM": 0.05}]
  said = "your left hand got there, but ended 10 deg off the turn asked"
  assert st.MOVE_TURN_SHORT.format(hand="left", deg=10.0) == said
  ran = _fly('def a():\n    reach("left", 0.3, 0.2, 0.9)\n    n = 0\n'
             '    if not move_hand("left", 0.3, 0.2, 0.9, fx=1, fy=0, fz=0, jx=0, jy=1, jz=0):\n'
             '        n = 1\n'
             '    move_hand("left", 0.3, 0.2, 0.95, fx=1, fy=0, fz=0, jx=0, jy=1, jz=0)\n', body)
  tested, failed = ran["steps"][-2], ran["steps"][-1]
  assert (tested["ok"], tested["reason"]) == (False, said), tested
  assert failed["reason"] == st.MOVE_WHY["held back"].format(hand="left"), failed
  [history] = lc.procedure_outcome("a", ran)
  assert f"1 test read false -- line 4, move_hand: {said}" in history, history
