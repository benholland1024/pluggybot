"""A way out from a hand already out keeps off the robot's own trunk (issue
#722; docs/Hands.md, SimNotes "A way out from a hand already out"), each rule
pinned without a flight -- the robot's own drawing and a bare reach
(docs/Testing.md). The flights that measured it are the SimNotes section's.

  1. a way out from a hand already out is judged against the trunk above the
     waist on the robot's model of itself, its end included: where it keeps
     `OWN_CLEAR_M` -- or, begun nearer, goes no nearer -- today's way is
     taken, untouched; a way from a hand at rest is never judged so;
  2. else the side way or the way drawn in that keeps so and dips no deeper
     over the work;
  3. else a verb's reach is refused and nothing moves, `own` saying which
     part it would have met; the loop's own reach takes the way that comes
     least near it;
  4. MuJoCo 3.10's box against a box that reads 0.0 is read again;
  5. the body's why is "own body", and the verb says so in words.
"""

from types import SimpleNamespace as NS

import mujoco
import numpy as np
import pytest

from pluggybot.body import StubBody
from pluggybot.humanoid import body as hb
from pluggybot.humanoid import model as hm
from pluggybot.humanoid import reach as rc
from pluggybot.humanoid.arms import HandGoal, SelfModel
from pluggybot.procedure import steps as st

from test_hands import _bare, _fly, _standing_reach

#: The 11-15 wake's t_0360, Luca's door_run31 flown at its live draw on
#: ac2538e8 (R5's rows): its left hand held at the cupboard's lever reached
#: again from there, and its straight way out ran the gripper 0.9 cm into
#: the torso (45.8 N on the truth).
AT_LEVER = np.array([0.1719, 0.1105, -0.2825, 2.0083, -0.1801, -0.3687])
LEVER_AGAIN = np.array([0.6363, 0.3043, -0.3642, 1.9979, -0.249, -0.3687])
#: ...and t_0408's, on the same tree: its hand out at the lever reaching on,
#: its forearm 1.2 cm from the torso as it begins and no nearer on the way.
ON_AT = np.array([0.3511, 0.0507, -0.1784, 1.3621, 0.8903, 0.0169])
ON_TO = np.array([0.6401, 0.123, -0.1261, 1.4637, 1.1071, 0.0169])
#: ...and our cube's at set-out 7 (`hands_check.PLACE`, a pass), from its
#: arm hanging: its plan's pre-grasp lies 3.5 mm into the head's back on the
#: robot's model of itself, and no part of the arm met it.
CUBE_7_PRE = np.array([-1.2369, 2.765, 0.5361, 1.8645, 1.0471, -1.121])
#: An arm raised behind and a pre-grasp low across the front (drawn):
#: straight, 4.4 cm through the trunk; the side way keeps 4.4 cm clear of it
#: but swings the arm 13.7 cm lower than it ends ahead of the robot; the way
#: drawn in, 6.2 cm through.
DIPS_FROM = np.array([-2.8, 0.36, 1.77, 0.49, 0.41, 0.83])
DIPS_TO = np.array([-2.93, 0.39, -1.93, 1.51, 0.36, 1.19])
#: An arm out in front and a pre-grasp beyond its other side (drawn on the
#: robot's own drawing): straight, the forearm passes 1.3 cm through the
#: torso; the side way keeps 7.5 cm clear of it.
ACROSS_FROM = np.array([-0.6, 0.9, -1.34, 1.14, 0.42, -0.53])
ACROSS_TO = np.array([1.37, 0.67, -1.74, 1.61, 1.33, 1.16])


def _own_room(q0, ways):
  return rc.own_room(SelfModel(), "left", q0, ways)[0]


# ---- 1-3: the way out judged, on the robot's own drawing ------------------------------------


def test_a_way_out_that_keeps_clear_of_the_trunk_is_todays_way():
  """Where today's way keeps `OWN_CLEAR_M` -- t_0408's left hand out at the
  lever reaching on, straight, 1.2 cm at its nearest -- the way is today's,
  and nothing is said."""
  r = _standing_reach()
  r.q_cmd["left"] = ON_AT.copy()
  assert _own_room(ON_AT, [ON_TO]) >= rc.OWN_CLEAR_M, "the premise: it keeps clear"
  way = r.way_out("left", ON_TO)
  assert way is not None and rc._same_way(way, [ON_TO]) and r.own["left"] is None


def test_a_way_out_from_a_hand_at_rest_is_never_judged_against_the_trunk():
  """From the arm hanging the way is today's, judged against nothing new,
  though our cube's plan at set-out 7 puts its pre-grasp in the head on the
  robot's model of itself: refused there, the pass was lost (the plan's to
  keep clear, SimNotes)."""
  assert _own_room(rc.seed("left", rc.SEEDS[0]), [CUBE_7_PRE]) < 0.0, "the premise: it ends in the head"
  r = _standing_reach()
  rest = r.rest["left"].copy()
  way = r.way_out("left", CUBE_7_PRE)
  sm = r._planner_model()
  r._pose_planner(sm, None)
  today, _ = r._picked(sm, "left", rest, CUBE_7_PRE, True)
  assert way is not None and rc._same_way(way, today) and r.own["left"] is None


def test_a_hand_out_whose_every_way_out_passes_through_the_trunk_is_refused():
  """t_0360's reach from the lever: the arm's pre-grasp puts the gripper in
  the torso, so no way there keeps clear -- a verb's reach is refused,
  nothing moves, and `own` says the torso and how far in."""
  assert _own_room(AT_LEVER, [LEVER_AGAIN]) < 0.0, "the premise: straight goes through"
  r = _standing_reach()
  r.q_cmd["left"], r.phase["left"] = AT_LEVER.copy(), "hold"
  assert r.way_out("left", LEVER_AGAIN) is None
  assert r.own["left"]["part"] == "torso" and r.own["left"]["m"] < 0.0
  # ...and the loop's own (a charge's) takes the way that comes least near
  r.own["left"] = None
  way = r.way_out("left", LEVER_AGAIN, refuse=False)
  assert way is not None and r.own["left"] is not None
  rooms = [_own_room(AT_LEVER, w) for w in ([LEVER_AGAIN], rc._aside("left", AT_LEVER, LEVER_AGAIN, True),
                                            rc._tucked_way("left", AT_LEVER, LEVER_AGAIN))]
  assert _own_room(AT_LEVER, way) == pytest.approx(max(rooms))


def test_a_hand_out_takes_the_side_way_where_straight_passes_through_the_trunk():
  assert _own_room(ACROSS_FROM, [ACROSS_TO]) < 0.0, "the premise: straight goes through"
  r = _standing_reach()
  r.q_cmd["left"] = ACROSS_FROM.copy()
  way = r.way_out("left", ACROSS_TO)
  assert rc._same_way(way, rc._aside("left", ACROSS_FROM, ACROSS_TO, True))
  assert _own_room(ACROSS_FROM, way) >= rc.OWN_CLEAR_M and r.own["left"] is None


def test_a_side_way_that_dips_deeper_over_the_work_is_not_taken():
  """#539's and #560's rule kept whole, as `_kept_off` keeps it: the side way
  keeps clear of the trunk but dips deeper ahead than going straight on, so
  with the way drawn in through the trunk too, the reach is refused."""
  r = _standing_reach()
  sm = r._planner_model()
  r._pose_planner(sm, None)
  dips = rc._dips(sm, "left", DIPS_FROM, DIPS_TO, rc.front(sm, "left", r.rest["left"]))
  aside = rc._aside("left", DIPS_FROM, DIPS_TO, True)
  assert _own_room(DIPS_FROM, [DIPS_TO]) < 0.0 and _own_room(DIPS_FROM, aside) >= rc.OWN_CLEAR_M
  assert dips(aside) > max(rc.HOME_DIP_M, dips([DIPS_TO])) + 0.1, "the premise: it dips deeper"
  assert _own_room(DIPS_FROM, rc._tucked_way("left", DIPS_FROM, DIPS_TO)) < 0.0
  r.q_cmd["left"] = DIPS_FROM.copy()
  assert r.way_out("left", DIPS_TO) is None and r.own["left"] is not None


def test_a_way_round_another_robot_keeps_off_the_trunk_too():
  """With another robot by (#678), a way other than today's is taken only
  where it keeps off the robot's own trunk too (`_own_kept`'s judge): #678's
  arm raised high, straight home meets the other robot and the side way
  keeps clear of it -- judged through the trunk, none is taken."""
  from test_arms_keep_off import RAISED_OUT, _at
  r = _standing_reach()
  r.keep_out = [_at(0.7, -30)]
  sm = r._planner_model()
  q0, q1 = RAISED_OUT.copy(), r.rest["right"].copy()
  r._pose_planner(sm, None)
  pick, dips = r._picked(sm, "right", q0, q1, False)
  assert r._kept_off(sm, "right", q0, q1, False, pick, dips, (None,), True) is not None
  r._pose_planner(sm, None)
  assert r._kept_off(sm, "right", q0, q1, False, pick, dips, (None,), True,
                     clear=lambda w: False) is None


def test_a_hand_outs_way_round_another_robot_is_judged_against_its_trunk():
  r = _standing_reach()
  r.keep_out = [(5.0, 5.0)]
  got = []
  real = rc.Reach._kept_off
  r._kept_off = lambda *a, **kw: (got.append(a[-1] if len(a) > 9 else kw.get("clear")), real(r, *a, **kw))[1]
  r.q_cmd["left"] = ON_AT.copy()
  assert r.way_out("left", ON_TO) is not None
  assert callable(got[-1]) and got[-1]([ON_TO]) and not got[-1]([LEVER_AGAIN])
  r.q_cmd["left"] = r.rest["left"].copy()
  r.way_out("left", ON_TO)
  assert got[-1] is None, "a way from rest is never judged so"


def test_each_pair_of_parts_begun_nearer_goes_no_nearer_and_the_others_stay_clear(monkeypatch):
  """Begun with one pair of parts in each other -- a gripper a stop left 2
  mm in the torso -- a way may take that pair no deeper, and every other
  pair stays out: judged on the least of all, the forearm could go 1.9 mm
  into the head because the gripper began 2 mm in the torso."""
  start = np.full(len(hm.ARM_GEOMS) * len(rc.UPPER_GEOMS), 0.05)
  start[0] = -0.002
  into = start.copy()
  into[1] = -0.0019
  deeper = start.copy()
  deeper[0] = -0.003
  same = start.copy()
  same[1] = 0.001
  monkeypatch.setattr(rc, "own_rooms", lambda sm, side, q0, ways: (
    start if not len(ways) else {0.1: into, 0.2: deeper, 0.3: same}[float(ways[-1][0])]))
  r = _standing_reach()
  sm = r._planner_model()
  for to, ok in ((0.1, False), (0.2, False), (0.3, True)):
    q1 = np.full(6, to)
    r.own["left"] = None
    way, _ = r._own_kept(sm, "left", np.zeros(6), q1, [q1], True)
    assert (way is not None) is ok, to


def test_an_arm_begun_on_the_trunk_may_go_no_nearer():
  """Begun nearer than `OWN_CLEAR_M` -- a gripper a stop left 2 mm into the
  torso -- a way may go no nearer than it began: the straight way back to
  where it began is taken, and on past it, deeper, it is not."""
  r = _standing_reach()
  q_on = AT_LEVER + (LEVER_AGAIN - AT_LEVER) * 0.94
  start = _own_room(q_on, [])
  assert start < rc.OWN_CLEAR_M and _own_room(q_on, [AT_LEVER]) >= start
  r.q_cmd["left"] = q_on.copy()
  assert rc._same_way(r.way_out("left", AT_LEVER), [AT_LEVER])
  r.q_cmd["left"] = q_on.copy()
  assert r.way_out("left", LEVER_AGAIN) is None


def _at_lever():
  """A bare reach with its left hand held at t_0360's lever, whose plan is
  t_0360's reach again from there (its pre-grasp's gripper in the torso)."""
  r = _standing_reach()
  r.q_cmd["left"], r.phase["left"] = AT_LEVER.copy(), "hold"
  plan_ = ((0.0, 0.0, 0.0), LEVER_AGAIN.copy(), LEVER_AGAIN.copy(), True)
  r._planned = lambda side, goal, approach, postures, flip: (plan_, goal, None)
  return r, HandGoal(np.array([0.3, 0.1, 0.9]), None, 1.0)


def test_a_verbs_reach_whose_way_out_meets_its_own_trunk_moves_nothing():
  r, goal = _at_lever()
  assert not r.reach("left", goal, verb=True)
  assert r.phase["left"] == "hold" and r.postures == [] and r.own["left"]["part"] == "torso"
  # ...and asked whether its way would be clear, it says "own" and keeps what
  # it said
  r.own["left"] = said = {"part": "head", "m": -0.01}
  assert r.clear_way("left", goal) == "own" and r.own["left"] is said


def test_the_loops_own_reach_takes_the_way_least_near_its_own_trunk():
  """A charge's reach (no `verb`) is never refused for its own trunk -- no
  wait frees it, and a failed charge is a death -- even asked to refuse, as
  the body's first try asks it (to wait out another robot, #678): it takes
  the way that comes least near, and `own` says how near."""
  r, goal = _at_lever()
  assert r.reach("left", goal, refuse=True, verb=False)
  rooms = [_own_room(AT_LEVER, w) for w in ([LEVER_AGAIN], rc._aside("left", AT_LEVER, LEVER_AGAIN, True),
                                            rc._tucked_way("left", AT_LEVER, LEVER_AGAIN))]
  took = [r.q_to["left"]] + r.outs["left"]
  assert r.phase["left"] == "move" and _own_room(AT_LEVER, took) == pytest.approx(max(rooms))
  assert r.own["left"]["m"] == pytest.approx(max(rooms), abs=1e-3)


def test_a_refused_way_whose_pre_grasp_lies_in_the_trunk_judges_no_other(monkeypatch):
  """Every way ends at the pre-grasp: where it lies in the trunk, a verb's
  reach is refused without judging the side way or the way drawn in (each
  costs a sampled walk on the physics thread)."""
  r, goal = _at_lever()
  assert _own_room(LEVER_AGAIN, []) < 0.0, "the premise: the pre-grasp is in the torso"
  monkeypatch.setattr(rc, "_aside", lambda *a: (_ for _ in ()).throw(AssertionError("judged")))
  monkeypatch.setattr(rc, "_tucked_way", lambda *a: (_ for _ in ()).throw(AssertionError("judged")))
  assert r.way_out("left", LEVER_AGAIN) is None and r.own["left"]["part"] == "torso"


def test_with_no_way_clear_of_its_trunk_the_loop_weighs_another_robot_as_before():
  """The loop's own reach with no way clear of its trunk takes the least near
  -- and hands `_kept_off` no judge of its trunk, so another robot by is
  weighed against every way, as #678 weighs them."""
  r, goal = _at_lever()
  r.keep_out = [(5.0, 5.0)]
  got = []
  real = rc.Reach._kept_off
  r._kept_off = lambda *a, **kw: (got.append(a[-1] if len(a) > 9 else kw.get("clear")), real(r, *a, **kw))[1]
  assert r.way_out("left", LEVER_AGAIN, refuse=False) is not None
  assert got == [None] and r.own["left"] is not None


def test_the_body_refuses_a_verbs_reach_for_its_own_trunk_and_never_its_own():
  """Through the body (`HumanoidBody._reached`, as `reach_routine` asks it):
  a procedure's reach is refused, "own body"; the loop's own goes."""
  for verb in (True, False):
    r, goal = _at_lever()
    me = NS(reach=r, _why={}, _kept={}, mission=NS(look_round=lambda: None, keep_out=lambda seen: []))
    went = hb.HumanoidBody._reached(me, "left", goal, rc.REACH_POSTURES, False, words=verb, verb=verb)
    assert went is (not verb) and me._why["left"] == ("own body" if verb else "out of reach")


def test_a_carried_hand_whose_reach_is_refused_is_still_carried():
  """Nothing moves, so a hand a walk carried stays carried (#682): its way
  home is still to its carry pose -- refused, `carry_q` dropped, the hand at
  the robot's side read as out at its work."""
  r, goal = _at_lever()
  r.phase["left"], r.carry_q["left"] = "carried", AT_LEVER.copy()
  assert not r.reach("left", goal, verb=True)
  assert r.carried("left") and np.allclose(r.home_to("left"), AT_LEVER)


# ---- 4: the box against a box ----------------------------------------------------------


def test_a_box_against_a_box_that_mujoco_reads_at_naught_is_read_again():
  """On t_0408's reach from the lever (R5's rows), 14/15 of the way: MuJoCo
  3.10 reads the gripper 0.0 from the torso, its from-to 4 cm long; it is 2.1
  cm apart, as the poses either side read."""
  sm = SelfModel()
  q0, q1 = ON_AT, ON_TO
  g, t = sm.m.geom("left_gripper").id, sm.m.geom("torso").id

  def read(f):
    sm.d.qpos[sm.qadr[sm.arm["left"]]] = q0 + (q1 - q0) * f
    mujoco.mj_kinematics(sm.m, sm.d)
    return mujoco.mj_geomDistance(sm.m, sm.d, g, t, rc.OWN_READ_M, None), rc.apart(sm.m, sm.d, g, t)

  assert read(14 / 15)[0] == 0.0, "the premise: MuJoCo reads naught"
  assert read(14 / 15)[1] == pytest.approx(0.0215, abs=0.001)
  for f in (13 / 15, 1.0):
    raw, got = read(f)
    assert raw > 0.0 and got == raw and rc._box_box(sm.m, sm.d, g, t) == pytest.approx(raw, abs=1e-9)


# ---- 5: the why and the words ----------------------------------------------------------


def test_a_reach_refused_for_its_own_trunk_is_the_bodys_why_own_body():
  said = {"part": "torso", "m": -0.009}

  class Arms:
    kept, own, kept_by, keep_out = {"left": None}, {"left": None}, {"left": None}, []

    def reach(self, side, goal, **kw):
      self.own[side] = said
      return False
  me = NS(reach=Arms(), _why={}, _kept={},
          mission=NS(look_round=lambda: None, keep_out=lambda seen: []))
  assert not hb.HumanoidBody._reached(me, "left", None, (), False, words=True)
  assert me._why["left"] == "own body" and "left" not in me._kept


def test_a_reach_kept_off_another_robot_says_the_other_hands_way_passes_through_its_body():
  class Off(StubBody):
    def reach_routine(self, side, pos, rot=None, stiffness=1.0, patience=20.0, path=None,
                      turns=None, refuse=False):
      return False
      yield

    def hand_why(self, side):
      return "kept off"

    def hand_kept(self, side):
      return {"m": 0.9, "x": 0.5, "y": -0.6, "home": False, "back": False, "other": "own"}
  step = _fly('def a():\n    reach("right", 0.3, -0.2, 0.9)\n', Off(*_bare()))["steps"][-1]
  assert step["reason"].endswith(st.OTHER_HAND_KEPT["own"].format(other="left"))


def test_a_reach_refused_for_its_own_body_says_so_and_whether_the_other_hand_could():
  class Own(StubBody):
    reaches = False

    def reach_routine(self, side, pos, rot=None, stiffness=1.0, patience=20.0, refuse=False):
      return False
      yield

    def hand_why(self, side):
      return "own body"

    def hand_reaches(self, side, pos, rot=None, verb=False):
      return self.reaches
  for reaches in (False, True):
    body = Own(*_bare())
    body.reaches = reaches
    step = _fly('def a():\n    reach("left", 0.3, 0.2, 0.9)\n', body)["steps"][-1]
    assert not step["ok"] and step["why"] == "own body"
    said = st.REACH_WHY["own body"].format(hand="left")
    assert step["reason"] == said + ("; your right hand could reach it as asked" if reaches else "")
