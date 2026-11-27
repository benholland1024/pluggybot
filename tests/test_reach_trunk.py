"""#763 (#736's call 6): a procedure's posture judges its trunk -- the torso
and the head's parts -- walked on the commands' clocks into the posture and
back up from it to the stand, against what its depth camera saw since its
feet last moved, tops kept (`Reach._trunk`, `HumanoidMission.trunk_seen`):
a posture whose trunk would come nearer than `TRUNK_CLEAR_M` is passed over
-- a reach's plan and its look, a carry's and a pull's at their ask and on
their way -- and where none keeps clear, a reach moves nothing and says so, a
carry is refused or held where it is, a pull held where it is. The loop's
own moves are never judged. SimNotes, "A reach's bow under a shelf's
board"."""

import math
from types import SimpleNamespace as NS

import mujoco
import numpy as np
import pytest
from scipy.spatial import cKDTree

from pluggybot.humanoid import body as hb
from pluggybot.humanoid import mission as hmis
from pluggybot.humanoid import model as hm
from pluggybot.humanoid import reach as rc
from pluggybot.humanoid.arms import HandGoal
from pluggybot.procedure import steps as st

from test_reach_swing import _floor

#: A square grip: the fingers straight ahead, the jaws closing across them
#: (`steps._grip_turn` of (1, 0, 0) and (0, 1, 0)).
SQUARE = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
#: The shop shelf's upper set-down pre-point in U35's k253 (#763): 0.256 m
#: ahead, 0.225 left, at the chest, its plan the deepest bow turned 60 deg.
K253 = HandGoal(np.array([0.256, 0.225, 1.066]), SQUARE, 1.0)
DEEP = (-0.25, 1.1, math.radians(60))
#: ...and three postures in the search's own order (`with_waist`): the
#: stand, the stand turned 45 deg, the deepest bow turned 60 -- the one
#: today's plan takes there with its slack, the turned stand the one it takes
#: at full stretch past it: the search kept short.
SHORT = ((0.0, 0.0, 0.0), (0.0, 0.0, math.radians(45)), DEEP)


def _board(front, depth=0.40, y0=-0.4, y1=0.8, top=0.96):
  """The shelf's upper board as the arms' memory keeps it, tops kept: 2 cm
  voxels over its top and down its front edge, from `front` m ahead."""
  xs = np.arange(front, front + depth + 1e-9, 0.02)
  ys = np.arange(y0, y1 + 1e-9, 0.02)
  return np.array([[x, y, top] for x in xs for y in ys] + [[front, y, top - 0.02] for y in ys])


def _view(pts):
  return cKDTree(pts), pts


@pytest.fixture(scope="module")
def floor():
  return _floor()


def _plan(reach, goal, seen, postures=SHORT):
  reach.trunk_view = None if seen is None else (lambda: seen)
  try:
    return reach._planned("left", goal, rc.APPROACH_M, postures, True, swing=True, passed=[])
  finally:
    reach.trunk_view = None


# ---- the judgment ---------------------------------------------------------------------------


def test_k253s_set_down_bow_brings_the_head_into_the_upper_board_and_is_passed_over(floor):
  """#763's case on the robot's model of itself: k253's pre-point plans the
  deepest bow turned 60 deg, and with the upper board seen 0.42 m ahead its
  head comes up into it on the way back to the stand (0.36 m: on the way
  in) -- passed over, the plan takes a posture whose trunk keeps clear;
  with nothing seen, today's plan bit for bit."""
  today = floor._planned("left", K253, rc.APPROACH_M, SHORT, True)[0]
  assert today[0] == DEEP, "the premise: today's plan is the deepest bow"
  lean = floor._lean()
  arms = {**floor._arms(but="left"), **rc._arm("left", today[2])}
  for front, way in ((0.42, "up"), (0.36, "in")):
    met = floor._trunk(today[0], arms, lean, _view(_board(front)))
    assert met is not None and met["m"] < rc.TRUNK_CLEAR_M and met["way"] == way, met
    assert met["part"] in rc.UPPER_GEOMS
    judged = _plan(floor, K253, _board(front))
    assert judged[0] is not None and judged[0][0] != DEEP and judged[3] is None
    assert floor._trunk(judged[0][0], {**floor._arms(but="left"), **rc._arm("left", judged[0][2])},
                        lean, _view(_board(front))) is None
  for seen in (None, np.zeros((0, 3))):
    same = _plan(floor, K253, seen)
    assert same[3] is None and same[0][0] == today[0] and same[0][3] == today[3]
    assert np.array_equal(same[0][1], today[1]) and np.array_equal(same[0][2], today[2])
  # ...and a board further off, which the deep bow's trunk keeps clear of: today's
  far = _plan(floor, K253, _board(0.50))
  assert far[0][0] == DEEP and np.array_equal(far[0][2], today[2])


def test_where_no_posture_keeps_the_trunk_clear_a_procedures_reach_moves_nothing():
  """Refused, nothing asked of the body: `swung` marked `trunk`, saying the
  part, how near, where and on which walk -- the body's why "head"; the
  loop's own reach (a charge's, `verb=False`) is never judged and goes."""
  reach = _floor()
  board = _board(0.42)
  reach.trunk_view = lambda: board
  posture, phase = reach.posture.copy(), dict(reach.phase)
  assert not reach.reach("left", K253, postures=(DEEP,), verb=True)
  assert reach.phase == phase and np.array_equal(reach.posture, posture)
  said = reach.swung["left"]
  assert said["trunk"] and said["part"] in rc.UPPER_GEOMS and said["way"] in ("in", "up")
  assert any(np.allclose(said["at"], p, atol=1e-3) for p in board)
  assert reach.reach("left", K253, postures=(DEEP,), verb=False)
  assert reach.swung["left"] is None and tuple(reach.posture) == DEEP


def test_a_plan_that_passed_over_a_posture_for_its_trunk_takes_no_look():
  """As for the arms' swing (#747): the posture it goes in from is the one
  the plan chose; a look could take back what the trunk was kept out of."""
  reach = _floor()
  board = _board(0.42)
  reach.trunk_view = lambda: board
  assert reach.reach("left", K253, postures=SHORT, verb=True)
  assert reach.looks["left"] == rc.LOOKS and tuple(reach.posture) != DEEP


def test_a_look_passes_over_a_posture_whose_trunk_meets_what_it_saw():
  """A look at the pre-grasp (#627) judges the trunk of the posture it would
  take: the deepest bow, its head up into the board, is not taken -- it
  keeps the one it holds; unseen, it takes it."""
  reach = _floor()
  here = reach.posture.copy()
  pre = rc.pregrasp(K253)
  for seen, takes in ((None, True), (_board(0.42), False)):
    reach.trunk_view = None if seen is None else (lambda s=seen: s)
    reach.goal["left"] = K253
    reach.cmd["left"] = HandGoal(pre.pos.copy(), SQUARE, 1.0)
    reach.allowed["left"], reach.left["left"], reach.looks["left"] = [DEEP], [], 0
    reach.seen_at["left"] = np.zeros((0, 3))
    stays = reach._look("left")
    assert stays is (not takes) and tuple(reach.posture) == (DEEP if takes else tuple(here))
    reach.hold_posture(*here)
  reach.trunk_view = None


def test_a_procedures_carry_and_pull_pass_over_a_posture_whose_trunk_meets_what_it_saw(floor):
  """`posture_for` and `path_for` judging (a procedure's carry or pull, at
  its ask and on its way): k253's point from the stand plans the deepest bow
  turned 60 deg; judged with the board seen 0.42 m ahead it is passed over,
  `trunked` saying what its trunk would meet; unjudged, or nothing seen,
  today's."""
  board = _board(0.42)
  short = ((0.0, 0.0, 0.0), DEEP)
  floor.trunk_view = lambda: board
  try:
    assert floor.posture_for("left", K253.pos, rot=SQUARE, postures=short) == DEEP, "the premise"
    assert floor.posture_for("left", K253.pos, rot=SQUARE, postures=short, judge=True) is None
    assert floor.trunked["left"] is not None and floor.trunked["left"]["part"] in rc.UPPER_GEOMS
    path = [K253.pos, K253.pos + [0.0, 0.0, -0.05]]
    assert floor.path_for("left", path, rot=SQUARE, postures=(DEEP,)) == DEEP
    assert floor.path_for("left", path, rot=SQUARE, postures=(DEEP,), judge=True) is None
    assert floor.trunked["left"] is not None
  finally:
    floor.trunk_view = None
  assert floor.posture_for("left", K253.pos, rot=SQUARE, postures=short, judge=True) == DEEP


def test_a_procedures_carry_and_pull_hand_their_judge_on_and_the_loops_own_do_not(monkeypatch):
  """The wiring: a carry's and a pull's ask, a carry's replans on its way and
  a pull's onward search judge where the move is a procedure's (`verb`), and
  a carry none of whose postures keeps the trunk clear is refused before it
  moves, its report saying so."""
  reach = _floor()
  asked = []
  met = {"part": "head_cap", "m": -0.02, "began": None, "at": [0.42, 0.2, 0.94], "way": "up",
         "t": 0.3}

  def posture_for(side, pos, rot=None, onward=False, past=(), postures=None, judge=False):
    asked.append(("posture_for", judge))
    reach.trunked[side] = met if judge else None
    return None

  def path_for(side, poses, rot=None, postures=None, judge=False):
    asked.append(("path_for", judge))
    reach.trunked[side] = met if judge else None
    return None
  monkeypatch.setattr(reach, "posture_for", posture_for)
  monkeypatch.setattr(reach, "path_for", path_for)
  monkeypatch.setattr(reach, "out", lambda side: True)
  for verb in (True, False):
    asked.clear()
    assert not reach.carry("left", K253.pos, verb=verb)
    assert asked == [("posture_for", verb)]
    assert (reach.report["left"].get("trunk") == met) is verb
    asked.clear()
    assert not reach.carry("left", K253.pos, path=[K253.pos], verb=verb)
    assert asked == [("path_for", verb)]
    asked.clear()
    reach.pull("left", (0.0, 0.0, -1.0), 6.0, 0.1, verb=verb)
    assert asked == [("path_for", verb)]
    assert bool(reach.pulls["left"].get("verb")) is verb
    assert (reach.pulls["left"].get("trunk") == met) is verb


def test_the_view_is_what_it_saw_since_its_feet_last_moved_tops_kept_and_a_pure_read():
  """`ArmsMemory.since`: the voxels whose newest sight is at or after the
  feet last moved, tops kept -- a board's 2 cm, a column's top whole, is
  there -- and nothing asked of the belief: a read that forgets nothing."""
  mem = hmis.ArmsMemory()
  pose, corrected = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
  before = np.array([[0.5, y, 0.96] for y in np.arange(-0.2, 0.2, 0.02)])
  after = np.array([[0.6, y, 0.94] for y in np.arange(-0.2, 0.2, 0.02)])
  mem.fold(before, 1.0, pose, corrected)
  mem.fold(after, 3.0, pose, corrected)
  pts, ts = mem.pts.copy(), mem.t.copy()
  seen = mem.since(pose, 2.0)
  assert len(seen) == len(after) and np.allclose(np.sort(seen[:, 0]), 0.6)
  assert len(mem.since(pose, 0.5)) == len(before) + len(after)
  assert len(mem.since(pose, 4.0)) == 0
  # ...the board's top is in it, where the faces leave a column's top out
  assert len(mem.faces(pose, corrected)) == 0
  # ...none where the belief jumped since the memory last read it (placed
  # where it no longer is), and nothing forgotten: it is a pure read
  mem.seen = np.array([0.0, 0.0, 0.0])
  assert len(mem.since(pose, 0.5, (0.05, 0.0, 0.0))) == len(before) + len(after)
  assert len(mem.since(pose, 0.5, (0.2, 0.0, 0.0))) == 0
  assert len(mem.since(pose, 0.5, (0.0, 0.0, 0.1))) == 0
  assert np.array_equal(mem.pts, pts) and np.array_equal(mem.t, ts)
  assert np.array_equal(mem.seen, [0.0, 0.0, 0.0])


def test_an_empty_view_judges_nothing_todays_plan_bit_for_bit(floor):
  """Just after a step, before the camera has seen again, nothing is
  judged (#763's call: today's behaviour)."""
  today = floor._planned("left", K253, rc.APPROACH_M, SHORT, True)[0]
  floor.trunk_view = lambda: np.zeros((0, 3))
  try:
    assert floor._trunk_view() == (None, None)
    got = floor._planned("left", K253, rc.APPROACH_M, SHORT, True, swing=True, passed=[])[0]
  finally:
    floor.trunk_view = None
  assert got[0] == today[0] and np.array_equal(got[2], today[2])


def test_the_trunks_shapes_are_read_to_their_surfaces():
  """`trunk_gap`'s signed distance for a box, a cylinder (its side, its end
  and its rim) and a sphere, outside positive and inside negative."""
  eye = np.eye(3)
  box = ("torso", "box", np.zeros(3), eye, np.array([0.1, 0.2, 0.3]))
  pts = np.array([[0.15, 0, 0], [0, 0.25, 0], [0, 0, 0.29], [0.13, 0.24, 0.0]])
  assert np.allclose(rc.trunk_gap(pts, box), [0.05, 0.05, -0.01, 0.05])
  cyl = ("head", "cylinder", np.array([0, 0, 1.0]), eye, np.array([0.068, 0.058, 0.0]))
  pts = np.array([[0.1, 0, 1.0], [0, 0, 1.1], [0.0, 0.0, 1.0], [0.068 + 0.03, 0, 1.058 + 0.04]])
  assert np.allclose(rc.trunk_gap(pts, cyl), [0.032, 0.042, -0.058, 0.05])
  sph = ("head_dome", "sphere", np.zeros(3), eye, np.array([0.066, 0.0, 0.0]))
  assert np.allclose(rc.trunk_gap(np.array([[0.1, 0, 0], [0.0, 0.0, 0.0]]), sph), [0.034, -0.066])
  assert rc.trunk_reach(cyl) == pytest.approx(math.hypot(0.068, 0.058))


def test_the_trunk_is_judged_on_the_robots_model_of_itself_never_the_simulators_truth():
  """A reach refused for its trunk, with the world's `MjData` a spy: nothing
  read of where the world says the bodies are or what they touch."""
  from pluggybot.humanoid.policy import StandDriver
  from test_humanoid import TRUTH, _Spy
  model = mujoco.MjModel.from_xml_string(hm.body_xml())
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  mujoco.mj_forward(model, data)
  spy = _Spy(data)
  stand = StandDriver(model, spy)
  reach = rc.Reach(model, spy, stand)
  for _ in range(int(0.3 / model.opt.timestep)):
    stand.step()
    reach.step()
    mujoco.mj_step(model, data)
  board = _board(0.42)
  reach.trunk_view = lambda: board
  spy.read.clear()
  assert not reach.reach("left", K253, postures=(DEEP,), verb=True)
  assert reach.swung["left"] is not None and reach.swung["left"]["trunk"], "the premise"
  assert reach.d is spy, "the premise: the reach meets the world through the spy"
  assert not spy.read & TRUTH, spy.read & TRUTH


def test_the_trunk_is_posed_on_a_model_of_its_own_never_the_planners():
  """A plan the trunk judges leaves the planner's model of itself -- its
  kept statues and the pose a later solve starts from -- as a plan with
  nothing seen leaves it: a flight the rule does not act on reads nothing
  it wrote."""
  states = []
  for seen in (None, _board(0.60)):
    reach = _floor()
    _plan(reach, K253, seen)
    sm = reach._planner_model()
    states.append((sm.d.qpos.copy(), sm.q.copy(), sorted(map(str, sm.statues))))
  assert np.array_equal(states[0][0], states[1][0]) and np.array_equal(states[0][1], states[1][1])
  assert states[0][2] == states[1][2]


# ---- through the body and the verbs ---------------------------------------------------------


def test_the_body_says_head_where_the_first_posture_brings_the_trunk_into_what_it_saw():
  """`HumanoidBody._reached`: a refusal marked `trunk` is "head" (an arm's
  swing stays "swing"), and `hand_swung` hands its met to the words."""
  said = {"part": "head_cap", "m": -0.013, "began": None, "at": [0.42, 0.02, 0.94], "way": "up",
          "t": 0.3, "trunk": True}

  class Arms:
    kept, own, kept_by, keep_out = {"left": None}, {"left": None}, {"left": None}, []
    swung, seen = {"left": None}, None

    def reach(self, side, goal, **kw):
      self.swung[side] = said
      return False
  arms = Arms()
  me = NS(reach=arms, _why={}, _kept={},
          mission=NS(look_round=lambda: None, keep_out=lambda seen: [], arms_seen=lambda: None))
  assert not hb.HumanoidBody._reached(me, "left", None, (), False, words=True, verb=True)
  assert me._why["left"] == "head" and hb.HumanoidBody.hand_swung(me, "left") == said


def test_a_reach_held_out_by_its_trunk_says_what_it_would_meet_and_whether_the_other_hand_could():
  """The words a procedure reads (#763, a draft for Ben): the part (the
  head's, or the trunk for the torso), the walk (taking the posture, or
  standing up from it) and where, its stance frame; and, as for the arms'
  swing, whether the other hand could."""
  from pluggybot.body import StubBody
  from test_hands import _bare, _fly

  class Held(StubBody):
    reaches = False

    def reach_routine(self, side, pos, rot=None, stiffness=1.0, patience=20.0, refuse=False):
      return False
      yield

    def hand_why(self, side):
      return "head"

    def hand_reaches(self, side, pos, rot=None, verb=False):
      return self.reaches
  for reaches, part, way in ((False, "head_cap", "up"), (True, "torso", "in")):
    body = Held(*_bare())
    body.reaches = reaches
    body.swung["left"] = {"part": part, "m": -0.013, "at": [0.42, 0.02, 0.94], "way": way,
                          "trunk": True}
    step = _fly('def a():\n    reach("left", 0.3, 0.2, 1.0)\n', body)["steps"][-1]
    assert not step["ok"] and step["why"] == "head"
    said = st.REACH_HEAD.format(hand="left", part="head" if part != "torso" else "trunk",
                                way=st.TRUNK_WAYS[way], x=0.42, y=0.02, z=0.94)
    assert ("standing up from it" if way == "up" else "taking it") in said
    assert step["reason"] == said + ("; your right hand could reach it as asked" if reaches else "")


def test_a_carry_held_where_it_is_by_its_trunk_says_so_and_a_pull_says_nothing_new():
  """`move_hand` whose report carries what every other posture's trunk would
  meet reads false saying so (a draft for Ben); a carry ended out of reach
  for any other cause keeps its words."""
  from pluggybot.body import StubBody
  from test_hands import _bare, _fly

  class Carried(StubBody):
    trunk = None

    def move_hand_routine(self, side, pos, stiffness=1.0, speed=0.15, patience=20.0, path=None,
                          turns=None, verb=False):
      return {"why": "out of reach", "seconds": 0.0, "moved": 0.0,
              **({"trunk": self.trunk} if self.trunk else {})}
      yield
  for trunk in ({"part": "head_rim", "m": -0.02, "at": [0.4, 0.0, 0.9], "way": "in"}, None):
    body = Carried(*_bare())
    body.hands["left"] = (0.3, 0.2, 1.0)
    body.trunk = trunk
    step = _fly('def a():\n    move_hand("left", 0.3, 0.2, 1.0)\n', body)["steps"][-1]
    assert not step["ok"]
    assert step["reason"] == (st.MOVE_HEAD.format(hand="left", part="head") if trunk
                              else st.MOVE_WHY["out of reach"].format(hand="left"))


def _stepped(reach, until, most_s):
  """The floor's reach stepped with its standing policy until `until()`."""
  model, data = reach.m, reach.d
  for _ in range(int(most_s / model.opt.timestep)):
    if until():
      return True
    reach.stand.step()
    reach.step()
    mujoco.mj_step(model, data)
  return until()


def test_a_procedures_carry_judges_its_trunk_on_its_way_and_the_loops_own_does_not(monkeypatch):
  """A carry's replan on its way (its arm run out of reach from the posture
  it holds) asks `posture_for` to judge the trunk where the carry is a
  procedure's, and not where it is the loop's own."""
  for verb in (True, False):
    reach = _floor()
    assert reach.reach("left", HandGoal(np.array([0.30, 0.10, 0.80]), None, 0.5), verb=verb)
    assert _stepped(reach, lambda: reach.phase["left"] == "hold", 4.0), "the premise: it holds"
    asked = []
    real = reach.posture_for

    def spy(side, pos, rot=None, onward=False, past=(), postures=None, judge=False):
      if not asked:
        asked.append(("ask", judge))
        return tuple(float(v) for v in reach.posture)
      asked.append(("on its way", judge))
      return real(side, pos, rot=rot, onward=onward, past=past, postures=postures, judge=judge)
    monkeypatch.setattr(reach, "posture_for", spy)
    assert reach.carry("left", np.array([0.62, 0.10, 0.80]), verb=verb)
    _stepped(reach, lambda: len(asked) > 1 or reach.phase["left"] != "carry", 2.0)
    assert asked[0] == ("ask", verb) and len(asked) > 1, asked
    assert asked[1] == ("on its way", verb)


def test_a_procedures_pull_judges_its_trunk_going_on_and_the_loops_own_does_not(monkeypatch):
  """A pull's onward search (its arm run out of reach going on, #525) asks
  `posture_for` to judge the trunk where the pull is a procedure's."""
  for verb in (True, False):
    reach = _floor()
    assert reach.reach("left", HandGoal(np.array([0.30, 0.10, 0.80]), None, 0.5), verb=verb)
    assert _stepped(reach, lambda: reach.phase["left"] == "hold", 4.0), "the premise: it holds"
    asked = []

    def spy(side, pos, rot=None, onward=False, past=(), postures=None, judge=False):
      asked.append((onward, judge))
      return None
    monkeypatch.setattr(reach, "posture_for", spy)
    # ...its ask taking none, as one further than any posture follows from the stand
    monkeypatch.setattr(reach, "path_for", lambda *a, **kw: None)
    assert reach.pull("left", (1.0, 0.0, 0.0), 10.0, 0.6, speed=0.25, verb=verb)
    _stepped(reach, lambda: any(o for o, _ in asked) or reach.phase["left"] != "carry", 2.5)
    assert (True, verb) in asked, asked


def test_the_body_hands_its_reach_what_it_saw_since_its_feet_moved_and_again_after_a_stand_up():
  """`HumanoidMission.trunk_seen`, the reach's `trunk_view` from its boot and
  again once a stand-up builds its reach anew: the arms' memory, tops kept,
  seen since `last_motion_t`, placed in its stance frame as `arms_seen`
  places its faces."""
  from test_guard import _floor_body
  body = _floor_body()
  try:
    mi = body.mission
    assert body.reach.trunk_view == mi.trunk_seen
    top = np.array([[0.4, y, 0.96] for y in np.arange(-0.2, 0.2, 0.02)])
    mi.swing.fold(top, 1.0, mi.pose, mi.odo.corrected)
    mi.last_motion_t = 0.5
    seen = mi.trunk_seen()
    assert len(seen) == len(top) and np.allclose(seen[:, 2], 0.96)
    assert len(mi.arms_seen()) == 0, "the premise: a board's top is no face"
    mi.last_motion_t = 2.0
    assert len(mi.trunk_seen()) == 0
    mi._stood_up()
    assert body.reach.trunk_view == mi.trunk_seen
  finally:
    body.close()


def test_the_posture_the_body_holds_is_never_judged_for_its_trunk(monkeypatch):
  """A reach planned in the posture the body holds, or has got to (the one
  #719 keeps it in), changes no posture: its trunk is not judged, whatever
  its walk back up to the stand would meet (round 1's review); another
  posture is."""
  reach = _floor()
  board = _board(0.42)
  goal = HandGoal(np.array([0.30, 0.10, 0.80]), None, 0.5)
  judged = []
  real = reach._trunk
  monkeypatch.setattr(reach, "_trunk", lambda posture, *a: judged.append(tuple(posture)) or real(posture, *a))
  crouch = (-0.08, 0.0, 0.0)
  assert _plan(reach, goal, board, postures=(crouch,))[0] is not None, "the premise: it plans there"
  assert crouch in judged, "the premise: another posture is judged"
  here = tuple(float(v) for v in reach.posture)
  assert here == reach._in_or_got_to() == (0.0, 0.0, 0.0)
  judged.clear()
  assert _plan(reach, goal, board, postures=(here,))[0] is not None
  assert judged == []


def test_a_replans_trunk_said_only_where_it_held_the_move():
  """`_trunk_said`: what the trunk met kept for the report where a
  procedure's replan took no posture for it; dropped where a later replan
  took one or none reached; nothing at all for the loop's own move."""
  reach = _floor()
  met = {"part": "head_cap", "m": -0.02, "way": "up"}
  st_ = {"verb": True}
  reach.trunked["left"] = met
  reach._trunk_said("left", st_, None)
  assert st_["trunk"] == met
  reach._trunk_said("left", st_, (0.0, 0.0, 0.0))
  assert "trunk" not in st_
  reach._trunk_said("left", st_, None)
  reach.trunked["left"] = None
  reach._trunk_said("left", st_, None)
  assert "trunk" not in st_
  own = {}
  reach.trunked["left"] = met
  reach._trunk_said("left", own, None)
  assert own == {}


def test_each_walk_begins_where_the_body_stands_and_the_walk_up_ends_at_its_lean(monkeypatch):
  """The trunk's walk into a posture begins with the body as it reads itself
  -- even where the posture shown already is the one asked -- and its walk
  back up ends at the stand's statue at the lean the posture's is balanced
  at (as every stand a way home is judged in, `_pose_planner`)."""
  reach = _floor()
  posed = []
  real = rc.trunk_parts
  monkeypatch.setattr(rc, "trunk_parts", lambda sm: posed.append(sm.d.qpos[:7].copy()) or real(sm))
  lean = reach._lean()
  arms = reach._arms()
  board = _board(0.42)
  # ...where the posture shown already is the one asked, every clock arrived
  reach.trim[:] = 0.0
  reach.stand.posture[:] = 0.0
  here = tuple(float(v) for v in reach.posture)
  assert here == (0.0, 0.0, 0.0) and not reach.waist_speed.any(), "the premise"
  reach._trunk(here, arms, lean, _view(board))
  assert not np.allclose(posed[1], reach.self_model.d.qpos[:7]), "the premise: its statue is elsewhere"
  assert np.allclose(posed[0], reach.self_model.d.qpos[:7])
  posed.clear()
  reach._trunk(DEEP, arms, lean, _view(board))
  sm = rc.SelfModel()
  q0 = np.array([hm.stand_qpos().get(n, 0.0) for n in hm.JOINT_NAMES])
  rc.statue(sm, (0.0, 0.0, 0.0), q0, arms, lean, fast=True)
  assert np.allclose(posed[-1], sm.d.qpos[:7], atol=1e-9)
