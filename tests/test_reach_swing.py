"""#747: a procedure's reach judges the posture it takes as it sets off -- the
body walked to the plan's posture while the arm walks out to its pre-grasp --
against what its depth camera saw where its arms swing (`mission.ArmsMemory`,
`Reach._swing`); a posture whose swing would bring either arm within
`SWING_CLEAR_M` of a face it saw is passed over, and where every one would,
the reach moves nothing and says so. The loop's own reach is never judged.
SimNotes, "A reach's posture swung into the counter"."""

import math
from types import SimpleNamespace as NS

import mujoco
import numpy as np
import pytest

from pluggybot.body import StubBody
from pluggybot.humanoid import body as hb
from pluggybot.humanoid import mission as hmis
from pluggybot.humanoid import model as hm
from pluggybot.humanoid import reach as rc
from pluggybot.humanoid.arms import HandGoal
from pluggybot.procedure import steps as st

from test_hands import _bare, _fly

#: The left hand across the body, 20 cm right of the stance at a counter's
#: height (Y6's grid, #747: from the stand its plan turns the waist 45 deg),
#: and to its own side, 10 cm left.
ACROSS = HandGoal(np.array([0.30, -0.20, 0.80]), None, 0.5)
OWN_SIDE = HandGoal(np.array([0.30, 0.10, 0.80]), None, 0.5)


def _floor():
  """A reach on the robot's own drawing on a bare floor, standing still."""
  from pluggybot.humanoid.policy import StandDriver
  model = mujoco.MjModel.from_xml_string(hm.body_xml())
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  mujoco.mj_forward(model, data)
  stand = StandDriver(model, data)
  reach = rc.Reach(model, data, stand)
  reach.standing = True
  for _ in range(int(round(0.3 / model.opt.timestep))):
    stand.step()
    reach.step()
    mujoco.mj_step(model, data)
  return reach


def _face(x, y0=-0.6, y1=0.6, z0=0.1, z1=0.70):
  """A counter's face as the memory gives it (`ArmsMemory.faces`), x m ahead
  of the stance: 2 cm voxels, its top's row left out."""
  ys, zs = np.arange(y0, y1 + 1e-9, 0.02), np.arange(z0, z1 + 1e-9, 0.02)
  return np.array([[x, y, z] for y in ys for z in zs])


@pytest.fixture(scope="module")
def floor():
  return _floor()


def _plan(reach, side, goal, seen, postures=rc.REACH_POSTURES):
  reach.seen = seen
  try:
    return reach._planned(side, goal, rc.APPROACH_M, postures, True, swing=True)
  finally:
    reach.seen = None


# ---- the clock ----------------------------------------------------------------------------------


def test_the_swing_clock_walks_each_command_at_its_own_rate():
  """The shown posture at the standing policy's rates, the waist on `turned`
  step for step, the arm along its way at `JOINT_RATE`, a tick each target it
  reaches (`Reach._joints`), until all have arrived."""
  from pluggybot.humanoid.policy import HEIGHT_RATE, PITCH_RATE
  q0 = np.zeros(6)
  ways = [np.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.0]), np.array([-1.0, 0.5, 0.0, 0.0, 0.0, 0.0])]
  clock = rc.swing_clock((0.0, 0.0), (-0.2, 0.55, -math.pi / 4), 0.0, 0.0, q0, ways, 0.01, every=0.01)
  ts = np.array([c[0] for c in clock])
  h = np.array([c[1] for c in clock])
  p = np.array([c[2] for c in clock])
  assert h[0] == 0.0 and p[0] == 0.0
  assert ts[np.argmax(h <= -0.2 + 1e-12)] == pytest.approx(0.2 / HEIGHT_RATE, abs=0.011)
  assert ts[np.argmax(p >= 0.55 - 1e-12)] == pytest.approx(0.55 / PITCH_RATE, abs=0.011)
  w, ws = np.zeros(1), np.zeros(1)
  for c in clock[1:]:
    w, ws = rc.turned(w, ws, np.array([-math.pi / 4]), 0.01)
    assert c[3] == w[0]
  q = np.array([c[4] for c in clock])
  first = int(np.argmax(np.all(q == ways[0], axis=1)))
  assert ts[first] == pytest.approx(0.5 / rc.JOINT_RATE, abs=0.011)
  assert np.array_equal(q[-1], ways[1]) and clock[-1][3] == -math.pi / 4


# ---- the memory ---------------------------------------------------------------------------------


def test_the_arms_memory_keeps_what_it_saw_where_its_arms_swing_its_tops_no_face():
  """In three dimensions, the knees' band and the arms' above it (`ARMS_TOP_Z`),
  placed by the belief; a column's highest voxel is the top of what stands
  there, no face (a hand comes down over it to its work); kept across a
  restart, forgotten where the belief jumps."""
  m = hmis.ArmsMemory()
  face = [[0.5, 0.0, z] for z in (0.02, 0.1, 0.3, 0.5, 0.7, 1.5)] + [[0.6, 0.0, 0.7]]
  m.fold(np.array(face), 0.0, (0.0, 0.0, 0.0), np.zeros(3))
  near = m.near((0.0, 0.0, 0.0), np.zeros(3))
  assert sorted(map(tuple, near.round(6))) == [(0.5, 0.0, 0.1), (0.5, 0.0, 0.3), (0.5, 0.0, 0.5),
                                               (0.5, 0.0, 0.7), (0.6, 0.0, 0.7)]
  faces = m.faces((0.0, 0.0, 0.0), np.zeros(3))
  assert sorted(map(tuple, faces.round(6))) == [(0.5, 0.0, 0.1), (0.5, 0.0, 0.3), (0.5, 0.0, 0.5)]
  # ...placed where the belief has it: a walk ahead brings it nearer
  assert np.allclose(sorted(m.faces((0.2, 0.0, 0.0), np.zeros(3))[:, 0]), [0.3, 0.3, 0.3])
  state, arrays = m.kept_state()
  back = hmis.ArmsMemory()
  back.restore_kept(state, arrays)
  assert np.array_equal(back.pts, m.pts) and np.array_equal(back.t, m.t)
  assert not len(m.near((0.0, 0.0, 0.0), np.array([hmis.KNEE_JUMP_M + 0.01, 0.0, 0.0])))


# ---- the judgment -------------------------------------------------------------------------------


def test_with_nothing_seen_a_procedures_plan_is_todays_bit_for_bit(floor):
  for goal in (ACROSS, OWN_SIDE):
    today = floor._planned("left", goal, rc.APPROACH_M, rc.REACH_POSTURES, True)
    for seen in (None, np.zeros((0, 3))):
      judged = _plan(floor, "left", goal, seen)
      assert judged[3] is None and judged[0][0] == today[0][0] and judged[0][3] == today[0][3]
      assert np.array_equal(judged[0][1], today[0][1]) and np.array_equal(judged[0][2], today[0][2])


def test_a_posture_whose_swing_meets_a_face_it_saw_is_passed_over(floor):
  """The premise: across the body from the stand the plan turns the waist,
  and with a counter's face 0.24 m ahead the turn carries the hand into it.
  Behind the hand's goal, 0.40 m ahead, the face is met by no posture; at
  0.24 m one is passed over, and the next that keeps clear is taken."""
  goal = HandGoal(np.array([0.30, -0.07, 0.80]), None, 0.5)
  today = floor._planned("left", goal, rc.APPROACH_M, rc.REACH_POSTURES, True)[0]
  assert floor._planned("left", goal, rc.APPROACH_M, rc.REACH_POSTURES, True)[0][0] == today[0]
  far = _plan(floor, "left", goal, _face(0.40))
  assert far[0][0] == today[0] and far[3] is None
  floor.seen = _face(0.24)
  try:
    met = floor._swing("left", today, goal, floor._lean())
  finally:
    floor.seen = None
  assert met is not None and met["side"] == "left", met
  near = _plan(floor, "left", goal, _face(0.24))
  assert near[0] is not None and near[0][0] != today[0] and near[3] is None
  floor.seen = _face(0.24)
  try:
    assert floor._swing("left", near[0], goal, floor._lean()) is None
  finally:
    floor.seen = None


def test_where_every_posture_swings_into_what_it_saw_a_procedures_reach_moves_nothing():
  """Refused, nothing asked of the body, `swung` saying which arm's part, how
  near and where; the loop's own reach -- a charge's -- never judged, goes."""
  reach = _floor()
  postures = rc.REACH_POSTURES[:12]
  reach.seen = _face(0.18)
  posture, phase = reach.posture.copy(), dict(reach.phase)
  assert not reach.reach("left", ACROSS, postures=postures, verb=True)
  assert reach.phase == phase and np.array_equal(reach.posture, posture)
  said = reach.swung["left"]
  assert said["side"] == "left" and said["part"] in rc.ARM_PARTS and said["m"] < rc.SWING_CLEAR_M
  assert said["at"][0] == pytest.approx(0.18)
  assert reach.reach("left", ACROSS, postures=postures, verb=False)
  assert reach.swung["left"] is None


def test_a_part_begun_nearer_than_the_clearance_goes_no_nearer(floor):
  """A hand hanging by a face as the reach begins (Y6: the left gripper
  resting on the counter at 7-10 N) may stay as near: met only where it goes
  nearer than it began, less `SWING_SLACK_M` -- here voxels a few mm under
  the hanging hand, which a reach to its own side lifts away from."""
  sm = rc.swing_model()
  floor._pose(floor.q_cmd)
  sm.q = floor.self_model.q.copy()
  sm.d.qpos[:] = floor.self_model.d.qpos
  mujoco.mj_kinematics(sm.m, sm.d)
  parts = rc.swing_parts(sm, ("left",))
  c = np.mean([p[3] for p in parts], axis=0)
  g = np.stack(np.meshgrid(np.arange(-0.1, 0.1, 0.005), np.arange(-0.1, 0.1, 0.005),
                           np.arange(-0.2, 0.0, 0.005), indexing="ij"), axis=-1).reshape(-1, 3) + c
  gap = np.min([rc.part_gap(g, p) for p in parts], axis=0)
  low = min(float(p[3][2]) for p in parts)
  under = g[(gap > 0.003) & (gap < 0.008) & (g[:, 2] < low - 0.02)]
  assert len(under) > 20 and np.min([rc.part_gap(under, p) for p in parts]) < rc.SWING_CLEAR_M, \
    "the premise: begun within the clearance"
  today = floor._planned("left", OWN_SIDE, rc.APPROACH_M, rc.REACH_POSTURES, True)[0]
  floor.seen = under
  try:
    assert floor._swing("left", today, OWN_SIDE, floor._lean()) is None
  finally:
    floor.seen = None


def test_the_work_round_the_goal_is_no_face(floor):
  """What lies within `SWING_WORK_M` of its goal is what it reaches for and
  what that stands on: the plan over a face round the goal alone is today's."""
  today = floor._planned("left", OWN_SIDE, rc.APPROACH_M, rc.REACH_POSTURES, True)[0]
  round_it = _face(0.30, y0=0.06, y1=0.14, z0=0.74, z1=0.86)
  assert np.linalg.norm(round_it - OWN_SIDE.pos, axis=1).max() <= rc.SWING_WORK_M + 0.03
  inside = round_it[np.linalg.norm(round_it - OWN_SIDE.pos, axis=1) <= rc.SWING_WORK_M]
  got = _plan(floor, "left", OWN_SIDE, inside)
  assert got[0][0] == today[0] and got[3] is None


# ---- honest -------------------------------------------------------------------------------------


def test_the_swing_is_judged_on_the_robots_model_of_itself_never_the_simulators_truth():
  """A reach refused for its swing, with the world's `MjData` a spy: nothing
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
  reach.seen = _face(0.18)
  spy.read.clear()
  assert not reach.reach("left", ACROSS, postures=rc.REACH_POSTURES[:12], verb=True)
  assert reach.swung["left"] is not None, "the premise: judged and refused"
  assert reach.d is spy, "the premise: the reach meets the world through the spy"
  assert not spy.read & TRUTH, spy.read & TRUTH


# ---- through the body and the verb --------------------------------------------------------------


def test_the_body_judges_a_procedures_reach_alone_and_says_swing():
  """`HumanoidBody._reached`: the memory handed to a verb's plan and taken
  back after; "swing" its why, `hand_swung` what it would meet; the loop's
  own reach planned with nothing seen."""
  said = {"arm": "this", "side": "left", "part": "gripper", "m": 0.004, "began": None,
          "at": [0.22, 0.31, 0.66], "t": 0.6}
  handed = []

  class Arms:
    kept, own, kept_by, keep_out = {"left": None}, {"left": None}, {"left": None}, []
    swung, seen = {"left": None}, None

    def reach(self, side, goal, **kw):
      handed.append(self.seen)
      self.swung[side] = said if kw.get("verb") else None
      return not kw.get("verb")
  wall = _face(0.2)
  for verb in (True, False):
    arms = Arms()
    me = NS(reach=arms, _why={}, _kept={},
            mission=NS(look_round=lambda: None, keep_out=lambda seen: [], arms_seen=lambda: wall))
    went = hb.HumanoidBody._reached(me, "left", None, (), False, words=verb, verb=verb)
    assert went is (not verb) and arms.seen is None
    assert (handed[-1] is wall) if verb else handed[-1] is None
    if verb:
      assert me._why["left"] == "swing"
      assert hb.HumanoidBody.hand_swung(me, "left") == said


def test_a_reach_refused_for_its_swing_says_what_it_would_meet_and_whether_the_other_hand_could():
  class Swung(StubBody):
    reaches = False

    def reach_routine(self, side, pos, rot=None, stiffness=1.0, patience=20.0, refuse=False):
      return False
      yield

    def hand_why(self, side):
      return "swing"

    def hand_reaches(self, side, pos, rot=None, verb=False):
      return self.reaches
  for reaches in (False, True):
    body = Swung(*_bare())
    body.reaches = reaches
    body.swung["left"] = {"side": "left", "part": "jaw_a_pad", "m": 0.0, "at": [0.214, 0.3, 0.664]}
    step = _fly('def a():\n    reach("left", 0.3, -0.2, 0.8)\n', body)["steps"][-1]
    assert not step["ok"] and step["why"] == "swing"
    said = st.REACH_SWUNG.format(hand="left", side="left", part="fingers", ahead=0.214, up=0.664)
    assert step["reason"] == said + ("; your right hand could reach it as asked" if reaches else "")
