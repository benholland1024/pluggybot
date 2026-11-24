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


def _face(x, turn=0.0, y0=-0.6, y1=0.6, z0=0.1, z1=0.68):
  """A counter's face as the memory gives it (`ArmsMemory.faces`), x m ahead
  of the stance along its facing, the stance turned `turn` deg to its right
  of square: 2 cm voxels, its top's own 3 cm left out."""
  t = math.radians(turn)
  ss, zs = np.arange(y0, y1 + 1e-9, 0.02), np.arange(z0, z1 + 1e-9, 0.02)
  return np.array([[x / math.cos(t) - s * math.sin(t), s * math.cos(t), z] for s in ss for z in zs])


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


def test_what_it_saw_where_its_arms_swing_rides_a_restart():
  """The memory goes into the body's kept state and comes back with it
  (#345: a new piece of state that decides anything); a save from before
  #747 comes back with none."""
  from test_guard import _floor_body
  body = _floor_body()
  try:
    mi = body.mission
    face = np.array([[0.2, y, z] for y in np.arange(-0.6, 0.6, 0.02) for z in np.arange(0.1, 0.8, 0.02)])
    mi.swing.fold(face, float(body._d.time), mi.pose, mi.odo.corrected)
    pts, ts = mi.swing.pts.copy(), mi.swing.t.copy()
    assert len(pts) > 100
    state, arrays = body.kept_state()
    mi.swing.forget()
    body.restore_kept(state, arrays)
    assert np.array_equal(mi.swing.pts, pts) and np.array_equal(mi.swing.t, ts)
    body.restore_kept(state, {k: v for k, v in arrays.items() if not k.startswith("arms")})
    assert not len(mi.swing.pts), "a save from before it: nothing seen"
  finally:
    body.close()


# ---- the judgment -------------------------------------------------------------------------------


def test_with_nothing_seen_a_procedures_plan_is_todays_bit_for_bit(floor):
  for goal in (ACROSS, OWN_SIDE):
    today = floor._planned("left", goal, rc.APPROACH_M, rc.REACH_POSTURES, True)
    for seen in (None, np.zeros((0, 3))):
      judged = _plan(floor, "left", goal, seen)
      assert judged[3] is None and judged[0][0] == today[0][0] and judged[0][3] == today[0][3]
      assert np.array_equal(judged[0][1], today[0][1]) and np.array_equal(judged[0][2], today[0][2])


def test_a_posture_whose_swing_meets_a_face_it_saw_is_passed_over(floor):
  """The premise: 7 cm across the body from the stand the plan crouches, and
  with a counter's face 0.15 m ahead, the stance turned 20 deg to its right
  of square as the robots stood at the cube (#747), the crouch carries the
  hanging hand into it; that posture is passed over, and the next that
  keeps clear is taken. Behind the goal, 0.40 m ahead, nothing changes."""
  goal = HandGoal(np.array([0.30, -0.07, 0.80]), None, 0.5)
  postures = rc.REACH_POSTURES[:12]
  today = floor._planned("left", goal, rc.APPROACH_M, postures, True)[0]
  far = _plan(floor, "left", goal, _face(0.40, 20), postures)
  assert far[0][0] == today[0] and far[3] is None
  face = _face(0.15, 20)
  floor.seen = face
  try:
    met = floor._swing("left", today, goal, floor._lean())
  finally:
    floor.seen = None
  assert met is not None and met["side"] == "left" and met["m"] < rc.SWING_CLEAR_M, met
  near = _plan(floor, "left", goal, face, postures)
  assert near[0] is not None and near[0][0] != today[0] and near[3] is None
  floor.seen = face
  try:
    assert floor._swing("left", near[0], goal, floor._lean()) is None
  finally:
    floor.seen = None


def test_a_reach_whose_plan_passed_over_a_posture_takes_no_look():
  """A plan that passed over a posture for its swing is the one its hand
  goes in from: its look takes none (`looks` spent) -- taken back by the
  look, the 0.55 bow the plan passed over went down in the approach, and
  kept off those, the look went on to the deepest bows, whose trunk and head
  met the counter (#747's census). Where nothing was passed over the look
  is as it was; the loop's own reach passes over nothing."""
  reach = _floor()
  goal = HandGoal(np.array([0.30, -0.07, 0.80]), None, 0.5)
  postures = rc.REACH_POSTURES[:12]
  reach.seen = _face(0.15, 20)
  assert reach.reach("left", goal, postures=postures, verb=True)
  assert reach.looks["left"] == rc.LOOKS and reach.seen_at["left"] is reach.seen
  reach.seen = None
  floor = _floor()
  floor.seen = _face(0.15, 20)
  assert floor.reach("left", goal, postures=postures, verb=False)
  assert floor.looks["left"] == 0 and floor.seen_at["left"] is None
  far = _floor()
  far.seen = _face(0.40, 20)
  assert far.reach("left", goal, postures=postures, verb=True)
  assert far.looks["left"] == 0


def test_a_look_passes_over_a_posture_whose_swing_meets_what_it_saw():
  """A look at the pre-grasp (#627) judges the posture it would take as the
  plan's are judged, its hand held where it waits (`_swing`'s `hold`): a
  crouch it would take lowers the other, hanging hand onto what was seen
  under it, so it keeps the posture it holds; unseen, it takes the crouch."""
  reach = _floor()
  reach._pose(reach.q_cmd)
  sm = rc.swing_model()
  sm.q = reach.self_model.q.copy()
  sm.d.qpos[:] = reach.self_model.d.qpos
  mujoco.mj_kinematics(sm.m, sm.d)
  right = [p for p in rc.swing_parts(sm, ("right",)) if p[1] == "gripper"][0]
  under = _around(right, 0.03, 0.06)
  under = under[under[:, 2] < right[3][2] - 0.03]
  left = [p for p in rc.swing_parts(sm, ("left",)) if p[1] == "gripper"][0]
  # ...and under the hand that waits: held at its pre-grasp, it goes nowhere
  below = _around(left, 0.03, 0.06)
  below = below[below[:, 2] < left[3][2] - 0.03]
  here = reach.posture.copy()
  for seen, takes in ((None, True), (under, False), (below, True)):
    reach.goal["left"] = HandGoal(np.array([0.30, 0.10, 0.70]), None, 1.0)
    reach.cmd["left"] = HandGoal(*reach.self_model.grip_pose("left"), 1.0)
    reach.allowed["left"], reach.left["left"], reach.looks["left"] = [(-0.16, 0.0, 0.0)], [], 0
    reach.seen_at["left"] = seen
    stays = reach._look("left")
    assert stays is (not takes) and tuple(reach.posture) == ((-0.16, 0.0, 0.0) if takes else tuple(here))
    reach.hold_posture(*here)


def test_where_every_posture_swings_into_what_it_saw_a_procedures_reach_moves_nothing():
  """Refused, nothing asked of the body, `swung` saying which arm's part, how
  near and where; the loop's own reach -- a charge's -- never judged, goes."""
  reach = _floor()
  postures = rc.REACH_POSTURES[:12]
  reach.seen = _face(0.15, 20)
  posture, phase = reach.posture.copy(), dict(reach.phase)
  assert not reach.reach("left", ACROSS, postures=postures, verb=True)
  assert reach.phase == phase and np.array_equal(reach.posture, posture)
  said = reach.swung["left"]
  assert said["side"] == "left" and said["part"] in rc.ARM_PARTS and said["m"] < rc.SWING_CLEAR_M
  assert any(np.allclose(said["at"], p, atol=1e-3) for p in _face(0.15, 20))
  assert reach.reach("left", ACROSS, postures=postures, verb=False)
  assert reach.swung["left"] is None


def _hand_parts(floor):
  """The hanging left hand's parts as the reach begins (`swing_parts`), the
  body as it reads itself, its arms where they are commanded."""
  sm = rc.swing_model()
  floor._pose(floor.q_cmd)
  sm.q = floor.self_model.q.copy()
  sm.d.qpos[:] = floor.self_model.d.qpos
  mujoco.mj_kinematics(sm.m, sm.d)
  return [p for p in rc.swing_parts(sm, ("left",)) if p[1] == "gripper"][0]


def _around(part, lo, hi):
  """Points a few cm round a part whose gap to it lies in (lo, hi), m."""
  g = np.stack(np.meshgrid(*(np.arange(-0.08, 0.08, 0.004),) * 3, indexing="ij"),
               axis=-1).reshape(-1, 3) + part[3]
  gap = rc.part_gap(g, part)
  return g[(gap > lo) & (gap < hi)]


def test_a_part_begun_further_in_than_the_clearance_goes_no_further(floor, monkeypatch):
  """A hand hanging on a face as the reach begins (Y6: the left gripper
  resting on the counter at 7-10 N) may stay as far in: met only where it
  goes further than it began, less `SWING_SLACK_M`. Its swing held still
  (the clock's commands all where they are), voxels 7-12 mm into its
  gripper are no meeting; a crouch of 10 cm carries it 3-6 cm into voxels
  under it."""
  gripper = _hand_parts(floor)
  today = floor._planned("left", OWN_SIDE, rc.APPROACH_M, rc.REACH_POSTURES, True)[0]
  inside = _around(gripper, -0.012, -0.007)
  assert len(inside) > 20 and rc.SWING_CLEAR_M > -0.007, "the premise: begun further in"

  def still(shown, posture, w, ws, q0, ways, tick, every=rc.SWING_DT, **kw):
    return [(k * every, float(shown[0]), float(shown[1]), float(w), np.asarray(q0, dtype=float))
            for k in range(3)]
  monkeypatch.setattr(rc, "swing_clock", still)
  floor.seen = inside
  try:
    assert floor._swing("left", today, OWN_SIDE, floor._lean()) is None
  finally:
    floor.seen = None
  # ...and carried further in: a crouch, the arm where it hangs
  under = _around(gripper, 0.03, 0.06)
  under = under[under[:, 2] < gripper[3][2] - 0.03]

  def crouch(shown, posture, w, ws, q0, ways, tick, every=rc.SWING_DT, **kw):
    return [(k * every, float(shown[0]) - 0.1 * k / 4, float(shown[1]), float(w),
             np.asarray(q0, dtype=float)) for k in range(5)]
  monkeypatch.setattr(rc, "swing_clock", crouch)
  floor.seen = under
  try:
    met = floor._swing("left", ((-0.1, 0.0, 0.0), *today[1:]), OWN_SIDE, floor._lean())
  finally:
    floor.seen = None
  assert met is not None and met["side"] == "left" and met["m"] < rc.SWING_CLEAR_M, met


def test_the_work_round_the_goal_is_no_face(floor, monkeypatch):
  """What lies within `SWING_WORK_M` across its goal, from `SWING_WORK_UNDER_M`
  under it up, is what it reaches for and what that stands on (the robots'
  goals hover 2.4 cm over the cube's board): voxels inside the hand as it
  waits at its pre-grasp, there, are no meeting -- with no work, they are."""
  today = floor._planned("left", OWN_SIDE, rc.APPROACH_M, rc.REACH_POSTURES, True)[0]
  sm = rc.swing_model()
  q0 = np.array([hm.stand_qpos().get(n, 0.0) for n in hm.JOINT_NAMES])
  rc.statue(sm, today[0], q0, {**floor._arms(but="left"), **rc._arm("left", today[1])}, floor._lean(),
            fast=True)
  sm._set_arm("left", today[1])
  gripper = [p for p in rc.swing_parts(sm, ("left",)) if p[1] == "gripper"][0]
  inside = _around(gripper, -0.02, -0.01)
  g = OWN_SIDE.pos
  inside = inside[(np.hypot(inside[:, 0] - g[0], inside[:, 1] - g[1]) <= rc.SWING_WORK_M)
                  & (inside[:, 2] >= g[2] - rc.SWING_WORK_UNDER_M)]
  assert len(inside) > 20, "the premise: the hand at its pre-grasp lies in its work"
  got = _plan(floor, "left", OWN_SIDE, inside)
  assert got[0][0] == today[0] and got[3] is None
  monkeypatch.setattr(rc, "SWING_WORK_M", 0.0)
  floor.seen = inside.copy()
  try:
    assert floor._swing("left", today, OWN_SIDE, floor._lean()) is not None
  finally:
    floor.seen = None


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
  reach.seen = _face(0.15, 20)
  spy.read.clear()
  assert not reach.reach("left", ACROSS, postures=rc.REACH_POSTURES[:12], verb=True)
  assert reach.swung["left"] is not None, "the premise: judged and refused"
  assert reach.d is spy, "the premise: the reach meets the world through the spy"
  assert not spy.read & TRUTH, spy.read & TRUTH


def test_what_it_saw_is_placed_and_asked_off_its_own_senses_never_the_simulators_truth():
  """The memory placed into the stance frame and a procedure's plan judged
  against it (`arms_seen`, `Body.hand_reaches(verb=True)`), with the body's
  every view of the world's `MjData` a spy: nothing read of where the world
  says the bodies are or what they touch."""
  from test_guard import _floor_body
  from test_humanoid import TRUTH, _Spy
  body = _floor_body()
  try:
    mi = body.mission
    # ...a face 0.2 m ahead, folded as a frame folds it (from the pelvis along its heading)
    face = np.array([[0.2, y, z] for y in np.arange(-0.6, 0.6, 0.02) for z in np.arange(0.1, 0.8, 0.02)])
    mi.swing.fold(face, float(body._d.time), mi.pose, mi.odo.corrected)
    data = body._d
    spy = _Spy(data)
    holders = [body, mi, body.reach, body.stand, body.stand.trunk]
    swapped = [(o, k) for o in holders for k, v in vars(o).items() if v is data]
    for o, k in swapped:
      setattr(o, k, spy)
    try:
      seen = mi.arms_seen()
      body.hand_reaches("left", (0.30, -0.20, 0.80), verb=True)
    finally:
      for o, k in swapped:
        setattr(o, k, data)
    assert len(seen) > 100 and swapped, "the premise: it saw the face, through the spy"
    assert not spy.read & TRUTH, spy.read & TRUTH
  finally:
    body.close()


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
