"""Carrying while walking (issue #682's stage A; Ben's calls of 2026-11-12;
SimNotes, "Carrying while walking, measured", and "Carrying, served"), each
rule pinned without a flight -- a fake reach, the stub body, a direct call
(docs/Testing.md):

  1. a walk's hand-over CARRIES a hand that is out whose gripper says it
     holds something -- lifted first, and carried only once it came away;
     its gripper as it was, its arm to the carry pose (`Reach.CARRY_POSE`,
     Ben's `close`) its own way -- and lets go of one that holds nothing or
     whose lift was held fast, as #669's does; an arm at its carry pose
     already goes nowhere;
  2. the carry pose keeps the hand turned as it is against the floor,
     solved on the statue of the stand -- where the walk carries it -- never
     the posture it holds now;
  3. a withdraw and a let-go end a carry: home is the rest again;
  4. the self-filter: the LIDAR's band and the depth camera's low layer
     leave out every return within `HELD_DROP_M` of a grip that holds,
     placed off the robot's model of itself -- and keep it with the grip
     empty;
  5. with another robot by, a held arm's way judged is the way to its carry
     pose (#678's rule composed), one already carried judged not at all;
     refused, the walk verbs say its way to your side;
  6. a walk verb's report says what it carried, and what it dropped.
"""

import math

import mujoco
import numpy as np
import pytest

from pluggybot.body import body_for
from pluggybot.humanoid import mission as hmis
from pluggybot.humanoid import model as hm
from pluggybot.humanoid import reach as rc
from pluggybot.humanoid.arms import SelfModel
from pluggybot.perception.lidar import BEARINGS, Cloud
from pluggybot.procedure import steps as st
from test_hands import _Life, _stub
from test_humanoid_body import _Out, _floor


# ---- 1: the walk's hand-over carries a held grip ----------------------------------


def _spy_opens(monkeypatch, body, log):
  from pluggybot.humanoid.arms import Gripper
  opened = Gripper.open

  def spied(g, *args, **kwargs):
    log.append(("open", next(s for s, h in body.mission.reach.grippers.items() if h is g),
                float(body.data.time)))
    opened(g, *args, **kwargs)
  monkeypatch.setattr(Gripper, "open", spied)


def test_a_walk_carries_a_hand_that_holds_and_lets_go_of_one_that_holds_nothing(monkeypatch):
  """#669 opened every hand that was out before a walk: a plate lifted for
  the kitchen's shelf would have been dropped at the walk's first step.
  Ben's call 2: the walk takes the carry pose itself -- the hand whose
  gripper reads held keeps its grip and goes to its carry pose its own way
  (out along its line, then home's way, ending at the pose), and the hand
  that holds nothing lets go and comes home, as before."""
  body = body_for(*_floor())
  m = body.mission
  log = []
  _spy_opens(monkeypatch, body, log)
  at = lambda: float(body.data.time)  # noqa: E731
  grippers = m.reach.grippers
  m.reach = _Out(grippers, ("left", "right"), log, at)
  for g in grippers.values():
    g.shut()
  monkeypatch.setattr(grippers["left"], "held", lambda: True)
  monkeypatch.setattr(grippers["right"], "held", lambda: False)
  body.run(m.to_walk_routine())
  assert m.legs == "walk"
  said = [e[:2] for e in log]
  assert ("open", "left") not in said, "the walk let go of what the hand held"
  assert said.index(("open", "right")) < said.index(("retreat", "right")), said
  assert said.index(("lift", "left")) < said.index(("retreat", "left")), \
    "the held arm was walked where the reach left it, or never lifted first"
  assert ("lift", "right") not in said, "a hand that holds nothing was lifted"
  assert m.reach.carry_q["left"] is not None and m.reach.carry_q["right"] is None
  assert m.reach.carried("left") and m.reach.phase["right"] == "rest"
  assert not grippers["left"].opened() and grippers["right"].opened()


def test_a_hand_whose_lift_is_held_fast_is_let_go_never_carried(monkeypatch):
  """What a hand holds is carried only once a lift shows it comes away:
  pulled up 2 cm at the most the standing body holds, a drawer's handle or
  a door's lever is held fast, and the walk lets go and brings the arm home,
  #669's way -- carried, a drawer's handle held came out 19 cm with the arm
  (flown, `held_handle.py`)."""
  body = body_for(*_floor())
  m = body.mission
  log = []
  _spy_opens(monkeypatch, body, log)
  at = lambda: float(body.data.time)  # noqa: E731
  grippers = m.reach.grippers
  m.reach = _Out(grippers, ("left",), log, at, lifts={"left": "held fast"})
  grippers["left"].shut()
  monkeypatch.setattr(grippers["left"], "held", lambda: True)
  body.run(m.to_walk_routine())
  said = [e[:2] for e in log]
  assert said == [("lift", "left"), ("open", "left"), ("retreat", "left")], said
  assert m.reach.carry_q["left"] is None and m.reach.phase["left"] == "rest"
  assert m.legs == "walk"


def test_an_arm_at_its_carry_pose_already_goes_nowhere(monkeypatch):
  """A second walk of a carry (a step after a step): the arm held at its
  carry pose stays there, its gripper shut -- no let-go, no way walked."""
  body = body_for(*_floor())
  m = body.mission
  log = []
  _spy_opens(monkeypatch, body, log)
  at = lambda: float(body.data.time)  # noqa: E731
  grippers = m.reach.grippers
  m.reach = _Out(grippers, (), log, at)
  m.reach.phase["left"] = "carried"
  m.reach.carry_q["left"] = np.array([0.15, 0.25, 0.0, 2.0, -1.2, -1.1])
  grippers["left"].shut()
  monkeypatch.setattr(grippers["left"], "held", lambda: True)
  body.run(m.to_walk_routine())
  assert m.legs == "walk" and log == [], log
  assert m.reach.carried("left")
  assert m.reach.carry_q["left"].tolist() == [0.15, 0.25, 0.0, 2.0, -1.2, -1.1]


def test_a_let_go_and_a_withdraw_end_a_carry():
  """A hand let go carries nothing: its way home is its rest again -- and a
  withdraw of a carried hand comes home, its way home judged to its rest
  (#678's judgment comes before the let-go), never back to its carry pose."""
  from test_arms_keep_off import _Arms, _beside
  body = body_for(*_floor())
  m = body.mission
  try:
    m.reach.carry_q["left"] = np.zeros(6)
    body.run(m.let_go_routine(("left",)))
    assert m.reach.carry_q["left"] is None
    assert np.allclose(m.reach.home_to("left"), m.reach.rest["left"])
    _beside(body)
    m.reach = _Arms(m.reach.grippers, ("right",), lambda: True, [],
                    lambda: float(body.data.time))
    m.reach.phase["right"], m.reach.carry_q["right"] = "carried", np.zeros(6)
    body.run(body.release_routine("right", refuse=True))
    assert m.reach.toward == [("right", None)], "a withdraw judged its way to the carry pose"
    assert m.reach.carry_q["right"] is None
  finally:
    body.close()


def test_a_new_reach_ends_a_carry():
  """A hand reached out again carries no longer: its way home is its rest."""
  body = body_for(*_floor())
  r = body.mission.reach
  try:
    r.carry_q["left"] = np.zeros(6)
    r.set_joints("left", r.q_cmd["left"])
    assert r.carry_q["left"] is None
  finally:
    body.close()


# ---- 2: the carry pose -------------------------------------------------------------


def _hand_up(sm: SelfModel, side: str) -> np.ndarray:
  return sm.d.site_xmat[sm.grip[side]].reshape(3, 3).T @ np.array([0.0, 0.0, 1.0])


@pytest.mark.parametrize("side", hm.SIDES)
def test_the_carry_pose_keeps_the_hand_turned_against_the_floor_solved_standing(side):
  """The wrist keeps what the hand holds as it is: the floor's up in the
  hand's frame after as before. ⚠ Solved where the walk carries it, on the
  statue of the stand -- solved on the model as it stood bowed for a reach,
  a plate lifted from a 0.3 rad bow was carried 15-17 deg tipped once the
  trunk stood up (flown, `carry_spike.py --served`)."""
  body = body_for(*_floor())
  r = body.mission.reach
  try:
    q0 = np.array([hm.stand_qpos().get(n, 0.0) for n in hm.JOINT_NAMES])
    # the hand as the plate's lift leaves it (`carry_spike.py`'s level wrist
    # at `close`), its turn read standing...
    want = rc.seed(side, (*rc.CARRY_POSE, -1.39, -0.98))
    stand = SelfModel()
    rc.statue(stand, (0.0, 0.0, 0.0), q0)
    stand._set_arm(side, want)
    turn = stand.grip_pose(side)[1].copy()
    up = _hand_up(stand, side)
    # ...while the robot's model of itself still stands in a reach's bow
    rc.statue(r.self_model, (-0.16, 0.3, 0.0), q0)
    r.grip_now = lambda s: (np.zeros(3), turn)
    q = r.carry_pose(side)
    assert np.allclose(q[:4], rc.seed(side, (*rc.CARRY_POSE, 0.0, 0.0))[:4])
    rc.statue(stand, (0.0, 0.0, 0.0), q0)
    stand._set_arm(side, q)
    off = math.degrees(math.acos(min(1.0, float(_hand_up(stand, side) @ up))))
    assert off < 1.0, f"the hand turned {off:.1f} deg against the floor"
  finally:
    body.close()


# ---- 4: the self-filter ------------------------------------------------------------


def _bare_lidar():
  from pluggybot.perception.lidar import Lidar
  m = mujoco.MjModel.from_xml_string(hm.body_xml())
  return Lidar(m, robot_body=hm.ROBOT_ROOT, noise=False)


def _cloud(lidar, near: float, far: float, bearing: int = 0) -> Cloud:
  """One bearing's rays: the lowest ring's struck `near` m out, as a plate
  held before the body is, every other ring's `far` m out at the band's
  middle height: in the sensor's frame, level, its turn the identity's."""
  n_el = len(lidar.elevations)
  rays = np.array([k * BEARINGS + bearing for k in range(n_el)])
  az = lidar.ray_angles[bearing]
  pts = []
  for k in range(n_el):
    r = near if k == 0 else far
    pts.append((r * math.cos(az), r * math.sin(az), -0.5 if k == 0 else 0.0))
  looked = np.zeros(n_el * BEARINGS, dtype=bool)
  looked[rays] = True
  return Cloud(np.array(pts), np.zeros((0, 3)), np.zeros(BEARINGS, dtype=bool), looked,
               rays, np.zeros(0, dtype=np.int64), np.zeros(n_el * BEARINGS, dtype=bool),
               np.zeros(0, dtype=np.int64))


def test_the_band_leaves_out_what_a_held_grip_holds_and_keeps_it_with_the_grip_empty():
  """Ben's call 5: a return within the volume of a grip that holds is the
  robot's own -- neither a point nor a ray that went on, so the bearing
  reads the wall behind the piece -- and the same return with no drop given
  (the grip empty) is a point as before: the front stop backed a robot off
  its own plate, 9-13 m backward (T7, 11-11)."""
  lidar = _bare_lidar()
  joints = {j: 0.0 for j in hm.WAIST}
  offset, turn = lidar.mount.pose(joints)
  height = 0.7
  lidar.cloud = lambda data: _cloud(lidar, 0.35, 3.0)
  plain = lidar.band(None, np.eye(3), height, joints=joints)
  # where the near return lies, levelled as `band` lays it
  near = _cloud(lidar, 0.35, 3.0).points[0] @ turn.T + offset
  centre = np.array([[near[0], near[1], height + near[2]]])
  dropped = lidar.band(None, np.eye(3), height, joints=joints,
                       drop=(centre, hmis.HELD_DROP_M))
  assert len(plain.ranges) and min(plain.ranges) < 0.5, "the premise: the plate is read"
  assert len(dropped.ranges) == 1 and dropped.ranges[0] == pytest.approx(3.0, abs=0.01), \
    "the held piece's return stood in the scan, or freed its bearing"
  far = lidar.band(None, np.eye(3), height, joints=joints,
                   drop=(centre + [[0.0, 0.0, hmis.HELD_DROP_M + 0.05]], hmis.HELD_DROP_M))
  assert min(far.ranges) < 0.5, "a return outside the volume was dropped too"
  # ...and a bearing the held piece was all its rays saw reads nothing --
  # never free to max range through what stands behind it: the level ring's
  # ray, stopped by the piece, would have stayed in the band to 8 m
  level = int(np.argmin(np.abs(lidar.elevations)))
  ray = level * BEARINGS
  only = Cloud(np.array([[0.3, 0.0, 0.0]]), np.zeros((0, 3)), np.zeros(BEARINGS, dtype=bool),
               np.isin(np.arange(len(lidar.elevations) * BEARINGS), [ray]), np.array([ray]),
               np.zeros(0, dtype=np.int64), np.zeros(len(lidar.elevations) * BEARINGS, dtype=bool),
               np.zeros(0, dtype=np.int64))
  lidar.cloud = lambda data: only
  piece = only.points[0] @ turn.T + offset
  held = lidar.band(None, np.eye(3), height, joints=joints,
                    drop=(np.array([[piece[0], piece[1], height + piece[2]]]), hmis.HELD_DROP_M))
  assert len(held.ranges) == 0, f"the piece's ray freed its bearing: {held.ranges}"


def test_the_mission_places_a_held_grip_off_its_model_of_itself_and_drops_it_from_its_layer():
  """The volume's middle is the grip as the robot's model of itself places
  it (its odometry's model, posed off its encoders and IMU), levelled as a
  scan's points are -- within a few cm of the truth -- only for a gripper
  that reads held; and the depth camera's low layer leaves out a point
  there that it folds in with the grip empty."""
  body = body_for(*_floor())
  m = body.mission
  try:
    body.run(body.hold_routine(0.2))
    assert m._held_drop() is None, "nothing held, nothing dropped"
    m.reach.grippers["left"].held = lambda: True
    d, model = m.data, m.model
    root = model.body(hm.ROBOT_ROOT).id

    def truth():
      r = d.xmat[root].reshape(3, 3)
      yaw = math.atan2(r[1, 0], r[0, 0])
      grip = d.site_xpos[model.site("left_grip").id]
      v = grip - d.xpos[root]
      # ...z over the bare world's floor, its plane at 0
      return np.array((math.cos(yaw) * v[0] + math.sin(yaw) * v[1],
                       -math.sin(yaw) * v[0] + math.cos(yaw) * v[1], grip[2]))
    centres, radius = m._held_drop()
    assert radius == hmis.HELD_DROP_M and centres.shape == (1, 3)
    assert np.linalg.norm(centres[0] - truth()) < 0.03, (centres[0], truth())
    # ...levelled along the PELVIS's heading, as a scan's points are -- its
    # soles turned 0.25 rad under it, the stance frame is no longer its frame
    sm = m.odo.sm
    q = d.qpos[sm.qadr].copy()
    for s in hm.SIDES:
      d.qpos[model.jnt_qposadr[model.joint(f"{s}_hip_yaw").id]] = 0.25
      q[hm.JOINT_NAMES.index(f"{s}_hip_yaw")] = 0.25
    mujoco.mj_forward(model, d)
    sm.pose(q, np.array([0.0, 0.0, -1.0]))
    centres, _ = m._held_drop()
    assert np.linalg.norm(centres[0] - truth()) < 0.03, (centres[0], truth())
    # ...and the low layer: a point at the grip folds in only unheld
    point = np.array([centres[0]])
    m.low[...] = 0.0
    m._fold_low(point)
    assert not (m.low != 0.0).any(), "the held grip's point went into the layer"
    m.reach.grippers["left"].held = lambda: False
    m._fold_low(point)
    assert (m.low > 0.0).any(), "the premise: unheld, the point is an obstacle"
  finally:
    body.close()


# ---- 5: composed with #678's rule --------------------------------------------------


def test_a_held_arms_way_judged_is_the_way_to_its_carry_pose(monkeypatch):
  """#678: a walk refused with a hand out whose every way home would meet
  another robot's body. A hand that holds goes not home but to its carry
  pose (#682), so that way is the one judged -- and an arm carried there
  already moves no more, so it is not judged at all; refused for a carried
  arm, the walk verbs say its way to your side."""
  from test_arms_keep_off import _Arms, _beside
  body = body_for(*_floor())
  m = body.mission
  at = lambda: float(body.data.time)  # noqa: E731
  grippers = m.reach.grippers
  _beside(body)
  m.reach = _Arms(grippers, ("left", "right"), lambda: True, [], at)
  monkeypatch.setattr(grippers["left"], "held", lambda: True)
  monkeypatch.setattr(grippers["right"], "held", lambda: False)
  assert body.arm_kept() is None
  ends = dict(m.reach.toward)
  assert ends["left"] == m.reach.carry_pose("left").tolist() and ends["right"] is None
  # ...one carried already is judged not at all
  m.reach.toward.clear()
  m.reach.phase["left"] = "carried"
  m.reach.carry_q["left"] = m.reach.carry_pose("left")
  body.arm_kept()
  assert [s for s, _ in m.reach.toward] == ["right"]
  # ...and refused, the carried arm's words
  m.reach = _Arms(grippers, ("left",), lambda: False, [], at)
  kept = body.arm_kept()
  assert kept["hand"] == "left" and kept["carried"]
  m.reach = _Arms(grippers, ("left",), lambda: False, [], at)
  said = st._arm_out(_Life(body), "step")["reason"]
  assert said == st.ARM_KEPT_OFF["step carried"].format(where=st._where(kept), **kept)
  assert "its way to your side, where a step carries it" in said


# ---- 6: what a walk's report says --------------------------------------------------


def test_a_step_holding_something_walks_and_its_report_says_what_it_carried():
  """`step` walked nothing holding something (`holding`, #525); now it walks,
  its hand-over carrying the hand that holds, and its report says so
  (`carried`) -- and `dropped`, a carried hand whose gripper holds nothing by
  the walk's end."""
  from pluggybot.procedure import lang
  body = _stub()
  body.hands["left"] = (0.4, 0.2, 0.8)
  body.grips["left"] = True
  facts = st.hand_facts()
  proc = lang.compile_procedure('def a():\n  step(0.2)\n  r = read("step.dx")\n', facts)
  ran = body.run(lang.run_procedure_routine(_Life(body), proc, facts))
  rep = ran["steps"][0]
  assert rep["ok"] and rep["why"] == "stepped" and rep["carried"] == ["left"], rep
  assert "dropped" not in rep and body.stepped == [(0.2, 0.0, 0.0)]
  assert ran["locals"]["r"] == pytest.approx(0.2)
  body.grips["left"] = False
  ran = body.run(lang.run_procedure_routine(_Life(body), proc, facts))
  assert ran["steps"][0]["dropped"] == ["left"]
