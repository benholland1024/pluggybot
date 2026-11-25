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
from pluggybot.humanoid.arms import HandGoal, SelfModel
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
    assert m.reach.toward == [("right", m.reach.rest["right"].tolist())], \
      "a withdraw judged its way to the carry pose"
    assert m.reach.carry_q["right"] is None
    # ...and the loop's own release (a charge's), which waits for its way home
    # rather than refuse it, judges it home too: its carry ended first
    m.reach.toward.clear()
    m.reach.phase["right"], m.reach.carry_q["right"] = "carried", np.zeros(6)
    body.run(body.release_routine("right"))
    assert m.reach.toward and all(end is None for _, end in m.reach.toward), m.reach.toward
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
    # ...while the robot's model of itself still stands in a reach's bow --
    # the turn its command keeps, and no joint read again (a read draws on
    # the drivers' noise: a check before a walk would move the flight)
    rc.statue(r.self_model, (-0.16, 0.3, 0.0), q0)
    r.cmd[side] = HandGoal(np.zeros(3), turn, 1.0)
    reads = []
    real_read = r.stand.joints.read
    r.stand.joints.read = lambda d: reads.append(1) or real_read(d)
    q = r.carry_pose(side)
    assert reads == [], "the carry pose read the joints again"
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
    # ...and the low layer: a point at the grip folds in only unheld (its
    # frame filtered once, `_held_out`, for the layer and the knees' memory)
    point = np.array([centres[0]])
    m.low[...] = 0.0
    m._fold_low(hmis._held_out(point, m._held_drop()))
    assert not (m.low != 0.0).any(), "the held grip's point went into the layer"
    m.reach.grippers["left"].held = lambda: False
    m._fold_low(hmis._held_out(point, m._held_drop()))
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
  # ...a hand carried whose grip reads empty by the walk's end was dropped
  # (the stub's walk is at once: its grip read as the walk ends)
  life = _Life(body)
  body.grips["left"] = False
  assert st._carried(life) == {"carried": ["left"], "dropped": ["left"]}
  # ...and the next walk carries it no longer: its hand-over let it go
  ran = body.run(lang.run_procedure_routine(_Life(body), proc, facts))
  assert "carried" not in ran["steps"][0] and not body.carrying("left")


def test_a_walks_report_says_an_arm_it_went_with_where_its_way_home_was_held():
  """#693's family: an arm whose way home something held -- raised and sent
  home again, and held still -- stays where it is held (`Body.arm_held`),
  and a walk goes with it there: its report says so (`armHeld`), and says
  nothing of it where no arm is."""
  from pluggybot.procedure import lang
  body = _stub()
  facts = st.hand_facts()
  proc = lang.compile_procedure('def a():\n  step(0.2)\n', facts)
  ran = body.run(lang.run_procedure_routine(_Life(body), proc, facts))
  assert "armHeld" not in ran["steps"][0]
  body.held_arms = {"right"}
  ran = body.run(lang.run_procedure_routine(_Life(body), proc, facts))
  rep = ran["steps"][0]
  assert rep["ok"] and rep["armHeld"] == ["right"], rep


def test_an_arm_left_held_is_saved_through_but_never_rested_with():
  """An arm left where its way home was held (phase "held") keeps its
  command where it is held, kept across a restart (`Reach.KEPT`), and moves
  until a withdraw or a walk: the world saves through it, as through a
  carried hand (#682) -- but the robot does not rest with it, so is never
  asked aside: an aside takes no arm home first, and would walk it there."""
  from pluggybot.humanoid import reach as rc
  body = body_for(*_floor())
  try:
    body.mission.reach.phase["left"] = "held"
    assert not body.working
    assert body.mission._hand_out()
    # ...but it moves nothing: the heading is held as the robot stands
    # (#693b's fourth review: counted busy, the hold never came)
    m = body.mission
    assert m._busy() and not m._busy(still=True)
    m._moved_t = m.last_motion_t = float(m.data.time) - 100.0
    assert m._settled()
    assert {"homing", "held_for", "lifted"} <= set(rc.Reach.KEPT)
  finally:
    body.close()


# ---- the review's: a carry rests, rides a restart, is bounded and judged home ------


def test_a_hand_carried_at_its_side_is_no_hand_at_work():
  """A carry lasts until a reach or a let-go: counted as a hand out, the
  world was never saved through it (`Keeper.busy`), and the robot never
  rested to be asked aside (#682's review). Held at its side as a hanging
  arm is, it is neither -- and the walks' aside carries it, as any walk."""
  body = body_for(*_floor())
  m = body.mission
  try:
    m.reach.carry_q["left"] = m.reach.q_cmd["left"].copy()
    m.reach.phase["left"] = "carried"
    assert not body.working and not m._hand_out()
    m.reach.phase["right"] = "hold"
    assert body.working and m._hand_out()
  finally:
    body.close()


def test_a_carried_arm_rides_a_restart():
  """What a restart keeps of the reach (`Reach.KEPT`) brings a carry back:
  its phase, its pose and the carry pose its way home ends at."""
  a, b = body_for(*_floor()), body_for(*_floor())
  try:
    ra, rb = a.mission.reach, b.mission.reach
    ra.carry_q["left"] = np.array([0.15, 0.25, 0.0, 2.0, -1.2, -1.1])
    ra.q_cmd["left"] = ra.carry_q["left"].copy()
    ra.phase["left"] = "carried"
    rb.restore_kept(ra.kept_state())
    assert rb.carried("left") and np.array_equal(rb.carry_q["left"], ra.carry_q["left"])
    assert rb.carry_q["right"] is None and np.array_equal(rb.home_to("left"), ra.carry_q["left"])
  finally:
    a.close()
    b.close()


def test_a_lift_that_never_ends_is_ended_and_carries_nothing(monkeypatch):
  """A hold that gives slowly kept a lift going and every walk waiting on it
  (the #668 class): it is ended where the hand is after `CARRY_LIFT_S`, read
  as not come away, and the hand let go."""
  body = body_for(*_floor())
  m = body.mission
  log = []
  _spy_opens(monkeypatch, body, log)
  at = lambda: float(body.data.time)  # noqa: E731
  grippers = m.reach.grippers
  m.reach = _Out(grippers, ("left",), log, at, lifts={"left": None})
  stopped = []

  def never(side, way, force, distance_m, speed=None, give=True, verb=False):
    log.append(("lift", side, at()))
    m.reach.phase[side] = "carry"
    return True
  m.reach.pull = never
  m.reach.step = lambda: None
  m.reach.stop = lambda side, why: stopped.append((side, why)) or m.reach.phase.update(
    {side: "hold"}) or m.reach.report.update({side: {"why": why}})
  grippers["left"].shut()
  monkeypatch.setattr(grippers["left"], "held", lambda: True)
  t0 = at()
  body.run(m.to_walk_routine())
  assert stopped == [("left", "out of time")]
  assert at() - t0 < hmis.CARRY_LIFT_S + 2.0 and m.legs == "walk"
  assert ("open", "left") in [e[:2] for e in log] and m.reach.carry_q["left"] is None


def test_only_a_hand_out_at_a_pose_of_its_own_or_carried_is_carried(monkeypatch):
  """A hand on its way anywhere, or walked joint by joint by `move`, holding
  something, has shown nothing comes away -- its lift needs a pose of its
  own (#682's review: a drawer's handle held through a `move` was dragged
  out) -- so a walk lets it go."""
  body = body_for(*_floor())
  m = body.mission
  log = []
  _spy_opens(monkeypatch, body, log)
  at = lambda: float(body.data.time)  # noqa: E731
  grippers = m.reach.grippers
  m.reach = _Out(grippers, ("left",), log, at)
  m.reach.phase["left"] = "joints"
  grippers["left"].shut()
  monkeypatch.setattr(grippers["left"], "held", lambda: True)
  assert m.carries() == {}
  body.run(m.to_walk_routine())
  said = [e[:2] for e in log]
  assert ("lift", "left") not in said and ("open", "left") in said, said


def test_a_withdraw_refused_keeps_its_carry_and_judges_the_way_to_its_rest():
  """A withdraw of a carried hand is judged home -- to its rest -- before
  anything ends the carry: refused for another robot's body, the hand stays
  carried, its carry as it was (#682's review: dropped first, the arm hung
  at its side carried by no one's account)."""
  from test_arms_keep_off import _Arms, _beside
  body = body_for(*_floor())
  m = body.mission
  try:
    _beside(body)
    m.reach = _Arms(m.reach.grippers, (), lambda: False, [], lambda: float(body.data.time))
    m.reach.phase["left"], m.reach.carry_q["left"] = "carried", np.zeros(6)
    assert body.run(body.release_routine("left", refuse=True)) == "kept off"
    assert m.reach.carried("left") and np.array_equal(m.reach.carry_q["left"], np.zeros(6))
    rest = m.reach.toward[0][1]
    assert m.reach.toward[0][0] == "left" and rest is not None and not np.allclose(rest, 0.0)
  finally:
    body.close()


def test_a_hand_home_stands_the_body_back_beside_one_carried():
  """The posture goes back to the stand once no hand is out at its work --
  and one carried at its side is none: kept the reach's bow, a withdraw of
  the other hand left the robot bowed, its plate at its side."""
  from test_hands import _bare_reach
  r = _bare_reach()
  r.phase["left"], r.carry_q["left"] = "carried", np.zeros(6)
  r.phase["right"] = "hold"
  r.way_home = lambda side, refuse=False: [np.zeros(6)]
  r._go_home("right")
  assert r.postures[-1][:2] == (0.0, 0.0)


def test_another_robots_returns_stand_within_a_held_grips_volume():
  """The self-filter is the room's returns' alone: another robot's are split
  out by the part's own reading (#316) and dropped, the front stop would be
  blind to it (#682's review)."""
  lidar = _bare_lidar()
  joints = {j: 0.0 for j in hm.WAIST}
  offset, turn = lidar.mount.pose(joints)
  height = 0.7
  n_el = len(lidar.elevations)
  ray = int(np.argmin(np.abs(lidar.elevations))) * BEARINGS
  peer = Cloud(np.zeros((0, 3)), np.array([[0.3, 0.0, 0.0]]), np.zeros(BEARINGS, dtype=bool),
               np.isin(np.arange(n_el * BEARINGS), [ray]), np.zeros(0, dtype=np.int64),
               np.array([ray]), np.zeros(n_el * BEARINGS, dtype=bool), np.array([0]))
  lidar.cloud = lambda data: peer
  q = peer.peers[0] @ turn.T + offset
  scan = lidar.band(None, np.eye(3), height, joints=joints,
                    drop=(np.array([[q[0], q[1], height + q[2]]]), hmis.HELD_DROP_M))
  assert len(scan.peer_ranges) == 1 and scan.peer_ranges[0] == pytest.approx(0.3, abs=0.01)


def test_the_lift_pulls_with_the_most_the_pull_verb_asks():
  """A typed copy, pinned: the lift's force is what the standing body holds
  at a hand, the `pull` verb's most (an import would be circular)."""
  assert hmis.CARRY_LIFT_N == st.PULL_MOST_N


def test_the_carrying_walker_trains_on_the_carry_pose_the_walk_takes():
  """`Pluggy-Humanoid-Walk-Carry` draws a share of its carries near the pose
  the served walk takes (#682's stage B, carry3): the two are one pose, read
  off training's source (its environment is its own: nothing imports it)."""
  import ast
  from pathlib import Path
  src = (Path(__file__).resolve().parents[1] / "training" / "humanoid_train" / "walk.py").read_text()
  close = next(ast.literal_eval(n.value) for n in ast.walk(ast.parse(src))
               if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "CARRY_CLOSE")
  assert tuple(close) == tuple(rc.CARRY_POSE)


# ---- 7: the walker shown its arms at rest while a hand is carried (#751) ---------


#: A carried left arm's six joints, folded as the carry pose folds it (the
#: elbow 2.0 rad: walk2's scripted arms went to 1.15).
_FOLDED = (0.15, 0.25, 0.0, 2.0, -1.2, -1.1)


def _shown(walker) -> dict:
  """Where each term of what the walking policy is shown sits in it."""
  pol = walker.policy
  nj = len(pol.joint_names)
  size = {"base_ang_vel": 3, "projected_gravity": 3, "joint_pos": nj, "joint_vel": nj,
          "actions": len(pol.action_joints), "command": 3}
  out, at = {}, 0
  for name in pol.observation_names:
    out[name] = slice(at, at + size[name])
    at += size[name]
  return out


def _arm_entries(walker) -> list[int]:
  """What the policy is shown of its arms: each arm joint's position and
  speed entries."""
  terms = _shown(walker)
  arms = [k for k, n in enumerate(walker.policy.joint_names) if n.split("_", 1)[1] in hm.ARM]
  return ([terms["joint_pos"].start + k for k in arms]
          + [terms["joint_vel"].start + k for k in arms])


def _fold_left(model, data) -> None:
  for j, v in zip(hm.ARM, _FOLDED):
    jid = model.joint(f"left_{j}").id
    data.qpos[model.jnt_qposadr[jid]] = v
    data.qvel[model.jnt_dofadr[jid]] = 0.8
  mujoco.mj_forward(model, data)


def test_the_walker_shown_its_arms_at_rest_sees_them_hang_still_and_draws_as_ever():
  """#751: walk2 trained with its arms scripted within half their range,
  hanging half the time; a carried hand folds the elbow to 1.6-2.0 rad, and
  seeing it walk2 stood still at the first carried step after a pick. Shown
  its arms at rest, every arm joint it sees (both arms, `hm.ARM`; no gripper,
  no waist) reads its policy's default -- where the served arms hang -- and
  still; everything else it is shown is as read, and the drivers are read as
  ever, the same draws (the standing policy shares them: a draw more or less
  would move every flight after it)."""
  from pluggybot.humanoid.walk import WalkDriver
  model, data = _floor()
  _fold_left(model, data)
  seen, rested = WalkDriver(model, data), WalkDriver(model, data)
  assert ({hm.JOINT_NAMES[i] for i in rested.arm_at}
          == {f"{s}_{j}" for s in hm.SIDES for j in hm.ARM})
  stand = hm.stand_qpos()
  assert np.array_equal(rested.arm_rest, [stand.get(hm.JOINT_NAMES[i], 0.0) for i in rested.arm_at]), \
    "the rest it is shown is not where the served arms hang"
  a, b = seen.observation(), rested.observation(arms_at_rest=True)
  arm = _arm_entries(rested)
  assert np.all(b[arm] == 0.0), "an arm it is shown off its default, or moving"
  elbow = _shown(seen)["joint_pos"].start + list(seen.policy.joint_names).index("left_elbow")
  assert a[elbow] == pytest.approx(2.0 - 0.3, abs=0.02), "the folded elbow, as its drivers read it"
  assert np.array_equal(np.delete(a, arm), np.delete(b, arm)), "it was shown something else apart"
  assert seen.joints.rng.bit_generator.state == rested.joints.rng.bit_generator.state, \
    "the drivers' noise drawn otherwise"


def test_a_walk_shows_its_walker_the_arms_at_rest_only_while_a_hand_is_carried():
  """The walker is shown its arms at rest while either hand is carried
  (`Reach.carried`), read off the reach as it stands at each of its policy's
  steps -- a carry whose grip has come to read empty is carried still (its
  arm stays folded at its pose until a reach or a let-go) -- and as read
  otherwise: an arm out, an arm left where its way home was held, a hand let
  go, and after a stand-up, whose reach is fresh."""
  body = body_for(*_floor())
  m = body.mission
  try:
    shown = []
    act = m.walker.policy.act

    def spy(obs):
      shown.append(np.array(obs, dtype=float))
      return act(obs)
    m.walker.policy.act = spy
    arm = _arm_entries(m.walker)
    _fold_left(m.model, m.data)

    def rested() -> bool:
      """One policy step of the walker's, and whether it was shown its arms at rest."""
      m.legs, m.walker.steps = "walk", 0
      m._before_step(hmis.STILL)
      return bool(np.all(shown[-1][arm] == 0.0))
    assert not rested(), "shown at rest with no hand carried"
    m.reach.carry_q["left"] = np.array(_FOLDED)
    m.reach.phase["left"] = "carried"
    assert rested()
    m.reach.grippers["left"].held = lambda: False
    assert rested(), "a carry whose grip reads empty, its arm folded still, shown as read"
    m.reach.drop_carry("left")
    m.reach.phase["left"] = "rest"
    assert not rested(), "a hand let go, shown at rest"
    m.reach.carry_q["right"] = m.reach.q_cmd["right"].copy()
    m.reach.phase["right"] = "carried"
    assert rested(), "the right hand's carry"
    m.reach.phase["left"], m.reach.phase["right"] = "held", "rest"
    m.reach.carry_q["right"] = None
    assert not rested(), "an arm left held, shown at rest"
    m.reach.carry_q["left"] = np.array(_FOLDED)
    m.reach.phase["left"] = "carried"
    m._stood_up()
    assert m.legs == "stand" and not rested(), "a stand-up's fresh reach, shown at rest"
  finally:
    body.close()


def test_a_restart_carries_on_showing_the_walker_its_arms_at_rest():
  """A restart mid-carry (#345: the reach's carry is kept state, `Reach.
  KEPT`) brings the rule back with it: the walker is told nothing a restart
  must keep apart, so the restored walk shows its arms at rest."""
  import json
  a, b = body_for(*_floor()), body_for(*_floor())
  try:
    ma, mb = a.mission, b.mission
    ma.reach.carry_q["left"] = np.array(_FOLDED)
    ma.reach.q_cmd["left"] = ma.reach.carry_q["left"].copy()
    ma.reach.phase["left"] = "carried"
    ma.legs = "walk"
    state, arrays = ma.kept_state()
    mb.restore_kept(json.loads(json.dumps(state)), arrays)
    shown = []
    act = mb.walker.policy.act
    mb.walker.policy.act = lambda obs: (shown.append(np.array(obs, dtype=float)), act(obs))[1]
    mb.walker.steps = 0
    mb._before_step(hmis.STILL)
    assert mb.legs == "walk" and np.all(shown[-1][_arm_entries(mb.walker)] == 0.0)
  finally:
    a.close()
    b.close()
