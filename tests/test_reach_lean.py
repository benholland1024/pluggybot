"""The reach planned where the robot stands (#627; SimNotes, "The reach
planned where the live trunk stands"):

  1. the plan's statue stands where the robot reads itself standing --
     its weight, its pelvis's place, turn and roll, the posture as it holds
     it -- at the stand, after a step with the other arm out of the way, and
     with its hand out at the table's plate (states it recorded,
     `humanoid_spike.py --lean --write`);
  2. a reach aimed at its grip arrives there on the live trunk, at the
     stand and with the other arm out of the way (flown, a bare floor);
  3. a hand at its pre-grasp looks where it stands before it goes in, and
     takes the posture its lean asks for, either way, waiting meanwhile;
  4. the seeded solve keeps the arm nearest the arm as it hangs;
  5. over the table cup's points the plan takes one posture -- no pockets.
"""

import json
import math
from pathlib import Path
from types import SimpleNamespace as NS

import mujoco
import numpy as np
import pytest

from pluggybot.humanoid import model as hm
from pluggybot.humanoid import reach as rc
from pluggybot.humanoid.arms import HandGoal, SelfModel, frame

from test_hands import _standing_reach, _stepped

FIXTURE = Path(__file__).parent / "fixtures" / "lean_states.json"
Q0 = np.array([hm.stand_qpos().get(n, 0.0) for n in hm.JOINT_NAMES])
#: The table's plate as its first reach asks it from its stance (#626).
PLATE = HandGoal(np.array([0.43, 0.09, 0.7734]), frame((0.5, 0.0, -0.866), (0.0, -1.0, 0.0)))


def _shoulders(sm) -> dict:
  return {s: sm.d.xpos[sm.m.body(f"{s}_shoulder_roll_link").id].copy() for s in hm.SIDES}


def test_the_plans_statue_stands_where_the_robot_reads_itself_standing():
  """At each state the robot recorded, its model of itself posed as its
  arm's loop poses it (its joints as read, its arms as commanded, its IMU)
  reads its lean, and the statue posed so stands each shoulder within 4 mm
  of where it truly stood -- where the statue the plan stood on before
  (balanced over its soles' middle, its arms hanging) stood them 1.5-6 cm
  off, the live trunk behind it (#627); asked the deepest bow, the policy
  holds less of it, and posed as asked, a shoulder stood 1.1 cm off."""
  states = json.loads(FIXTURE.read_text())["states"]
  assert set(states) == {"stand", "bow", "stepped", "plate"}
  for name, st in states.items():
    live, planner = SelfModel(), SelfModel()
    live.pose(np.array(st["q"]), np.array(st["gravity"]))
    for k, s in enumerate(hm.SIDES):
      live._set_arm(s, np.array(st["arms"][6 * k:6 * k + 6]))
    true = {s: np.array(v) for s, v in st["shoulders"].items()}
    lean = rc.Lean.read(live, st["posture"], planner)
    arms = {n: float(live.q[hm.JOINT_NAMES.index(n)])
            for s in hm.SIDES for n in (f"{s}_{j}" for j in hm.ARM)}
    rc.statue(planner, st["posture"], Q0, arms, lean)
    for s, p in _shoulders(planner).items():
      assert np.abs(p - true[s])[:2].max() < 0.004, (name, s, p - true[s])
    rc.stands(planner, st["posture"], Q0)
    off = max(float(np.abs(p - true[s])[:2].max()) for s, p in _shoulders(planner).items())
    assert off > 0.015, (name, off)


def test_a_reach_reads_its_lean_off_its_model_of_itself(monkeypatch):
  """The lean a reach plans on is its model of itself as its arm's loop
  last posed it (`Reach._lean`) -- never its joints read again -- here as it
  stood after a step with its other arm out of the way, 1.7 cm of its weight
  behind its soles' middle; and none before its first reading."""
  st = json.loads(FIXTURE.read_text())["states"]["stepped"]
  r = _standing_reach()
  assert r._lean() is None
  r.self_model.pose(np.array(st["q"]), np.array(st["gravity"]))
  for k, s in enumerate(hm.SIDES):
    r.q_cmd[s] = np.array(st["arms"][6 * k:6 * k + 6])
    r.self_model._set_arm(s, r.q_cmd[s])
  r.sensed, r.posture = np.array(st["q"]), np.array(st["posture"], dtype=float)
  lean = r._lean()
  assert lean is not None and -0.02 < lean.com < -0.014, lean
  seen = []
  monkeypatch.setattr(rc, "plan", lambda *a, **kw: seen.append(kw) or None)
  r.plans("left", PLATE)
  assert seen and seen[0]["lean"] == lean
  assert all(seen[0]["arms"][f"right_{j}"] == v for j, v in zip(hm.ARM, r.q_cmd["right"]))


def _floor():
  """The robot on a bare floor at the stand, its standing policy and its
  reach -- the robot's own drawing, nothing to touch."""
  from pluggybot.humanoid.policy import StandDriver
  from pluggybot.humanoid.reach import Reach
  model = mujoco.MjModel.from_xml_string(hm.body_xml())
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  mujoco.mj_forward(model, data)
  stand = StandDriver(model, data)
  reach = Reach(model, data, stand)
  reach.standing = True

  def fly(seconds, until=lambda: False):
    for _ in range(int(round(seconds / model.opt.timestep))):
      stand.step()
      reach.step()
      mujoco.mj_step(model, data)
      if until():
        return
  return model, data, reach, fly


def _true_stance(model, data):
  soles = [model.geom(f"{s}_sole").id for s in hm.SIDES]
  heads = [math.atan2(data.geom_xmat[g][3], data.geom_xmat[g][0]) for g in soles]
  yaw = math.atan2(sum(math.sin(h) for h in heads), sum(math.cos(h) for h in heads))
  mid = np.mean([data.geom_xpos[g] - data.geom_xmat[g].reshape(3, 3)[:, 2] * model.geom_size[g][2]
                 for g in soles], axis=0)
  c, s = math.cos(yaw), math.sin(yaw)
  return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]), mid


@pytest.mark.parametrize("aside", [False, True], ids=["arms hanging", "the other arm out"])
def test_a_reach_aimed_at_its_grip_arrives_there_on_the_live_trunk(aside):
  """The table's plate's grip, from the stand, the right arm hanging or held
  out of the way as the table's procedure holds it: the hand holds at its
  grip -- its own solve within a few mm of its command, and truly within 1
  cm of the grip in the frame its feet stand in. Planned on a statue
  balanced over the soles' middle, it stopped 3.2 and 5.1 cm short
  (`blocked`): the live trunk stood 2-6 cm behind that statue (#627)."""
  model, data, reach, fly = _floor()
  if aside:
    reach.set_joints("right", np.array([0.0, -0.9, 0.0, 1.4, 0.0, 0.0]))
    fly(2.5)
  else:
    fly(1.0)
  assert reach.reach("left", HandGoal(PLATE.pos.copy(), PLATE.rot.copy(), 1.0))
  fly(12.0, until=lambda: reach.done("left"))
  fly(0.3)
  assert reach.phase["left"] == "hold", reach.phase["left"]
  assert reach.miss["left"] < 0.005, reach.miss["left"]
  rot, origin = _true_stance(model, data)
  grip = rot.T @ (data.site_xpos[model.site("left_grip").id] - origin)
  assert np.linalg.norm(grip - PLATE.pos) < 0.01, grip - PLATE.pos


def _at_pre_grasp(monkeypatch, reaching, first, allowed):
  """A bare reach's left hand at its pre-grasp, in `first`, its body there
  and still; a posture puts its goal in reach where `reaching` says so, as
  the robot reads itself standing (`_reached` stubbed; the lean whatever it
  reads)."""
  r = _stepped(_standing_reach())
  r.stand = NS(settled=True)
  there = {"now": True}
  r._posture_there = lambda dt: there["now"]
  r._lean = lambda: rc.Lean(com=-0.02)

  def hold(height, pitch, waist=0.0):
    r.posture = np.array([height, pitch, waist], dtype=float)
    r.postures.append((height, pitch, waist))
  r.hold_posture = hold
  asked = []
  monkeypatch.setattr(rc, "stands", lambda sm, posture, q0: True)
  monkeypatch.setattr(rc, "out_of_reach", lambda *a, **kw: False)

  def reached(sm, side, posture, q0, goal, pre, slack, lean, arms, start):
    asked.append((tuple(posture), slack, lean))
    return (start, start) if reaching(tuple(posture), slack) else None
  monkeypatch.setattr(rc, "_reached", reached)
  q = r.self_model.solve("left", np.array([0.32, 0.2, 0.9]), None)[0]
  r.q_cmd["left"], r.q_to["left"] = q.copy(), q.copy()
  r.self_model._set_arm("left", q)
  r.goal["left"] = HandGoal(np.array([0.40, 0.2, 0.9]), None, 1.0)
  r.allowed["left"] = [tuple(float(v) for v in p_) for p_ in allowed]
  r.posture = np.array(first, dtype=float)
  r.phase["left"] = "move"
  return r, there, asked


SQUARE = [(0.0, 0.0, 0.0), (-0.08, 0.0, 0.0), (-0.16, 0.0, 0.0), (-0.16, 0.3, 0.0)]


def test_a_hand_at_its_pre_grasp_looks_where_it_stands_before_it_goes_in(monkeypatch):
  """Its arm out, the robot reads itself standing (`Lean`) and the least
  bent posture that puts its goal in reach is taken before the hand goes in
  -- deeper where the lean is further back than its plan read, LESS bent
  where it is further forward (an arm over the top leans the policy
  forward) -- its hand waiting at its pre-grasp while its body comes, never
  a posture it has left, at most `LOOKS` times; and where the posture it
  holds reaches, in it goes."""
  # the posture held reaches: in at once
  r, there, asked = _at_pre_grasp(monkeypatch, lambda p, slack: slack, SQUARE[0], SQUARE)
  r._update(0.01)
  assert r.phase["left"] == "approach" and not r.looking["left"] and not r.postures
  assert asked and asked[0][2] == rc.Lean(com=-0.02)
  # ...it does not, and only the third does: taken, the hand waiting
  r, there, asked = _at_pre_grasp(monkeypatch, lambda p, slack: p == SQUARE[2], SQUARE[0], SQUARE)
  r._update(0.01)
  assert r.phase["left"] == "approach" and r.looking["left"]
  assert r.postures == [SQUARE[2]] and r.left["left"] == [SQUARE[0]]
  waiting = r.cmd["left"].pos.copy()
  there["now"] = False
  for _ in range(30):
    r._update(0.01)
  assert r.phase["left"] == "approach" and np.allclose(r.cmd["left"].pos, waiting)
  # ...there, it looks again, and in it goes along its line
  there["now"] = True
  r._update(0.01)
  r._update(0.01)
  assert not r.looking["left"] and not np.allclose(r.cmd["left"].pos, waiting)
  assert r.looks["left"] == 2
  # a less bent posture reaches: taken too
  r, there, asked = _at_pre_grasp(monkeypatch, lambda p, slack: p == SQUARE[1], SQUARE[3], SQUARE)
  r._update(0.01)
  assert r.postures == [SQUARE[1]]
  # never one it has left, and no more than `LOOKS` looks: then in from where it is
  flip = {"to": SQUARE[1]}
  r, there, asked = _at_pre_grasp(monkeypatch, lambda p, slack: p == flip["to"], SQUARE[0], SQUARE)
  r._update(0.01)
  assert r.postures == [SQUARE[1]]
  flip["to"] = SQUARE[0]                      # left already: never taken again
  asked.clear()
  r._update(0.01)
  assert r.looking["left"] is False and r.postures == [SQUARE[1]]
  assert asked and all(p != SQUARE[0] for p, _, _ in asked)
  r, there, asked = _at_pre_grasp(monkeypatch, lambda p, slack: False, SQUARE[0], SQUARE)
  r.looks["left"] = rc.LOOKS
  r._update(0.01)
  assert not r.looking["left"] and not asked


def test_the_plan_balances_each_statue_with_the_arm_where_its_solve_put_it(monkeypatch):
  """Each posture the plan weighs is posed twice: with the reaching arm where
  it is, then balanced with it where the first solve put it and solved again
  -- an arm held out moves the trunk back 0.7-2.2 cm (#493, #627), and both
  times as the robot reads itself standing and with its other arm where it
  is."""
  posed = []
  real = rc.statue

  def spy(sm, posture, q0, arms=None, lean=None, fast=False):
    posed.append((tuple(posture), dict(arms or {}), lean))
    real(sm, posture, q0, arms, lean, fast)
  monkeypatch.setattr(rc, "statue", spy)
  lean, other = rc.Lean(com=-0.015), rc._arm("right", np.array([0.0, -0.9, 0.0, 1.4, 0.0, 0.0]))
  got = rc.plan("left", PLATE, SelfModel(), lean=lean, arms=other)
  assert got is not None
  last = [p for p in posed if p[0] == got[0]][-2:]
  assert len(last) == 2 and all(p[2] == lean for p in last)
  assert all(p[1][k] == v for p in last for k, v in other.items())
  assert not any(k.startswith("left_") for k in last[0][1])
  out = np.array([last[1][1][f"left_{j}"] for j in hm.ARM])
  hanging = Q0[[hm.JOINT_NAMES.index(f"left_{j}") for j in hm.ARM]]
  assert np.abs(out - hanging).max() > 0.3 and np.abs(out - got[2]).max() < 0.15, (out, got[2])


def test_the_fast_balance_is_the_least_squares_answer():
  """A plan weighs dozens of statues, each balanced at the lean it read:
  balanced from the kept statue by Newton's method (`statue(fast=True)`),
  each stands within 0.1 mm of the least squares' -- linear in the lean
  alone, it was 8 mm off in a deep bow with an arm held level."""
  sm = SelfModel()
  rng = np.random.default_rng(627)
  for posture in ((0.0, 0.0, 0.0), (-0.16, 0.3, math.radians(-30)), (-0.25, 0.95, 0.0)):
    lean = rc.Lean(com=float(rng.uniform(-0.035, 0.03)), roll=0.01, turn=0.02)
    arms = {**rc._arm("left", np.array([-1.4, 0.1, 0.0, 0.3, 0.0, 0.0])),
            **rc._arm("right", np.array([0.0, -0.9, 0.0, 1.4, 0.0, 0.0]))}
    rc.statue(sm, posture, Q0, arms, lean)
    exact = _shoulders(sm)
    rc.statue(sm, posture, Q0, arms, lean, fast=True)
    for s, p in _shoulders(sm).items():
      assert np.abs(p - exact[s]).max() < 1e-4, (posture, s, p - exact[s])


def test_the_seeded_solve_keeps_the_arm_nearest_where_it_hangs():
  """A left hand 0.40 m ahead at a counter's height, crouched: five seeds
  reach it with the arm in front, and the hanging arm's own seed -- tried
  first -- with the shoulder thrown back and the arm over the top. Kept in
  its seeds' order, that arm was planned; so held, the standing policy leant
  4.4 cm forward where the arm in front leant it 0.8 back (#627)."""
  sm = SelfModel()
  rc.stands(sm, (-0.16, 0.0, 0.0), Q0)
  goal = np.array([0.40, 0.18, 0.85])
  rot = frame((1.0, 0.0, -0.3), (0.0, 0.0, 1.0))
  sm.q[sm.arm["left"]] = rc.seed("left", rc.SEEDS[0])
  first = sm.solve("left", goal, rot, iters=80)
  assert first[1] < 0.01 and first[0][0] > 1.0, first    # the premise: over the top
  q, ep, er = rc.solve_seeded(sm, "left", goal, rot)
  assert ep < 0.01 and er < 0.1
  assert q[0] < 0.0 and abs(q[1]) < 0.5, q                 # in front, out a little


def test_the_plan_takes_one_posture_over_the_cups_points():
  """10 cm over the table's cup, its grip 45 deg down from the right, at
  points 5 mm apart: the plan took its 0.3 rad bow at most and 0.55 or 0.8
  rad -- no tag in view -- at scattered others, none of seven seeds
  converging there for the goal or the point its slack asks for, where each
  solved from the other's arm reaches (#627)."""
  sm = SelfModel()
  rot = frame((0.707, 0.0, -0.707), (0.0, 1.0, 0.0))
  for x, y in ((0.405, -0.435), (0.425, -0.435), (0.43, -0.42), (0.39, -0.44),
               (0.41, -0.43), (0.42, -0.44)):
    got = rc.plan("right", HandGoal(np.array([x, y, 0.88]), rot), sm)
    assert got is not None and got[0] == (-0.16, 0.3, 0.0), (x, y, got and got[0])
