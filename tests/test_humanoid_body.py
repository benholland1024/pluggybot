"""The humanoid behind `Body` (issue #507, stages 3 and 4): every member
answered as `Body` declares it, the one place a body is chosen choosing it,
and its hands' members flown -- the chest's knob found by its tag, reached
and gripped, the grip judged by the gripper's own reading and graded
against the truth by us. Each as cheap as it fails for the right reason;
SimNotes, "The humanoid behind Body", has the design."""

import inspect
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

from pluggybot.body import Body, body_for
from pluggybot.humanoid import model as hm

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def floor_body():
  from pluggybot.humanoid.body import HumanoidBody
  model = mujoco.MjModel.from_xml_string(hm.body_xml())
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  mujoco.mj_forward(model, data)
  body = body_for(model, data)
  assert isinstance(body, HumanoidBody)
  return body


def test_the_humanoid_answers_every_body_member_as_body_declares_it(floor_body):
  """No member left abstract, and every one takes what `Body` says it takes:
  the loop calls them by `Body`'s signatures."""
  assert not type(floor_body).__abstractmethods__
  for name, member in vars(Body).items():
    if name.startswith("_") or not getattr(member, "__isabstractmethod__", False):
      continue
    if isinstance(member, property):
      getattr(floor_body, name)
      continue
    want = list(inspect.signature(member).parameters)
    have = list(inspect.signature(getattr(type(floor_body), name)).parameters)
    assert want == have, (name, want, have)


def test_a_world_with_neither_body_is_refused_one():
  model = mujoco.MjModel.from_xml_string(
    '<mujoco><worldbody><geom type="plane" size="1 1 0.1"/></worldbody></mujoco>')
  with pytest.raises(ValueError, match="quadruped and the humanoid"):
    body_for(model, mujoco.MjData(model))


def test_the_quadruped_refuses_a_hand():
  """The hands' members are the humanoid's: the quadruped's planar arm says
  so rather than pretending (no world needed to ask it)."""
  from pluggybot.legs.body import QuadBody
  quad = object.__new__(QuadBody)
  with pytest.raises(NotImplementedError, match="planar|plane"):
    next(quad.reach_routine("left", (0.0, 0.0, 1.0)))
  with pytest.raises(NotImplementedError):
    quad.gripped("left")


def test_the_humanoid_finds_reaches_and_grips_the_chests_knob_through_body():
  """The `demo` scene's set-out, every act a `Body` member: the knob found
  by its tag in the colour imager (within a centimetre of where it truly
  is), reached with the fingers tilted down onto it, the gripper shut --
  HELD by the gripper's own reading, and the pads truly on the knob (our
  grading), still holding it after a hold. MuJoCo's, in the house's own
  light."""
  sys.path.insert(0, str(ROOT / "scripts"))
  import humanoid_spike as sp
  model, data, body = sp.demo_world()
  records = []
  try:
    for _ in sp.demo_scene(body, model, data, records):
      pass
  finally:
    body.close()
  rec = records[0]
  assert rec["seen_off_m"] < 0.01, rec
  assert rec["reached"] and rec["held"] and rec["pads"] and rec["still_pads"], rec
  assert not rec["fell"]


def _floor():
  model = mujoco.MjModel.from_xml_string(hm.body_xml())
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  mujoco.mj_forward(model, data)
  return model, data


class _Sent:
  """A reach that only says where its hand is: sent, the hand goes to
  `end` and stays there. `done` is `Reach.done`'s."""

  def __init__(self, end: str):
    self.end = end
    self.phase = {s: "rest" for s in hm.SIDES}

  def reach(self, side, goal, approach=None) -> bool:
    self.phase[side] = self.end
    return True

  def done(self, side) -> bool:
    return self.phase[side] in ("hold", "rest", "blocked")

  def step(self) -> None:
    pass

  def held_pitch(self) -> float:
    return 0.0


class _Rising:
  """A get-up that stands the body after `n` steps, whatever it does."""

  def __init__(self, n: int):
    self.n, self.steps = n, 0

  def begin(self) -> None:
    pass

  def step(self) -> None:
    self.steps += 1

  @property
  def stood(self) -> bool:
    return self.steps >= self.n


def test_a_reach_says_true_only_of_a_hand_holding_there():
  """`Reach.done` is not arrival (#520): it is also a hand stopped on its
  way (`blocked`: out of reach of the posture truly held, or pushing on
  something) -- the floor's reach stopped 6 cm short of it and said True."""
  for end, said in (("hold", True), ("blocked", False)):
    body = body_for(*_floor())
    body.mission.reach = _Sent(end)
    assert body.run(body.reach_routine("left", (0.4, 0.2, 0.9))) is said, end


def test_a_reach_a_fall_let_go_of_never_arrived():
  """A fall's get-up lets go of the reach, and its stand-up builds a fresh
  one whose hands hang -- which `Reach.done` counts: the reach said True
  (#520). It ends at the fall, False; the next routine waits out the
  get-up."""
  body = body_for(*_floor())
  m = body.mission
  m.reach, m.getup = _Sent("move"), _Rising(50)
  falls = iter([True])
  m.fall.falling = lambda data, pitch=0.0: next(falls, False)
  assert body.run(body.reach_routine("left", (0.4, 0.2, 0.9))) is False
  assert m.falls == 1 and m.legs == "getup"


def test_through_body_the_hand_reaches_the_floor_and_the_bow_is_no_fall():
  """The humanoid's deepest reach, flown through `Body` with its get-up
  loaded, as served: the floor by its toes, fingers down, from the deepest
  bow its standing policy holds (1.1 rad). It holds there, the grip truly
  within 1.5 cm of the goal (our grading), and its own watch takes no fall.
  ⚠ Its watch read off upright took the bow for a fall and the get-up took
  the joints; its approach begun as the bow stopped turning was stopped
  6 cm short by the force its currents read (#520)."""
  import math
  from pluggybot.humanoid.arms import DOWN
  from pluggybot.humanoid.body import _rz
  model, data = _floor()
  body = body_for(model, data)
  assert body.rights_itself
  goal = np.array([0.17, 0.2, 0.09])                        # the `reach` scene's floor
  x, y, yaw = body._true_stance()
  truly = _rz(yaw) @ goal + [x, y, 0.0]
  got = body.run(body.reach_routine("left", body.to_world(goal), _rz(body.stance[2]) @ DOWN,
                                    patience=10.0))
  assert got is True and body.falls == 0
  assert body.reach.posture[1] == pytest.approx(1.1)
  assert np.linalg.norm(data.site_xpos[model.site("left_grip").id] - truly) < 0.015
  assert math.degrees(math.acos(data.xmat[body.root][8])) > 55.0     # bowed past the old 35


# ---- stage 4: walking ----------------------------------------------------------------


def test_the_walking_policy_walks_and_turns_in_our_physics():
  """The walking policy flown in the served sim's physics, off the robot's
  own senses: it tracks a walk, and from standing it turns -- its heading
  over 4 s of a turn asked 0.5 rad/s. ⚠ Trained with the hips' yaw held as
  tightly as the G1's (0.15 rad), a policy asked 0.5 rad/s turned 0.08
  (`training/humanoid_train/walk.py`, `STD_WALKING`). Turning is still its
  weakest command (SimNotes, "The humanoid walks")."""
  import math
  from pluggybot.humanoid.policy import StandDriver
  from pluggybot.humanoid.reach import Reach
  from pluggybot.humanoid.walk import WalkDriver

  def robot():
    model, data = _floor()
    stand = StandDriver(model, data)
    reach = Reach(model, data, stand)
    return model, data, reach, WalkDriver(model, data, trunk=stand.trunk, joints=stand.joints)

  def heading(model, data):
    rot = data.xmat[model.body(hm.ROBOT_ROOT).id].reshape(3, 3)
    return math.atan2(rot[1, 0], rot[0, 0])

  def fly(model, data, reach, walk, seconds):
    for _ in range(int(seconds / model.opt.timestep)):
      walk.step()
      reach.step()
      mujoco.mj_step(model, data)

  model, data, reach, walk = robot()
  walk.ask(0.8, 0.0, 0.0)
  fly(model, data, reach, walk, 2.0)
  x0, t0 = float(data.xpos[model.body(hm.ROBOT_ROOT).id][0]), data.time
  fly(model, data, reach, walk, 2.0)
  v = (float(data.xpos[model.body(hm.ROBOT_ROOT).id][0]) - x0) / (data.time - t0)
  assert v == pytest.approx(0.8, rel=0.15)

  model, data, reach, walk = robot()
  walk.ask(0.0, 0.0, 0.5)
  fly(model, data, reach, walk, 1.0)
  h0, turned = heading(model, data), 0.0
  for _ in range(4):
    fly(model, data, reach, walk, 1.0)
    h1 = heading(model, data)
    turned += math.remainder(h1 - h0, 2 * math.pi)
    h0 = h1
  assert turned / 4.0 > 0.3, turned / 4.0
  assert data.xpos[model.body(hm.ROBOT_ROOT).id][2] > 0.6


def test_the_walker_that_sees_the_ground_steps_onto_a_doorstep_and_off_it():
  """Rough ground in our physics (`humanoid_spike.course_flight`): a 5 cm
  doorstep, twice, walked straight ahead at 0.6 m/s from a start drawn
  askew, by the walker that sees the ground, on its own map. ⚠ The flat
  walker -- the served one -- fell on it eight times in eight (SimNotes,
  "The humanoid walks")."""
  sys.path.insert(0, str(ROOT / "scripts"))
  import humanoid_spike as sp
  rec = sp.course_flight("doorstep", 0, str(sp.SEEING_NPZ))
  assert not rec["fell"] and rec["made_m"] > 5.0, rec


def test_the_humanoid_walks_to_a_point_through_the_navigator():
  """`Body.go_to_routine` over the Navigator (`humanoid.mission`): it plans,
  hands its legs to the walking policy and walks there; its belief -- its
  legs' odometry and IMU, nothing on a bare floor to match -- within 8 % of
  the way of where it truly is."""
  import math
  model, data = _floor()
  body = body_for(model, data, grid_bounds=(-3, -3, 7, 7))
  assert body.run(body.go_to_routine(2.5, 0.5, timeout=30.0))
  bx, by, _ = body.pose
  tx, ty, _ = body.true_pose()
  assert math.hypot(tx - 2.5, ty - 0.5) < 0.25
  assert math.hypot(bx - tx, by - ty) < 0.08 * math.hypot(2.5, 0.5)
  assert body.mission.legs == "walk"


def test_the_legs_meet_each_other():
  """The legs' parts collide with each other (`model.LEG_GEOMS`'s pairs):
  the robot's own geoms touch nothing of the robot's, so without the pairs
  a gait could pass one leg through the other. Crossed, they meet."""
  model, data = _floor()
  for name, value in (("left_hip_roll", -0.5), ("right_hip_roll", 0.5)):
    data.qpos[model.jnt_qposadr[model.joint(name).id]] = value
  mujoco.mj_forward(model, data)
  left = {model.geom(f"left_{p}").id for p in hm.LEG_GEOMS}
  right = {model.geom(f"right_{p}").id for p in hm.LEG_GEOMS}
  pairs = {(int(c.geom1), int(c.geom2)) for c in data.contact[:data.ncon]}
  assert any((a in left and b in right) or (a in right and b in left) for a, b in pairs)
