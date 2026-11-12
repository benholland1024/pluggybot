"""The humanoid behind `Body` (issue #507, stages 3 and 4): every member
answered as `Body` declares it, the one place a body is chosen choosing it,
and its hands' members flown -- the chest's knob found by its tag, reached
and gripped, the grip judged by the gripper's own reading and graded
against the truth by us. Each as cheap as it fails for the right reason;
SimNotes, "The humanoid behind Body", has the design."""

import inspect
import math
import sys
from pathlib import Path
from types import SimpleNamespace as NS

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


def test_a_world_with_no_humanoid_is_refused_a_body():
  """The humanoid is the one body `body_for` builds (the quadruped left in
  #559): a world with none under this robot's names is refused one."""
  model = mujoco.MjModel.from_xml_string(
    '<mujoco><worldbody><geom type="plane" size="1 1 0.1"/></worldbody></mujoco>')
  with pytest.raises(ValueError, match="no body under"):
    body_for(model, mujoco.MjData(model))


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
  # ...and let go, its arm home by its drivers (#539: the walk home stepped
  # and arrived, never only begun)
  assert rec["released"] == "home", rec
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
    self.grippers = {}

  def reach(self, side, goal, approach=None, postures=None, flip=True) -> bool:
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


def test_a_ramp_a_fall_let_go_of_ends_there():
  """A fall lets go of the arm: ramped on through the get-up, `move` wrote
  its target into the stand-up's fresh reach, driving an arm whose phase
  says it hangs -- which a walk's hand-over reads as home (#520)."""
  body = body_for(*_floor())
  m = body.mission
  m.getup = _Rising(50)
  falls = iter([True])
  m.fall.falling = lambda data, pitch=0.0: next(falls, False)
  body.run(body.ramp_routine(body.actuator("left_elbow"), 1.5, 1.0))
  assert m.legs == "getup", "the ramp ran on past the fall"
  body.run(m._getup_routine())
  assert np.allclose(m.reach.q_to["left"], m.reach.rest["left"])


class _Swinging(_Rising):
  """A get-up that swings one leg's hip forward as it rises, `n` steps, the
  other's knee folded up out of the way, and stands that way: the foot it
  stands on moves under its pelvis, as a real get-up's feet do."""

  def __init__(self, n: int, data, hip: int, knee: int):
    super().__init__(n)
    self.d, self.hip, self.knee = data, hip, knee

  def step(self) -> None:
    super().step()
    k = min(1.0, self.steps / self.n)
    self.d.qpos[self.hip], self.d.qpos[self.knee] = -0.5 * k, 1.2 * k


def test_a_get_up_walks_no_belief():
  """#572: the odometry stepped while the get-up had the legs, following a
  tumbling body's feet -- a fall walked the belief 1.7-2.1 m in flights and
  Luca's 3.7 m live, past the wide search's 2 m. It is held through the
  get-up, and stood, it counts from the foot it stands on then: the foot it
  stood on before the fall counted the whole get-up's move at once."""
  model, data = _floor()
  body = body_for(model, data)
  m = body.mission
  body.run(body.settle_routine(model.opt.timestep * 3 * m._odo_every))
  planted, other = hm.SIDES[m.odo.foot], hm.SIDES[1 - m.odo.foot]
  adr = lambda name: model.jnt_qposadr[model.joint(name).id]  # noqa: E731
  m.getup = _Swinging(300, data, adr(f"{planted}_hip_pitch"), adr(f"{other}_knee"))
  falls = iter([True])
  m.fall.falling = lambda data, pitch=0.0: next(falls, False)
  body.run(body.settle_routine(model.opt.timestep * m._fall_every))  # the watch takes a fall
  assert m.legs == "getup"
  before = (m.odo.x, m.odo.y)
  body.run(m._getup_routine())                              # ...and it stands
  assert m.legs == "stand"
  assert (m.odo.x, m.odo.y) == before, "the get-up walked the belief"
  body.run(body.settle_routine(model.opt.timestep * m._odo_every))
  assert math.hypot(m.odo.x - before[0], m.odo.y - before[1]) < 0.05, \
    "stood, it counted the get-up's move off the foot it stood on before"


def _opened(grippers: dict) -> dict:
  """The body's grippers, each opened -- a release first opens one that is
  not (#669) -- for a test of what comes after."""
  for g in grippers.values():
    g.open()
  return grippers


class _Homing:
  """A reach whose arm its drivers read home `n` steps after it is sent,
  its phase saying it hangs from the first: as the walk home's did as it
  began (#539) -- its grippers the body's own, and each hand sent home said
  in `log` beside the sim time `at` gives."""

  def __init__(self, n: int, grippers: dict, log=None, at=None, on_step=None):
    self.n, self.steps, self.sent = n, 0, False
    self.phase = {s: "rest" for s in hm.SIDES}
    self.grippers, self.log, self.at, self.on_step = grippers, log, at, on_step

  def retreat(self, side, approach=None) -> None:
    self.sent, self.steps = True, 0
    if self.log is not None:
      self.log.append(("retreat", side, self.at()))

  def home(self, side) -> bool:
    return self.sent and self.steps >= self.n

  def step(self) -> None:
    self.steps += 1
    if self.on_step is not None:
      self.on_step(self.steps)

  def hold_waist(self) -> None:
    pass

  def held_pitch(self) -> float:
    return 0.0


class _Out(_Homing):
  """...one whose `out` hands are out (holding), each sent home at once:
  for a walk's hand-over (`to_walk_routine`)."""

  def __init__(self, grippers: dict, out, log, at):
    super().__init__(0, grippers, log, at)
    self.phase = {s: "hold" if s in out else "rest" for s in hm.SIDES}

  def retreat(self, side, approach=None) -> None:
    super().retreat(side, approach)
    self.phase[side] = "rest"

  def hold_posture(self, height, pitch) -> None:
    pass

  def waist_home(self) -> bool:
    return True


def test_a_release_waits_until_its_drivers_read_the_arm_home(monkeypatch):
  """#539: it returned the moment the walk home began -- the phase already
  the rest's -- and the procedure's next verb, or the grader's hold, began
  with the arm seconds from home. It says "home" once the arm is; "held"
  where it is not by `RELEASE_S`; "fell" at a fall."""
  from pluggybot.humanoid import body as hb
  body = body_for(*_floor())
  m = body.mission
  m.reach = _Homing(300, _opened(m.reach.grippers))
  assert body.run(body.release_routine("left")) == "home" and m.reach.steps >= 300
  monkeypatch.setattr(hb, "RELEASE_S", 0.1)
  m.reach = _Homing(10 ** 6, _opened(m.reach.grippers))
  assert body.run(body.release_routine("left")) == "held"
  m.reach, m.getup = _Homing(10 ** 6, _opened(m.reach.grippers)), _Rising(10 ** 6)
  falls = iter([True])
  m.fall.falling = lambda data, pitch=0.0: next(falls, False)
  t0 = float(body.data.time)
  assert body.run(body.release_routine("left")) == "fell"
  assert float(body.data.time) - t0 < 0.1, "a fall ends it at once"


def test_a_release_begun_in_a_get_up_waits_it_out_then_comes_home(monkeypatch):
  """Every hand routine waits out a get-up under way first: begun in one, a
  release spent its `RELEASE_S` on the arm the fall had let go of, and said
  "held" of an arm the stand-up's fresh reach hung at home."""
  from pluggybot.humanoid import body as hb
  monkeypatch.setattr(hb, "RELEASE_S", 0.1)
  body = body_for(*_floor())
  m = body.mission
  m.reach, m.getup = _Homing(10 ** 6, _opened(m.reach.grippers)), _Rising(200)
  m._fell()
  assert body.run(body.release_routine("left")) == "home"
  assert m.legs == "stand"


def test_a_fall_while_the_posture_comes_back_is_a_fall():
  """A release waits for its posture to come back to the stand: a fall then
  said "held" -- the stand-up's fresh reach read nothing yet -- not "fell";
  and it ends there, never waiting on a posture the get-up has taken."""
  body = body_for(*_floor())
  m = body.mission
  m.reach, m.getup = _Homing(0, _opened(m.reach.grippers)), _Rising(200)
  m.stand.ask(-0.15, 0.0)
  falls = iter([True])
  m.fall.falling = lambda data, pitch=0.0: next(falls, False)
  t0 = float(body.data.time)
  assert body.run(body.release_routine("left")) == "fell"
  assert float(body.data.time) - t0 < 0.1, "a fall ends it at once"


def test_a_release_opens_its_gripper_before_its_hand_comes_out(monkeypatch):
  """#669: at 14:25 on 11-08 Rowan's jaws were shut on a door's lever and
  read nothing held, so its procedure never opened them, and the way home
  levered its trunk into the counter (9 of 9 flown). Its gripper is opened
  first, `GRIP_S` before the hand is sent home, whatever it says it holds --
  told to shut, or together as it boots; one already open is left as it is,
  its hand sent home at once as before; and a fall as the jaws open ends it
  there, the hand never sent. The verb's text says so."""
  from pluggybot.humanoid import body as hb
  from pluggybot.humanoid.arms import Gripper
  from pluggybot.procedure.steps import HANDS_VERBS
  body = body_for(*_floor())
  m = body.mission
  log = []
  at = lambda: float(body.data.time)  # noqa: E731
  opened = Gripper.open

  def spied(g, *args, **kwargs):
    log.append(("open", next(s for s, h in m.reach.grippers.items() if h is g), at()))
    opened(g, *args, **kwargs)
  monkeypatch.setattr(Gripper, "open", spied)
  grippers = m.reach.grippers
  dt = m.model.opt.timestep
  m.reach = _Homing(10, grippers, log, at)
  grippers["left"].shut()
  assert not grippers["left"].held(), "the premise: it says it holds nothing"
  assert body.run(body.release_routine("left")) == "home"
  assert [e[:2] for e in log] == [("open", "left"), ("retreat", "left")], log
  assert log[1][2] - log[0][2] == pytest.approx(hb.GRIP_S, abs=1.5 * dt)
  assert grippers["left"].opened()
  # ...together as it boots, never told either way: opened too
  log.clear()
  m.reach = _Homing(10, grippers, log, at)
  assert not grippers["right"].shut_asked and not grippers["right"].opened()
  assert body.run(body.release_routine("right")) == "home"
  assert [e[:2] for e in log] == [("open", "right"), ("retreat", "right")], log
  # ...and one already open is left as it is
  log.clear()
  m.reach = _Homing(10, grippers, log, at)
  t0 = at()
  assert body.run(body.release_routine("right")) == "home"
  assert log == [("retreat", "right", t0)], "an open gripper is left as it is"
  # ...and a fall as the jaws open ends it the step it comes, its last too
  # (#669's review: checked before each step, the last one's fall sent the
  # hand home on a body the get-up had)
  n = int(round(hb.GRIP_S / dt))
  for at_step in (1, n):
    log.clear()
    m.getup = _Rising(10 ** 6)
    m.reach = _Homing(10, grippers, log, at,
                      on_step=lambda k, at_step=at_step: k == at_step and m._fell())
    grippers["left"].shut()
    t0 = at()
    assert body.run(body.release_routine("left")) == "fell"
    assert [e[0] for e in log] == ["open"], (at_step, log)
    assert at() - t0 == pytest.approx(at_step * dt, abs=1.5 * dt), at_step
    m.legs = "stand"
  doc = HANDS_VERBS["withdraw"].doc
  assert doc.startswith("open a hand's gripper, letting go of whatever it holds, then"), doc


def test_a_walk_lets_go_of_a_hand_that_is_out_before_it_comes_home(monkeypatch):
  """#669's review: a walk's hand-over brought an arm that was out home with
  its gripper as it was -- `step` refuses only a grip that reads held, and a
  procedure that ended with a hand out walks after -- so jaws shut on a
  lever that read nothing held levered the trunk over as a withdraw's had.
  An arm that is out lets go first, as a withdraw's does; one at rest is
  left as it is, its gripper too, and its walk begins as before."""
  from pluggybot.humanoid import body as hb
  from pluggybot.humanoid.arms import Gripper
  body = body_for(*_floor())
  m = body.mission
  log = []
  at = lambda: float(body.data.time)  # noqa: E731
  opened = Gripper.open

  def spied(g, *args, **kwargs):
    log.append(("open", next(s for s, h in m.reach.grippers.items() if h is g), at()))
    opened(g, *args, **kwargs)
  monkeypatch.setattr(Gripper, "open", spied)
  grippers = m.reach.grippers
  dt = m.model.opt.timestep
  m.reach = _Out(grippers, ("left",), log, at)
  grippers["left"].shut()
  assert not grippers["right"].opened(), "the premise: the arm at rest boots shut"
  t0 = at()
  body.run(m.to_walk_routine())
  assert m.legs == "walk"
  assert [e[:2] for e in log] == [("open", "left"), ("retreat", "left")], log
  assert log[0][2] == t0 and log[1][2] - t0 == pytest.approx(hb.GRIP_S, abs=1.5 * dt)
  assert grippers["left"].opened() and not grippers["right"].opened()


def _capped(body, routine, seconds, each=None):
  """A routine driven as `Body.run` drives it, `seconds` of sim at most:
  (whether it returned, what it returned). `each` is called before every
  step with the command."""
  from pluggybot import tick
  step = tick.Step(routine, "capped")
  cmd = step.tick()
  for _ in range(int(round(seconds / body.mission.model.opt.timestep))):
    if cmd is None:
      break
    if each is not None:
      each(cmd)
    body.mission.stepper.step(cmd)
    cmd = step.tick()
  if cmd is not None:
    step.routine.close()
  return step.done, step.result


def test_a_release_asked_aside_as_its_arm_comes_home_takes_the_legs_back():
  """#668: Luca's `withdraw` waited 2,191 s. Its arm read home a while after
  its command came to rest (the counter held it), so it was resting, and
  the other robot's walk asked it aside: the walker took its legs, nothing
  gave them back, and the release waited on a standing posture that moves
  only while the standing policy steps. Once the arm is home it takes them
  back, and its posture is waited on `STAND_BACK_S` at most. ⚠ Unchanged,
  this robot's release never returned: its step aside ended at its spot,
  and the walk's trim (`Reach._trim`) had moved the standing goal off the
  posture the walker left standing still."""
  from pluggybot.humanoid import body as hb
  from pluggybot.humanoid.mission import SETTLE_S
  from pluggybot.humanoid.way import REST_S
  from test_humanoid_served import _open_floor
  body = body_for(*_floor())
  m = body.mission
  try:
    x, y, _ = body.pose
    _open_floor(m, x, y)
    m.last_motion_t = float(body.data.time) - REST_S
    # ...its arm read home 6 s after its command rests: a stand-in for an
    # arm the counter held until the step aside walked it free
    free_t, home = float(body.data.time) + 6.0, m.reach.home
    m.reach.home = lambda side: float(body.data.time) >= free_t and home(side)
    seen = {"asked": False, "walked": False}

    def each(cmd):
      if not seen["asked"]:
        seen["asked"] = body.make_way([(x - 1.5, y), (x + 1.5, y)], "r2_pluggybot")
      seen["walked"] = seen["walked"] or m.legs == "walk"
    done, said = _capped(body, body.release_routine("right"), 30.0, each)
    assert seen["asked"] and seen["walked"], "the premise: a step aside took its legs"
    assert done, "the release never returned"
    assert said == "home" and m.legs == "stand" and m._aside is None, (said, m.legs)
    assert float(body.data.time) <= free_t + SETTLE_S + hb.STAND_BACK_S
  finally:
    body.close()


def test_a_release_waits_on_its_posture_a_bounded_time(monkeypatch):
  """#668: a release waited on its standing posture with no bound, a loop
  with no budget and no answer. A posture that never comes back is waited
  on `STAND_BACK_S`, and the arm's own word is the answer."""
  from pluggybot.humanoid import body as hb
  from pluggybot.humanoid.policy import StandDriver
  monkeypatch.setattr(hb, "STAND_BACK_S", 0.2)
  monkeypatch.setattr(StandDriver, "settled", property(lambda self: False))
  body = body_for(*_floor())
  m = body.mission
  m.reach = _Homing(10, _opened(m.reach.grippers))
  t0 = float(body.data.time)
  assert _capped(body, body.release_routine("left"), 2.0) == (True, "home")
  assert float(body.data.time) - t0 == pytest.approx(0.2 + 10 * m.model.opt.timestep, abs=0.005)


def test_a_fall_as_a_release_takes_its_legs_back_ends_it_at_once():
  """#668's review: the walker holding the legs as the arm comes home, a
  fall while the release hands them back to the standing policy ends it
  "fell" at once, as any fall in it does: the get-up is the next routine's
  to wait out (`to_stand_routine(getup=False)`)."""
  body = body_for(*_floor())
  m = body.mission
  m.reach, m.getup = _Homing(10, _opened(m.reach.grippers)), _Rising(10 ** 6)
  home = []

  def falling(data, pitch=0.0):
    # ...the watch's second look with the arm home: the release is past its
    # wait for the arm, handing the legs back
    home.append(m.reach.home("left"))
    return home.count(True) == 2

  def walker(cmd):
    m.legs = "walk" if m.legs == "stand" and not m.reach.home("left") else m.legs
  m.fall.falling = falling
  t0 = float(body.data.time)
  assert _capped(body, body.release_routine("left"), 2.0, walker) == (True, "fell")
  assert home.count(True) == 2, "the premise: it fell as the legs were handed back"
  assert float(body.data.time) - t0 < 0.1, "a fall ends it at once"
  # ...and begun with the get-up under way, the hand-back ends at once too
  assert m.legs == "getup" and _capped(body, m.to_stand_routine(getup=False), 0.1) == (True, None)


def test_an_aside_asked_as_the_legs_come_back_is_ended_and_the_settle_stays_still():
  """#668's review: `to_stand_routine` ended a step aside under way as it
  began, but not one asked during its settle, the walker held still
  `SETTLE_S` before the standing policy takes the legs: the aside walked
  0.6 s of the settle, the legs went over as it walked, and the aside took
  them back. A release hands them back after its robot may have rested on
  the walker's legs: the robot the other's walk asks aside."""
  from pluggybot.humanoid.mission import STILL
  from pluggybot.humanoid.way import REST_S
  from test_humanoid_served import _open_floor
  body = body_for(*_floor())
  m = body.mission
  try:
    x, y, _ = body.pose
    _open_floor(m, x, y)
    body.run(m.to_walk_routine())
    body.run(body.hold_routine(1.0))
    m.last_motion_t = float(body.data.time) - REST_S
    seen = {"steps": 0, "asked": False, "walked": 0}

    def each(cmd):
      seen["steps"] += 1
      if seen["steps"] == 100:                 # 0.2 s into the settle
        seen["asked"] = body.make_way([(x - 1.5, y), (x + 1.5, y)], "r2_pluggybot")

    def walked():
      seen["walked"] += m.legs == "walk" and tuple(m.walker.twist) != STILL
    body.step_hooks.append(walked)
    assert _capped(body, m.to_stand_routine(), 2.0, each) == (True, None)
    assert seen["asked"], "the premise: asked aside as it settled"
    assert seen["walked"] <= 1, f"the walker walked {seen['walked']} steps of the settle"
    assert m.legs == "stand" and m._aside is None
    assert body.last_aside["why"] == "its own move"
  finally:
    body.close()


def test_through_body_the_hand_reaches_the_floor_and_the_bow_is_no_fall():
  """The humanoid's deepest reach, flown through `Body` with its get-up
  loaded, as served: the floor by its toes, fingers down, from a deep bow
  turned at the waist (#560's stage (c3): without the waist, the deepest
  its standing policy holds, 1.1 rad). It holds there, the grip truly
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
  assert body.reach.posture[1] >= 0.95 and body.reach.posture[2] != 0.0
  assert np.linalg.norm(data.site_xpos[model.site("left_grip").id] - truly) < 0.015
  assert math.degrees(math.acos(data.xmat[body.root][8])) > 50.0     # bowed past the old 35


# ---- stage 4: walking ----------------------------------------------------------------


def test_the_walking_policy_walks_and_turns_in_our_physics():
  """The walking policy flown in the served sim's physics, off the robot's
  own senses: it tracks a walk, and from standing it turns -- its heading
  over 4 s of a turn asked 0.5 rad/s. ⚠ Trained with the hips' yaw held as
  tightly as the G1's (0.15 rad), a policy asked 0.5 rad/s turned 0.08
  (`training/humanoid_train/walk.py`, `STD_WALKING`). Turning is still its
  weakest command (SimNotes, "The humanoid walks"): asked 0.5 it turns in
  one of two gaits, 0.22-0.27 or 0.37-0.43 rad/s, by the draw of its
  senses' noise -- on #581's head and the one before it alike (8 draws a
  head and a way; asked 0.4 or 0.6, the heads turned alike)."""
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
  assert turned / 4.0 > 0.2, turned / 4.0
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


# ---- the waist (#560, stage (c3)) ----------------------------------------------------


def test_the_head_sensors_are_laid_through_the_waist_the_odometry_read():
  """The LIDAR's band and the depth camera's frame are each laid through the
  waist's angle as its driver read it -- the reading the odometry posed its
  model of itself with (`HumanoidMission.waist_read`), never the
  simulator's, never a zero."""
  body = body_for(*_floor())
  m = body.mission
  try:
    sm = m.odo.sm
    sm.q[sm.waist] = 0.3
    handed = []
    real_band, real_frame = m.lidar.band, m.depth.frame
    m.lidar.band = lambda data, level, height, **kw: handed.append(kw["joints"]) or real_band(
      data, level, height, **kw)
    m.depth.frame = lambda data, joints=None: handed.append(joints) or real_frame(data, joints)
    m._scan()
    m._frame_t = -1e9
    m.latest_depth()
    assert handed == [{"waist_yaw": 0.3}, {"waist_yaw": 0.3}]
  finally:
    body.close()


def test_a_walk_takes_the_legs_only_once_the_waist_reads_home():
  """The walking policy knows no waist: turned for a reach, the waist is
  walked home first (`Reach.waist_home`: commanded there and read there),
  and only then do the legs go to the walker -- as the arms come home first."""
  body = body_for(*_floor())
  m = body.mission
  try:
    m.reach.hold_posture(0.0, 0.0, 0.6)
    body.run(body.hold_routine(1.5))
    w = m.reach.self_model.waist
    assert m.reach.sensed[w][0] > 0.55 and not m.reach.waist_home()
    gen = m.to_walk_routine()
    for _ in range(int(6.0 / m.model.opt.timestep)):
      try:
        command = next(gen)
      except StopIteration:
        break
      m.stepper.step(command)
    assert m.legs == "walk" and m.reach.waist_home()
    assert abs(m.reach.sensed[w][0]) <= 0.05
  finally:
    body.close()


def test_the_get_up_holds_the_waist_home_and_reads_nothing():
  """The get-up policy drives the legs and the arms, never the waist: while
  it has them the waist is walked home at `WAIST_RATE` and held there
  (`Reach.hold_waist`), a command each `EVERY_S` -- nothing read, no model
  posed, while it tumbles."""
  from pluggybot.humanoid import reach as rc
  from test_hands import _bare_reach
  r = _bare_reach()
  r.m = NS(opt=NS(timestep=0.002))
  r.every, r.steps = 5, 0
  r.posture[2], r.waist_cmd = 0.5, np.array([0.5])
  r._pose = lambda *a, **k: (_ for _ in ()).throw(AssertionError("read while it tumbles"))
  for _ in range(5 * 200):
    r.hold_waist()
  assert len(r.waist_sent) == 200
  steps = np.array([float(v[0]) for v in r.waist_sent])
  assert np.all(np.diff(steps) <= 1e-12) and np.all(np.diff(steps) >= -rc.WAIST_RATE * 0.01 - 1e-12)
  assert steps[-1] == 0.0 and r.posture[2] == 0.0
