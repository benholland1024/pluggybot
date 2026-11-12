"""The step (issue #525; docs/Hands.md): the body's feet moved to its work,
each rule pinned without a flight -- the stub body, a fake walk, the
odometry fed its soles (docs/Testing.md). The flights that measured them
are SimNotes' "Stepping toward the work".

  1. `step` reports where the stance went in the frame it stood in before,
     read as sensors, and a place read before is turned by them;
  2. a step holding something walks, its hand-over carrying it (#682;
     `tests/test_carry.py` pins the carry);
  3. the humanoid's step hands its legs to the walker before it measures
     anything, turns first -- its heading counted unwrapped -- stops its
     walk where what its legs reckon it has walked and its stride's carry reach
     the step, walks an oblique step the walker would skew as two, reports a
     turn as on the spot, says `blocked` where its walk gets nowhere and
     `fell` for a fall up to its last settle, commands nothing once fallen,
     and walks holding something;
  4. its odometry takes a shuffle's swing foot off the floor;
  5. a walk its own legs reckon went nowhere reads false with its reason, and
     a turn closes on the STANCE's heading (issue #649), its text saying so
     off its constants;
  6. a step keeps off another robot as a walk does (issue #670): the walks'
     disc round its middle, judged where the stride's carry would take it,
     and begun inside it no nearer; the walks' front stop turned to the way
     it walks, on the LIDAR's returns off that robot alone; each read false
     saying how, its text off the walks' own sizes.
"""

import ast
import math
import re
from types import SimpleNamespace as NS

import numpy as np
import pytest

from pluggybot import tick
from pluggybot.body import KeepClear, StubBody
from pluggybot.challenge import solutions as sol
from pluggybot.evaluation import hands_check as hc
from pluggybot.humanoid import body as hb
from pluggybot.humanoid import odometry as od
from pluggybot.humanoid.mission import HumanoidMission
from pluggybot.navigator import GRID_M, Navigator
from pluggybot.perception.lidar import LIDAR_PERIOD
from pluggybot.procedure import steps as st

from test_guard import _floor_body
from test_hands import _bare, _fly


def test_a_step_reports_where_the_feet_went_and_a_place_read_before_turns_by_it():
  body = StubBody(*_bare())
  body.start_at(2.0, 1.0, math.pi / 2)
  ran = _fly('def a():\n    step(0.2, 0.1, 0.3)\n    dx = read("step.dx")\n'
             '    dy = read("step.dy")\n    t = read("step.turn")\n', body)
  assert ran["ok"] and body.stepped == [(0.2, 0.1, 0.3)]
  dx, dy, t = (ran["locals"][k] for k in ("dx", "dy", "t"))
  assert t == pytest.approx(0.3)
  # the stub turns, then walks (0.2, 0.1) in the frame it then stands in
  assert (dx, dy) == pytest.approx((0.2 * math.cos(0.3) - 0.1 * math.sin(0.3),
                                    0.2 * math.sin(0.3) + 0.1 * math.cos(0.3)), abs=1e-4)
  # ...and the verb's own formula turns a place read before into the frame
  # it stands in now: a world point stays where it was
  wx, wy = 2.0 - 0.15, 1.0 + 0.4                   # stance (0.4, 0.15) before
  c, s = math.cos(t), math.sin(t)
  x, y = c * (0.4 - dx) + s * (0.15 - dy), c * (0.15 - dy) - s * (0.4 - dx)
  bx, by, byaw = body.stance
  assert (bx + math.cos(byaw) * x - math.sin(byaw) * y,
          by + math.sin(byaw) * x + math.cos(byaw) * y) == pytest.approx((wx, wy), abs=1e-3)
  # ...and before any step there is no reading
  ran = _fly('def a():\n    x = read("step.dx")\n    wait(1)\n')
  assert "failedAt" in ran and "step" in ran["steps"][-1]["reason"]


def test_the_stubs_step_and_grip_read_as_the_humanoids():
  """A step brings the hands home, as the walker takes its legs; a gripper
  starts shut, and shut on nothing reads closed (#525)."""
  body = StubBody(*_bare())
  assert body.grip_gap("left") == 0.0
  ran = _fly('def a():\n    reach("left", 0.3, 0.2, 0.9)\n    open_hand("left")\n'
             '    g = read("left.gap")\n    step(0.2)\n    if not pull("left", 0, 0, 1, 10, 0.05):\n'
             '        wait(0.1)\n', body)
  pull = [s_ for s_ in ran["steps"] if s_["verb"] == "pull"][0]
  assert ran["locals"]["g"] == pytest.approx(0.09) and "is not out" in pull["reason"]
  ran = _fly('def a():\n    if not close_hand("left"):\n        g = read("left.gap")\n', body)
  assert ran["locals"]["g"] == 0.0


def test_a_step_carries_what_a_hand_holds():
  """It refused, `holding` (#525); a walk carries a hand that holds now (#682)."""
  body = StubBody(*_bare())
  ran = _fly('def a():\n    reach("left", 0.3, 0.2, 0.9)\n    close_hand("left")\n'
             '    if step(0.2):\n        x = read("step.dx")\n', body)
  step = ran["steps"][2]
  assert step["ok"] and step["why"] == "stepped" and step["carried"] == ["left"]
  assert ran["locals"]["x"] == pytest.approx(0.2) and body.pose[:2] == pytest.approx((0.2, 0.0))


# ---- the humanoid's step, on a fake walk ------------------------------------------


class Walk:
  """The walking policy as the step drives it: each command a physics step
  at the twist asked, its believed pose (the pelvis's) moving so; a stride's
  carry after the last command (`coast`); `wall`, where its walk gets
  nowhere; the legs handed to it in `handover_s`, the pelvis coming
  `handover_m` forward over the feet meanwhile (out of a bow); a fall
  at its command `fall_at`; and its first `shuffle_s` of walking a
  shuffle, the believed pose still (the feet slide). Another robot is where
  `others` says (#670: the Navigator's own `_bodies` and `_cells`, on the
  humanoid's sizes), and `seen`, (bearings, ranges) off its body in the
  trunk's frame, is the LIDAR's every scan's (`LIDAR_PERIOD`)."""

  OTHER_ROBOT_CELLS = HumanoidMission.OTHER_ROBOT_CELLS
  DOWN_ROBOT_CELLS = HumanoidMission.DOWN_ROBOT_CELLS
  FRONT_STOP_RANGE, FRONT_HALF_M = HumanoidMission.FRONT_STOP_RANGE, HumanoidMission.FRONT_HALF_M
  grid = NS(resolution=GRID_M)
  _bodies, _cells, in_front = Navigator._bodies, Navigator._cells, HumanoidMission.in_front
  held_s = Navigator.held_s

  def __init__(self, coast_s=0.22, wall=None, turn_drift=0.0, handover_s=0.0, handover_m=0.0,
               fall_at=None, settle_m=0.0, shuffle_s=0.0, planted_s=0.0, feet=1.0, others=(),
               seen=None):
    self.others, self.seen, self.peer_returns = list(others), seen, None
    self.pose, self.falls, self.legs, self.t = (0.0, 0.0, 0.0), 0, "stand", 0.0
    self.coast_s, self.wall, self.v, self.asked = coast_s, wall, (0.0, 0.0), []
    #: how far a turn's shuffle walks the BELIEF a command, m
    self.turn_drift = turn_drift
    self.handover_s, self.handover_m, self.fall_at = handover_s, handover_m, fall_at
    #: how far a turn's feet settle as the legs are handed back, m
    self.settle_m = settle_m
    #: where the pelvis stands ahead of the feet's middle, m
    self.lean = -handover_m
    #: its reckoning's ledger of corrections (`StanceOdometry.corrected`),
    #: and one a matcher makes `correct_at` its command: (dx, dy, dyaw)
    self.odo = NS(corrected=np.zeros(3))
    self.correct_at, self.correction = None, (0.0, 0.0, 0.0)
    self.shuffle_s, self.walked_s = shuffle_s, 0.0
    #: its first `planted_s` of walking its pelvis leans over its feet, the
    #: stance staying; and what `feet` of a turn's command the feet follow,
    #: the stance's heading its own (the pelvis's leads, #649)
    self.planted_s, self.feet, self.stance_yaw = planted_s, feet, 0.0

  @property
  def stance(self):
    x, y, yaw = self.pose
    return (x - math.cos(yaw) * self.lean, y - math.sin(yaw) * self.lean,
            yaw if self.feet == 1.0 else self.stance_yaw)

  def _getup_routine(self):
    return
    yield

  def to_walk_routine(self):
    if self.legs == "walk":
      return
    for k in range(int(round(self.handover_s / DT))):
      x, y, yaw = self.pose
      came = self.handover_m / max(1, int(round(self.handover_s / DT)))
      self.pose, self.lean = (x + math.cos(yaw) * came, y + math.sin(yaw) * came, yaw), self.lean + came
      self.t += DT
      yield hb.STILL
    self.legs = "walk"

  def _carry(self):
    """A walk's stride carries on after its last command."""
    if self.asked and any(self.asked[-1][:2]):
      x, y, yaw = self.pose
      self.pose = (x + self.v[0] * self.coast_s, y + self.v[1] * self.coast_s, yaw)
    self.v = (0.0, 0.0)

  def _twist_routine(self, vx, vy, w):
    yield from self.to_walk_routine()
    if self.asked and self.asked[-1] != (vx, vy, w):
      self._carry()
    self.asked.append((vx, vy, w))
    if self.fall_at is not None and len(self.asked) == self.fall_at:
      self.falls += 1
    if self.correct_at is not None and len(self.asked) == self.correct_at:
      # a correction from outside the legs' reckoning: the belief moves, the body not
      x, y, yaw = self.pose
      d = self.correction
      self.pose = (x + d[0], y + d[1], yaw + d[2])
      self.odo.corrected = self.odo.corrected + np.array(d)
    x, y, yaw = self.pose
    c, s = math.cos(yaw), math.sin(yaw)
    nx, ny = x + (c * vx - s * vy) * DT, y + (s * vx + c * vy) * DT
    if vx or vy:
      self.walked_s += DT
      if self.walked_s <= self.shuffle_s:
        nx, ny = x, y
    if self.planted_s and self.walked_s <= self.planted_s and (vx or vy):
      # its pelvis goes on over feet that stay
      self.lean += math.hypot(nx - x, ny - y)
    self.stance_yaw += self.feet * w * DT
    if w and not (vx or vy):
      nx += self.turn_drift
    if self.wall is not None and nx > self.wall:
      nx = x
    self.v = ((nx - x) / DT, (ny - y) / DT)
    self.pose = (nx, ny, yaw + w * DT)
    self.t += DT
    if self.seen is not None and round(self.t / DT) % round(LIDAR_PERIOD / DT) == 0:
      self.peer_returns = (self.t, *(np.array(v, dtype=float) for v in self.seen))
    yield hb.STILL

  def to_stand_routine(self):
    if self.asked and not any(self.asked[-1][:2]):
      x, y, yaw = self.pose
      self.pose = (x + self.settle_m, y, yaw)
    self._carry()
    self.legs = "stand"
    return
    yield


DT = 0.002


def _walker(walk) -> hb.HumanoidBody:
  body = object.__new__(Stepper)
  body.mission = walk
  body._d = NS(time=0.0)
  body._m = NS(opt=NS(timestep=DT))
  return body


class Stepper(hb.HumanoidBody):
  """The humanoid's `step_routine` on a fake walk: its stance the walk's
  pose, its time the walk's, its grippers holding `held`."""

  held = False

  @property
  def stance(self):
    return self.mission.stance

  def gripped(self, side):
    return self.held


def _step(body, *args, fall_in_settle=False):
  """The step driven to its end, its clock the walk's -- a fall, where
  asked, once its legs are handed back to the standing policy."""
  routine = body.step_routine(*args)
  try:
    while True:
      next(routine)
      body._d.time = body.mission.t
      if fall_in_settle and body.mission.legs == "stand" and body.mission.asked:
        body.mission.falls, fall_in_settle = body.mission.falls + 1, False
  except StopIteration as e:
    return e.value


def test_the_humanoids_step_stops_where_its_stride_carries_it_to_the_step():
  """At the walker's slowest its feet slid and its odometry counted a
  stride backward; at `STEP_V` it lifts them, and stopped commanding at
  what its legs reckon it has walked plus its speed times `STEP_STOP_S`, the
  stride's carry takes it to the step asked (#525). The fake walk is at
  its pace from the first command, which its speed's reading takes 0.2 s
  to follow, so the steps here are long enough for the reading to have."""
  walk = Walk(coast_s=hb.STEP_STOP_S)
  rep = _step(_walker(walk), 0.3)
  assert rep["why"] == "stepped" and rep["dx"] == pytest.approx(0.3, abs=0.004)
  assert {round(vx, 3) for vx, _, _ in walk.asked} == {hb.STEP_V}
  # ...backing up at its own pace, and sideways in one motion
  walk = Walk(coast_s=hb.STEP_STOP_S)
  rep = _step(_walker(walk), -0.3, 0.2)
  assert (rep["dx"], rep["dy"]) == pytest.approx((-0.3, 0.2), abs=0.004)
  assert math.hypot(*walk.asked[0][:2]) == pytest.approx(hb.BACK_V)


def test_the_humanoids_step_turns_first_and_takes_the_turn_as_on_the_spot():
  walk = Walk(coast_s=hb.STEP_STOP_S)
  rep = _step(_walker(walk), 0.3, 0.0, 0.5)
  turns = [w for _, _, w in walk.asked if w]
  walks = [i for i, (vx, _, _) in enumerate(walk.asked) if vx]
  assert turns and all(w == hb.TURN_W for w in turns)
  assert max(i for i, (_, _, w) in enumerate(walk.asked) if w) < min(walks)
  assert rep["turn"] == pytest.approx(0.5, abs=hb.TURN_STOP_RAD + TURN_STEP)
  assert (rep["dx"], rep["dy"]) == pytest.approx((0.3 * math.cos(rep["turn"]),
                                                  0.3 * math.sin(rep["turn"])), abs=0.004)
  # ...and its turn's shuffle walks the belief, which the step leaves out:
  # turned alone, the feet went nowhere; turned and walked, the walk alone
  walk = Walk(turn_drift=0.0001, settle_m=0.01)
  rep = _step(_walker(walk), 0.0, 0.0, -0.4)
  assert walk.pose[0] > 0.01 and (rep["dx"], rep["dy"]) == (0.0, 0.0) and rep["turn"] < -0.35
  walk = Walk(coast_s=hb.STEP_STOP_S, turn_drift=0.0001)
  rep = _step(_walker(walk), 0.3, 0.0, 0.5)
  assert math.hypot(rep["dx"], rep["dy"]) == pytest.approx(0.3, abs=0.004)


TURN_STEP = hb.TURN_W * DT


def test_the_humanoids_step_measures_nothing_until_its_legs_are_the_walkers():
  """Out of a deep bow the pelvis came 16 cm forward over the feet as the
  arms came home, which a walk measured from before it counted as walked
  (ok, without a step); and the hand-over outlasted `STEP_STUCK_S`, which
  read as `blocked` on a bare floor (#525)."""
  walk = Walk(coast_s=hb.STEP_STOP_S, handover_s=1.5, handover_m=0.16)
  rep = _step(_walker(walk), 0.3)
  assert rep["why"] == "stepped" and rep["dx"] == pytest.approx(0.3, abs=0.01)
  walk = Walk(coast_s=hb.STEP_STOP_S, handover_s=1.5)
  rep = _step(_walker(walk), -0.3)
  assert rep["why"] == "stepped" and rep["dx"] == pytest.approx(-0.3, abs=0.01)


def test_an_oblique_step_the_walker_would_skew_is_walked_as_two():
  """The walker raises a forward speed under `V_MIN` to it and leaves the
  sideways one: a step at `STEP_V` more than 48 deg off its way walked 12 cm
  further ahead than asked (#525)."""
  assert hb._step_walks(0.2, 0.1) == [(0.2, 0.1)]
  assert hb._step_walks(0.0, 0.2) == [(0.0, 0.2)] and hb._step_walks(-0.3, 0.0) == [(-0.3, 0.0)]
  assert hb._step_walks(0.03, 0.15) == [(0.03, 0.0), (0.0, 0.15)]
  assert hb._step_walks(-0.05, 0.2) == [(-0.05, 0.0), (0.0, 0.2)]
  assert hb._step_walks(0.0, 0.0) == []
  walk = Walk(coast_s=hb.STEP_STOP_S)
  rep = _step(_walker(walk), 0.3, 0.45)
  assert all(abs(vx) < 1e-9 or abs(vx) >= hb.V_MIN for vx, _, _ in walk.asked)
  assert (rep["dx"], rep["dy"]) == pytest.approx((0.3, 0.45), abs=0.006)


def test_a_turn_past_pi_ends_and_a_fall_up_to_the_last_settle_is_a_fall():
  walk = Walk()
  rep = _step(_walker(walk), 0.0, 0.0, 3.2)
  assert rep["why"] == "stepped" and walk.t < 4.0
  assert rep["turn"] == pytest.approx(math.remainder(3.2, 2 * math.pi), abs=hb.TURN_STOP_RAD + TURN_STEP)
  # ...a fall as it settles is a fall, and its report no step
  walk = Walk(coast_s=hb.STEP_STOP_S)
  assert _step(_walker(walk), 0.3, fall_in_settle=True)["why"] == "fell"
  # ...and fallen, it commands nothing more
  walk = Walk(fall_at=10)
  assert _step(_walker(walk), 0.3)["why"] == "fell" and len(walk.asked) == 10
  walk = Walk(fall_at=10)
  assert _step(_walker(walk), 0.0, 0.0, 1.0)["why"] == "fell" and len(walk.asked) == 10


def test_a_step_walks_by_its_legs_never_by_a_correction_the_matcher_made_meanwhile():
  """A correction is no step (#581's stage 2): the band's matcher at its
  station moved the belief 6 cm in a step's 2 s, the step walked by the
  belief stopped 3.6 cm inside the seat, and the walker cannot back up so
  little. A step walks, turns and reports by its legs' reckoning alone: the
  belief jumping 6 cm along its way mid-walk, its feet still go 0.3 m, and
  its heading corrected 0.1 rad mid-turn, it still turns 0.5."""
  walk = Walk(coast_s=hb.STEP_STOP_S)
  walk.correct_at, walk.correction = 60, (0.06, 0.0, 0.0)
  rep = _step(_walker(walk), 0.3)
  went = walk.pose[0] - walk.odo.corrected[0]
  assert rep["why"] == "stepped" and went == pytest.approx(0.3, abs=0.004)
  assert rep["dx"] == pytest.approx(0.3, abs=0.004), "its report counted the correction"
  walk = Walk()
  walk.correct_at, walk.correction = 100, (0.0, 0.0, 0.1)
  rep = _step(_walker(walk), 0.0, 0.0, 0.5)
  turned = walk.pose[2] - walk.odo.corrected[2]
  assert turned == pytest.approx(0.5, abs=hb.TURN_STOP_RAD + TURN_STEP)
  assert rep["turn"] == pytest.approx(0.5, abs=hb.TURN_STOP_RAD + TURN_STEP)


def test_a_step_is_given_its_first_window_to_get_going():
  """From the standing policy's stance after a reach, a sidestep to the
  right slid up to 1.1 s before its first stride, unseen by its legs'
  reckoning, and 2 of 7 at the table read "blocked" in a 1 s window (#560):
  a walk's first window is `STEP_START_S`."""
  walk = Walk(coast_s=hb.STEP_STOP_S, shuffle_s=1.1)
  rep = _step(_walker(walk), 0.0, -0.21)
  assert rep["why"] == "stepped", "a slide before its first stride read as blocked"
  assert rep["dy"] == pytest.approx(-0.21, abs=0.01)


def test_a_step_that_gets_nowhere_says_it_is_blocked_and_one_holding_walks():
  walk = Walk(wall=0.05)
  rep = _step(_walker(walk), 0.3)
  assert rep["why"] == "blocked" and rep["dx"] == pytest.approx(0.05, abs=0.002)
  assert walk.t <= hb.STEP_START_S + hb.STEP_STUCK_S + 0.01
  # ...and a walk given less time than it takes runs out of it
  walk = Walk()
  assert _step(_walker(walk), 0.3, 0.0, 0.0, 0.1)["why"] == "out of time"
  # ...and one that never gets going, once its first window is over
  walk = Walk(shuffle_s=60.0)
  rep = _step(_walker(walk), 0.0, -0.21)
  assert rep["why"] == "blocked" and hb.STEP_START_S <= walk.t <= hb.STEP_START_S + 0.01
  # ...and with something in its hand it turns and walks all the same (#682:
  # the walk's hand-over carries it; it refused, `holding`, before)
  walk = Walk()
  body = _walker(walk)
  body.held = True
  rep = _step(body, 0.3, 0.0, 0.5)
  assert rep["why"] == "stepped" and walk.asked != []


def test_the_body_text_says_a_walk_carries_a_hand_that_holds_something():
  """#669's second review read the body text against what a step does (a
  step holding a knob, read as letting go and walking, 12 times of 12): a
  walk carries a hand that holds something now (#682; Ben, 2026-11-12: the
  walk takes the carry pose itself and says so). What the robot is served
  says it, beside the walks' let-go, in the practice's body and the served
  rule alike -- and no longer that a step holding something moves nothing."""
  from pluggybot.mind import overseer as ov
  from pluggybot.mind import practice as mp
  walk = Walk()
  body = _walker(walk)
  body.held = True
  rep = _step(body, 0.3, 0.0, 0.5)
  assert rep["why"] == "stepped" and walk.asked != [], "the premise: a step holding walks"
  for text in (dict(mp.system_sections(st.hand_facts()))["your body"],
               ov.hands_procedure_rule()):
    flat = " ".join(text.split())
    assert "a hand that is out comes home, its gripper opened as `withdraw` opens it" in flat
    assert ("a hand that is out whose gripper says it holds something is CARRIED: it comes "
            "back out along the way it went in, then its arm folds to your side, the forearm "
            "up and the hand at your hips, turned as it was against the floor, its gripper "
            "still shut.") in flat, "the carry goes unsaid beside the let-go"
    assert "says so in its report (`carried`;" in flat
    assert "moves nothing, and reads false" not in flat, "the old refusal is still said"
  assert "holding" not in st.STEP_WHY
  assert "a hand that holds something carried at your side" in st.HANDS_VERBS["step"].doc


def test_a_walk_its_stance_did_not_make_says_it_was_too_short():
  """#649: 3 cm ahead, the walker's first 0.3 s are its pelvis leaning over
  planted feet; the walk's stop reading the pelvis ended it there, nothing
  had moved and it read `stepped`, and a robot's creep loop of six went
  nowhere believing it had. Judged by what its own legs reckon the stance
  did, a walk of at least `STEP_LEAST_M` that went under it says so."""
  rep = _step(_walker(Walk(coast_s=0.0, planted_s=0.1)), 0.03)
  assert rep["why"] == "too short", "the stance stayed, and it read stepped"
  assert (rep["dx"], rep["dy"]) == pytest.approx((0.0, 0.0), abs=0.002)
  # ...the same walk asked long enough goes, once its stance moves
  rep = _step(_walker(Walk(coast_s=hb.STEP_STOP_S, planted_s=0.1)), 0.2)
  assert rep["why"] == "stepped" and rep["dx"] > 0.1
  # ...and an ask under the floor is no walk to judge
  rep = _step(_walker(Walk(coast_s=0.0, planted_s=1.0)), hb.STEP_LEAST_M / 2.0)
  assert rep["why"] == "stepped"
  # ...nor does a turn with no walk asked read short
  assert _step(_walker(Walk()), 0.0, 0.0, 0.5)["why"] == "stepped"


def test_a_step_too_short_reads_false_with_its_reason_and_says_where_it_got():
  body = StubBody(*_bare())
  body.step_routine = lambda *a, **kw: tick.result(
    {"why": "too short", "seconds": 1.5, "dx": 0.002, "dy": 0.0, "turn": 0.0})
  ran = _fly('def a():\n    if not step(0.03):\n        d = read("step.dx")\n', body)
  step = ran["steps"][0]
  assert not step["ok"] and step["why"] == "too short"
  assert step["reason"] == "your step was too short to walk" == st.STEP_WHY["too short"]
  assert ran["locals"]["d"] == pytest.approx(0.002)


# ---- a step keeps off another robot as a walk does (#670) ---------------------------


def _apart(walk, b):
  return math.hypot(b[0] - walk.pose[0], b[1] - walk.pose[1])


def test_a_step_stops_before_its_carry_takes_it_into_the_other_robots_disc():
  """11:48 on 11-08 (#670): Rowan's walk to the cupboard stopped at the disc
  it keeps round Luca, and its procedure stepped onto its work spot anyway --
  `step` kept no disc. A step's walk keeps the walks' disc round the other
  robot's middle (`Body.others`: 0.65 m standing), judged where its stride's
  carry would take it, and stops there `blocked`, saying how."""
  walk = Walk(coast_s=hb.STEP_STOP_S, others=[lambda: (1.0, 0.0)])
  rep = _step(_walker(walk), 0.5)
  assert rep["why"] == "blocked", "the step walked into the other robot's disc"
  assert rep["keptOff"] == {"by": "keep", "m": pytest.approx(1.0 - rep["dx"], abs=0.01),
                            "keepM": pytest.approx(hb.STEP_KEEP_M)}
  assert _apart(walk, (1.0, 0.0)) >= hb.STEP_KEEP_M - 0.01, "its stride carried it into the disc"
  assert rep["dx"] == pytest.approx(1.0 - hb.STEP_KEEP_M, abs=0.01)
  # ...round one lying down, its own wider disc (`DOWN_ROBOT_CELLS`)
  walk = Walk(coast_s=hb.STEP_STOP_S, others=[lambda: KeepClear(1.5, 0.0, down=True)])
  rep = _step(_walker(walk), 0.5)
  assert rep["why"] == "blocked" and rep["keptOff"]["keepM"] == pytest.approx(hb.STEP_KEEP_DOWN_M)
  assert _apart(walk, (1.5, 0.0)) >= hb.STEP_KEEP_DOWN_M - 0.01
  # ...and a walk that passes the disc by goes its whole way
  walk = Walk(coast_s=hb.STEP_STOP_S, others=[lambda: (0.5, 0.7)])
  rep = _step(_walker(walk), 0.5)
  assert rep["why"] == "stepped" and "keptOff" not in rep


def test_a_step_begun_inside_the_other_robots_disc_goes_only_further_from_it():
  """A walk inside the other robot's disc walks out of it (#455), never
  through it: a step begun 0.5 m from its middle whose walk heads nearer goes
  nowhere -- nor turns, nor hands its legs to the walker: flown, each such
  step took 1.3 s and moved its feet -- and goes away from it, or past it,
  as asked, a turn first included."""
  where = [lambda: (0.3, 0.4)]                    # 0.5 m off, ahead and to the left
  for ask in ((0.3,), (0.0, 0.2), (0.3, 0.0, math.pi / 2), (-0.3, 0.0, math.pi)):
    walk = Walk(coast_s=hb.STEP_STOP_S, others=where)
    rep = _step(_walker(walk), *ask)
    assert rep["why"] == "blocked" and rep["keptOff"]["by"] == "keep", ask
    assert rep["keptOff"]["m"] == pytest.approx(0.5, abs=1e-3)
    assert walk.asked == [] and walk.legs == "stand", f"{ask} moved"
    assert (rep["seconds"], rep["dx"], rep["dy"], rep["turn"]) == (0.0, 0.0, 0.0, 0.0)
  for ask in ((-0.3,), (0.0, -0.2), (0.3, 0.0, math.pi)):
    walk = Walk(coast_s=hb.STEP_STOP_S, others=where)
    assert _step(_walker(walk), *ask)["why"] == "stepped", ask
    assert _apart(walk, (0.3, 0.4)) > 0.5


def test_a_step_stops_once_the_lidar_puts_the_other_robots_body_in_its_way():
  """The walks' front stop reads the LIDAR, which sees the other robot
  (#316): a step's reads it too, turned to the way it walks, from the scans
  the step has taken, on the other robot's returns alone (its body 0.35 m
  ahead stops a step ahead, and one to its left a sidestep left; one behind
  or to its right, none)."""
  ahead, to_left = ((0.0,), (0.35,)), ((math.pi / 2,), (0.3,))
  behind, to_right = ((math.pi,), (0.3,)), ((-math.pi / 2,), (0.3,))
  for seen, ask in ((ahead, (0.3,)), (to_left, (0.0, 0.3))):
    walk = Walk(coast_s=hb.STEP_STOP_S, seen=seen)
    rep = _step(_walker(walk), *ask)
    assert rep["why"] == "blocked" and rep["keptOff"]["by"] == "body", ask
    assert rep["keptOff"]["m"] == pytest.approx(seen[1][0], abs=1e-6)
    assert walk.t < 2 * LIDAR_PERIOD + 0.6, "it walked on past the scan that saw it"
  for seen, ask in ((behind, (0.3,)), (to_right, (0.0, 0.3))):
    walk = Walk(coast_s=hb.STEP_STOP_S, seen=seen)
    assert _step(_walker(walk), *ask)["why"] == "stepped", ask
  # ...nor beyond the front stop's reach and the stride's carry at the
  # step's pace, nor its half-width either side
  carry = hb.STEP_V * hb.STEP_STOP_S
  for seen in (((0.0,), (hb.STEP_FRONT_M + carry + 0.01,)),
               ((math.atan2(hb.STEP_FRONT_HALF_M + 0.02, 0.2),),
                (math.hypot(hb.STEP_FRONT_HALF_M + 0.02, 0.2),))):
    walk = Walk(coast_s=hb.STEP_STOP_S, seen=seen)
    assert _step(_walker(walk), 0.3)["why"] == "stepped", seen
  # ...and a scan from before the step began is not read: the body put
  # there, its scans stopped as the step began
  walk = Walk(coast_s=hb.STEP_STOP_S)
  walk.peer_returns = (0.0, np.array([0.0]), np.array([0.3]))
  walk.t = 0.001
  body = _walker(walk)
  body._d.time = walk.t
  assert _step(body, 0.3)["why"] == "stepped"


def test_a_steps_front_stop_counts_its_carry_and_every_scan_the_step_took():
  """#672's review: stopped where the body came within the walks' 0.40 m, the
  stride's carry walked it 6-8 cm nearer, sideways about 5 cm off the robot's
  own hands: the body is judged where the carry would take the step, as the
  disc is. And a step's second walk (ahead, then left: `_step_walks`) reads
  the scans its first walk took: read only from its own start, it walked
  toward the body until the next scan."""
  walk = Walk(coast_s=hb.STEP_STOP_S, seen=((0.0,), (hb.STEP_FRONT_M + 0.03,)))
  rep = _step(_walker(walk), 0.3)
  assert rep["why"] == "blocked" and rep["keptOff"]["by"] == "body"
  assert rep["dx"] < 0.3 - 0.05, "its carry took it on into the stop"
  # ...and the second walk of a split step, the body to its left, takes no
  # stride left: the first walk's scans saw it
  assert hb._step_walks(0.1, 0.4) == [(0.1, 0.0), (0.0, 0.4)]
  walk = Walk(coast_s=hb.STEP_STOP_S, seen=((math.pi / 2,), (0.3,)))
  rep = _step(_walker(walk), 0.1, 0.4)
  assert rep["why"] == "blocked" and rep["keptOff"]["by"] == "body"
  assert any(vx for vx, _, _ in walk.asked), "its first walk did not go"
  assert not any(vy for _, vy, _ in walk.asked), "its second walk stepped toward the body"


def test_a_steps_every_walk_is_judged_before_it_moves_and_it_names_the_robot_it_kept_off():
  """#672's review: a split step's first walk away from the other robot was
  walked, legs handed over, and only then its second, heading nearer from
  inside the disc, refused; each of a step's walks is judged from where the
  one before it leaves it, before anything moves. And with two robots about,
  the distance a blocked step says is to the one whose disc stopped it."""
  # 0.56 m off, behind and to the left: ahead goes away, then left goes nearer
  walk = Walk(coast_s=hb.STEP_STOP_S, others=[lambda: (-0.05, 0.55)])
  rep = _step(_walker(walk), 0.1, 0.4)
  assert rep["why"] == "blocked" and rep["keptOff"]["by"] == "keep"
  assert walk.asked == [] and walk.legs == "stand", "its first walk went before the refusal"
  # ...a fallen robot's 1.2 m disc stops a step ahead; one standing 0.9 m to
  # its right is nearer, and is not the one it kept off
  walk = Walk(coast_s=hb.STEP_STOP_S, others=[lambda: KeepClear(1.5, 0.0, down=True),
                                              lambda: (0.25, -0.9)])
  rep = _step(_walker(walk), 0.5)
  assert rep["why"] == "blocked" and rep["keptOff"]["keepM"] == pytest.approx(hb.STEP_KEEP_DOWN_M)
  assert rep["keptOff"]["m"] == pytest.approx(_apart(walk, (1.5, 0.0)), abs=1e-3)
  assert "of" not in rep["keptOff"]


def test_the_scan_keeps_its_returns_off_another_robot_for_a_step():
  """The one wire from the LIDAR to a step's front stop: each scan keeps the
  returns it split off as another robot's, stamped, beside the map's and the
  walks' front stop's (`Navigator._scan_step`)."""
  from pluggybot.perception.lidar import Scan
  nav = object.__new__(Navigator)
  peers = (np.array([0.1]), np.array([0.6]))
  nav.data, nav._next_scan, nav.backoff_until = NS(time=3.0), 0.0, 99.0
  nav.LIDAR_ORIGIN, nav.peer_returns = (0.0, 0.0), None
  nav._scan = lambda: Scan(np.array([0.0]), np.array([2.0]), *peers)
  nav.maps = lambda: False
  nav.matcher = None
  Navigator._scan_step(nav)
  assert nav.peer_returns[0] == 3.0
  assert [list(a) for a in nav.peer_returns[1:]] == [list(a) for a in peers]


def test_a_step_kept_off_another_robot_reads_false_saying_how():
  """The words (#670): a step blocked by another robot says so, with the
  distance and the disc it keeps -- never "something stands in your way"."""
  for kept, said in (({"by": "keep", "m": 0.6, "keepM": 0.65},
                      "your step stopped short of another robot: a step keeps your middle "
                      "0.65 m from its middle, and within that goes only further from it "
                      "(0.60 m now)"),
                     ({"by": "body", "m": 0.34},
                      "your step stopped short of another robot: its body was 0.34 m from "
                      "your middle in the way you were stepping")):
    body = StubBody(*_bare())
    body.step_routine = lambda *a, kept=kept, **kw: tick.result(
      {"why": "blocked", "seconds": 1.0, "dx": 0.04, "dy": 0.0, "turn": 0.0, "keptOff": kept})
    ran = _fly('def a():\n    if not step(0.3):\n        d = read("step.dx")\n', body)
    step = ran["steps"][0]
    assert not step["ok"] and step["why"] == "blocked" and step["reason"] == said
    assert ran["locals"]["d"] == pytest.approx(0.04)
  # ...and a walk that got nowhere says what it always did
  assert st.STEP_WHY["blocked"] == "your step got nowhere: something stands in your way"


def test_a_small_turn_is_closed_on_the_stance_not_the_pelvis():
  """#649, #643's option (b): the walker yaws its pelvis over planted feet
  first, so a turn stopped on the pelvis's heading left the feet where they
  were: asked 3 deg, the stance turned 0.3, and 15 to the right 9. The
  step's turn pulses on, read after a settle, until the STANCE'S heading is
  within `TURN_TOL_RAD`, at most `TURN_PULSES`."""
  asked = math.radians(3.0)
  walk = Walk(feet=0.25)
  rep = _step(_walker(walk), 0.0, 0.0, asked)
  assert rep["why"] == "stepped"
  assert walk.stance_yaw == pytest.approx(asked, abs=hb.TURN_TOL_RAD + 2 * TURN_STEP), (
    "the feet follow the pelvis a quarter of the way: stopped on the pelvis it turned 0.75 deg")
  # ...and the other way, and a big turn's walker part less what its feet left
  walk = Walk(feet=0.8)
  rep = _step(_walker(walk), 0.0, 0.0, -0.5)
  assert walk.stance_yaw == pytest.approx(-0.5, abs=hb.TURN_TOL_RAD + 2 * TURN_STEP)
  # ...and feet that never follow end it within its pulses, never forever, and say so
  walk = Walk(feet=0.0)
  rep = _step(_walker(walk), 0.0, 0.0, asked)
  assert rep["why"] == "turned short" and walk.stance_yaw == 0.0
  assert walk.t <= hb.TURN_PULSES * (max(hb._turn_pulse(1.0), 0.14) + hb.TURN_SETTLE_S) + 3.0
  # ...and a turn its pulses left nearly there is no failure
  assert _step(_walker(Walk(feet=0.0)), 0.0, 0.0, 0.9 * hb.TURN_MISS_RAD)["why"] == "stepped"


def test_a_turns_closing_has_its_own_time_over_the_steps_patience():
  """#649's review: the closing's settles and pulses spent the step's one
  patience, so a big turn from an arm out with a long walk could read out
  of time for the walk it never began."""
  walk = Walk(feet=0.25)
  rep = _step(_walker(walk), 0.0, 0.0, math.radians(3.0), 1.0)
  assert rep["why"] == "stepped" and walk.t > 1.0, "the closing ran out the step's patience"
  # ...and a step with no turn keeps its patience as it was
  assert _step(_walker(Walk()), 0.3, 0.0, 0.0, 0.1)["why"] == "out of time"


def test_a_reference_takes_a_correction_too_short_to_walk_in_its_stride():
  """#649's review: a step computed from a reading can come out a few cm,
  and a walk that did not go now reads false: unguarded, a reference ended
  there where it used to pass through. Each of our references' computed
  corrections is an `if`'s test of its own."""
  pat = re.compile(r"%\(\w+\)[gds]")
  for name, src in (("APPROACH", sol.APPROACH), ("TABLE_APPROACH", sol.TABLE_APPROACH),
                    ("PLACE", hc.PLACE)):
    tree = ast.parse(pat.sub("0", src))
    tested = {id(n.test.operand) for n in ast.walk(tree)
              if isinstance(n, ast.If) and isinstance(n.test, ast.UnaryOp)
              and isinstance(n.test.op, ast.Not) and [type(b) for b in n.body] == [ast.Pass]}
    steps = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "step"]
    assert steps and all(id(c) in tested for c in steps), f"a computed step in {name} stands alone"


def test_a_turns_pulses_are_shorter_than_a_stride_and_never_read_a_settling_walker():
  """A pulse is sized by what it has left to turn, and every one under the
  knee at 0.2 s where a foot lifts and the stance jumps 7-14 deg; a settle
  of `TURN_SETTLE_S` between (0.3 s after the walker's own turn, a 0.14 s
  pulse moved the stance 7.8 deg, from rest 2-3)."""
  sizes = [hb._turn_pulse(math.radians(d)) for d in (0.3, 0.9, 1.5, 3.0, 20.0)]
  assert sizes == sorted(sizes) and max(sizes) < 0.2 and min(sizes) > 0.0
  assert hb.TURN_SETTLE_S >= 0.4 and hb.TURN_SETTLE_BIG_S > hb.TURN_SETTLE_S
  assert hb.TURN_TOL_RAD < math.radians(1.0)


def test_the_steps_text_says_what_a_short_ask_does_off_its_constants():
  """#649: the text read "A walk goes at least a stride, about 10 cm": asked
  3 cm it went nowhere, and a robot built theories on it. Each number here
  is the constant's; a flown sweep measured it (SimNotes, "Stepping toward
  the work")."""
  doc = st.HANDS_VERBS["step"].doc
  assert f"under {100 * hb.STEP_SHORT_M:.0f} cm may take none" in doc
  assert f"asked at least {100 * hb.STEP_LEAST_M:.0f} cm" in doc
  assert "reads false and says your step was too short to walk" in doc
  assert f"{100 * hb.STEP_DY_SHORT_M[0]:.0f}-{100 * hb.STEP_DY_SHORT_M[1]:.0f} cm short" in doc
  assert f"up to {100 * hb.STEP_DY_LEFT_AFTER_M:.0f} cm or more short" in doc
  assert "at least a stride" not in doc
  # ...and the two ends a step reads false by, with their reasons
  assert st.STEP_WHY["too short"] == "your step was too short to walk"
  assert st.STEP_WHY["turned short"] == "your turn stopped short of where you asked"


def test_the_steps_text_says_how_it_keeps_off_another_robot_off_the_walks_sizes():
  """#670: what keeps a step off another robot is the walks' own disc and
  front stop, and its text says them off the same numbers the check reads
  (`Navigator._cells` times the map's cell, the body's front stop)."""
  doc = st.HANDS_VERBS["step"].doc
  # ...every number read back off the text, never formatted as the text
  # formats it (#672's review: a 1.25 m disc printed `:.1f` read "1.2")
  said = re.search(r"within ([0-9.]+) m of that robot's middle \(([0-9.]+) m of one that has "
                   r"fallen\).*within ([0-9.]+) m of your middle in the way you step and "
                   r"([0-9.]+) m either side of it", doc)
  assert said, doc
  assert [float(v) for v in said.groups()] == pytest.approx(
    [hb.STEP_KEEP_M, hb.STEP_KEEP_DOWN_M, hb.STEP_FRONT_M, hb.STEP_FRONT_HALF_M], abs=1e-9)
  assert "standing within that already" in doc
  assert "a step whose walk would take you nearer neither turns nor walks" in doc
  walk = Walk()
  assert walk._cells(KeepClear(0.0, 0.0)) * walk.grid.resolution == hb.STEP_KEEP_M
  assert walk._cells(KeepClear(0.0, 0.0, down=True)) * walk.grid.resolution == hb.STEP_KEEP_DOWN_M
  assert (hb.STEP_FRONT_M, hb.STEP_FRONT_HALF_M) == (
    HumanoidMission.FRONT_STOP_RANGE, HumanoidMission.FRONT_HALF_M)


def _flown_stance_step(*ask):
  """`ask` flown on the humanoid on a bare floor, stood still first: how it
  ended, and how the stance truly went in the frame it stood in."""
  body = _floor_body()
  body.run(body.hold_routine(1.0))
  a = body._true_stance()
  rep = body.run(body.step_routine(*ask))
  b = body._true_stance()
  c, s = math.cos(a[2]), math.sin(a[2])
  dx, dy = b[0] - a[0], b[1] - a[1]
  return rep, c * dx + s * dy, -s * dx + c * dy, math.remainder(b[2] - a[2], 2 * math.pi)


def test_a_short_walk_flown_either_goes_about_what_it_asks_or_reads_false():
  """#649, flown (a few seconds of sim): six `step(0, 0.03, 0)` ended with
  Rowan's knob still at `ky` 0.143 (tk_0054). Never `stepped` at under 1 cm."""
  rep, along, _, _ = _flown_stance_step(0.03)
  assert rep["why"] == "too short" or along >= 0.01, (
    f"read {rep['why']}, moved {100 * along:.1f} cm of 3")
  assert rep["why"] != "stepped" or along >= 0.01


@pytest.mark.parametrize("deg", [3.0, -3.0])
def test_a_small_turn_flown_turns_the_stance_within_a_degree(deg):
  """#649 (and #643): asked 1.7, 2.9 and 5.7 deg, the stance turned 0.03,
  0.3 and 1.1."""
  rep, _, _, turned = _flown_stance_step(0.0, 0.0, math.radians(deg))
  assert rep["why"] == "stepped" and abs(math.degrees(turned) - deg) <= 1.0, (
    f"asked {deg}, the stance turned {math.degrees(turned):.2f}")


def test_a_mid_turn_to_the_right_flown_is_not_left_short():
  """Stopped on the pelvis, a 15 deg turn to the right turned the stance
  6-13 deg from the same three starts (#649, measured; the pulses close it)."""
  rep, _, _, turned = _flown_stance_step(0.0, 0.0, math.radians(-15.0))
  assert abs(math.degrees(turned) + 15.0) <= 3.0, (
    f"the stance turned {math.degrees(turned):.1f} of -15")


# ---- the odometry and a shuffle ---------------------------------------------------


def _shuffle(odo, ticks=40, lift=0.005, stride=0.04, advance=0.02):
  """Feed the odometry a shuffle: the left sole lifted `lift` and slid
  `stride` forward while the right one, planted, goes `advance` back
  under the pelvis -- the walker's first stride at its slowest."""
  for k in range(ticks + 1):
    f = k / ticks
    up = lift if k else 0.0                        # both down at first
    mids = np.array([[-0.05 + stride * f, 0.1, -0.9 + up], [0.05 - advance * f, -0.1, -0.9]])

    def soles(q, g, mids=mids):
      odo.corners = np.array([[m + [a, b, 0.0] for a in (-0.1, 0.1) for b in (-0.04, 0.04)]
                              for m in mids])
      odo.flat = np.array([True, True])
      odo.low = odo.corners[:, :, 2].min(axis=1)
      return mids
    odo._soles = soles
    odo.step(None, None)


def test_the_odometry_takes_a_shuffles_swing_foot_off_the_floor(monkeypatch):
  """A shuffle lifts its swing foot under a centimetre: at 1 cm the
  odometry kept it planted and read the pelvis going BACK as it slid
  forward -- a short step's first stride lost -- and a sideways walk read
  21 % short (#525)."""
  att = NS(yaw=lambda: 0.0)
  odo = od.StanceOdometry(NS(soles=[0, 1]), att)
  odo.z = 0.9
  _shuffle(odo)
  assert odo.foot == 1 and odo.x == pytest.approx(0.02, abs=0.002)
  # ...where 1 cm kept the sliding foot planted, and the pelvis went back
  monkeypatch.setattr(od, "SWAP_M", 0.01)
  odo = od.StanceOdometry(NS(soles=[0, 1]), att)
  odo.z = 0.9
  _shuffle(odo)
  assert odo.foot == 0 and odo.x < -0.03


def test_the_odometry_sums_what_corrections_moved_its_belief_by():
  """A step walks by its legs' reckoning: the belief less what corrections
  from outside it moved it by since the step began (`_by_legs`, #581's
  stage 2). The ledger is every correction's move summed, its turn the
  short way round."""
  att = NS(yaw=lambda: 0.3)
  odo = od.StanceOdometry(NS(soles=[0, 1]), att, x=1.0, y=2.0)
  odo.correct(1.1, 1.95, 0.35)
  odo.correct(1.0, 2.05, 0.25)
  assert odo.corrected == pytest.approx([0.0, 0.05, -0.05])
  assert (odo.x, odo.y, odo.yaw) == pytest.approx((1.0, 2.05, 0.25))
  odo = od.StanceOdometry(NS(soles=[0, 1]), NS(yaw=lambda: 3.1))
  odo.correct(0.0, 0.0, -3.1)
  assert odo.corrected[2] == pytest.approx(2 * math.pi - 6.2)
