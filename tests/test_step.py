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
from pluggybot.humanoid import mission as hmis
from pluggybot.humanoid import odometry as od
from pluggybot.humanoid.mission import HumanoidMission, KneeMemory
from pluggybot.navigator import GRID_M, Navigator
from pluggybot.perception import lidar
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
               seen=None, over_wall=None, settled=True):
    self.others, self.seen, self.peer_returns = list(others), seen, None
    #: what its settle says as it hands the legs back (#740): True, both soles
    #: read down; False, its bound ran out first
    self.settled = settled
    #: what its LIDAR reads at its head's height and what it saw at its
    #: knees (#717): none, unless a test says -- `over_wall` a wall across
    #: the world at that x, read at every scan
    self.over_returns, self.knee, self.over_wall = None, KneeMemory(), over_wall
    self.pose, self.falls, self.legs, self.t = (0.0, 0.0, 0.0), 0, "stand", 0.0
    self.coast_s, self.wall, self.v, self.asked = coast_s, wall, (0.0, 0.0), []
    #: how far a turn's shuffle walks the BELIEF a command, m
    self.turn_drift = turn_drift
    self.handover_s, self.handover_m, self.fall_at = handover_s, handover_m, fall_at
    #: how far a turn's feet settle as the legs are handed back, m
    self.settle_m = settle_m
    #: where the pelvis stands ahead of the feet's middle, m -- and where a
    #: walk's first `planted_s` leant it over feet that stayed, m in the world
    self.lean, self.lean_xy = -handover_m, np.zeros(2)
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
    return (x - math.cos(yaw) * self.lean - self.lean_xy[0],
            y - math.sin(yaw) * self.lean - self.lean_xy[1],
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
    if self.correct_at is not None and len(self.asked) in (
        self.correct_at if isinstance(self.correct_at, (set, tuple)) else (self.correct_at,)):
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
      # its pelvis goes on over feet that stay, whichever way it walks
      self.lean_xy = self.lean_xy + (nx - x, ny - y)
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
    if round(self.t / DT) % round(LIDAR_PERIOD / DT) == 0:
      self._over_scan()
    yield hb.STILL

  def to_stand_routine(self):
    if self.asked and not any(self.asked[-1][:2]):
      x, y, yaw = self.pose
      self.pose = (x + self.settle_m, y, yaw)
    self._carry()
    self._over_scan()
    self.legs = "stand"
    return self.settled
    yield

  def look_scan(self):
    """A scan taken now (`_Room`'s, where a step has none, #717)."""
    self.scans_taken = getattr(self, "scans_taken", 0) + 1
    self._over_scan()

  def _over_scan(self):
    """Its LIDAR's scan at its head's height, where `over_wall` stands."""
    if self.over_wall is not None:
      x, y, yaw = self.pose
      a = np.radians(np.arange(-80.0, 81.0))
      self.over_returns = (self.t, a, (self.over_wall - x) / np.cos(a + yaw), (0.0, 0.0))


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


def test_a_step_whose_legs_went_back_before_both_soles_read_down_says_so():
  """#740: the settle hands the legs to the standing policy once both soles
  have read planted, and where its bound runs out first hands them over all
  the same: the step then says so (`feetDown`: false), in the body's report
  and in the verb's, which the robot reads -- and nothing where they did."""
  rep = _step(_walker(Walk(settled=False)), 0.2)
  assert rep["why"] == "stepped" and rep["feetDown"] is False
  assert "feetDown" not in _step(_walker(Walk()), 0.2)
  body = StubBody(*_bare())
  stepped = body.step_routine

  def bound_out(*a, **kw):
    return {**(yield from stepped(*a, **kw)), "feetDown": False}
  body.step_routine = bound_out
  ran = _fly('def a():\n    step(0.2)\n', body)
  assert ran["ok"] and ran["steps"][0]["feetDown"] is False
  body.step_routine = stepped
  ran = _fly('def a():\n    step(0.2)\n', body)
  assert ran["ok"] and "feetDown" not in ran["steps"][0]


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


class SideWall(Walk):
  """...its sideways walk held `side_m` along (0: none of it walked): what
  an oblique step's second walk (`hb._step_walks`) meets."""

  def __init__(self, side_m=0.0, **kw):
    super().__init__(**kw)
    self.side_m = side_m

  def _twist_routine(self, vx, vy, w):
    yield from super()._twist_routine(vx, vy, w)
    x, y, yaw = self.pose
    if abs(y) > self.side_m:
      self.pose, self.v = (x, math.copysign(self.side_m, y), yaw), (self.v[0], 0.0)


class Planted(SideWall):
  """...and its stance never moving, its pelvis leaning over planted feet
  (#649): an oblique step's walk ahead ends on its stop rule so, never
  having walked."""

  @property
  def stance(self):
    return (0.0, 0.0, self.pose[2])


#: A stall's words, typed: the test reads the template, never formats it.
GOT_NOWHERE = ("your step got nowhere: your legs counted under 1 cm along the way you stepped "
               "in the first 2 s of its walk")
STALLED_1S = ("your step stalled: once it was under way, your legs counted under 1 cm along the "
              "way you stepped in 1 s")


@pytest.mark.parametrize("walk, ask, window, under_way, way, said", [
  # P1: never got going, its walk's first window out
  (lambda: Walk(shuffle_s=60.0), (0.0, -0.21), "start", False, None, GOT_NOWHERE),
  # P2: 5 cm, then a later window with none
  (lambda: Walk(wall=0.05), (0.3,), "stuck", True, None, STALLED_1S),
  # an oblique step walked as two, its walk ahead never getting going: its own way
  (lambda: Walk(shuffle_s=60.0), (0.03, 0.15), "start", False, "ahead",
   GOT_NOWHERE.replace("along the way you stepped", "ahead")),
  # P3: ...its walk ahead walked, its sidestep never got going
  (lambda: SideWall(0.0), (0.03, 0.15), "start", True, "left",
   STALLED_1S.replace("along the way you stepped in 1 s", "to your left in 2 s")),
  # ...its walk ahead ended on its stop rule over planted feet (round 2 of the
  # review): it never walked, and the step got nowhere
  (lambda: Planted(0.0, coast_s=0.0), (0.03, -0.15), "start", False, "right",
   GOT_NOWHERE.replace("along the way you stepped", "to your right")),
  # P4: ...or its sidestep got 5 cm and then none in a later window
  (lambda: SideWall(0.05), (0.03, 0.15), "stuck", True, "left",
   STALLED_1S.replace("along the way you stepped", "to your left")),
])
def test_a_stalled_step_says_what_its_legs_counted_in_the_window_that_ended_it(
    walk, ask, window, under_way, way, said):
  """#751: a carried hand's first step after a pick stood still and read
  "your step got nowhere: something stands in your way" with nothing there;
  and a walk that got 5 cm before a window with none read "got nowhere" too.
  A walk whose legs count under `STEP_STUCK_M` along its way in a window says
  so, in that window's words -- its first, or one once under way -- and never
  what stood there, which the step does not measure. One of an oblique
  step's two walks names its own way, and the second is under way only where
  the first walked, by its stance (#649's rule)."""
  w = walk()
  rep = _step(_walker(w), *ask)
  assert rep["why"] == "blocked" and "keptOff" not in rep, rep
  assert rep["stuck"] == {"m": hb.STEP_STUCK_M,
                          "s": hb.STEP_START_S if window == "start" else hb.STEP_STUCK_S,
                          "underWay": under_way, **({"way": way} if way else {})}, rep
  assert st.step_stuck_said(rep["stuck"]) == said
  assert "in your way" not in said
  # ...its head, the shorter form History may say, no other reason's: not the
  # bare fallback's, nor the start of a keep-off's
  head = said.partition(":")[0]
  assert head != "your step stopped short"
  assert not any(t.startswith(head) for t in st.STEP_KEPT_OFF.values())


def test_a_stall_is_the_only_step_record_that_says_how_its_walk_got_nowhere():
  """...a step that walked, one kept off another robot and one that fell in
  its settle after a stall carry no `stuck`."""
  assert "stuck" not in _step(_walker(Walk(coast_s=hb.STEP_STOP_S)), 0.3)
  kept = _step(_walker(Walk(coast_s=hb.STEP_STOP_S, others=[lambda: (1.0, 0.0)])), 0.5)
  assert kept["why"] == "blocked" and "keptOff" in kept and "stuck" not in kept, kept
  fell = _step(_walker(Walk(shuffle_s=60.0)), 0.0, -0.21, fall_in_settle=True)
  assert fell["why"] == "fell" and "stuck" not in fell, fell


def test_a_stalls_words_say_their_numbers_off_the_constants(monkeypatch):
  """#640's rule: moved, the words move with them -- the distance its legs
  must count, and each window's length, as the walk judged them."""
  monkeypatch.setattr(hb, "STEP_STUCK_M", 0.02)
  monkeypatch.setattr(hb, "STEP_START_S", 3.0)
  monkeypatch.setattr(hb, "STEP_STUCK_S", 1.5)
  rep = _step(_walker(Walk(shuffle_s=60.0)), 0.0, -0.21)
  assert rep["stuck"] == {"m": 0.02, "s": 3.0, "underWay": False}
  assert st.step_stuck_said(rep["stuck"]) == (
    "your step got nowhere: your legs counted under 2 cm along the way you stepped in the "
    "first 3 s of its walk")
  rep = _step(_walker(Walk(wall=0.05)), 0.3)
  assert rep["stuck"] == {"m": 0.02, "s": 1.5, "underWay": True}
  assert st.step_stuck_said(rep["stuck"]) == (
    "your step stalled: once it was under way, your legs counted under 2 cm along the way you "
    "stepped in 1.5 s")


def test_a_step_held_through_the_other_robots_run_counts_none_of_it():
  """#690's second review: a robot held standing through the other robot's
  blocking run mid-step (its stepper told, into `held_s`) reads the step as
  it would have gone, its walk's window and its patience counting none of
  the hold. Shown to fail with both on the bare clock: held 30 s, the step
  read "out of time"."""
  class Held(Walk):
    def _twist_routine(self, vx, vy, w):
      if len(self.asked) == 50 and not hasattr(self, "paused"):
        # ...as `tick._Loop.run` holds it: the world goes on, the body stands,
        # and its stepper is told (`HumanoidStepper.held`)
        self.paused, self.t, self.held_s = True, self.t + 30.0, self.held_s + 30.0
      yield from super()._twist_routine(vx, vy, w)
  walk = Held()
  rep = _step(_walker(walk), 0.3)
  assert walk.paused and rep["why"] == "stepped", rep
  assert rep["dx"] == pytest.approx(0.3, abs=0.02)


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
    assert ("a hand that is out whose gripper says it holds something is first raised "
            f"{100 * hmis.CARRY_LIFT_M:g} cm, pulling up with at most {hmis.CARRY_LIFT_N:g} N, "
            "and if what it holds came up with it, it is CARRIED: it "
            "comes back out along the way it went in, then its arm folds to your side, the "
            "forearm up and the hand at your hips, turned as it was against the floor, its "
            "gripper still shut.") in flat, "the carry goes unsaid beside the let-go"
    assert "says so in its report (`carried`;" in flat
    assert "If what it holds did not come with it, the hand lets go and comes home." in flat
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


def test_a_short_walk_to_the_left_is_held_on_until_its_first_foot_moves():
  """A leftward walk's first 0.16-0.58 s are its pelvis leaning over planted
  feet, and its first move is the right foot closing in, 2.4-3.5 cm of
  stance (SimNotes, "Stepping toward the work"): its stop rule, reading the
  pelvis, ended a short left ask in the lean and nothing moved -- 47 of 245
  left asks over eight stances read `too short`, and 40 of the robots' own
  212 left steps of 6 cm. Held on until its legs count its stance
  `STEP_LEAST_M` along its way, it takes that first move."""
  rep = _step(_walker(Walk(coast_s=0.0, planted_s=0.3)), 0.0, 0.04)
  assert rep["why"] == "stepped", "a left walk ended in its lean, its feet unmoved"
  # ...held only that long, then its stop rule as before: here at once, its
  # pelvis having leant past (no step closes on its stance)
  assert rep["dy"] == pytest.approx(hb.STEP_LEAST_M, abs=0.003), rep["dy"]
  # ...carrying something too: the walk is the same walk
  body = _walker(Walk(coast_s=0.0, planted_s=0.3))
  body.held = True
  assert _step(body, 0.0, 0.04)["why"] == "stepped"
  # ...and an oblique step's walk to the left, its second, as much (its
  # verdict is the whole step's, judged along its oblique way)
  rep = _step(_walker(Walk(coast_s=0.0, planted_s=0.3)), 0.02, 0.06)
  assert rep["dy"] >= hb.STEP_LEAST_M, rep
  # ...LEFT ONLY: to the right, ahead and back the walker's first move is a
  # whole stride, and held on, a short ask went a stride too far
  for ask in ((0.0, -0.04), (0.04, 0.0), (-0.04, 0.0)):
    rep = _step(_walker(Walk(coast_s=0.0, planted_s=0.3)), *ask)
    assert rep["why"] == "too short", (ask, rep)


def test_a_left_walk_is_held_on_at_most_its_bound_of_walking():
  """...its feet never moving, held `STEP_LEFT_HOLD_S` of its walking at
  most, then its stop rule as before (a stance pressed where it is reads
  as it did, a second later at most) -- and never counting the time it is
  held through the other robot's run (#690), as its windows do not."""
  walk = Walk(coast_s=0.0, planted_s=60.0)
  rep = _step(_walker(walk), 0.0, 0.04)
  walked = DT * sum(1 for _, vy, _ in walk.asked if vy)
  assert rep["why"] == "too short" and walked == pytest.approx(hb.STEP_LEFT_HOLD_S, abs=2 * DT), walked

  class Held(Walk):
    def _twist_routine(self, vx, vy, w):
      if len(self.asked) == 50 and not hasattr(self, "paused"):
        # ...as `tick._Loop.run` holds it: the world goes on, the body stands
        self.paused, self.t, self.held_s = True, self.t + 30.0, self.held_s + 30.0
      yield from super()._twist_routine(vx, vy, w)
  walk = Held(coast_s=0.0, planted_s=60.0)
  rep = _step(_walker(walk), 0.0, 0.04)
  walked = DT * sum(1 for _, vy, _ in walk.asked if vy)
  assert walk.paused and walked == pytest.approx(hb.STEP_LEFT_HOLD_S, abs=2 * DT), walked
  # ...and a left walk that gets nowhere at all still stalls in its first window
  rep = _step(_walker(SideWall(0.0, coast_s=0.0)), 0.0, 0.04)
  assert rep["why"] == "blocked" and rep["stuck"]["s"] == hb.STEP_START_S, rep


def test_a_left_walk_held_on_still_falls_and_keeps_off_what_a_step_keeps_off():
  """...every tick it is held is a tick of its walk as before: a fall ends it
  `fell`, another robot's disc and what its depth camera saw at its knees
  end it `blocked`, saying how."""
  assert _step(_walker(Walk(coast_s=0.0, planted_s=60.0, fall_at=100)), 0.0, 0.04)["why"] == "fell"
  walk = Walk(coast_s=0.0, planted_s=60.0, others=[lambda: (0.0, 0.9)])
  rep = _step(_walker(walk), 0.0, 0.04)
  assert rep["why"] == "blocked" and rep["keptOff"]["by"] == "keep", rep
  walk = Walk(coast_s=0.0, planted_s=60.0)
  side = np.array([[x, 0.3, 0.3] for x in np.linspace(-0.3, 0.3, 31)])
  walk.knee.fold(side, 0.0, walk.pose, walk.odo.corrected)
  rep = _step(_walker(walk), 0.0, 0.04)
  assert rep["why"] == "blocked" and rep["keptOff"]["by"] == "low", rep


def test_a_left_walk_held_on_ends_where_a_stand_up_or_a_stop_closes_it():
  """...and closed from outside mid-hold -- a stand-up landing (#348), a
  stop thrown in -- it ends there, holding nothing on."""
  for end in ("close", "throw"):
    walk = Walk(coast_s=0.0, planted_s=60.0)
    body = _walker(walk)
    routine = body.step_routine(0.0, 0.04)
    for _ in range(200):
      next(routine)
      body._d.time = walk.t
    assert walk.legs == "walk" and walk.asked[-1][1] > 0.0, "not walking left at 0.4 s"
    if end == "close":
      routine.close()
    else:
      with pytest.raises(tick.MissionAborted):
        routine.throw(tick.MissionAborted("stop"))
    with pytest.raises(StopIteration):
      next(routine)


def test_a_short_step_to_the_left_flown_takes_its_first_foot():
  """Flown (a few seconds of sim): from the stand, `step(0, 0.04)` went
  0.8-1.0 cm and read `too short`, its walk stopped in the lean; held on, the
  right foot closes in and it goes about 3 cm."""
  rep, _, left, _ = _flown_stance_step(0.0, 0.04)
  assert rep["why"] == "stepped" and left >= 0.02, (rep["why"], f"{100 * left:.1f} cm")


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


def test_a_stalled_step_reads_false_saying_what_its_legs_counted_on_historys_line():
  """#751: a step its walk got nowhere in reads false saying what its legs
  counted and in which window -- in its report, and on the run's line
  History keeps, as a test that read false and as the step the run stopped
  at -- never "something stands in your way"; a body that says neither how
  it kept off nor how its walk got nowhere is told its step stopped short."""
  from pluggybot import lifecycle as lc
  from pluggybot.lifecycle import errand_from
  from pluggybot.mind.overseer import Decision
  from pluggybot.mind.thoughts import HISTORY
  from pluggybot.procedure import library as lib
  from test_body import stub_life
  life = stub_life()
  stalls = iter([{"m": 0.01, "s": 2.0, "underWay": False}, {"m": 0.01, "s": 1.0, "underWay": True}])
  life.body.step_routine = lambda *a, **kw: tick.result(
    {"why": "blocked", "seconds": 3.3, "dx": 0.002, "dy": 0.0, "turn": 0.0, "stuck": next(stalls)})
  L = lib.Library(lc.world_facts("home_humanoid"))
  L.define("creep", 'def creep():\n  if not step(0, -0.15):\n    d = read("step.dx")\n'
                    '  step(0.11)\n')
  out = life.run_errand(errand_from(Decision(action="procedure:creep"), library=L))
  assert [s["reason"] for s in out["procedure"]["steps"]] == [GOT_NOWHERE, STALLED_1S]
  [line] = [x for x in life.thoughts.lines(HISTORY) if "procedure creep" in x]
  assert line.endswith(
    f"the procedure creep did not finish -- it stopped at line 4, step: {STALLED_1S} (1 step "
    f"had run; 1 test read false -- line 2, step: {GOT_NOWHERE}) -- it ended with d = 0.002")
  assert "in your way" not in life.thoughts.read("History.md")
  # ...and a record that says neither is told only what is true of any
  body = StubBody(*_bare())
  body.step_routine = lambda *a, **kw: tick.result(
    {"why": "blocked", "seconds": 3.3, "dx": 0.0, "dy": 0.0, "turn": 0.0})
  step = _fly('def a():\n    if not step(0.1):\n        pass\n', body)["steps"][0]
  assert not step["ok"] and step["reason"] == "your step stopped short"


@pytest.mark.parametrize("stuck, head", [({"m": 0.01, "s": 2.0, "underWay": False},
                                          "your step got nowhere"),
                                         ({"m": 0.01, "s": 1.0, "underWay": True},
                                          "your step stalled")])
def test_a_stalls_words_give_way_to_their_head_where_historys_line_has_no_room(stuck, head):
  """#751 on #728's shorter form: rebuilt off five wakes (11-14 to 11-25),
  the stall's words in full took the step's own reason off the run's line in
  up to 2 of the 23 runs with one, where the old 51 characters were said,
  and a readout onto a line of its own in 5. Its head, before the colon, is
  its shorter form: said where the rest would move anything else, and the
  line otherwise as one with no shorter form."""
  from pluggybot import lifecycle as lc
  body = StubBody(*_bare())
  body.step_routine = lambda *a, **kw: tick.result(
    {"why": "blocked", "seconds": 3.3, "dx": 0.0, "dy": 0.0, "turn": 0.0, "stuck": stuck})
  step = _fly('def a():\n    if not step(0.1):\n        pass\n', body)["steps"][0]
  assert step["brief"] == head and step["reason"].startswith(head + ": ")
  walk = {"verb": "drive_to", "line": 4, "ok": False, "tested": True, "reason": "w" * 220}
  run = {"ok": True, "completed": 9, "total": 9, "steps": [walk, {**step, "line": 10}]}
  [line] = lc.procedure_outcome("p", run)
  assert line == (f"ran the procedure p to its end (9 steps; 2 tests read false -- line 4, "
                  f"drive_to: {'w' * 220}; line 10, step: {head})")
  # ...and in full where it moves nothing else
  walk["reason"] = "w" * 100
  [line] = lc.procedure_outcome("p", run)
  assert line.endswith(f"line 10, step: {step['reason']})")


def test_a_step_stops_short_of_a_wall_its_lidar_reads_at_its_heads_height():
  """#717 (Ben's option 1): Rowan's step loops walked into the lobby's wall
  to 0.12 m and the storeroom's corner to 0.01 m, every step "stepped", and
  the look's bow there felled it. A step keeps its LIDAR `STEP_WALL_M` off
  what that reads at its head's height ahead along its way, its stride's
  carry counted, and says so -- stood that near already, it goes along the
  wall but no nearer."""
  walk = Walk(coast_s=hb.STEP_STOP_S, over_wall=0.6)
  rep = _step(_walker(walk), 0.5)
  assert rep["why"] == "blocked" and rep["keptOff"]["by"] == "wall", rep
  assert rep["keptOff"]["keepM"] == hb.STEP_WALL_M
  assert 0.6 - walk.pose[0] == pytest.approx(hb.STEP_WALL_M, abs=0.02), "its carry took it in"
  assert rep["keptOff"]["m"] == pytest.approx(0.6 - walk.pose[0], abs=0.005)
  # ...a sidestep along it, stood 0.25 m off it, goes on; a step nearer does not
  walk = Walk(coast_s=hb.STEP_STOP_S, over_wall=0.25)
  assert _step(_walker(walk), 0.0, 0.2)["why"] == "stepped"
  walk = Walk(coast_s=hb.STEP_STOP_S, over_wall=0.25)
  rep = _step(_walker(walk), 0.2)
  assert rep["why"] == "blocked" and rep["keptOff"]["by"] == "wall"
  # ...stopped within its first strides off a scan taken as it set off: its
  # walk's own first scan comes a tenth of a second in
  assert walk.scans_taken == 1 and walk.t < LIDAR_PERIOD, walk.t
  # ...and a wall behind it keeps no step ahead
  walk = Walk(coast_s=hb.STEP_STOP_S, over_wall=-0.2)
  assert _step(_walker(walk), 0.3)["why"] == "stepped"


def test_a_step_keeps_off_what_it_saw_at_its_knees_where_its_belief_has_it():
  """#717: up close the depth camera sees none of a counter's face (its
  picture's bottom), and pressed there a thigh met it at 150-400 N. What it
  saw at its knees' height is kept where its belief placed it -- a correction
  the matcher makes is the body's truer place, a jump past `KNEE_JUMP_M` a
  belief that no longer knows where it saw it -- and a step keeps its middle
  `STEP_LOW_M` off it."""
  face = np.array([[0.45, y, 0.3] for y in np.linspace(-0.3, 0.3, 31)])
  for correction, short in (((0.0, 0.0, 0.0), 0.0), ((0.05, 0.0, 0.0), 0.05)):
    walk = Walk(coast_s=hb.STEP_STOP_S)
    walk.knee.fold(face, 0.0, walk.pose, walk.odo.corrected)
    walk.correct_at, walk.correction = 40, correction
    rep = _step(_walker(walk), 0.4)
    went = walk.pose[0] - walk.odo.corrected[0]
    assert rep["why"] == "blocked" and rep["keptOff"]["by"] == "low", (correction, rep)
    assert 0.45 - went == pytest.approx(hb.STEP_LOW_M + short, abs=0.02), (correction, went)
    assert rep["keptOff"]["m"] == pytest.approx(0.45 - walk.pose[0], abs=0.02)
  # ...a belief that jumped knows no longer where it saw it
  walk = Walk(coast_s=hb.STEP_STOP_S)
  walk.knee.fold(face, 0.0, walk.pose, walk.odo.corrected)
  walk.correct_at, walk.correction = 40, (hmis.KNEE_JUMP_M + 0.05, 0.0, 0.0)
  assert _step(_walker(walk), 0.4)["why"] == "stepped"
  # ...and what stands higher than its knees -- a table's top -- is no part of it
  walk = Walk(coast_s=hb.STEP_STOP_S)
  walk.knee.fold(face + [0.0, 0.0, hmis.KNEE_Z], 0.0, walk.pose, walk.odo.corrected)
  assert _step(_walker(walk), 0.4)["why"] == "stepped"


def test_the_knee_memory_is_placed_by_the_belief_and_forgets_where_it_cannot_be():
  """What it saw stays where its belief placed it: a walk ahead brings it
  nearer, a turn swings it round; a frame that sees a place afresh replaces
  what was kept there (never nearer than the camera holds the knees' whole
  height); a jump of the belief, and a get-up, which no leg counts, forget it
  (#717)."""
  k = hmis.KneeMemory()
  k.fold(np.array([[1.0, 0.0, 0.3], [0.4, 0.2, 0.3]]), 0.0, (0.0, 0.0, 0.0), np.zeros(3))
  assert np.allclose(sorted(map(tuple, k.near((0.3, 0.0, 0.0), np.zeros(3)))),
                     [(0.1, 0.2), (0.7, 0.0)])
  assert np.allclose(sorted(map(tuple, k.near((0.0, 0.0, math.pi / 2), np.zeros(3)))),
                     [(0.0, -1.0), (0.2, -0.4)])
  # ...a frame that sees nothing past the far one keeps it (another robot, a
  # carried piece or its own body may hide it); one that sees the floor past
  # it takes it out; the near one, under the camera's whole view of the knees'
  # height, it keeps
  k.fold(np.zeros((0, 3)), 1.0, (0.0, 0.0, 0.0), np.zeros(3))
  assert len(k.near((0.0, 0.0, 0.0), np.zeros(3))) == 2
  k.fold(np.array([[1.3, 0.0, 0.0]]), 1.0, (0.0, 0.0, 0.0), np.zeros(3))
  assert np.allclose(k.near((0.0, 0.0, 0.0), np.zeros(3)), [(0.4, 0.2)])
  state, arrays = k.kept_state()
  back = hmis.KneeMemory()
  back.restore_kept(state, arrays)
  assert np.array_equal(back.pts, k.pts) and np.array_equal(back.seen, k.seen)
  assert not len(k.near((0.0, 0.0, 0.0), np.array([0.0, 0.0, hmis.KNEE_JUMP_RAD + 0.01]))), \
    "a heading's jump kept what it saw"
  k.fold(np.array([[0.4, 0.0, 0.3]]), 2.0, (0.0, 0.0, 0.0), np.zeros(3))
  # ...and what it saw where its arms swing with it (#747)
  swing = hmis.ArmsMemory()
  swing.fold(np.array([[0.4, 0.0, 0.7]]), 2.0, (0.0, 0.0, 0.0), np.zeros(3))
  mission = NS(falls=0, legs="stand", knee=k, swing=swing, getup=NS(begin=lambda: None))
  HumanoidMission._fell(mission)
  assert not len(k.pts), "a fall kept what it saw"
  assert not len(swing.pts), "a fall kept what it saw where its arms swing"


def test_a_steps_own_scan_draws_noise_of_its_own():
  """The scan a step takes as it sets off, and a look's (`look_scan`), draw
  their noise from a stream of their own (#717): every other scan the LIDAR
  takes draws as it did, so a flight no new rule stops is the flight it was.
  Drawn from the LIDAR's, the tour's table lays parted from today's at their
  first step and fell in a withdraw three times."""
  body = _floor_body()
  try:
    mi = body.mission
    lidar = (mi.lidar.rng.bit_generator.state, mi.lidar.peer_rng.bit_generator.state)
    look = mi._look_rngs[0].bit_generator.state
    seen = (mi.peer_returns, mi.scanned_t, dict(mi.seen_t))
    mi.look_scan()
    assert (mi.lidar.rng.bit_generator.state, mi.lidar.peer_rng.bit_generator.state) == lidar
    # ...nor whom it saw, nor the front stop's reading: with another robot by,
    # those move the pair's flight (#717's review)
    assert (mi.peer_returns, mi.scanned_t, dict(mi.seen_t)) == seen
    assert mi._look_rngs[0].bit_generator.state != look
    assert mi.over_returns is not None and mi.over_returns[0] == float(body._d.time)
    state, _ = mi.kept_state()
    assert state["lookRng"][0] == mi._look_rngs[0].bit_generator.state
  finally:
    body.close()


def test_a_pressed_step_reads_false_and_says_what_its_scans_do():
  """#718: pressed on a counter, a stride lands where it began and the legs
  count 7-9 cm a step where the body went 1-3; Rowan's procedures took
  `step.dx` off where a tag was, 32 cm that never happened in four steps. A
  step its scans of the room say went under `STEP_PRESSED_SHARE` of what its
  legs count reads false, saying both, and `step.*` says the scans'."""
  walk = Walk(coast_s=hb.STEP_STOP_S)
  # ...its matcher taking back what the legs count, 2 cm a scan
  walk.correct_at, walk.correction = (20, 40, 60, 80), (-0.02, 0.0, 0.0)
  rep = _step(_walker(walk), 0.12)
  assert rep["why"] == "stepped" and rep["mdx"] == pytest.approx(rep["dx"] - 0.08, abs=0.002)
  assert rep["jump"] == pytest.approx(0.02, abs=1e-6)
  out = st._pressed(rep, 0.12, 0.0)
  assert out["why"] == "pressed" and out["dx"] == pytest.approx(rep["mdx"])
  said = st.STEP_PRESSED.format(**out)
  assert said == (f"your legs counted {100 * rep['dx']:.0f} cm along the way you stepped, but "
                  f"your scans of the room put you only {100 * rep['mdx']:.0f} cm on: "
                  "something held your body back")
  # ...never a step its scans keep up with, nor one under STEP_PRESSED_M, nor
  # one its belief jumped across (a relocation, a doubt's take, a fix)
  for mdx, dx, jump in ((0.06, 0.1, 0.0), (0.0, hb.STEP_PRESSED_M - 0.005, 0.0),
                        (0.01, 0.1, hb.STEP_JUMP_M + 0.01)):
    rep = {"why": "stepped", "dx": dx, "dy": 0.0, "turn": 0.0, "mdx": mdx, "mdy": 0.0,
           "jump": jump}
    assert st._pressed(rep, 0.12, 0.0)["why"] == "stepped", (mdx, dx, jump)
  # ...and a step that turned first is judged along the way it walked after
  rep = {"why": "stepped", "dx": 0.002, "dy": 0.09, "turn": math.pi / 2, "mdx": 0.0, "mdy": 0.01}
  assert st._pressed(rep, 0.12, 0.0)["why"] == "pressed"
  rep = {**rep, "mdy": 0.08}
  assert st._pressed(rep, 0.12, 0.0)["why"] == "stepped"
  body = StubBody(*_bare())
  body.step_routine = lambda *a, **kw: tick.result(
    {"why": "stepped", "seconds": 1.0, "dx": 0.08, "dy": 0.0, "turn": 0.0, "mdx": 0.01, "mdy": 0.0})
  ran = _fly('def a():\n    if not step(0.12):\n        d = read("step.dx")\n', body)
  step = ran["steps"][0]
  assert not step["ok"] and step["why"] == "pressed" and step["reason"].startswith("your legs")
  assert ran["locals"]["d"] == pytest.approx(0.01)


def test_the_steps_text_says_what_it_keeps_off_and_a_pressed_step_off_its_constants():
  """#717, #718: the numbers the text says are the constants the checks
  read."""
  doc = st.HANDS_VERBS["step"].doc
  said = re.search(r"within ([0-9.]+) m of anything that reads at your head's height "
                   r"\(([0-9.]+) m over the floor and up\).*your middle within ([0-9.]+) m of what "
                   r"your depth camera has seen in front of you at your knees' height \(([0-9.]+) m "
                   r"over the floor and lower\)", doc)
  assert said, doc
  assert [float(v) for v in said.groups()] == pytest.approx(
    [hb.STEP_WALL_M, lidar.OVER_Z, hb.STEP_LOW_M, hmis.KNEE_Z], abs=1e-9)
  assert "a walk goes along it but no nearer" in doc
  said = re.search(r"went under ([0-9.]+) of what your legs count, these ([0-9]+) cm or more", doc)
  assert said and float(said.group(1)) == hb.STEP_PRESSED_SHARE
  assert float(said.group(2)) == pytest.approx(100 * hb.STEP_PRESSED_M)
  for by, kept in (("wall", {"m": 0.28, "keepM": 0.3}), ("low", {"m": 0.19, "keepM": 0.2})):
    assert f"{kept['keepM']:.2f} m" in st.STEP_KEPT_OFF[by].format(**kept)


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
