"""The step (issue #525; docs/Hands.md): the body's feet moved to its work,
each rule pinned without a flight -- the stub body, a fake walk, the
odometry fed its soles (docs/Testing.md). The flights that measured them
are SimNotes' "Stepping toward the work".

  1. `step` reports where the stance went in the frame it stood in before,
     read as sensors, and a place read before is turned by them;
  2. a step carries nothing: holding something, nothing moves;
  3. the humanoid's step turns first, stops its walk where what it believes
     it has walked and its stride's carry reach the step, reports a turn as
     on the spot, says `blocked` where its walk gets nowhere, and does not
     move holding something;
  4. its odometry takes a shuffle's swing foot off the floor.
"""

import math
from types import SimpleNamespace as NS

import numpy as np
import pytest

from pluggybot.body import StubBody
from pluggybot.humanoid import body as hb
from pluggybot.humanoid import odometry as od
from pluggybot.procedure import steps as st

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


def test_a_step_carries_nothing():
  body = StubBody(*_bare())
  ran = _fly('def a():\n    reach("left", 0.3, 0.2, 0.9)\n    close_hand("left")\n'
             '    if not step(0.2):\n        x = read("step.dx")\n', body)
  step = ran["steps"][2]
  assert not step["ok"] and step["why"] == "holding" and step["reason"] == st.STEP_WHY["holding"]
  assert ran["locals"]["x"] == 0.0 and body.pose[:2] == pytest.approx((0.0, 0.0))


# ---- the humanoid's step, on a fake walk ------------------------------------------


class Walk:
  """The walking policy as the step drives it: each command a physics step
  at the twist asked, its believed pose moving so; a stride's carry after
  the last command (`coast`); `wall`, where its walk gets nowhere."""

  def __init__(self, coast_s=0.22, wall=None, turn_drift=0.0):
    self.pose, self.falls, self.legs, self.t = (0.0, 0.0, 0.0), 0, "stand", 0.0
    self.coast_s, self.wall, self.v, self.asked = coast_s, wall, (0.0, 0.0), []
    #: how far a turn's shuffle walks the BELIEF a command, m
    self.turn_drift = turn_drift

  def _getup_routine(self):
    return
    yield

  def _twist_routine(self, vx, vy, w):
    self.legs = "walk"
    self.asked.append((vx, vy, w))
    x, y, yaw = self.pose
    c, s = math.cos(yaw), math.sin(yaw)
    nx, ny = x + (c * vx - s * vy) * DT, y + (s * vx + c * vy) * DT
    if w and not (vx or vy):
      nx += self.turn_drift
    if self.wall is not None and nx > self.wall:
      nx = x
    self.v = ((nx - x) / DT, (ny - y) / DT)
    self.pose = (nx, ny, yaw + w * DT)
    self.t += DT
    yield hb.STILL

  def to_stand_routine(self):
    x, y, yaw = self.pose
    self.pose = (x + self.v[0] * self.coast_s, y + self.v[1] * self.coast_s, yaw)
    self.v, self.legs = (0.0, 0.0), "stand"
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
    return self.mission.pose

  def gripped(self, side):
    return self.held


def _step(body, *args):
  routine = body.step_routine(*args)
  try:
    while True:
      next(routine)
      body._d.time = body.mission.t
  except StopIteration as e:
    return e.value


def test_the_humanoids_step_stops_where_its_stride_carries_it_to_the_step():
  """At the walker's slowest its feet slid and its odometry counted a
  stride backward; at `STEP_V` it lifts them, and stopped commanding at
  what it believes it has walked plus its speed times `STEP_STOP_S`, the
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
  walk = Walk(turn_drift=0.0001)
  rep = _step(_walker(walk), 0.0, 0.0, -0.4)
  assert walk.pose[0] > 0.01 and (rep["dx"], rep["dy"]) == (0.0, 0.0) and rep["turn"] < -0.35
  walk = Walk(coast_s=hb.STEP_STOP_S, turn_drift=0.0001)
  rep = _step(_walker(walk), 0.3, 0.0, 0.5)
  assert math.hypot(rep["dx"], rep["dy"]) == pytest.approx(0.3, abs=0.004)


TURN_STEP = hb.TURN_W * DT


def test_a_step_that_gets_nowhere_says_it_is_blocked_and_one_holding_does_not_move():
  walk = Walk(wall=0.05)
  rep = _step(_walker(walk), 0.3)
  assert rep["why"] == "blocked" and rep["dx"] == pytest.approx(0.05, abs=0.002)
  assert walk.t <= 2 * hb.STEP_STUCK_S + 0.01
  # ...and with something in its hand it neither turns nor walks
  walk = Walk()
  body = _walker(walk)
  body.held = True
  rep = _step(body, 0.3, 0.0, 0.5)
  assert rep["why"] == "holding" and walk.asked == []


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
