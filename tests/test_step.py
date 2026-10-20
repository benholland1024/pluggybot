"""The step (issue #525; docs/Hands.md): the body's feet moved to its work,
each rule pinned without a flight -- the stub body, a fake walk, the
odometry fed its soles (docs/Testing.md). The flights that measured them
are SimNotes' "Stepping toward the work".

  1. `step` reports where the stance went in the frame it stood in before,
     read as sensors, and a place read before is turned by them;
  2. a step carries nothing: holding something, nothing moves;
  3. the humanoid's step hands its legs to the walker before it measures
     anything, turns first -- its heading counted unwrapped -- stops its
     walk where what it believes it has walked and its stride's carry reach
     the step, walks an oblique step the walker would skew as two, reports a
     turn as on the spot, says `blocked` where its walk gets nowhere and
     `fell` for a fall up to its last settle, commands nothing once fallen,
     and does not move holding something;
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
  at the twist asked, its believed pose (the pelvis's) moving so; a stride's
  carry after the last command (`coast`); `wall`, where its walk gets
  nowhere; the legs handed to it in `handover_s`, the pelvis coming
  `handover_m` forward over the feet meanwhile (out of a bow); and a fall
  at its command `fall_at`."""

  def __init__(self, coast_s=0.22, wall=None, turn_drift=0.0, handover_s=0.0, handover_m=0.0,
               fall_at=None, settle_m=0.0):
    self.pose, self.falls, self.legs, self.t = (0.0, 0.0, 0.0), 0, "stand", 0.0
    self.coast_s, self.wall, self.v, self.asked = coast_s, wall, (0.0, 0.0), []
    #: how far a turn's shuffle walks the BELIEF a command, m
    self.turn_drift = turn_drift
    self.handover_s, self.handover_m, self.fall_at = handover_s, handover_m, fall_at
    #: how far a turn's feet settle as the legs are handed back, m
    self.settle_m = settle_m
    #: where the pelvis stands ahead of the feet's middle, m
    self.lean = -handover_m

  @property
  def stance(self):
    x, y, yaw = self.pose
    return (x - math.cos(yaw) * self.lean, y - math.sin(yaw) * self.lean, yaw)

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


def test_a_step_that_gets_nowhere_says_it_is_blocked_and_one_holding_does_not_move():
  walk = Walk(wall=0.05)
  rep = _step(_walker(walk), 0.3)
  assert rep["why"] == "blocked" and rep["dx"] == pytest.approx(0.05, abs=0.002)
  assert walk.t <= 2 * hb.STEP_STUCK_S + 0.01
  # ...and a walk given less time than it takes runs out of it
  walk = Walk()
  assert _step(_walker(walk), 0.3, 0.0, 0.0, 0.1)["why"] == "out of time"
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
